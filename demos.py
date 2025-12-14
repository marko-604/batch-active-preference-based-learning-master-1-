from simulation_utils import create_env, get_feedback, run_algo
import numpy as np

from sampling import Sampler
from metrics import ResultsTracker, save_results
from LLM_ASSISTED import get_feedback_mixed, predictive_entropy


# ---- alignment metric (paper Eq. 15) ----
def alignment_metric(w_true, w_hat):
    """
    m = (w_true^T w_hat) / (||w_true|| * ||w_hat||)
    Returns a float or None if norms are zero.
    """
    if w_true is None or w_hat is None:
        return None

    w_true = np.asarray(w_true, dtype=float).flatten()
    w_hat = np.asarray(w_hat, dtype=float).flatten()

    if w_true.shape != w_hat.shape:
        return None

    n_true = np.linalg.norm(w_true)
    n_hat = np.linalg.norm(w_hat)
    if n_true == 0.0 or n_hat == 0.0:
        return None

    return float(np.dot(w_true, w_hat) / (n_true * n_hat))


def _compute_batch_entropies(simulation_object, inputA_set, inputB_set, w_samples):
    """
    Computes (psi_j, H_j) for each query j in the batch.
    psi_j = phi(A) - phi(B)
    H_j   = predictive entropy under w_samples
    """
    b = len(inputA_set)
    psis = []
    Hs = []

    for j in range(b):
        simulation_object.feed(inputA_set[j])
        phi_A = simulation_object.get_features()

        simulation_object.feed(inputB_set[j])
        phi_B = simulation_object.get_features()

        psi = np.asarray(phi_A, dtype=float) - np.asarray(phi_B, dtype=float)
        H = predictive_entropy(w_samples, psi)

        psis.append(psi)
        Hs.append(H)

    return psis, Hs


def batch(task, method, N, M, b, human_topk=2):
    """
    Batched active preference-based learning with:
      - mixed human/LLM feedback
      - EXACTLY `human_topk` human queries per batch chosen as top entropy queries
      - alignment metric (if env.w_true exists)
      - avg log-likelihood
      - JSON logging for plotting

    CLI: python run.py <task> <method> N M b
    example: python run.py driver greedy 60 1000 10
    """
    if N % b != 0:
        print("N must be divisible by b")
        return

    if human_topk < 0 or human_topk > b:
        raise ValueError("human_topk must be between 0 and b")

    # ----------------------------------------------------------
    # 1) Environment & sampler
    # ----------------------------------------------------------
    simulation_object = create_env(task)

    # Determine feature dimension d
    if hasattr(simulation_object, "num_of_features"):
        d = int(simulation_object.num_of_features)
    else:
        # Fallback: infer from one feature evaluation
        simulation_object.feed(simulation_object.u0)
        phi0 = simulation_object.get_features()
        d = len(phi0)

    sampler = Sampler(d)

    # Collected data so far
    psi_set = []
    s_set = []

    # Ground-truth weights for alignment (if present in env)
    w_true = getattr(simulation_object, "w_true", None)

    # Tracker
    tracker = ResultsTracker(task_name=task, method_name=method)

    num_batches = N // b

    # ----------------------------------------------------------
    # 2) Batch loop
    # ----------------------------------------------------------
    for batch_idx in range(num_batches):
        # 2.1 Update sampler with all data so far
        if psi_set:
            sampler.A = np.vstack(psi_set)
            sampler.y = np.array(s_set).reshape(-1, 1)

        # 2.2 Sample posterior weights and compute mean
        w_samples = sampler.sample(M)
        mean_w = np.mean(w_samples, axis=0)

        # Print normalized estimate
        if np.linalg.norm(mean_w) > 0:
            mean_w_unit = mean_w / np.linalg.norm(mean_w)
            print(f"[batch {batch_idx+1}/{num_batches}] w-estimate = {mean_w_unit}")
        else:
            print(f"[batch {batch_idx+1}/{num_batches}] w-estimate undefined (no data yet)")

        # 2.3 Log metrics BEFORE new queries
        tracker.log_iteration(
            iteration=batch_idx + 1,
            num_queries=len(s_set),
            w_est=mean_w,
            w_true=w_true,
            psi_list=psi_set if psi_set else None,
            s_list=s_set if s_set else None,
        )

        # 2.4 Select next batch using run_algo
        inputA_set, inputB_set = run_algo(method, simulation_object, w_samples, b)

        # 2.5 Compute entropy for each query and choose top-k as human
        _, Hs = _compute_batch_entropies(simulation_object, inputA_set, inputB_set, w_samples)
        if human_topk == 0:
            human_idx = set()
        else:
            human_idx = set(np.argsort(Hs)[-human_topk:].tolist())

        H_str = ", ".join([f"{h:.3f}" for h in Hs])
        print(f"[batch {batch_idx+1}/{num_batches}] Hs = [{H_str}] | HUMAN idx={sorted(human_idx)}")

        # 2.6 Collect feedback for each query in batch
        for j in range(b):
            psi_obs, s, source = get_feedback_mixed(
                task=task,
                simulation_object=simulation_object,
                input_A=inputA_set[j],
                input_B=inputB_set[j],
                query_index=j,
                batch_size=b,
                tracker=tracker,
                w_samples=w_samples,
                force_human=(j in human_idx),
                routing_mode="topk",
            )
            psi_set.append(psi_obs)
            s_set.append(s)

    # ----------------------------------------------------------
    # 3) Final posterior estimate after all N queries
    # ----------------------------------------------------------
    if psi_set:
        sampler.A = np.vstack(psi_set)
        sampler.y = np.array(s_set).reshape(-1, 1)
        w_samples = sampler.sample(M)
        mean_w = np.mean(w_samples, axis=0)

        if np.linalg.norm(mean_w) > 0:
            mean_w_unit = mean_w / np.linalg.norm(mean_w)
            print(f"Final w-estimate = {mean_w_unit}")

        # final metrics point
        tracker.log_iteration(
            iteration=num_batches + 1,
            num_queries=len(s_set),
            w_est=mean_w,
            w_true=w_true,
            psi_list=psi_set,
            s_list=s_set,
        )

    # ----------------------------------------------------------
    # 4) Save metrics to JSON
    # ----------------------------------------------------------
    save_results(tracker)


def nonbatch(task, method, N, M):
    """
    Kept for completeness (unchanged logic except small cleanup).
    Non-batch querying uses get_feedback (human) only.
    """
    simulation_object = create_env(task)
    d = simulation_object.num_of_features

    lower_input_bound = [x[0] for x in simulation_object.feed_bounds]
    upper_input_bound = [x[1] for x in simulation_object.feed_bounds]

    w_sampler = Sampler(d)
    psi_set = []
    s_set = []

    input_A = np.random.uniform(
        low=2 * np.array(lower_input_bound),
        high=2 * np.array(upper_input_bound),
        size=(2 * simulation_object.feed_size),
    )
    input_B = np.random.uniform(
        low=2 * np.array(lower_input_bound),
        high=2 * np.array(upper_input_bound),
        size=(2 * simulation_object.feed_size),
    )

    psi, s = get_feedback(simulation_object, input_A, input_B)
    psi_set.append(psi)
    s_set.append(s)

    for i in range(1, N):
        w_sampler.A = np.vstack(psi_set)
        w_sampler.y = np.array(s_set).reshape(-1, 1)

        w_samples = w_sampler.sample(M)
        mean_w_samples = np.mean(w_samples, axis=0)

        if np.linalg.norm(mean_w_samples) > 0:
            print("w-estimate = {}".format(mean_w_samples / np.linalg.norm(mean_w_samples)))
        else:
            print("w-estimate undefined (no data yet)")

        input_A, input_B = run_algo(method, simulation_object, w_samples)
        psi, s = get_feedback(simulation_object, input_A, input_B)

        psi_set.append(psi)
        s_set.append(s)

    w_sampler.A = np.vstack(psi_set)
    w_sampler.y = np.array(s_set).reshape(-1, 1)
    w_samples = w_sampler.sample(M)
    mean_w_samples = np.mean(w_samples, axis=0)

    if np.linalg.norm(mean_w_samples) > 0:
        print("Final w-estimate = {}".format(mean_w_samples / np.linalg.norm(mean_w_samples)))
    else:
        print("Final w-estimate undefined")
