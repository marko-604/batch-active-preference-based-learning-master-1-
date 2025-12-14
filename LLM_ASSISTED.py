import requests
import numpy as np

from models import Driver
from simulation_utils import get_feedback as human_get_feedback
from driver_extra_feature import EXTRA_FEATURE_BANK

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "mistral"

# Fallback-only heuristic (used only if w_samples is None and you don't force-human)
INFO_GAIN_THRESHOLD = 0.65

# global history of driver preferences: list of {"psi": [...], "s": int}
DRIVER_HISTORY = []


# -----------------------------
# Human feedback wrapper
# -----------------------------
def _human_feedback_psi(simulation_object, input_A, input_B):
    result = human_get_feedback(simulation_object, input_A, input_B)
    if isinstance(result, tuple) and len(result) >= 2:
        psi, s = result[0], result[1]
    else:
        raise RuntimeError("Unexpected return format from human_get_feedback.")
    return np.array(psi), int(s)


def _log_driver_choice(psi, s):
    """
    Store each driver comparison in a simple Python structure
    so we can later ask the LLM if a new feature is needed.
    """
    psi_arr = np.asarray(psi, dtype=float).tolist()
    DRIVER_HISTORY.append({"psi": psi_arr, "s": int(s)})


# -----------------------------
# Driver feature interpretation
# -----------------------------
def describe_driver_features(phi):
    """
    Turn the 4-dimensional Driver feature vector into a human-readable description.

    Driver.get_features() returns:
        1. staying_in_lane      (higher is better)
        2. keeping_speed        (mean squared error from speed 1; LOWER is better)
        3. heading              (mean sin(heading); closer to 1 ~ straight / forward)
        4. collision_avoidance  (higher ~ more overlap / near-collisions; LOWER is better)
    """
    staying_in_lane, keeping_speed, heading, collision_avoidance = map(float, phi)

    desc = [
        f"Staying in lane (higher is better): {staying_in_lane:.3f}",
        f"Speed deviation (lower is better): {keeping_speed:.3f}",
        f"Heading straightness (higher is better): {heading:.3f}",
        f"Collision risk (lower is better): {collision_avoidance:.3f}",
    ]
    return "\n".join(desc)


# -----------------------------
# Ollama call + parsing
# -----------------------------
def call_ollama_mistral(prompt: str) -> str:
    """
    Call a local Ollama model with a prompt. Returns raw text response.
    """
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "stream": False}

    # Best practice: always set a timeout so your experiment doesn’t hang forever.
    resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def parse_preference_from_text(text: str) -> int:
    """
    STRICT parser:
      "1" or "a" -> +1 (prefer A)
      "2" or "b" -> -1 (prefer B)
      "0"/tie/equal -> 0
    Anything else -> 0 (tie) to avoid poisoning learning.
    """
    t = (text or "").strip().lower()

    if t in {"1", "a"}:
        return 1
    if t in {"2", "b"}:
        return -1
    if t in {"0", "tie", "equal"}:
        return 0

    return 0


def llm_preference_for_driver(simulation_object, input_A, input_B):
    """
    Returns: (psi, s)
      psi = phi_A - phi_B
      s   = +1 if prefers A, -1 if prefers B, 0 tie
    """
    # A
    simulation_object.feed(input_A)
    phi_A = simulation_object.get_features()

    # B
    simulation_object.feed(input_B)
    phi_B = simulation_object.get_features()

    psi = np.array(phi_A) - np.array(phi_B)

    desc_A = describe_driver_features(phi_A)
    desc_B = describe_driver_features(phi_B)

    prompt = f"""
You are helping to compare two short car driving trajectories in a simple simulator.

The driver has four features:
1. Staying in lane (higher is better).
2. Speed deviation from the desired speed (lower is better).
3. Heading straightness (higher is better).
4. Collision risk (lower is better).

Here are the numeric summaries of the two trajectories:

Trajectory A:
{desc_A}

Trajectory B:
{desc_B}

Question:
Based on these summaries, which trajectory is better overall for natural, safe driving?

Answer STRICTLY with one symbol:
- "1" if Trajectory A is better
- "2" if Trajectory B is better
- "0" if they are equally good / cannot decide

Do NOT add any explanation, only return 0, 1, or 2.
"""

    try:
        raw = call_ollama_mistral(prompt)
        s = parse_preference_from_text(raw)
    except Exception as e:
        # If the LLM call fails, safest fallback is tie (0).
        print(f"[WARN] LLM call failed ({type(e).__name__}): {e} -> returning tie")
        s = 0

    # log this comparison for later feature selection
    if isinstance(simulation_object, Driver):
        _log_driver_choice(psi, s)

    return psi, s


# -----------------------------
# Uncertainty / routing metrics
# -----------------------------
def predictive_entropy(w_samples: np.ndarray, psi: np.ndarray) -> float:
    """
    Posterior predictive entropy for preference label under samples of w:
      p = E_w[ sigmoid(w^T psi) ]
      H(p) = -p log p - (1-p) log(1-p)
    Max is ln(2)=0.693...
    """
    psi = np.asarray(psi, dtype=float).reshape(-1)
    W = np.asarray(w_samples, dtype=float)
    logits = W @ psi
    p = 1.0 / (1.0 + np.exp(-logits))
    pbar = float(np.mean(p))
    eps = 1e-12
    pbar = min(max(pbar, eps), 1.0 - eps)
    return float(-(pbar * np.log(pbar) + (1.0 - pbar) * np.log(1.0 - pbar)))


def estimate_information_gain_from_inputs(simulation_object, input_A, input_B):
    """
    Fallback IG heuristic based only on ||psi||:
        psi  = phi(A) - phi(B)
        IG   = 1 / (1 + ||psi||)

    High IG means trajectories are similar -> (in your old logic) ask human.
    This is NOT posterior-aware. Prefer entropy routing when possible.
    """
    simulation_object.feed(input_A)
    phi_A = simulation_object.get_features()

    simulation_object.feed(input_B)
    phi_B = simulation_object.get_features()

    psi = np.array(phi_A) - np.array(phi_B)
    norm = float(np.linalg.norm(psi))
    info_gain = 1.0 / (1.0 + norm)

    print(f"[DEBUG] Driver IG heuristic: ||psi||={norm:.3f}, IG={info_gain:.3f}")
    return info_gain


# -----------------------------
# Mixed feedback function (main entry)
# -----------------------------
def get_feedback_mixed(
    task,
    simulation_object,
    input_A,
    input_B,
    query_index,
    batch_size,
    tracker=None,
    w_samples=None,
    force_human=False,
    routing_mode="topk",   # "topk" or "entropy"
    entropy_threshold=0.55,
):
    if force_human:
        psi, s = _human_feedback_psi(simulation_object, input_A, input_B)
        if tracker: tracker.record_query_source("human")
        print(f"[INFO] Driver query {query_index+1}/{batch_size}: FORCED HUMAN")
        return psi, s, "human"

    # if you're in topk mode, everything else should be LLM
    if routing_mode == "topk":
        psi, s = llm_preference_for_driver(simulation_object, input_A, input_B)
        if tracker: tracker.record_query_source("llm")
        print(f"[INFO] Driver query {query_index+1}/{batch_size}: TOPK MODE -> LLM")
        return psi, s, "llm"


    # Compute psi once
    simulation_object.feed(input_A)
    phi_A = simulation_object.get_features()
    simulation_object.feed(input_B)
    phi_B = simulation_object.get_features()
    psi = np.array(phi_A) - np.array(phi_B)

    # Posterior-aware routing
    if w_samples is not None:
        H = predictive_entropy(w_samples, psi)

        # human if uncertain; llm if confident
        if H > float(entropy_threshold):
            print(
                f"[INFO] Driver query {query_index+1}/{batch_size}: "
                f"H={H:.3f} > {entropy_threshold:.2f} -> HUMAN"
            )
            psi2, s = _human_feedback_psi(simulation_object, input_A, input_B)
            source = "human"
        else:
            print(
                f"[INFO] Driver query {query_index+1}/{batch_size}: "
                f"H={H:.3f} <= {entropy_threshold:.2f} -> LLM"
            )
            psi2, s = llm_preference_for_driver(simulation_object, input_A, input_B)
            source = "llm"

        if tracker is not None:
            tracker.record_query_source(source)
        return psi2, s, source

    # Fallback heuristic routing (no w_samples)
    info_gain = 1.0 / (1.0 + float(np.linalg.norm(psi)))
    if info_gain > INFO_GAIN_THRESHOLD:
        print(
            f"[INFO] Driver query {query_index+1}/{batch_size}: "
            f"IG={info_gain:.3f} > {INFO_GAIN_THRESHOLD:.2f} -> HUMAN"
        )
        psi2, s = _human_feedback_psi(simulation_object, input_A, input_B)
        source = "human"
    else:
        print(
            f"[INFO] Driver query {query_index+1}/{batch_size}: "
            f"IG={info_gain:.3f} <= {INFO_GAIN_THRESHOLD:.2f} -> LLM"
        )
        psi2, s = llm_preference_for_driver(simulation_object, input_A, input_B)
        source = "llm"

    if tracker is not None:
        tracker.record_query_source(source)
    return psi2, s, source


# -----------------------------
# Extra feature suggestion (optional)
# -----------------------------
def _summarize_driver_history():
    if not DRIVER_HISTORY:
        return "No comparisons have been collected yet."

    num = len(DRIVER_HISTORY)
    num_pref_A = sum(1 for h in DRIVER_HISTORY if h["s"] == 1)
    num_pref_B = sum(1 for h in DRIVER_HISTORY if h["s"] == -1)
    num_ties = sum(1 for h in DRIVER_HISTORY if h["s"] == 0)

    psi_mat = np.array([h["psi"] for h in DRIVER_HISTORY], dtype=float)
    mean_abs_psi = np.mean(np.abs(psi_mat), axis=0)

    summary = []
    summary.append(f"Total comparisons: {num}")
    summary.append(f"Preferences: {num_pref_A} for A, {num_pref_B} for B, {num_ties} ties.")
    summary.append("Mean absolute psi per dimension (current features):")
    summary.append(f"  dim1 (staying in lane):        {mean_abs_psi[0]:.4f}")
    summary.append(f"  dim2 (speed deviation):        {mean_abs_psi[1]:.4f}")
    summary.append(f"  dim3 (heading):                {mean_abs_psi[2]:.4f}")
    summary.append(f"  dim4 (collision avoidance):    {mean_abs_psi[3]:.4f}")

    return "\n".join(summary)


def suggest_extra_feature_from_history(min_samples: int = 20) -> str:
    """
    Ask the LLM whether we should add a 5th feature for the driver task,
    based on the accumulated DRIVER_HISTORY.

    Returns: feature_id in EXTRA_FEATURE_BANK keys, or 'none'
    """
    if len(DRIVER_HISTORY) < min_samples:
        return "none"

    history_summary = _summarize_driver_history()

    options_lines = []
    for fid, meta in EXTRA_FEATURE_BANK.items():
        if fid == "none":
            continue
        options_lines.append(f"- id='{fid}': {meta['name']} — {meta['description']}")
    options_text = "\n".join(options_lines) if options_lines else "No extra features defined."

    prompt = f"""
You are helping design a reward model for a car driving preference-elicitation system.

The current model uses FOUR features:
1. Staying in lane (higher is better).
2. Speed deviation from the desired speed (lower is better).
3. Heading straightness (higher is better).
4. Collision risk (lower is better).

We are running pairwise comparisons between trajectories.
Each comparison yields:
  psi = phi_A - phi_B   (4D difference in these features)
  s   in {{1, -1, 0}}   (1: prefers A, -1: prefers B, 0: tie / no preference).

Here is a summary of all comparisons so far:
{history_summary}

We suspect that the four current features might not fully capture the preferences.
We have a BANK of candidate extra features we could add as a FIFTH feature:

{options_text}

Question:
Based on the summary, decide whether adding ONE extra feature from the bank would likely
help explain the preferences better. If NONE of them clearly help, answer with "none".

You MUST answer with exactly ONE token, matching either:
- one of the feature ids in the list above
- or "none"

Do NOT add explanations, just return the id.
"""

    try:
        raw = call_ollama_mistral(prompt).strip().lower()
    except Exception as e:
        print(f"[WARN] LLM feature suggestion failed ({type(e).__name__}): {e} -> 'none'")
        return "none"

    if "none" in raw:
        return "none"

    for fid in EXTRA_FEATURE_BANK.keys():
        if fid != "none" and fid.lower() in raw:
            return fid

    return "none"
