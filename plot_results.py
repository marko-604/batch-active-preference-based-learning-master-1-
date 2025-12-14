# plot_results.py
import json
import sys
import os

import numpy as np
import matplotlib.pyplot as plt


def load_results(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def plot_alignment(result_dict, label=None):
    hist = result_dict.get("history", [])
    xs = []
    ys = []
    for entry in hist:
        a = entry.get("alignment", None)
        if a is not None:
            xs.append(entry["num_queries"])
            ys.append(a)
    if not xs:
        print("[WARN] No alignment data found.")
        return
    xs = np.array(xs)
    ys = np.array(ys)
    plt.plot(xs, ys, marker="o", label=label or "alignment")
    plt.xlabel("Number of queries")
    plt.ylabel("Alignment")
    plt.ylim(-1.0, 1.0)
    plt.grid(True)


def plot_log_likelihood(result_dict, label=None):
    hist = result_dict.get("history", [])
    xs = []
    ys = []
    for entry in hist:
        ll = entry.get("avg_log_likelihood", None)
        if ll is not None:
            xs.append(entry["num_queries"])
            ys.append(ll)
    if not xs:
        print("[WARN] No log-likelihood data found.")
        return
    xs = np.array(xs)
    ys = np.array(ys)
    plt.plot(xs, ys, marker="o", label=label or "avg log-likelihood")
    plt.xlabel("Number of queries")
    plt.ylabel("Average log-likelihood")
    plt.grid(True)


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_results.py results_file1.json [results_file2.json ...]")
        sys.exit(1)

    paths = sys.argv[1:]

    # --- Plot alignment ---
    plt.figure()
    for p in paths:
        res = load_results(p)
        label = os.path.basename(p)
        plot_alignment(res, label=label)
    plt.legend()
    plt.title("Alignment vs Number of Queries")
    plt.tight_layout()
    plt.savefig("FINAL_FINAL_allign.png", dpi=200)
    print("[Plot] Saved alignment_vs_queries_3_w_ent.png")

    # --- Plot avg log-likelihood ---
    plt.figure()
    for p in paths:
        res = load_results(p)
        label = os.path.basename(p)
        plot_log_likelihood(res, label=label)
    plt.legend()
    plt.title("Average Log-Likelihood vs Number of Queries")
    plt.tight_layout()
    plt.savefig("FINAL_FINAL_log.png", dpi=200)
    print("[Plot] Saved log_3_with_ent.png")

    plt.show()


if __name__ == "__main__":
    main()
