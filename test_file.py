#!/usr/bin/env python
"""
test_file.py
Read a project definition (JSON) and run the dataless scheduler.
Figures are saved in ./figs at 300 dpi.
"""

import os, sys, argparse, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scheduler import DatalessProjectScheduler
from project_loader import load_project


# ------------------- helper to save figures -------------------
def save_fig(fig, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"[saved] {fname}")


def save_convergence(loss, span, viol, lam, gnorm, max_viol, fname, early_stop_tol=0.01):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(11, 6), sharex=True)

    # 1️⃣  Loss (linear)
    axs[0,0].plot(loss)
    axs[0,0].set_title("Loss"); axs[0,0].grid(alpha=.3)

    # 2️⃣  Makespan
    axs[0,1].plot(span, color="tab:red")
    axs[0,1].set_title("Makespan"); axs[0,1].grid(alpha=.3)

    # 3️⃣  Violations (log scale) - Mean vs Max
    axs[1,0].semilogy(viol, color="tab:green", alpha=0.7, label="Mean")
    axs[1,0].semilogy(max_viol, color="tab:red", label="Max (early stopping)")
    axs[1,0].axhline(y=early_stop_tol, color="red", linestyle="--", alpha=0.5, label="Early stop threshold")
    axs[1,0].set_title("Violations (log)")
    axs[1,0].grid(alpha=.3, which="both")
    axs[1,0].legend()

    # 4️⃣  λ and ‖grad‖  —> **log y-axis**
    ax4 = axs[1,1]
    ax4.plot(lam,   color="tab:purple", label="λ")
    ax4.plot(gnorm, color="tab:gray",   label="‖grad‖", alpha=.6)
    ax4.set_yscale("log")               # ← NEW LINE
    ax4.set_title("λ and gradient norm (log-y)")
    ax4.grid(alpha=.3, which="both")
    ax4.legend()

    for ax in axs.ravel(): ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"[saved] {fname}")




def save_gantt(sched, names, fname):
    sol = sched.solution()
    n   = len(names); dur = sched.durations.cpu().numpy()
    fig_h = 2 + 0.25 * n
    fig, ax = plt.subplots(figsize=(10, fig_h))
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    for i, nm in enumerate(names):
        if dur[i] > 0:
            ax.barh(i, dur[i], left=sol['start'][i],
                    color=colors[i], edgecolor="black", alpha=.7)
            ax.text(sol['start'][i] + dur[i]/2, i, nm,
                    ha="center", va="center", fontsize=8)
    ax.set_yticks(range(n)), ax.set_yticklabels(names), ax.invert_yaxis()
    ax.set_xlabel("time"), ax.set_title("Gantt Chart")
    plt.tight_layout(); save_fig(fig, fname)


# ------------------- main -------------------
if __name__ == "__main__":

    # ---- friendly early-exit guard ---------------------------
    if len(sys.argv) == 1:          # only the script name present
        print("Usage:  python test_file.py <project.json>")
        print("Error : You must supply a project file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Run dataless scheduler on a JSON file.")
    parser.add_argument("file", help="Path to project JSON (durations, precedences, names)")
    parser.add_argument("--epochs", type=int, default=400, help="training epochs")
    parser.add_argument("--lr",     type=float, default=0.1, help="learning rate")
    args = parser.parse_args()

    durations, precedences, names = load_project(args.file)

    sched = DatalessProjectScheduler(durations, precedences)
    (loss_hist, span_hist,
    viol_hist, lambda_hist,
    gnorm_hist, max_viol_hist) = sched.solve(epochs=args.epochs,
                                          lr=args.lr,
                                          verbose=True)

    sched.show(names)

    base = os.path.splitext(os.path.basename(args.file))[0]
    save_convergence(loss_hist, span_hist,
                 viol_hist, lambda_hist,
                 gnorm_hist, max_viol_hist,
                 f"figs/{base}_conv.png", sched.early_stop_tol)

    save_gantt(sched, names, f"figs/{base}_gantt.png")

    print("All figures saved in ./figs/ (300 dpi)")
