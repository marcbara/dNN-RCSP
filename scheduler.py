"""
Dataless Neural Networks for Project Scheduling
===============================================

Core implementation of the DatalessProjectScheduler class.
This module provides the main scheduling class and methods.

Based on the dataless neural network approach from:
- Alkhouri, I. R., Atia, G. K., & Velasquez, A. (2022). 
  A differentiable approach to the maximum independent set problem using dataless neural networks.

Author: Marc Bara
Date: 06/July/2025
"""

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np





class DatalessProjectScheduler:
    def __init__(
        self,
        durations,
        precedences,
        beta=5.0,                  # smooth-max temperature
        penalty_init=10.0,         # initial λ for precedence violations
        penalty_anneal=1.5,        # multiply λ every `anneal_every` epochs
        anneal_every=50,
        early_stop_tol=1e-6,       # early stopping when max violation < this
        device="cpu"
    ):
        """
        durations   : list/array of activity durations
        precedences : list of (pred_idx, succ_idx) tuples
        """
        self.device     = torch.device(device)                # store once
        self.durations  = torch.as_tensor(durations, dtype=torch.float32).to(self.device)
        self.n = len(durations)


        # precedence tensors (vectorised)
        self.pred_idx = torch.tensor([p for p, _ in precedences], device=device, dtype=torch.long)
        self.succ_idx = torch.tensor([s for _, s in precedences], device=device, dtype=torch.long)

        # trainable raw parameters (softplus will keep them ≥0)
        self.raw_times = torch.nn.Parameter(torch.zeros(self.n, device=device))

        # hyper-params
        self.beta = beta
        self.penalty = penalty_init
        self.penalty_anneal = penalty_anneal
        self.anneal_every = anneal_every
        self.early_stop_tol = early_stop_tol

        # utility: index of START assumed 0
        self.start_idx = 0


    # ---- helper: convert raw params -> feasible-ish start times ----------
    def _start_times(self):
        s = torch.nn.functional.softplus(self.raw_times)     # ≥0
        s = s - s.min()                                      # anchor earliest at 0
        return s

    # ---- forward: smooth makespan + squared violation penalty -----------
    def forward(self):
        s = self._start_times()                              # (n,)
        finish = s + self.durations                          # (n,)

        # smooth max (“log-sum-exp”) for nicer gradients
        makespan = torch.logsumexp(finish * self.beta, dim=0) / self.beta

        # vectorised precedence violations
        pred_finish = finish[self.pred_idx]                  # |P|
        succ_start  = s[self.succ_idx]

        # --- smoother, always-gradient penalty -----------------------
        def smooth_penalty(v: torch.Tensor, thresh: float = 1.0) -> torch.Tensor:
            """Quadratic when v<thresh, linear tail after – keeps gradients alive."""
            quad   = 0.5 * v**2
            linear = thresh * (v - 0.5 * thresh)
            return torch.where(v < thresh, quad, linear)

        viol = torch.relu(pred_finish - succ_start)          # |P|
        penalty_term = smooth_penalty(viol).mean()

        total_loss = makespan + self.penalty * penalty_term
        return total_loss, makespan, penalty_term, viol

    # ------------------------------------------------------------------
    def solve(self, epochs: int = 400, lr: float = 0.1, verbose: bool = True):
        """
        Gradient-descent optimiser with rapid λ escalation and early β sharpening.
        * λ doubles every epoch while mean-slack > 1 (cap 4 e6)
        * β doubles once λ is large OR slack < 2
        * LR is gently reheated after λ saturates to avoid Adam stalls
        * Early stopping when MAX individual violation < early_stop_tol time units
        """
        opt = torch.optim.Adam([self.raw_times], lr=lr)

        loss_hist, span_hist   = [], []
        viol_hist, lambda_hist = [], []
        max_viol_hist          = []
        gnorm_hist             = []
        PENALTY_CAP = 100_000_000

        for ep in range(epochs):
            opt.zero_grad()
            loss, span, viol_mean, viol_raw = self.forward()
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.raw_times, max_norm=10.0)
            
            gnorm = torch.norm(self.raw_times.grad).item()
            opt.step()

            # ── 1️⃣  λ escalation  (every epoch until mean-slack ≤ 1) ─────────
            if viol_mean.item() > 1.0 and self.penalty < PENALTY_CAP:
                self.penalty *= 2

            # ── 2️⃣  β sharpening  (as soon as λ big OR slack small) ──────────
            if ((self.penalty > 1_000) or (viol_mean.item() < 2.0)) and self.beta < 1e4:
                self.beta *= 2

            # ── 3️⃣  LR reheating once λ has frozen  ──────────────────────────
            if self.penalty >= PENALTY_CAP and (ep + 1) % 50 == 0:
                for g in opt.param_groups:
                    g["lr"] *= 1.1            # small boost to keep Adam moving

            # ---- history logging --------------------------------------------
            loss_hist.append(loss.item())
            span_hist.append(span.item())
            viol_hist.append(viol_mean.item())
            lambda_hist.append(self.penalty)
            gnorm_hist.append(gnorm)

            # ── 4️⃣  Early stopping when MAX violation is tiny ─────────────────
            max_viol = torch.max(viol_raw).item()
            max_viol_hist.append(max_viol)
            if max_viol < self.early_stop_tol:
                if verbose:
                    print(f"Early stopping at epoch {ep}: max violation {max_viol:.3f} < {self.early_stop_tol} time units")
                break

            if verbose and ep % 100 == 0:
                print(f"ep {ep:4d} | span {span.item():6.2f} | "
                    f"viol {viol_mean.item():.2e} | λ {self.penalty:.1e} | "
                    f"∥grad∥ {gnorm:.2f}")

        return (loss_hist, span_hist, viol_hist, lambda_hist, gnorm_hist, max_viol_hist)




    # ---- utilities ------------------------------------------------------
    def solution(self):
        with torch.no_grad():
            s = self._start_times().cpu().numpy()
            e = s + self.durations.cpu().numpy()
            return dict(start=s, end=e, makespan=e.max())

    def clean_schedule(self, tol=1e-3):
        sol = self.solution()
        sol['start']    = np.where(sol['start']  < tol, 0.0, sol['start'])
        sol['end']      = np.where(sol['end']    < tol, 0.0, sol['end'])
        sol['makespan'] = round(sol['makespan'], 2)
        return sol




    def print_solution(self, names=None):
        sol = self.solution()
        names = names or [f"Act {i}" for i in range(self.n)]
        print("\nFinal schedule".center(40, "═"))
        for i, nm in enumerate(names):
            print(f"{nm:10s}: {sol['start'][i]:5.2f} → {sol['end'][i]:5.2f}")
        print("─" * 40)
        print(f"Makespan: {sol['makespan']:.2f}")

    # ---------------------------------------------
    # pretty printer
    # ---------------------------------------------
    def show(self, names=None):
        names = names or [f"Act {i}" for i in range(self.n)]
        sol   = self.solution()
        title = "CONTINUOUS SCHEDULE"
        print(f"\n{title.center(46, '═')}")
        for i, nm in enumerate(names):
            print(f"{nm:10s}: {sol['start'][i]:6.2f} → {sol['end'][i]:6.2f}")
        print("─" * 46)
        print(f"Makespan: {sol['makespan']:.2f}")

    # ---------------------------------------------
    # Gantt wrapper
    # ---------------------------------------------
    def plot(self, names=None):
        sol   = self.solution()
        names = names or [f"Act {i}" for i in range(self.n)]
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots(figsize=(8, .5 * self.n + 2))
        colors = plt.cm.Set3(np.linspace(0, 1, self.n))
        for i, nm in enumerate(names):
            if self.durations[i] > 0:
                ax.barh(i, self.durations[i], left=sol['start'][i],
                        color=colors[i], edgecolor="black", alpha=.7)
                ax.text(sol['start'][i] + self.durations[i] / 2, i, nm,
                        ha="center", va="center", fontsize=8)
        ax.set_yticks(range(self.n)); ax.set_yticklabels(names); ax.invert_yaxis()
        ax.set_xlabel("time"); ax.set_title("Gantt Chart")
        plt.tight_layout(); plt.show()


# Example usage and basic validation
if __name__ == "__main__":
    print("DatalessProjectScheduler module loaded successfully!")
    print("Import this module to use the scheduler in your experiments.")
    print("\nExample usage:")
    print("  from scheduler import DatalessProjectScheduler")
    print("  scheduler = DatalessProjectScheduler(durations, precedences)")
    print("  scheduler.solve()")
    print("  scheduler.show()")
