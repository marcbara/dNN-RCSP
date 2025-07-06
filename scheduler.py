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
import numpy as np





class DatalessProjectScheduler:
    def __init__(
        self,
        durations,
        precedences,
        # Conservative starts
        beta=5.0,
        penalty_init=10.0,
        
        # Less aggressive adaptive mechanisms
        beta_sharpening_threshold=0.1,        # Wait until violations much smaller
        violation_escalation_threshold=0.01,   # Less aggressive λ escalation  
        recovery_sensitivity=3.0,             # Less sensitive recovery
        penalty_beta_threshold=10000,         # Higher threshold for β sharpening
        
        # Penalty reduction (make it more reluctant)
        penalty_reduction_factor=0.95,        # Reduce λ more slowly (was 0.9)
        violation_reduction_threshold=1e-5,   # Much stricter threshold to reduce λ (was 1e-4)
        
        # Trend analysis (make it more stable)
        lookback_window=50,                   # Longer trend analysis (was 20)
        trend_window=20,                      # Longer recovery detection (was 10)  
        min_history_for_recovery=100,         # Wait longer before recovery kicks in (was 30)
        
        # Legacy parameters (probably not used but keep conservative)
        penalty_anneal=1.2,                   # Slower escalation if used (was 1.5)
        anneal_every=100,                     # Less frequent if used (was 50)
        
        # Your settings
        feasible_threshold=1e-3,
        early_stop_tol=0.5e-3,
        enable_early_stopping=False,
        device="cpu"
    ):
        """
        durations   : list/array of activity durations
        precedences : list of (pred_idx, succ_idx) tuples
        
        Key hyperparameters for adaptive penalty method:
        - feasible_threshold: store candidates when max_viol < this (default: 1e-6)
        - violation_reduction_threshold: reduce λ only when violations < this (default: 0.001)
        - violation_escalation_threshold: escalate λ when violations > this (default: 0.01)
        - recovery_sensitivity: trigger recovery when violations increase by this factor (default: 2.0)
        - beta_sharpening_threshold: trigger β doubling when violations < this (default: 2.0)
        - penalty_beta_threshold: trigger β doubling when λ > this (default: 1000)
        - trend_window: epochs to look back for recovery detection (default: 10)
        - min_history_for_recovery: minimum epochs before recovery mechanism activates (default: 30)
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
        self.penalty_init = penalty_init  # Store original for lower bound
        self.penalty_anneal = penalty_anneal
        self.anneal_every = anneal_every
        self.early_stop_tol = early_stop_tol
        self.enable_early_stopping = enable_early_stopping
        
        # Simple parameters for all problems
        self.penalty_reduction_factor = penalty_reduction_factor
        self.lookback_window = lookback_window
        self.violation_reduction_threshold = violation_reduction_threshold
        self.violation_escalation_threshold = violation_escalation_threshold
        self.recovery_sensitivity = recovery_sensitivity
        self.beta_sharpening_threshold = beta_sharpening_threshold
        self.penalty_beta_threshold = penalty_beta_threshold
        self.trend_window = trend_window
        self.min_history_for_recovery = min_history_for_recovery

        # adaptive penalty tracking
        self.viol_history = []
        
        # feasible solution tracking
        self.feasible_candidates = []
        self.feasible_threshold = feasible_threshold  # Store candidates when max_viol < this

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
        Gradient-descent optimiser with adaptive λ management and early β sharpening.
        * λ escalates when violations are large, reduces when stable and small
        * β doubles once λ is large OR violations are small
        * LR is gently reheated after λ saturates to avoid Adam stalls
        * Early stopping when enable_early_stopping=True and MAX individual violation < early_stop_tol time units
        """
        opt = torch.optim.Adam([self.raw_times], lr=lr)

        loss_hist, span_hist   = [], []
        viol_hist, lambda_hist = [], []
        max_viol_hist          = []
        gnorm_hist             = []
        PENALTY_CAP = 10_000_000  # 1e7 - conservative cap to prevent FP32 numerical issues

        for ep in range(epochs):
            opt.zero_grad()
            loss, span, viol_mean, viol_raw = self.forward()
            loss.backward()

            # No gradient clipping - let Adam handle large gradients naturally
            # (Previous clipping at 10.0 was destroying penalty signals when λ > 1e6)
            
            gnorm = torch.norm(self.raw_times.grad).item()
            opt.step()

            # ── 1️⃣  Adaptive λ management (based on MAX violations) ─────────────────────────────────────
            # Track MAX violation history for trend analysis (not mean - max determines feasibility!)
            max_viol = torch.max(viol_raw).item()
            self.viol_history.append(max_viol)
            
            if len(self.viol_history) >= self.lookback_window:
                # Calculate violation improvement rate over lookback window
                recent_window = self.viol_history[-self.lookback_window:]
                earlier_window = self.viol_history[-self.lookback_window:-self.lookback_window//2]
                
                recent_avg = sum(recent_window) / len(recent_window)
                earlier_avg = sum(earlier_window) / len(earlier_window)
                
                # Calculate improvement rate (positive = getting better)
                improvement_rate = (earlier_avg - recent_avg) / max(earlier_avg, 1e-8)
                
                # Simple robust logic with safety valve and recovery
                min_penalty = self.penalty_init * 10  # Simple safety valve
                
                # Check if violations are trending upward (need recovery)
                if len(self.viol_history) >= self.min_history_for_recovery:
                    very_recent = sum(self.viol_history[-self.trend_window:]) / self.trend_window
                    less_recent = sum(self.viol_history[-2*self.trend_window:-self.trend_window]) / self.trend_window
                    
                    # If violations are increasing, boost λ immediately (RECOVERY MODE)
                    if very_recent > less_recent * self.recovery_sensitivity and self.penalty < PENALTY_CAP:
                        old_penalty = self.penalty
                        self.penalty = min(self.penalty * 2, PENALTY_CAP)

                    # If MAX violations are small and stable, reduce penalty (with safety valve)
                    elif recent_avg < self.violation_reduction_threshold:
                        old_penalty = self.penalty
                        self.penalty = max(self.penalty * self.penalty_reduction_factor, min_penalty)

                    # If MAX violations are moderate to large, escalate aggressively
                    elif max_viol > self.violation_escalation_threshold and self.penalty < PENALTY_CAP:
                        self.penalty *= 2
                # Standard escalation for early epochs
                elif max_viol > self.violation_escalation_threshold and self.penalty < PENALTY_CAP:
                    self.penalty *= 2
            else:
                # Very early epochs (before lookback window): use original escalation logic
                if max_viol > self.violation_escalation_threshold and self.penalty < PENALTY_CAP:
                    self.penalty *= 2

            # ── 2️⃣  β sharpening  (as soon as λ big OR max violations small) ──────────
            if ((self.penalty > self.penalty_beta_threshold) or (max_viol < self.beta_sharpening_threshold)) and self.beta < 1e4:
                self.beta *= 2

            # ── 3️⃣  LR reheating once λ has frozen  ──────────────────────────
            # DISABLED: Testing if LR reheating causes violation spikes
            # if self.penalty >= PENALTY_CAP and (ep + 1) % 50 == 0:
            #     for g in opt.param_groups:
            #         g["lr"] *= 1.1            # small boost to keep Adam moving

            # ---- history logging --------------------------------------------
            loss_hist.append(loss.item())
            span_hist.append(span.item())
            viol_hist.append(viol_mean.item())
            lambda_hist.append(self.penalty)
            gnorm_hist.append(gnorm)

            # ── 4️⃣  Store feasible candidates and early stopping ─────────────────
            max_viol_hist.append(max_viol)
            
            # Store feasible candidate when violations are essentially zero
            if max_viol < self.feasible_threshold:
                current_sol = self.solution()
                candidate = {
                    'epoch': ep,
                    'makespan': current_sol['makespan'],
                    'start_times': current_sol['start'].copy(),
                    'end_times': current_sol['end'].copy(),
                    'max_violation': max_viol,
                    'mean_violation': viol_mean.item()
                }
                self.feasible_candidates.append(candidate)
            
            if self.enable_early_stopping and max_viol < self.early_stop_tol:
                if verbose:
                    print(f"Early stopping at epoch {ep}: max violation {max_viol:.3f} < {self.early_stop_tol} time units")
                break

            if verbose and ep % 100 == 0:
                print(f"ep {ep:4d} | span {span.item():6.2f} | "
                    f"viol {viol_mean.item():.2e} | λ {self.penalty:.1e} | "
                    f"∥grad∥ {gnorm:.2f}")

        # ── 5️⃣  Select and restore best feasible candidate ─────────────────
        selected_candidate = self.select_best_candidate(verbose)
        if selected_candidate is not None:
            self.restore_candidate(selected_candidate)
            # Store the best candidate as the current solution state
            self.best_solution = selected_candidate
            if verbose:
                print(f"✓ BEST SOLUTION: epoch {selected_candidate['epoch']}, "
                      f"makespan {selected_candidate['makespan']:.3f}, "
                      f"max_viol {selected_candidate['max_violation']:.2e}")
        else:
            # No feasible candidates - use current solution
            self.best_solution = None

        return (loss_hist, span_hist, viol_hist, lambda_hist, gnorm_hist, max_viol_hist)

    def select_best_candidate(self, verbose=True):
        """Select the best feasible candidate (minimum makespan among feasible solutions)"""
        if not self.feasible_candidates:
            if verbose:
                print("⚠ No feasible candidates found - using final solution")
            return None
        
        # Among feasible solutions, select the one with minimum makespan
        best_candidate = min(self.feasible_candidates, key=lambda x: x['makespan'])
        
        if verbose:
            print(f"📊 Found {len(self.feasible_candidates)} feasible candidates")
            print(f"   Best makespan: {best_candidate['makespan']:.3f} at epoch {best_candidate['epoch']}")
            makespan_range = (
                min(c['makespan'] for c in self.feasible_candidates),
                max(c['makespan'] for c in self.feasible_candidates)
            )
            print(f"   Makespan range: {makespan_range[0]:.3f} - {makespan_range[1]:.3f}")
        
        return best_candidate
    
    def restore_candidate(self, candidate):
        """Restore the scheduler state to the given candidate solution"""
        with torch.no_grad():
            # Calculate raw_times from start_times (reverse the _start_times transformation)
            start_times = torch.tensor(candidate['start_times'], device=self.device, dtype=torch.float32)
            
            # Since _start_times does: softplus(raw_times) - min(softplus(raw_times))
            # We need to find raw_times that produce the desired start_times
            # Simple approach: use inverse softplus (log(exp(x) - 1)) and add offset
            
            # Inverse softplus: log(exp(x + ε) - 1) with epsilon inside exp for numerical stability
            # This prevents log(0) even for tiny start times
            raw_times_est = torch.log(torch.exp(start_times + 1e-6) - 1.0)
            
            # Update the raw_times parameter
            self.raw_times.data = raw_times_est

    # ---- utilities ------------------------------------------------------
    def solution(self):
        # If we have a best solution from feasible candidates, use it
        if hasattr(self, 'best_solution') and self.best_solution is not None:
            return {
                'start': self.best_solution['start_times'],
                'end': self.best_solution['end_times'],
                'makespan': self.best_solution['makespan'],
                'max_violation': self.best_solution['max_violation'],
                'mean_violation': self.best_solution['mean_violation']
            }
        
        # Otherwise, calculate from current state
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

    # Add a new simplified solve method
    def solve_simple(self, epochs: int = 400, lr: float = 0.1, verbose: bool = True, fixed_penalty: float = 10_000_000):
        """
        Simplified solver with fixed high penalty - no complex adaptive mechanism.
        Use this when violations plateau and adaptive λ becomes useless.
        """
        self.penalty = fixed_penalty  # Set high penalty from start
        opt = torch.optim.Adam([self.raw_times], lr=lr)
        
        loss_hist, span_hist   = [], []
        viol_hist, lambda_hist = [], []
        max_viol_hist          = []
        gnorm_hist             = []
        
        for ep in range(epochs):
            opt.zero_grad()
            loss, span, viol_mean, viol_raw = self.forward()
            loss.backward()
            
            # No gradient clipping - let Adam handle large gradients naturally
            gnorm = torch.norm(self.raw_times.grad).item()
            opt.step()
            
            # Simple β sharpening when max violations get small
            max_viol = torch.max(viol_raw).item()
            if max_viol < 0.1 and self.beta < 1e4:
                self.beta *= 2
            
            # ---- history logging --------------------------------------------
            loss_hist.append(loss.item())
            span_hist.append(span.item())
            viol_hist.append(viol_mean.item())
            lambda_hist.append(self.penalty)
            gnorm_hist.append(gnorm)
            
            # ── Store feasible candidates (with relaxed threshold) ─────────────────
            max_viol_hist.append(max_viol)
            
            # Store "practical feasible" candidates when violations < 0.1 time units
            if max_viol < 0.1:  # Much more relaxed threshold
                current_sol = self.solution()
                candidate = {
                    'epoch': ep,
                    'makespan': current_sol['makespan'],
                    'start_times': current_sol['start'].copy(),
                    'end_times': current_sol['end'].copy(),
                    'max_violation': max_viol,
                    'mean_violation': viol_mean.item()
                }
                self.feasible_candidates.append(candidate)
            
            if verbose and ep % 100 == 0:
                print(f"ep {ep:4d} | span {span.item():6.2f} | "
                      f"viol {viol_mean.item():.2e} | λ {self.penalty:.1e} | "
                      f"∥grad∥ {gnorm:.2f}")
        
        # Select and restore best candidate
        selected_candidate = self.select_best_candidate(verbose)
        if selected_candidate is not None:
            self.restore_candidate(selected_candidate)
            self.best_solution = selected_candidate
            if verbose:
                print(f"✓ BEST SOLUTION: epoch {selected_candidate['epoch']}, "
                      f"makespan {selected_candidate['makespan']:.3f}, "
                      f"max_viol {selected_candidate['max_violation']:.2e}")
        else:
            self.best_solution = None
        
        return (loss_hist, span_hist, viol_hist, lambda_hist, gnorm_hist, max_viol_hist)


# Example usage and basic validation
if __name__ == "__main__":
    print("DatalessProjectScheduler module loaded successfully!")
    print("Import this module to use the scheduler in your experiments.")
    print("\nExample usage:")
    print("  from scheduler import DatalessProjectScheduler")
    print("  scheduler = DatalessProjectScheduler(durations, precedences)")
    print("  scheduler.solve()")
    print("  scheduler.show()")
