#!/usr/bin/env python
"""
find_solution.py

"""

import os, sys, argparse, matplotlib, json
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

from scheduler import DatalessProjectScheduler
from project_loader import load_project


class SolutionAnalyzer:
    """Analyze and save solutions professionally"""
    
    def __init__(self, scheduler, names, project_file):
        self.scheduler = scheduler
        self.names = names
        self.project_file = project_file
        
    def analyze_solution(self, histories):
        """Analyze the final solution and optimization history"""
        loss_hist, span_hist, viol_hist, lambda_hist, gnorm_hist, max_viol_hist = histories
        
        # Get final solution
        sol = self.scheduler.solution()
        
        # Get best solution info from feasible candidate system if available
        if hasattr(self.scheduler, 'best_solution') and self.scheduler.best_solution is not None:
            # Use the actual selected feasible candidate
            best_viol_epoch = self.scheduler.best_solution['epoch']
            best_max_viol = self.scheduler.best_solution['max_violation']
            best_mean_viol = self.scheduler.best_solution['mean_violation']
            best_span_at_best_viol = self.scheduler.best_solution['makespan']
            num_feasible_candidates = len(self.scheduler.feasible_candidates)
        else:
            # Fallback to history analysis (when no feasible candidates found)
            best_viol_idx = np.argmin(max_viol_hist)
            best_viol_epoch = best_viol_idx
            best_max_viol = max_viol_hist[best_viol_idx]
            best_mean_viol = viol_hist[best_viol_idx]
            best_span_at_best_viol = span_hist[best_viol_idx]
            num_feasible_candidates = 0
        
                # Get violation information from solution or history
        if 'max_violation' in sol and 'mean_violation' in sol:
            # Use stored violation information from best solution
            final_max_viol = float(sol['max_violation'])
            final_mean_viol = float(sol['mean_violation'])
        else:
            # Fallback to history (current state)
            final_max_viol = float(max_viol_hist[-1])
            final_mean_viol = float(viol_hist[-1])
        
        # Create comprehensive metrics
        metrics = {
            'final_solution': {
                'makespan': round(float(sol['makespan']), 3),
                'max_violation': round(final_max_viol, 6),
                'mean_violation': round(final_mean_viol, 6),
                'feasible': bool(final_max_viol < 1e-6),
                'solution_epoch': int(best_viol_epoch),
                'total_epochs_run': len(span_hist),
                'feasible_candidates_found': num_feasible_candidates
            },
            'resource_analysis': {
                'total_activities': len(self.names),
                'activities_with_duration': sum(1 for i in range(len(self.names)) if self.scheduler.durations[i] > 0),
                'max_activity_duration': float(self.scheduler.durations.max()),
                'total_work_content': float(self.scheduler.durations.sum())
            }
        }
        
        return metrics
        
    def save_solution(self, metrics, filename):
        """Save comprehensive solution record"""
        sol = self.scheduler.solution()
        
        # Create solution record with reasonable precision (3 decimal places)
        solution_record = {
            'metadata': {
                'project_file': self.project_file,
                'timestamp': datetime.now().isoformat(),
                'solver': 'DatalessProjectScheduler',
                'version': '1.0'
            },
            'solution': {
                'start_times': {name: round(float(sol['start'][i]), 3) for i, name in enumerate(self.names)},
                'finish_times': {name: round(float(sol['end'][i]), 3) for i, name in enumerate(self.names)},
                'makespan': round(float(sol['makespan']), 3)
            },
            'metrics': metrics,
            'activity_schedule': [
                {
                    'name': name,
                    'start': round(float(sol['start'][i]), 3),
                    'finish': round(float(sol['end'][i]), 3),
                    'duration': float(self.scheduler.durations[i])
                }
                for i, name in enumerate(self.names)
                if self.scheduler.durations[i] > 0  # Only include real activities
            ]
        }
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(solution_record, f, indent=2)
            
        print(f"[saved] {filename}")
        return solution_record
        
    def print_solution_summary(self, metrics):
        """Print comprehensive solution summary"""
        print("\n" + "="*70)
        print("                      SOLUTION SUMMARY")
        print("="*70)
        
        # Final solution quality
        final = metrics['final_solution']
        print(f"Final makespan:        {final['makespan']:.3f}")
        print(f"Max violation:         {final['max_violation']:.2e}")
        print(f"Mean violation:        {final['mean_violation']:.2e}")
        print(f"Feasible:              {'✓ YES' if final['feasible'] else '✗ NO'}")
        print(f"Solution epoch:        {final['solution_epoch']}")
        print(f"Total epochs run:      {final['total_epochs_run']}")
        print(f"Feasible candidates:   {final['feasible_candidates_found']}")
        
        # Resource analysis
        res = metrics['resource_analysis']
        print(f"\nProblem characteristics:")
        print(f"Total activities:      {res['total_activities']}")
        print(f"Active activities:     {res['activities_with_duration']}")
        print(f"Max activity duration: {res['max_activity_duration']:.1f}")
        print(f"Total work content:    {res['total_work_content']:.1f}")
        
        print("="*70)
        
    def print_detailed_schedule(self):
        """Print detailed activity schedule"""
        sol = self.scheduler.solution()
        
        print("\n" + "="*70)
        print("                    DETAILED SCHEDULE")
        print("="*70)
        
        # Create schedule list sorted by start time
        schedule_items = []
        for i, name in enumerate(self.names):
            if self.scheduler.durations[i] > 0:  # Only real activities
                                 schedule_items.append({
                     'name': name,
                     'start': sol['start'][i],
                     'finish': sol['end'][i],
                     'duration': float(self.scheduler.durations[i])
                 })
        
        # Sort by start time
        schedule_items.sort(key=lambda x: x['start'])
        
        print("Activity              Start      Finish     Duration")
        print("-" * 70)
        for item in schedule_items:
            print(f"{item['name']:<18} {item['start']:8.2f} → {item['finish']:8.2f}  ({item['duration']:6.1f})")
        
        print("-" * 70)
        print(f"{'PROJECT COMPLETION':<18} {sol['makespan']:8.2f}")
        print("="*70)


# ------------------- helper functions -------------------
def save_fig(fig, fname):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig.savefig(fname, dpi=300)
    plt.close(fig)
    print(f"[saved] {fname}")


def save_convergence(loss, span, viol, lam, gnorm, max_viol, fname, early_stop_tol=0.01):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Loss
    axs[0,0].plot(loss, linewidth=1.5)
    axs[0,0].set_title("Loss", fontsize=12, fontweight='bold')
    axs[0,0].grid(alpha=.3)

    # Makespan
    axs[0,1].plot(span, color="tab:red", linewidth=1.5)
    axs[0,1].set_title("Makespan", fontsize=12, fontweight='bold')
    axs[0,1].grid(alpha=.3)

    # Violations (log scale)
    axs[1,0].semilogy(viol, color="tab:green", alpha=0.8, linewidth=1.5, label="Mean")
    axs[1,0].semilogy(max_viol, color="tab:red", linewidth=1.5, label="Max")
    axs[1,0].axhline(y=early_stop_tol, color="red", linestyle="--", alpha=0.7, label="Early stop threshold")
    axs[1,0].set_title("Violations (log scale)", fontsize=12, fontweight='bold')
    axs[1,0].grid(alpha=.3, which="both")
    axs[1,0].legend()

    # λ and gradient norm
    ax4 = axs[1,1]
    ax4.plot(lam, color="tab:purple", linewidth=1.5, label="λ")
    ax4.plot(gnorm, color="tab:gray", linewidth=1.5, alpha=.7, label="‖grad‖")
    ax4.set_yscale("log")
    ax4.set_title("λ and Gradient Norm (log scale)", fontsize=12, fontweight='bold')
    ax4.grid(alpha=.3, which="both")
    ax4.legend()

    for ax in axs.ravel(): 
        ax.set_xlabel("Epoch", fontsize=10)
        ax.tick_params(labelsize=9)
    
    fig.suptitle("Optimization Convergence Analysis", fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {fname}")


def save_gantt(sol, names, durations, fname):
    """Enhanced Gantt chart"""
    # Filter out zero-duration activities
    real_activities = [(i, names[i]) for i in range(len(names)) if durations[i] > 0]
    
    if not real_activities:
        print("No activities to plot in Gantt chart")
        return
    
    n = len(real_activities)
    fig_h = max(6, 2 + 0.3 * n)
    fig, ax = plt.subplots(figsize=(12, fig_h))
    
    # Create color map
    colors = plt.cm.Set3(np.linspace(0, 1, n))
    
    # Plot activities
    for plot_idx, (orig_idx, name) in enumerate(real_activities):
        duration = durations[orig_idx]
        start = sol['start'][orig_idx]
        
        ax.barh(plot_idx, duration, left=start,
                color=colors[plot_idx], edgecolor="black", 
                alpha=0.8, linewidth=0.5)
        
        # Add activity name in center of bar
        ax.text(start + duration/2, plot_idx, name,
                ha="center", va="center", fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_yticks(range(n))
    ax.set_yticklabels([name for _, name in real_activities])
    ax.invert_yaxis()
    ax.set_xlabel("Time", fontsize=12)
    ax.set_title(f"Project Schedule - Makespan: {sol['makespan']:.2f}", 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add makespan line
    ax.axvline(x=sol['makespan'], color='red', linestyle='--', 
               alpha=0.7, label=f'Makespan: {sol["makespan"]:.2f}')
    ax.legend()
    
    plt.tight_layout()
    save_fig(fig, fname)


# ------------------- main -------------------
if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage:  python test_file_enhanced.py <project.json>")
        print("Error : You must supply a project file.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Enhanced dataless scheduler with professional output.")
    parser.add_argument("file", help="Path to project JSON")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--save-solution", action="store_true", help="Save solution to JSON file")
    parser.add_argument("--detailed-schedule", action="store_true", help="Print detailed schedule")
    args = parser.parse_args()

    durations, precedences, names = load_project(args.file)
    
    print(f"Loading project: {args.file}")
    print(f"Activities: {len(names)}, Precedences: {len(precedences)}")
    
    # Run optimization
    sched = DatalessProjectScheduler(durations, precedences)
    histories = sched.solve(epochs=args.epochs, lr=args.lr, verbose=True)
    
    # Analyze solution
    analyzer = SolutionAnalyzer(sched, names, args.file)
    metrics = analyzer.analyze_solution(histories)
    
    # Print results
    analyzer.print_solution_summary(metrics)
    
    if args.detailed_schedule:
        analyzer.print_detailed_schedule()
    
    # Save solution if requested
    if args.save_solution:
        base = os.path.splitext(os.path.basename(args.file))[0]
        solution_file = f"solutions/{base}_solution.json"
        analyzer.save_solution(metrics, solution_file)
    
    # Generate plots
    base = os.path.splitext(os.path.basename(args.file))[0]
    loss_hist, span_hist, viol_hist, lambda_hist, gnorm_hist, max_viol_hist = histories
    
    save_convergence(loss_hist, span_hist, viol_hist, lambda_hist, gnorm_hist, 
                    max_viol_hist, f"figs/{base}_conv.png", sched.early_stop_tol)
    
    save_gantt(sched.solution(), names, sched.durations.cpu().numpy(), 
               f"figs/{base}_gantt.png")
    
    print(f"\nAll outputs saved with '{base}' prefix")
    print("Use --save-solution to save JSON, --detailed-schedule for full schedule") 