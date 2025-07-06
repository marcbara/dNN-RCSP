# Dataless Neural Networks for Resource Constrained Scheduling Problems (dNN-RCSP)

A novel approach to solving Resource Constrained Scheduling Problems using dataless neural networks with adaptive penalty methods and intelligent feasible candidate selection.

## Overview

This project implements a **dataless neural network** that learns to schedule project activities while respecting precedence constraints. Unlike traditional approaches, the neural network doesn't train on data but rather optimizes directly on the scheduling problem using gradient descent with sophisticated penalty methods.

## Key Features

### 🧠 **Dataless Neural Network Architecture**
- **Trainable Parameters**: Raw start times transformed via softplus activation
- **Zero Initialization**: All activities want to start immediately, network learns to space them
- **Continuous Scheduling**: Produces fractional start times (realistic for project management)

### 🎯 **Adaptive Penalty Method**
- **Escalation Phase**: λ doubles when violations > threshold (up to 100M cap)
- **Reduction Phase**: λ reduces when violations < 0.0001 (maintains constraint pressure)
- **Recovery Mechanism**: λ spikes when violations increase (prevents constraint degradation)
- **Safety Valve**: λ never drops below penalty_init × 10

### 🔍 **Intelligent Feasible Candidate System**
- **Continuous Collection**: Stores all solutions with violations < 1e-6 during optimization
- **Best Selection**: Automatically selects minimum makespan among feasible solutions
- **State Restoration**: Restores neural network to reproduce best solution

### 📊 **Professional Output**
- **Clean Terminal**: Essential progress without spam
- **Comprehensive Metrics**: Solution quality, feasible candidates count, optimization statistics
- **Multiple Formats**: JSON solutions, convergence plots, Gantt charts
- **Sensible Precision**: 3 decimal places for scheduling (not 15+ meaningless digits)

## Project Structure

```
dNN-RCSP/
├── find_solution.py          # Main solver script
├── scheduler.py              # Core DatalessProjectScheduler class
├── project_loader.py         # JSON project file loader
├── psplib_converter.py       # Convert PSPLIB .sm files to JSON
├── critical_path_analyzer.py # Theoretical optimal makespan calculation
├── solutions/                # Generated solution files (.json)
├── figs/                     # Generated plots (.png)
├── j30_json/                 # PSPLIB benchmark instances (480 projects)
├── mini7.json               # Simple test problem (7 activities)
├── realistic20.json         # Medium test problem (20 activities)
├── realistic40.json         # Large test problem (40 activities)
└── README.md               # This documentation
```

## Algorithm Details

### Neural Network Formulation

**Parameters**: `raw_times` ∈ ℝⁿ (trainable start times)

**Transformation**: `start_times = softplus(raw_times) - min(softplus(raw_times))`

**Loss Function**: `L = makespan + λ × penalty_term`

Where:
- `makespan = smooth_max(finish_times, β)` (differentiable via log-sum-exp)
- `penalty_term = Σ relu(pred_finish - succ_start)` (precedence violations)

### Adaptive Penalty Algorithm

```python
# Escalation: violations are significant
if violations > 0.001:
    λ *= 2  # Double penalty (up to 100M cap)

# Reduction: violations are tiny and stable  
elif violations < 0.0001 and stable:
    λ *= 0.8  # Reduce penalty (with safety valve)

# Recovery: violations increasing (algorithm testing boundaries)
elif violations_trend_up:
    λ *= 2  # Boost penalty immediately
```

### Feasible Candidate Collection

```python
# During optimization (every epoch)
if max_violation < 1e-6:
    candidate = {
        'epoch': current_epoch,
        'makespan': current_makespan,
        'start_times': current_solution.copy(),
        'max_violation': max_violation
    }
    feasible_candidates.append(candidate)

# At end: select best among all feasible
best = min(feasible_candidates, key=lambda x: x['makespan'])
```

## Usage

### Basic Usage

```bash
# Simple test problem
python find_solution.py mini7.json --epochs 300 --lr 0.1

# PSPLIB benchmark instance  
python find_solution.py j30_json/j301_1.json --epochs 600 --lr 0.05

# Save solution and show detailed schedule
python find_solution.py realistic20.json --epochs 400 --save-solution --detailed-schedule
```

### Command Line Options

```
python find_solution.py <project.json> [options]

Required:
  project.json              Path to project file

Optional:
  --epochs INT              Number of optimization epochs (default: 400)
  --lr FLOAT               Learning rate (default: 0.1)
  --save-solution          Save solution to JSON file in solutions/
  --detailed-schedule      Print detailed activity schedule
```

### Project File Format

```json
{
  "activities": {
    "START": {"duration": 0},
    "A": {"duration": 4},
    "B": {"duration": 3},
    "END": {"duration": 0}
  },
  "precedences": [
    ["START", "A"],
    ["START", "B"], 
    ["A", "END"],
    ["B", "END"]
  ]
}
```

## Output Examples

### Terminal Output
```
Loading project: realistic20.json
Activities: 21, Precedences: 25

ep    0 | span  10.00 | viol 3.45e+00 | λ 2.0e+01 | ∥grad∥ 2.14
ep  100 | span  35.20 | viol 2.10e+00 | λ 1.3e+08 | ∥grad∥ 10.00
ep  200 | span  42.15 | viol 5.23e-02 | λ 1.3e+08 | ∥grad∥ 10.00
ep  300 | span  40.85 | viol 2.14e-06 | λ 4.2e+07 | ∥grad∥ 8.50

📊 Found 45 feasible candidates
   Best makespan: 40.000 at epoch 347
   Makespan range: 40.000 - 40.950
✓ BEST SOLUTION: epoch 347, makespan 40.000, max_viol 0.00e+00

======================================================================
                      SOLUTION SUMMARY
======================================================================
Final makespan:        40.000
Max violation:         0.00e+00
Mean violation:        0.00e+00
Feasible:              ✓ YES
Solution epoch:        347
Total epochs run:      400
Feasible candidates:   45

Problem characteristics:
Total activities:      21
Active activities:     19
Max activity duration: 8.0
Total work content:    95.0
======================================================================

[saved] figs/realistic20_conv.png
[saved] figs/realistic20_gantt.png

All outputs saved with 'realistic20' prefix
```

### Generated Files

**Solution File** (`solutions/realistic20_solution.json`):
```json
{
  "metadata": {
    "project_file": "realistic20.json",
    "timestamp": "2025-07-06T18:15:42.123456",
    "solver": "DatalessProjectScheduler",
    "version": "1.0"
  },
  "solution": {
    "start_times": {
      "START": 0.0,
      "Planning": 0.0,
      "Design": 5.0,
      "Development": 13.0,
      "Testing": 25.0,
      "Deployment": 35.0,
      "END": 40.0
    },
    "makespan": 40.0
  },
  "metrics": {
    "final_solution": {
      "makespan": 40.0,
      "max_violation": 0.0,
      "feasible": true,
      "solution_epoch": 347,
      "feasible_candidates_found": 45
    }
  }
}
```

**Generated Plots**:
- `figs/{project}_conv.png` - Convergence analysis (loss, makespan, violations, λ)
- `figs/{project}_gantt.png` - Gantt chart visualization

## PSPLIB Integration

### Converting PSPLIB Files

```bash
# Convert single file
python psplib_converter.py j30.sm/j301_1.sm

# Convert all files in directory  
python psplib_converter.py j30.sm/ --output-dir j30_json/
```

### Benchmark Results

**J30 Benchmark Set** (30 activities + START/END):
- **480 instances** available in `j30_json/`
- **Theoretical optimals** computed via critical path analysis
- **Typical results**: 38-45 makespan for j301_x instances

**Example Results**:
- `j301_1`: 38.6 makespan (23 feasible candidates found)
- `mini7`: 8.13 makespan vs 8.0 optimal (1.6% gap)
- `realistic20`: 40.0 makespan (45 feasible candidates, optimal achieved)

## Algorithm Insights

### Why Dataless Works

1. **Zero Initialization Philosophy**: "Everyone wants to start now" - natural starting point
2. **Gradient-Based Learning**: Network learns precedence relationships through violation penalties  
3. **Continuous Solutions**: Fractional start times are realistic (0.048 time units = 1.15 hours)
4. **Adaptive Exploration**: Recovery mechanism creates intelligent boundary testing

### Key Innovations

- **Feasible Candidate System**: Collects multiple solutions, selects best (not first found)
- **Recovery Mechanism**: Acts as "constraint conscience" preventing infeasible solutions
- **Adaptive Penalty**: Self-regulating λ maintains constraint pressure while optimizing
- **Professional Output**: Clean metrics without misleading "convergence analysis"

## Dependencies

```bash
pip install torch numpy matplotlib 
```

## Research Context

This approach demonstrates that:
- Neural networks can solve scheduling without training data
- Gradient descent with adaptive penalties finds high-quality solutions  
- Continuous scheduling is often superior to discrete post-processing
- Recovery mechanisms prevent constraint degradation during optimization

## Future Work

- Multi-resource constraints (resource-constrained project scheduling)
- Larger problem instances (100+ activities)
- Parallelization for multiple project optimization
- Integration with real project management systems

---

**Citation**: If you use this code in research, please cite:
```
Dataless Neural Networks for Resource Constrained Scheduling Problems
[Your Name], 2025
``` 