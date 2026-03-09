# ACC Optimization Playground

A simple research playground for **Adaptive Cruise Control (ACC)** and **metaheuristic optimization**.

This repository can be used in three different ways:

- **ACC demo** — see how the controller manages speed, distance, and vehicle response.
- **Metaheuristic playground** — test optimization algorithms on mathematical benchmark functions.
- **Research pipeline** — run larger ACC optimization experiments and review the results.

The goal of this project is to make it easy for readers to explore the part they care about most, without needing to understand the whole codebase at once.

---

## What is inside this project?

This project combines two main ideas.

### 1. Adaptive Cruise Control (ACC)
The ACC part simulates how an ego vehicle follows a lead vehicle while trying to stay safe, stable, and smooth.

The controller is optimized through three gains:

- `verr_gain`
- `vx_gain`
- `xerr_gain`

Main related files:

- `benchmarks/sisken_prastiyanto.py`
- `core/physics.py`
- `run_acc.py`

### 2. Metaheuristic Optimization
The optimization part searches for good parameters automatically.

Included algorithms:

- **PSO** — Particle Swarm Optimization
- **GA** — Genetic Algorithm
- **KMA** — Komodo Mlipir Algorithm
- **SKA** — Stochastic Komodo Algorithm
- **HOA** — Hiking Optimization Algorithm

Main related files:

- `optimizers/pso.py`
- `optimizers/ga.py`
- `optimizers/kma.py`
- `optimizers/ska.py`
- `optimizers/hoa.py`

---

## Recommended branches

### `main`
Use this branch if you want a cleaner version of the project focused on the code.

Best for:

- reading the source code
- learning the project structure
- modifying or extending the program

### `prerun`
Use this branch if you want code **plus prepared outputs**, such as saved data, plots, tables, or checkpoints.

Best for:

- exploring previous experiment results
- reproducing the research more quickly
- reviewing outputs without running everything from the beginning

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/Mousynano/ACC-Optimization.git
cd ACC-Optimization
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

> Note: the current file is named `requirement.txt`. In the future, it would be better to rename it to `requirements.txt`.

---

## Choose your path

## A. I only want to try the ACC simulation

Run:

```bash
python run_acc.py
```

This is the easiest entry point for new readers.

It will help you:

- run one ACC simulation
- observe the controller behavior
- view plots related to motion and system response

Useful files:

- `run_acc.py`
- `benchmarks/sisken_prastiyanto.py`
- `core/physics.py`
- `core/utils.py`

This path is good for:

- quick demonstrations
- visual understanding
- readers who care more about ACC than optimization

---

## B. I only want to try the metaheuristics

You can use this repository as a general optimization playground.

The simplest path is to focus on the mathematical benchmark functions instead of the ACC simulation.

Useful files:

- `benchmarks/simple_funcs.py`
- `optimizers/*.py`
- `optimize.py`
- `optimize_parallel.py`

Suggested idea:

- activate a simple benchmark function
- compare two or more algorithms
- observe convergence behavior

This path is good for:

- learning how metaheuristics behave
- comparing optimization methods
- experimenting without the ACC setup

---

## C. I want to run the full ACC optimization experiment

Run:

```bash
python optimize_parallel.py
```

This script is designed for larger experiment workflows.

It can help you:

- create optimization jobs
- test multiple algorithms and objectives
- save checkpoints
- generate reports and plots

Common output folders include:

- `checkpoint_jobs/`
- `output/`
- and in `prerun`, additional processed results inside `data/`

This path is good for:

- research experiments
- repeated testing
- comparison across algorithms and objectives

---

## Available objective functions

This project supports several common error-based objective functions:

- **IAE** — Integral of Absolute Error
- **ISE** — Integral of Squared Error
- **ITAE** — Integral of Time-weighted Absolute Error
- **ITSE** — Integral of Time-weighted Squared Error

These utilities are mainly handled in:

- `core/utils.py`

In simple terms:

- **IAE** focuses on total absolute error
- **ISE** punishes large errors more strongly
- **ITAE** gives more penalty to errors that last longer
- **ITSE** combines time sensitivity and squared error

---

## Project structure

```text
ACC-Optimization/
├── benchmarks/
│   ├── simple_funcs.py
│   └── sisken_prastiyanto.py
├── core/
│   ├── physics.py
│   ├── utils.py
│   ├── report.py
│   ├── job_worker.py
│   ├── checkpoint_jobs.py
│   └── xp.py
├── optimizers/
│   ├── pso.py
│   ├── ga.py
│   ├── kma.py
│   ├── ska.py
│   └── hoa.py
├── optimize.py
├── optimize_parallel.py
├── run_acc.py
└── run_export.py
```

---

## How you can explore and create

This repository is flexible. You do not need to use every part of it.

### Try a different optimizer
You can activate one algorithm only, compare a few algorithms, or add your own.

This is useful if you want to study algorithm behavior rather than vehicle control.

### Try a different objective
You can compare how the same optimizer behaves under different objective functions such as `IAE` or `ITAE`.

This is useful if you want to understand which objective gives a better control response.

### Switch between simple functions and ACC
You can use this project in two modes:

- **simple mathematical benchmarks**
- **ACC-based simulation benchmarks**

This makes the repository useful both for control-related research and for general optimization experiments.

### Test different ACC conditions
Inside the ACC benchmark, you can explore different conditions such as:

- road angle
- vehicle mass
- rolling resistance

This is one of the most interesting parts if you want to study robustness.

---

## What can you expect from the `prerun` branch?

The `prerun` branch is useful for readers who want to inspect results quickly.

Depending on what is included, you may find:

- saved checkpoints
- CSV summary tables
- generated plots
- motion profiles
- robustness-related outputs
- experiment summaries

This branch is helpful for:

- thesis presentation
- result inspection
- quick review by supervisors or examiners
- readers who do not want to run the full pipeline first

---

## Who is this repository for?

This project can be useful for:

- students learning ACC concepts
- readers who want to try metaheuristic algorithms
- researchers comparing optimization methods
- thesis reviewers who want to inspect code and outputs
- developers who want to extend the experiments

---

## Final note

This repository is not only an ACC project. It can also be used as:

- an **ACC simulation sandbox**
- a **metaheuristic testing playground**
- a **research experiment pipeline**
- a **reproducible result archive** through the `prerun` branch

If you only want to understand the controller, start with `run_acc.py`.

If you only want to study optimization, start with the benchmark functions and the files inside `optimizers/`.

If you want the full research workflow, use `optimize_parallel.py` and explore the `prerun` branch.
