# Refactored ACC stack

This folder splits the old `sisken_prastiyanto_wltc.py` monolith into maintainable modules.

## Main structure

```text
acc_refactor/
  controllers/
    classical_acc.py   # old ACC controller, extracted cleanly
    fopid.py           # new FOPID ACC controller
    base.py            # shared controller interface
  wltc.py              # WLTC CSV/profile/noise/delay helpers
  scenarios.py         # train/test scenario builders
  scenario_registry.py # active global scenarios for optimizer calls
  metrics.py           # TTC, tracking objective, constraints
  simulator.py         # controller-agnostic simulation loop
  benchmark.py         # optimizer-facing functions
  sisken_prastiyanto_wltc_refactored.py # compatibility adapter
```

## Controller parameter vectors

### Classical ACC

```python
[Kve, Kvrel, Kde]
```

or, if you want the optimizer to tune spacing policy too:

```python
[Kve, Kvrel, Kde, time_gap, standstill_distance]
```

### FOPID ACC

Basic FOPID:

```python
[Kp, Ki, Kd, lambda_, mu]
```

FOPID + spacing policy:

```python
[Kp, Ki, Kd, lambda_, mu, time_gap, standstill_distance]
```

FOPID + spacing policy + relative velocity damping:

```python
[Kp, Ki, Kd, lambda_, mu, time_gap, standstill_distance, Kvrel]
```

The FOPID controller uses a finite-memory Grunwald-Letnikov approximation and is NumPy-only.

## Minimal use

```python
from acc_refactor.scenarios import use_wltc_scenarios
from acc_refactor.benchmark import acc_fopid_fitness_func, acc_constraint_evaluator

use_wltc_scenarios("data/wltc_class3b.csv", dt=1/60, mode="train")

params = [1.0, 0.05, 0.01, 0.9, 0.7, 1.2, 20.0, 0.5]
objective = acc_fopid_fitness_func(params)
constraints = acc_constraint_evaluator(params, controller_type="fopid")

print(objective)
print(constraints["feasible"])
print(constraints["total_violation"])
```

## Why this refactor matters

The old file mixed these concerns in one place:

- controller law;
- WLTC loading;
- scenario construction;
- vehicle integration;
- objective computation;
- constraint evaluation;
- optimizer-facing API.

The new layout separates them, so changing from classical ACC to FOPID only means changing the controller factory/fitness function. The simulator and WLTC scenario generation stay the same.
