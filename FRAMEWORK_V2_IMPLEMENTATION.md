# Framework V2.2-FJE implementation contract

## Status

FJE is now the **active checked-out framework** for implementation/testing:

```text
FJE = [speed RMSE, spacing RMSE, realized jerk RMS, positive tractive Wh/km]
```

This checkout is an experimental-method choice, not a claim that the earlier
ablation proved FJE superior. A fresh 4D calibration and main campaign are
required before research conclusions are updated.

## What changed from V2.1 FJ

1. Added `core/energy.py` for readable wheel-level energy diagnostics.
2. Added `core/fast_acc_energy.py` for the compiled energy-aware kernels.
3. Added `benchmarks/acc_cost_fje_v2.py` as the active four-component evaluator.
4. Added `calibrate_cost_fje_v2.py` for 4-objective MOPSO, Sahib weights, and PSO verification.
5. Added `experiment_manifest_acc_cost_fje_v2_2.json` and a separate unfrozen FJE profile.
6. Changed the default manifest to V2.2-FJE.
7. Isolated main checkpoints under `acc_cost_sahib_fje_v2 / proposed_cost_fje`.
8. Updated validation plotting and analysis to expose the energy component.

## What did NOT change

- controller equations;
- gain vector and bounds;
- scenario selection and public trajectory preprocessing;
- 60 Hz sampling;
- sensor architecture;
- longitudinal plant and ±6000 N force saturation;
- no post-hoc denoising;
- no acceleration clamp;
- no jerk clamp;
- six final optimizers and 30×30×100 campaign protocol.

## Energy semantics

For every simulation interval:

```text
P = F_traction * v_ego
P+ = max(P, 0)
E+ = Σ P+ Δt
f_E = E+ / distance  [Wh/km]
```

`F_traction` is the same realized wheel-force output already produced by the
plant after force saturation. Negative wheel power gets no energy credit.

Required terminology: **positive tractive energy** or **wheel-level positive
tractive-energy demand**. The V2.2 model does not justify fuel, battery, or regen
claims.

## Recalibration rule

V2.1 weights are invalid for V2.2-FJE. Run:

```bash
python run_pipeline.py --force-calibration --calibrate-only
```

or simply `python run_pipeline.py` from an unfrozen FJE checkout.

A frozen FJE profile must contain four positive weights, the four objective
names, the current scenario-manifest SHA-256, and existing reference-front/PSO
verification artifacts.

## Main-run rule

Only after `PREFLIGHT PASS [main]` may the 180-job benchmark start. Existing FJ
checkpoints are historical evidence and cannot be resumed because the benchmark
namespace changed.
