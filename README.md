# ACC Evolutionary Benchmark — canonical RQ1–RQ4 scope

The active study uses the frozen V2.2-FJE training campaign and all 30 controllers per optimizer.
The primary evidence scope is now frozen to:

- **RQ1:** six-optimizer solution quality, variability, convergence, runtime, and actual FE;
- **RQ2:** all 180 controllers on nominal/synthetic/held-out downstream profiles;
- **RQ3:** all 180 fixed-gain controllers on the predeclared **12 physical conditions** with matched compensation;
- **RQ4:** paired canonical-DE re-optimization for `full`, `without_jerk`, `without_energy`, and `tracking_only`.

The following are **not part of the primary study**: sensor noise/bias campaigns, separate sensor sampling, explicit sensor filtering, and actual-versus-nominal compensation mismatch. Equal-FE remains deferred. Custom actuator scenarios remain reporting-only and opt-in through `--actuator-config`; they are not part of the canonical RQ3 evidence.

```bash
pip install -r requirements.txt
# RQ1-RQ3 from existing main checkpoints; no main optimization rerun:
python research_pipeline.py --workers 4
# RQ1-RQ4. Reuses a compatible completed RQ4 campaign when present:
python research_pipeline.py --all --workers 4
# Same orchestration, regression tests first:
python run_pipeline.py --research-all
```

`research_outputs/11d1197769c4/` is retained as a **pre-scope-freeze nominal evidence snapshot**. Its RQ1-RQ4 nominal CSV/figures remain scientifically relevant, but its original evaluation contract predates this scope freeze. Run the current pipeline once if a fresh canonical contract/hash is required for final archival submission.

## Frozen training framework V2.2-FJE

Reproducible pipeline for tuning the three-gain constant-time-gap ACC controller
and benchmarking six evolutionary optimizers against one **frozen FJE cost**.

Framework V2.2-FJE promotes the energy-aware formulation tested in the prior
ablation branch. The active performance vector is:

```text
FJE = [
  cruise speed RMSE,
  following CTG-spacing RMSE,
  realized longitudinal jerk RMS,
  positive tractive-energy intensity (Wh/km)
]
```

The energy term is **wheel-level positive tractive energy**, not fuel consumption
and not battery consumption. Negative tractive power receives no energy credit.
The framework adds no post-hoc denoising and no acceleration/jerk output clamp.

The older three-term FJ framework is retained through
`experiment_manifest_acc_cost_v2.json`; V2.2-FJE uses a new benchmark namespace
and a new cost-profile file so stale FJ weights/checkpoints cannot be resumed by
accident.

## Quick start

Ubuntu/Linux:

```bash
./bootstrap.sh
```

Manual environment setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
```

PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_pipeline.py
```

## What the main pipeline does

`python run_pipeline.py` performs, in order:

1. unit/regression tests;
2. public trajectory acquisition and deterministic scenario freeze;
3. calibration preflight;
4. **4-objective MOPSO** over FJE;
5. Sahib–Ahmed inverse-mean contribution-weight construction;
6. 10-trial PSO verification;
7. freeze of `cost_profile_acc_cost_fje_v2_2.json` against the exact scenario hash;
8. main preflight;
9. the six-optimizer, 30-run benchmark.

The main campaign remains 180 jobs:

```text
6 algorithms × 30 independent runs × 1 frozen FJE scalar objective
```

Algorithms: GA, DE, FVRADE, RNEGDE, Neighborhood-SHADE, and DE-NPS-FP.

## Fast smoke test

Before spending serious compute:

```bash
python run_pipeline.py --smoke-test
```

This uses only development scenarios, tiny MOPSO/PSO budgets, and temporary
equal FJE weights. It never freezes a research cost profile.

## Stage commands

```bash
# Download/preprocess/freeze exact public scenarios only
python run_pipeline.py --prepare-only

# Prepare + run fresh 4D FJE calibration + freeze cost profile
python run_pipeline.py --calibrate-only

# Require frozen scenarios/profile, then launch/resume the 180-job benchmark
python run_pipeline.py --main-only

# Force source refresh
python run_pipeline.py --force-data --prepare-only

# Force FJE recalibration
python run_pipeline.py --force-calibration --calibrate-only
```

Low-level equivalent:

```bash
python prepare_scenarios_v2.py
python preflight_v2.py --stage calibration
python calibrate_cost_fje_v2.py --freeze
python preflight_v2.py --stage main
python optimize_parallel.py
```

## Active experiment contract

The default active manifest is:

```text
experiment_manifest_acc_cost_fje_v2_2.json
```

Core controller/plant contract:

- gains: `[k_ve, k_de, k_vrel]`;
- bounds: `[-100, 100]^3`;
- CTG: `d_safe = 20 + 1.2 * v_ego`;
- simulation/controller rate: `60 Hz`;
- vehicle mass: `1500 kg`;
- air density: `1.225 kg/m³`;
- drag coefficient: `0.36`;
- frontal area: `2.42 m²`;
- rolling resistance: `0.015`;
- road slope: `0°`;
- plant force capability: `±6000 N`;
- nominal range-sensor noise/bias: zero.

No additional actuator smoothing, acceleration clamp, or jerk clamp is applied.

## Positive tractive-energy term

The plant already computes the realized traction force after force saturation.
For each plant interval the FJE evaluator computes:

```text
P_tractive = F_traction * max(v_ego, 0)
P_positive = max(P_tractive, 0)
E_positive = sum(P_positive * dt)
f_E        = E_positive / travelled_distance  [Wh/km]
```

Therefore braking never subtracts from cumulative positive energy. This is an
energy-demand surrogate at the wheels. Do not interpret it as battery/fuel use
or regenerative-energy recovery.

For the calibration bank, per-scenario Wh/km values are aggregated with the
same declared scenario weights used for the tracking/jerk components.

## FJE Pareto calibration

MOPSO constructs a four-dimensional feasible nondominated reference front:

```text
[f_v, f_d, f_j, f_E]
```

where:

- `f_v`: cruise set-speed RMSE;
- `f_d`: following CTG-spacing RMSE;
- `f_j`: realized ego jerk RMS;
- `f_E`: positive tractive-energy intensity in Wh/km.

For each dimension, the merged-front mean `mu_i` is converted to a contribution
weight:

```text
w_i = (1 / mu_i) / sum_j(1 / mu_j)
```

The resulting four weights are frozen only after PSO verification. The old
three-objective FJ profile is never reused because adding energy changes Pareto
dominance and the scalar landscape.

## Public scenario bank

Scenario semantics are intentionally unchanged from V2.1 so the FJ→FJE change
is isolated to the performance formulation.

| Role | Source | Rule |
|---|---|---|
| calibration | synthetic cruise | deterministic set-speed events |
| calibration | OpenACC AstaZero | deterministic 90 s leader-kinematic excitation window |
| calibration | OpenACC Vicolungo | deterministic 90 s leader-kinematic excitation window |
| held-out | OpenACC Casale | 50–120 s relative to first valid part timestamp |
| held-out | JRCSTU WLTC Class 3b | full trace |

The processed inputs and exact hashes are recorded in
`scenario_manifest_acc_cost_v2.json`. The scenario-bank contract remains the V2.1
bank because its data/window semantics did not change when FJE was promoted.

## Validation plots from a checkpoint

`plot_validation_motion.py` reads `best_solution` from any main checkpoint and
plots held-out + synthetic motion. It also reports energy, jerk P95/P99, and
traction/braking reversal diagnostics.

```bash
python plot_validation_motion.py checkpoint_jobs/<checkpoint>.pkl

# Synthetic motion diagnostics only
python plot_validation_motion.py checkpoint_jobs/<checkpoint>.pkl --synthetic-only
```

Outputs include PNGs, a multi-page PDF, CSV metrics, and JSON provenance.


## Comprehensive post-processing / manuscript evidence

The main benchmark now has a reusable post-processing layer inspired by the
legacy analysis while preserving the active FJE research contract:

```bash
python analysis_pipeline.py --clean
```

It generates `analysis_outputs/` with:

- scalar objective distributions, run stability, gain-boundary audit, and all-run
  FJE physical-component decomposition;
- median/IQR convergence;
- Q-Q + Shapiro-Wilk diagnostics -> independent Kruskal-Wallis / Mann-Whitney
  comparisons with Holm and Bonferroni correction;
- comparative trajectory motion for periodic cruise, OpenACC AstaZero/Vicolungo/
  Casale, WLTC, and sinusoidal following;
- TTC and explicitly operational/observed TTE diagnostics;
- rise time, settling time, overshoot, peak time, and steady-state error for all
  ten cruise set-speed transitions;
- the inherited 12-case mass/rolling-resistance/road-slope robustness stress test;
- consistency metadata proving scalar-cost reconstruction from the four weighted
  FJE components.

`analysis.ipynb` is the manuscript-facing viewer/orchestrator for these outputs.
The prior minimal FJE notebook is retained as `analysis_minimal_v2_2_fje.ipynb`.
TTC/TTE, time response, held-out trajectories, and robustness remain validation
evidence and do not modify the frozen optimization objective.

## Fail-fast checks

```bash
python preflight_v2.py --stage development
python preflight_v2.py --stage calibration
python preflight_v2.py --stage main
```

Main preflight verifies that:

- exact scenario inputs are frozen and hash-valid;
- the compiled evaluator returns four finite FJE terms;
- the energy term is positive on the calibration bank;
- the frozen profile contains exactly four positive weights summing to one;
- the profile was generated from the exact current scenario hash;
- referenced Pareto/verification artifacts still exist.

## Checkpoint isolation

V2.2-FJE uses:

```text
benchmark = acc_cost_sahib_fje_v2
objective = proposed_cost_fje
```

so old `acc_cost_sahib_v2` checkpoints cannot masquerade as FJE results. After
promoting FJE, the main six-optimizer campaign must be rerun under the new
namespace.

## Repository map

```text
.
├── run_pipeline.py
├── prepare_scenarios_v2.py
├── preflight_v2.py
├── calibrate_cost_fje_v2.py
├── optimize_parallel.py
├── plot_validation_motion.py
├── experiment_manifest_acc_cost_fje_v2_2.json   # default active contract
├── cost_profile_acc_cost_fje_v2_2.json           # generated/frozen by calibration
├── scenario_manifest_acc_cost_v2.json            # unchanged scenario-bank contract
├── benchmarks/
│   ├── acc_cost_fje_v2.py                         # active FJE evaluator
│   └── acc_cost_v2.py                             # retained historical FJ evaluator
├── core/
│   ├── energy.py                                  # readable energy diagnostics
│   ├── fast_acc_energy.py                         # compiled FJE kernels
│   ├── physics.py
│   ├── sensing.py
│   ├── scenarios_v2.py
│   └── data_sources_v2.py
├── optimizers/
├── tests/
├── analysis.ipynb
└── checkpoint_jobs/
```

## Reproduce the older FJ framework

The older three-component framework is still present. Run old-compatible scripts
under its manifest explicitly:

```bash
ACC_MANIFEST=experiment_manifest_acc_cost_v2.json python <script>.py
```

Do not mix its cost profile/checkpoints with FJE.
