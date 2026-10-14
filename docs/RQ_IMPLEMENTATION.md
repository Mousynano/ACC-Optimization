# RQ implementation — canonical scope freeze, 9 October 2026

The four research questions live in `research_protocol.json`. This file is an
evaluation/ablation contract, not a replacement for the frozen training manifest.

## Run the approved pipeline

```bash
pip install -r requirements.txt
# Reuse the existing 180 optimization checkpoints: RQ1, RQ2, RQ3.
python research_pipeline.py
# Also run/resume the 120 paired DE ablation jobs and generate RQ4 evidence.
python research_pipeline.py --all --workers 4
# Equivalent orchestration with regression tests first:
python run_pipeline.py --research-all
```

`research_outputs/<contract-hash>/` contains all four RQ folders, the exact
evaluation contract, `evidence_status.json`, and `RQ_evidence.md`. Output folders
are versioned by input/source/checkpoint hashes. Evaluation cache records contain
controller identity, gain vector, trajectory hash, physical/actuator condition,
and contract identity. Completed simulations can be resumed without rerunning
optimization. No production weights or main checkpoints are overwritten.

Use `research_analysis.ipynb` as the viewer. `analysis.ipynb` remains a
representative-controller viewer for the older directory layout; its inference
has been updated to independent-run comparisons. The original `data/` and
pre-existing `analysis_outputs/` are historical snapshots, not the new RQ evidence.

## Research questions and evidence

| RQ | Question | Methods/code | Required evidence | Conclusion boundary |
|---|---|---|---|---|
| RQ1 | How do the six optimizers differ in quality, consistency, convergence, and cost under a fixed formulation? | `analysis_pipeline.py`, `core/research_statistics.py` | Score/component distributions, median/IQR convergence, actual runtime/FE, independent-run inference and effect magnitude | The tested implementations, formulation, and iteration/population protocol; no equal-FE claim |
| RQ2 | To what extent do optimization differences appear in vehicle responses, including held-out profiles? | `research_pipeline.py`, `core/research_simulation.py` | All controllers on Casale/WLTC, periodic cruise, sinusoidal following, bounded recovery, inherited regression profile; tails/failures/CI, directional transient and association analyses | The tested profiles and controllers; no real-vehicle or universal safety guarantee |
| RQ3 | How do fixed-gain performance, recovery, safety, and energy change across the predeclared physical conditions? | `research_pipeline.py`, `core/research_simulation.py` | All 180 controllers on the 12-case matched physical grid; controller distributions, paired change from each controller nominal, recovery, TTC, and energy | Tested 12-condition grid, fixed gains, matched compensation, nominal sensing |
| RQ4 | How does including jerk and energy change optimized gains and behavior after retuning? | `run_objective_ablation.py` | Four objective profiles, 30 paired DE blocks, calibration components/gains and downstream distributions/paired differences | Conditional on canonical DE, this model, these profiles and frozen surviving-weight ratios |

For the manuscript: motivate each problem in Introduction; give its comparison,
controls, unit of replication and extraction rules in Methods; explicitly answer
the question with these outputs in Results/Discussion; repeat the bounded answer
in Conclusion. This implementation does not edit the manuscript or invent its
eventual conclusions.

## RQ1 statistical correction

Main checkpoints contain optimizer-specific random seeds. A shared run number
is not an established block of equivalent random inputs. RQ1 therefore uses
independent Kruskal-Wallis omnibus comparisons and planned Mann-Whitney U
comparisons, corrected with Holm/Bonferroni within each metric family. Cliff's
delta, median differences and bootstrap intervals accompany p-values. Shapiro
and Q-Q plots are distribution diagnostics, not proof of a test's validity.

Original Friedman/paired-Wilcoxon files are not regenerated as primary evidence.
Existing manuscript statistical statements must be revisited using the new
outputs before publication. Actual FE/runtime continue to be reported; an
equal-FE optimization campaign remains postponed.

## RQ2/RQ3 evaluation semantics

- Controller ID = benchmark + optimizer + run ID. Nominal and perturbed/profile
  evaluations retain this identity. Thirty controllers on two profiles are not
  sixty independent controllers.
- Simulation states are recorded before each interval update. The controller
  uses ego speed and sensed range/range difference; lead truth is evaluator-side.
- All performance metrics use realized dynamics. Energy integrates the applied
  force times interval-start ego speed, including the actual physical condition.
  Negative wheel power receives no positive-energy credit.
- Pointwise median/IQR convergence uses the 30 optimization histories. Physical
  distributions include finite counts, missing/infinite counts, P05/P95, extrema,
  and median bootstrap intervals. Finite-value summaries are conditional on the
  stated finite count; an infinite TTC with no closing is not a failure.
- Collision or nonfinite simulation is a retained failed evaluation. Partial
  trajectories are not reported as successful full-profile metrics.
- TTC below-threshold fraction now uses the full trace as denominator. A separate
  closing-only fraction is exported so the two definitions are not conflated.
  The 1.5 s line is a study reporting threshold, not a certified safe-gap policy.
- Cruise transitions are first summarized within controller and direction;
  bootstrap operates on controllers, not individual samples or transitions.
  Rise time follows the existing 10–90% rule; settling follows the existing
  observed final-mean band rule, so inspect steady-state error alongside it.
- Training-to-held-out Spearman associations are evaluated within optimizer and
  profile. Small-sample permutation p-values and family Holm corrections are
  reported. These are exploratory associations, not causal term attribution.
- Representative trajectories remain useful illustrations alongside all-run
  distributions. Their raw range, relative-speed estimate, command, realized
  acceleration, jerk, requested/target/applied force, and wheel power are saved
  as compressed telemetry in RQ2's telemetry folder. Nominal sensing is retained throughout the primary study; sensor uncertainty, separate sampling, and filtering are outside scope.

### Recovery

The new study-defined diagnostic starts at CTG equilibrium at 25 m/s, applies
lead deceleration −1 m/s² from 10 to 13 s, then holds 22 m/s until 60 s.
The initial gap is derived from the existing CTG policy (50 m nominal).

Recovery requires relative speed ≤0.2 m/s and true spacing error within the
larger of 0.75 m or 2% of desired spacing for 2 s. TTE measures delay to the
start of the qualifying hold; completion delay is also exported. Event output
includes the available observation window, recovery status and nonrecovery
reason. Nonrecovered events stay in the recovery-fraction denominator, and a
recovered-only TTE median is explicitly conditional. The final admissible hold
window is included. This is an operational diagnostic, not an analytical TTE
or a formal stability/string-stability proof.

Casale retains disturbance detection for its changing lead trajectory. WLTC
and continuous sinusoidal/regression profiles are not assigned equilibrium TTE.

## RQ4 isolation and pairing

The four profiles are `full`, `without_jerk`, `without_energy`, `tracking_only`.
Removed coefficients become zero only in this explicit ablation context.
Ratios of surviving frozen weights remain fixed and are renormalized to sum
to one. The production loader still requires four positive finite weights.
The plant, tracking definitions, gain bounds and feasibility penalty remain
the same. All four physical components are measured even when inactive.

Each block explicitly supplies the same initial population and local DE seed
to all four profiles. Initial-population hashes, seeds, function calls, gains,
components and convergence histories are checkpointed and checked on resume.
The full-DE arm is rerun: the old 30 DE results cannot be presumed paired with
these new initial populations. Therefore the complete design has 120 new jobs.

Compare common physical metrics and their paired changes, not raw scalar costs
from unlike objectives. Each arm also receives the original full-objective score
for a common reference. Paired Wilcoxon, paired median-difference bootstrap and
rank-biserial effects are reported, with finite-pair counts. The 2×2 interaction
contrast is exploratory and conditional on this design.

```bash
# Tiny integration check: separate namespace, research=false.
python run_objective_ablation.py --smoke-test --workers 2
# Research tuning only; resumable, never overwrites production artifacts.
python run_objective_ablation.py --workers 4
# Analyze a specific completed research campaign:
python research_pipeline.py --ablation-dir research_ablation/research_<hash>
```

## Explicitly omitted from the primary study

Sensor noise/bias perturbations, separate sensor sampling, explicit sensor filtering, and actual-versus-nominal compensation mismatch are not evaluated in the canonical RQ1–RQ4 evidence. They introduce distinct sensing/estimation/model-uncertainty questions beyond the physical-condition robustness scope of RQ3. Custom actuator reporting remains opt-in only and must not be mixed with the primary evidence.

## Still postponed

No equal-FE optimization campaign is introduced. The frozen nominal training
manifest remains noise/bias free. Sensor/model-uncertainty campaigns are outside
the active study rather than hidden inside another evaluation branch. Adopting
another primary sensing or tuning plant would require a new declared protocol.
The frozen 180-controller main campaign is not retuned by this scope freeze.

## Verification

```bash
pip install pytest
python -m pytest -q
```

New regression tests cover nominal Python/Numba equivalence, saturation/grade,
actual-force energy, optional actuator response and power bounds, collision
retention, recovery-window endpoints, explicit DE-population reproducibility,
ablation pairing/resume rejection, production-weight guards, controller-level
counts, independent inference, permutation correlation and evaluation caching.
