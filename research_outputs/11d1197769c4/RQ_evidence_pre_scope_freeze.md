# Research-question evidence

Generated from the declared frozen simulation protocol; no manuscript files were edited.

## RQ1: optimizer quality and cost

| Optimizer | Median score | Mean runtime (s) | Mean FE |
|---|---:|---:|---:|
| ga | 1.44307884 | 1.4864 | 3000 |
| de | 1.40832417 | 2.0124 | 3100 |
| fvrade | 1.40195657 | 2.2985 | 3100 |
| rnegde | 1.40202516 | 2.4394 | 3100 |
| nshade | 1.40297308 | 2.5822 | 3100 |
| denps | 1.40251735 | 2.3856 | 3187.8 |

Independent-run inference is in RQ1_optimizer/. Runtime/FE are recorded effort, not an equal-FE comparison.

## RQ2: closed-loop behavior

Evaluated 1080 controller-profile combinations, including 360 held-out evaluations. Failed evaluations: 0.

controller_distributions.csv reports finite counts, tails, and controller-bootstrap intervals. Recovery fractions include nonrecovered events; recovered-only TTE is conditional. Correlations are exploratory.

## RQ3: physical sensitivity

Evaluated 4320 controller-profile-condition combinations across 12 configurations. Failed evaluations: 0.

Relative changes compare each fixed-gain controller with its own nominal result. Compensation uses the same physical parameters as the plant, so these are matched-model sensitivities.

## RQ4: objective contribution

complete: 4 objective profiles x 30 paired DE blocks

Use common physical metrics and paired differences, not comparisons of unlike ablated scalar fitness values. Results are conditional on canonical DE.

## Intro--Methods--Results--Conclusion traceability

| RQ | Intro problem | Methods | Results evidence | Conclusion boundary |
|---|---|---|---|---|
| RQ1 | Stochastic quality-cost trade-off | 30 runs/optimizer, independent inference | RQ1_optimizer | Tested optimizers and declared budget |
| RQ2 | Scores may not translate to motion | Frozen gains, held-out profiles, diagnostics | RQ2_behavior | Tested controllers and profiles |
| RQ3 | Nominal behavior does not establish condition sensitivity | Fixed gains, 12 matched-model configurations | RQ3_physical | Ideal compensation and tested grid |
| RQ4 | Full-objective components do not isolate objective contributions | Paired DE retuning, four objective profiles | RQ4_objective | Conditional on DE and declared model |

## RQ3: approved sensor/model/actuator expansion

62100 additional evaluations across 50 conditions; 0 failed evaluations. Seed repetitions are within-controller measurements. The matched physical grid above is one RQ3 stratum; extension evidence additionally covers range uncertainty, causal filtering/sampling, frozen-nominal compensation and actuator limits/lag. See RQ3_extension_link.json, extension distributions, paired contrasts with completion exclusions, and failure counts. Conclusion scope: the tested frozen controllers, profiles, parameter grid and sensor seeds; accurate ego speed and the single-leader range-only architecture remain assumptions.
