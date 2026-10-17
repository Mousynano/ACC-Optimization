# Evolutionary Optimization for Adaptive Cruise Control

Research software and reproducibility package for a comparative study of evolutionary optimization methods applied to a three-gain Adaptive Cruise Control (ACC) controller.

The public release is intentionally organized around **one Python entry point** and **one analysis notebook**:

```text
main.py          # prepare data, run optimization, regenerate evidence
analysis.ipynb   # explore the archived RQ1-RQ4 results without rerunning experiments
```

The lower-level implementation remains available in `core/`, `benchmarks/`, `optimizers/`, and `scripts/` for readers who want to inspect or extend the study.

## Fastest way to explore the study

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Then open:

```text
analysis.ipynb
```

The notebook reads the archived canonical results from `research_outputs/canonical/`. You do **not** need to rerun the optimization campaign just to inspect the tables and figures reported by the research pipeline.

You can also verify that the bundled configuration and canonical evidence are present:

```bash
python main.py verify
```

## Run the optimization yourself

The study uses public OpenACC and WLTC trajectories. The raw third-party files are not redistributed in this repository; the preparation command downloads them from their original public sources, preprocesses them deterministically, and verifies the frozen scenario hash.

```bash
python main.py prepare
python main.py optimize --workers 4
```

`optimize` runs the frozen benchmark:

```text
6 optimizers × 30 independent runs = 180 optimization jobs
```

The six compared methods are:

- Genetic Algorithm (GA)
- Differential Evolution (DE)
- FVRADE
- RNEGDE
- Neighborhood-SHADE
- DE-NPS-FP

Raw run checkpoints are written to `checkpoint_jobs/` and are intentionally excluded from the archival release because they are reproducible runtime artifacts.

## Regenerate the research evidence

After a local optimization campaign has finished:

```bash
python main.py analyze --workers 4
```

To also run or resume the paired canonical-DE objective-ablation campaign used for RQ4:

```bash
python main.py analyze --with-ablation --workers 4
```

A complete prepare → optimize → analyze workflow is also available:

```bash
python main.py all --with-ablation --workers 4
```

This is computationally expensive. Opening `analysis.ipynb` is the sensible option if the goal is simply to inspect the published evidence rather than donate several CPU-hours to science for sport.

## Frozen optimization formulation

The decision vector is:

```text
[k_ve, k_de, k_vrel]
```

with bounds `[-100, 100]^3`.

The frozen FJE performance vector contains:

1. cruise-speed RMSE;
2. following CTG-spacing RMSE;
3. realized longitudinal jerk RMS;
4. positive tractive-energy intensity (Wh/km).

The scalar weights used by the published campaign are stored in `cost_profile_acc_cost_fje_v2_2.json`. The public entry point treats this profile as a **frozen research artifact** rather than silently recalibrating it.

The energy term represents positive wheel-level tractive energy. It is not a fuel-consumption or battery-consumption model, and negative tractive power does not receive energy credit.

## Research scope

The archived evidence is organized around four research questions:

- **RQ1:** solution quality, variability, convergence, runtime, and recorded function evaluations across the six optimizers;
- **RQ2:** fixed-gain controller behavior on synthetic and held-out driving profiles;
- **RQ3:** fixed-gain sensitivity across 12 predeclared physical conditions with matched compensation;
- **RQ4:** paired canonical-DE objective-term ablation for the full, no-jerk, no-energy, and tracking-only formulations.

The active study does not claim sensor-noise robustness, explicit filtering robustness, compensation-parameter mismatch robustness, or equal-function-evaluation comparison.

## Repository layout

```text
.
├── main.py                         # public command-line entry point
├── analysis.ipynb                  # public result walkthrough
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── experiment_manifest_acc_cost_fje_v2_2.json
├── cost_profile_acc_cost_fje_v2_2.json
├── research_protocol.json
├── benchmarks/                     # frozen FJE objective
├── core/                           # ACC simulation, physics, statistics, analysis helpers
├── optimizers/                     # six compared evolutionary optimizers
├── scripts/                        # internal workflow modules used by main.py
├── research_outputs/canonical/     # archived RQ1-RQ4 evidence
├── research_ablation/              # paired RQ4 run records
├── data_external/                  # downloaded/preprocessed public trajectories (generated)
└── tests/                          # lightweight release checks
```

## External data

The preparation workflow uses public trajectory data from:

- JRC OpenACC AstaZero;
- JRC OpenACC Vicolungo;
- JRC OpenACC Casale;
- JRCSTU WLTC Class 3b.

Source URLs, preprocessing rules, selected windows, and SHA-256 hashes are recorded in the generated `scenario_manifest_acc_cost_v2.json`.

## Citation

Citation metadata are provided in `CITATION.cff`. If you use this software or its archived research outputs, cite the corresponding Zenodo software release and, where applicable, the associated research article.

## License

See `LICENSE` for the software license. External datasets retain their original terms and are downloaded from their original sources rather than redistributed here.
