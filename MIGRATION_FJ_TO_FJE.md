# Migration note: V2.1 FJ -> V2.2-FJE

The active cost framework changed from:

```text
FJ = [cruise speed RMSE, following spacing RMSE, realized jerk RMS]
```

to:

```text
FJE = FJ + [positive tractive-energy intensity, Wh/km]
```

Because this changes Pareto dominance, Sahib contribution means/weights, and the
final scalar fitness landscape, the following V2.1 artifacts are **not reusable**
for V2.2-FJE research results:

- `cost_profile_acc_cost_v2.json`;
- V2.1 Pareto/calibration outputs;
- `acc_cost_sahib_v2` main checkpoints.

The public scenario bank itself is unchanged and may be reused when its frozen
hashes remain valid.

Recommended transition:

```bash
python run_pipeline.py --force-calibration --calibrate-only
python plot_validation_motion.py checkpoint_jobs/<new_FJE_checkpoint>.pkl
python run_pipeline.py --main-only
```

Or start from the unfrozen checkout with `python run_pipeline.py`, which will
prepare data if needed, calibrate FJE, preflight, and then start the new campaign.

The framework intentionally does not add output clamping or denoising. Energy is
computed from realized plant traction force and ego speed, with negative wheel
power contributing zero to positive tractive energy.
