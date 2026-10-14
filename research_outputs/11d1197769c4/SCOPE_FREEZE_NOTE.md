# Scope-freeze note — 9 October 2026

This directory was generated before the final omission decision, so `evaluation_contract.json` preserves the exact historical protocol metadata and must not be rewritten retroactively. The retained RQ1, RQ2, canonical 12-condition RQ3, and RQ4 nominal result files are unchanged. The later RQ3 sensor/model/actuator extension outputs and link were removed from the clean project.

For a submission archive whose evaluation contract itself reflects `RQ-2026-10-09-canonical-v1`, run:

```bash
python research_pipeline.py --all --workers 4
```

This does **not** require rerunning the frozen 180-run main optimizer campaign. A compatible completed RQ4 ablation campaign is reused automatically.
