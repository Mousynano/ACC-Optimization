# Canonical scope freeze — 9 October 2026

## Active primary study

- RQ1: six optimizers, 30 runs each, frozen V2.2-FJE formulation.
- RQ2: all 180 optimized controllers on nominal/synthetic/held-out downstream profiles.
- RQ3: all 180 fixed-gain controllers on the predeclared 12 physical conditions with matched road-load compensation.
- RQ4: canonical DE, four objective formulations, 30 paired blocks per formulation, fresh re-optimization within the ablation campaign.

## Explicit omissions

The following are not part of the primary RQ1-RQ4 study:

- sensor noise/bias perturbation campaign;
- separate sensor sampling clock;
- explicit sensor filtering;
- actual-versus-nominal compensation mismatch.

Equal-FE remains deferred. Custom actuator conditions remain optional reporting only and are not part of canonical RQ3.

## Existing evidence and rerun policy

- Main campaign: 180/180 frozen checkpoints present. **Do not rerun.**
- RQ2: 1080 nominal/downstream evaluations previously completed, 0 failed. No new data collection is scientifically required.
- RQ3: 4320 canonical matched-physical evaluations previously completed over 12 conditions, 0 failed. No new data collection is scientifically required.
- RQ4: `research_ablation/research_5f9d42f13d77/` contains 120/120 paired canonical-DE jobs. **Do not rerun** unless the optimizer, objective, frozen weights, scenario manifest, or other critical simulation source changes.
- `research_outputs/11d1197769c4/` is retained as a nominal pre-scope-freeze evidence snapshot. Its canonical RQ1-RQ4 numerical files remain useful, but the historical `evaluation_contract.json` records the protocol metadata that existed when those files were generated.

For a fresh evidence directory whose contract/hash itself reflects the current canonical protocol, run:

```bash
python research_pipeline.py --all --workers 4
```

The cleaned pipeline searches for a scientifically compatible completed RQ4 campaign first. Therefore this command regenerates evaluation/analysis evidence and a fresh contract, but does **not** rerun the 120 RQ4 optimization jobs when the retained campaign is compatible. It also never reruns the 180 main optimization jobs.

## Packaging repair

The user-supplied `(9).zip` was a stripped package: it contained the checkpoints and top-level scripts but omitted required source directories/files such as most of `core/`, `optimizers/`, `data_external/`, tests, and the completed research evidence. The complete `fin.zip` project snapshot was used only to restore these missing files. All 311 non-venv files common to both packages had identical SHA-256 content, confirming that the stripped upload and complete snapshot share the same code revision for their overlapping files.
