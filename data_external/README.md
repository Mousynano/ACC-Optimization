# External trajectory cache

This directory is populated by `python main.py prepare`.

- `raw/` contains public third-party source files downloaded from JRC/OpenACC and JRCSTU.
- `processed/` contains compact project-local leader-speed traces.

Raw data are not bundled in the repository archive. Exact URLs, SHA-256 hashes,
source windows, speed columns, and resampling semantics are written to the frozen
`scenario_manifest_acc_cost_v2.json`.

Do not manually edit processed trajectory files after freeze. The production
loader verifies their hashes and will fail if they change.
