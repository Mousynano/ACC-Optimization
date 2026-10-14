# Implementasi canonical RQ1–RQ4 — scope freeze 9 Oktober 2026

Basis: frozen V2.2-FJE main campaign dengan 180 checkpoint utama. Gain, bobot, scenario calibration, dan plant tuning tidak diubah.

| RQ | Implementasi aktif |
|---|---|
| RQ1 | 6 optimizer × 30 runs; score/component distributions, convergence median–IQR, runtime/FE aktual, inferensi independent-run |
| RQ2 | Seluruh 180 controller; 1080 evaluasi nominal/downstream termasuk held-out Casale/WLTC, transient, recovery, dan association audit |
| RQ3 | Seluruh 180 controller; 12 physical conditions × 2 profiles = 4320 evaluasi; fixed gains; matched actual/compensation parameters |
| RQ4 | 4 objective formulations × 30 paired canonical-DE blocks; 120 optimization jobs; 720 downstream evaluations |

## Omission / deferral

Tidak masuk primary study:

- sensor noise/bias campaign;
- separate sensor sampling;
- explicit sensor filtering;
- actual-versus-nominal compensation mismatch.

Equal-FE tetap ditunda. Actuator extensions hanya tersedia sebagai custom reporting opt-in (`--actuator-config`), bukan bagian dari canonical RQ3.

## Rerun policy

- **Main 180-run optimization: tidak perlu rerun.** Checkpoint lengkap dan frozen.
- **RQ2/RQ3 data collection: secara ilmiah tidak perlu rerun** karena 1080 + 4320 canonical evaluations sudah selesai tanpa failure.
- **RQ4 optimization: tidak perlu rerun.** `research_ablation/research_5f9d42f13d77/` lengkap 120/120 jobs.
- Untuk membuat **fresh canonical evidence contract/hash setelah scope cleanup**, jalankan `python research_pipeline.py --all --workers 4`. Pipeline akan melakukan ulang evaluasi/analysis RQ1–RQ3 dan downstream RQ4, tetapi akan memakai ulang compatible completed RQ4 campaign sehingga tidak menjalankan 120 optimization jobs lagi.

Keluaran `research_outputs/11d1197769c4/` dipertahankan sebagai snapshot nominal sebelum scope freeze; cache dan telemetry besar yang reproducible dibuang dari paket clean.
