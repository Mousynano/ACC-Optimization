#!/usr/bin/env python3
"""Acquire public trajectories and freeze the Framework V2 scenario manifest.

This is the only script that mutates ``scenario_manifest_acc_cost_v2.json``.
It records source URLs, source/processed hashes, exact selected windows, units,
sampling semantics, and the deterministic selection rule used for calibration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.config import ROOT_DIR, load_manifest
from core.data_sources_v2 import prepare_public_trajectories
from core.scenarios_v2 import load_scenario_suite, sha256_file


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR))


def _prov(t):
    return {
        "source": t.source_name,
        "source_url": t.source_url,
        "raw_path": _rel(t.raw_path),
        "raw_sha256": t.raw_sha256,
        "processed_path": _rel(t.processed_path),
        "processed_sha256": t.processed_sha256,
        "source_sampling_hz_estimate": t.source_sampling_hz,
        "source_window_start_s": t.start_s,
        "source_window_end_s": t.end_s,
        "source_speed_column": t.speed_column,
        "selection_rule": t.selection_rule,
        "excitation_score": t.excitation_score,
    }


def build_manifest(prepared: dict) -> dict:
    exp = load_manifest()
    controller = exp["controller"]
    sampling_hz = int(exp["execution_backend"]["sampling_hz"])
    d0 = float(controller["standstill_distance_m"])
    h = float(controller["time_gap_s"])

    asta = prepared["asta_calibration"]
    vico = prepared["vicolungo_calibration"]
    casale = prepared["casale_validation"]
    wltc = prepared["wltc_class3b"]

    def following_entry(sid, traj, weight=1.0):
        v0 = float(traj.initial_speed_mps)
        return {
            "id": sid,
            "type": "csv_lead_speed",
            "path": _rel(traj.processed_path),
            "time_column": "time_s",
            "lead_speed_column": "lead_speed_mps",
            "lead_speed_unit": "m/s",
            "start_s": 0.0,
            "end_s": round(float(traj.duration_s), 6),
            "initial_lead_speed_mps": v0,
            "initial_ego_speed_mps": v0,
            "initial_gap_m": d0 + h * v0,
            "weight": weight,
            "resampling": {
                "target_sampling_hz": sampling_hz,
                "method": "linear_interpolation_of_leader_speed",
                "note": "OpenACC source is approximately 10 Hz; controller simulation runs on the frozen 60 Hz grid."
            },
            "provenance": _prov(traj),
        }

    # Explicit synthetic cruise experiment. The speed levels are moderate relative
    # to the plant force bound and give repeated positive/negative set-point changes.
    cruise = {
        "id": "C0_CRUISE_STEPS",
        "type": "cruise_events",
        "duration_s": 80.0,
        "initial_ego_speed_mps": 20.0,
        "events": [[0.0, 20.0], [20.0, 25.0], [40.0, 30.0], [60.0, 25.0]],
        "weight": 1.0,
        "source": "deterministic_synthetic_cruise_profile_v2",
        "design_rationale": "Repeated 5 m/s set-speed changes within the nominal longitudinal-force envelope; no external controller response used in construction."
    }

    validation = {
        "V1_CASALE": {
            **following_entry("V1_CASALE", casale),
            "role": "held_out_empirical_following_validation",
            "note": "Casale part6 fixed 50-120 s window relative to the first valid timestamp in the part file; excluded from cost calibration."
        },
        "V2_WLTC": {
            **following_entry("V2_WLTC", wltc),
            "role": "standardized_out_of_distribution_lead_speed_validation",
            "note": "Full WLTC Class 3b speed trace; excluded from cost calibration."
        },
        "V3_LEGACY": {
            "id": "V3_LEGACY",
            "type": "synthetic_sinusoidal_lead",
            "duration_s": 60.0,
            "initial_ego_speed_mps": 22.0,
            "initial_lead_speed_mps": 22.0,
            "initial_gap_m": 60.0,
            "lead_accel_amplitude_mps2": 1.0,
            "lead_phase_rate_deg_per_s": 30.0,
            "role": "backward_regression_only",
        },
    }

    return {
        "version": "2.1",
        "status": "frozen",
        "sampling_hz": sampling_hz,
        "calibration": {
            "cruise": [cruise],
            "following": [
                following_entry("C1_OPENACC_ASTAZERO", asta),
                following_entry("C2_OPENACC_VICOLUNGO", vico),
            ],
        },
        "validation": {
            "scenarios": list(validation.values()),
            "policy": "Held-out scenarios never influence Pareto weights after calibration freeze."
        },
        "development_only": {
            "cruise": [{
                "id": "DEV_C0",
                "type": "cruise_events",
                "duration_s": 40.0,
                "initial_ego_speed_mps": 22.0,
                "events": [[0.0, 25.0], [10.0, 30.0], [20.0, 22.0], [30.0, 25.0]],
                "weight": 1.0,
                "research_use": False,
            }],
            "following": [{
                "id": "DEV_F0",
                "type": "synthetic_sinusoidal_lead",
                "duration_s": 40.0,
                "initial_ego_speed_mps": 22.0,
                "initial_lead_speed_mps": 22.0,
                "initial_gap_m": 60.0,
                "lead_accel_amplitude_mps2": 1.0,
                "lead_phase_rate_deg_per_s": 30.0,
                "weight": 1.0,
                "research_use": False,
            }],
        },
        "freeze_policy": {
            "window_selection": "AstaZero/Vicolungo use the predeclared max-leader-kinematic-excitation rule; Casale uses a fixed 50-120 s offset from the first valid timestamp of part6. All rules are independent of controller results.",
            "data_integrity": "Raw and processed SHA-256 hashes are recorded per scenario.",
            "resampling": "Lead speed is linearly interpolated from source timestamps to the 60 Hz controller grid.",
            "change_control": "Changing any calibration source/window requires a new scenario-manifest version and recalibration of the cost profile."
        }
    }


def main():
    ap = argparse.ArgumentParser(description="Download/preprocess/freeze Framework V2 scenarios")
    ap.add_argument("--force-download", action="store_true", help="redownload public raw files")
    ap.add_argument("--manifest", default="scenario_manifest_acc_cost_v2.json")
    args = ap.parse_args()

    print("Preparing public trajectory sources...")
    prepared = prepare_public_trajectories(force_download=args.force_download)
    for key, traj in prepared.items():
        score = "" if traj.excitation_score is None else f" score={traj.excitation_score:.4g}"
        print(
            f"  {key:24s} {traj.start_s:.1f}-{traj.end_s:.1f}s "
            f"({traj.source_sampling_hz:.2f} Hz){score}"
        )

    manifest = build_manifest(prepared)
    path = ROOT_DIR / args.manifest
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Loading via the production loader catches path/column/resampling mistakes now,
    # before an expensive Pareto campaign starts.
    suite = load_scenario_suite(path, development=False, require_frozen=True)
    print(f"Frozen scenario manifest: {path}")
    print(f"SHA-256: {suite.manifest_sha256}")
    print(f"Calibration scenarios: {len(suite.cruise)} cruise + {len(suite.following)} following")


if __name__ == "__main__":
    main()
