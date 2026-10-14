#!/usr/bin/env python3
"""Fail-fast integrity checks for the active ACC framework.

Framework V2.2-FJE is the default.  The older V2.1 FJ manifest remains usable
through ACC_MANIFEST for historical reproduction.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys

import numpy as np

from core.config import load_manifest, resolve_project_path
from core.scenarios_v2 import load_scenario_suite, load_validation_scenarios


def _active_cost_api(manifest):
    key = manifest["benchmark"]
    if key == "acc_cost_sahib_fje_v2":
        from benchmarks.acc_cost_fje_v2 import evaluate_performance_vector, load_cost_profile, PERFORMANCE_NAMES
        return evaluate_performance_vector, load_cost_profile, PERFORMANCE_NAMES
    if key == "acc_cost_sahib_v2":
        from benchmarks.acc_cost_v2 import evaluate_performance_vector, load_cost_profile
        return evaluate_performance_vector, load_cost_profile, (
            "cruise_speed_rmse", "following_spacing_rmse", "realized_jerk_rms"
        )
    raise RuntimeError(f"preflight_v2.py does not handle benchmark {key!r}")


def check(stage: str) -> None:
    manifest = load_manifest()
    cf = manifest["cost_framework"]
    opt = manifest["optimization"]
    ctrl = manifest["controller"]
    evaluate_performance_vector, load_cost_profile, performance_names = _active_cost_api(manifest)

    expected_jobs = len(opt["algorithms"]) * int(opt["independent_runs"]) * len(opt["objectives"])
    if int(opt["total_jobs"]) != expected_jobs:
        raise RuntimeError(f"manifest total_jobs={opt['total_jobs']} but contract implies {expected_jobs}")
    if not (len(ctrl["decision_vector"]) == len(ctrl["min_bounds"]) == len(ctrl["max_bounds"]) == 3):
        raise RuntimeError("controller decision vector/bounds must all have dimension 3")
    if int(manifest["execution_backend"]["sampling_hz"]) != 60:
        raise RuntimeError("active framework is frozen to a 60 Hz controller/simulation grid")
    if list(cf["performance_vector"]) != list(performance_names):
        raise RuntimeError(
            f"manifest performance vector {cf['performance_vector']} does not match evaluator {list(performance_names)}"
        )

    print(f"Framework: {manifest['framework_version']} | benchmark={manifest['benchmark']}")
    print(f"Python: {platform.python_version()} | platform={platform.platform()}")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >= 3.10 is required")

    scenario_path = resolve_project_path(cf["scenario_manifest_path"])
    if not scenario_path.exists():
        raise FileNotFoundError(f"missing scenario manifest: {scenario_path}")
    scenario_json = json.loads(scenario_path.read_text(encoding="utf-8"))
    if int(scenario_json.get("sampling_hz", -1)) != int(manifest["execution_backend"]["sampling_hz"]):
        raise RuntimeError("scenario sampling_hz does not match experiment execution_backend sampling_hz")

    if stage == "development":
        suite = load_scenario_suite(scenario_path, development=True, require_frozen=False)
        print(f"Development scenarios OK: {len(suite.cruise)} cruise, {len(suite.following)} following")
        perf = evaluate_performance_vector([1.0, 1.0, 1.0], suite)
        values = perf.as_array()
        if values.shape != (len(performance_names),) or not np.isfinite(values).all():
            raise RuntimeError(f"development metric smoke failed: {values}")
        print(f"Metric smoke: {dict(zip(performance_names, values.tolist()))} feasible={perf.feasible}")
        print(f"PREFLIGHT PASS [{stage}]")
        return

    suite = load_scenario_suite(scenario_path, development=False, require_frozen=True)
    print(f"Scenario manifest frozen: {scenario_path.name}")
    print(f"Scenario SHA-256: {suite.manifest_sha256}")
    print(f"Calibration bank: {len(suite.cruise)} cruise + {len(suite.following)} following")
    validation = load_validation_scenarios(scenario_path, require_frozen=True)
    print(f"Validation bank: {len(validation)} following/regression scenarios (paths + hashes verified)")

    neutral = [1.0, 1.0, 1.0]
    perf = evaluate_performance_vector(neutral, suite)
    values = perf.as_array()
    if values.shape != (len(performance_names),) or not np.isfinite(values).all():
        raise RuntimeError("performance-vector preflight returned malformed/non-finite values")
    if manifest["benchmark"] == "acc_cost_sahib_fje_v2" and values[3] <= 0.0:
        raise RuntimeError(f"FJE energy term must be positive on the calibration bank; got {values[3]}")
    print(f"Kernel/data preflight: {dict(zip(performance_names, values.tolist()))} feasible={perf.feasible}")

    if stage == "main":
        profile = load_cost_profile(require_frozen=True)
        if profile.scenario_manifest_sha256 != suite.manifest_sha256:
            raise RuntimeError(
                "Frozen cost profile does not match current scenario manifest. Re-run the active calibration script."
            )
        if profile.weights.shape != (len(performance_names),):
            raise RuntimeError(f"wrong frozen weight dimension: {profile.weights}")
        if np.any(profile.weights <= 0.0) or not np.isclose(profile.weights.sum(), 1.0):
            raise RuntimeError(f"invalid frozen weights: {profile.weights}")
        profile_json = json.loads(profile.path.read_text(encoding="utf-8"))
        if list(profile_json.get("performance_vector", [])) != list(performance_names):
            raise RuntimeError("frozen cost-profile performance vector does not match active evaluator")
        for field in ("reference_front_path", "verification_path"):
            value = profile_json.get(field)
            if not value:
                raise RuntimeError(f"frozen cost profile is missing {field}")
            artifact = resolve_project_path(str(value))
            if not artifact.exists():
                raise FileNotFoundError(f"frozen cost profile references missing artifact: {artifact}")
        print(f"Cost profile frozen: weights={dict(zip(performance_names, profile.weights.tolist()))}")

    print(f"PREFLIGHT PASS [{stage}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["development", "calibration", "main"], default="main")
    args = ap.parse_args()
    check(args.stage)


if __name__ == "__main__":
    main()
