#!/usr/bin/env python3
"""One-command orchestration for the active ACC framework.

Default Framework V2.2-FJE path:
  1. run tests;
  2. acquire/freeze public scenario inputs;
  3. preflight calibration inputs;
  4. run 4-objective FJE MOPSO -> Sahib weights -> PSO verification;
  5. freeze the FJE cost profile;
  6. preflight and launch/resume the 180-job six-optimizer benchmark.

Older manifests can still be selected with ACC_MANIFEST.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from core.config import ROOT_DIR, load_manifest, resolve_project_path
from core.scenarios_v2 import sha256_file


def run(*args: str) -> None:
    cmd = [sys.executable, *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def _manifest():
    return load_manifest()


def scenario_is_ready() -> bool:
    exp = _manifest()
    path = resolve_project_path(exp["cost_framework"]["scenario_manifest_path"])
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("status") != "frozen":
        return False
    for group in ("cruise", "following"):
        for entry in data.get("calibration", {}).get(group, []):
            pval = entry.get("path")
            if not pval:  # synthetic cruise may be embedded rather than file-backed
                continue
            p = resolve_project_path(str(pval))
            if not p.exists():
                return False
            expected = str(entry.get("provenance", {}).get("processed_sha256", ""))
            if expected and sha256_file(p) != expected:
                return False
    return True


def cost_is_ready() -> bool:
    exp = _manifest()
    sp = resolve_project_path(exp["cost_framework"]["scenario_manifest_path"])
    cp = resolve_project_path(exp["cost_framework"]["cost_profile_path"])
    if not sp.exists() or not cp.exists():
        return False
    try:
        data = json.loads(cp.read_text(encoding="utf-8"))
    except Exception:
        return False
    return (
        data.get("status") == "frozen"
        and data.get("scenario_manifest_sha256") == sha256_file(sp)
        and data.get("performance_vector") == exp["cost_framework"].get("performance_vector")
    )


def calibration_script() -> str:
    exp = _manifest()
    return str(exp["cost_framework"].get("calibration_script", "calibrate_cost_v2.py"))


def calibration_output_dir(smoke: bool = False) -> str:
    exp = _manifest()
    if exp["benchmark"] == "acc_cost_sahib_fje_v2":
        return "output/calibration_fje_v2_2_smoke" if smoke else "output/calibration_fje_v2_2"
    return "output/calibration_v2_smoke" if smoke else "output/calibration_v2"


def main():
    ap = argparse.ArgumentParser(description="Prepare, calibrate, and run the ACC benchmark")
    ap.add_argument("--smoke-test", action="store_true", help="fast offline integration path")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--force-data", action="store_true", help="redownload public trajectory data")
    ap.add_argument("--force-calibration", action="store_true", help="rebuild frozen cost profile")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--calibrate-only", action="store_true")
    ap.add_argument("--main-only", action="store_true", help="require already-frozen scenario + cost profile")
    ap.add_argument("--research-only", action="store_true", help="evaluate existing checkpoints for RQ1-RQ3 without recalibration")
    ap.add_argument("--research-all", action="store_true", help="RQ1-RQ4, including isolated paired objective ablation")
    args = ap.parse_args()
    if sum((args.research_only,args.research_all)) > 1:
        ap.error("choose one research entry point")
    if args.research_only or args.research_all:
        if any((args.smoke_test,args.force_data,args.force_calibration,args.prepare_only,args.calibrate_only,args.main_only)):
            ap.error("research evaluation cannot be combined with training/calibration stage flags")
        if not args.skip_tests:
            run("-m", "unittest", "discover", "-s", "tests", "-v")
        command=["research_pipeline.py"]
        if args.research_all: command.append("--all")
        run(*command)
        return

    exp = _manifest()
    print(f"Active framework: {exp['framework_version']} | benchmark={exp['benchmark']}")

    if args.smoke_test:
        if not args.skip_tests:
            run("-m", "unittest", "discover", "-s", "tests", "-v")
        run(calibration_script(), "--smoke-test", "--output-dir", calibration_output_dir(smoke=True))
        run("optimize_parallel.py", "--smoke-test")
        print("\nSMOKE PIPELINE PASS")
        return

    if not args.skip_tests:
        run("-m", "unittest", "discover", "-s", "tests", "-v")

    if args.main_only:
        run("preflight_v2.py", "--stage", "main")
        run("optimize_parallel.py")
        return

    if args.force_data or not scenario_is_ready():
        cmd = [str(exp["cost_framework"].get("scenario_bootstrap", "prepare_scenarios_v2.py"))]
        if args.force_data:
            cmd.append("--force-download")
        run(*cmd)
    else:
        print("\nScenario bundle already frozen and hash-valid; reusing it.")

    run("preflight_v2.py", "--stage", "calibration")
    if args.prepare_only:
        print("\nPREPARE-ONLY COMPLETE")
        return

    if args.force_calibration or not cost_is_ready():
        run(calibration_script(), "--freeze", "--output-dir", calibration_output_dir(smoke=False))
    else:
        print("\nFrozen cost profile matches scenario manifest and active performance vector; reusing it.")

    run("preflight_v2.py", "--stage", "main")
    if args.calibrate_only:
        print("\nCALIBRATE-ONLY COMPLETE")
        return

    run("optimize_parallel.py")


if __name__ == "__main__":
    main()
