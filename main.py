#!/usr/bin/env python3
"""Public entry point for the ACC evolutionary-optimization release.

Typical use:
    python main.py prepare
    python main.py optimize --workers 4
    python main.py analyze --with-ablation --workers 4

The archived paper-facing results can be explored directly in analysis.ipynb
without rerunning the expensive optimization campaign.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from core.config import ROOT_DIR, load_manifest, resolve_project_path
from core.scenarios_v2 import sha256_file


CANONICAL_RESULTS = ROOT_DIR / "research_outputs" / "canonical"


def run_module(module: str, *args: str, env: dict[str, str] | None = None) -> None:
    cmd = [sys.executable, "-m", module, *args]
    print("\n$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=True)


def verify_frozen_inputs(*, require_scenarios: bool) -> None:
    manifest = load_manifest()
    if manifest.get("benchmark") != "acc_cost_sahib_fje_v2":
        raise RuntimeError("This public release supports only the frozen V2.2-FJE benchmark.")

    profile_path = resolve_project_path(manifest["cost_framework"]["cost_profile_path"])
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    expected_vector = manifest["cost_framework"]["performance_vector"]
    if profile.get("status") != "frozen":
        raise RuntimeError("The bundled cost profile is not marked frozen.")
    if profile.get("performance_vector") != expected_vector:
        raise RuntimeError("Cost-profile performance vector does not match the experiment manifest.")

    scenario_path = resolve_project_path(manifest["cost_framework"]["scenario_manifest_path"])
    if require_scenarios:
        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Missing {scenario_path.name}. Run `python main.py prepare` first."
            )
        actual_hash = sha256_file(scenario_path)
        expected_hash = str(profile.get("scenario_manifest_sha256", ""))
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Prepared scenario manifest does not match the frozen cost profile. "
                "Do not continue with mixed data/configuration versions."
            )


def command_prepare(_args: argparse.Namespace) -> None:
    run_module("scripts.prepare_data")
    verify_frozen_inputs(require_scenarios=True)
    print("\nPrepared public trajectories match the frozen study configuration.")


def command_optimize(args: argparse.Namespace) -> None:
    scenario_path = resolve_project_path(load_manifest()["cost_framework"]["scenario_manifest_path"])
    if not scenario_path.exists():
        print("Scenario bundle is not prepared yet; preparing it now.")
        command_prepare(args)
    verify_frozen_inputs(require_scenarios=True)
    run_module("scripts.preflight", "--stage", "main")
    env = os.environ.copy()
    env["MAX_CONCURRENT_ALGOS"] = str(args.workers)
    run_module("scripts.optimize", env=env)


def command_analyze(args: argparse.Namespace) -> None:
    verify_frozen_inputs(require_scenarios=True)
    checkpoint_dir = ROOT_DIR / "checkpoint_jobs"
    if not checkpoint_dir.exists() or not any(checkpoint_dir.glob("*.pkl")):
        raise FileNotFoundError(
            "No optimizer checkpoints found. Run `python main.py optimize` first, "
            "or open analysis.ipynb to inspect the archived canonical results."
        )
    cmd = ["--workers", str(args.workers)]
    if args.with_ablation:
        cmd.append("--all")
    run_module("scripts.analyze", *cmd)


def command_all(args: argparse.Namespace) -> None:
    command_prepare(args)
    command_optimize(args)
    command_analyze(args)


def command_verify(_args: argparse.Namespace) -> None:
    verify_frozen_inputs(require_scenarios=False)
    status_path = CANONICAL_RESULTS / "evidence_status.json"
    evidence_path = CANONICAL_RESULTS / "RQ_evidence.md"
    if not status_path.exists() or not evidence_path.exists():
        raise FileNotFoundError("Archived canonical results are incomplete.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    issues = status.get("issues", [])
    if issues:
        raise RuntimeError("Canonical evidence reports issues: " + "; ".join(map(str, issues)))
    print("Public release check: PASS")
    print(f"Canonical results: {CANONICAL_RESULTS.relative_to(ROOT_DIR)}")
    print("For a no-compute walkthrough, open analysis.ipynb.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore or reproduce the frozen ACC evolutionary-optimization study."
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("prepare", help="download/process the public OpenACC and WLTC trajectories")

    p_opt = sub.add_parser("optimize", help="run the six-optimizer, 30-run benchmark")
    p_opt.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))

    p_an = sub.add_parser("analyze", help="regenerate RQ1-RQ4 evidence from local checkpoints")
    p_an.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    p_an.add_argument(
        "--with-ablation",
        action="store_true",
        help="also run/resume the paired canonical-DE objective-ablation campaign",
    )

    p_all = sub.add_parser("all", help="prepare data, optimize, then regenerate research evidence")
    p_all.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    p_all.add_argument("--with-ablation", action="store_true")

    sub.add_parser("verify", help="check bundled configuration and archived canonical evidence")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        print("\nFastest path: open analysis.ipynb to explore the archived results.")
        return 0
    if hasattr(args, "workers") and args.workers < 1:
        parser.error("--workers must be at least 1")
    {
        "prepare": command_prepare,
        "optimize": command_optimize,
        "analyze": command_analyze,
        "all": command_all,
        "verify": command_verify,
    }[args.command](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
