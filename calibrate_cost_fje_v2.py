#!/usr/bin/env python3
"""Calibrate and optionally freeze the Framework V2.2-FJE scalar cost.

Research mode requires a frozen scenario manifest. ``--smoke-test`` uses only
explicit development scenarios with tiny budgets and never writes a frozen
research profile.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from benchmarks.acc_cost_fje_v2 import (
    PERFORMANCE_NAMES,
    evaluate_performance_vector,
    sahib_contribution_weights,
    scalar_cost_from_performance,
)
from core.config import ROOT_DIR, load_manifest, resolve_project_path
from core.scenarios_v2 import load_scenario_suite
from optimizers.mopso import MOEval, merge_reference_front, run_mopso
from optimizers.pso import run_pso


MANIFEST = load_manifest()
CTRL = MANIFEST["controller"]
COST = MANIFEST["cost_framework"]
CAL = MANIFEST["calibration"]
LO = list(map(float, CTRL["min_bounds"]))
HI = list(map(float, CTRL["max_bounds"]))


def _utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mo_objective(suite):
    def evaluate(params):
        perf = evaluate_performance_vector(params, suite)
        return MOEval(perf.as_array(), perf.feasible, perf.constraint_violation)
    return evaluate


def _serialize_archive(front):
    rows = []
    for item in front:
        row = {
            "k_ve": float(item.position[0]),
            "k_de": float(item.position[1]),
            "k_vrel": float(item.position[2]),
            "feasible": bool(item.evaluation.feasible),
            "constraint_violation": float(item.evaluation.violation),
        }
        row.update({name: float(value) for name, value in zip(PERFORMANCE_NAMES, item.evaluation.values)})
        rows.append(row)
    return rows


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("cannot write empty Pareto front")
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description="Calibrate Framework V2.2-FJE")
    ap.add_argument("--smoke-test", action="store_true", help="tiny development-only integration run")
    ap.add_argument("--freeze", action="store_true", help="write frozen FJE research cost profile")
    ap.add_argument("--output-dir", default="output/calibration_fje_v2_2", help="calibration artifact directory")
    args = ap.parse_args()

    if args.smoke_test and args.freeze:
        raise SystemExit("--smoke-test can never --freeze a research cost profile")
    if MANIFEST["benchmark"] != "acc_cost_sahib_fje_v2":
        raise RuntimeError(
            "calibrate_cost_fje_v2.py requires the FJE manifest. "
            "Use ACC_MANIFEST=experiment_manifest_acc_cost_fje_v2_2.json or the default config."
        )

    suite = load_scenario_suite(
        COST["scenario_manifest_path"],
        development=args.smoke_test,
        require_frozen=not args.smoke_test,
    )
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = dict(CAL["mopso"])
    pso_cfg = dict(CAL["pso_verification"])
    if args.smoke_test:
        cfg.update(independent_runs=1, population_size=8, max_iterations=2, archive_size=16, grid_divisions=5)
        pso_cfg.update(trials=1, population_size=8, max_iterations=2)

    fronts = []
    objective = _mo_objective(suite)
    for run_idx in range(int(cfg["independent_runs"])):
        seed = int(cfg["base_seed"]) + run_idx
        front = run_mopso(
            objective,
            LO,
            HI,
            population_size=int(cfg["population_size"]),
            max_iter=int(cfg["max_iterations"]),
            archive_size=int(cfg["archive_size"]),
            grid_divisions=int(cfg["grid_divisions"]),
            grid_inflation=float(cfg["grid_inflation"]),
            inertia=float(cfg["inertia"]),
            inertia_damping=float(cfg["inertia_damping"]),
            c1=float(cfg["c1"]),
            c2=float(cfg["c2"]),
            leader_pressure=float(cfg["leader_pressure"]),
            deletion_pressure=float(cfg["deletion_pressure"]),
            mutation_rate=float(cfg["mutation_rate"]),
            seed=seed,
        )
        fronts.append(front)
        _write_csv(output_dir / f"pareto_run_{run_idx:02d}.csv", _serialize_archive(front))
        print(f"FJE MOPSO run {run_idx+1}/{cfg['independent_runs']}: archive={len(front)}")

    reference = [x for x in merge_reference_front(fronts) if x.evaluation.feasible]
    if not reference:
        raise RuntimeError("No feasible FJE nondominated solutions found; calibration cannot define weights")
    reference_path = output_dir / "reference_front.csv"
    _write_csv(reference_path, _serialize_archive(reference))

    objective_matrix = np.asarray([x.evaluation.values for x in reference], dtype=float)
    weights = sahib_contribution_weights(objective_matrix)
    means = objective_matrix.mean(axis=0)
    print("FJE contribution means:", dict(zip(PERFORMANCE_NAMES, means.tolist())))
    print("FJE Sahib weights:", dict(zip(PERFORMANCE_NAMES, weights.tolist())))

    def scalar_fitness(params):
        return scalar_cost_from_performance(evaluate_performance_vector(params, suite), weights)

    verification = []
    for trial in range(int(pso_cfg["trials"])):
        seed = int(pso_cfg["base_seed"]) + trial
        sol, fit, curve = run_pso(
            scalar_fitness,
            LO,
            HI,
            population_size=int(pso_cfg["population_size"]),
            max_iter=int(pso_cfg["max_iterations"]),
            c1=float(pso_cfg["c1"]),
            c2=float(pso_cfg["c2"]),
            inertia_start=float(pso_cfg["inertia_start"]),
            inertia_end=float(pso_cfg["inertia_end"]),
            seed=seed,
        )
        perf = evaluate_performance_vector(sol, suite)
        record = {
            "trial": trial,
            "seed": seed,
            "best_solution": list(map(float, sol)),
            "best_fitness": float(fit),
            "performance": perf.as_dict(),
            "feasible": bool(perf.feasible),
            "constraint_violation": float(perf.constraint_violation),
            "curve": list(map(float, curve)),
        }
        verification.append(record)
        print(f"FJE PSO verify {trial+1}/{pso_cfg['trials']}: J={fit:.8g}, gains={sol}")

    fitnesses = np.asarray([r["best_fitness"] for r in verification], dtype=float)
    median_fit = float(np.median(fitnesses))
    representative = min(verification, key=lambda r: abs(r["best_fitness"] - median_fit))
    verify_summary = {
        "mean_best_fitness": float(np.mean(fitnesses)),
        "median_best_fitness": median_fit,
        "std_best_fitness": float(np.std(fitnesses, ddof=1)) if fitnesses.size > 1 else 0.0,
        "min_best_fitness": float(np.min(fitnesses)),
        "max_best_fitness": float(np.max(fitnesses)),
        "representative_rule": "observed trial nearest median best fitness",
    }
    verification_path = output_dir / "pso_verification.json"
    verification_path.write_text(
        json.dumps({"summary": verify_summary, "trials": verification, "representative": representative}, indent=2),
        encoding="utf-8",
    )

    calibration_summary = {
        "framework_version": MANIFEST["framework_version"],
        "benchmark": MANIFEST["benchmark"],
        "created_at_utc": _utc_now(),
        "mode": "development_smoke" if args.smoke_test else "research_calibration",
        "scenario_manifest": str(suite.manifest_path),
        "scenario_manifest_sha256": suite.manifest_sha256,
        "reference_front_size": len(reference),
        "performance_vector": list(PERFORMANCE_NAMES),
        "objective_means": {name: float(v) for name, v in zip(PERFORMANCE_NAMES, means)},
        "weights": {name: float(v) for name, v in zip(PERFORMANCE_NAMES, weights)},
        "energy_definition": COST.get("energy_definition", {}),
        "mopso": cfg,
        "pso_verification": pso_cfg,
        "verification_summary": verify_summary,
        "representative_verification": representative,
    }
    (output_dir / "calibration_summary.json").write_text(json.dumps(calibration_summary, indent=2), encoding="utf-8")

    if args.freeze:
        profile_path = resolve_project_path(COST["cost_profile_path"])
        profile = {
            "framework_version": MANIFEST["framework_version"],
            "benchmark": MANIFEST["benchmark"],
            "status": "frozen",
            "frozen_at_utc": _utc_now(),
            "weights": calibration_summary["weights"],
            "performance_vector": list(PERFORMANCE_NAMES),
            "energy_definition": COST.get("energy_definition", {}),
            "generated_by": "calibrate_cost_fje_v2.py --freeze",
            "scenario_manifest_sha256": suite.manifest_sha256,
            "reference_front_path": str(reference_path.relative_to(ROOT_DIR)),
            "verification_path": str(verification_path.relative_to(ROOT_DIR)),
            "verification_summary": verify_summary,
            "representative_verification": representative,
            "note": "FJE profile. Changing scenarios, physics, energy semantics, or weights requires a new framework/profile version.",
        }
        profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
        print(f"FROZEN FJE cost profile written to {profile_path}")
    elif args.smoke_test:
        print("Development FJE smoke completed; no research profile was frozen.")
    else:
        print("FJE calibration completed. Inspect artifacts, then rerun with --freeze.")


if __name__ == "__main__":
    main()
