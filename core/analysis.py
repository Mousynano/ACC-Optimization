"""Comprehensive post-processing and reporting for the active ACC FJE campaign.

This module keeps post-processing separate from the frozen optimization
fitness and uses the active Framework V2.2-FJE experiment contract and data sources.

Generated evidence covers:
  * all-run scalar objective and four FJE physical components;
  * median/IQR convergence;
  * Shapiro diagnostics -> independent Kruskal-Wallis/Mann-Whitney with corrections;
  * representative-controller motion for OpenACC, WLTC, periodic cruise, and
    sinusoidal car-following;
  * TTC and an explicitly operational/observed TTE diagnostic;
  * cruise time-response metrics for every synthetic set-speed transition;
  * the inherited 12-scenario physical-parameter robustness stress test;
  * reproducibility / consistency audit tables and metadata.

The observed TTE implemented here is a reporting metric, not the analytical TTE
formula of Zhuo & Zhu.  For each detected finite lead-vehicle disturbance, it is
the elapsed time from disturbance end until both relative speed and CTG spacing
error enter predefined equilibrium bands and remain there for a hold window.
Those thresholds are written to analysis_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import shutil
from dataclasses import dataclass, asdict
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from benchmarks.acc_cost_fje_v2 import (
    PERFORMANCE_NAMES,
    evaluate_performance_vector,
    load_cost_profile,
)
from core.checkpoint_jobs import load_all_job_checkpoints
from core.config import load_manifest, resolve_project_path
from core.energy import trajectory_traction_energy
from core.physics import VehiclePhysics
from core.scenarios_v2 import (
    CruiseScenario,
    FollowingScenario,
    load_scenario_suite,
    load_validation_scenarios,
)
from core.sensing import RangeOnlySensor, RangeSensorConfig
from core.utils import compute_ttc_series, step_response_metrics


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis_outputs"
GAIN_NAMES = ("k_ve", "k_de", "k_vrel")

# Reporting-only safety / equilibrium diagnostics. These are deliberately kept
# outside the optimization cost and persisted in metadata for auditability.
TTC_REPORT_THRESHOLD_S = 1.5
TTC_PLOT_CAP_S = 20.0
TTE_LEAD_ACCEL_THRESHOLD_MPS2 = 0.25
TTE_MIN_DISTURBANCE_S = 0.50
TTE_MERGE_GAP_S = 0.75
TTE_SPEED_TOL_MPS = 0.20
TTE_SPACING_TOL_M = 0.75
TTE_SPACING_TOL_FRACTION = 0.02
TTE_HOLD_S = 2.0

# Physical robustness scenarios used by the frozen study. ±10 degree
# slopes are stress cases rather than nominal-road claims.
ROBUSTNESS_SCENARIOS = (
    ("S0",  "Nominal",          0.0,  1.00, 1.00),
    ("S1",  "Mass +10%",        0.0,  1.10, 1.00),
    ("S2",  "Mass -10%",        0.0,  0.90, 1.00),
    ("S3",  "Crr +20%",         0.0,  1.00, 1.20),
    ("S4",  "Uphill +10deg",   10.0,  1.00, 1.00),
    ("S5",  "Up + Mass +10%",  10.0,  1.10, 1.00),
    ("S6",  "Up + Mass -10%",  10.0,  0.90, 1.00),
    ("S7",  "Up + Crr +20%",   10.0,  1.00, 1.20),
    ("S8",  "Downhill -10deg", -10.0, 1.00, 1.00),
    ("S9",  "Down + Mass +10%",-10.0, 1.10, 1.00),
    ("S10", "Down + Mass -10%",-10.0, 0.90, 1.00),
    ("S11", "Down + Crr +20%", -10.0, 1.00, 1.20),
)

# Stable paper-facing algorithm order / styling.
STYLE = {
    "ga":     dict(label="GA",                 color="#8e44ad", marker="o", ls="-"),
    "de":     dict(label="DE",                 color="#34495e", marker="s", ls="--"),
    "fvrade": dict(label="FVRADE",             color="#16a085", marker="^", ls="-"),
    "rnegde": dict(label="RNEGDE",             color="#c0392b", marker="D", ls="-."),
    "nshade": dict(label="Neighborhood-SHADE", color="#2980b9", marker="v", ls="-"),
    "denps":  dict(label="DE-NPS-FP",          color="#d68910", marker="P", ls=":"),
}


def _label(algo: str) -> str:
    return STYLE.get(algo, {}).get("label", algo.upper())


def _ensure_dirs(base: Path) -> dict[str, Path]:
    dirs = {
        "objective": base / "01_objective",
        "convergence": base / "02_convergence",
        "statistics": base / "03_statistics",
        "motion": base / "04_motion",
        "safety": base / "05_safety",
        "time_response": base / "06_time_response",
        "robustness": base / "07_robustness",
        "audit": base / "08_audit",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    (dirs["statistics"] / "qq").mkdir(parents=True, exist_ok=True)
    return dirs


def _to_float_or_nan(value):
    try:
        if value is None:
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def load_run_frame(manifest: dict) -> pd.DataFrame:
    benchmark = manifest["benchmark"]
    algos = list(manifest["optimization"]["algorithms"])
    objectives = list(manifest["optimization"]["objectives"])
    if len(objectives) != 1:
        raise RuntimeError("This FJE reporting pipeline expects one frozen scalar objective")
    obj = objectives[0]
    stats_map = load_all_job_checkpoints(benchmark, algos, [obj])
    rows = []
    for algo in algos:
        bucket = stats_map[obj][algo]
        for i, run_id in enumerate(bucket["run_ids"]):
            sol = np.asarray(bucket["best_solutions"][i], dtype=float).reshape(-1)
            rows.append({
                "algorithm": algo,
                "algorithm_label": _label(algo),
                "run_id": int(run_id),
                "seed": bucket["seeds"][i],
                "controller_id": f"{benchmark}:{algo}:{int(run_id)}",
                "fitness": _to_float_or_nan(bucket["best_fitness"][i]),
                "runtime_s": _to_float_or_nan(bucket["durations"][i]),
                "function_evaluations": _to_float_or_nan(bucket["function_evaluations"][i]),
                "solution": sol,
                "curve": bucket["curves"][i],
                **{name: float(sol[j]) if j < sol.size else np.nan for j, name in enumerate(GAIN_NAMES)},
            })
    return pd.DataFrame(rows)


def select_observed_median_representatives(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo, group in runs.groupby("algorithm", sort=False):
        median = float(group["fitness"].median())
        idx = (group["fitness"] - median).abs().idxmin()
        row = group.loc[idx]
        rows.append({
            "algorithm": algo,
            "algorithm_label": _label(algo),
            "run_id": int(row["run_id"]),
            "fitness": float(row["fitness"]),
            "k_ve": float(row["k_ve"]),
            "k_de": float(row["k_de"]),
            "k_vrel": float(row["k_vrel"]),
        })
    return pd.DataFrame(rows)


def evaluate_all_components(runs: pd.DataFrame, suite, profile) -> pd.DataFrame:
    rows = []
    for _, row in runs.iterrows():
        gains = np.asarray([row[n] for n in GAIN_NAMES], dtype=float)
        perf = evaluate_performance_vector(gains, suite)
        rec = {
            "algorithm": row["algorithm"],
            "algorithm_label": row["algorithm_label"],
            "run_id": int(row["run_id"]),
            "fitness": float(row["fitness"]),
            "feasible": bool(perf.feasible),
        }
        for i, name in enumerate(PERFORMANCE_NAMES):
            value = float(perf.as_array()[i])
            rec[name] = value
            rec[f"weighted_{name}"] = float(profile.weights[i] * value)
        rec["reconstructed_cost"] = float(sum(rec[f"weighted_{n}"] for n in PERFORMANCE_NAMES))
        rec["cost_reconstruction_abs_error"] = abs(rec["reconstructed_cost"] - rec["fitness"])
        rows.append(rec)
    return pd.DataFrame(rows)


def objective_outputs(runs, components, reps, manifest, profile, out_dir: Path):
    runs_export = runs.drop(columns=["solution", "curve"], errors="ignore").copy()
    runs_export.to_csv(out_dir / "checkpoint_runs.csv", index=False)
    components.to_csv(out_dir / "performance_components_all_runs.csv", index=False)
    reps.to_csv(out_dir / "representative_controllers.csv", index=False)

    summary_rows = []
    for algo, g in runs.groupby("algorithm", sort=False):
        summary_rows.append({
            "algorithm": algo,
            "algorithm_label": _label(algo),
            "n": len(g),
            "fitness_median": float(g.fitness.median()),
            "fitness_iqr": float(g.fitness.quantile(.75) - g.fitness.quantile(.25)),
            "fitness_mean": float(g.fitness.mean()),
            "fitness_std": float(g.fitness.std(ddof=1)),
            "fitness_min": float(g.fitness.min()),
            "fitness_max": float(g.fitness.max()),
            "runtime_mean_s": float(g.runtime_s.mean()),
            "runtime_median_s": float(g.runtime_s.median()),
            "fe_mean": float(g.function_evaluations.mean()),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "objective_summary.csv", index=False)

    # Scalar objective distribution.
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    data = [runs.loc[runs.algorithm == a, "fitness"].to_numpy(float) for a in manifest["optimization"]["algorithms"]]
    labels = [_label(a) for a in manifest["optimization"]["algorithms"]]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
    for patch, algo in zip(bp["boxes"], manifest["optimization"]["algorithms"]):
        patch.set_facecolor(STYLE[algo]["color"])
        patch.set_alpha(.55)
    ax.set_ylabel("Frozen FJE scalar cost")
    ax.set_title("Final objective distribution across independent runs")
    ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    fig.savefig(out_dir / "objective_distribution_boxplot.png", dpi=300)
    plt.close(fig)

    # Run-index stability across independent optimization runs.
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for algo in manifest["optimization"]["algorithms"]:
        g = runs[runs.algorithm == algo].sort_values("run_id")
        st = STYLE[algo]
        ax.plot(g.run_id + 1, g.fitness, label=st["label"], color=st["color"],
                marker=st["marker"], linestyle=st["ls"], markersize=4, linewidth=1.2, alpha=.85)
    ax.set_xlabel("Independent run index")
    ax.set_ylabel("Best frozen FJE scalar cost")
    ax.set_title("Best-fitness stability across runs")
    ax.grid(alpha=.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "objective_stability_by_run.png", dpi=300)
    plt.close(fig)

    # Physical component distributions across all runs. Each metric gets its own
    # panel because their units/scales differ by design.
    for metric in PERFORMANCE_NAMES:
        fig, ax = plt.subplots(figsize=(10.5, 5.2))
        data = [components.loc[components.algorithm == a, metric].to_numpy(float)
                for a in manifest["optimization"]["algorithms"]]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=True)
        for patch, algo in zip(bp["boxes"], manifest["optimization"]["algorithms"]):
            patch.set_facecolor(STYLE[algo]["color"]); patch.set_alpha(.55)
        ax.set_title(f"Physical component distribution: {metric}")
        ax.set_ylabel(metric.replace("_", " "))
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"component_{metric}_boxplot.png", dpi=300)
        plt.close(fig)

    # Cost contribution decomposition for each observed-median representative.
    rep_comp = components.merge(reps[["algorithm", "run_id"]], on=["algorithm", "run_id"], how="inner")
    rep_comp.to_csv(out_dir / "representative_component_decomposition.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(rep_comp))
    bottom = np.zeros(len(rep_comp))
    component_colors = ["#4c78a8", "#f58518", "#54a24b", "#e45756"]
    for color, name in zip(component_colors, PERFORMANCE_NAMES):
        vals = rep_comp[f"weighted_{name}"].to_numpy(float)
        ax.bar(x, vals, bottom=bottom, label=name.replace("_", " "), color=color)
        bottom += vals
    ax.set_xticks(x, [_label(a) for a in rep_comp.algorithm])
    ax.set_ylabel("Contribution to scalar FJE cost")
    ax.set_title("Representative-controller FJE cost decomposition")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "representative_cost_decomposition.png", dpi=300)
    plt.close(fig)

    # Gain/boundary audit.
    bounds_lo = np.asarray(manifest["controller"]["min_bounds"], dtype=float)
    bounds_hi = np.asarray(manifest["controller"]["max_bounds"], dtype=float)
    eps = 1e-8
    boundary_rows = []
    for _, row in runs.iterrows():
        for j, name in enumerate(GAIN_NAMES):
            value = float(row[name])
            boundary_rows.append({
                "algorithm": row.algorithm,
                "algorithm_label": row.algorithm_label,
                "run_id": int(row.run_id),
                "gain": name,
                "value": value,
                "at_lower_bound": abs(value - bounds_lo[j]) <= eps,
                "at_upper_bound": abs(value - bounds_hi[j]) <= eps,
                "normalized_position": (value - bounds_lo[j]) / (bounds_hi[j] - bounds_lo[j]),
            })
    pd.DataFrame(boundary_rows).to_csv(out_dir / "gain_boundary_audit.csv", index=False)


def convergence_outputs(runs: pd.DataFrame, manifest: dict, out_dir: Path):
    rows = []
    fig, ax = plt.subplots(figsize=(11, 6))
    for algo in manifest["optimization"]["algorithms"]:
        curves = [np.asarray(c, dtype=float).reshape(-1) for c in runs.loc[runs.algorithm == algo, "curve"] if c is not None]
        if not curves:
            continue
        n = min(map(len, curves))
        arr = np.vstack([c[:n] for c in curves])
        it = np.arange(1, n + 1)
        med = np.median(arr, axis=0)
        q1 = np.quantile(arr, .25, axis=0)
        q3 = np.quantile(arr, .75, axis=0)
        st = STYLE[algo]
        ax.plot(it, med, label=st["label"], color=st["color"], linestyle=st["ls"], linewidth=1.8)
        ax.fill_between(it, q1, q3, color=st["color"], alpha=.12)
        for i in range(n):
            rows.append({"algorithm": algo, "iteration": i+1, "median": med[i], "q1": q1[i], "q3": q3[i]})
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Frozen FJE scalar cost")
    ax.set_title("Convergence across independent runs: median with IQR")
    ax.grid(alpha=.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_median_iqr.png", dpi=300)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "convergence_median_iqr.csv", index=False)


def _rank_biserial_paired(x: np.ndarray, y: np.ndarray) -> float:
    d = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    d = d[np.isfinite(d) & (d != 0)]
    if d.size == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(d), method="average")
    pos = float(ranks[d > 0].sum())
    neg = float(ranks[d < 0].sum())
    denom = pos + neg
    return (pos - neg) / denom if denom > 0 else 0.0


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = min(1.0, (m - rank) * p[idx])
        running = max(running, val)
        adjusted[idx] = running
    return adjusted


def statistical_outputs(runs, components, manifest, out_dir: Path):
    """Independent optimizer runs; no run-id pairing without a designed block."""
    from core.research_statistics import independent_comparisons
    merged = runs[["algorithm", "run_id", "fitness"]].merge(
        components[["algorithm", "run_id", *PERFORMANCE_NAMES]],
        on=["algorithm", "run_id"], validate="one_to_one")
    metrics = ["fitness", *PERFORMANCE_NAMES]
    rows = []
    for metric in metrics:
        for algorithm, group in merged.groupby("algorithm", sort=False):
            values = group[metric].to_numpy(float)
            values = values[np.isfinite(values)]
            w, p = stats.shapiro(values) if len(values) >= 3 else (np.nan, np.nan)
            rows.append({"algorithm": algorithm, "metric": metric, "n": len(values), "W": w, "p_value": p})
            if len(values) >= 3:
                fig, ax = plt.subplots(figsize=(5.4, 5))
                stats.probplot(values, dist="norm", plot=ax)
                ax.set_title(f"Q-Q: {_label(algorithm)} / {metric}")
                fig.tight_layout(); fig.savefig(out_dir / "qq" / f"qq_{metric}_{algorithm}.png", dpi=240)
                plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "shapiro_wilk.csv", index=False)
    omnibus, pairs = independent_comparisons(merged, "algorithm", metrics)
    omnibus.to_csv(out_dir / "kruskal_wallis.csv", index=False)
    pairs.to_csv(out_dir / "independent_mann_whitney_corrected.csv", index=False)
    for obsolete in ("friedman.csv", "paired_wilcoxon_corrected.csv"):
        (out_dir / obsolete).unlink(missing_ok=True)


def build_periodic_cruise() -> CruiseScenario:
    speeds = np.asarray([25., 20., 30., 22., 28., 24., 29., 21., 27., 23., 25.], dtype=float)
    plateau_s = 20.0
    times = np.arange(len(speeds), dtype=float) * plateau_s
    return CruiseScenario(
        scenario_id="SYN_CRUISE_VSET_10X",
        duration_s=float(len(speeds) * plateau_s),
        initial_ego_speed_mps=22.0,
        event_times_s=times,
        set_speeds_mps=speeds,
        weight=1.0,
        provenance="reporting-only periodic/step cruise profile with ten set-speed transitions",
    )


def build_sinusoidal_following(sampling_hz: int) -> FollowingScenario:
    duration_s = 60.0
    dt = 1.0 / sampling_hz
    steps = int(round(duration_s * sampling_hz))
    lead = np.empty(steps)
    v, phase = 22.0, 0.0
    for i in range(steps):
        phase = (phase + 30.0 * dt) % 360.0
        v = max(v + math.sin(math.radians(phase)) * dt, 0.0)
        lead[i] = v
    return FollowingScenario(
        scenario_id="SYN_FOLLOW_SINUSOIDAL",
        duration_s=duration_s,
        initial_ego_speed_mps=22.0,
        initial_lead_speed_mps=22.0,
        initial_gap_m=60.0,
        lead_speed_mps=lead,
        weight=1.0,
        provenance="reporting-only sinusoidal lead acceleration, 1 m/s^2 amplitude, 30 deg/s phase rate",
    )


@dataclass(frozen=True)
class PhysicsOverride:
    mass_kg: float
    rolling_resistance: float
    road_slope_deg: float


def _nominal_physics(manifest: dict) -> PhysicsOverride:
    p = manifest["physics"]
    return PhysicsOverride(float(p["nominal_mass_kg"]), float(p["rolling_resistance"]), float(p["road_slope_deg"]))


def simulate_cruise_reference(gains: Sequence[float], scenario: CruiseScenario, manifest: dict,
                    physics_override: PhysicsOverride | None = None) -> dict[str, list[float]]:
    p = manifest["physics"]
    c = manifest["controller"]
    hz = int(manifest["execution_backend"]["sampling_hz"])
    dt = 1.0 / hz
    po = physics_override or _nominal_physics(manifest)
    physics = VehiclePhysics(M=po.mass_kg, Cd=float(p.get("drag_coefficient", .36)),
                             A=float(p.get("frontal_area_m2", 2.42)), rho=float(p.get("air_density_kg_m3", 1.225)),
                             Croll=po.rolling_resistance, g=float(p.get("gravity_m_per_s2", 9.81)), theta=po.road_slope_deg)
    fmax = float(p["max_longitudinal_force_n"])
    k_ve = float(gains[0])
    v, a, prev_a, event_idx = float(scenario.initial_ego_speed_mps), 0.0, None, 0
    hist = {k: [] for k in ["time", "v_set", "v_ego", "a_cmd", "a_ego", "jerk", "speed_error",
                                  "traction_force_n", "tractive_power_w", "positive_tractive_power_w"]}
    for i in range(int(round(scenario.duration_s * hz))):
        t = i * dt
        while event_idx + 1 < len(scenario.event_times_s) and t >= scenario.event_times_s[event_idx+1]:
            event_idx += 1
        vset = float(scenario.set_speeds_mps[event_idx])
        err = vset - v
        acmd = k_ve * err
        v_before = v
        v_new, a_new, force = physics.step_acceleration_command(v, acmd, dt, Fmax=fmax)
        jerk = 0.0 if prev_a is None else (a_new - prev_a) / dt
        power = force * max(v_before, 0.0)
        vals = dict(time=t, v_set=vset, v_ego=v_before, a_cmd=acmd, a_ego=a_new, jerk=jerk,
                    speed_error=err, traction_force_n=force, tractive_power_w=power,
                    positive_tractive_power_w=max(power, 0.0))
        for k, val in vals.items(): hist[k].append(float(val))
        prev_a, v, a = a_new, v_new, a_new
    return hist


def simulate_following_reference(gains: Sequence[float], scenario: FollowingScenario, manifest: dict,
                       physics_override: PhysicsOverride | None = None,
                       sensor_override: RangeSensorConfig | None = None) -> dict[str, list[float]]:
    p = manifest["physics"]
    c = manifest["controller"]
    s = manifest["sensing"]
    hz = int(manifest["execution_backend"]["sampling_hz"])
    dt = 1.0 / hz
    po = physics_override or _nominal_physics(manifest)
    physics = VehiclePhysics(M=po.mass_kg, Cd=float(p.get("drag_coefficient", .36)),
                             A=float(p.get("frontal_area_m2", 2.42)), rho=float(p.get("air_density_kg_m3", 1.225)),
                             Croll=po.rolling_resistance, g=float(p.get("gravity_m_per_s2", 9.81)), theta=po.road_slope_deg)
    sensor_cfg = sensor_override or RangeSensorConfig(
        noise_std_m=float(s["nominal_noise_std_m"]), bias_m=float(s["nominal_bias_m"]),
        min_distance_m=float(s["nominal_min_distance_m"]), seed=int(s["nominal_noise_seed"]),
    )
    sensor = RangeOnlySensor(dt, sensor_cfg)
    fmax = float(p["max_longitudinal_force_n"])
    d0, tg = float(c["standstill_distance_m"]), float(c["time_gap_s"])
    k_de, k_vrel = float(gains[1]), float(gains[2])
    xe, ve = 0.0, float(scenario.initial_ego_speed_mps)
    xl, vl = float(scenario.initial_gap_m), float(scenario.initial_lead_speed_mps)
    prev_a = None
    hist = {k: [] for k in ["time", "v_ego", "v_lead", "dist", "dist_sensed", "d_safe", "v_rel_est",
                                  "v_rel_true", "a_cmd", "a_ego", "jerk", "spacing_error", "traction_force_n",
                                  "tractive_power_w", "positive_tractive_power_w"]}
    for i, lead_next in enumerate(np.asarray(scenario.lead_speed_mps, dtype=float)):
        t = i * dt
        dist = xl - xe
        reading = sensor.sense(dist)
        dsafe = d0 + tg * ve
        err = dist - dsafe
        acmd = k_de * (reading.distance - dsafe) + k_vrel * reading.relative_velocity
        v_before = ve
        vnew, anew, force = physics.step_acceleration_command(ve, acmd, dt, Fmax=fmax)
        jerk = 0.0 if prev_a is None else (anew - prev_a) / dt
        power = force * max(v_before, 0.0)
        vals = dict(time=t, v_ego=v_before, v_lead=vl, dist=dist, dist_sensed=reading.distance,
                    d_safe=dsafe, v_rel_est=reading.relative_velocity, v_rel_true=vl-v_before,
                    a_cmd=acmd, a_ego=anew, jerk=jerk, spacing_error=err,
                    traction_force_n=force, tractive_power_w=power, positive_tractive_power_w=max(power,0.0))
        for k, val in vals.items(): hist[k].append(float(val))
        prev_a, ve, vl = anew, vnew, float(lead_next)
        xe += ve * dt; xl += vl * dt
    return hist


from core.research_simulation import simulate_cruise, simulate_following

def _rms(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x*x))) if x.size else np.nan


def _energy_metrics(hist, manifest, physics_override: PhysicsOverride | None = None):
    if "traction_force_n" in hist:
        from core.energy import applied_force_energy
        return applied_force_energy(hist["v_ego"],hist["traction_force_n"],
            1.0/int(manifest["execution_backend"]["sampling_hz"])).positive_wh_per_km
    p = manifest["physics"]
    po = physics_override or _nominal_physics(manifest)
    t = np.asarray(hist["time"], float); v = np.asarray(hist["v_ego"], float); a = np.asarray(hist["a_ego"], float)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0 / int(manifest["execution_backend"]["sampling_hz"])
    e = trajectory_traction_energy(v, a, dt, mass_kg=po.mass_kg,
        rho_kg_m3=float(p.get("air_density_kg_m3",1.225)), drag_coefficient=float(p.get("drag_coefficient",.36)),
        frontal_area_m2=float(p.get("frontal_area_m2",2.42)), rolling_resistance=po.rolling_resistance,
        gravity_mps2=float(p.get("gravity_m_per_s2",9.81)), road_slope_deg=po.road_slope_deg)
    return float(e.positive_wh_per_km)


def following_metrics(hist, manifest, physics_override=None):
    spacing = np.asarray(hist["dist"],float); dsafe=np.asarray(hist["d_safe"],float)
    ve=np.asarray(hist["v_ego"],float); vl=np.asarray(hist["v_lead"],float)
    jerk=np.asarray(hist["jerk"],float); accel=np.asarray(hist["a_ego"],float)
    ttc=compute_ttc_series(spacing,ve,vl); finite=ttc[np.isfinite(ttc)]
    return {
        "spacing_rmse_m": _rms(spacing-dsafe), "speed_rmse_mps": _rms(ve-vl),
        "jerk_rms_mps3": _rms(jerk), "peak_abs_jerk_mps3": float(np.max(np.abs(jerk))),
        "peak_abs_accel_mps2": float(np.max(np.abs(accel))), "min_spacing_m": float(np.min(spacing)),
        "ttc_min_s": float(np.min(finite)) if finite.size else np.inf,
        "ttc_below_threshold_fraction": float(np.mean(ttc < TTC_REPORT_THRESHOLD_S)) if ttc.size else 0.0,
        "ttc_closing_below_threshold_fraction": float(np.mean(finite < TTC_REPORT_THRESHOLD_S)) if finite.size else 0.0,
        "positive_tractive_energy_wh_per_km": _energy_metrics(hist, manifest, physics_override),
    }


def cruise_metrics(hist, manifest, physics_override=None):
    ve=np.asarray(hist["v_ego"],float); vs=np.asarray(hist["v_set"],float); jerk=np.asarray(hist["jerk"],float); accel=np.asarray(hist["a_ego"],float)
    return {
        "cruise_speed_rmse_mps": _rms(ve-vs), "cruise_speed_mae_mps": float(np.mean(np.abs(ve-vs))),
        "jerk_rms_mps3": _rms(jerk), "peak_abs_jerk_mps3": float(np.max(np.abs(jerk))),
        "peak_abs_accel_mps2": float(np.max(np.abs(accel))),
        "positive_tractive_energy_wh_per_km": _energy_metrics(hist, manifest, physics_override),
    }


def detect_disturbances(time, lead_speed) -> list[tuple[int,int]]:
    t=np.asarray(time,float); v=np.asarray(lead_speed,float)
    if len(t)<3: return []
    dt=float(np.median(np.diff(t))); a=np.gradient(v,dt)
    active=np.abs(a)>=TTE_LEAD_ACCEL_THRESHOLD_MPS2
    idx=np.where(active)[0]
    if not len(idx): return []
    groups=[]; start=prev=int(idx[0])
    max_gap=max(1,int(round(TTE_MERGE_GAP_S/dt)))
    for cur in idx[1:]:
        cur=int(cur)
        if cur-prev<=max_gap: prev=cur
        else: groups.append((start,prev)); start=prev=cur
    groups.append((start,prev))
    min_n=max(1,int(round(TTE_MIN_DISTURBANCE_S/dt)))
    return [(a,b) for a,b in groups if b-a+1>=min_n]


def observed_tte_events(hist) -> pd.DataFrame:
    t=np.asarray(hist["time"],float); vl=np.asarray(hist["v_lead"],float); ve=np.asarray(hist["v_ego"],float)
    dist=np.asarray(hist["dist"],float); dsafe=np.asarray(hist["d_safe"],float)
    if len(t)<3: return pd.DataFrame()
    dt=float(np.median(np.diff(t))); hold_n=max(1,int(round(TTE_HOLD_S/dt)))
    events=detect_disturbances(t,vl)
    rows=[]
    for eid,(start,end) in enumerate(events):
        next_start=events[eid+1][0] if eid+1<len(events) else len(t)
        search_start=min(end+1,len(t)-1)
        found=None
        for i in range(search_start, max(search_start, next_start-hold_n+1)):
            j=i+hold_n
            if j>next_start: break
            rel=np.abs(ve[i:j]-vl[i:j])
            spacing=np.abs(dist[i:j]-dsafe[i:j])
            spacing_band=np.maximum(TTE_SPACING_TOL_M, TTE_SPACING_TOL_FRACTION*np.abs(dsafe[i:j]))
            if np.all(rel<=TTE_SPEED_TOL_MPS) and np.all(spacing<=spacing_band):
                found=i; break
        rows.append({
            "event_id": eid, "disturbance_start_s": float(t[start]), "disturbance_end_s": float(t[end]),
            "next_disturbance_start_s": float(t[next_start]) if next_start<len(t) else np.nan,
            "tte_s": float(t[found]-t[end]) if found is not None else np.nan,
            "recovered_before_next_disturbance": found is not None,
        })
    return pd.DataFrame(rows)


def _decimate(n: int, max_points=5000):
    if n<=max_points: return np.arange(n)
    step=int(math.ceil(n/max_points)); idx=np.arange(0,n,step)
    return np.append(idx,n-1) if idx[-1]!=n-1 else idx


def plot_following_comparison(histories: dict[str,dict], scenario: FollowingScenario, path: Path, title: str):
    first=next(iter(histories.values())); t=np.asarray(first["time"],float); idx=_decimate(len(t)); tt=t[idx]
    fig,axes=plt.subplots(4,1,figsize=(12,13),sharex=True)
    # Reference curves shown once.
    axes[0].plot(tt,np.asarray(first["d_safe"])[idx],color="black",ls="--",lw=1.2,label="CTG desired spacing")
    axes[1].plot(tt,np.asarray(first["v_lead"])[idx],color="black",ls="--",lw=1.2,label="Lead")
    for algo,h in histories.items():
        st=STYLE[algo]
        axes[0].plot(tt,np.asarray(h["dist"])[idx],color=st["color"],ls=st["ls"],lw=1.1,label=st["label"])
        axes[1].plot(tt,np.asarray(h["v_ego"])[idx],color=st["color"],ls=st["ls"],lw=1.1,label=st["label"])
        axes[2].plot(tt,np.asarray(h["a_ego"])[idx],color=st["color"],ls=st["ls"],lw=1.0,label=st["label"])
        axes[3].plot(tt,np.asarray(h["jerk"])[idx],color=st["color"],ls=st["ls"],lw=.9,label=st["label"])
    axes[0].set_ylabel("Spacing (m)"); axes[1].set_ylabel("Speed (m/s)"); axes[2].set_ylabel("Accel. (m/s²)"); axes[3].set_ylabel("Jerk (m/s³)"); axes[3].set_xlabel("Time (s)")
    fig.suptitle(title)
    for ax in axes: ax.grid(alpha=.22); ax.legend(ncol=4,fontsize=7)
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(path,dpi=280); plt.close(fig)


def plot_cruise_comparison(histories: dict[str,dict], scenario: CruiseScenario, path: Path, title: str):
    first=next(iter(histories.values())); t=np.asarray(first["time"],float); idx=_decimate(len(t)); tt=t[idx]
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    axes[0].plot(tt,np.asarray(first["v_set"])[idx],color="black",ls="--",lw=1.2,label="Set speed")
    for algo,h in histories.items():
        st=STYLE[algo]
        axes[0].plot(tt,np.asarray(h["v_ego"])[idx],color=st["color"],ls=st["ls"],lw=1.1,label=st["label"])
        axes[1].plot(tt,np.asarray(h["a_ego"])[idx],color=st["color"],ls=st["ls"],lw=1.0,label=st["label"])
        axes[2].plot(tt,np.asarray(h["jerk"])[idx],color=st["color"],ls=st["ls"],lw=.9,label=st["label"])
    for ev in scenario.event_times_s[1:]:
        for ax in axes: ax.axvline(float(ev),color="grey",ls=":",lw=.6,alpha=.35)
    axes[0].set_ylabel("Speed (m/s)"); axes[1].set_ylabel("Accel. (m/s²)"); axes[2].set_ylabel("Jerk (m/s³)"); axes[2].set_xlabel("Time (s)")
    fig.suptitle(title)
    for ax in axes: ax.grid(alpha=.22); ax.legend(ncol=4,fontsize=7)
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(path,dpi=280); plt.close(fig)


def motion_safety_outputs(reps: pd.DataFrame, manifest: dict, suite, out_motion: Path, out_safety: Path):
    hz=int(manifest["execution_backend"]["sampling_hz"])
    periodic=build_periodic_cruise(); sinus=build_sinusoidal_following(hz)
    validations={s.scenario_id:s for s in load_validation_scenarios(manifest["cost_framework"]["scenario_manifest_path"], require_frozen=True)}
    calibration={s.scenario_id:s for s in suite.following}
    scenarios={**calibration, **validations, sinus.scenario_id:sinus}

    rep_gains={r.algorithm:np.asarray([r.k_ve,r.k_de,r.k_vrel],float) for _,r in reps.iterrows()}
    safety_rows=[]; tte_rows=[]

    # Periodic cruise motion.
    cruise_hist={a:simulate_cruise(g,periodic,manifest) for a,g in rep_gains.items()}
    plot_cruise_comparison(cruise_hist,periodic,out_motion/"cruise_control_periodic.png","Periodic cruise-control motion: 10 set-speed transitions")

    # Following sources: both calibration OpenACC windows, held-out Casale,
    # standardized WLTC, plus synthetic sinusoidal regression case.
    for sid,scenario in scenarios.items():
        hist={a:simulate_following(g,scenario,manifest) for a,g in rep_gains.items()}
        if sid.startswith("C1_") or sid.startswith("C2_") or sid.startswith("V1_"):
            family="OpenACC"
        elif "WLTC" in sid:
            family="WLTC"
        elif "SINUSOIDAL" in sid:
            family="Synthetic sinusoidal"
        else:
            family="Validation"
        plot_following_comparison(hist,scenario,out_motion/f"{sid.lower()}_motion.png",f"{family} trajectory motion: {sid}")

        # TTC comparison.
        fig,ax=plt.subplots(figsize=(11,5.2))
        for algo,h in hist.items():
            t=np.asarray(h["time"],float); ttc=compute_ttc_series(h["dist"],h["v_ego"],h["v_lead"])
            finite=ttc[np.isfinite(ttc)]
            st=STYLE[algo]
            ax.plot(t,np.where(np.isfinite(ttc),np.minimum(ttc,TTC_PLOT_CAP_S),TTC_PLOT_CAP_S),
                    color=st["color"],ls=st["ls"],lw=1.0,label=st["label"])
            fm=following_metrics(h,manifest)
            safety_rows.append({"scenario_id":sid,"algorithm":algo,"algorithm_label":_label(algo),**fm})
            # TTE is only reported for the frozen Casale 50--120 s window.
            # That window was selected specifically for local equilibrium / finite
            # deceleration-acceleration disturbances. WLTC and sinusoidal motion are
            # continuous-excitation cases, where a TTE number would look precise
            # while answering the wrong question.
            if sid == "V1_CASALE":
                ev=observed_tte_events(h)
                if not ev.empty:
                    ev.insert(0,"algorithm_label",_label(algo)); ev.insert(0,"algorithm",algo); ev.insert(0,"scenario_id",sid)
                    tte_rows.extend(ev.to_dict("records"))
        ax.axhline(TTC_REPORT_THRESHOLD_S,color="red",ls="--",lw=1.0,label=f"Reporting threshold {TTC_REPORT_THRESHOLD_S:g}s")
        ax.set_ylim(0,TTC_PLOT_CAP_S*1.03); ax.set_xlabel("Time (s)"); ax.set_ylabel(f"TTC (s), capped at {TTC_PLOT_CAP_S:g}s")
        ax.set_title(f"TTC diagnostic: {sid}"); ax.grid(alpha=.25); ax.legend(ncol=4,fontsize=7)
        fig.tight_layout(); fig.savefig(out_safety/f"ttc_{sid.lower()}.png",dpi=280); plt.close(fig)

    safety=pd.DataFrame(safety_rows); safety.to_csv(out_safety/"ttc_and_following_summary.csv",index=False)
    pd.DataFrame([
        {"scenario_id": sid, "tte_applicable": sid == "V1_CASALE",
         "reason": ("Frozen OpenACC Casale 50-120 s local-equilibrium disturbance window"
                    if sid == "V1_CASALE" else
                    "TTE intentionally not reported for continuous/non-preregistered equilibrium disturbances")}
        for sid in scenarios
    ]).to_csv(out_safety/"tte_applicability.csv", index=False)
    tte=pd.DataFrame(tte_rows); tte.to_csv(out_safety/"observed_tte_events.csv",index=False)
    if not tte.empty:
        summary=tte.groupby(["scenario_id","algorithm","algorithm_label"],as_index=False).agg(
            disturbances=("event_id","count"), recovered=("recovered_before_next_disturbance","sum"),
            median_tte_s=("tte_s","median"), mean_tte_s=("tte_s","mean"), max_tte_s=("tte_s","max"),
        )
        summary["recovery_fraction"] = summary.recovered / summary.disturbances
        summary.to_csv(out_safety/"observed_tte_summary.csv",index=False)
    return periodic, sinus, validations


def time_response_outputs(reps, manifest, periodic: CruiseScenario, out_dir: Path):
    rows=[]
    for _,r in reps.iterrows():
        gains=np.asarray([r.k_ve,r.k_de,r.k_vrel],float); h=simulate_cruise(gains,periodic,manifest)
        t=np.asarray(h["time"],float); v=np.asarray(h["v_ego"],float)
        events=np.asarray(periodic.event_times_s,float); refs=np.asarray(periodic.set_speeds_mps,float)
        for i in range(1,len(events)):
            start=events[i]; end=events[i+1] if i+1<len(events) else periodic.duration_s
            mask=(t>=start)&(t<end)
            if mask.sum()<3: continue
            ts=t[mask]-start; ys=v[mask]
            m=step_response_metrics(ts,ys,float(refs[i]),y0=float(ys[0]),steady_window=min(2.0,float(ts[-1])),tol=.02)
            rows.append({"algorithm":r.algorithm,"algorithm_label":r.algorithm_label,"transition_index":i,
                         "start_s":start,"end_s":end,"from_set_mps":float(refs[i-1]),"to_set_mps":float(refs[i]),
                         **m})
    df=pd.DataFrame(rows); df.to_csv(out_dir/"cruise_transition_metrics.csv",index=False)
    numeric=["rise_time","settling_time","overshoot_pct","steady_state_error","peak_time"]
    agg=[]
    for algo,g in df.groupby("algorithm",sort=False):
        rec={"algorithm":algo,"algorithm_label":_label(algo),"transitions":len(g)}
        for m in numeric:
            vals=pd.to_numeric(g[m],errors="coerce")
            rec[f"{m}_median"]=float(vals.median()) if vals.notna().any() else np.nan
            rec[f"{m}_iqr"]=float(vals.quantile(.75)-vals.quantile(.25)) if vals.notna().any() else np.nan
        agg.append(rec)
    pd.DataFrame(agg).to_csv(out_dir/"cruise_time_response_summary.csv",index=False)

    for metric,title,ylabel in [
        ("rise_time","Cruise rise time by transition","Rise time (s)"),
        ("settling_time","Cruise settling time by transition","Settling time (s)"),
        ("overshoot_pct","Cruise overshoot by transition","Overshoot (%)"),
        ("steady_state_error","Cruise steady-state error by transition","Steady-state error (m/s)"),
    ]:
        fig,ax=plt.subplots(figsize=(11,5.2))
        for algo in reps.algorithm:
            g=df[df.algorithm==algo]; st=STYLE[algo]
            ax.plot(g.transition_index,g[metric],label=st["label"],color=st["color"],marker=st["marker"],ls=st["ls"],lw=1.1,ms=4)
        ax.set_xlabel("Set-speed transition index"); ax.set_ylabel(ylabel); ax.set_title(title); ax.grid(alpha=.25); ax.legend(ncol=3,fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir/f"{metric}_by_transition.png",dpi=280); plt.close(fig)


def robustness_outputs(reps, manifest, periodic, casale: FollowingScenario, out_dir: Path):
    p=manifest["physics"]; base_mass=float(p["nominal_mass_kg"]); base_crr=float(p["rolling_resistance"])
    rows=[]
    for sid,desc,slope,mass_factor,crr_factor in ROBUSTNESS_SCENARIOS:
        override=PhysicsOverride(base_mass*mass_factor,base_crr*crr_factor,slope)
        for _,r in reps.iterrows():
            gains=np.asarray([r.k_ve,r.k_de,r.k_vrel],float)
            hc=simulate_cruise(gains,periodic,manifest,override); hf=simulate_following(gains,casale,manifest,override)
            cm=cruise_metrics(hc,manifest,override); fm=following_metrics(hf,manifest,override)
            tr=observed_tte_events(hf)
            # final transition response as one interpretable time-domain stress metric
            t=np.asarray(hc["time"],float); v=np.asarray(hc["v_ego"],float); start=float(periodic.event_times_s[-1]); mask=t>=start
            rm=step_response_metrics(t[mask]-start,v[mask],float(periodic.set_speeds_mps[-1]),y0=float(v[mask][0]),tol=.02) if mask.sum()>3 else {}
            rows.append({
                "scenario_id":sid,"description":desc,"slope_deg":slope,"mass_factor":mass_factor,"crr_factor":crr_factor,
                "algorithm":r.algorithm,"algorithm_label":r.algorithm_label,
                **{f"cruise_{k}":v for k,v in cm.items()},
                **{f"following_{k}":v for k,v in fm.items()},
                "casale_tte_median_s":float(tr.tte_s.median()) if (not tr.empty and tr.tte_s.notna().any()) else np.nan,
                "casale_tte_recovery_fraction":float(tr.recovered_before_next_disturbance.mean()) if not tr.empty else np.nan,
                "final_transition_settling_time_s":rm.get("settling_time",np.nan),
                "final_transition_overshoot_pct":rm.get("overshoot_pct",np.nan),
            })
    df=pd.DataFrame(rows); df.to_csv(out_dir/"physical_robustness_metrics.csv",index=False)

    # Relative-to-nominal tables preserve metric units while making stress sensitivity visible.
    metrics=["cruise_cruise_speed_rmse_mps","following_spacing_rmse_m","following_ttc_min_s",
             "following_positive_tractive_energy_wh_per_km","casale_tte_median_s"]
    rel_rows=[]
    for algo,g in df.groupby("algorithm"):
        nominal=g[g.scenario_id=="S0"].iloc[0]
        for _,r in g.iterrows():
            rec={"algorithm":algo,"algorithm_label":_label(algo),"scenario_id":r.scenario_id,"description":r.description}
            for metric in metrics:
                base=float(nominal[metric]); val=float(r[metric]) if pd.notna(r[metric]) else np.nan
                rec[f"{metric}_ratio_to_nominal"] = val/base if np.isfinite(val) and np.isfinite(base) and abs(base)>1e-15 else np.nan
            rel_rows.append(rec)
    pd.DataFrame(rel_rows).to_csv(out_dir/"physical_robustness_relative_to_nominal.csv",index=False)

    # One plot per key physical/safety metric.
    plot_metrics=[
        ("cruise_cruise_speed_rmse_mps","Cruise speed RMSE (m/s)"),
        ("following_spacing_rmse_m","Casale spacing RMSE (m)"),
        ("following_ttc_min_s","Casale minimum TTC (s)"),
        ("following_positive_tractive_energy_wh_per_km","Casale positive tractive energy (Wh/km)"),
        ("casale_tte_median_s","Casale observed TTE median (s)"),
    ]
    for metric,ylabel in plot_metrics:
        fig,ax=plt.subplots(figsize=(12,5.8))
        x=np.arange(len(ROBUSTNESS_SCENARIOS))
        for algo in reps.algorithm:
            g=df[df.algorithm==algo].set_index("scenario_id").reindex([s[0] for s in ROBUSTNESS_SCENARIOS])
            st=STYLE[algo]
            ax.plot(x,g[metric],label=st["label"],color=st["color"],marker=st["marker"],ls=st["ls"],lw=1.1,ms=4)
        ax.set_xticks(x,[s[0] for s in ROBUSTNESS_SCENARIOS]); ax.set_xlabel("Physical robustness scenario"); ax.set_ylabel(ylabel)
        ax.set_title(ylabel+" across 12 inherited physical stress scenarios"); ax.grid(alpha=.25); ax.legend(ncol=3,fontsize=8)
        fig.tight_layout(); fig.savefig(out_dir/f"robustness_{metric}.png",dpi=280); plt.close(fig)


def audit_outputs(runs, components, reps, manifest, suite, profile, out_dir: Path):
    issues=[]
    opt=manifest["optimization"]; expected=int(opt["independent_runs"]); iterations=int(opt["max_iterations"])
    for algo in opt["algorithms"]:
        g=runs[runs.algorithm==algo]
        if len(g)!=expected: issues.append(("coverage",algo,f"{len(g)}/{expected} runs"))
        for _,r in g.iterrows():
            if r.curve is None or len(np.asarray(r.curve).reshape(-1))!=iterations:
                issues.append(("curve_length",algo,f"run {int(r.run_id)} has {0 if r.curve is None else len(np.asarray(r.curve).reshape(-1))}/{iterations}"))
            if not np.isfinite(r.function_evaluations) or r.function_evaluations<=0:
                issues.append(("function_evaluations",algo,f"run {int(r.run_id)} invalid FE"))
    max_recon=float(components.cost_reconstruction_abs_error.max()) if len(components) else np.nan
    if np.isfinite(max_recon) and max_recon>1e-8: issues.append(("cost_reconstruction","all",f"max abs error={max_recon:g}"))
    pd.DataFrame(issues,columns=["check","scope","detail"]).to_csv(out_dir/"consistency_issues.csv",index=False)
    metadata={
        "benchmark":manifest["benchmark"],"framework_version":manifest.get("framework_version"),
        "algorithms":list(opt["algorithms"]),"independent_runs":expected,"iterations":iterations,
        "population_size":int(opt["population_size"]),"representative_policy":"observed run nearest algorithm median final fitness",
        "scenario_manifest":str(suite.manifest_path),"scenario_manifest_sha256":suite.manifest_sha256,
        "cost_profile":str(profile.path),"cost_profile_status":profile.status,
        "performance_names":list(PERFORMANCE_NAMES),"weights":{n:float(w) for n,w in zip(PERFORMANCE_NAMES,profile.weights)},
        "ttc_reporting_threshold_s":TTC_REPORT_THRESHOLD_S,
        "observed_tte_definition":{
            "lead_acceleration_threshold_mps2":TTE_LEAD_ACCEL_THRESHOLD_MPS2,
            "minimum_disturbance_duration_s":TTE_MIN_DISTURBANCE_S,"merge_gap_s":TTE_MERGE_GAP_S,
            "relative_speed_tolerance_mps":TTE_SPEED_TOL_MPS,"spacing_absolute_tolerance_m":TTE_SPACING_TOL_M,
            "spacing_relative_tolerance_fraction":TTE_SPACING_TOL_FRACTION,"hold_window_s":TTE_HOLD_S,
            "note":"Operational observed recovery metric for reporting; not Zhuo-Zhu analytical TTE.",
        },
        "robustness_scenarios":[{"id":x[0],"description":x[1],"slope_deg":x[2],"mass_factor":x[3],"crr_factor":x[4]} for x in ROBUSTNESS_SCENARIOS],
        "sensor_robustness":"not executed: active research manifest defines nominal sensing only and no preregistered perturbation grid",
        "max_cost_reconstruction_abs_error":max_recon,
        "issue_count":len(issues),
    }
    (out_dir/"analysis_metadata.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")


def write_index(base: Path):
    text="""# Analysis output index\n\nGenerated by the internal analysis module from the frozen Framework V2.2-FJE campaign.\n\n- `01_objective/`: all-run scalar cost, FJE physical components, representative controllers, gain-boundary audit.\n- `02_convergence/`: median/IQR convergence across 30 runs per algorithm.\n- `03_statistics/`: Q-Q, Shapiro-Wilk, independent Kruskal-Wallis and Mann-Whitney comparisons with corrections.\n- `04_motion/`: comparative trajectory motion for OpenACC calibration/validation, WLTC, periodic cruise, sinusoidal following.\n- `05_safety/`: TTC time series and observed/operational TTE event tables.\n- `06_time_response/`: rise, settling, overshoot, steady-state error for all ten cruise set-speed transitions.\n- `07_robustness/`: inherited 12-case physical-parameter stress test.\n- `08_audit/`: experiment consistency checks and analysis contract metadata.\n\nTTC/TTE and robustness are validation evidence only; they do not modify the frozen optimization fitness.\n"""
    (base/"README.md").write_text(text,encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir",type=Path,default=DEFAULT_OUT)
    ap.add_argument("--clean",action="store_true",help="replace existing analysis output directory")
    args=ap.parse_args(argv)
    out=args.out_dir if args.out_dir.is_absolute() else ROOT/args.out_dir
    if args.clean and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True); dirs=_ensure_dirs(out)

    manifest=load_manifest()
    if manifest["benchmark"]!="acc_cost_sahib_fje_v2":
        raise RuntimeError("core.analysis targets the active FJE benchmark")
    suite=load_scenario_suite(manifest["cost_framework"]["scenario_manifest_path"],development=False,require_frozen=True)
    profile=load_cost_profile(require_frozen=True)

    print("[1/8] Loading 180 optimization checkpoints...")
    runs=load_run_frame(manifest)
    reps=select_observed_median_representatives(runs)
    print("[2/8] Re-evaluating all-run FJE physical components...")
    components=evaluate_all_components(runs,suite,profile)
    objective_outputs(runs,components,reps,manifest,profile,dirs["objective"])
    print("[3/8] Convergence and statistical audit...")
    convergence_outputs(runs,manifest,dirs["convergence"])
    statistical_outputs(runs,components,manifest,dirs["statistics"])
    print("[4/8] Motion trajectories + TTC/TTE validation...")
    periodic,sinus,validations=motion_safety_outputs(reps,manifest,suite,dirs["motion"],dirs["safety"])
    print("[5/8] Cruise time-response metrics...")
    time_response_outputs(reps,manifest,periodic,dirs["time_response"])
    print("[6/8] Physical robustness stress test...")
    casale=validations.get("V1_CASALE")
    if casale is None: raise RuntimeError("V1_CASALE held-out validation scenario is required for robustness reporting")
    robustness_outputs(reps,manifest,periodic,casale,dirs["robustness"])
    print("[7/8] Reproducibility / consistency audit...")
    audit_outputs(runs,components,reps,manifest,suite,profile,dirs["audit"])
    write_index(out)
    print(f"[8/8] Done. Analysis outputs: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
