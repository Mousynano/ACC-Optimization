#!/usr/bin/env python3
"""Plot held-out and synthetic ACC validation motion for one checkpoint.

The script reads ``best_solution = [k_ve, k_de, k_vrel]`` from one
``checkpoint_jobs/*.pkl`` file and evaluates that same gain vector on:

1. every held-out following scenario declared in the frozen Framework V2
   scenario manifest; and
2. two deterministic synthetic diagnostics:
   * ``SYN_FOLLOW_SINUSOIDAL``: 60 s sinusoidal lead acceleration;
   * ``SYN_CRUISE_VSET_10X``: cruise-control stress test containing exactly
     ten set-speed changes (eleven plateaus).

The synthetic diagnostics are reporting-only.  They do not participate in
Pareto calibration, weight construction, or optimizer fitness.

Examples
--------
Plot held-out + both synthetic diagnostics::

    python plot_validation_motion.py checkpoint_jobs/<checkpoint>.pkl

Plot only the two synthetic diagnostics (does not require a frozen scenario
manifest)::

    python plot_validation_motion.py checkpoint_jobs/<checkpoint>.pkl \
        --synthetic-only

Plot only the frozen held-out scenarios::

    python plot_validation_motion.py checkpoint_jobs/<checkpoint>.pkl \
        --manifest-only

Plot selected cases::

    python plot_validation_motion.py checkpoint_jobs/<checkpoint>.pkl \
        --scenario V1_CASALE \
        --scenario SYN_FOLLOW_SINUSOIDAL \
        --scenario SYN_CRUISE_VSET_10X

Outputs
-------
* one PNG per validation case;
* one multi-page PDF with all selected figures;
* one CSV containing full-resolution numerical diagnostics;
* one JSON containing checkpoint metadata, gains, scenario definitions, and
  metrics.

Long traces may be decimated only for rendering.  All numerical diagnostics
use the full 60 Hz simulation output.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd

from core.config import ROOT_DIR, load_manifest
from core.energy import trajectory_traction_energy

_ACTIVE_MANIFEST = load_manifest()
if _ACTIVE_MANIFEST["benchmark"] == "acc_cost_sahib_fje_v2":
    from benchmarks.acc_cost_fje_v2 import simulate_cruise_history, simulate_following_history
else:
    from benchmarks.acc_cost_v2 import simulate_cruise_history, simulate_following_history
from core.scenarios_v2 import (
    CruiseScenario,
    FollowingScenario,
    load_validation_scenarios,
    sha256_file,
)


GAIN_NAMES = ("k_ve", "k_de", "k_vrel")

# ---------------------------------------------------------------------------
# Reporting-only synthetic validation definitions.
# ---------------------------------------------------------------------------
SYNTHETIC_FOLLOW_ID = "SYN_FOLLOW_SINUSOIDAL"
SYNTHETIC_CRUISE_ID = "SYN_CRUISE_VSET_10X"

SYN_FOLLOW_DURATION_S = 60.0
SYN_FOLLOW_INITIAL_SPEED_MPS = 22.0
SYN_FOLLOW_INITIAL_GAP_M = 60.0
SYN_FOLLOW_ACCEL_AMPLITUDE_MPS2 = 1.0
SYN_FOLLOW_PHASE_RATE_DEG_PER_S = 30.0

# Eleven plateaus -> exactly ten changes after t=0.  Speeds stay inside the
# 20--30 m/s envelope already used by the V2 cruise calibration design.
SYN_CRUISE_PLATEAU_S = 20.0
SYN_CRUISE_INITIAL_EGO_SPEED_MPS = 22.0
SYN_CRUISE_SET_SPEEDS_MPS = np.asarray(
    [25.0, 20.0, 30.0, 22.0, 28.0, 24.0, 29.0, 21.0, 27.0, 23.0, 25.0],
    dtype=float,
)


@dataclass(frozen=True)
class ValidationCase:
    scenario_id: str
    scenario_type: Literal["following", "cruise"]
    source_kind: Literal["held_out_manifest", "synthetic_diagnostic"]
    scenario: FollowingScenario | CruiseScenario
    description: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load best_solution from one checkpoint and plot held-out plus "
            "synthetic ACC validation motion."
        )
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to one checkpoint_jobs/*.pkl file.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Scenario manifest path. Default: cost_framework.scenario_manifest_path "
            "from the active experiment manifest."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--synthetic-only",
        action="store_true",
        help=(
            "Run only SYN_FOLLOW_SINUSOIDAL and SYN_CRUISE_VSET_10X. "
            "This mode does not require a frozen external scenario manifest."
        ),
    )
    mode.add_argument(
        "--manifest-only",
        action="store_true",
        help="Run only held-out scenarios declared in the frozen manifest.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help=(
            "Plot only this scenario ID. Repeat to select multiple cases. "
            "Default: all cases enabled by the selected mode."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("validation_motion_plots"),
        help="Output directory (default: validation_motion_plots).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=220,
        help="PNG resolution (default: 220).",
    )
    parser.add_argument(
        "--max-plot-points",
        type=int,
        default=12000,
        help=(
            "Maximum displayed samples per trace. Metrics always use full-resolution "
            "data (default: 12000)."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List enabled validation cases and exit without simulation.",
    )
    return parser.parse_args()


def _resolve_from_root(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    cwd_candidate = Path.cwd() / path
    if cwd_candidate.exists():
        return cwd_candidate.resolve()
    return (ROOT_DIR / path).resolve()


def load_checkpoint(path: Path) -> tuple[dict[str, Any], np.ndarray]:
    path = _resolve_from_root(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    with path.open("rb") as fh:
        payload = pickle.load(fh)

    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a dict, got {type(payload).__name__}")
    if "best_solution" not in payload:
        raise KeyError("Checkpoint does not contain 'best_solution'.")

    gains = np.asarray(payload["best_solution"], dtype=float).reshape(-1)
    if gains.shape != (3,):
        raise ValueError(
            "Framework V2 checkpoint best_solution must contain exactly three gains "
            "[k_ve, k_de, k_vrel]; got shape " + str(gains.shape)
        )
    if not np.isfinite(gains).all():
        raise ValueError(f"Checkpoint best_solution contains non-finite values: {gains}")

    return payload, gains


def resolve_manifest_path(cli_path: Path | None) -> Path:
    if cli_path is not None:
        return _resolve_from_root(cli_path)
    exp = load_manifest()
    rel = Path(exp["cost_framework"]["scenario_manifest_path"])
    return _resolve_from_root(rel)


def _sampling_hz() -> int:
    exp = load_manifest()
    return int(exp["execution_backend"]["sampling_hz"])


def _build_sinusoidal_following(sampling_hz: int) -> FollowingScenario:
    dt = 1.0 / float(sampling_hz)
    steps = int(round(SYN_FOLLOW_DURATION_S * sampling_hz))
    lead_speed = np.empty(steps, dtype=float)

    v = float(SYN_FOLLOW_INITIAL_SPEED_MPS)
    phase_deg = 0.0
    for i in range(steps):
        phase_deg = (
            phase_deg + SYN_FOLLOW_PHASE_RATE_DEG_PER_S * dt
        ) % 360.0
        a_lead = SYN_FOLLOW_ACCEL_AMPLITUDE_MPS2 * math.sin(math.radians(phase_deg))
        v = max(v + a_lead * dt, 0.0)
        lead_speed[i] = v

    return FollowingScenario(
        scenario_id=SYNTHETIC_FOLLOW_ID,
        duration_s=SYN_FOLLOW_DURATION_S,
        initial_ego_speed_mps=SYN_FOLLOW_INITIAL_SPEED_MPS,
        initial_lead_speed_mps=SYN_FOLLOW_INITIAL_SPEED_MPS,
        initial_gap_m=SYN_FOLLOW_INITIAL_GAP_M,
        lead_speed_mps=lead_speed,
        weight=1.0,
        provenance=(
            "reporting-only synthetic validation: sinusoidal lead acceleration; "
            f"A={SYN_FOLLOW_ACCEL_AMPLITUDE_MPS2:g} m/s^2, "
            f"phase_rate={SYN_FOLLOW_PHASE_RATE_DEG_PER_S:g} deg/s"
        ),
    )


def _build_ten_change_cruise() -> CruiseScenario:
    speeds = SYN_CRUISE_SET_SPEEDS_MPS.copy()
    event_times = np.arange(speeds.size, dtype=float) * SYN_CRUISE_PLATEAU_S
    duration_s = float(speeds.size * SYN_CRUISE_PLATEAU_S)

    # There must be exactly ten transitions after the initial set point.
    if speeds.size - 1 != 10:
        raise AssertionError("SYN_CRUISE_VSET_10X must contain exactly 10 set-speed changes")

    return CruiseScenario(
        scenario_id=SYNTHETIC_CRUISE_ID,
        duration_s=duration_s,
        initial_ego_speed_mps=SYN_CRUISE_INITIAL_EGO_SPEED_MPS,
        event_times_s=event_times,
        set_speeds_mps=speeds,
        weight=1.0,
        provenance=(
            "reporting-only synthetic validation: 10 set-speed changes, "
            f"{SYN_CRUISE_PLATEAU_S:g} s per plateau"
        ),
    )


def synthetic_validation_cases() -> list[ValidationCase]:
    hz = _sampling_hz()
    follow = _build_sinusoidal_following(hz)
    cruise = _build_ten_change_cruise()
    return [
        ValidationCase(
            scenario_id=follow.scenario_id,
            scenario_type="following",
            source_kind="synthetic_diagnostic",
            scenario=follow,
            description=(
                "Sinusoidal car-following validation, matching the legacy lead-motion "
                "stress pattern (60 s, 1 m/s² acceleration amplitude, 30 deg/s phase rate)."
            ),
        ),
        ValidationCase(
            scenario_id=cruise.scenario_id,
            scenario_type="cruise",
            source_kind="synthetic_diagnostic",
            scenario=cruise,
            description=(
                "Cruise-control validation with exactly ten v_set transitions across "
                "eleven 20 s plateaus within 20-30 m/s."
            ),
        ),
    ]


def manifest_validation_cases(manifest_path: Path) -> list[ValidationCase]:
    scenarios = load_validation_scenarios(manifest_path, require_frozen=True)
    return [
        ValidationCase(
            scenario_id=s.scenario_id,
            scenario_type="following",
            source_kind="held_out_manifest",
            scenario=s,
            description=f"Frozen held-out/regression following trajectory: {s.provenance}",
        )
        for s in scenarios
    ]


def collect_cases(args: argparse.Namespace, manifest_path: Path) -> list[ValidationCase]:
    cases: list[ValidationCase] = []
    if not args.synthetic_only:
        cases.extend(manifest_validation_cases(manifest_path))
    if not args.manifest_only:
        cases.extend(synthetic_validation_cases())

    # Guard against accidental duplicate IDs between manifest and synthetic cases.
    ids = [c.scenario_id for c in cases]
    duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
    if duplicates:
        raise RuntimeError("Duplicate validation scenario IDs: " + ", ".join(duplicates))

    if args.scenario:
        available = {c.scenario_id: c for c in cases}
        missing = [sid for sid in args.scenario if sid not in available]
        if missing:
            raise KeyError(
                "Unknown validation scenario(s): "
                + ", ".join(missing)
                + ". Available: "
                + ", ".join(available)
            )
        cases = [available[sid] for sid in args.scenario]

    if not cases:
        raise RuntimeError("No validation cases selected")
    return cases


def _decimation_indices(n: int, max_points: int) -> np.ndarray:
    if max_points <= 0 or n <= max_points:
        return np.arange(n, dtype=int)
    stride = int(math.ceil(n / max_points))
    idx = np.arange(0, n, stride, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(values))))


def _energy_metrics(history: dict[str, list[float]]) -> dict[str, float]:
    t = np.asarray(history["time"], dtype=float)
    v = np.asarray(history["v_ego"], dtype=float)
    a = np.asarray(history["a_ego"], dtype=float)
    if t.size > 1:
        dt = float(np.median(np.diff(t)))
    else:
        dt = 1.0 / float(_ACTIVE_MANIFEST["execution_backend"]["sampling_hz"])
    ph = _ACTIVE_MANIFEST["physics"]
    e = trajectory_traction_energy(
        v, a, dt,
        mass_kg=float(ph["nominal_mass_kg"]),
        rho_kg_m3=float(ph.get("air_density_kg_m3", 1.225)),
        drag_coefficient=float(ph.get("drag_coefficient", 0.36)),
        frontal_area_m2=float(ph.get("frontal_area_m2", 2.42)),
        rolling_resistance=float(ph["rolling_resistance"]),
        gravity_mps2=float(ph.get("gravity_m_per_s2", 9.81)),
        road_slope_deg=float(ph["road_slope_deg"]),
    )
    return {
        "positive_energy_wh_per_km": float(e.positive_wh_per_km),
        "braking_energy_wh_per_km": float(e.braking_wh_per_km),
        "traction_brake_reversals": float(e.traction_brake_reversals),
    }


def compute_following_metrics(history: dict[str, list[float]]) -> dict[str, float]:
    t = np.asarray(history["time"], dtype=float)
    v_ego = np.asarray(history["v_ego"], dtype=float)
    v_lead = np.asarray(history["v_lead"], dtype=float)
    spacing = np.asarray(history["dist"], dtype=float)
    d_safe = np.asarray(history["d_safe"], dtype=float)
    accel = np.asarray(history["a_ego"], dtype=float)
    jerk = np.asarray(history["jerk"], dtype=float)

    spacing_error = spacing - d_safe
    speed_error = v_ego - v_lead

    abs_jerk = np.abs(jerk)
    metrics = {
        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "spacing_rmse_m": _rms(spacing_error),
        "speed_rmse_mps": _rms(speed_error),
        "jerk_rms_mps3": _rms(jerk),
        "jerk_p95_mps3": float(np.percentile(abs_jerk, 95)),
        "jerk_p99_mps3": float(np.percentile(abs_jerk, 99)),
        "peak_abs_jerk_mps3": float(np.max(abs_jerk)),
        "peak_abs_accel_mps2": float(np.max(np.abs(accel))),
        "min_spacing_m": float(np.min(spacing)),
        "mean_spacing_m": float(np.mean(spacing)),
        "min_ego_speed_mps": float(np.min(v_ego)),
        "max_ego_speed_mps": float(np.max(v_ego)),
        "min_lead_speed_mps": float(np.min(v_lead)),
        "max_lead_speed_mps": float(np.max(v_lead)),
    }
    metrics.update(_energy_metrics(history))
    return metrics


def compute_cruise_metrics(
    history: dict[str, list[float]], scenario: CruiseScenario
) -> dict[str, float]:
    t = np.asarray(history["time"], dtype=float)
    v_ego = np.asarray(history["v_ego"], dtype=float)
    v_set = np.asarray(history["v_set"], dtype=float)
    accel = np.asarray(history["a_ego"], dtype=float)
    jerk = np.asarray(history["jerk"], dtype=float)
    speed_error = v_ego - v_set

    abs_jerk = np.abs(jerk)
    metrics = {
        "duration_s": float(t[-1] - t[0]) if t.size > 1 else 0.0,
        "vset_change_count": float(len(scenario.event_times_s) - 1),
        "set_speed_rmse_mps": _rms(speed_error),
        "set_speed_mae_mps": float(np.mean(np.abs(speed_error))),
        "final_abs_speed_error_mps": float(abs(speed_error[-1])),
        "jerk_rms_mps3": _rms(jerk),
        "jerk_p95_mps3": float(np.percentile(abs_jerk, 95)),
        "jerk_p99_mps3": float(np.percentile(abs_jerk, 99)),
        "peak_abs_jerk_mps3": float(np.max(abs_jerk)),
        "peak_abs_accel_mps2": float(np.max(np.abs(accel))),
        "min_ego_speed_mps": float(np.min(v_ego)),
        "max_ego_speed_mps": float(np.max(v_ego)),
        "min_set_speed_mps": float(np.min(v_set)),
        "max_set_speed_mps": float(np.max(v_set)),
    }
    metrics.update(_energy_metrics(history))
    return metrics


def plot_following_motion(
    *,
    scenario_id: str,
    history: dict[str, list[float]],
    gains: np.ndarray,
    checkpoint_name: str,
    max_plot_points: int,
):
    t = np.asarray(history["time"], dtype=float)
    v_ego = np.asarray(history["v_ego"], dtype=float)
    v_lead = np.asarray(history["v_lead"], dtype=float)
    spacing = np.asarray(history["dist"], dtype=float)
    d_safe = np.asarray(history["d_safe"], dtype=float)
    a_ego = np.asarray(history["a_ego"], dtype=float)
    jerk = np.asarray(history["jerk"], dtype=float)

    if t.size < 2:
        raise RuntimeError(f"{scenario_id}: simulation returned fewer than two samples")

    dt = float(np.median(np.diff(t)))
    a_lead = np.gradient(v_lead, dt)
    idx = _decimation_indices(t.size, max_plot_points)
    tt = t[idx]

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    axes[0].plot(tt, spacing[idx], label="Actual spacing")
    axes[0].plot(tt, d_safe[idx], linestyle="--", label="CTG desired spacing")
    axes[0].set_ylabel("Spacing (m)")

    axes[1].plot(tt, v_lead[idx], linestyle="--", label="Lead speed")
    axes[1].plot(tt, v_ego[idx], label="Ego speed")
    axes[1].set_ylabel("Speed (m/s)")

    axes[2].plot(tt, a_lead[idx], linestyle="--", label="Lead acceleration")
    axes[2].plot(tt, a_ego[idx], label="Ego acceleration")
    axes[2].axhline(0.0, linewidth=0.8)
    axes[2].set_ylabel("Acceleration (m/s²)")

    axes[3].plot(tt, jerk[idx], label="Ego realized jerk")
    axes[3].axhline(0.0, linewidth=0.8)
    axes[3].set_ylabel("Jerk (m/s³)")
    axes[3].set_xlabel("Time (s)")

    gain_text = ", ".join(f"{name}={value:.6g}" for name, value in zip(GAIN_NAMES, gains))
    fig.suptitle(
        f"Following validation: {scenario_id}\n{gain_text} | checkpoint={checkpoint_name}",
        fontsize=12,
    )

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_cruise_motion(
    *,
    scenario: CruiseScenario,
    history: dict[str, list[float]],
    gains: np.ndarray,
    checkpoint_name: str,
    max_plot_points: int,
):
    t = np.asarray(history["time"], dtype=float)
    v_ego = np.asarray(history["v_ego"], dtype=float)
    v_set = np.asarray(history["v_set"], dtype=float)
    a_ego = np.asarray(history["a_ego"], dtype=float)
    jerk = np.asarray(history["jerk"], dtype=float)

    if t.size < 2:
        raise RuntimeError(f"{scenario.scenario_id}: simulation returned fewer than two samples")

    idx = _decimation_indices(t.size, max_plot_points)
    tt = t[idx]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    axes[0].plot(tt, v_set[idx], linestyle="--", label="Set speed")
    axes[0].plot(tt, v_ego[idx], label="Ego speed")
    axes[0].set_ylabel("Speed (m/s)")

    axes[1].plot(tt, a_ego[idx], label="Ego acceleration")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_ylabel("Acceleration (m/s²)")

    axes[2].plot(tt, jerk[idx], label="Ego realized jerk")
    axes[2].axhline(0.0, linewidth=0.8)
    axes[2].set_ylabel("Jerk (m/s³)")
    axes[2].set_xlabel("Time (s)")

    # Mark every actual v_set transition.  The first event at t=0 establishes the
    # initial set point and is not counted as a change.
    for event_time in np.asarray(scenario.event_times_s[1:], dtype=float):
        for ax in axes:
            ax.axvline(event_time, linestyle=":", linewidth=0.8, alpha=0.35)

    gain_text = ", ".join(f"{name}={value:.6g}" for name, value in zip(GAIN_NAMES, gains))
    fig.suptitle(
        f"Cruise validation: {scenario.scenario_id} | "
        f"{len(scenario.event_times_s) - 1} v_set changes\n"
        f"{gain_text} | checkpoint={checkpoint_name}",
        fontsize=12,
    )

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def scenario_metadata(case: ValidationCase) -> dict[str, Any]:
    s = case.scenario
    base: dict[str, Any] = {
        "scenario_id": case.scenario_id,
        "scenario_type": case.scenario_type,
        "source_kind": case.source_kind,
        "description": case.description,
        "duration_s": float(s.duration_s),
        "provenance": s.provenance,
    }
    if isinstance(s, CruiseScenario):
        base.update(
            {
                "initial_ego_speed_mps": float(s.initial_ego_speed_mps),
                "event_times_s": [float(x) for x in s.event_times_s],
                "set_speeds_mps": [float(x) for x in s.set_speeds_mps],
                "vset_change_count": int(len(s.event_times_s) - 1),
            }
        )
    else:
        base.update(
            {
                "initial_ego_speed_mps": float(s.initial_ego_speed_mps),
                "initial_lead_speed_mps": float(s.initial_lead_speed_mps),
                "initial_gap_m": float(s.initial_gap_m),
                "lead_trace_samples": int(len(s.lead_speed_mps)),
            }
        )
    return base


def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_from_root(args.checkpoint)
    payload, gains = load_checkpoint(checkpoint_path)
    manifest_path = resolve_manifest_path(args.manifest)

    cases = collect_cases(args, manifest_path)

    if args.list_scenarios:
        print("Enabled validation cases:")
        for case in cases:
            s = case.scenario
            print(
                f"  {case.scenario_id:26s} "
                f"{case.scenario_type:9s} "
                f"{case.source_kind:20s} "
                f"{s.duration_s:8.3f} s  {s.provenance}"
            )
        return

    out_dir = _resolve_from_root(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_tag = checkpoint_path.stem
    pdf_path = out_dir / f"validation_motion_{checkpoint_tag}.pdf"
    csv_path = out_dir / f"validation_motion_metrics_{checkpoint_tag}.csv"
    json_path = out_dir / f"validation_motion_metadata_{checkpoint_tag}.json"

    metric_rows: list[dict[str, Any]] = []
    case_meta: list[dict[str, Any]] = []
    written_pngs: list[Path] = []

    print(f"Checkpoint : {checkpoint_path}")
    print(
        "Best gains : "
        + ", ".join(f"{name}={value:.9g}" for name, value in zip(GAIN_NAMES, gains))
    )
    if payload.get("best_fitness") is not None:
        print(f"Fitness    : {payload['best_fitness']}")
    if payload.get("algo_name") is not None:
        print(f"Algorithm  : {payload['algo_name']}")
    if payload.get("run_id") is not None:
        print(f"Run ID     : {payload['run_id']}")
    if not args.synthetic_only:
        print(f"Manifest   : {manifest_path}")
    print(f"Cases      : {', '.join(c.scenario_id for c in cases)}")

    with PdfPages(pdf_path) as pdf:
        for case in cases:
            scenario = case.scenario
            print(
                f"\nSimulating {case.scenario_id} [{case.scenario_type}] "
                f"({scenario.duration_s:.3f} s)..."
            )

            if case.scenario_type == "following":
                assert isinstance(scenario, FollowingScenario)
                history = simulate_following_history(gains, scenario)
                metrics = compute_following_metrics(history)
                fig = plot_following_motion(
                    scenario_id=case.scenario_id,
                    history=history,
                    gains=gains,
                    checkpoint_name=checkpoint_path.name,
                    max_plot_points=args.max_plot_points,
                )
                print(
                    "  "
                    f"spacing RMSE={metrics['spacing_rmse_m']:.6g} m | "
                    f"speed RMSE={metrics['speed_rmse_mps']:.6g} m/s | "
                    f"jerk RMS={metrics['jerk_rms_mps3']:.6g} m/s³ | "
                    f"peak |jerk|={metrics['peak_abs_jerk_mps3']:.6g} m/s³"
                )
            else:
                assert isinstance(scenario, CruiseScenario)
                history = simulate_cruise_history(gains, scenario)
                metrics = compute_cruise_metrics(history, scenario)
                fig = plot_cruise_motion(
                    scenario=scenario,
                    history=history,
                    gains=gains,
                    checkpoint_name=checkpoint_path.name,
                    max_plot_points=args.max_plot_points,
                )
                print(
                    "  "
                    f"v_set changes={int(metrics['vset_change_count'])} | "
                    f"speed RMSE={metrics['set_speed_rmse_mps']:.6g} m/s | "
                    f"jerk RMS={metrics['jerk_rms_mps3']:.6g} m/s³ | "
                    f"peak |jerk|={metrics['peak_abs_jerk_mps3']:.6g} m/s³"
                )

            row: dict[str, Any] = {
                "scenario_id": case.scenario_id,
                "scenario_type": case.scenario_type,
                "source_kind": case.source_kind,
                "provenance": scenario.provenance,
                **metrics,
            }
            metric_rows.append(row)
            case_meta.append({**scenario_metadata(case), "metrics": metrics})

            png_path = out_dir / f"{case.scenario_id.lower()}_{checkpoint_tag}.png"
            fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            written_pngs.append(png_path)

            if args.show:
                plt.show()
            plt.close(fig)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(csv_path, index=False)

    metadata: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_metadata": {
            key: payload.get(key)
            for key in (
                "fun_name",
                "run_id",
                "algo_name",
                "obj_name",
                "best_fitness",
                "duration",
                "function_evaluations",
            )
            if key in payload
        },
        "best_solution": {name: float(value) for name, value in zip(GAIN_NAMES, gains)},
        "validation_cases": case_meta,
        "synthetic_validation_contract": {
            "sinusoidal_following": {
                "id": SYNTHETIC_FOLLOW_ID,
                "duration_s": SYN_FOLLOW_DURATION_S,
                "initial_speed_mps": SYN_FOLLOW_INITIAL_SPEED_MPS,
                "initial_gap_m": SYN_FOLLOW_INITIAL_GAP_M,
                "lead_accel_amplitude_mps2": SYN_FOLLOW_ACCEL_AMPLITUDE_MPS2,
                "lead_phase_rate_deg_per_s": SYN_FOLLOW_PHASE_RATE_DEG_PER_S,
            },
            "cruise_10x_vset": {
                "id": SYNTHETIC_CRUISE_ID,
                "plateau_duration_s": SYN_CRUISE_PLATEAU_S,
                "set_speeds_mps": [float(x) for x in SYN_CRUISE_SET_SPEEDS_MPS],
                "change_count": int(SYN_CRUISE_SET_SPEEDS_MPS.size - 1),
                "initial_ego_speed_mps": SYN_CRUISE_INITIAL_EGO_SPEED_MPS,
            },
        },
        "plot_note": (
            "All metrics use full-resolution simulation data. Long traces may be "
            "deterministically decimated only for rendering. Synthetic diagnostics "
            "are reporting-only and are not part of calibration or optimizer fitness."
        ),
    }
    if not args.synthetic_only:
        metadata["scenario_manifest"] = str(manifest_path)
        metadata["scenario_manifest_sha256"] = sha256_file(manifest_path)

    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nWrote:")
    for path in written_pngs:
        print(f"  {path}")
    print(f"  {pdf_path}")
    print(f"  {csv_path}")
    print(f"  {json_path}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        message = str(exc)
        if "not 'frozen'" in message or "no validation scenarios" in message.lower():
            print(f"ERROR: {message}", file=sys.stderr)
            print(
                "Run `python prepare_scenarios_v2.py` first for held-out validation, "
                "or use `--synthetic-only` to run the two synthetic diagnostics now.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        raise
