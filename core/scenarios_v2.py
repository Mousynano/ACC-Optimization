"""Scenario loading/preprocessing for the v2 ACC cost framework.

Research runs consume only scenarios declared in ``scenario_manifest_acc_cost_v2.json``
when that manifest is explicitly frozen.  Development smoke tests use a separate,
clearly labelled synthetic section and must never be treated as research data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from core.config import ROOT_DIR, resolve_project_path


@dataclass(frozen=True)
class CruiseScenario:
    scenario_id: str
    duration_s: float
    initial_ego_speed_mps: float
    event_times_s: np.ndarray
    set_speeds_mps: np.ndarray
    weight: float = 1.0
    provenance: str = ""


@dataclass(frozen=True)
class FollowingScenario:
    scenario_id: str
    duration_s: float
    initial_ego_speed_mps: float
    initial_lead_speed_mps: float
    initial_gap_m: float
    lead_speed_mps: np.ndarray
    weight: float = 1.0
    provenance: str = ""


@dataclass(frozen=True)
class ScenarioSuite:
    cruise: Tuple[CruiseScenario, ...]
    following: Tuple[FollowingScenario, ...]
    manifest_path: Path
    manifest_sha256: str
    status: str
    development_only: bool


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _validate_events(events: Sequence[Sequence[float]], duration_s: float):
    if not events:
        raise ValueError("cruise_events scenario requires at least one event")
    times = np.asarray([float(x[0]) for x in events], dtype=float)
    speeds = np.asarray([float(x[1]) for x in events], dtype=float)
    if times[0] != 0.0:
        raise ValueError("first cruise set-speed event must start at t=0")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("cruise event times must be strictly increasing")
    if times[-1] >= float(duration_s):
        raise ValueError("last cruise event must occur before scenario end")
    if np.any(~np.isfinite(speeds)) or np.any(speeds < 0.0):
        raise ValueError("set speeds must be finite and nonnegative")
    return times, speeds


def _build_cruise(entry: dict) -> CruiseScenario:
    if entry.get("type") != "cruise_events":
        raise ValueError(f"unsupported cruise scenario type: {entry.get('type')}")
    duration = float(entry["duration_s"])
    times, speeds = _validate_events(entry["events"], duration)
    return CruiseScenario(
        scenario_id=str(entry["id"]),
        duration_s=duration,
        initial_ego_speed_mps=float(entry["initial_ego_speed_mps"]),
        event_times_s=times,
        set_speeds_mps=speeds,
        weight=float(entry.get("weight", 1.0)),
        provenance=str(entry.get("source", entry.get("note", ""))),
    )


def _generate_sinusoidal_lead(entry: dict, sampling_hz: int) -> FollowingScenario:
    dt = 1.0 / float(sampling_hz)
    duration = float(entry["duration_s"])
    steps = int(round(duration * sampling_hz))
    lead_speed = np.empty(steps, dtype=float)
    v = float(entry["initial_lead_speed_mps"])
    phase = 0.0
    amplitude = float(entry.get("lead_accel_amplitude_mps2", 1.0))
    phase_rate = float(entry.get("lead_phase_rate_deg_per_s", 30.0))
    for i in range(steps):
        phase = (phase + phase_rate * dt) % 360.0
        a = amplitude * np.sin(np.deg2rad(phase))
        v = max(v + a * dt, 0.0)
        lead_speed[i] = v
    return FollowingScenario(
        scenario_id=str(entry["id"]),
        duration_s=duration,
        initial_ego_speed_mps=float(entry["initial_ego_speed_mps"]),
        initial_lead_speed_mps=float(entry["initial_lead_speed_mps"]),
        initial_gap_m=float(entry["initial_gap_m"]),
        lead_speed_mps=lead_speed,
        weight=float(entry.get("weight", 1.0)),
        provenance=str(entry.get("note", "synthetic sinusoidal lead")),
    )


def _unit_speed(values: np.ndarray, unit: str) -> np.ndarray:
    unit = unit.strip().lower()
    if unit in {"m/s", "mps", "m_per_s", "ms-1"}:
        return values.astype(float)
    if unit in {"km/h", "kmh", "kph"}:
        return values.astype(float) / 3.6
    raise ValueError(f"unsupported speed unit: {unit}")


def _load_csv_following(entry: dict, sampling_hz: int) -> FollowingScenario:
    source = resolve_project_path(str(entry["path"]))
    if not source.exists():
        raise FileNotFoundError(f"following scenario file not found: {source}")

    provenance = entry.get("provenance", {})
    expected_sha = str(provenance.get("processed_sha256", "")).strip()
    if expected_sha:
        actual_sha = sha256_file(source)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"{source}: SHA-256 mismatch. Expected {expected_sha}, got {actual_sha}. "
                "Run `python main.py prepare` instead of using silently changed trajectory data."
            )

    resampling = entry.get("resampling", {})
    method = str(resampling.get("method", "linear_interpolation_of_leader_speed"))
    if method != "linear_interpolation_of_leader_speed":
        raise ValueError(f"unsupported following resampling method: {method}")

    df = pd.read_csv(source)
    tcol = str(entry["time_column"])
    vcol = str(entry["lead_speed_column"])
    if tcol not in df or vcol not in df:
        raise KeyError(f"{source}: missing required columns {tcol!r}/{vcol!r}")

    t = np.asarray(df[tcol], dtype=float)
    v = _unit_speed(np.asarray(df[vcol], dtype=float), str(entry["lead_speed_unit"]))
    finite = np.isfinite(t) & np.isfinite(v)
    t, v = t[finite], v[finite]
    if t.size < 2:
        raise ValueError(f"{source}: insufficient finite time/speed samples")
    order = np.argsort(t)
    t, v = t[order], v[order]

    start = float(entry.get("start_s", t[0]))
    end = float(entry.get("end_s", t[-1]))
    if end <= start:
        raise ValueError("end_s must be greater than start_s")
    keep = (t >= start) & (t <= end)
    t, v = t[keep], v[keep]
    if t.size < 2:
        raise ValueError(f"{source}: selected window contains fewer than 2 samples")

    # Remove duplicate timestamps deterministically, keeping the first occurrence.
    unique_t, unique_idx = np.unique(t, return_index=True)
    t, v = unique_t, v[unique_idx]
    dt = 1.0 / float(sampling_hz)
    duration = float(end - start)
    target_t = start + np.arange(int(round(duration * sampling_hz)), dtype=float) * dt
    lead_speed = np.interp(target_t, t, v)
    initial_lead = float(entry.get("initial_lead_speed_mps", lead_speed[0]))
    initial_ego = float(entry.get("initial_ego_speed_mps", initial_lead))

    return FollowingScenario(
        scenario_id=str(entry["id"]),
        duration_s=float(lead_speed.size / sampling_hz),
        initial_ego_speed_mps=initial_ego,
        initial_lead_speed_mps=initial_lead,
        initial_gap_m=float(entry["initial_gap_m"]),
        lead_speed_mps=np.asarray(lead_speed, dtype=float),
        weight=float(entry.get("weight", 1.0)),
        provenance=f"{source}:{start:g}-{end:g}s",
    )


def _build_following(entry: dict, sampling_hz: int) -> FollowingScenario:
    kind = entry.get("type")
    if kind == "synthetic_sinusoidal_lead":
        return _generate_sinusoidal_lead(entry, sampling_hz)
    if kind == "csv_lead_speed":
        return _load_csv_following(entry, sampling_hz)
    raise ValueError(f"unsupported following scenario type: {kind}")


def load_scenario_suite(
    manifest_path: str | Path = "scenario_manifest_acc_cost_v2.json",
    *,
    development: bool = False,
    require_frozen: bool = True,
) -> ScenarioSuite:
    path = Path(manifest_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    data = _load_manifest(path)
    sampling_hz = int(data.get("sampling_hz", 60))
    status = str(data.get("status", "unknown"))

    if development:
        section = data.get("development_only", {})
    else:
        if require_frozen and status != "frozen":
            raise RuntimeError(
                f"Scenario manifest is {status!r}, not 'frozen'. Audit/freeze exact "
                "external files/windows before research calibration. Use development=True "
                "only for smoke tests."
            )
        section = data.get("calibration", {})

    cruise_entries = section.get("cruise", [])
    following_entries = section.get("following", [])
    cruise = tuple(_build_cruise(e) for e in cruise_entries)
    following = tuple(_build_following(e, sampling_hz) for e in following_entries)
    if not cruise:
        raise RuntimeError("Scenario suite must contain at least one cruise scenario")
    if not following:
        raise RuntimeError("Scenario suite must contain at least one following scenario")

    return ScenarioSuite(
        cruise=cruise,
        following=following,
        manifest_path=path,
        manifest_sha256=sha256_file(path),
        status=status,
        development_only=bool(development),
    )


def load_validation_scenarios(
    manifest_path: str | Path = "scenario_manifest_acc_cost_v2.json",
    *,
    require_frozen: bool = True,
) -> tuple[FollowingScenario, ...]:
    """Load held-out/regression following scenarios declared in the frozen manifest.

    Validation scenarios never participate in Pareto calibration.  This loader exists
    so preflight/analysis code can verify that the held-out trajectory bundle is
    complete and hash-valid before a long campaign is started.
    """
    path = Path(manifest_path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    data = _load_manifest(path)
    status = str(data.get("status", "unknown"))
    if require_frozen and status != "frozen":
        raise RuntimeError(f"Scenario manifest is {status!r}, not 'frozen'.")
    sampling_hz = int(data.get("sampling_hz", 60))
    entries = data.get("validation", {}).get("scenarios", [])
    if not entries:
        raise RuntimeError("Frozen scenario manifest contains no validation scenarios")
    scenarios = tuple(_build_following(e, sampling_hz) for e in entries)
    if not scenarios:
        raise RuntimeError("No loadable validation scenarios were found")
    return scenarios
