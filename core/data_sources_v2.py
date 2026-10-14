"""Public trajectory acquisition and deterministic preprocessing for Framework V2.

The main research pipeline intentionally does not bundle third-party OpenACC raw
files.  This module downloads them from the European Commission JRC public data
server, converts the leader-speed traces to a small project-local format, and
records all provenance needed to reproduce the scenario manifest.

No controller output is used when selecting calibration windows.  The fixed rule
selects the valid window with the largest *leader kinematic excitation score*
(speed standard deviation plus a small acceleration-activity term).  This keeps
window selection deterministic and independent of optimizer/controller results.
"""

from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import time
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from core.config import ROOT_DIR


OPENACC_ROOT = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/"
    "TransportExpData/JRCDBT0001/LATEST"
)

SOURCE_REGISTRY = {
    "asta_calibration": {
        "url": f"{OPENACC_ROOT}/AstaZero/ASta_040719_platoon3.csv",
        "raw_path": "data_external/raw/openacc/ASta_040719_platoon3.csv",
        "processed_path": "data_external/processed/asta_calibration.csv",
        "source": "OpenACC AstaZero / ASta_040719_platoon3.csv",
        "campaign": "AstaZero",
        "window_s": 90.0,
        "selection": "max_leader_kinematic_excitation",
    },
    "vicolungo_calibration": {
        "url": f"{OPENACC_ROOT}/Vicolungo/JRC-VC_280219_part4_highway.csv",
        "raw_path": "data_external/raw/openacc/JRC-VC_280219_part4_highway.csv",
        "processed_path": "data_external/processed/vicolungo_calibration.csv",
        "source": "OpenACC Vicolungo / JRC-VC_280219_part4_highway.csv",
        "campaign": "Vicolungo",
        "window_s": 90.0,
        "selection": "max_leader_kinematic_excitation",
    },
    "casale_validation": {
        "url": f"{OPENACC_ROOT}/Casale/part6.csv",
        "raw_path": "data_external/raw/openacc/Casale_part6.csv",
        "processed_path": "data_external/processed/casale_validation_offset_50_120.csv",
        "source": "OpenACC Casale / part6.csv",
        "campaign": "Casale",
        "window_s": 70.0,
        "selection": "fixed_offset_50_120_s",
    },
    "wltc_class3b": {
        "url": "https://raw.githubusercontent.com/JRCSTU/wltp/master/wltp/cycles/V_class3b.txt",
        "raw_path": "data_external/raw/wltc/V_class3b.txt",
        "processed_path": "data_external/processed/wltc_class3b.csv",
        "source": "JRCSTU WLTC Class 3b speed trace",
        "campaign": "WLTC Class 3b",
        "selection": "full_cycle",
    },
}


@dataclass(frozen=True)
class ProcessedTrajectory:
    source_key: str
    processed_path: Path
    raw_path: Path
    raw_sha256: str
    processed_sha256: str
    source_url: str
    source_name: str
    source_sampling_hz: float
    start_s: float
    end_s: float
    duration_s: float
    initial_speed_mps: float
    speed_column: str
    selection_rule: str
    excitation_score: float | None = None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, destination: Path, *, force: bool = False, retries: int = 3) -> Path:
    """Download one public source with a small retry loop.

    Existing non-empty files are reused unless ``force`` is true.  The server-side
    raw files can be several MB, so data acquisition is intentionally separate from
    Python package installation.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return destination

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": "ACC-Framework-V2/1.0"})
            with urlopen(req, timeout=120) as response, destination.open("wb") as out:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            if destination.stat().st_size == 0:
                raise RuntimeError(f"downloaded empty file from {url}")
            return destination
        except Exception as exc:  # pragma: no cover - network dependent
            last_error = exc
            destination.unlink(missing_ok=True)
            if attempt < retries:
                time.sleep(1.5 * attempt)
    raise RuntimeError(
        f"Could not download {url} after {retries} attempts. "
        "Check internet access or place the raw file at "
        f"{destination.relative_to(ROOT_DIR) if destination.is_relative_to(ROOT_DIR) else destination}."
    ) from last_error


def _openacc_header_line(path: Path) -> int:
    """Return zero-based row containing the actual Time,... table header."""
    with path.open("r", encoding="utf-8-sig", errors="replace") as fh:
        for idx, line in enumerate(fh):
            first = line.split(",", 1)[0].strip().strip('"')
            if first.lower() == "time":
                return idx
            if idx > 30:
                break
    raise ValueError(f"{path}: could not locate OpenACC table header beginning with 'Time'")


def read_openacc_csv(path: Path) -> pd.DataFrame:
    header_row = _openacc_header_line(path)
    df = pd.read_csv(path, skiprows=header_row)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def detect_openacc_leader_speed_column(df: pd.DataFrame) -> str:
    """Detect the first vehicle's speed column in the original OpenACC schema."""
    cols = [str(c).strip() for c in df.columns]
    exact = [c for c in cols if re.fullmatch(r"Speed\s*1", c, flags=re.I)]
    if exact:
        return exact[0]
    numbered = []
    for c in cols:
        m = re.fullmatch(r"Speed\s*(\d+)", c, flags=re.I)
        if m:
            numbered.append((int(m.group(1)), c))
    if numbered:
        return min(numbered)[1]
    if "Speed" in cols:
        return "Speed"
    candidates = [c for c in cols if "speed" in c.lower()]
    if candidates:
        return candidates[0]
    raise KeyError(f"No leader speed column found. Available columns: {cols[:30]}")


def _clean_time_speed(df: pd.DataFrame, speed_column: str) -> tuple[np.ndarray, np.ndarray]:
    if "Time" not in df.columns:
        raise KeyError("OpenACC data must contain a 'Time' column")
    t = pd.to_numeric(df["Time"], errors="coerce").to_numpy(dtype=float)
    v = pd.to_numeric(df[speed_column], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(t) & np.isfinite(v) & (v >= 0.0) & (v <= 70.0)
    t, v = t[ok], v[ok]
    order = np.argsort(t)
    t, v = t[order], v[order]
    t, unique_idx = np.unique(t, return_index=True)
    v = v[unique_idx]
    if t.size < 10:
        raise ValueError("OpenACC source contains too few valid time/speed samples")
    return t, v


def infer_sampling_hz(t: np.ndarray) -> float:
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise ValueError("cannot infer sampling rate from timestamps")
    return float(1.0 / np.median(diffs))


def select_excitation_window(
    t: np.ndarray,
    v: np.ndarray,
    *,
    window_s: float = 90.0,
    stride_s: float = 5.0,
    min_mean_speed_mps: float = 5.0,
) -> tuple[float, float, float]:
    """Deterministically select a kinematically informative leader-speed window.

    Score = std(speed) + 0.25 * mean(abs(dv/dt)).  Only leader kinematics enter
    the rule; controller response, optimized gains, or objective values never do.
    """
    start_min = float(t[0])
    start_max = float(t[-1] - window_s)
    if start_max <= start_min:
        raise ValueError(f"source is shorter than required {window_s:g}s window")
    candidates = np.arange(start_min, start_max + 1e-9, stride_s)
    global_dt = np.diff(t)
    global_dt = global_dt[np.isfinite(global_dt) & (global_dt > 0.0)]
    median_dt = float(np.median(global_dt))
    expected_samples = max(2, int(round(window_s / median_dt)))
    best: tuple[float, float] | None = None
    for start in candidates:
        end = start + window_s
        mask = (t >= start) & (t <= end)
        tw, vw = t[mask], v[mask]
        # Reject windows spanning large holes or containing too little of the nominal
        # source sampling density. This avoids selecting a high-variance artifact
        # created by a discontinuity in an otherwise valid trip file.
        if tw.size < max(20, int(0.9 * expected_samples)):
            continue
        if np.max(np.diff(tw)) > max(1.0, 5.0 * median_dt):
            continue
        if float(np.mean(vw)) < min_mean_speed_mps:
            continue
        dt = np.diff(tw)
        dv = np.diff(vw)
        valid = dt > 0
        accel_activity = float(np.mean(np.abs(dv[valid] / dt[valid]))) if np.any(valid) else 0.0
        score = float(np.std(vw) + 0.25 * accel_activity)
        if best is None or score > best[1]:
            best = (float(start), score)
    if best is None:
        raise ValueError("no valid excitation window found in source trajectory")
    return best[0], best[0] + float(window_s), best[1]


def write_processed_window(
    *,
    source_key: str,
    raw_path: Path,
    processed_path: Path,
    source_url: str,
    source_name: str,
    selection: str,
    window_s: float,
) -> ProcessedTrajectory:
    df = read_openacc_csv(raw_path)
    speed_col = detect_openacc_leader_speed_column(df)
    t, v = _clean_time_speed(df, speed_col)
    source_hz = infer_sampling_hz(t)

    if selection == "fixed_offset_50_120_s":
        # OpenACC Casale uses a *common experiment time frame* across the trip
        # rather than resetting each part file to t=0. Therefore ``part6.csv``
        # can legitimately begin at a timestamp far above 120 s. The held-out
        # window is defined relative to the beginning of the selected part: the
        # interval [file_start + 50 s, file_start + 120 s].
        origin = float(t[0])
        start, end, score = origin + 50.0, origin + 120.0, None
        if t[-1] + 1e-9 < end:
            available = float(t[-1] - origin)
            raise ValueError(
                f"{raw_path}: Casale part is only {available:.1f}s long after its first "
                "valid sample; need at least 120s for the fixed relative 50-120s window"
            )
    elif selection == "max_leader_kinematic_excitation":
        start, end, score = select_excitation_window(t, v, window_s=window_s)
    else:
        raise ValueError(f"unsupported OpenACC selection rule: {selection}")

    mask = (t >= start) & (t <= end)
    tw, vw = t[mask], v[mask]
    if tw.size < 2:
        raise ValueError(f"{raw_path}: selected window is empty")
    tw = tw - tw[0]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": tw, "lead_speed_mps": vw}).to_csv(processed_path, index=False)

    return ProcessedTrajectory(
        source_key=source_key,
        processed_path=processed_path,
        raw_path=raw_path,
        raw_sha256=sha256_file(raw_path),
        processed_sha256=sha256_file(processed_path),
        source_url=source_url,
        source_name=source_name,
        source_sampling_hz=source_hz,
        start_s=float(start),
        end_s=float(end),
        duration_s=float(tw[-1] - tw[0]),
        initial_speed_mps=float(vw[0]),
        speed_column=speed_col,
        selection_rule=selection,
        excitation_score=score,
    )


def process_wltc_class3b(raw_path: Path, processed_path: Path, source_url: str) -> ProcessedTrajectory:
    values = []
    for token in raw_path.read_text(encoding="utf-8", errors="replace").split():
        try:
            values.append(float(token))
        except ValueError:
            continue
    if len(values) < 1700:
        raise ValueError(f"{raw_path}: expected a full Class 3b trace; parsed only {len(values)} values")
    speed_kmh = np.asarray(values, dtype=float)
    speed_mps = speed_kmh / 3.6
    t = np.arange(speed_mps.size, dtype=float)  # public V_class3b trace is 1 Hz
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"time_s": t, "lead_speed_mps": speed_mps}).to_csv(processed_path, index=False)
    return ProcessedTrajectory(
        source_key="wltc_class3b",
        processed_path=processed_path,
        raw_path=raw_path,
        raw_sha256=sha256_file(raw_path),
        processed_sha256=sha256_file(processed_path),
        source_url=source_url,
        source_name="JRCSTU WLTC Class 3b speed trace",
        source_sampling_hz=1.0,
        start_s=0.0,
        end_s=float(t[-1]),
        duration_s=float(t[-1]),
        initial_speed_mps=float(speed_mps[0]),
        speed_column="V_class3b",
        selection_rule="full_cycle",
        excitation_score=None,
    )


def project_path(relative: str) -> Path:
    return ROOT_DIR / relative


def prepare_public_trajectories(*, force_download: bool = False) -> dict[str, ProcessedTrajectory]:
    prepared: dict[str, ProcessedTrajectory] = {}
    for key, spec in SOURCE_REGISTRY.items():
        raw = project_path(spec["raw_path"])
        processed = project_path(spec["processed_path"])
        download_file(spec["url"], raw, force=force_download)
        if key == "wltc_class3b":
            prepared[key] = process_wltc_class3b(raw, processed, spec["url"])
        else:
            prepared[key] = write_processed_window(
                source_key=key,
                raw_path=raw,
                processed_path=processed,
                source_url=spec["url"],
                source_name=spec["source"],
                selection=spec["selection"],
                window_s=float(spec["window_s"]),
            )
    return prepared
