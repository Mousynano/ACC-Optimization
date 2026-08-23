"""WLTC lead-vehicle profile utilities.

This module contains only lead-profile and sensor-perturbation logic. It does
not know anything about controllers, objectives, optimizers, or vehicle physics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def moving_average(x, window: int = 5) -> np.ndarray:
    """Smooth a 1-D signal with a simple moving average."""
    x = np.asarray(x, dtype=float)
    if window is None or int(window) <= 1:
        return x
    window = int(window)
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, kernel, mode="same")


def load_wltc_csv(csv_path: str | Path = "wltc_class3b.csv") -> Tuple[np.ndarray, np.ndarray]:
    """Load WLTC CSV and return ``(time_s, speed_mps)``.

    Supported headers:
    - time: ``time_s`` or ``time``; if absent, data are assumed to be 1 Hz.
    - speed: ``speed_mps``, ``v_mps``, ``speed_kmh``, ``v_kmh``, or ``speed``.
      A generic ``speed`` column is treated as km/h because WLTC tables are
      commonly distributed in km/h.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"WLTC CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.dtype.names is None:
        raise ValueError("WLTC CSV must have a header, for example: time_s,speed_kmh")

    cols = {name.lower().strip(): name for name in data.dtype.names}

    if "time_s" in cols:
        t_raw = np.asarray(data[cols["time_s"]], dtype=float)
    elif "time" in cols:
        t_raw = np.asarray(data[cols["time"]], dtype=float)
    else:
        t_raw = np.arange(len(data), dtype=float)

    if "speed_mps" in cols:
        v_raw = np.asarray(data[cols["speed_mps"]], dtype=float)
    elif "v_mps" in cols:
        v_raw = np.asarray(data[cols["v_mps"]], dtype=float)
    elif "speed_kmh" in cols:
        v_raw = np.asarray(data[cols["speed_kmh"]], dtype=float) / 3.6
    elif "v_kmh" in cols:
        v_raw = np.asarray(data[cols["v_kmh"]], dtype=float) / 3.6
    elif "speed" in cols:
        v_raw = np.asarray(data[cols["speed"]], dtype=float) / 3.6
    else:
        raise ValueError("WLTC CSV must contain speed_kmh, v_kmh, speed_mps, v_mps, or speed.")

    order = np.argsort(t_raw)
    return t_raw[order], v_raw[order]


def get_wltc_phase_window(phase: str) -> Tuple[float, float]:
    """Return WLTC Class-3 phase window in seconds."""
    phase = str(phase).lower().strip().replace("-", "_")
    windows = {
        "low": (0.0, 589.0),
        "medium": (589.0, 1022.0),
        "high": (1022.0, 1477.0),
        "extra_high": (1477.0, 1800.0),
        "extrahigh": (1477.0, 1800.0),
        "city": (0.0, 1022.0),
        "full": (0.0, 1800.0),
    }
    if phase not in windows:
        raise ValueError(f"Unknown WLTC phase: {phase}. Available: {list(windows.keys())}")
    return windows[phase]


def make_wltc_lead_profile(
    csv_path: str | Path = "wltc_class3b.csv",
    dt: float = 1.0 / 60.0,
    phase: str = "full",
    t_start: float | None = None,
    t_end: float | None = None,
    speed_scale: float = 1.0,
    smoothing_window: int = 5,
) -> Dict[str, object]:
    """Build a lead-vehicle trajectory from WLTC.

    Returns a dictionary compatible with the previous script:
    ``t``, ``v_lead``, ``a_lead``, ``x_lead``, ``dt``, ``phase``, ``source``.
    """
    t_raw, v_raw = load_wltc_csv(csv_path)

    if t_start is None or t_end is None:
        p_start, p_end = get_wltc_phase_window(phase)
        t_start = p_start if t_start is None else t_start
        t_end = p_end if t_end is None else t_end

    mask = (t_raw >= float(t_start)) & (t_raw <= float(t_end))
    t_raw = t_raw[mask]
    v_raw = v_raw[mask] * float(speed_scale)

    if len(t_raw) < 2:
        raise ValueError("WLTC segment is too short after filtering.")

    t_raw_local = t_raw - float(t_raw[0])
    t = np.arange(0.0, float(t_raw_local[-1]) + float(dt), float(dt))
    v_lead = np.maximum(np.interp(t, t_raw_local, v_raw), 0.0)
    a_lead = moving_average(np.gradient(v_lead, float(dt)), window=smoothing_window)

    x_lead = np.zeros_like(t)
    if len(t) > 1:
        x_lead[1:] = np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * float(dt))

    return {
        "t": t,
        "v_lead": v_lead,
        "a_lead": a_lead,
        "x_lead": x_lead,
        "dt": float(dt),
        "phase": phase,
        "source": str(csv_path),
    }


def apply_sensor_noise(
    profile: Dict[str, object],
    noise_std_v: float = 0.0,
    noise_std_x: float = 0.0,
    seed: int | None = None,
) -> Dict[str, object]:
    """Return a copy of ``profile`` with noise on perceived lead speed/position."""
    if noise_std_v <= 0 and noise_std_x <= 0:
        return profile

    rng = np.random.default_rng(seed)
    out = dict(profile)
    v = np.array(profile["v_lead"], dtype=float, copy=True)
    x = np.array(profile["x_lead"], dtype=float, copy=True)

    if noise_std_v > 0:
        v += rng.normal(0.0, float(noise_std_v), size=v.shape)
        v = np.maximum(v, 0.0)
    if noise_std_x > 0:
        x += rng.normal(0.0, float(noise_std_x), size=x.shape)

    out["v_lead"] = v
    out["x_lead"] = x
    out["a_lead"] = moving_average(np.gradient(v, float(profile["dt"])), window=5)
    return out


def apply_sensor_delay(profile: Dict[str, object], delay_s: float = 0.0) -> Dict[str, object]:
    """Return a copy of ``profile`` delayed by ``delay_s`` seconds."""
    if delay_s <= 0:
        return profile

    out = dict(profile)
    dt = float(profile["dt"])
    delay_steps = int(round(float(delay_s) / dt))

    def delay_array(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if delay_steps <= 0:
            return x.copy()
        return np.concatenate([np.full(delay_steps, x[0]), x[:-delay_steps]])

    out["v_lead"] = delay_array(profile["v_lead"])
    out["a_lead"] = delay_array(profile["a_lead"])
    out["x_lead"] = delay_array(profile["x_lead"])
    return out
