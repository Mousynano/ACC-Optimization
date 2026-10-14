"""Numerical utilities used by the active ACC optimization and analysis pipeline."""

import math
import numpy as np


def iae(error):
    return abs(error)


def ise(error):
    return error * error


def itae(error, time):
    return abs(error) * time


def itse(error, time):
    return error * error * time


def lerp(a, b, t):
    return a + t * (b - a)


def compute_jerk(acceleration, dt):
    a = np.asarray(acceleration, dtype=float)
    if a.size < 2:
        return np.zeros_like(a)
    return np.concatenate(([0.0], np.diff(a) / float(dt)))


def compute_ttc_series(distance, v_ego, v_lead, eps=1e-9):
    """Evaluator-side TTC series using simulator ground-truth lead velocity."""
    d = np.asarray(distance, dtype=float)
    ego = np.asarray(v_ego, dtype=float)
    lead = np.asarray(v_lead, dtype=float)
    if not (d.shape == ego.shape == lead.shape):
        raise ValueError("distance, v_ego, and v_lead must have matching shapes")

    closing_speed = ego - lead
    ttc = np.full_like(d, np.inf, dtype=float)
    collision = d <= 0.0
    closing = (~collision) & (closing_speed > eps)
    ttc[collision] = 0.0
    ttc[closing] = d[closing] / closing_speed[closing]
    return ttc


def _interp_crossing_time(t, y, level, idx):
    t0, t1 = t[idx - 1], t[idx]
    y0, y1 = y[idx - 1], y[idx]
    if y1 == y0:
        return float(t1)
    alpha = (level - y0) / (y1 - y0)
    return float(t0 + alpha * (t1 - t0))


def _first_crossing_time(t, y, level, direction):
    if len(t) < 2:
        return None
    if direction == "rising":
        cond = (y[:-1] < level) & (y[1:] >= level)
    else:
        cond = (y[:-1] > level) & (y[1:] <= level)
    idxs = np.where(cond)[0]
    if idxs.size == 0:
        return None
    return _interp_crossing_time(t, y, level, int(idxs[0] + 1))


def _settling_time(t, y, target, band, start_time=None):
    if start_time is not None:
        mask = t >= start_time
        t, y = t[mask], y[mask]
    if len(t) == 0:
        return None
    outside = np.abs(y - target) > abs(float(band))
    if not np.any(outside):
        return float(t[0])
    last_outside = int(np.where(outside)[0][-1])
    if last_outside == len(t) - 1:
        return None
    return float(t[last_outside + 1])


def step_response_metrics(time, y, ref, y0=None, steady_window=2.0, tol=0.02):
    """Compute rise, peak, overshoot, settling, and steady-state metrics."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) == 0:
        raise ValueError("Empty time series")
    if y0 is None:
        y0 = float(y[0])

    mask = t >= (t[-1] - steady_window)
    y_ss = float(np.mean(y[mask])) if np.any(mask) else float(y[-1])
    steady_state_error = float(ref - y_ss)

    rising = ref >= y0
    peak_idx = int(np.argmax(y) if rising else np.argmin(y))
    peak_value = float(y[peak_idx])
    peak_time = float(t[peak_idx])
    overshoot = max(0.0, peak_value - ref) if rising else max(0.0, ref - peak_value)
    step_amp = abs(ref - y0)
    overshoot_pct = 100.0 * overshoot / step_amp if step_amp > 0.0 else 0.0

    low_level = y0 + 0.1 * (ref - y0)
    high_level = y0 + 0.9 * (ref - y0)
    direction = "rising" if rising else "falling"
    t10 = _first_crossing_time(t, y, low_level, direction)
    t90 = _first_crossing_time(t, y, high_level, direction)
    rise_time = t90 - t10 if t10 is not None and t90 is not None and t90 >= t10 else None

    band_scale = step_amp if step_amp > 0.0 else max(abs(ref), 1.0)
    settling = _settling_time(t, y, y_ss, tol * band_scale, start_time=t10)

    return {
        "y0": float(y0),
        "y_ss": y_ss,
        "steady_state_error": steady_state_error,
        "peak_value": peak_value,
        "peak_time": peak_time,
        "overshoot": float(overshoot),
        "overshoot_pct": float(overshoot_pct),
        "rise_time": float(rise_time) if rise_time is not None else None,
        "settling_time": settling,
    }
