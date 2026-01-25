import math
import random
import json
import os
import numpy as np

def seconds_to_hms(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{int(hours)} Jam, {int(minutes)} Menit, {int(seconds)} Detik'

def sin_angle(x):
    radian = x * (math.pi / 180)
    result = math.sin(radian)
    return result

def cos_angle(x):
    radian = x * (math.pi / 180)
    result = math.cos(radian)
    return result

def iae(error):
    return abs(error)

def ise(error):
    return error * error

def itae(error, time):
    return abs(error) * time

def itse(error, time):
    return error * error * time

def cappiello_iae(weight, error):
    return weight * abs(error)

def cappiello_ise(weight, error):
    return weight * error**2

def cappiello_itae(weight, error, time):
    return weight * abs(error) * time

def cappiello_itse(weight, error, time):
    return weight * error**2 * time

def dropna(data):
    return [value for value in data if value is not None and not (isinstance(value, float) and math.isnan(value))]

def clip(arr, min_value, max_value):
    clipped_arr = []
    for value in arr:
        clipped_value = max(min(value, max_value), min_value)
        clipped_arr.append(clipped_value)
    return clipped_arr

def random_uniform(low, high, size=None):
    if size is None:
        return random.uniform(low, high)
    else:
        return [random.uniform(low, high) for _ in range(size)]
    
def lerp(a, b, t):
    return a + t * (b - a)

def save_results(cars_history, best_car_params, fitness_arr, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    fitness_history_path = f'{output_folder}/fitness_history.json'
    with open(fitness_history_path, 'w') as fitness_file:
        json.dump(fitness_arr, fitness_file)

    cars_history_path = f'{output_folder}/cars_history.json'
    with open(cars_history_path, 'w') as cars_file:
        json.dump(cars_history, cars_file)

    best_car_params_path = f'{output_folder}/best_car_params.json'
    with open(best_car_params_path, 'w') as best_params_file:
        json.dump(best_car_params, best_params_file)

    print('Results saved successfully.')

def compute_jerk(a_ego, dt):
    a = np.array(a_ego)
    jerk = np.diff(a) / dt
    return np.concatenate([[0], jerk])  # align length

def compute_ttc(d_actual, v_ego, v_lead, eps=1e-9):
    """
    TTC definition (classic):
    TTC = d / (v_ego - v_lead) when v_ego > v_lead and d > 0
    otherwise TTC = inf
    """
    if d_actual <= 0:
        return 0.0  # already collided / overlap (or negative distance)
    dv = v_ego - v_lead
    if dv <= eps:
        return math.inf
    return d_actual / dv


import numpy as np

def _interp_crossing_time(t, y, level, idx):
    """
    Linear interpolation to estimate crossing time between idx-1 and idx
    where y crosses 'level'. Assumes (y[idx-1]-level) and (y[idx]-level) have opposite signs.
    """
    t0, t1 = t[idx - 1], t[idx]
    y0, y1 = y[idx - 1], y[idx]
    if y1 == y0:
        return float(t1)
    alpha = (level - y0) / (y1 - y0)
    return float(t0 + alpha * (t1 - t0))

def _first_crossing_time(t, y, level, direction="auto"):
    """
    Returns the first time y crosses 'level' (with interpolation).
    direction:
      - "rising": crossing from below to above
      - "falling": crossing from above to below
      - "auto": decide from start/end trend
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) < 2:
        return None

    if direction == "auto":
        direction = "rising" if (y[-1] >= y[0]) else "falling"

    if direction == "rising":
        cond = (y[:-1] < level) & (y[1:] >= level)
    else:
        cond = (y[:-1] > level) & (y[1:] <= level)

    idxs = np.where(cond)[0]
    if idxs.size == 0:
        return None

    i = int(idxs[0] + 1)
    return _interp_crossing_time(t, y, level, i)

def settling_time(t, y, y_final, tol=0.02, start_time=None):
    """
    Settling time: earliest time after which y stays within +/- tol*|y_final|
    (or absolute band if y_final == 0) for the remainder.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(t) == 0:
        return None

    if start_time is not None:
        mask = t >= start_time
        t2, y2 = t[mask], y[mask]
    else:
        t2, y2 = t, y

    if len(t2) == 0:
        return None

    band = tol * abs(y_final) if y_final != 0 else tol
    outside = np.abs(y2 - y_final) > band

    if not np.any(outside):
        return float(t2[0])  # already settled at start

    last_outside_idx = int(np.where(outside)[0][-1])
    if last_outside_idx == len(t2) - 1:
        return None  # never settles within horizon
    return float(t2[last_outside_idx + 1])

def step_response_metrics(time, y, ref, y0=None, steady_window=2.0, tol=0.02, rise_frac=(0.1, 0.9)):
    """
    Compute step-response metrics for a step-to-constant reference.
    - time: array (seconds)
    - y: measured signal (e.g., v_ego)
    - ref: target setpoint (e.g., 25 m/s)
    - y0: initial value (if None uses y[0])
    - steady_window: seconds from the end used to estimate steady-state value (mean of last window)
    - tol: settling band ratio, e.g., 0.02 = 2%
    - rise_frac: (low, high) fractions of step amplitude (default 10%-90%)

    Returns dict with:
      y0, y_ss, ss_error,
      peak_value, peak_time,
      overshoot, overshoot_pct,
      rise_time, t10, t90,
      settling_time
    """
    t = np.asarray(time, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) == 0:
        raise ValueError("Empty time series.")

    if y0 is None:
        y0 = float(y[0])

    # steady-state estimate from last steady_window seconds (mean)
    t_end = float(t[-1])
    if steady_window is not None and steady_window > 0:
        mask = t >= (t_end - steady_window)
        y_ss = float(np.mean(y[mask])) if np.any(mask) else float(y[-1])
    else:
        y_ss = float(y[-1])

    ss_error = float(ref - y_ss)

    # Determine step direction by comparing ref vs y0
    rising = ref >= y0
    direction = "rising" if rising else "falling"

    # Peak / extreme
    if rising:
        peak_idx = int(np.argmax(y))
        peak_value = float(y[peak_idx])
        peak_time = float(t[peak_idx])
        overshoot = max(0.0, peak_value - ref)
    else:
        peak_idx = int(np.argmin(y))
        peak_value = float(y[peak_idx])
        peak_time = float(t[peak_idx])
        overshoot = max(0.0, ref - peak_value)

    step_amp = abs(ref - y0)
    overshoot_pct = float(100.0 * overshoot / step_amp) if step_amp > 0 else 0.0

    # Rise time 10%-90% of step amplitude (based on y0 -> ref)
    low_f, high_f = rise_frac
    low_level = y0 + (ref - y0) * low_f
    high_level = y0 + (ref - y0) * high_f

    t10 = _first_crossing_time(t, y, low_level, direction=direction)
    t90 = _first_crossing_time(t, y, high_level, direction=direction)
    rise_time = (t90 - t10) if (t10 is not None and t90 is not None and t90 >= t10) else None

    # Settling time relative to ref (common in control)
    ts = settling_time(t, y, y_final=ref, tol=tol, start_time=t10)

    return {
        "y0": float(y0),
        "y_ss": float(y_ss),
        "steady_state_error": ss_error,

        "peak_value": peak_value,
        "peak_time": peak_time,

        "overshoot": float(overshoot),
        "overshoot_pct": float(overshoot_pct),

        "t10": t10,
        "t90": t90,
        "rise_time": float(rise_time) if rise_time is not None else None,

        "settling_time": ts,
        "ref": float(ref),
        "tol": float(tol),
    }
