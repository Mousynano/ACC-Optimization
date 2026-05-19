"""
sisken_prastiyanto_wltc.py

Drop-in transition version of sisken_prastiyanto.py for WLTC-based ACC scenarios.

Core idea:
- Keep the old ACC controller + VehiclePhysics integration.
- Replace the hand-crafted lead vehicle update with a WLTC lead profile.
- Still support the old sine-wave lead vehicle when no lead_profile is supplied.

Expected WLTC CSV format:
    time_s,speed_kmh
or:
    time_s,speed_mps

Recommended default:
    data/wltc_class3b.csv
"""

import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from core.utils import sin_angle, itse  # kept for backward compatibility
from core.physics import VehiclePhysics


# ============================================================
# Legacy scenario list.
# You can overwrite SCENARIOS by calling use_wltc_scenarios(...)
# ============================================================

SCENARIOS = [
    {"name": "legacy_nominal", "theta_angle": 0, "vehicle_mass": 1500, "rolling_resistance": 0.06},
]


# ============================================================
# WLTC helpers
# ============================================================

def _moving_average(x, window=5):
    """Simple smoothing for acceleration estimated from speed."""
    x = np.asarray(x, dtype=float)
    if window is None or int(window) <= 1:
        return x

    window = int(window)
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(x, kernel, mode="same")


def load_wltc_csv(csv_path="wltc_class3b.csv"):
    """
    Load WLTC CSV.

    Supported columns:
    - time_s / time
    - speed_mps / v_mps
    - speed_kmh / v_kmh / speed

    If only speed is found, it is assumed to be km/h.
    If no time column is found, data is assumed to be 1 Hz.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"WLTC CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=float, encoding="utf-8")

    if data.dtype.names is None:
        raise ValueError(
            "WLTC CSV must have a header, for example: time_s,speed_kmh"
        )

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
        # WLTC tables are usually in km/h.
        v_raw = np.asarray(data[cols["speed"]], dtype=float) / 3.6
    else:
        raise ValueError(
            "WLTC CSV must contain speed_kmh, v_kmh, speed_mps, v_mps, or speed."
        )

    order = np.argsort(t_raw)
    t_raw = t_raw[order]
    v_raw = v_raw[order]

    return t_raw, v_raw


def get_wltc_phase_window(phase):
    """
    WLTC Class 3 phase windows in seconds.
    Works for both Class 3a and 3b because phase durations are aligned.
    """
    phase = str(phase).lower().strip().replace("-", "_")

    windows = {
        "low": (0.0, 589.0),
        "medium": (589.0, 1022.0),
        "high": (1022.0, 1477.0),
        "extra_high": (1477.0, 1800.0),
        "extrahigh": (1477.0, 1800.0),
        "city": (0.0, 1022.0),       # low + medium
        "full": (0.0, 1800.0),
    }

    if phase not in windows:
        raise ValueError(f"Unknown WLTC phase: {phase}. Available: {list(windows.keys())}")

    return windows[phase]


def make_wltc_lead_profile(
    csv_path="wltc_class3b.csv",
    dt=1 / 60,
    phase="full",
    t_start=None,
    t_end=None,
    speed_scale=1.0,
    smoothing_window=5,
):
    """
    Build lead vehicle profile from WLTC.

    Returns:
        {
            "t": array [s],
            "v_lead": array [m/s],
            "a_lead": array [m/s^2],
            "x_lead": array [m], starts from zero,
            "dt": float
        }
    """
    t_raw, v_raw = load_wltc_csv(csv_path)

    if t_start is None or t_end is None:
        p_start, p_end = get_wltc_phase_window(phase)
        if t_start is None:
            t_start = p_start
        if t_end is None:
            t_end = p_end

    mask = (t_raw >= t_start) & (t_raw <= t_end)
    t_raw = t_raw[mask]
    v_raw = v_raw[mask] * float(speed_scale)

    if len(t_raw) < 2:
        raise ValueError("WLTC segment is too short after filtering.")

    # Start the scenario clock from zero to keep the simulator simple.
    t0 = float(t_raw[0])
    t_raw_local = t_raw - t0
    t_end_local = float(t_raw_local[-1])

    t = np.arange(0.0, t_end_local + dt, dt)
    v_lead = np.interp(t, t_raw_local, v_raw)
    v_lead = np.maximum(v_lead, 0.0)

    a_lead = np.gradient(v_lead, dt)
    a_lead = _moving_average(a_lead, window=smoothing_window)

    x_lead = np.zeros_like(t)
    if len(t) > 1:
        x_lead[1:] = np.cumsum(0.5 * (v_lead[1:] + v_lead[:-1]) * dt)

    return {
        "t": t,
        "v_lead": v_lead,
        "a_lead": a_lead,
        "x_lead": x_lead,
        "dt": float(dt),
        "phase": phase,
        "source": str(csv_path),
    }


def apply_sensor_noise(profile, noise_std_v=0.0, noise_std_x=0.0, seed=None):
    """Apply noise to perceived lead velocity and position."""
    if noise_std_v <= 0 and noise_std_x <= 0:
        return profile

    rng = np.random.default_rng(seed)
    out = dict(profile)
    v = np.array(profile["v_lead"], dtype=float, copy=True)
    x = np.array(profile["x_lead"], dtype=float, copy=True)

    if noise_std_v > 0:
        v += rng.normal(0.0, noise_std_v, size=v.shape)
        v = np.maximum(v, 0.0)

    if noise_std_x > 0:
        x += rng.normal(0.0, noise_std_x, size=x.shape)

    out["v_lead"] = v
    out["x_lead"] = x
    out["a_lead"] = _moving_average(np.gradient(v, profile["dt"]), window=5)
    return out


def apply_sensor_delay(profile, delay_s=0.0):
    """Delay perceived lead profile by delay_s seconds."""
    if delay_s <= 0:
        return profile

    out = dict(profile)
    dt = float(profile["dt"])
    delay_steps = int(round(delay_s / dt))

    def delay_array(x):
        x = np.asarray(x, dtype=float)
        if delay_steps <= 0:
            return x.copy()
        return np.concatenate([np.full(delay_steps, x[0]), x[:-delay_steps]])

    out["v_lead"] = delay_array(profile["v_lead"])
    out["a_lead"] = delay_array(profile["a_lead"])
    out["x_lead"] = delay_array(profile["x_lead"])
    return out


def make_wltc_scenario(
    csv_path="wltc_class3b.csv",
    phase="full",
    dt=1 / 60,
    name=None,
    initial_gap=60.0,
    ego_initial_speed=None,
    theta_angle=0,
    vehicle_mass=1500,
    rolling_resistance=0.06,
    noise_std_v=0.0,
    noise_std_x=0.0,
    delay_s=0.0,
    speed_scale=1.0,
    seed=None,
):
    """Create one scenario dict compatible with acc_single_scenario(...)."""
    profile = make_wltc_lead_profile(
        csv_path=csv_path,
        dt=dt,
        phase=phase,
        speed_scale=speed_scale,
    )

    if noise_std_v > 0 or noise_std_x > 0:
        profile = apply_sensor_noise(profile, noise_std_v=noise_std_v, noise_std_x=noise_std_x, seed=seed)

    if delay_s > 0:
        profile = apply_sensor_delay(profile, delay_s=delay_s)

    if ego_initial_speed is None:
        ego_initial_speed = float(profile["v_lead"][0])

    if name is None:
        name = f"wltc_{phase}"

    return {
        "name": name,
        "lead_profile": profile,
        "initial_gap": float(initial_gap),
        "ego_initial_speed": float(ego_initial_speed),
        "theta_angle": theta_angle,
        "vehicle_mass": vehicle_mass,
        "rolling_resistance": rolling_resistance,
        "noise_std_v": noise_std_v,
        "noise_std_x": noise_std_x,
        "delay_s": delay_s,
    }


def build_wltc_scenarios(csv_path="wltc_class3b.csv", dt=1 / 60, mode="train"):
    """
    Build WLTC scenario bundle.

    mode="train": smaller bundle for optimization.
    mode="test":  larger/held-out bundle for validation.
    """
    mode = str(mode).lower().strip()

    if mode == "train":
        return [
            make_wltc_scenario(csv_path, phase="city", dt=dt, name="train_city_nominal", initial_gap=35.0),
            make_wltc_scenario(csv_path, phase="high", dt=dt, name="train_high_nominal", initial_gap=50.0),
            make_wltc_scenario(
                csv_path,
                phase="medium",
                dt=dt,
                name="train_medium_noise_delay",
                initial_gap=35.0,
                noise_std_v=0.15,
                noise_std_x=0.30,
                delay_s=0.10,
                seed=101,
            ),
        ]

    if mode == "test":
        return [
            make_wltc_scenario(csv_path, phase="full", dt=dt, name="test_full_nominal", initial_gap=45.0),
            make_wltc_scenario(
                csv_path,
                phase="city",
                dt=dt,
                name="test_city_noise_delay",
                initial_gap=30.0,
                noise_std_v=0.25,
                noise_std_x=0.50,
                delay_s=0.20,
                seed=202,
            ),
            make_wltc_scenario(
                csv_path,
                phase="high",
                dt=dt,
                name="test_high_heavy_vehicle",
                initial_gap=55.0,
                vehicle_mass=1500 * 1.15,
                rolling_resistance=0.06,
            ),
            make_wltc_scenario(
                csv_path,
                phase="extra_high",
                dt=dt,
                name="test_extra_high_roll_variation",
                initial_gap=65.0,
                vehicle_mass=1500,
                rolling_resistance=0.06 * 1.20,
            ),
        ]

    raise ValueError("mode must be 'train' or 'test'.")


def use_wltc_scenarios(csv_path="wltc_class3b.csv", dt=1 / 60, mode="train"):
    """
    Replace global SCENARIOS with WLTC scenarios.

    Example:
        from sisken_prastiyanto_wltc import use_wltc_scenarios
        use_wltc_scenarios("data/wltc_class3b.csv", dt=1/60, mode="train")
    """
    global SCENARIOS
    SCENARIOS = build_wltc_scenarios(csv_path, dt=dt, mode=mode)
    return SCENARIOS


# ============================================================
# ACC simulation
# ============================================================

def compute_ttc(d_actual, v_ego, v_lead, eps=1e-9):
    if d_actual <= 0:
        return 0.0
    dv = v_ego - v_lead
    if dv <= eps:
        return math.inf
    return d_actual / dv


def error_synthesizer(derr, verr, c_nrg, time, obj_fun):
    w1_pos = 20.6537
    w2_vel = 1.74043
    w3_nrg = 0.074934

    try:
        J = obj_fun(w1_pos, derr, time) + obj_fun(w2_vel, verr, time)
        # If you want to activate energy term:
        # J += obj_fun(w3_nrg, c_nrg, time)
    except TypeError:
        J = obj_fun(w1_pos, derr) + obj_fun(w2_vel, verr)
        # If you want to activate energy term:
        # J += obj_fun(w3_nrg, c_nrg)
    return J


def lean_simulate_system(
    params,
    obj_fun,
    return_history=False,
    ttc_threshold=1.5,
    sim_time=60,
    theta_angle=0,
    vehicle_mass=1500,
    rolling_resistance=0.06,
    lead_profile=None,
    initial_gap=60.0,
    ego_initial_speed=None,
    vset=25.0,
    ddef=20.0,
    Tg=1.2,
):
    """
    Simulate ACC.

    Legacy mode:
        lead_profile is None -> original sine-wave lead car update is used.

    WLTC mode:
        lead_profile is dict from make_wltc_lead_profile(...).
        The lead vehicle speed/position/acceleration are read from WLTC arrays.
    """
    physics = VehiclePhysics(theta=theta_angle, M=vehicle_mass, Croll=rolling_resistance)

    Kve, Kvrel, Kde = params

    # ------------------------------------------------------------
    # Timing and initial condition
    # ------------------------------------------------------------
    if lead_profile is None:
        Ts = 1 / 60
        simulation_steps = int(sim_time / Ts)
        t_array = None

        xego = 0.0
        vego = 22.0 if ego_initial_speed is None else float(ego_initial_speed)
        aego = 0.0

        xlead = float(initial_gap)
        vlead = 22.0
        alead = 0.0
        theta = 0.0
    else:
        Ts = float(lead_profile["dt"])
        t_array = np.asarray(lead_profile["t"], dtype=float)
        simulation_steps = len(t_array)

        v_lead_profile = np.asarray(lead_profile["v_lead"], dtype=float)
        a_lead_profile = np.asarray(lead_profile["a_lead"], dtype=float)
        x_lead_profile = np.asarray(lead_profile["x_lead"], dtype=float)

        xego = 0.0
        vego = float(v_lead_profile[0]) if ego_initial_speed is None else float(ego_initial_speed)
        aego = 0.0

        # WLTC x_lead starts from 0. Add initial_gap so the lead starts ahead of ego.
        xlead = float(initial_gap) + float(x_lead_profile[0])
        vlead = float(v_lead_profile[0])
        alead = float(a_lead_profile[0])
        theta = 0.0

    obj_val = 0.0
    TET = 0.0
    TIT = 0.0
    ttc_min = math.inf
    collision = False
    sat_count = 0

    prev_aego = aego

    history = {
        "time": [],
        "v_ego": [],
        "v_lead": [],
        "x_ego": [],
        "x_lead": [],
        "car_force": [],
        "dist": [],
        "a_ego": [],
        "a_lead": [],
        "jerk": [],
        "u": [],
        "err": [],
        "ttc": [],
        "obj_val": [],
        "ttc_min": [],
        "TET": [],
        "TIT": [],
        "d_safe": [],
        "collision": False,
        "saturation_ratio": 0.0,
    }

    # ------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------
    for i in range(simulation_steps):
        if lead_profile is None:
            t = i * Ts
        else:
            t = float(t_array[i])
            vlead = float(v_lead_profile[i])
            alead = float(a_lead_profile[i])
            xlead = float(initial_gap) + float(x_lead_profile[i])

        # Compute ACC errors
        dactual = xlead - xego
        dsafe = ddef + Tg * vego

        vrel = vlead - vego
        derr = dsafe - dactual
        verr = vset - vego

        # Controller from original script
        u_v = Kve * verr
        u_x = vrel * Kvrel + Kde * derr
        u = min(u_x, u_v)

        # Update ego car using existing physics model
        vego, aego, Ft = physics.step(vego, u, Ts)
        xego += vego * Ts

        # Legacy lead vehicle update only when WLTC is not supplied
        if lead_profile is None:
            theta = (theta + 30 * Ts) % 360
            alead = sin_angle(theta)
            vlead += alead * Ts
            xlead += vlead * Ts

        c_nrg = max(0.0, Ft)
        err = error_synthesizer(derr, verr, c_nrg, t, obj_fun)
        obj_val += err

        # Safety metrics
        ttc = compute_ttc(dactual, vego, vlead)
        if ttc < ttc_min:
            ttc_min = ttc
        if ttc < ttc_threshold:
            TET += Ts
            TIT += (ttc_threshold - ttc) * Ts
        if dactual <= 0:
            collision = True

        jerk = (aego - prev_aego) / Ts
        prev_aego = aego

        # Generic saturation indicator. Adjust threshold to match your physics model.
        if abs(u) > 1e6:
            sat_count += 1

        if return_history:
            history["time"].append(t)
            history["v_ego"].append(vego)
            history["v_lead"].append(vlead)
            history["x_ego"].append(xego)
            history["x_lead"].append(xlead)
            history["car_force"].append(Ft)
            history["dist"].append(dactual)
            history["d_safe"].append(dsafe)
            history["a_ego"].append(aego)
            history["a_lead"].append(alead)
            history["jerk"].append(jerk)
            history["u"].append(u)
            history["err"].append(err)
            history["ttc"].append(ttc)
            history["obj_val"].append(obj_val)

    history["TET"] = TET
    history["TIT"] = TIT
    history["ttc_min"] = ttc_min if ttc_min != math.inf else None
    history["collision"] = collision
    history["saturation_ratio"] = sat_count / max(simulation_steps, 1)

    return [obj_val, history] if return_history else obj_val


def acc_single_scenario(params, obj_function, scenario):
    return lean_simulate_system(
        params,
        obj_function,
        theta_angle=scenario.get("theta_angle", 0),
        vehicle_mass=scenario.get("vehicle_mass", 1500),
        rolling_resistance=scenario.get("rolling_resistance", 0.06),
        lead_profile=scenario.get("lead_profile", None),
        initial_gap=scenario.get("initial_gap", 60.0),
        ego_initial_speed=scenario.get("ego_initial_speed", None),
        sim_time=scenario.get("sim_time", 60),
    )


def acc_single_scenario_history(params, obj_function, scenario):
    """Convenience function for plotting/debugging one scenario."""
    return lean_simulate_system(
        params,
        obj_function,
        return_history=True,
        theta_angle=scenario.get("theta_angle", 0),
        vehicle_mass=scenario.get("vehicle_mass", 1500),
        rolling_resistance=scenario.get("rolling_resistance", 0.06),
        lead_profile=scenario.get("lead_profile", None),
        initial_gap=scenario.get("initial_gap", 60.0),
        ego_initial_speed=scenario.get("ego_initial_speed", None),
        sim_time=scenario.get("sim_time", 60),
    )


def acc_fitness_func(params, obj_function):
    """Fitness over all active scenarios."""
    with ThreadPoolExecutor(max_workers=min(4, len(SCENARIOS))) as pool:
        futures = [
            pool.submit(acc_single_scenario, params, obj_function, scen)
            for scen in SCENARIOS
        ]
        return sum(f.result() for f in futures)


def acc_constraint_evaluator(params, obj_function=itse, ttc_threshold=1.5):
    """
    Optional constraint evaluator for SA-DA-RIME.

    Returns a dict compatible with the draft SA-DA-RIME constraint_evaluator interface.
    """
    total_violation = 0.0
    collision_any = False

    for scen in SCENARIOS:
        _, hist = acc_single_scenario_history(params, obj_function, scen)

        collision = bool(hist.get("collision", False))
        collision_any = collision_any or collision

        min_ttc = hist.get("ttc_min", None)
        if min_ttc is None or not np.isfinite(min_ttc):
            min_ttc = math.inf

        distances = np.asarray(hist.get("dist", []), dtype=float)
        d_safe = np.asarray(hist.get("d_safe", []), dtype=float)
        if len(distances) and len(d_safe):
            min_gap_margin = float(np.min(distances - d_safe))
        else:
            min_gap_margin = 0.0

        violation = 0.0
        if collision:
            violation += 1000.0
        if min_ttc < ttc_threshold:
            violation += ttc_threshold - min_ttc
        if min_gap_margin < 0:
            violation += abs(min_gap_margin)

        total_violation += violation

    return {
        "feasible": total_violation <= 0.0,
        "safety_violation": float(total_violation),
        "collision": collision_any,
    }


BENCHMARKS_ACC = {
    "acc": acc_fitness_func,
}
