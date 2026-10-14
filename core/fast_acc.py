"""Validated Numba execution backend for the ACC nominal objective.

IMPORTANT RESEARCH DESIGN NOTE
------------------------------
This module is an execution backend only.  The readable/reference scientific
implementation remains ``benchmarks.sisken_prastiyanto.lean_simulate_system``.
The functions below mirror the same equations and state-update order so the
optimizer can evaluate candidates without paying Python interpreter overhead at
every 60 Hz simulation step.

Only the P0 numerical hot path is JIT-compiled here.  Optimizer internals,
checkpointing, progress reporting, statistics, and post-processing are left
unchanged.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


# Objective identifiers used at the Python -> Numba boundary.
OBJECTIVE_IAE = 0
OBJECTIVE_ISE = 1
OBJECTIVE_ITAE = 2
OBJECTIVE_ITSE = 3




# ---------------------------------------------------------------------------
# Deterministic sensor-noise stream
# Mirrors core.sensing.deterministic_standard_normal() exactly so noisy sensor
# studies remain numerically aligned between the Python and Numba simulators.
# Noise is indexed by sample number rather than shared mutable RNG state.
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False, inline="always")
def _sensor_hash_u64_jit(seed, sample_index, stream):
    gamma = np.uint64(0x9E3779B97F4A7C15)
    mul1 = np.uint64(0xBF58476D1CE4E5B9)
    mul2 = np.uint64(0x94D049BB133111EB)
    salt = np.uint64(0xD1B54A32D192ED03)

    x = (
        np.uint64(seed)
        ^ (np.uint64(sample_index + 1) * gamma)
        ^ (np.uint64(stream + 1) * salt)
    )
    z = x + gamma
    z = (z ^ (z >> np.uint64(30))) * mul1
    z = (z ^ (z >> np.uint64(27))) * mul2
    z = z ^ (z >> np.uint64(31))
    return z


@njit(cache=True, fastmath=False, inline="always")
def deterministic_standard_normal_jit(seed, sample_index):
    inv_2_pow_53 = 1.0 / 9007199254740992.0
    h1 = _sensor_hash_u64_jit(seed, sample_index, 0)
    h2 = _sensor_hash_u64_jit(seed, sample_index, 1)
    u1 = (float(h1 >> np.uint64(11)) + 0.5) * inv_2_pow_53
    u2 = (float(h2 >> np.uint64(11)) + 0.5) * inv_2_pow_53
    if u1 < 1e-15:
        u1 = 1e-15
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


# ---------------------------------------------------------------------------
# NUMBA-P0: controller math
# This is the compiled mirror of pd_acc_control().  It is intentionally kept as
# a small named function so the equations remain easy to map to the manuscript.
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False, inline="always")
def pd_acc_control_jit(
    k_v,
    k_p,
    k_d,
    v_ref,
    v_ego,
    d_sensed,
    v_rel_est,
    d_def,
    time_gap,
):
    d_safe = d_def + time_gap * v_ego
    speed_error = v_ref - v_ego
    gap_error = d_sensed - d_safe

    a_speed = k_v * speed_error
    a_follow = k_p * gap_error + k_d * v_rel_est
    a_cmd = a_speed if a_speed < a_follow else a_follow

    return a_cmd, d_safe, speed_error, gap_error


# ---------------------------------------------------------------------------
# NUMBA-P0: normalized IAE / ISE / ITAE / ITSE instantaneous terms
# Python callables are deliberately replaced by an integer objective id inside
# the compiled kernel; the mathematical definitions are unchanged.
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False, inline="always")
def normalized_objective_term_jit(
    spacing_error,
    speed_error,
    time,
    objective_id,
    spacing_scale,
    speed_scale,
):
    e_d = spacing_error / spacing_scale
    e_v = speed_error / speed_scale

    if objective_id == OBJECTIVE_IAE:
        return abs(e_d) + abs(e_v)
    if objective_id == OBJECTIVE_ISE:
        return e_d * e_d + e_v * e_v
    if objective_id == OBJECTIVE_ITAE:
        return time * (abs(e_d) + abs(e_v))
    if objective_id == OBJECTIVE_ITSE:
        return time * (e_d * e_d + e_v * e_v)

    # Fail loudly rather than silently evaluating a different objective.
    raise ValueError("Unknown ACC objective id")


# ---------------------------------------------------------------------------
# NUMBA-P0: vehicle-force physics helpers
# These functions mirror VehiclePhysics.aerodynamic_drag(),
# rolling_resistance(), gravitational_force(), and
# step_acceleration_command().  No alternative physics model is introduced.
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False, inline="always")
def aerodynamic_drag_jit(v, rho, cd, frontal_area):
    return 0.5 * rho * cd * frontal_area * v * v


@njit(cache=True, fastmath=False, inline="always")
def rolling_resistance_jit(mass, g, c_roll, theta_rad):
    return c_roll * mass * g * math.cos(theta_rad)


@njit(cache=True, fastmath=False, inline="always")
def gravitational_force_jit(mass, g, theta_rad):
    return mass * g * math.sin(theta_rad)


@njit(cache=True, fastmath=False, inline="always")
def vehicle_step_acceleration_command_jit(
    v,
    a_cmd,
    dt,
    mass,
    rho,
    cd,
    frontal_area,
    c_roll,
    g,
    theta_rad,
    f_max,
):
    fa = aerodynamic_drag_jit(v, rho, cd, frontal_area)
    frr = rolling_resistance_jit(mass, g, c_roll, theta_rad)
    fg = gravitational_force_jit(mass, g, theta_rad)
    f_resist = fa + frr + fg

    ft_required = mass * a_cmd + f_resist
    if ft_required > f_max:
        ft = f_max
    elif ft_required < -f_max:
        ft = -f_max
    else:
        ft = ft_required

    acceleration = (ft - f_resist) / mass
    v_new = v + acceleration * dt
    if v_new < 0.0:
        v_new = 0.0

    return v_new, acceleration, ft


# ---------------------------------------------------------------------------
# NUMBA-P0: complete sequential nominal ACC simulation loop
# This is the primary hot path.  The timestep recurrence stays sequential.
# Numba removes Python interpreter overhead; it does NOT parallelize timesteps,
# memoize fitness values, reduce the 60 Hz sample rate, or shorten the horizon.
#
# Range sensing is kept as scalar state inside this kernel rather than compiling
# the Python RangeOnlySensor dataclass.  The equations and first-sample behavior
# are the same: range is clamped at zero and the first range-rate sample is 0.
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def acc_objective_kernel_jit(
    k_v,
    k_p,
    k_d,
    objective_id,
    sim_time,
    theta_angle_deg,
    vehicle_mass,
    rolling_resistance,
    sensor_noise_std_m,
    sensor_bias_m,
    sensor_seed,
    sensor_min_distance_m,
):
    # Frozen reference/controller constants from the active v3 benchmark.
    v_ref = 25.0
    d_def = 20.0
    time_gap = 1.2
    speed_error_scale = 25.0
    spacing_error_scale = 50.0
    dt = 1.0 / 60.0

    # VehiclePhysics defaults used by the reference implementation.
    cd = 0.36
    frontal_area = 2.42
    rho = 1.225
    g = 9.81
    f_max = 6000.0
    theta_rad = theta_angle_deg * (math.pi / 180.0)

    # Initial world state matches lean_simulate_system().
    phase_deg = 0.0
    x_ego = 0.0
    v_ego = 22.0
    x_lead = 60.0
    v_lead = 22.0

    previous_sensed_distance = 0.0
    has_previous_distance = False
    objective_value = 0.0
    simulation_steps = int(sim_time / dt)

    for i in range(simulation_steps):
        time = i * dt

        # NUMBA-P0 sensing: explicit range-sensor boundary. The nominal
        # benchmark passes noise_std=0 and bias=0; robustness studies can enable
        # deterministic seeded Gaussian noise without changing the controller.
        distance_true = x_lead - x_ego
        sensor_noise = 0.0
        if sensor_noise_std_m > 0.0:
            sensor_noise = sensor_noise_std_m * deterministic_standard_normal_jit(
                sensor_seed, i
            )
        distance_raw = distance_true + sensor_bias_m + sensor_noise
        distance_sensed = (
            distance_raw
            if distance_raw > sensor_min_distance_m
            else sensor_min_distance_m
        )

        if has_previous_distance:
            v_rel_est = (distance_sensed - previous_sensed_distance) / dt
        else:
            v_rel_est = 0.0
            has_previous_distance = True
        previous_sensed_distance = distance_sensed

        # NUMBA-P0 controller: same CTG + PD acceleration-command equations.
        a_cmd, d_safe, speed_error, _gap_error = pd_acc_control_jit(
            k_v,
            k_p,
            k_d,
            v_ref,
            v_ego,
            distance_sensed,
            v_rel_est,
            d_def,
            time_gap,
        )

        # NUMBA-P0 objective: same normalized error and rectangle-rule integration.
        spacing_error = distance_true - d_safe
        objective_value += normalized_objective_term_jit(
            spacing_error,
            speed_error,
            time,
            objective_id,
            spacing_error_scale,
            speed_error_scale,
        ) * dt

        # NUMBA-P0 plant: same acceleration-to-force conversion and saturation.
        v_ego, _a_ego, _traction_force = vehicle_step_acceleration_command_jit(
            v_ego,
            a_cmd,
            dt,
            vehicle_mass,
            rho,
            cd,
            frontal_area,
            rolling_resistance,
            g,
            theta_rad,
            f_max,
        )
        x_ego += v_ego * dt

        # NUMBA-P0 lead trajectory: same sequential sinusoidal acceleration profile.
        phase_deg = (phase_deg + 30.0 * dt) % 360.0
        a_lead = math.sin(phase_deg * (math.pi / 180.0))
        v_lead = v_lead + a_lead * dt
        if v_lead < 0.0:
            v_lead = 0.0
        x_lead += v_lead * dt

    return objective_value


# ---------------------------------------------------------------------------
# Python-facing adapter.  This small wrapper is intentionally NOT jitted: it
# validates/converts the three-gain vector once, then calls the P0 compiled
# numerical kernel.  All expensive timestep work stays inside Numba.
# ---------------------------------------------------------------------------
def fast_acc_objective(
    params,
    objective_id,
    sim_time=60.0,
    theta_angle=0.0,
    vehicle_mass=1500.0,
    rolling_resistance=0.015,
    sensor_noise_std_m=0.0,
    sensor_bias_m=0.0,
    sensor_seed=0,
    sensor_min_distance_m=0.0,
):
    if len(params) != 3:
        raise ValueError("ACC decision vector must contain exactly [k_v, k_p, k_d]")
    k_v = float(params[0])
    k_p = float(params[1])
    k_d = float(params[2])
    sensor_noise_std_m = float(sensor_noise_std_m)
    sensor_bias_m = float(sensor_bias_m)
    sensor_min_distance_m = float(sensor_min_distance_m)
    sensor_seed = int(sensor_seed)
    if sensor_noise_std_m < 0.0 or not math.isfinite(sensor_noise_std_m):
        raise ValueError("sensor_noise_std_m must be finite and nonnegative")
    if not math.isfinite(sensor_bias_m):
        raise ValueError("sensor_bias_m must be finite")
    if not math.isfinite(sensor_min_distance_m):
        raise ValueError("sensor_min_distance_m must be finite")
    return float(
        acc_objective_kernel_jit(
            k_v,
            k_p,
            k_d,
            int(objective_id),
            float(sim_time),
            float(theta_angle),
            float(vehicle_mass),
            float(rolling_resistance),
            sensor_noise_std_m,
            sensor_bias_m,
            sensor_seed,
            sensor_min_distance_m,
        )
    )

# ---------------------------------------------------------------------------
# Compiled downstream-validation helpers.
# These use the same equations as the optimizer-facing kernel but return only
# the evidence needed by the analysis notebook. Keeping batch robustness work
# here avoids hundreds of slow Python timestep loops without changing the model.
# ---------------------------------------------------------------------------
@njit(cache=True, fastmath=False)
def acc_validation_metrics_kernel_jit(
    k_v,
    k_p,
    k_d,
    objective_id,
    sim_time,
    theta_angle_deg,
    vehicle_mass,
    rolling_resistance,
    sensor_noise_std_m,
    sensor_bias_m,
    sensor_seed,
    sensor_min_distance_m,
):
    v_ref = 25.0
    d_def = 20.0
    time_gap = 1.2
    speed_error_scale = 25.0
    spacing_error_scale = 50.0
    dt = 1.0 / 60.0

    cd = 0.36
    frontal_area = 2.42
    rho = 1.225
    g = 9.81
    f_max = 6000.0
    theta_rad = theta_angle_deg * (math.pi / 180.0)

    phase_deg = 0.0
    x_ego = 0.0
    v_ego = 22.0
    a_ego = 0.0
    x_lead = 60.0
    v_lead = 22.0

    previous_sensed_distance = 0.0
    has_previous_distance = False
    previous_recorded_acceleration = 0.0
    has_previous_acceleration = False

    objective_value = 0.0
    min_distance = math.inf
    min_ttc = math.inf
    max_abs_jerk = 0.0
    simulation_steps = int(sim_time / dt)

    for i in range(simulation_steps):
        time = i * dt
        distance_true = x_lead - x_ego
        if distance_true < min_distance:
            min_distance = distance_true

        closing_speed = v_ego - v_lead
        if closing_speed > 0.0:
            ttc = distance_true / closing_speed
            if ttc < min_ttc:
                min_ttc = ttc

        if has_previous_acceleration:
            jerk = abs((a_ego - previous_recorded_acceleration) / dt)
            if jerk > max_abs_jerk:
                max_abs_jerk = jerk
        else:
            has_previous_acceleration = True
        previous_recorded_acceleration = a_ego

        sensor_noise = 0.0
        if sensor_noise_std_m > 0.0:
            sensor_noise = sensor_noise_std_m * deterministic_standard_normal_jit(
                sensor_seed, i
            )
        distance_raw = distance_true + sensor_bias_m + sensor_noise
        distance_sensed = (
            distance_raw
            if distance_raw > sensor_min_distance_m
            else sensor_min_distance_m
        )

        if has_previous_distance:
            v_rel_est = (distance_sensed - previous_sensed_distance) / dt
        else:
            v_rel_est = 0.0
            has_previous_distance = True
        previous_sensed_distance = distance_sensed

        a_cmd, d_safe, speed_error, _gap_error = pd_acc_control_jit(
            k_v,
            k_p,
            k_d,
            v_ref,
            v_ego,
            distance_sensed,
            v_rel_est,
            d_def,
            time_gap,
        )

        spacing_error = distance_true - d_safe
        objective_value += normalized_objective_term_jit(
            spacing_error,
            speed_error,
            time,
            objective_id,
            spacing_error_scale,
            speed_error_scale,
        ) * dt

        v_ego, a_ego, _traction_force = vehicle_step_acceleration_command_jit(
            v_ego,
            a_cmd,
            dt,
            vehicle_mass,
            rho,
            cd,
            frontal_area,
            rolling_resistance,
            g,
            theta_rad,
            f_max,
        )
        x_ego += v_ego * dt

        phase_deg = (phase_deg + 30.0 * dt) % 360.0
        a_lead = math.sin(phase_deg * (math.pi / 180.0))
        v_lead = v_lead + a_lead * dt
        if v_lead < 0.0:
            v_lead = 0.0
        x_lead += v_lead * dt

    return objective_value, min_ttc, max_abs_jerk, min_distance


@njit(cache=True, fastmath=False)
def cruise_history_kernel_jit(
    k_v,
    sim_time,
    theta_angle_deg,
    vehicle_mass,
    rolling_resistance,
    v_ref,
    v_ego_init,
):
    dt = 1.0 / 60.0
    cd = 0.36
    frontal_area = 2.42
    rho = 1.225
    g = 9.81
    f_max = 6000.0
    theta_rad = theta_angle_deg * (math.pi / 180.0)
    steps = int(sim_time / dt)

    time_hist = np.empty(steps, dtype=np.float64)
    speed_hist = np.empty(steps, dtype=np.float64)
    ref_hist = np.empty(steps, dtype=np.float64)
    command_hist = np.empty(steps, dtype=np.float64)
    acceleration_hist = np.empty(steps, dtype=np.float64)
    force_hist = np.empty(steps, dtype=np.float64)

    v_ego = v_ego_init
    a_ego = 0.0
    for i in range(steps):
        time = i * dt
        a_cmd = k_v * (v_ref - v_ego)

        time_hist[i] = time
        speed_hist[i] = v_ego
        ref_hist[i] = v_ref
        command_hist[i] = a_cmd
        acceleration_hist[i] = a_ego

        v_ego, a_ego, force = vehicle_step_acceleration_command_jit(
            v_ego,
            a_cmd,
            dt,
            vehicle_mass,
            rho,
            cd,
            frontal_area,
            rolling_resistance,
            g,
            theta_rad,
            f_max,
        )
        force_hist[i] = force

    return (
        time_hist,
        speed_hist,
        ref_hist,
        command_hist,
        acceleration_hist,
        force_hist,
    )


def fast_acc_validation_metrics(
    params,
    objective_id,
    sim_time=60.0,
    theta_angle=0.0,
    vehicle_mass=1500.0,
    rolling_resistance=0.015,
    sensor_noise_std_m=0.0,
    sensor_bias_m=0.0,
    sensor_seed=0,
    sensor_min_distance_m=0.0,
):
    if len(params) != 3:
        raise ValueError("ACC decision vector must contain exactly [k_v, k_p, k_d]")
    return tuple(
        float(x)
        for x in acc_validation_metrics_kernel_jit(
            float(params[0]),
            float(params[1]),
            float(params[2]),
            int(objective_id),
            float(sim_time),
            float(theta_angle),
            float(vehicle_mass),
            float(rolling_resistance),
            float(sensor_noise_std_m),
            float(sensor_bias_m),
            int(sensor_seed),
            float(sensor_min_distance_m),
        )
    )


def fast_cruise_history(
    params,
    sim_time=60.0,
    theta_angle=0.0,
    vehicle_mass=1500.0,
    rolling_resistance=0.015,
    v_ref=25.0,
    v_ego_init=22.0,
):
    if len(params) != 3:
        raise ValueError("ACC decision vector must contain exactly [k_v, k_p, k_d]")
    arrays = cruise_history_kernel_jit(
        float(params[0]),
        float(sim_time),
        float(theta_angle),
        float(vehicle_mass),
        float(rolling_resistance),
        float(v_ref),
        float(v_ego_init),
    )
    return {
        "time": arrays[0],
        "v_ego": arrays[1],
        "v_ref": arrays[2],
        "a_cmd": arrays[3],
        "a_ego": arrays[4],
        "tractive_force": arrays[5],
    }
