"""Numba kernels for the v2 Sahib-derived ACC cost framework.

V2 evaluates two explicit ACC operating situations:
  * cruise: speed-control branch only, with event-based user set speed;
  * following: spacing/relative-velocity branch only, driven by a lead trace.

The scalar research cost is built outside this module from three metrics:
cruise speed RMSE, following spacing RMSE, and realized jerk RMS.
"""

from __future__ import annotations

import math
import numpy as np
from numba import njit

from core.fast_acc import (
    deterministic_standard_normal_jit,
    vehicle_step_acceleration_command_jit,
)


@njit(cache=True, fastmath=False, inline="always")
def _event_set_speed_jit(time_s, event_times, event_speeds):
    idx = 0
    n = event_times.size
    for i in range(1, n):
        if time_s >= event_times[i]:
            idx = i
        else:
            break
    return event_speeds[idx]


@njit(cache=True, fastmath=False)
def cruise_metrics_kernel_jit(
    k_ve,
    duration_s,
    initial_ego_speed_mps,
    event_times_s,
    set_speeds_mps,
    sampling_hz,
    theta_angle_deg,
    vehicle_mass,
    rolling_resistance,
    f_max,
):
    """Return speed MSE, jerk MSE, sample counts, feasibility, violation."""
    dt = 1.0 / sampling_hz
    theta_rad = theta_angle_deg * math.pi / 180.0
    rho = 1.225
    cd = 0.36
    frontal_area = 2.42
    g = 9.81

    v_ego = initial_ego_speed_mps
    a_ego = 0.0
    prev_acc = 0.0
    has_prev_acc = False

    speed_sq = 0.0
    speed_n = 0
    jerk_sq = 0.0
    jerk_n = 0
    violation = 0.0
    steps = int(round(duration_s * sampling_hz))

    for i in range(steps):
        time_s = i * dt
        v_set = _event_set_speed_jit(time_s, event_times_s, set_speeds_mps)
        speed_error = v_set - v_ego
        a_cmd = k_ve * speed_error

        speed_sq += speed_error * speed_error
        speed_n += 1

        v_ego, a_ego, _ = vehicle_step_acceleration_command_jit(
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

        if not math.isfinite(v_ego) or not math.isfinite(a_ego):
            return 1e24, 1e24, speed_n, max(jerk_n, 1), False, 1e12

        if has_prev_acc:
            jerk = (a_ego - prev_acc) / dt
            jerk_sq += jerk * jerk
            jerk_n += 1
        else:
            has_prev_acc = True
        prev_acc = a_ego

    if speed_n <= 0 or jerk_n <= 0:
        return 1e24, 1e24, max(speed_n, 1), max(jerk_n, 1), False, 1e9
    return speed_sq / speed_n, jerk_sq / jerk_n, speed_n, jerk_n, True, violation


@njit(cache=True, fastmath=False)
def following_metrics_kernel_jit(
    k_de,
    k_vrel,
    initial_ego_speed_mps,
    initial_lead_speed_mps,
    initial_gap_m,
    lead_speed_mps,
    sampling_hz,
    standstill_distance_m,
    time_gap_s,
    theta_angle_deg,
    vehicle_mass,
    rolling_resistance,
    f_max,
    sensor_noise_std_m,
    sensor_bias_m,
    sensor_seed,
    sensor_min_distance_m,
):
    """Return spacing MSE, jerk MSE, counts, feasibility, violation.

    The following branch uses sensed spacing and finite-difference sensed range
    rate. Objective spacing error uses simulator ground-truth separation.
    """
    dt = 1.0 / sampling_hz
    theta_rad = theta_angle_deg * math.pi / 180.0
    rho = 1.225
    cd = 0.36
    frontal_area = 2.42
    g = 9.81

    x_ego = 0.0
    x_lead = initial_gap_m
    v_ego = initial_ego_speed_mps
    v_lead = initial_lead_speed_mps
    a_ego = 0.0

    prev_sensed = 0.0
    has_prev_sensed = False
    prev_acc = 0.0
    has_prev_acc = False

    spacing_sq = 0.0
    spacing_n = 0
    jerk_sq = 0.0
    jerk_n = 0
    violation = 0.0

    steps = lead_speed_mps.size
    for i in range(steps):
        distance_true = x_lead - x_ego
        if not math.isfinite(distance_true) or distance_true <= 0.0:
            collision_depth = 1.0 + max(-distance_true, 0.0)
            return 1e24, 1e24, max(spacing_n, 1), max(jerk_n, 1), False, collision_depth

        noise = 0.0
        if sensor_noise_std_m > 0.0:
            noise = sensor_noise_std_m * deterministic_standard_normal_jit(sensor_seed, i)
        raw = distance_true + sensor_bias_m + noise
        sensed = raw if raw > sensor_min_distance_m else sensor_min_distance_m

        if has_prev_sensed:
            v_rel_est = (sensed - prev_sensed) / dt
        else:
            v_rel_est = 0.0
            has_prev_sensed = True
        prev_sensed = sensed

        d_safe = standstill_distance_m + time_gap_s * v_ego
        gap_error_for_control = sensed - d_safe
        a_cmd = k_de * gap_error_for_control + k_vrel * v_rel_est

        spacing_error = distance_true - d_safe
        spacing_sq += spacing_error * spacing_error
        spacing_n += 1

        v_ego, a_ego, _ = vehicle_step_acceleration_command_jit(
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
        if not math.isfinite(v_ego) or not math.isfinite(a_ego):
            return 1e24, 1e24, spacing_n, max(jerk_n, 1), False, 1e12

        if has_prev_acc:
            jerk = (a_ego - prev_acc) / dt
            jerk_sq += jerk * jerk
            jerk_n += 1
        else:
            has_prev_acc = True
        prev_acc = a_ego

        # Lead speed is an exogenous trace. Integrate position with the supplied
        # sample speed so calibration does not infer privileged lead acceleration.
        v_lead = lead_speed_mps[i]
        if v_lead < 0.0 or not math.isfinite(v_lead):
            return 1e24, 1e24, spacing_n, max(jerk_n, 1), False, 1e12
        x_ego += v_ego * dt
        x_lead += v_lead * dt

    if spacing_n <= 0 or jerk_n <= 0:
        return 1e24, 1e24, max(spacing_n, 1), max(jerk_n, 1), False, 1e9
    return spacing_sq / spacing_n, jerk_sq / jerk_n, spacing_n, jerk_n, True, violation
