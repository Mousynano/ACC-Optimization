"""Active ACC benchmark used by the optimization and analysis pipeline.

The controller receives only ego speed, sensed longitudinal range, and a
finite-difference range-rate estimate. Simulator-side lead states remain
available only to the world model and evaluator.
"""

import math

from core.config import load_json_relative
from core.fast_acc import (
    OBJECTIVE_IAE,
    OBJECTIVE_ISE,
    OBJECTIVE_ITAE,
    OBJECTIVE_ITSE,
    fast_acc_objective,
)
from core.physics import VehiclePhysics
from core.sensing import RangeOnlySensor, RangeSensorConfig
from core.utils import iae, ise, itae, itse


_MANIFEST = load_json_relative("experiment_manifest_acc_range_pd_norm_v3.json")
_CONTROLLER = _MANIFEST["controller"]
_PHYSICS = _MANIFEST["physics"]
_SENSING = _MANIFEST["sensing"]
_EXECUTION = _MANIFEST["execution_backend"]

NOMINAL_SPEED_REFERENCE = float(_CONTROLLER["speed_reference_m_per_s"])
NOMINAL_STANDSTILL_DISTANCE = float(_CONTROLLER["standstill_distance_m"])
NOMINAL_TIME_GAP = float(_CONTROLLER["time_gap_s"])
SPEED_ERROR_SCALE = NOMINAL_SPEED_REFERENCE
SPACING_ERROR_SCALE = (
    NOMINAL_STANDSTILL_DISTANCE
    + NOMINAL_TIME_GAP * NOMINAL_SPEED_REFERENCE
)
SIMULATION_HZ = int(_EXECUTION["sampling_hz"])
SIMULATION_DT = 1.0 / SIMULATION_HZ
SIMULATION_HORIZON_S = float(_EXECUTION["horizon_s"])

NOMINAL_SCENARIO = {
    "theta_angle": float(_PHYSICS["road_slope_deg"]),
    "vehicle_mass": float(_PHYSICS["nominal_mass_kg"]),
    "rolling_resistance": float(_PHYSICS["rolling_resistance"]),
    "sensor_noise_std_m": float(_SENSING["nominal_noise_std_m"]),
    "sensor_bias_m": float(_SENSING["nominal_bias_m"]),
    "sensor_seed": int(_SENSING["nominal_noise_seed"]),
    "sensor_min_distance_m": float(_SENSING["nominal_min_distance_m"]),
}


def pd_acc_control(
    k_v,
    k_p,
    k_d,
    v_ref,
    v_ego,
    d_sensed,
    v_rel_est,
    d_def=NOMINAL_STANDSTILL_DISTANCE,
    time_gap=NOMINAL_TIME_GAP,
):
    """CTG speed/spacing controller that outputs desired acceleration."""
    d_safe = d_def + time_gap * v_ego
    speed_error = v_ref - v_ego
    gap_error = d_sensed - d_safe
    a_speed = k_v * speed_error
    a_follow = k_p * gap_error + k_d * v_rel_est
    return min(a_speed, a_follow), d_safe, speed_error, gap_error


def normalized_error_cost(spacing_error, speed_error, time, obj_fun):
    """Instantaneous normalized IAE/ISE/ITAE/ITSE contribution."""
    e_d = float(spacing_error) / SPACING_ERROR_SCALE
    e_v = float(speed_error) / SPEED_ERROR_SCALE
    if obj_fun is iae:
        return iae(e_d) + iae(e_v)
    if obj_fun is ise:
        return ise(e_d) + ise(e_v)
    if obj_fun is itae:
        return itae(e_d, time) + itae(e_v, time)
    if obj_fun is itse:
        return itse(e_d, time) + itse(e_v, time)
    raise ValueError("ACC benchmark supports only iae/ise/itae/itse")


def lean_simulate_system(
    params,
    obj_fun,
    return_history=False,
    sim_time=SIMULATION_HORIZON_S,
    theta_angle=NOMINAL_SCENARIO["theta_angle"],
    vehicle_mass=NOMINAL_SCENARIO["vehicle_mass"],
    rolling_resistance=NOMINAL_SCENARIO["rolling_resistance"],
    sensor_noise_std_m=NOMINAL_SCENARIO["sensor_noise_std_m"],
    sensor_bias_m=NOMINAL_SCENARIO["sensor_bias_m"],
    sensor_seed=NOMINAL_SCENARIO["sensor_seed"],
    sensor_min_distance_m=NOMINAL_SCENARIO["sensor_min_distance_m"],
):
    """Readable reference simulator for nominal and sensor-robustness analysis."""
    if len(params) != 3:
        raise ValueError("ACC decision vector must contain [k_v, k_p, k_d]")
    k_v, k_p, k_d = map(float, params)

    physics = VehiclePhysics(
        theta=theta_angle,
        M=vehicle_mass,
        Croll=rolling_resistance,
    )
    sensor = RangeOnlySensor(
        dt=SIMULATION_DT,
        config=RangeSensorConfig(
            noise_std_m=sensor_noise_std_m,
            bias_m=sensor_bias_m,
            min_distance_m=sensor_min_distance_m,
            seed=sensor_seed,
        ),
    )

    phase_deg = 0.0
    x_ego, v_ego, a_ego = 0.0, 22.0, 0.0
    x_lead, v_lead, a_lead = 60.0, 22.0, 0.0
    objective_value = 0.0
    steps = int(float(sim_time) / SIMULATION_DT)

    history = None
    if return_history:
        history = {
            "time": [],
            "v_ego": [],
            "v_lead": [],
            "x_ego": [],
            "x_lead": [],
            "a_ego": [],
            "a_lead": [],
            "car_force": [],
            "dist": [],
            "dist_sensed": [],
            "d_safe": [],
            "v_rel_est": [],
            "v_rel_true": [],
            "range_rate_valid": [],
            "sensor_noise": [],
            "sensor_error": [],
            "a_cmd": [],
            "spacing_error": [],
            "speed_error": [],
            "err": [],
        }

    for i in range(steps):
        time = i * SIMULATION_DT
        distance_true = x_lead - x_ego
        reading = sensor.sense(distance_true)

        a_cmd, d_safe, speed_error, _gap_error = pd_acc_control(
            k_v,
            k_p,
            k_d,
            NOMINAL_SPEED_REFERENCE,
            v_ego,
            reading.distance,
            reading.relative_velocity,
        )

        spacing_error = distance_true - d_safe
        instantaneous_cost = normalized_error_cost(
            spacing_error,
            speed_error,
            time,
            obj_fun,
        )
        objective_value += instantaneous_cost * SIMULATION_DT
        v_rel_true = v_lead - v_ego

        if return_history:
            history["time"].append(time)
            history["v_ego"].append(v_ego)
            history["v_lead"].append(v_lead)
            history["x_ego"].append(x_ego)
            history["x_lead"].append(x_lead)
            history["a_ego"].append(a_ego)
            history["a_lead"].append(a_lead)
            history["dist"].append(distance_true)
            history["dist_sensed"].append(reading.distance)
            history["d_safe"].append(d_safe)
            history["v_rel_est"].append(reading.relative_velocity)
            history["v_rel_true"].append(v_rel_true)
            history["range_rate_valid"].append(reading.rate_valid)
            history["sensor_noise"].append(reading.injected_noise)
            history["sensor_error"].append(reading.measurement_error)
            history["a_cmd"].append(a_cmd)
            history["spacing_error"].append(spacing_error)
            history["speed_error"].append(speed_error)
            history["err"].append(instantaneous_cost)

        v_ego, a_ego, traction_force = physics.step_acceleration_command(
            v_ego,
            a_cmd,
            SIMULATION_DT,
        )
        x_ego += v_ego * SIMULATION_DT
        if return_history:
            history["car_force"].append(traction_force)

        phase_deg = (phase_deg + 30.0 * SIMULATION_DT) % 360.0
        a_lead = math.sin(math.radians(phase_deg))
        v_lead = max(v_lead + a_lead * SIMULATION_DT, 0.0)
        x_lead += v_lead * SIMULATION_DT

    return (objective_value, history) if return_history else objective_value


def _objective_id_for_numba(obj_function):
    if obj_function is iae:
        return OBJECTIVE_IAE
    if obj_function is ise:
        return OBJECTIVE_ISE
    if obj_function is itae:
        return OBJECTIVE_ITAE
    if obj_function is itse:
        return OBJECTIVE_ITSE
    raise ValueError("ACC benchmark supports only iae/ise/itae/itse")


def acc_fitness_func(params, obj_function):
    """Compiled optimizer-facing evaluation of the active nominal scenario."""
    return fast_acc_objective(
        params,
        _objective_id_for_numba(obj_function),
        sim_time=SIMULATION_HORIZON_S,
        theta_angle=NOMINAL_SCENARIO["theta_angle"],
        vehicle_mass=NOMINAL_SCENARIO["vehicle_mass"],
        rolling_resistance=NOMINAL_SCENARIO["rolling_resistance"],
        sensor_noise_std_m=NOMINAL_SCENARIO["sensor_noise_std_m"],
        sensor_bias_m=NOMINAL_SCENARIO["sensor_bias_m"],
        sensor_seed=NOMINAL_SCENARIO["sensor_seed"],
        sensor_min_distance_m=NOMINAL_SCENARIO["sensor_min_distance_m"],
    )


BENCHMARKS_ACC = {_MANIFEST["benchmark"]: acc_fitness_func}
