"""Framework V2.2-FJE ACC benchmark.

Active performance vector:
    [cruise speed RMSE,
     following CTG-spacing RMSE,
     realized longitudinal jerk RMS,
     positive tractive-energy intensity (Wh/km)]

The first three terms are unchanged from Framework V2.1.  The fourth term is
wheel-level and powertrain-agnostic.  It integrates only positive tractive
power; braking receives no energy credit.  No fuel/battery claim, post-hoc
denoising, acceleration clamp, or jerk clamp is introduced by this evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np

from core.config import load_manifest, resolve_project_path
from core.fast_acc_energy import (
    cruise_metrics_energy_kernel_jit,
    following_metrics_energy_kernel_jit,
)
from core.scenarios_v2 import ScenarioSuite, CruiseScenario, FollowingScenario, load_scenario_suite
from core.physics import VehiclePhysics
from core.sensing import RangeOnlySensor, RangeSensorConfig


_MANIFEST = load_manifest()
_CONTROLLER = _MANIFEST["controller"]
_PHYSICS = _MANIFEST["physics"]
_SENSING = _MANIFEST["sensing"]
_EXECUTION = _MANIFEST["execution_backend"]
_COST = _MANIFEST["cost_framework"]

SAMPLING_HZ = int(_EXECUTION["sampling_hz"])
D0 = float(_CONTROLLER["standstill_distance_m"])
TIME_GAP = float(_CONTROLLER["time_gap_s"])
THETA = float(_PHYSICS["road_slope_deg"])
MASS = float(_PHYSICS["nominal_mass_kg"])
RHO = float(_PHYSICS.get("air_density_kg_m3", 1.225))
CD = float(_PHYSICS.get("drag_coefficient", 0.36))
AREA = float(_PHYSICS.get("frontal_area_m2", 2.42))
CROLL = float(_PHYSICS["rolling_resistance"])
GRAVITY = float(_PHYSICS.get("gravity_m_per_s2", 9.81))
F_MAX = float(_PHYSICS["max_longitudinal_force_n"])
SENSOR_NOISE_STD = float(_SENSING["nominal_noise_std_m"])
SENSOR_BIAS = float(_SENSING["nominal_bias_m"])
SENSOR_SEED = int(_SENSING["nominal_noise_seed"])
SENSOR_MIN = float(_SENSING["nominal_min_distance_m"])

PERFORMANCE_NAMES = (
    "cruise_speed_rmse",
    "following_spacing_rmse",
    "realized_jerk_rms",
    "positive_tractive_energy_wh_per_km",
)


@dataclass(frozen=True)
class PerformanceVectorFJE:
    cruise_speed_rmse: float
    following_spacing_rmse: float
    realized_jerk_rms: float
    positive_tractive_energy_wh_per_km: float
    feasible: bool
    constraint_violation: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.cruise_speed_rmse,
                self.following_spacing_rmse,
                self.realized_jerk_rms,
                self.positive_tractive_energy_wh_per_km,
            ],
            dtype=float,
        )

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in zip(PERFORMANCE_NAMES, self.as_array())}


@dataclass(frozen=True)
class CostProfile:
    weights: np.ndarray
    status: str
    scenario_manifest_sha256: str
    path: Path


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.size == 0 or weights.size != values.size:
        raise ValueError("weighted mean requires non-empty aligned values/weights")
    if np.any(weights < 0.0) or not np.isfinite(weights).all() or weights.sum() <= 0.0:
        raise ValueError("scenario weights must be finite, nonnegative, and sum > 0")
    return float(np.sum(values * weights) / np.sum(weights))


def evaluate_performance_vector(params, suite: ScenarioSuite) -> PerformanceVectorFJE:
    """Evaluate FJE on one scenario suite using the compiled 60 Hz kernels."""
    if len(params) != 3:
        raise ValueError("FJE ACC decision vector must be [k_ve, k_de, k_vrel]")
    k_ve, k_de, k_vrel = map(float, params)

    cruise_mse, cruise_w = [], []
    follow_mse, follow_w = [], []
    jerk_mse, jerk_w = [], []
    energy_whkm, energy_w = [], []
    feasible = True
    violation = 0.0

    for scenario in suite.cruise:
        s_mse, j_mse, e_whkm, _sn, _jn, ok, vio = cruise_metrics_energy_kernel_jit(
            k_ve,
            float(scenario.duration_s),
            float(scenario.initial_ego_speed_mps),
            np.asarray(scenario.event_times_s, dtype=np.float64),
            np.asarray(scenario.set_speeds_mps, dtype=np.float64),
            float(SAMPLING_HZ),
            THETA,
            MASS,
            RHO,
            CD,
            AREA,
            CROLL,
            GRAVITY,
            F_MAX,
        )
        cruise_mse.append(float(s_mse)); cruise_w.append(float(scenario.weight))
        jerk_mse.append(float(j_mse)); jerk_w.append(float(scenario.weight))
        energy_whkm.append(float(e_whkm)); energy_w.append(float(scenario.weight))
        feasible = feasible and bool(ok)
        violation += float(vio)

    for scenario in suite.following:
        d_mse, j_mse, e_whkm, _dn, _jn, ok, vio = following_metrics_energy_kernel_jit(
            k_de,
            k_vrel,
            float(scenario.initial_ego_speed_mps),
            float(scenario.initial_lead_speed_mps),
            float(scenario.initial_gap_m),
            np.asarray(scenario.lead_speed_mps, dtype=np.float64),
            float(SAMPLING_HZ),
            D0,
            TIME_GAP,
            THETA,
            MASS,
            RHO,
            CD,
            AREA,
            CROLL,
            GRAVITY,
            F_MAX,
            SENSOR_NOISE_STD,
            SENSOR_BIAS,
            SENSOR_SEED,
            SENSOR_MIN,
        )
        follow_mse.append(float(d_mse)); follow_w.append(float(scenario.weight))
        jerk_mse.append(float(j_mse)); jerk_w.append(float(scenario.weight))
        energy_whkm.append(float(e_whkm)); energy_w.append(float(scenario.weight))
        feasible = feasible and bool(ok)
        violation += float(vio)

    if not feasible:
        return PerformanceVectorFJE(1e12, 1e12, 1e12, 1e12, False, max(violation, 1.0))

    fv = math.sqrt(max(_weighted_mean(cruise_mse, cruise_w), 0.0))
    fd = math.sqrt(max(_weighted_mean(follow_mse, follow_w), 0.0))
    fj = math.sqrt(max(_weighted_mean(jerk_mse, jerk_w), 0.0))
    fe = _weighted_mean(energy_whkm, energy_w)
    if not all(math.isfinite(x) for x in (fv, fd, fj, fe)):
        return PerformanceVectorFJE(1e12, 1e12, 1e12, 1e12, False, 1e12)
    return PerformanceVectorFJE(fv, fd, fj, fe, True, 0.0)


def sahib_contribution_weights(objective_matrix: np.ndarray) -> np.ndarray:
    """Compute sum-to-one inverse-mean Sahib weights for the four FJE terms."""
    arr = np.asarray(objective_matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] < 1:
        raise ValueError("objective_matrix must have shape (n>=1, 4) for FJE")
    if np.any(~np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError("Pareto objective values must be finite and nonnegative")
    mu = np.mean(arr, axis=0)
    if np.any(mu <= 0.0) or np.any(~np.isfinite(mu)):
        raise ValueError(f"Pareto objective means must be finite and positive; got {mu}")
    inv = 1.0 / mu
    return (inv / np.sum(inv)).astype(float)


def scalar_cost_from_performance(perf: PerformanceVectorFJE, weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    if w.shape != (4,) or np.any(w < 0.0) or not np.isfinite(w).all():
        raise ValueError("FJE weights must be four finite nonnegative values")
    if not math.isclose(float(w.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("FJE cost-profile weights must sum to 1")
    if not perf.feasible:
        return float(1e12 + 1e6 * max(perf.constraint_violation, 0.0))
    return float(np.dot(w, perf.as_array()))


def load_cost_profile(*, require_frozen: bool = True) -> CostProfile:
    path = resolve_project_path(str(_COST["cost_profile_path"]))
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    status = str(data.get("status", "unknown"))
    if require_frozen and status != "frozen":
        raise RuntimeError(
            f"FJE cost profile is {status!r}, not 'frozen'. Run FJE calibration before the research benchmark."
        )
    weights = data.get("weights")
    if weights is None:
        raise RuntimeError("FJE cost profile has no weights")
    w = np.asarray([weights[name] for name in PERFORMANCE_NAMES], dtype=float)
    if w.shape != (4,) or not np.isfinite(w).all() or np.any(w <= 0.0) or not np.isclose(w.sum(), 1.0):
        raise RuntimeError(f"invalid frozen FJE weights: {w}")
    return CostProfile(
        weights=w,
        status=status,
        scenario_manifest_sha256=str(data.get("scenario_manifest_sha256", "")),
        path=path,
    )


_PRODUCTION_SUITE = None
_PRODUCTION_PROFILE = None


def _get_production_context():
    global _PRODUCTION_SUITE, _PRODUCTION_PROFILE
    if _PRODUCTION_SUITE is None:
        _PRODUCTION_SUITE = load_scenario_suite(
            _COST["scenario_manifest_path"], development=False, require_frozen=True
        )
    if _PRODUCTION_PROFILE is None:
        _PRODUCTION_PROFILE = load_cost_profile(require_frozen=True)
    if _PRODUCTION_PROFILE.scenario_manifest_sha256 != _PRODUCTION_SUITE.manifest_sha256:
        raise RuntimeError(
            "Frozen FJE cost profile was derived from a different scenario manifest. Recalibrate."
        )
    return _PRODUCTION_SUITE, _PRODUCTION_PROFILE


def acc_cost_fje_fitness(params, _objective_token=None):
    suite, profile = _get_production_context()
    perf = evaluate_performance_vector(params, suite)
    return scalar_cost_from_performance(perf, profile.weights)


def development_cost_fitness(params, weights=(0.25, 0.25, 0.25, 0.25)):
    suite = load_scenario_suite(_COST["scenario_manifest_path"], development=True, require_frozen=False)
    perf = evaluate_performance_vector(params, suite)
    return scalar_cost_from_performance(perf, weights)


def _physics() -> VehiclePhysics:
    return VehiclePhysics(M=MASS, Cd=CD, A=AREA, rho=RHO, Croll=CROLL, g=GRAVITY, theta=THETA)


def simulate_cruise_history(params, scenario: CruiseScenario):
    """Readable cruise simulation including wheel-force/power telemetry."""
    k_ve, _k_de, _k_vrel = map(float, params)
    dt = 1.0 / SAMPLING_HZ
    physics = _physics()
    v = float(scenario.initial_ego_speed_mps)
    a = 0.0
    history = {k: [] for k in [
        "time", "v_set", "v_ego", "a_cmd", "a_ego", "jerk", "speed_error",
        "traction_force_n", "tractive_power_w", "positive_tractive_power_w",
    ]}
    event_idx = 0
    prev_a = None
    steps = int(round(scenario.duration_s * SAMPLING_HZ))
    for i in range(steps):
        t = i * dt
        while event_idx + 1 < len(scenario.event_times_s) and t >= scenario.event_times_s[event_idx + 1]:
            event_idx += 1
        v_set = float(scenario.set_speeds_mps[event_idx])
        err = v_set - v
        a_cmd = k_ve * err
        v_before = v
        v_new, a_new, force = physics.step_acceleration_command(v, a_cmd, dt, Fmax=F_MAX)
        jerk = 0.0 if prev_a is None else (a_new - prev_a) / dt
        power = float(force) * max(v_before, 0.0)
        values = {
            "time": t, "v_set": v_set, "v_ego": v_before, "a_cmd": a_cmd,
            "a_ego": a_new, "jerk": jerk, "speed_error": err,
            "traction_force_n": force, "tractive_power_w": power,
            "positive_tractive_power_w": max(power, 0.0),
        }
        for key, value in values.items():
            history[key].append(float(value))
        prev_a = a_new
        v, a = v_new, a_new
    return history


def simulate_following_history(params, scenario: FollowingScenario):
    """Readable following simulation including wheel-force/power telemetry."""
    _k_ve, k_de, k_vrel = map(float, params)
    dt = 1.0 / SAMPLING_HZ
    physics = _physics()
    sensor = RangeOnlySensor(
        dt=dt,
        config=RangeSensorConfig(
            noise_std_m=SENSOR_NOISE_STD,
            bias_m=SENSOR_BIAS,
            min_distance_m=SENSOR_MIN,
            seed=SENSOR_SEED,
        ),
    )
    x_e, v_e = 0.0, float(scenario.initial_ego_speed_mps)
    x_l, v_l = float(scenario.initial_gap_m), float(scenario.initial_lead_speed_mps)
    prev_a = None
    history = {k: [] for k in [
        "time", "v_ego", "v_lead", "dist", "dist_sensed", "d_safe", "v_rel_est",
        "a_cmd", "a_ego", "jerk", "spacing_error", "traction_force_n",
        "tractive_power_w", "positive_tractive_power_w",
    ]}
    for i, lead_speed in enumerate(np.asarray(scenario.lead_speed_mps, dtype=float)):
        t = i * dt
        dist = x_l - x_e
        reading = sensor.sense(dist)
        d_safe = D0 + TIME_GAP * v_e
        spacing_error = dist - d_safe
        a_cmd = k_de * (reading.distance - d_safe) + k_vrel * reading.relative_velocity
        v_before = v_e
        v_new, a_new, force = physics.step_acceleration_command(v_e, a_cmd, dt, Fmax=F_MAX)
        jerk = 0.0 if prev_a is None else (a_new - prev_a) / dt
        power = float(force) * max(v_before, 0.0)
        values = {
            "time": t, "v_ego": v_before, "v_lead": v_l, "dist": dist,
            "dist_sensed": reading.distance, "d_safe": d_safe,
            "v_rel_est": reading.relative_velocity, "a_cmd": a_cmd,
            "a_ego": a_new, "jerk": jerk, "spacing_error": spacing_error,
            "traction_force_n": force, "tractive_power_w": power,
            "positive_tractive_power_w": max(power, 0.0),
        }
        for key, value in values.items():
            history[key].append(float(value))
        prev_a = a_new
        v_e = v_new
        v_l = float(lead_speed)
        x_e += v_e * dt
        x_l += v_l * dt
    return history


BENCHMARKS_ACC_FJE_V2 = {_MANIFEST["benchmark"]: acc_cost_fje_fitness}
