"""V2 ACC benchmark: Sahib-derived scalar cost from direct performance metrics.

The research objective is *not* available until calibration has produced a frozen
``cost_profile_acc_cost_v2.json`` against a frozen scenario manifest. Development
smoke tests can explicitly request the synthetic development suite and temporary
equal weights, but those values are never used by the normal benchmark registry.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

from core.config import ROOT_DIR, load_manifest, resolve_project_path
from core.fast_acc_v2 import cruise_metrics_kernel_jit, following_metrics_kernel_jit
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
CROLL = float(_PHYSICS["rolling_resistance"])
F_MAX = float(_PHYSICS["max_longitudinal_force_n"])
SENSOR_NOISE_STD = float(_SENSING["nominal_noise_std_m"])
SENSOR_BIAS = float(_SENSING["nominal_bias_m"])
SENSOR_SEED = int(_SENSING["nominal_noise_seed"])
SENSOR_MIN = float(_SENSING["nominal_min_distance_m"])


@dataclass(frozen=True)
class PerformanceVector:
    cruise_speed_rmse: float
    following_spacing_rmse: float
    realized_jerk_rms: float
    feasible: bool
    constraint_violation: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [self.cruise_speed_rmse, self.following_spacing_rmse, self.realized_jerk_rms],
            dtype=float,
        )


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


def evaluate_performance_vector(params, suite: ScenarioSuite) -> PerformanceVector:
    """Evaluate [cruise RMSE, following RMSE, jerk RMS] on one frozen suite."""
    if len(params) != 3:
        raise ValueError("v2 ACC decision vector must be [k_ve, k_de, k_vrel]")
    k_ve, k_de, k_vrel = map(float, params)

    cruise_mse = []
    cruise_w = []
    follow_mse = []
    follow_w = []
    jerk_mse = []
    jerk_w = []
    feasible = True
    violation = 0.0

    for scenario in suite.cruise:
        s_mse, j_mse, _sn, _jn, ok, vio = cruise_metrics_kernel_jit(
            k_ve,
            float(scenario.duration_s),
            float(scenario.initial_ego_speed_mps),
            np.asarray(scenario.event_times_s, dtype=np.float64),
            np.asarray(scenario.set_speeds_mps, dtype=np.float64),
            float(SAMPLING_HZ),
            THETA,
            MASS,
            CROLL,
            F_MAX,
        )
        cruise_mse.append(float(s_mse))
        cruise_w.append(float(scenario.weight))
        jerk_mse.append(float(j_mse))
        jerk_w.append(float(scenario.weight))
        feasible = feasible and bool(ok)
        violation += float(vio)

    for scenario in suite.following:
        d_mse, j_mse, _dn, _jn, ok, vio = following_metrics_kernel_jit(
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
            CROLL,
            F_MAX,
            SENSOR_NOISE_STD,
            SENSOR_BIAS,
            SENSOR_SEED,
            SENSOR_MIN,
        )
        follow_mse.append(float(d_mse))
        follow_w.append(float(scenario.weight))
        jerk_mse.append(float(j_mse))
        jerk_w.append(float(scenario.weight))
        feasible = feasible and bool(ok)
        violation += float(vio)

    if not feasible:
        # Objective values remain finite for optimizer bookkeeping, while
        # constraint dominance uses the explicit feasibility flag/violation.
        return PerformanceVector(1e12, 1e12, 1e12, False, max(violation, 1.0))

    fv = math.sqrt(max(_weighted_mean(cruise_mse, cruise_w), 0.0))
    fd = math.sqrt(max(_weighted_mean(follow_mse, follow_w), 0.0))
    fj = math.sqrt(max(_weighted_mean(jerk_mse, jerk_w), 0.0))
    if not all(math.isfinite(x) for x in (fv, fd, fj)):
        return PerformanceVector(1e12, 1e12, 1e12, False, 1e12)
    return PerformanceVector(fv, fd, fj, True, 0.0)


def sahib_contribution_weights(objective_matrix: np.ndarray) -> np.ndarray:
    """Compute sum-to-one inverse-mean contribution weights.

    Rows are Pareto solutions; columns are [f_v, f_d, f_j].
    """
    arr = np.asarray(objective_matrix, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
        raise ValueError("objective_matrix must have shape (n>=1, 3)")
    if np.any(~np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("Pareto objective values must be finite and strictly positive")
    mu = np.mean(arr, axis=0)
    inv = 1.0 / mu
    weights = inv / np.sum(inv)
    return weights.astype(float)


def scalar_cost_from_performance(perf: PerformanceVector, weights: Sequence[float]) -> float:
    w = np.asarray(weights, dtype=float)
    if w.shape != (3,) or np.any(w < 0.0) or not np.isfinite(w).all():
        raise ValueError("weights must be three finite nonnegative values")
    if not math.isclose(float(w.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise ValueError("v2 cost-profile weights must sum to 1")
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
            f"Cost profile is {status!r}, not 'frozen'. Run/freeze calibration before "
            "the v2 research benchmark."
        )
    weights = data.get("weights")
    if weights is None:
        raise RuntimeError("cost profile has no weights")
    w = np.asarray(
        [weights["cruise_speed_rmse"], weights["following_spacing_rmse"], weights["realized_jerk_rms"]],
        dtype=float,
    )
    if not math.isclose(float(w.sum()), 1.0, rel_tol=1e-8, abs_tol=1e-8):
        raise RuntimeError(f"frozen weights must sum to 1; got {w}")
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
            "Frozen cost profile was derived from a different scenario manifest. "
            "Recalibrate rather than silently mixing versions."
        )
    return _PRODUCTION_SUITE, _PRODUCTION_PROFILE


def acc_cost_fitness(params, _objective_token=None):
    suite, profile = _get_production_context()
    perf = evaluate_performance_vector(params, suite)
    return scalar_cost_from_performance(perf, profile.weights)


def development_cost_fitness(params, weights=(1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)):
    suite = load_scenario_suite(_COST["scenario_manifest_path"], development=True, require_frozen=False)
    perf = evaluate_performance_vector(params, suite)
    return scalar_cost_from_performance(perf, weights)


def simulate_cruise_history(params, scenario: CruiseScenario):
    """Readable V2 cruise-mode simulation for plots/reporting."""
    k_ve, _k_de, _k_vrel = map(float, params)
    dt = 1.0 / SAMPLING_HZ
    physics = VehiclePhysics(M=MASS, Croll=CROLL, theta=THETA)
    v = float(scenario.initial_ego_speed_mps)
    a = 0.0
    history = {k: [] for k in ["time","v_set","v_ego","a_cmd","a_ego","jerk","speed_error"]}
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
        v_new, a_new, _ = physics.step_acceleration_command(v, a_cmd, dt, Fmax=F_MAX)
        jerk = 0.0 if prev_a is None else (a_new - prev_a) / dt
        for key, value in [("time",t),("v_set",v_set),("v_ego",v),("a_cmd",a_cmd),("a_ego",a_new),("jerk",jerk),("speed_error",err)]:
            history[key].append(float(value))
        prev_a = a_new
        v, a = v_new, a_new
    return history


def simulate_following_history(params, scenario: FollowingScenario):
    """Readable V2 following-mode simulation for plots/reporting."""
    _k_ve, k_de, k_vrel = map(float, params)
    dt = 1.0 / SAMPLING_HZ
    physics = VehiclePhysics(M=MASS, Croll=CROLL, theta=THETA)
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
        "time","v_ego","v_lead","dist","dist_sensed","d_safe","v_rel_est",
        "a_cmd","a_ego","jerk","spacing_error"
    ]}
    for i, lead_speed in enumerate(np.asarray(scenario.lead_speed_mps, dtype=float)):
        t = i * dt
        dist = x_l - x_e
        reading = sensor.sense(dist)
        d_safe = D0 + TIME_GAP * v_e
        spacing_error = dist - d_safe
        a_cmd = k_de * (reading.distance - d_safe) + k_vrel * reading.relative_velocity
        v_new, a_new, _ = physics.step_acceleration_command(v_e, a_cmd, dt, Fmax=F_MAX)
        jerk = 0.0 if prev_a is None else (a_new - prev_a) / dt
        for key, value in [
            ("time",t),("v_ego",v_e),("v_lead",v_l),("dist",dist),
            ("dist_sensed",reading.distance),("d_safe",d_safe),("v_rel_est",reading.relative_velocity),
            ("a_cmd",a_cmd),("a_ego",a_new),("jerk",jerk),("spacing_error",spacing_error)
        ]:
            history[key].append(float(value))
        prev_a = a_new
        v_e = v_new
        v_l = float(lead_speed)
        x_e += v_e * dt
        x_l += v_l * dt
    return history


BENCHMARKS_ACC_V2 = {_MANIFEST["benchmark"]: acc_cost_fitness}
