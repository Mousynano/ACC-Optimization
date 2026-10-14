"""Versioned evaluation-only sensing and plant extensions (no tuning changes).

The controller receives measured range and its causal range-rate estimate only.
Matched compensation is an explicit ideal-model assumption; frozen_nominal and
custom compensation never read the actual plant parameter vector.
"""
from dataclasses import dataclass, field, asdict
import math
import numpy as np
from numba import njit

from core.fast_acc import deterministic_standard_normal_jit
from core.sensing import RangeSensorConfig
from core.research_simulation import ActuatorConfig, CRUISE_KEYS, FOLLOW_KEYS, _context, _history


@dataclass(frozen=True)
class SensorProcessingConfig:
    sampling_hz: float | None = None
    range_filter_tau_s: float = 0.0
    rate_filter_tau_s: float = 0.0

    def resolved(self, control_hz):
        hz = float(control_hz if self.sampling_hz is None else self.sampling_hz)
        values = (hz, self.range_filter_tau_s, self.rate_filter_tau_s)
        if not all(math.isfinite(v) for v in values) or hz <= 0 or hz > control_hz:
            raise ValueError("Sensor rate must be finite, positive, and no faster than control")
        if min(self.range_filter_tau_s, self.rate_filter_tau_s) < 0:
            raise ValueError("Filter time constants must be nonnegative")
        stride = int(round(control_hz / hz))
        if not math.isclose(stride * hz, control_hz, rel_tol=1e-12):
            raise ValueError("Sensor period must be an integer multiple of the control period")
        return np.array([stride, self.range_filter_tau_s, self.rate_filter_tau_s], float)


@dataclass(frozen=True)
class CompensationConfig:
    mode: str = "matched"
    parameters: dict = field(default_factory=dict)

    def resolved(self, manifest, actual):
        if self.mode not in ("matched", "frozen_nominal", "custom"):
            raise ValueError("Compensation mode must be matched, frozen_nominal, or custom")
        if self.mode != "custom" and self.parameters:
            raise ValueError("Only custom compensation accepts parameter overrides")
        if self.mode == "matched":
            return actual.copy()
        nominal = _context([0., 0., 0.], manifest, None, None)[1]
        if self.mode == "custom":
            keys = ["mass_kg", "air_density_kg_m3", "drag_coefficient", "frontal_area_m2",
                    "rolling_resistance", "gravity_m_per_s2", "road_slope_deg"]
            if set(self.parameters) - set(keys):
                raise ValueError("Unknown compensation parameter")
            for k, value in self.parameters.items():
                nominal[keys.index(k)] = math.radians(value) if k == "road_slope_deg" else value
        if not np.isfinite(nominal).all() or np.any(nominal[:6] < 0) or nominal[0] <= 0 or nominal[5] <= 0:
            raise ValueError("Invalid compensation physics")
        return nominal


@njit(cache=True, fastmath=False)
def road_load_jit(speed, physics):
    mass, rho, cd, area, crr, gravity, slope = physics
    return (0.5*rho*cd*area*speed**2 + crr*mass*gravity*math.cos(slope)
            + mass*gravity*math.sin(slope))


@njit(cache=True, fastmath=False)
def force_step_mismatch_jit(speed, command, dt, actual, compensation, actuator, previous_force):
    road = road_load_jit(speed, actual)
    estimated_road = road_load_jit(speed, compensation)
    requested = compensation[0]*command + estimated_road
    tau, traction, braking, power, _ = actuator
    upper = min(traction, power/max(speed, 1e-9)) if power > 0 else traction
    target = min(max(requested, -braking), upper)
    force = target if tau == 0 else previous_force + (1-math.exp(-dt/tau))*(target-previous_force)
    force = min(max(force, -braking), upper)
    acceleration = (force-road)/actual[0]
    return max(speed+acceleration*dt, 0.), acceleration, force, requested, target, road, estimated_road


@njit(cache=True, fastmath=False)
def sensor_update_jit(raw, previous_filtered, previous_rate, elapsed, range_tau, rate_tau, first):
    if first:
        return raw, 0., 0.
    alpha = 1. if range_tau == 0 else 1-math.exp(-elapsed/range_tau)
    filtered = raw if range_tau == 0 else previous_filtered + alpha*(raw-previous_filtered)
    raw_rate = (filtered-previous_filtered)/elapsed
    beta = 1. if rate_tau == 0 else 1-math.exp(-elapsed/rate_tau)
    rate = raw_rate if rate_tau == 0 else previous_rate+beta*(raw_rate-previous_rate)
    return filtered, rate, raw_rate


EXT_CRUISE_KEYS = (*CRUISE_KEYS, "estimated_road_load_force_n")
EXT_FOLLOW_KEYS = (*FOLLOW_KEYS, "estimated_road_load_force_n", "dist_raw",
    "range_noise_m", "sensor_updated", "measurement_time_s", "measurement_age_s",
    "v_rel_before_rate_filter")


@njit(cache=True, fastmath=False)
def cruise_extended_jit(gains, duration, initial_speed, events, speeds, hz, physics, compensation, actuator):
    steps = int(round(duration*hz)); dt = 1./hz
    out = np.empty((steps, len(EXT_CRUISE_KEYS))); speed = initial_speed
    previous_acc = 0.; previous_force = actuator[4]; event = 0
    for i in range(steps):
        time = i*dt
        while event+1 < len(events) and time >= events[event+1]:
            event += 1
        ref = speeds[event]; error = ref-speed; command = gains[0]*error
        vn, acc, force, request, target, road, estimated = force_step_mismatch_jit(
            speed, command, dt, physics, compensation, actuator, previous_force)
        jerk = 0. if i == 0 else (acc-previous_acc)/dt
        power = force*max(speed, 0.)
        out[i] = (time, ref, speed, command, acc, jerk, error, force, power, max(power, 0.),
                  request, target, road, estimated)
        if not np.isfinite(out[i]).all(): return out[:i+1], 2
        speed = vn; previous_acc = acc; previous_force = force
    return out, 0


@njit(cache=True, fastmath=False)
def following_extended_jit(gains, lead, ego0, lead0, gap0, hz, d0, headway,
                           physics, compensation, actuator, sensor, processing):
    dt = 1./hz; out = np.empty((len(lead), len(EXT_FOLLOW_KEYS)))
    xe = 0.; xl = gap0; ve = ego0; vl = lead0
    filtered = 0.; relative = 0.; raw = 0.; raw_rate = 0.; noise_value = 0.
    previous_acc = 0.; previous_force = actuator[4]; last_tick = 0; sample_index = 0
    noise, bias, seed, minimum = sensor
    stride, range_tau, rate_tau = processing
    for i in range(len(lead)):
        gap = xl-xe
        updated = i % int(stride) == 0
        if updated:
            noise_value = noise*deterministic_standard_normal_jit(int(seed), sample_index) if noise > 0 else 0.
            raw = max(gap+bias+noise_value, minimum)
            # Differentiate at the measurement clock, never at held controller ticks.
            elapsed = (i-last_tick)*dt if i > 0 else dt
            filtered, relative, raw_rate = sensor_update_jit(raw, filtered, relative, elapsed,
                                                           range_tau, rate_tau, i == 0)
            last_tick = i; sample_index += 1
        desired = d0+headway*ve
        command = gains[1]*(filtered-desired)+gains[2]*relative
        vn, acc, force, request, target, road, estimated = force_step_mismatch_jit(
            ve, command, dt, physics, compensation, actuator, previous_force)
        jerk = 0. if i == 0 else (acc-previous_acc)/dt
        power = force*max(ve, 0.)
        out[i] = (i*dt, ve, vl, gap, filtered, desired, relative, vl-ve, command, acc, jerk,
                  gap-desired, force, power, max(power, 0.), request, target, road, estimated,
                  raw, noise_value, 1. if updated else 0., last_tick*dt, (i-last_tick)*dt, raw_rate)
        if not np.isfinite(out[i]).all(): return out[:i+1], 2
        if gap <= 0: return out[:i+1], 1
        previous_acc = acc; previous_force = force
        ve = vn; vl = lead[i]; xe += ve*dt; xl += vl*dt
    return out, 0


def simulate_extended(gains, scenario, manifest, physics_override=None, sensor_override=None,
                      actuator=None, sensor_processing=None, compensation=None):
    gains, physics, av, config = _context(gains, manifest, physics_override, actuator)
    comp = compensation or CompensationConfig()
    cp = comp.resolved(manifest, physics)
    processing = sensor_processing or SensorProcessingConfig()
    hz = int(manifest["execution_backend"]["sampling_hz"])
    pv = processing.resolved(hz)
    s = manifest["sensing"]
    sensor = sensor_override or RangeSensorConfig(s["nominal_noise_std_m"], s["nominal_bias_m"],
                                                s["nominal_min_distance_m"], s["nominal_noise_seed"])
    sv = np.array([sensor.noise_std_m, sensor.bias_m, sensor.seed, sensor.min_distance_m], float)
    if hasattr(scenario, "lead_speed_mps"):
        c = manifest["controller"]
        array, code = following_extended_jit(gains, scenario.lead_speed_mps,
            scenario.initial_ego_speed_mps, scenario.initial_lead_speed_mps, scenario.initial_gap_m,
            hz, c["standstill_distance_m"], c["time_gap_s"], physics, cp, av, sv, pv)
        keys = EXT_FOLLOW_KEYS; expected = len(scenario.lead_speed_mps)
    else:
        array, code = cruise_extended_jit(gains, scenario.duration_s, scenario.initial_ego_speed_mps,
            scenario.event_times_s, scenario.set_speeds_mps, hz, physics, cp, av)
        keys = EXT_CRUISE_KEYS; expected = int(round(scenario.duration_s*hz))
    history = _history(array, code, keys, expected, config)
    history["_status"].update(model="evaluation_extensions_v2", compensation=asdict(comp),
        actual_physics=physics.tolist(), compensation_physics=cp.tolist(),
        sensor=asdict(sensor), sensor_processing=asdict(processing), control_hz=hz)
    return history
