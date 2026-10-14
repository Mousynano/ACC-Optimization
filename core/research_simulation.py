"""RQ evaluation backend; preserves the frozen nominal controller/plant.

Optional evaluation extensions are dispatched explicitly; nominal behavior is frozen.
All states are interval-start states, matching the production FJE kernels.
Optional actuator scenarios are reporting-only and must name their parameters.
"""
from dataclasses import dataclass, asdict
import math
import numpy as np
from numba import njit

from core.fast_acc import deterministic_standard_normal_jit


@dataclass(frozen=True)
class ActuatorConfig:
    lag_s: float = 0.0
    traction_limit_n: float | None = None
    braking_limit_n: float | None = None
    max_power_w: float | None = None
    initial_force_n: float = 0.0

    def resolved(self, nominal_limit):
        traction = nominal_limit if self.traction_limit_n is None else self.traction_limit_n
        braking = nominal_limit if self.braking_limit_n is None else self.braking_limit_n
        values = (self.lag_s, traction, braking, self.initial_force_n)
        if not all(math.isfinite(float(v)) for v in values):
            raise ValueError("Actuator parameters must be finite")
        if self.lag_s < 0 or traction <= 0 or braking <= 0:
            raise ValueError("Lag must be nonnegative; force limits must be positive")
        power = 0.0 if self.max_power_w is None else float(self.max_power_w)
        if not math.isfinite(power) or (self.max_power_w is not None and power <= 0):
            raise ValueError("An explicit power limit must be finite and positive")
        if not -braking <= self.initial_force_n <= traction:
            raise ValueError("Initial actuator force is outside amplitude limits")
        return np.array([self.lag_s, traction, braking, power, self.initial_force_n], dtype=float)


@njit(cache=True, fastmath=False)
def force_step_jit(speed, command, dt, physics, actuator, previous_force):
    mass, rho, cd, area, crr, gravity, slope = physics
    resistance = (0.5 * rho * cd * area * speed**2
                  + crr * mass * gravity * math.cos(slope)
                  + mass * gravity * math.sin(slope))
    requested = mass * command + resistance
    tau, traction, braking, power, _initial = actuator
    upper = traction
    if power > 0:
        upper = min(upper, power / max(speed, 1e-9))
    target = min(max(requested, -braking), upper)
    if tau == 0:
        force = target
    else:
        beta = 1.0 - math.exp(-dt / tau)
        force = previous_force + beta * (target - previous_force)
        # Instantaneous envelope applies even when speed changes the power cap.
        force = min(max(force, -braking), upper)
    acceleration = (force - resistance) / mass
    return max(speed + acceleration * dt, 0.0), acceleration, force, requested, target, resistance


def force_step_reference(speed, command, dt, physics, actuator, previous_force):
    """Readable reference for nominal and optional actuator conditions."""
    mass, rho, cd, area, crr, gravity, slope = map(float, physics)
    resistance = (0.5*rho*cd*area*speed**2 + crr*mass*gravity*math.cos(slope)
                  + mass*gravity*math.sin(slope))
    requested = mass * command + resistance
    tau, traction, braking, power, _ = map(float, actuator)
    upper = min(traction, power/max(speed, 1e-9)) if power > 0 else traction
    target = float(np.clip(requested, -braking, upper))
    force = target if tau == 0 else previous_force + (1-math.exp(-dt/tau))*(target-previous_force)
    force = float(np.clip(force, -braking, upper))
    acceleration = (force-resistance)/mass
    return max(speed+acceleration*dt, 0.0), acceleration, force, requested, target, resistance


CRUISE_KEYS = ("time", "v_set", "v_ego", "a_cmd", "a_ego", "jerk", "speed_error",
               "traction_force_n", "tractive_power_w", "positive_tractive_power_w",
               "requested_force_n", "target_force_n", "road_load_force_n")
FOLLOW_KEYS = ("time", "v_ego", "v_lead", "dist", "dist_sensed", "d_safe", "v_rel_est",
               "v_rel_true", "a_cmd", "a_ego", "jerk", "spacing_error", "traction_force_n",
               "tractive_power_w", "positive_tractive_power_w", "requested_force_n",
               "target_force_n", "road_load_force_n")


@njit(cache=True, fastmath=False)
def cruise_history_jit(gains, duration, initial_speed, events, speeds, hz, physics, actuator):
    steps = int(round(duration*hz)); dt = 1.0/hz
    out = np.empty((steps, 13)); speed = initial_speed
    previous_acc = 0.0; previous_force = actuator[4]; event = 0
    for i in range(steps):
        time = i*dt
        while event+1 < len(events) and time >= events[event+1]:
            event += 1
        ref = speeds[event]; error = ref-speed; command = gains[0]*error
        new_speed, acc, force, request, target, road = force_step_jit(speed,command,dt,physics,actuator,previous_force)
        jerk = 0.0 if i == 0 else (acc-previous_acc)/dt
        power = force*max(speed,0.0)
        out[i] = (time,ref,speed,command,acc,jerk,error,force,power,max(power,0.0),request,target,road)
        if not np.isfinite(out[i]).all():
            return out[:i+1], 2
        speed = new_speed; previous_acc = acc; previous_force = force
    return out, 0


@njit(cache=True, fastmath=False)
def following_history_jit(gains, lead, ego0, lead0, gap0, hz, d0, headway, physics, actuator, sensor):
    dt = 1.0/hz; out = np.empty((len(lead),18))
    xe = 0.0; xl = gap0; ve = ego0; vl = lead0
    previous_range = 0.0; previous_acc = 0.0; previous_force = actuator[4]
    noise, bias, seed, minimum = sensor
    for i in range(len(lead)):
        gap = xl-xe
        sensed = gap+bias
        if noise > 0:
            sensed += noise*deterministic_standard_normal_jit(int(seed),i)
        sensed = max(sensed,minimum)
        relative = 0.0 if i == 0 else (sensed-previous_range)/dt
        desired = d0+headway*ve
        command = gains[1]*(sensed-desired)+gains[2]*relative
        vn, acc, force, request, target, road = force_step_jit(ve,command,dt,physics,actuator,previous_force)
        jerk = 0.0 if i == 0 else (acc-previous_acc)/dt
        power = force*max(ve,0.0)
        out[i] = (i*dt,ve,vl,gap,sensed,desired,relative,vl-ve,command,acc,jerk,gap-desired,
                  force,power,max(power,0.0),request,target,road)
        if not np.isfinite(out[i]).all():
            return out[:i+1], 2
        if gap <= 0:
            return out[:i+1], 1
        previous_range = sensed; previous_acc = acc; previous_force = force
        ve = vn; vl = lead[i]; xe += ve*dt; xl += vl*dt
    return out, 0


def _context(gains, manifest, override, actuator):
    gains = np.asarray(gains, dtype=float)
    if gains.shape != (3,) or not np.isfinite(gains).all():
        raise ValueError("Three finite gains required")
    p = manifest["physics"]
    mass = p["nominal_mass_kg"] if override is None else override.mass_kg
    crr = p["rolling_resistance"] if override is None else override.rolling_resistance
    slope = p["road_slope_deg"] if override is None else override.road_slope_deg
    physics = np.array([mass,p.get("air_density_kg_m3",1.225),p.get("drag_coefficient",.36),
                        p.get("frontal_area_m2",2.42),crr,p.get("gravity_m_per_s2",9.81),math.radians(slope)],float)
    if not np.isfinite(physics).all() or mass <= 0:
        raise ValueError("Invalid physical parameters")
    config = actuator or ActuatorConfig()
    return gains, physics, config.resolved(p["max_longitudinal_force_n"]), config


def _history(array, code, keys, expected, actuator):
    history = {k: array[:,i] for i,k in enumerate(keys)}
    history["_status"] = {"failed":code != 0,"reason":{0:"",1:"collision",2:"nonfinite"}[code],
                          "steps_completed":len(array),"expected_steps":expected,
                          "model":"ideal_compensation_with_force_limits_v1" if actuator == ActuatorConfig()
                          else "reporting_actuator_extension_v1", "actuator":asdict(actuator)}
    return history


def simulate_cruise(gains, scenario, manifest, physics_override=None, actuator=None, compensation=None):
    if compensation is not None:
        from core.evaluation_extensions import simulate_extended
        return simulate_extended(gains, scenario, manifest, physics_override, actuator=actuator, compensation=compensation)
    gains, physics, av, config = _context(gains,manifest,physics_override,actuator)
    hz = int(manifest["execution_backend"]["sampling_hz"])
    array, code = cruise_history_jit(gains,scenario.duration_s,scenario.initial_ego_speed_mps,
                                   scenario.event_times_s,scenario.set_speeds_mps,hz,physics,av)
    return _history(array,code,CRUISE_KEYS,int(round(scenario.duration_s*hz)),config)


def simulate_following(gains, scenario, manifest, physics_override=None, sensor_override=None, actuator=None,
                       sensor_processing=None, compensation=None):
    if sensor_processing is not None or compensation is not None:
        from core.evaluation_extensions import simulate_extended
        return simulate_extended(gains, scenario, manifest, physics_override, sensor_override, actuator,
                                 sensor_processing, compensation)
    gains, physics, av, config = _context(gains,manifest,physics_override,actuator)
    s = manifest["sensing"]; c = manifest["controller"]
    if sensor_override is None:
        sensor = np.array([s["nominal_noise_std_m"],s["nominal_bias_m"],s["nominal_noise_seed"],s["nominal_min_distance_m"]],float)
    else:
        sensor = np.array([sensor_override.noise_std_m,sensor_override.bias_m,sensor_override.seed,sensor_override.min_distance_m],float)
    array, code = following_history_jit(gains,scenario.lead_speed_mps,scenario.initial_ego_speed_mps,
        scenario.initial_lead_speed_mps,scenario.initial_gap_m,int(manifest["execution_backend"]["sampling_hz"]),
        c["standstill_distance_m"],c["time_gap_s"],physics,av,sensor)
    return _history(array,code,FOLLOW_KEYS,len(scenario.lead_speed_mps),config)
