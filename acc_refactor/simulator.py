"""Controller-agnostic ACC simulator.

This is the main refactor point: the simulator owns vehicle integration,
lead-vehicle playback, history logging, and metric accumulation. It does not
contain any controller law. Classical ACC and FOPID are injected as objects
that implement ``BaseACCController``.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from core.physics import VehiclePhysics
from core.utils import sin_angle

from acc_refactor.controllers.base import BaseACCController
from acc_refactor.metrics import compute_ttc, normalized_tracking_step_cost
from acc_refactor.types import ACCState, SimulationConfig


def _empty_history() -> Dict[str, object]:
    return {
        "time": [],
        "mode": [],
        "v_ego": [],
        "v_lead": [],
        "x_ego": [],
        "x_lead": [],
        "car_force": [],
        "dist": [],
        "d_ref": [],
        "d_safe": [],
        "a_ego": [],
        "a_lead": [],
        "jerk": [],
        "u": [],
        "controller_error": [],
        "tracking_cost": [],
        "obj_val": [],
        "ttc": [],
        "TET": 0.0,
        "TIT": 0.0,
        "ttc_min": None,
        "collision": False,
        "saturation_ratio": 0.0,
    }


def _scenario_value(scenario: Dict[str, object], name: str, default):
    return scenario.get(name, default)


def simulate_acc(
    controller: BaseACCController,
    scenario: Optional[Dict[str, object]] = None,
    config: Optional[SimulationConfig] = None,
    return_history: bool = False,
) -> float | list[object]:
    """Simulate one ACC scenario with any controller implementing the interface.

    Parameters
    ----------
    controller:
        Controller object, e.g. ``ClassicalACCController`` or
        ``FOPIDACCController``.
    scenario:
        Plain dictionary containing vehicle parameters and optional
        ``lead_profile``. If no lead profile is supplied, the legacy sine-wave
        lead car is used.
    config:
        Simulation and constraint configuration.
    return_history:
        If true, returns ``[objective, history]``. Otherwise returns objective.
    """
    scenario = {} if scenario is None else dict(scenario)
    config = SimulationConfig() if config is None else config

    controller.reset()

    theta_angle = float(_scenario_value(scenario, "theta_angle", 0.0))
    vehicle_mass = float(_scenario_value(scenario, "vehicle_mass", 1500.0))
    rolling_resistance = float(_scenario_value(scenario, "rolling_resistance", 0.06))
    initial_gap = float(_scenario_value(scenario, "initial_gap", 60.0))
    ego_initial_speed = _scenario_value(scenario, "ego_initial_speed", None)
    lead_profile = _scenario_value(scenario, "lead_profile", None)

    physics = VehiclePhysics(theta=theta_angle, M=vehicle_mass, Croll=rolling_resistance)

    if lead_profile is None:
        dt = float(config.dt)
        simulation_steps = int(float(config.sim_time) / dt)
        t_array = None
        v_lead_profile = a_lead_profile = x_lead_profile = None

        x_ego = 0.0
        v_ego = 22.0 if ego_initial_speed is None else float(ego_initial_speed)
        a_ego = 0.0

        x_lead = initial_gap
        v_lead = 22.0
        a_lead = 0.0
        lead_theta = 0.0
    else:
        dt = float(lead_profile["dt"])
        t_array = np.asarray(lead_profile["t"], dtype=float)
        simulation_steps = len(t_array)
        v_lead_profile = np.asarray(lead_profile["v_lead"], dtype=float)
        a_lead_profile = np.asarray(lead_profile["a_lead"], dtype=float)
        x_lead_profile = np.asarray(lead_profile["x_lead"], dtype=float)

        x_ego = 0.0
        v_ego = float(v_lead_profile[0]) if ego_initial_speed is None else float(ego_initial_speed)
        a_ego = 0.0

        x_lead = initial_gap + float(x_lead_profile[0])
        v_lead = float(v_lead_profile[0])
        a_lead = float(a_lead_profile[0])
        lead_theta = 0.0

    history = _empty_history()
    objective_sum = 0.0
    TET = 0.0
    TIT = 0.0
    ttc_min = np.inf
    collision = False
    saturation_count = 0
    previous_a_ego = a_ego

    for i in range(simulation_steps):
        if lead_profile is None:
            t = i * dt
        else:
            t = float(t_array[i])
            v_lead = float(v_lead_profile[i])
            a_lead = float(a_lead_profile[i])
            x_lead = initial_gap + float(x_lead_profile[i])

        d_actual = x_lead - x_ego
        d_ref_guess = config.standstill_distance + config.time_gap * v_ego

        state = ACCState(
            t=float(t),
            dt=float(dt),
            x_ego=float(x_ego),
            v_ego=float(v_ego),
            a_ego=float(a_ego),
            x_lead=float(x_lead),
            v_lead=float(v_lead),
            a_lead=float(a_lead),
            v_set=float(config.v_set),
            d_actual=float(d_actual),
            d_ref=float(d_ref_guess),
            d_safe=float(d_ref_guess),
        )

        output = controller.step(state)
        u = float(output.u)
        d_ref = float(output.d_ref)
        d_safe = d_ref

        v_ego, a_ego, traction_force = physics.step(v_ego, u, dt)
        x_ego += v_ego * dt

        if lead_profile is None:
            lead_theta = (lead_theta + 30.0 * dt) % 360.0
            a_lead = sin_angle(lead_theta)
            v_lead += a_lead * dt
            x_lead += v_lead * dt

        jerk = (a_ego - previous_a_ego) / dt
        previous_a_ego = a_ego

        ttc = compute_ttc(d_actual=d_actual, v_ego=v_ego, v_lead=v_lead)
        if ttc < ttc_min:
            ttc_min = ttc
        if ttc < config.ttc_threshold:
            TET += dt
            TIT += (config.ttc_threshold - ttc) * dt
        if d_actual <= 0.0:
            collision = True

        if config.control_limit is not None and abs(u) > abs(float(config.control_limit)):
            saturation_count += 1

        step_cost = normalized_tracking_step_cost(
            d_actual=d_actual,
            d_ref=d_ref,
            v_ego=v_ego,
            v_lead=v_lead,
            v_set=config.v_set,
            eps=config.eps,
        )
        objective_sum += step_cost
        objective_value = objective_sum / float(i + 1)

        if return_history:
            history["time"].append(float(t))
            history["mode"].append(output.mode)
            history["v_ego"].append(float(v_ego))
            history["v_lead"].append(float(v_lead))
            history["x_ego"].append(float(x_ego))
            history["x_lead"].append(float(x_lead))
            history["car_force"].append(float(traction_force))
            history["dist"].append(float(d_actual))
            history["d_ref"].append(float(d_ref))
            history["d_safe"].append(float(d_safe))
            history["a_ego"].append(float(a_ego))
            history["a_lead"].append(float(a_lead))
            history["jerk"].append(float(jerk))
            history["u"].append(float(u))
            history["controller_error"].append(float(output.error))
            history["tracking_cost"].append(float(step_cost))
            history["obj_val"].append(float(objective_value))
            history["ttc"].append(float(ttc))

    if return_history:
        history["TET"] = float(TET)
        history["TIT"] = float(TIT)
        history["ttc_min"] = None if ttc_min == np.inf else float(ttc_min)
        history["collision"] = bool(collision)
        history["saturation_ratio"] = float(saturation_count / max(simulation_steps, 1))
        return [float(objective_sum / max(simulation_steps, 1)), history]

    return float(objective_sum / max(simulation_steps, 1))
