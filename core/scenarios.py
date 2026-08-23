"""Scenario builders for ACC experiments.

A scenario is a plain dictionary on purpose. This keeps the module compatible
with your current checkpoint/optimizer pipeline and makes it easy to serialize.
"""

from __future__ import annotations

from typing import Dict, List

from core.scenario_registry import set_scenarios
from core.wltc import apply_sensor_delay, apply_sensor_noise, make_wltc_lead_profile


def make_wltc_scenario(
    csv_path: str = "wltc_class3b.csv",
    phase: str = "full",
    dt: float = 1.0 / 60.0,
    name: str | None = None,
    initial_gap: float = 60.0,
    ego_initial_speed: float | None = None,
    theta_angle: float = 0.0,
    vehicle_mass: float = 1500.0,
    rolling_resistance: float = 0.06,
    noise_std_v: float = 0.0,
    noise_std_x: float = 0.0,
    delay_s: float = 0.0,
    speed_scale: float = 1.0,
    seed: int | None = None,
) -> Dict[str, object]:
    """Create one WLTC scenario compatible with ``simulate_acc``."""
    profile = make_wltc_lead_profile(csv_path=csv_path, dt=dt, phase=phase, speed_scale=speed_scale)
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
        "theta_angle": float(theta_angle),
        "vehicle_mass": float(vehicle_mass),
        "rolling_resistance": float(rolling_resistance),
        "noise_std_v": float(noise_std_v),
        "noise_std_x": float(noise_std_x),
        "delay_s": float(delay_s),
    }


def build_wltc_scenarios(csv_path: str = "wltc_class3b.csv", dt: float = 1.0 / 60.0, mode: str = "train") -> List[Dict[str, object]]:
    """Build a train/test bundle of WLTC-derived scenarios."""
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
                vehicle_mass=1500.0 * 1.15,
                rolling_resistance=0.06,
            ),
            make_wltc_scenario(
                csv_path,
                phase="extra_high",
                dt=dt,
                name="test_extra_high_roll_variation",
                initial_gap=65.0,
                vehicle_mass=1500.0,
                rolling_resistance=0.06 * 1.20,
            ),
        ]

    raise ValueError("mode must be 'train' or 'test'.")


def use_wltc_scenarios(csv_path: str = "wltc_class3b.csv", dt: float = 1.0 / 60.0, mode: str = "train") -> List[Dict[str, object]]:
    """Replace the active global scenario registry with WLTC scenarios."""
    return set_scenarios(build_wltc_scenarios(csv_path=csv_path, dt=dt, mode=mode))
