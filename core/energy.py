"""Powertrain-agnostic longitudinal traction-energy metrics.

This module deliberately models *wheel-level positive tractive energy*, not
fuel consumption and not battery energy.  Negative tractive power receives no
energy credit.  The implementation reuses the same longitudinal road-load
physics as the ACC plant so the metric adds no fictional powertrain model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TractionEnergyMetrics:
    positive_energy_j: float
    braking_energy_j: float
    distance_m: float
    positive_wh_per_km: float
    braking_wh_per_km: float
    traction_brake_reversals: int


def road_load_force_n(
    speed_mps: float,
    *,
    mass_kg: float = 1500.0,
    rho_kg_m3: float = 1.225,
    drag_coefficient: float = 0.36,
    frontal_area_m2: float = 2.42,
    rolling_resistance: float = 0.015,
    gravity_mps2: float = 9.81,
    road_slope_deg: float = 0.0,
) -> float:
    """Return aerodynamic + rolling + grade road load in newtons."""
    v = max(float(speed_mps), 0.0)
    theta = math.radians(float(road_slope_deg))
    drag = 0.5 * float(rho_kg_m3) * float(drag_coefficient) * float(frontal_area_m2) * v * v
    rolling = float(rolling_resistance) * float(mass_kg) * float(gravity_mps2) * math.cos(theta)
    grade = float(mass_kg) * float(gravity_mps2) * math.sin(theta)
    return float(drag + rolling + grade)


def reconstructed_tractive_force_n(
    speed_mps: float,
    acceleration_mps2: float,
    **physics,
) -> float:
    """Reconstruct wheel traction force from realized longitudinal motion."""
    mass = float(physics.get("mass_kg", 1500.0))
    return mass * float(acceleration_mps2) + road_load_force_n(speed_mps, **physics)


def positive_tractive_power_w(
    speed_mps: float,
    acceleration_mps2: float,
    **physics,
) -> float:
    """Positive-only wheel power. Negative tractive power receives no credit."""
    v = max(float(speed_mps), 0.0)
    p = reconstructed_tractive_force_n(v, acceleration_mps2, **physics) * v
    return float(max(p, 0.0))


def wh_per_km(energy_j: float, distance_m: float) -> float:
    """Convert integrated joules over distance to Wh/km."""
    d = float(distance_m)
    if not math.isfinite(d) or d <= 1e-12:
        return float("inf")
    return float(float(energy_j) * 1000.0 / (3600.0 * d))


def applied_force_energy(speed, force_n, dt_s):
    """Integrate actual interval-start wheel power, never nominal reconstruction."""
    v, force = np.asarray(speed,float), np.asarray(force_n,float)
    if v.shape != force.shape or v.ndim != 1 or not len(v):
        raise ValueError("Aligned nonempty speed and applied-force vectors required")
    if not np.isfinite(v).all() or not np.isfinite(force).all() or not math.isfinite(dt_s) or dt_s <= 0:
        raise ValueError("Energy inputs must be finite; dt must be positive")
    power = force*np.maximum(v,0.0)
    positive = float(np.maximum(power,0).sum()*dt_s)
    braking = float(np.maximum(-power,0).sum()*dt_s)
    distance = float(np.maximum(v,0).sum()*dt_s)
    active = np.sign(power[np.abs(power)>1e-9])
    return TractionEnergyMetrics(positive,braking,distance,wh_per_km(positive,distance),
        wh_per_km(braking,distance),int(np.sum(active[1:] != active[:-1])) if len(active)>1 else 0)


def trajectory_traction_energy(
    speed_mps: Sequence[float],
    acceleration_mps2: Sequence[float],
    dt_s: float,
    *,
    mass_kg: float = 1500.0,
    rho_kg_m3: float = 1.225,
    drag_coefficient: float = 0.36,
    frontal_area_m2: float = 2.42,
    rolling_resistance: float = 0.015,
    gravity_mps2: float = 9.81,
    road_slope_deg: float = 0.0,
) -> TractionEnergyMetrics:
    """Compute positive/braking wheel work and traction/brake reversals.

    ``speed_mps[k]`` is interpreted as the speed at the beginning of the same
    plant interval whose realized acceleration is ``acceleration_mps2[k]``.
    This matches the explicit time semantics used by the readable ACC simulator.
    """
    v = np.asarray(speed_mps, dtype=float)
    a = np.asarray(acceleration_mps2, dtype=float)
    if v.shape != a.shape or v.ndim != 1 or v.size == 0:
        raise ValueError("speed and acceleration must be non-empty aligned 1-D arrays")
    if not np.isfinite(v).all() or not np.isfinite(a).all():
        raise ValueError("speed/acceleration contain non-finite values")
    dt = float(dt_s)
    if not math.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt_s must be finite and positive")

    theta = math.radians(float(road_slope_deg))
    vv = np.maximum(v, 0.0)
    drag = 0.5 * float(rho_kg_m3) * float(drag_coefficient) * float(frontal_area_m2) * vv**2
    rolling = float(rolling_resistance) * float(mass_kg) * float(gravity_mps2) * math.cos(theta)
    grade = float(mass_kg) * float(gravity_mps2) * math.sin(theta)
    force = float(mass_kg) * a + drag + rolling + grade
    power = force * vv

    positive_j = float(np.sum(np.maximum(power, 0.0)) * dt)
    braking_j = float(np.sum(np.maximum(-power, 0.0)) * dt)
    distance_m = float(np.sum(vv) * dt)

    # Ignore exact/near-zero force samples when counting meaningful traction↔brake changes.
    active = np.sign(power[np.abs(power) > 1e-9])
    reversals = int(np.sum(active[1:] != active[:-1])) if active.size > 1 else 0

    return TractionEnergyMetrics(
        positive_energy_j=positive_j,
        braking_energy_j=braking_j,
        distance_m=distance_m,
        positive_wh_per_km=wh_per_km(positive_j, distance_m),
        braking_wh_per_km=wh_per_km(braking_j, distance_m),
        traction_brake_reversals=reversals,
    )
