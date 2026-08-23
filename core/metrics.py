"""Objective and constraint metrics for ACC experiments."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np


def compute_ttc(d_actual: float, v_ego: float, v_lead: float, eps: float = 1e-9) -> float:
    """Compute 1-D time-to-collision.

    Returns ``math.inf`` when the ego vehicle is not closing in on the lead
    vehicle. Returns 0 when the gap is already non-positive.
    """
    if d_actual <= 0:
        return 0.0
    closing_speed = float(v_ego) - float(v_lead)
    if closing_speed <= eps:
        return math.inf
    return float(d_actual) / closing_speed


def normalized_tracking_step_cost(
    d_actual: float,
    d_ref: float,
    v_ego: float,
    v_lead: float,
    v_set: float,
    eps: float = 1e-9,
) -> float:
    """Tracking objective used by the refactored benchmark.

    The cost is deliberately simple and weight-free:
    - normalized spacing error;
    - normalized ego/lead relative-speed error.

    This follows the current methodology decision: tracking is the objective,
    while safety, comfort, and actuator limits are constraints.
    """
    d_scale = max(abs(float(d_ref)), eps)
    v_scale = max(abs(float(v_set)), eps)
    spacing_term = ((float(d_actual) - float(d_ref)) / d_scale) ** 2
    relative_speed_term = ((float(v_ego) - float(v_lead)) / v_scale) ** 2
    return float(spacing_term + relative_speed_term)


def _mean_normalized_positive_violation(values: np.ndarray, limit: float, eps: float = 1e-9) -> float:
    """Mean max(0, values-limit)/limit."""
    if limit is None or limit <= 0 or len(values) == 0:
        return 0.0
    return float(np.mean(np.maximum(0.0, values - float(limit)) / max(abs(float(limit)), eps)))


def evaluate_history_constraints(
    history: Dict[str, object],
    ttc_threshold: float = 1.5,
    acceleration_limit: Optional[float] = None,
    jerk_limit: Optional[float] = None,
    control_limit: Optional[float] = None,
    min_gap_margin: float = 0.0,
    collision_penalty: float = 1000.0,
    eps: float = 1e-9,
) -> Dict[str, object]:
    """Evaluate safety/comfort/actuator feasibility from simulation history.

    The returned ``total_violation`` is not a weighted performance objective.
    It is used only for feasibility-rule ranking when candidates are infeasible.
    """
    dist = np.asarray(history.get("dist", []), dtype=float)
    d_safe = np.asarray(history.get("d_safe", []), dtype=float)
    ttc = np.asarray(history.get("ttc", []), dtype=float)
    a_ego = np.asarray(history.get("a_ego", []), dtype=float)
    jerk = np.asarray(history.get("jerk", []), dtype=float)
    u = np.asarray(history.get("u", []), dtype=float)

    collision = bool(history.get("collision", False))

    if len(dist) and len(d_safe):
        gap_margin = dist - d_safe
        gap_violation = float(np.mean(np.maximum(0.0, float(min_gap_margin) - gap_margin) / (np.maximum(np.abs(d_safe), eps))))
        min_gap_margin_value = float(np.min(gap_margin))
    else:
        gap_violation = 0.0
        min_gap_margin_value = math.inf

    finite_ttc = ttc[np.isfinite(ttc)]
    if len(finite_ttc):
        ttc_violation = float(np.mean(np.maximum(0.0, float(ttc_threshold) - finite_ttc) / max(float(ttc_threshold), eps)))
        min_ttc = float(np.min(finite_ttc))
    else:
        ttc_violation = 0.0
        min_ttc = math.inf

    acceleration_violation = _mean_normalized_positive_violation(np.abs(a_ego), acceleration_limit, eps=eps)
    jerk_violation = _mean_normalized_positive_violation(np.abs(jerk), jerk_limit, eps=eps)
    control_violation = _mean_normalized_positive_violation(np.abs(u), control_limit, eps=eps)
    collision_violation = float(collision_penalty if collision else 0.0)

    total_violation = float(
        gap_violation
        + ttc_violation
        + acceleration_violation
        + jerk_violation
        + control_violation
        + collision_violation
    )

    return {
        "feasible": bool(total_violation <= 0.0),
        "total_violation": total_violation,
        "safety_violation": total_violation,  # legacy alias for SA-DA-RIME drafts
        "collision": collision,
        "min_ttc": None if min_ttc == math.inf else min_ttc,
        "min_gap_margin": None if min_gap_margin_value == math.inf else min_gap_margin_value,
        "gap_violation": gap_violation,
        "ttc_violation": ttc_violation,
        "acceleration_violation": acceleration_violation,
        "jerk_violation": jerk_violation,
        "control_violation": control_violation,
        "collision_violation": collision_violation,
    }


def deb_feasibility_key(objective_value: float, constraint_result: Dict[str, object]) -> tuple[int, float]:
    """Sort key implementing Deb-style feasibility rules.

    Smaller is better:
    - feasible candidates sort before infeasible candidates;
    - among feasible candidates, objective value decides;
    - among infeasible candidates, total violation decides.
    """
    feasible = bool(constraint_result.get("feasible", False))
    if feasible:
        return (0, float(objective_value))
    return (1, float(constraint_result.get("total_violation", math.inf)))


def mean_objective(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return math.inf
    return float(np.mean(np.asarray(values, dtype=float)))
