"""Global scenario registry used by optimizer-facing benchmark functions."""

from __future__ import annotations

from typing import Dict, List

SCENARIOS: List[Dict[str, object]] = [
    {"name": "legacy_nominal", "theta_angle": 0, "vehicle_mass": 1500, "rolling_resistance": 0.06},
]


def set_scenarios(scenarios: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Replace active scenarios and return the new list."""
    # global SCENARIOS
    # SCENARIOS = list(scenarios)
    SCENARIOS.clear()
    SCENARIOS.extend(scenarios)
    return SCENARIOS
