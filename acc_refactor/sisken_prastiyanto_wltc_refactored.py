"""Compatibility adapter for the old ``sisken_prastiyanto_wltc.py`` imports.

Use this file if your current pipeline expects one benchmark file. Internally,
it delegates to the refactored modules.

Recommended new imports:
    from acc_refactor.benchmark import acc_fopid_fitness_func
    from acc_refactor.scenarios import use_wltc_scenarios

Temporary old-style imports:
    from acc_refactor.sisken_prastiyanto_wltc_refactored import acc_fitness_func
"""

from __future__ import annotations

from acc_refactor.benchmark import (
    BENCHMARKS_ACC,
    acc_classical_fitness_func,
    acc_constraint_evaluator,
    acc_fopid_fitness_func,
    acc_single_scenario,
    acc_single_scenario_history,
)
from acc_refactor.metrics import compute_ttc
from acc_refactor.scenario_registry import SCENARIOS, set_scenarios
from acc_refactor.scenarios import build_wltc_scenarios, make_wltc_scenario, use_wltc_scenarios
from acc_refactor.wltc import (
    apply_sensor_delay,
    apply_sensor_noise,
    get_wltc_phase_window,
    load_wltc_csv,
    make_wltc_lead_profile,
)

# Default old name. Keep this classical so existing experiments do not silently
# switch controller family. For FOPID, use ``acc_fopid_fitness_func`` explicitly.
acc_fitness_func = acc_classical_fitness_func

__all__ = [
    "SCENARIOS",
    "set_scenarios",
    "use_wltc_scenarios",
    "build_wltc_scenarios",
    "make_wltc_scenario",
    "load_wltc_csv",
    "get_wltc_phase_window",
    "make_wltc_lead_profile",
    "apply_sensor_noise",
    "apply_sensor_delay",
    "compute_ttc",
    "acc_single_scenario",
    "acc_single_scenario_history",
    "acc_fitness_func",
    "acc_classical_fitness_func",
    "acc_fopid_fitness_func",
    "acc_constraint_evaluator",
    "BENCHMARKS_ACC",
]
