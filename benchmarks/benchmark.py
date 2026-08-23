"""Optimizer-facing benchmark functions.

These functions keep the optimizer API simple while the internals remain
modular. Each fitness call creates a fresh controller per scenario, which is
important for FOPID because it has memory.

Key design choice
-----------------
The constraint evaluator can infer the controller architecture from the
fitness function itself. This keeps ``optimize_parallel.py`` simple:

    fitness_functions = [("acc_fopid", acc_fopid_fitness_func)]
    constraint_evaluator = make_constraint_evaluator_from_objective(fitness_func)

No controller-type string has to be duplicated in the optimizer script.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from controllers import ClassicalACCController, FOPIDACCController, AugmentedFOPIDACCController
from core.metrics import evaluate_history_constraints, mean_objective
from core.scenario_registry import SCENARIOS
from core.simulator import simulate_acc
from core.types import SimulationConfig


# -----------------------------------------------------------------------------
# Controller-type metadata helpers
# -----------------------------------------------------------------------------

_CONTROLLER_ALIASES = {
    "classical": "classical",
    "classical_acc": "classical",
    "legacy": "classical",
    "acc_classical": "classical",
    "fopid": "fopid",
    "fopid_acc": "fopid",
    "acc_fopid": "fopid",
    "pure_fopid": "fopid",
    "pure_fopid_acc": "fopid",
    "augmented_fopid": "augmented_fopid",
    "augmented_fopid_acc": "augmented_fopid",
    "acc_augmented_fopid": "augmented_fopid",
    "aug_fopid": "augmented_fopid",
    "aug_fopid_acc": "augmented_fopid",
}


def normalize_controller_type(controller_type: str) -> str:
    """Normalize controller names to: classical, fopid, augmented_fopid."""
    key = str(controller_type).lower().strip()
    if key not in _CONTROLLER_ALIASES:
        valid = ", ".join(sorted(_CONTROLLER_ALIASES))
        raise ValueError(f"Unknown controller_type: {controller_type!r}. Valid aliases: {valid}")
    return _CONTROLLER_ALIASES[key]


def controller_fitness(controller_type: str):
    """Attach controller metadata to an optimizer-facing fitness function.

    This is more robust than parsing strings in ``optimize_parallel.py``.
    Name-based detection is still provided as a fallback.
    """
    normalized = normalize_controller_type(controller_type)

    def _decorate(func: Callable) -> Callable:
        setattr(func, "controller_type", normalized)
        setattr(func, "is_acc_fitness", True)
        return func

    return _decorate


def _callable_name(func: Optional[Callable]) -> str:
    """Return a stable name for a function or functools.partial object."""
    if func is None:
        return ""
    # functools.partial keeps the original function in .func
    base_func = getattr(func, "func", func)
    return str(getattr(base_func, "__name__", "")).lower().strip()


def infer_controller_type_from_objective(
    obj_function: Optional[Callable] = None,
    controller_type: Optional[str] = None,
    default: Optional[str] = None,
) -> str:
    """Infer controller type from an explicit argument or fitness function.

    Priority:
    1. Explicit ``controller_type`` if provided.
    2. Metadata attached to the objective/fitness function.
    3. Function-name fallback, e.g. ``acc_fopid_fitness_func`` -> fopid.
    4. ``default`` if provided.

    This lets the optimizer pass only the fitness function and avoids repeating
    controller strings in multiple files.
    """
    if controller_type is not None:
        return normalize_controller_type(controller_type)

    if obj_function is not None:
        metadata = getattr(obj_function, "controller_type", None)
        if metadata is None and isinstance(obj_function, partial):
            metadata = getattr(obj_function.func, "controller_type", None)
        if metadata is not None:
            return normalize_controller_type(metadata)

        name = _callable_name(obj_function)
        # Check augmented before fopid because augmented_fopid contains fopid.
        if "augmented" in name and "fopid" in name:
            return "augmented_fopid"
        if "aug_fopid" in name:
            return "augmented_fopid"
        if "fopid" in name:
            return "fopid"
        if "classical" in name or "legacy" in name:
            return "classical"

    if default is not None:
        return normalize_controller_type(default)

    raise ValueError(
        "Could not infer controller type. Pass a fitness function decorated with "
        "@controller_fitness(...), or pass controller_type explicitly."
    )


# -----------------------------------------------------------------------------
# Controller factories
# -----------------------------------------------------------------------------

def make_classical_controller(
    params: Sequence[float],
    config: Optional[SimulationConfig] = None,
) -> ClassicalACCController:
    """Factory for the classical ACC controller."""
    config = SimulationConfig() if config is None else config
    return ClassicalACCController.from_params(
        params,
        default_time_gap=config.time_gap,
        default_standstill_distance=config.standstill_distance,
    )


def make_fopid_controller(
    params: Sequence[float],
    config: Optional[SimulationConfig] = None,
    memory_size: int = 300,
    output_limit: Optional[float] = None,
    derivative_filter: float = 1.0,
) -> FOPIDACCController:
    """Factory for the pure FOPID ACC controller."""
    config = SimulationConfig() if config is None else config
    return FOPIDACCController.from_params(
        params,
        default_time_gap=config.time_gap,
        default_standstill_distance=config.standstill_distance,
        memory_size=memory_size,
        output_limit=output_limit,
        derivative_filter=derivative_filter,
    )


def make_augmented_fopid_controller(
    params: Sequence[float],
    config: Optional[SimulationConfig] = None,
    memory_size: int = 300,
    output_limit: Optional[float] = None,
    derivative_filter: float = 1.0,
) -> AugmentedFOPIDACCController:
    """Factory for the augmented FOPID ACC controller."""
    config = SimulationConfig() if config is None else config
    return AugmentedFOPIDACCController.from_params(
        params,
        default_time_gap=config.time_gap,
        default_standstill_distance=config.standstill_distance,
        memory_size=memory_size,
        output_limit=output_limit,
        derivative_filter=derivative_filter,
    )


def _controller_factory(
    controller_type: str,
    params: Sequence[float],
    config: SimulationConfig,
):
    controller_type = normalize_controller_type(controller_type)
    if controller_type == "classical":
        return make_classical_controller(params, config=config)
    if controller_type == "fopid":
        return make_fopid_controller(params, config=config, output_limit=config.control_limit)
    if controller_type == "augmented_fopid":
        return make_augmented_fopid_controller(params, config=config, output_limit=config.control_limit)
    raise ValueError(f"Unknown normalized controller_type: {controller_type}")


# -----------------------------------------------------------------------------
# Scenario simulation wrappers
# -----------------------------------------------------------------------------

def acc_single_scenario(
    params: Sequence[float],
    scenario: Dict[str, object],
    controller_type: str = "classical",
    config: Optional[SimulationConfig] = None,
) -> float:
    """Run one scenario and return the tracking objective."""
    config = SimulationConfig() if config is None else config
    controller = _controller_factory(controller_type, params, config)
    return float(simulate_acc(controller=controller, scenario=scenario, config=config, return_history=False))


def acc_single_scenario_history(
    params: Sequence[float],
    scenario: Dict[str, object],
    controller_type: str = "classical",
    config: Optional[SimulationConfig] = None,
) -> list[object]:
    """Run one scenario and return ``[objective, history]``."""
    config = SimulationConfig() if config is None else config
    controller = _controller_factory(controller_type, params, config)
    return simulate_acc(controller=controller, scenario=scenario, config=config, return_history=True)


def acc_robust_fitness_func(
    params: Sequence[float],
    controller_type: str = "classical",
    config: Optional[SimulationConfig] = None,
    max_workers: int = 4,
) -> float:
    """Mean tracking objective across all active scenarios."""
    config = SimulationConfig() if config is None else config
    controller_type = normalize_controller_type(controller_type)
    scenarios = list(SCENARIOS)
    if not scenarios:
        return float("inf")

    with ThreadPoolExecutor(max_workers=min(int(max_workers), len(scenarios))) as pool:
        futures = [
            pool.submit(acc_single_scenario, params, scen, controller_type, config)
            for scen in scenarios
        ]
        values = [future.result() for future in futures]
    return mean_objective(values)


# -----------------------------------------------------------------------------
# Optimizer-facing fitness functions
# -----------------------------------------------------------------------------

@controller_fitness("classical")
def acc_classical_fitness_func(params: Sequence[float], obj_function=None) -> float:
    """Optimizer-compatible fitness for classical ACC.

    ``obj_function`` is accepted for backward compatibility with the old
    pipeline, but the refactored benchmark uses the normalized tracking
    objective rather than IAE/ISE/ITAE wrappers.
    """
    return acc_robust_fitness_func(params=params, controller_type="classical")


@controller_fitness("fopid")
def acc_fopid_fitness_func(params: Sequence[float], obj_function=None) -> float:
    """Optimizer-compatible fitness for pure FOPID ACC."""
    return acc_robust_fitness_func(params=params, controller_type="fopid")


@controller_fitness("augmented_fopid")
def acc_augmented_fopid_fitness_func(params: Sequence[float], obj_function=None) -> float:
    """Optimizer-compatible fitness for augmented FOPID ACC."""
    return acc_robust_fitness_func(params=params, controller_type="augmented_fopid")


# -----------------------------------------------------------------------------
# Constraint evaluators
# -----------------------------------------------------------------------------

def acc_constraint_evaluator(
    params: Sequence[float],
    obj_function: Optional[Callable] = None,
    controller_type: Optional[str] = None,
    config: Optional[SimulationConfig] = None,
) -> Dict[str, object]:
    """Constraint evaluator compatible with SA-DA-RIME style optimizers.

    The controller architecture is inferred from ``obj_function`` whenever
    possible. For example:

        acc_constraint_evaluator(params, obj_function=acc_fopid_fitness_func)

    will automatically evaluate constraints using the pure FOPID controller.
    """
    config = SimulationConfig() if config is None else config
    resolved_controller_type = infer_controller_type_from_objective(
        obj_function=obj_function,
        controller_type=controller_type,
        default="fopid",
    )

    total_violation = 0.0
    collision_any = False
    scenario_results = []

    for scenario in list(SCENARIOS):
        objective, history = acc_single_scenario_history(
            params=params,
            scenario=scenario,
            controller_type=resolved_controller_type,
            config=config,
        )
        result = evaluate_history_constraints(
            history,
            ttc_threshold=config.ttc_threshold,
            acceleration_limit=config.acceleration_limit,
            jerk_limit=config.jerk_limit,
            control_limit=config.control_limit,
            min_gap_margin=config.min_gap_margin,
        )
        result["scenario"] = scenario.get("name", "unnamed")
        result["objective"] = float(objective)
        result["controller_type"] = resolved_controller_type
        scenario_results.append(result)
        total_violation += float(result["total_violation"])
        collision_any = collision_any or bool(result["collision"])

    return {
        "feasible": bool(total_violation <= 0.0),
        "total_violation": float(total_violation),
        "safety_violation": float(total_violation),
        "collision": bool(collision_any),
        "controller_type": resolved_controller_type,
        "scenario_results": scenario_results,
    }


def make_constraint_evaluator(
    controller_type: Optional[str] = None,
    obj_function: Optional[Callable] = None,
    config: Optional[SimulationConfig] = None,
) -> Callable[[Sequence[float]], Dict[str, object]]:
    """Return a controller-specific constraint evaluator for optimizers.

    You can call this in either style:

        make_constraint_evaluator(controller_type="fopid")
        make_constraint_evaluator(obj_function=acc_fopid_fitness_func)
    """
    resolved_controller_type = infer_controller_type_from_objective(
        obj_function=obj_function,
        controller_type=controller_type,
        default="fopid",
    )

    def _evaluate(params: Sequence[float], obj_function_override=None) -> Dict[str, object]:
        effective_obj = obj_function if obj_function_override is None else obj_function_override
        return acc_constraint_evaluator(
            params=params,
            obj_function=effective_obj,
            controller_type=resolved_controller_type,
            config=config,
        )

    setattr(_evaluate, "controller_type", resolved_controller_type)
    return _evaluate


def make_constraint_evaluator_from_objective(
    obj_function: Callable,
    config: Optional[SimulationConfig] = None,
) -> Callable[[Sequence[float]], Dict[str, object]]:
    """Build a constraint evaluator by reading metadata from a fitness function."""
    return make_constraint_evaluator(obj_function=obj_function, config=config)


BENCHMARKS_ACC = {
    "acc_classical": acc_classical_fitness_func,
    "acc_fopid": acc_fopid_fitness_func,
    "acc_augmented_fopid": acc_augmented_fopid_fitness_func,
}


BENCHMARK_CONSTRAINTS_ACC = {
    name: make_constraint_evaluator_from_objective(func)
    for name, func in BENCHMARKS_ACC.items()
}
