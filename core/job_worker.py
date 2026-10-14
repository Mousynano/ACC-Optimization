"""One optimization-job execution unit for the active ACC benchmark."""

import time
from typing import Any, Callable, Dict

from core.seeding import seed_everything


def run_job(
    fun_name: str,
    run_id: int,
    algo_name: str,
    obj_name: str,
    obj_fn: Callable[..., float],
    algo_runner: Callable[..., Any],
    fitness_fn: Callable[..., float],
    min_params,
    max_params,
    population_size: int,
    max_iter: int,
    seed: int,
    progress_dict,
    status_dict,
    current_obj_dict,
) -> Dict[str, Any]:
    key = (fun_name, run_id, algo_name, obj_name)
    if progress_dict is not None:
        progress_dict[key] = 0

    def progress_callback(step):
        if progress_dict is not None:
            progress_dict[key] = int(max(0, min(step, max_iter)))

    function_evaluations = 0

    def fitness_wrapper(params):
        nonlocal function_evaluations
        function_evaluations += 1
        return fitness_fn(params, obj_fn)

    seed_everything(seed)
    if current_obj_dict is not None:
        current_obj_dict[(fun_name, run_id, algo_name)] = obj_name

    start_t = time.perf_counter()
    solution, best_fitness, curve = algo_runner(
        func=fitness_wrapper,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        verbose=False,
        progress_callback=progress_callback,
        seed=seed,
    )
    duration = time.perf_counter() - start_t

    if status_dict is not None:
        status_dict[key] = {"fitness": best_fitness}
    if progress_dict is not None:
        progress_dict[key] = max_iter
    if current_obj_dict is not None:
        current_obj_dict[(fun_name, run_id, algo_name)] = "done"

    return {
        "fun_name": fun_name,
        "run_id": run_id,
        "algo_name": algo_name,
        "obj_name": obj_name,
        "seed": seed,
        "best_fitness": best_fitness,
        "best_solution": solution,
        "curve": curve,
        "duration": duration,
        "function_evaluations": function_evaluations,
    }
