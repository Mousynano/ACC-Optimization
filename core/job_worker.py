# core/job_worker.py
import os

from core.seeding import seed_everything
from core.xp import xp
from core.checkpoint_jobs import job_done
from core.optimization_job import OptimizationJob
import inspect
import time
from typing import Dict, Callable, Any

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

    def cb(step):
        if progress_dict is None:
            return
        # clamp biar aman
        if step < 0:
            step = 0
        if step > max_iter:
            step = max_iter
        progress_dict[key] = int(step)

    def fitness_wrapper(params):
        # benchmark signature: (params, obj_fn) atau (params)
        if len(inspect.signature(fitness_fn).parameters) == 2:
            return fitness_fn(params, obj_fn)
        return fitness_fn(params)

    # seed sekali per job (per objective) → aman & deterministik
    if seed is not None:
        seed_everything(seed, xp=xp)

    if current_obj_dict is not None:
        current_obj_dict[(fun_name, run_id, algo_name)] = obj_name

    start_t = time.time()
    sol, best_fit, curve = algo_runner(
        func=fitness_wrapper,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        verbose=False,
        progress_callback=cb,
        seed=seed,   # kalau algo_runner support
    )
    end_t = time.time()
    duration = end_t - start_t
    if status_dict is not None:
        status_dict[key] = {"fitness": best_fit}

    if progress_dict is not None:
        progress_dict[key] = max_iter

    if current_obj_dict is not None:
        current_obj_dict[(fun_name, run_id, algo_name)] = "done"

    return {
        "fun_name": fun_name,
        "run_id": run_id,
        "algo_name": algo_name,
        "obj_name": obj_name,
        "best_fitness": best_fit,
        "best_solution": sol,
        "curve": curve,
        "duration": duration
    }

def make_jobs(fitness_functions, algorithms, objectives, N_RUNS) -> list:
    jobs = []
    for fun_name, fitness_fn in fitness_functions:
        for run_id in range(N_RUNS):
            for algo_name in algorithms.keys():
                for obj_name, obj_fn in objectives.items():
                    if job_done(fun_name, run_id, algo_name, obj_name):
                        continue
                    jobs.append(OptimizationJob(
                        fun_name=fun_name, 
                        fitness_fn=fitness_fn,
                        run_id=run_id,
                        algo_name=algo_name,
                        obj_name=obj_name,
                        obj_fn=obj_fn,
                    ))
    return jobs