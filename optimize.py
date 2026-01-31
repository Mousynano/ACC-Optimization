import time
import numpy as np
import os

from tqdm import tqdm
from multiprocessing import Process, Manager

from benchmarks.sisken_prastiyanto import BENCHMARKS_ACC as SISKEN_BENCH
from benchmarks.simple_funcs import BENCHMARKS 

from optimizers.pso import run_pso
from optimizers.ga import run_ga
from optimizers.kma import run_kma
from optimizers.ska import run_ska
from optimizers.hoa import run_hoa

from core.report import generate_statistical_report, generate_convergence_plot
from core.algorithm_worker import algo_worker
from core.utils import iae, ise, itae, itse
from core.checkpoint import save_checkpoint, load_existing_checkpoints

# -----------------------------
# Algorithm registry
# -----------------------------
algorithms = {
    # "kma": run_kma,
    "pso": run_pso,
    # "ska": run_ska,
    # "hoa": run_hoa,
    # "ga": run_ga,
}

# -----------------------------
# Objective functions
# -----------------------------
objectives = {
    "iae": iae,
    "ise": ise,
    "itae": itae,
    "itse": itse,
}

# -----------------------------
# MAIN PROCESS
# -----------------------------
if __name__ == "__main__":

    # ========== SHARED MEMORY ==========
    manager = Manager()

    progress = manager.dict()
    sub_progress = manager.dict()
    status = manager.dict()

    # ========== CONFIG ==========
    N_RUNS = 20
    max_iter = 300
    population_size = 100

    fitness_functions = list(SISKEN_BENCH.items())
    # fitness_functions = list(BENCHMARKS.items())

    # for f_idx, f in fitness_functions:
    for fun_name, fitness_fn in fitness_functions:
        # statistics container
        stats = {
            algo: {
                "best_fitness": [],
                "best_solutions": [],
                "curves": [],
            } for algo in algorithms
        }
        print(f"\nStarting optimization benchmarks on {fun_name}!\n")
        completed_runs = load_existing_checkpoints(fun_name, stats, algorithms)

        iterator = tqdm(range(N_RUNS), desc="Benchmark Runs", ncols=100)

        for run_id in iterator:

            if run_id in completed_runs:
                print(f"\nSkipping run {run_id} (already checkpointed).")
                continue

            print(f"\n🌀 Run {run_id+1}/{N_RUNS}")

            # Launch algorithm workers
            processes = []
            for algo_name, algo_runner in algorithms.items():
                p = Process(
                    target=algo_worker,
                    args=(
                        algo_name, algo_runner,
                        fitness_fn, objectives,
                        [-10, -10, -10],
                        [10, 10, 10],
                        population_size,
                        max_iter,
                        progress,
                        sub_progress,
                        status
                    )
                )
                p.start()
                processes.append(p)


            # ========== PROGRESS BARS ==========
            bars_algo = {}
            bars_obj = {}

            position = 0

            for algo_name in algorithms:

                # Algorithm-level bar
                bars_algo[algo_name] = tqdm(
                    total=len(objectives),
                    desc=f"[ALGO] {algo_name.upper()}",
                    position=position
                )
                position += 1
    
                # Objective bars — MUST BE INSIDE THIS LOOP!
                bars_obj[algo_name] = {}
                for obj_name in objectives:
                    bars_obj[algo_name][obj_name] = tqdm(
                        total=max_iter,
                        desc=f" > {obj_name.upper()}",
                        position=position
                    )
                    position += 1


            # ========== LIVE UPDATE LOOP ==========
            running = True
            while running:
                running = any(p.is_alive() for p in processes)

                for algo_name in algorithms:
                    bars_algo[algo_name].n = progress.get(algo_name, 0)
                    bars_algo[algo_name].refresh()

                    for obj_name in objectives:
                        key = (algo_name, obj_name)
                        step = sub_progress.get(key, 0)
                        bars_obj[algo_name][obj_name].n = step
                        bars_obj[algo_name][obj_name].refresh()

                time.sleep(0.05)

            # Wait for all processes
            for p in processes:
                p.join()        

            # ========== COLLECT RESULTS ==========
            history = {}

            for algo in algorithms:
                curves_per_obj = {}
                solutions_per_obj = {}
                fitness_per_obj = {}
                for obj_fun in objectives:
                    key = (algo, obj_fun)
                    res = status[key]
                    curves_per_obj[obj_fun] = res["curve"]
                    solutions_per_obj[obj_fun] = res["solution"]
                    fitness_per_obj[obj_fun] = res["fitness"]

                history[algo] = curves_per_obj
                stats[algo]["best_solutions"].append(solutions_per_obj)
                stats[algo]["best_fitness"].append(fitness_per_obj)
                stats[algo]["curves"].append(curves_per_obj)

            # ========== GENERATE CONVERGENCE ==========
            convergence_history = {}
            for algo in history:
                for obj_fun in history[algo]:
                    convergence_history.setdefault(obj_fun, {})
                    convergence_history[obj_fun][algo] = history[algo][obj_fun]

            for obj_fun in convergence_history:
                generate_convergence_plot(
                    convergence_history[obj_fun],
                    title=f"Convergence Plot [{obj_fun.upper()}] {fun_name.upper()} Run {run_id+1}",
                    filename=f"output/convergence_{fun_name}_{obj_fun}_run_{run_id+1}.png"
                )

            print(f"\nSaving checkpoint for run {run_id}...")
            save_checkpoint(run_id, stats, fun_name)
        
        # ========== FINAL REPORT ==========
        generate_statistical_report(stats, fun_name, filename=f"output/{fun_name}_statistical_report.txt")
        print("\nAll benchmarks completed. Reports and plots are saved.")
