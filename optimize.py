import time
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Manager

from benchmarks.sisken_prastiyanto import BENCHMARKS_ACC as SISKEN_BENCH

from optimizers.pso import run_pso
from optimizers.ga import run_ga
from optimizers.kma import run_kma
from optimizers.ska import run_ska
from optimizers.hoa import run_hoa

from core.report import generate_statistical_report, generate_convergence_plot
from core.algorithm_worker import algo_worker
from core.utils import iae, ise, itae, itse

# -----------------------------
# Algorithm registry
# -----------------------------
algorithms = {
    "kma": run_kma,
    "pso": run_pso,
    "ska": run_ska,
    "hoa": run_hoa,
    "ga": run_ga,
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

    progress.update({a: 0 for a in algorithms})
    sub_progress.update({
        a: {obj: 0 for obj in objectives} for a in algorithms
    })

    # ========== CONFIG ==========
    N_RUNS = 20
    fun_name = "acc"
    fitness_fn = SISKEN_BENCH[fun_name]

    population_size = 100
    max_iter = 300

    # statistics container
    stats = {
        algo: {
            "best_fitness": [],
            "best_solutions": [],
            "curves": [],
        } for algo in algorithms
    }

    iterator = tqdm(range(N_RUNS), desc="Benchmark Runs", ncols=100)
    print("Starting optimization benchmarks!")

    for run_id in iterator:

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

        # Algorithm-level bars
        for algo_name in algorithms:
            bars_algo[algo_name] = tqdm(
                total=len(objectives),
                desc=f"[ALGO] {algo_name.upper()}",
                position=position
            )
            position += 1

            # Objective bars
            bars_obj[algo_name] = {}
            for obj_name in objectives:
                bars_obj[algo_name][obj_name] = tqdm(
                    total=max_iter,
                    desc=f"   > {obj_name.upper()}",
                    position=position
                )
                position += 1

        # ========== LIVE UPDATE LOOP ==========
        running = True
        while running:
            running = any(p.is_alive() for p in processes)

            for algo_name in algorithms:
                bars_algo[algo_name].n = progress[algo_name]
                bars_algo[algo_name].refresh()

                for obj_name in objectives:
                    bars_obj[algo_name][obj_name].n = sub_progress[algo_name][obj_name]
                    bars_obj[algo_name][obj_name].refresh()

            time.sleep(0.05)

        # Wait for all processes
        for p in processes:
            p.join()

        # ========== COLLECT RESULTS ==========
        history = {}
        for algo in algorithms:
            res = status[algo]
            history[algo] = res["curve"]

            stats[algo]["best_solutions"].append(res["solutions"])
            stats[algo]["best_fitness"].append(res["fitness"])

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

    # ========== FINAL REPORT ==========
    generate_statistical_report(stats, filename="output/statistical_report.txt")
    print("\nAll benchmarks completed. Reports and plots are saved.")
