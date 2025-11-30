import numpy as np
from tqdm import tqdm

from benchmarks.simple_funcs import BENCHMARKS as SIMPLE_BENCH
from benchmarks.sisken_prastiyanto import BENCHMARKS_ACC as SISKEN_BENCH

from optimizers.pso import run_pso
from optimizers.ga import run_ga
from optimizers.kma import run_kma
from optimizers.ska import run_ska
from optimizers.hoa import run_hoa

from core.report import generate_statistical_report, generate_convergence_plot

algorithms = {
    "kma": run_kma,
    "pso": run_pso,
    "ska": run_ska,
    "hoa": run_hoa,
    "ga": run_ga,
}

if __name__ == "__main__":
    N_RUNS = 2
    # fitness_fn = SISKEN_BENCH["acc"]
    fun_name = "rastrigin"
    fitness_fn = SIMPLE_BENCH[fun_name]

    stats = {
        algo: {
            "best_fitness": [],
            "best_solutions": [],
            "curves": [],
        } for algo in algorithms.keys()
    }

    iterator = tqdm(range(N_RUNS), desc="Benchmark Runs", ncols=100)
    print("Starting optimization benchmarks!")

    for run_id in iterator:
        history = {}
        print(f"\n🌀 Run {run_id+1}/{N_RUNS}")
        for algo_name, algo_runner in algorithms.items():
            print(f"\n🔷 Algorithm: {algo_name.upper()}")

            sol, best_fit, curve = algo_runner(
                func=fitness_fn,
                min_params=[-10, -10, -10],
                max_params=[10, 10, 10],
                population_size=100,
                max_iter=300,
                verbose=True
            )

            history[algo_name] = curve

            stats[algo_name]["best_solutions"].append(sol)
            stats[algo_name]["best_fitness"].append(best_fit)

        convergence_history = {}
        for algo in history:
            for obj_fun in history[algo]:
                if obj_fun not in convergence_history:
                    convergence_history[obj_fun] = {}
                if algo not in convergence_history[obj_fun]:
                    convergence_history[obj_fun][algo] = []
                convergence_history[obj_fun][algo] = history[algo][obj_fun]

        for obj_fun in convergence_history:
            generate_convergence_plot(convergence_history[obj_fun], title=f"Convergence Plot for {obj_fun.upper()} on {fun_name.upper()} - Run {run_id+1}", filename=f"output/convergence_{fun_name}_{obj_fun}_run_{run_id+1}.png")

    generate_statistical_report(stats, filename="output/statistical_report.txt")
    print("\nAll benchmarks completed. Reports and plots are saved in the 'output' folder.")
