import numpy as np
from tqdm import tqdm

from benchmarks.simple_funcs import BENCHMARKS as SIMPLE_BENCH
from benchmarks.sisken_prastiyanto import BENCHMARKS_ACC as SISKEN_BENCH

from optimizers.pso import run_pso
from optimizers.ga import run_ga
from optimizers.kma import run_kma
from optimizers.ska import run_ska
from optimizers.hoa import run_hoa

from core.report import generate_statisical_report, generate_convergence_plot

algorithms = {
    "kma": run_kma,
    "pso": run_pso,
    "ska": run_ska,
    "hoa": run_hoa,
    "ga": run_ga,
}

if __name__ == "__main__":
    N_RUNS = 1
    fitness_fn = SISKEN_BENCH["acc"]

    stats = {
        algo: {
            "best_fitness": {},
            "best_solutions": {}
        } for algo in algorithms.keys()
    }
    convergence_history = {}

    iterator = tqdm(range(N_RUNS), desc="Benchmark Runs", ncols=100)
    print("Starting optimization benchmarks!")

    for run_id in iterator:
        history = {}
        print(f"\n🌀 Run {run_id+1}/20")
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

            stats[algo_name]["best_solutions"] = sol
            stats[algo_name]["best_fitness"] = best_fit
            stats[algo_name]["curves"] = curve

        generate_convergence_plot(history, filename=f"output/convergence_run_{run_id+1}.png")
    generate_statisical_report(stats, filename="output/statistical_report.txt")
    print("\nAll benchmarks completed. Reports and plots are saved in the 'output' folder.")