import matplotlib.pyplot as plt
import numpy as np

def generate_statisical_report(all_runs, filename="results/statistical_report.txt"):

    with open(filename, "w") as f:
        f.write("Statistical Report of Metaheuristic Benchmarking\n")
        f.write("===============================================\n\n")

        for func_name, runs in all_runs.items():
            f.write(f"\n\n===== FUNCTION: {func_name.upper()} =====\n\n")

            stats = {
                "kma": [],
                "pso": [],
                "ska": [],
                "hoa": [],
                "ga": [],
            }

            for run in runs:
                for algo in stats:
                    stats[algo].append(run[algo]["best_fitness"])

            # tulis statistik untuk setiap algoritma
            for algo, fitness_list in stats.items():
                arr = np.array(fitness_list)
                f.write(f"[{algo.upper()}]\n")
                f.write(f"  Mean Fitness:   {arr.mean():.6f}\n")
                f.write(f"  Std Dev:        {arr.std():.6f}\n")
                f.write(f"  Best Fitness:   {arr.max():.6f}\n")
                f.write(f"  Worst Fitness:  {arr.min():.6f}\n\n")

def generate_convergence_plot(history, filename="results/convergence_plot.png"):
    plt.figure(figsize=(10, 5))
    for algo, curve in history.items():
        plt.plot(curve, label=algo.upper())
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title("Convergence Comparison of Metaheuristic Algorithms")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

    print(f"Convergence plot saved to {filename}")