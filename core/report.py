import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def generate_statistical_report(all_runs, filename="results/statistical_report.txt"):
    # Membuat struktur 'runs' yang lebih terorganisir
    runs = {}

# all_runs[algo][metrics] ini tuh list of object yang isinya ada value dari objective function

    # Mengatur runs berdasarkan obj_func -> algo -> metrics
    for algo in all_runs:
        print(f"Processing algorithm: {algo}")
        print(f"Raw data for algorithm {algo}: {all_runs[algo]}")
        for metrics in all_runs[algo]:
            print(f"  Processing metric: {metrics}")
            print(f"  Raw metric data: {all_runs[algo][metrics]}")
            for i, run in enumerate(all_runs[algo][metrics]):
                print(f"    Processing run: {i}: {run}")
                print(f"    Raw run data: {all_runs[algo][metrics]}")
                for obj_function in run:
                    print(f"      Processing objective function: {obj_function}")
                    if obj_function not in runs:
                        runs[obj_function] = {}
                    if metrics not in runs[obj_function]:
                        runs[obj_function][metrics] = {}
                    if algo not in runs[obj_function][metrics]:
                        runs[obj_function][metrics][algo] = []
                    runs[obj_function][metrics][algo].append(run[obj_function])
                    
    with open("output/raw_runs.txt", "w") as f:
        f.write(str(runs))

    # Menulis laporan statistik ke file
    with open(filename, "w") as f:
        f.write("Statistical Report of Metaheuristic Benchmarking\n")
        f.write("===============================================\n\n")

        # Iterasi untuk setiap fungsi objektif
        for func_name, func_data in runs.items():
            f.write(f"\n\n===== FUNCTION: {func_name.upper()} =====\n\n")
            print(f"Generating report for function: {func_name}")
            print(f"Function data: {func_data}")

            # Dictionary untuk menyimpan fitness dan solutions untuk setiap algoritma
            all_algorithms = set()
            for metrics_data in func_data.values():
                all_algorithms.update(metrics_data.keys())

            stats = {algo: {"best_fitness": [], "best_solutions": []} for algo in all_algorithms}

            # Get all metrics from func_data
            for metrics, metrics_data in func_data.items():
                print(f"  Processing metrics: {metrics}")
                print(f"  Raw metrics data: {metrics_data}")

                # Mengumpulkan data fitness dan solutions untuk setiap algoritma
                for algo, algo_data in metrics_data.items():
                    print(f"    Processing algorithm data: {algo}")
                    print(f"    Raw algorithm data: {algo_data}")
                    for run_data in algo_data:
                        print(f"      Processing run data: {run_data}")
                        print(f"      Raw run data: {run_data}")
                        stats[algo][metrics].append(run_data)

            # Menulis statistik untuk setiap algoritma
            for algo, algo_stats in stats.items():
                fitness_values = algo_stats["best_fitness"]
                solutions = algo_stats["best_solutions"]

                sorted_indices = np.argsort(fitness_values)
                sorted_fitness = np.array(fitness_values)[sorted_indices]
                sorted_solutions = np.array(solutions)[sorted_indices]

                best_fitness = sorted_fitness[0]
                worst_fitness = sorted_fitness[-1]
                mean_fitness = np.mean(fitness_values)
                std_fitness = np.std(fitness_values)

                best_solution = sorted_solutions[0]
                worst_solution = sorted_solutions[-1]
                mean_solution = np.mean(solutions, axis=0)
                std_solution = np.std(solutions, axis=0)

                f.write(f"Algorithm: {algo.upper()}\n")
                f.write(f"  Best Fitness: {best_fitness}\n")
                f.write(f"  Worst Fitness: {worst_fitness}\n")
                f.write(f"  Mean Fitness: {mean_fitness}\n")
                f.write(f"  Std Dev Fitness: {std_fitness}\n")
                f.write(f"  Best Solution: {best_solution}\n")
                f.write(f"  Worst Solution: {worst_solution}\n")
                f.write(f"  Mean Solution: {mean_solution}\n")
                f.write(f"  Std Dev Solution: {std_solution}\n\n")

                

def generate_convergence_plot(history, title, filename="results/convergence_plot.png"):
    print(f"Raw history data: {history}")
    plt.figure(figsize=(10, 5))
    for algo, curve in history.items():
        plt.plot(curve, label=algo.upper())
    plt.xlabel("Iteration")
    plt.ylabel("Fitness")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

    print(f"Convergence plot saved to {filename}")