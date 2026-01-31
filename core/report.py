from typing import List, Dict, Any
import numpy as np
import os
import matplotlib.pyplot as plt
import csv

def generate_statistical_report(all_runs, fun_name, filename="results/statistical_report.txt"):
    """
    Generates report from aggregated data structure:
    all_runs[obj][algo] = {
        "best_fitness": [run0, run1...], 
        "best_solutions": [sol0, sol1...], 
        "curves": [...]
    }
    """
    
    # Pastikan folder output ada
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    print(f"Writing statistical report to {filename}...")

    with open(filename, "w") as f:
        f.write(f"Statistical Report of Metaheuristic Benchmarking on {fun_name.upper()}\n")
        f.write("===============================================\n")

        # 1. Loop per Objective Function (IAE, ISE, dll)
        for obj_name, algos_dict in all_runs.items():
            f.write(f"\n\n===== OBJECTIVE: {obj_name.upper()} =====\n")
            f.write("-----------------------------------------\n")
            print(f"Processing objective: {obj_name}")

            # 2. Loop per Algorithm (PSO, GA, dll)
            for algo_name, metrics_data in algos_dict.items():
                
                # Ambil list data
                fitness_values = metrics_data.get("best_fitness", [])
                solutions = metrics_data.get("best_solutions", [])
                
                # Cek jika data kosong
                if not fitness_values:
                    f.write(f"\nAlgorithm: {algo_name.upper()} (No Data)\n")
                    continue

                # Konversi ke Numpy Array untuk statistik
                fitness_arr = np.array(fitness_values)
                solutions_arr = np.array(solutions)

                # 3. Hitung Statistik
                # Cari index terbaik (Minimum Fitness)
                best_idx = np.argmin(fitness_arr)
                worst_idx = np.argmax(fitness_arr)

                best_fitness = fitness_arr[best_idx]
                worst_fitness = fitness_arr[worst_idx]
                mean_fitness = np.mean(fitness_arr)
                median_fitness = np.median(fitness_arr)  # <--- TAMBAHAN: Hitung Median
                std_fitness = np.std(fitness_arr)

                # Statistik Solution Vector
                best_solution = solutions_arr[best_idx]
                worst_solution = solutions_arr[worst_idx] 
                mean_solution = np.mean(solutions_arr, axis=0)
                std_solution = np.std(solutions_arr, axis=0)

                # Duration statistics (if available)
                durations = np.array(metrics_data.get("durations", []), dtype=float) if metrics_data.get("durations") else None
                if durations is not None and durations.size > 0:
                    total_time = np.sum(durations)
                    mean_time = np.mean(durations)
                    median_time = np.median(durations)
                    std_time = np.std(durations)
                else:
                    total_time = mean_time = median_time = std_time = None

                # 4. Tulis ke File
                f.write(f"\nAlgorithm: {algo_name.upper()}\n")
                f.write(f"  Best Fitness:     {best_fitness:.6f}\n")
                f.write(f"  Worst Fitness:    {worst_fitness:.6f}\n")
                f.write(f"  Mean Fitness:     {mean_fitness:.6f}\n")
                f.write(f"  Median Fitness:   {median_fitness:.6f}\n") # <--- TAMBAHAN: Tulis Median
                f.write(f"  Std Dev Fitness:  {std_fitness:.6f}\n")
                f.write(f"  -----------------------------------\n")
                f.write(f"  Best Solution:    {best_solution}\n")
                f.write(f"  Worst Solution:   {worst_solution}\n")
                f.write(f"  Mean Solution:    {mean_solution}\n")
                f.write(f"  Std Dev Solution: {std_solution}\n")
                if mean_time is not None:
                    f.write("  ----- Time Consumption (seconds) -----\n")
                    f.write(f"  Total Time:       {total_time:.3f}\n")
                    f.write(f"  Mean Time:        {mean_time:.3f}\n")
                    f.write(f"  Median Time:      {median_time:.3f}\n")
                    f.write(f"  Std Dev Time:     {std_time:.3f}\n")
                else:
                    f.write("  Time data:        (not available)\n")

    print("Report generation done.")

def generate_convergence_plot(history_dict, title, filename="results/convergence_plot.png"):
    """
    history_dict: dict { "algo_name": data }
    
    Fitur Baru:
    - Bisa menerima Multi-run (List of Lists) -> Dirata-rata
    - Bisa menerima Single-run (Flat List) -> Diplot langsung (Auto-wrap)
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    
    for algo, curves in history_dict.items():
        if not curves: continue
        
        # --- 1. AUTO-FIX DIMENSI (The Fix) ---
        # Cek elemen pertama. Jika itu angka (float/int/numpy number), berarti ini Flat List (1D).
        # Kita bungkus jadi List of List (2D) biar logic di bawah jalan.
        # Contoh: [100, 90, 80] -> [[100, 90, 80]]
        if isinstance(curves[0], (int, float, np.number)):
            curves = [curves] 
        
        # --- 2. VALIDASI ---
        # Pastikan isinya valid (punya len > 0)
        valid_curves = [c for c in curves if hasattr(c, '__len__') and len(c) > 0]
        
        if not valid_curves:
            print(f"[WARN] Skipping {algo}, invalid data structure.")
            continue

        # --- 3. PLOTTING ---
        try:
            # Cari panjang minimum (jika ada data yang tidak rata panjangnya)
            min_len = min(len(c) for c in valid_curves)
            
            # Potong data ke panjang minimum
            truncated_curves = [c[:min_len] for c in valid_curves]
            
            # Hitung Rata-rata 
            # Jika cuma 1 run (karena tadi di-wrap), mean-nya ya dirinya sendiri.
            mean_curve = np.mean(truncated_curves, axis=0)
            
            plt.plot(mean_curve, label=algo.upper(), linewidth=2)
            
        except Exception as e:
            print(f"[ERROR] Failed plotting {algo}: {e}")
            continue

    plt.xlabel("Iteration")
    plt.ylabel("Fitness (Cost)")
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    # print(f"Plot saved: {filename}")
    
def generate_time_plots(all_runs, fun_name, outdir="results"):
    """
    Generate bar charts of mean duration (seconds) per algorithm for each objective.
    """
    os.makedirs(outdir, exist_ok=True)

    for obj_name, algos_dict in all_runs.items():
        algos = []
        means = []
        for algo_name, metrics in algos_dict.items():
            durations = metrics.get("durations", [])
            if durations:
                algos.append(algo_name.upper())
                means.append(np.mean(durations))

        if not algos:
            print(f"[INFO] No duration data for objective {obj_name}")
            continue

        plt.figure(figsize=(8,4))
        x = np.arange(len(algos))
        plt.bar(x, means, color="C0")
        plt.xticks(x, algos)
        plt.ylabel("Mean Duration (s)")
        plt.title(f"Time Consumption [{obj_name.upper()}] {fun_name.upper()}")
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        fname = os.path.join(outdir, f"time_{fun_name}_{obj_name}.png")
        plt.savefig(fname)
        plt.close()
        print(f"Saved time plot: {fname}")
def export_results_to_csv(all_runs, filename="results/summary.csv"):
    """
    Export tabel summary ke CSV.
    Baris: Run ID
    Kolom: Objective, Algo, Fitness, Solution...
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    rows = []
    
    for obj_name, algos_dict in all_runs.items():
        for algo_name, metrics in algos_dict.items():
            fitnesses = metrics.get("best_fitness", [])
            solutions = metrics.get("best_solutions", [])
            
            for i, (fit, sol) in enumerate(zip(fitnesses, solutions)):
                dur = None
                if isinstance(metrics.get("durations"), (list, tuple)) and len(metrics.get("durations")) > i:
                    dur = metrics.get("durations")[i]

                row = {
                    "objective": obj_name,
                    "algorithm": algo_name,
                    "run_id": i,
                    "fitness": fit,
                    "duration": dur,
                    "solution": str(list(sol)) # Convert array to string biar masuk 1 cell
                }
                rows.append(row)
                
    if not rows:
        return

    keys = rows[0].keys()
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV exported to {filename}")