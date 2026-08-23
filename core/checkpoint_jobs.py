import os
import pickle
from typing import Dict, Any, List
from tqdm import tqdm

from core.seeding import seed_for_run

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoint_jobs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def ckpt_path(fun_name, run_id, algo_name, obj_name):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{fun_name}_run_{run_id}_{algo_name}_{obj_name}.pkl")

def save_job_checkpoint(fun_name, run_id, algo_name, obj_name, payload):
    path = ckpt_path(fun_name, run_id, algo_name, obj_name)
    with open(path, "wb") as f:
        pickle.dump(payload, f)

def job_done(fun_name, run_id, algo_name, obj_name):
    return os.path.exists(ckpt_path(fun_name, run_id, algo_name, obj_name))

def load_all_job_checkpoints(fun_name: str, algorithms: List[str], objectives: List[str]) -> Dict[str, Any]:
    """
    Returns aggregated structure:
    stats[objective][algo]["best_fitness"] = list sorted by run_id
    stats[objective][algo]["best_solutions"] = list sorted by run_id
    stats[objective][algo]["curves"] = list sorted by run_id
    """

    stats = {}
    for obj in objectives:
        stats[obj] = {}
        for algo in algorithms:
            stats[obj][algo] = {"best_fitness": [], "best_solutions": [], "curves": [], "durations": []}

    # 2. Siapkan penampung sementara untuk sorting: temp[obj][algo] = [(run_id, payload), ...]
    temp_storage = {obj: {algo: [] for algo in algorithms} for obj in objectives}

    # 3. Scan file
    if os.path.exists(CHECKPOINT_DIR):
        files = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith(f"{fun_name}_run_") and f.endswith(".pkl")]

        for fn in files:
            try:
                # Parse filename: "acc_run_19_ska_itse.pkl"
                # Ambil bagian setelah "..._run_" dan buang ".pkl"
                part = fn.split("_run_")[1].replace(".pkl", "")
                tokens = part.split("_")
                run_id = int(tokens[0])
                obj = tokens[-1]
                algo = "_".join(tokens[1:-1])
                
                # Filter sesuai request
                if obj in objectives and algo in algorithms:
                    with open(os.path.join(CHECKPOINT_DIR, fn), "rb") as f:
                        payload = pickle.load(f)
                    temp_storage[obj][algo].append((run_id, payload))

            except Exception as e:
                print(f"Skipping file {fn}: {e}")

    # 4. Sorting & Flattening
    for obj in objectives:
        for algo in algorithms:
            # Sort berdasarkan run_id (index 0) agar urutan data benar
            runs = sorted(temp_storage[obj][algo], key=lambda x: x[0])
            
            for _, data in runs:
                stats[obj][algo]["best_fitness"].append(data.get("best_fitness"))
                # Note: Di pickle key-nya 'best_solution' (singular), kita simpan ke list 'best_solutions' (plural)
                stats[obj][algo]["best_solutions"].append(data.get("best_solution"))
                stats[obj][algo]["curves"].append(data.get("curve"))
                # tambahkan durasi eksekusi (detik)
                stats[obj][algo]["durations"].append(data.get("duration"))

    return stats

def create_placeholder_checkpoints(jobs, min_params, max_params):
    for (
        fun_name,
        fitness_fn,
        run_id,
        algo_name,
        obj_name,
        obj_fn,
    ) in tqdm(jobs):

        seed = seed_for_run(fun_name, run_id)

        midpoint = [
            (a + b) / 2
            for a, b in zip(min_params, max_params)
        ]

        save_job_checkpoint(
            fun_name,
            run_id,
            algo_name,
            obj_name,
            {
                "seed": seed,
                "best_fitness": float("inf"),
                "best_solution": midpoint,
                "curve": [],
                "duration": 0.0,
            },
        )

