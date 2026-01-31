import pickle
import os

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoint")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def save_checkpoint(run_id, stats, fun_name):
    filename = f"{CHECKPOINT_DIR}/{fun_name}_run_{run_id}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(stats, f)

def load_existing_checkpoints(fun_name, stats, algorithms):
    completed = []

    # reset stats (anti double count)
    for algo in algorithms:
        stats[algo]["best_fitness"].clear()
        stats[algo]["best_solutions"].clear()
        stats[algo]["curves"].clear()

    for file in os.listdir(CHECKPOINT_DIR):
        if not (file.startswith(fun_name) and file.endswith(".pkl")):
            continue

        run_id = int(file.split("_run_")[1].split(".")[0])
        completed.append(run_id)

        with open(os.path.join(CHECKPOINT_DIR, file), "rb") as f:
            old_stats = pickle.load(f)

        for algo in algorithms:
            stats[algo]["best_fitness"] = old_stats[algo]["best_fitness"]
            stats[algo]["best_solutions"] = old_stats[algo]["best_solutions"]
            stats[algo]["curves"] = old_stats[algo]["curves"]

    return completed

