"""Checkpoint persistence for one optimizer/objective/run artifact per file."""

import os
import pickle
from typing import Any, Dict, List

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoint_jobs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def ckpt_path(fun_name, run_id, algo_name, obj_name):
    return os.path.join(
        CHECKPOINT_DIR,
        f"{fun_name}_run_{run_id}_{algo_name}_{obj_name}.pkl",
    )


def save_job_checkpoint(fun_name, run_id, algo_name, obj_name, payload):
    with open(ckpt_path(fun_name, run_id, algo_name, obj_name), "wb") as fh:
        pickle.dump(payload, fh)


def job_done(fun_name, run_id, algo_name, obj_name):
    return os.path.exists(ckpt_path(fun_name, run_id, algo_name, obj_name))


def load_all_job_checkpoints(
    fun_name: str,
    algorithms: List[str],
    objectives: List[str],
) -> Dict[str, Any]:
    """Load completed artifacts sorted by run id for ETA/runtime modeling."""
    stats = {
        obj: {
            algo: {
                "best_fitness": [],
                "best_solutions": [],
                "curves": [],
                "durations": [],
                "function_evaluations": [],
                "run_ids": [],
                "seeds": [],
            }
            for algo in algorithms
        }
        for obj in objectives
    }
    temp = {obj: {algo: [] for algo in algorithms} for obj in objectives}

    prefix = f"{fun_name}_run_"
    for filename in os.listdir(CHECKPOINT_DIR):
        if not (filename.startswith(prefix) and filename.endswith(".pkl")):
            continue
        body = filename[len(prefix):-4]
        first_sep = body.find("_")
        if first_sep < 0:
            continue
        try:
            run_id = int(body[:first_sep])
        except ValueError:
            continue

        rest = body[first_sep + 1:]
        matched = None
        for obj in sorted(objectives, key=len, reverse=True):
            suffix = f"_{obj}"
            if rest.endswith(suffix):
                algo = rest[:-len(suffix)]
                if algo in algorithms:
                    matched = (obj, algo)
                break
        if matched is None:
            continue

        obj, algo = matched
        with open(os.path.join(CHECKPOINT_DIR, filename), "rb") as fh:
            temp[obj][algo].append((run_id, pickle.load(fh)))

    for obj in objectives:
        for algo in algorithms:
            for run_id, data in sorted(temp[obj][algo], key=lambda item: item[0]):
                bucket = stats[obj][algo]
                bucket["best_fitness"].append(data.get("best_fitness"))
                bucket["best_solutions"].append(data.get("best_solution"))
                bucket["curves"].append(data.get("curve"))
                bucket["durations"].append(data.get("duration"))
                bucket["function_evaluations"].append(data.get("function_evaluations"))
                bucket["run_ids"].append(run_id)
                bucket["seeds"].append(data.get("seed"))

    return stats
