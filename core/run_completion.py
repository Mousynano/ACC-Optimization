# core/run_completion.py
import os
import pickle
from typing import Dict, List

CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", "checkpoint_jobs")

def is_run_complete(fun_name: str, run_id: int, algorithms, objectives) -> bool:
    for algo in algorithms:
        for obj in objectives:
            fname = f"{fun_name}_run_{run_id}_{algo}_{obj}.pkl"
            if not os.path.exists(os.path.join(CHECKPOINT_DIR, fname)):
                return False
    return True


def load_run_curves(fun_name: str, run_id: int, algorithms, obj_name: str):
    curves = {}
    for algo in algorithms:
        path = os.path.join(CHECKPOINT_DIR, f"{fun_name}_run_{run_id}_{algo}_{obj_name}.pkl")
        with open(path, "rb") as f:
            payload = pickle.load(f)
        curves[algo] = payload["curve"]
    return curves

