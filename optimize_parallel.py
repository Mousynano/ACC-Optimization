# Optimize_parallel.py
import os
import hashlib

import concurrent.futures

from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from tqdm import tqdm

from benchmarks.sisken_prastiyanto import BENCHMARKS_ACC as SISKEN_BENCH
from benchmarks.simple_funcs import BENCHMARKS as SIMPLE_BENCH

from optimizers.pso import run_pso
from optimizers.ga import run_ga
from optimizers.kma import run_kma
from optimizers.ska import run_ska
from optimizers.hoa import run_hoa

from core.utils import iae, ise, itae, itse, cappiello_iae, cappiello_ise, cappiello_itae, cappiello_itse
from core.report import generate_statistical_report, generate_convergence_plot
from core.job_worker import run_job
from core.checkpoint_jobs import save_job_checkpoint, job_done, load_all_job_checkpoints
from core.run_completion import is_run_complete, load_run_curves

from typing import Tuple, Dict, Any, List

# -----------------------------
# Registry
# -----------------------------
algorithms = {
    # "kma": run_kma,
    "pso": run_pso,
    # "ska": run_ska,
    # "hoa": run_hoa,
    # "ga": run_ga,
}

objectives = {
    # "iae": iae,
    # "ise": ise,
    # "itae": itae,
    "itse": itse,
    # "cappiello_iae": cappiello_iae,
    # "cappiello_ise": cappiello_ise,
    # "cappiello_itae": cappiello_itae,
    # "cappiello_itse": cappiello_itse
}

# -----------------------------
# Config
# -----------------------------
fitness_functions = SISKEN_BENCH.items()
# fitness_functions = SIMPLE_BENCH.items()
N_RUNS = 1
max_iter = 300
population_size = 100

seeds = [i + 1 for i in range(N_RUNS)]

MAX_CONCURRENT_ALGOS = 1
MIN_PARAMS = [-10, -10, -10]
MAX_PARAMS = [10, 10, 10]

def seed_for_run(fun_name: str, run_id: int, base_seed: int = 20260111) -> int:
    s = f"{base_seed}:{fun_name}:{run_id}".encode("utf-8")
    digest = hashlib.md5(s).hexdigest()  # stabil lintas run/machine
    return int(digest[:8], 16)  # 32-bit

def ensure_outdir():
    os.makedirs("output", exist_ok=True)

def make_jobs():
    jobs = []
    for fun_name, fitness_fn in fitness_functions:
        for run_id in range(N_RUNS):
            for algo_name in algorithms.keys():
                for obj_name, obj_fn in objectives.items():
                    if job_done(fun_name, run_id, algo_name, obj_name):
                        continue
                    jobs.append((fun_name, fitness_fn, run_id, algo_name, obj_name, obj_fn))
    return jobs

def job_progress_fraction(progress_dict, fun_name: str, run_id: int, algo_name: str, objectives: Dict[str, Any], max_iter: int) -> float:
    """
    Return 0.0..1.0
    Progress job dihitung sebagai mean progress dari semua objective.
    """
    if not objectives:
        return 0.0

    total = 0.0
    for obj_name in objectives.keys():
        key = (fun_name, run_id, algo_name, obj_name)
        step = progress_dict.get(key, 0)
        # clamp
        if step < 0:
            step = 0
        if step > max_iter:
            step = max_iter
        total += step / float(max_iter)

    return total / float(len(objectives))

def slot_progress_fraction(progress_dict, fun_name, run_id, algo_name, obj_name, max_iter):
    step = progress_dict.get((fun_name, run_id, algo_name, obj_name), 0)
    if step < 0: step = 0
    if step > max_iter: step = max_iter
    return step / float(max_iter)

def main():
    ensure_outdir()

    manager = Manager()
    progress = manager.dict()  # key: (fun,run,algo,obj) -> iter
    status = manager.dict()    # optional
    current_obj_dict = manager.dict()  # NEW: (fun, run, algo) -> obj_name

    jobs = make_jobs()
    if not jobs:
        print("All jobs already checkpointed. Generating final reports only...")
    else:
        print(f"Total pending jobs: {len(jobs)} (concurrency={MAX_CONCURRENT_ALGOS})")

        # progress bar overall
        overall = tqdm(total=len(jobs), desc="All Jobs", ncols=100)

    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_ALGOS) as ex:
        futures = {}          # fut -> (fun, run, algo)
        future_slot = {}      # fut -> slot_id (0..MAX_CONCURRENT-1)
        slot_current = {}     # slot_id -> (fun, run, algo)
        free_slots = list(range(MAX_CONCURRENT_ALGOS))
        job_iter = iter(jobs)

        # progress bar overall
        overall = tqdm(total=len(jobs), desc="All Jobs", ncols=100, position=0)

        # slot bars (position 1..MAX_CONCURRENT)
        slot_bars = []
        for i in range(MAX_CONCURRENT_ALGOS):
            bar = tqdm(
                total=max_iter,
                desc=f"Slot {i+1}",
                ncols=100,
                position=i+1,
                leave=False
            )

            bar.set_postfix_str("idle")
            slot_bars.append(bar)

        def submit_next_job(slot_id: int) -> bool:
            """Submit 1 job ke slot tertentu. Return True kalau ada job yang disubmit."""
            try:
                fun_name, fitness_fn, run_id, algo_name, obj_name, obj_fn = next(job_iter)
                seed = seed_for_run(fun_name, run_id)   # atau seed_for_run(fun_name, run_id, obj_name) kalau kamu mau beda per objective

                fut = ex.submit(
                    run_job,
                    fun_name, run_id, algo_name, obj_name, obj_fn,
                    algorithms[algo_name], fitness_fn,
                    MIN_PARAMS, MAX_PARAMS,
                    population_size, max_iter,
                    seed, progress, status, current_obj_dict
                )
            except StopIteration:
                return False
            
            futures[fut] = (fun_name, run_id, algo_name, obj_name)
            future_slot[fut] = slot_id
            slot_current[slot_id] = (fun_name, run_id, algo_name, obj_name)

            # reset slot bar
            sb = slot_bars[slot_id]
            sb.n = 0
            sb.total = max_iter
            sb.set_postfix_str(f"{algo_name} | {fun_name} | run {run_id+1} | {obj_name}")
            sb.refresh()

            return True

        # submit initial batch (isi semua slot semampunya)
        for _ in range(MAX_CONCURRENT_ALGOS):
            if not free_slots:
                break
            slot_id = free_slots.pop(0)
            ok = submit_next_job(slot_id)
            if not ok:
                free_slots.append(slot_id)
                break

        # loop utama: update slot bars sambil menunggu completion
        while futures:
            done, _ = concurrent.futures.wait(
                futures,
                timeout=0.2,
                return_when=concurrent.futures.FIRST_COMPLETED
            )

            # 1) refresh progress slot (yang sedang jalan)
            for slot_id, meta in list(slot_current.items()):
                fun_name, run_id, algo_name, obj_name = meta

                step = progress.get((fun_name, run_id, algo_name, obj_name), 0)

                # clamp biar aman
                if step < 0:
                    step = 0
                if step > max_iter:
                    step = max_iter

                sb = slot_bars[slot_id]
                delta = step - sb.n
                if delta > 0:
                    sb.update(delta)

                sb.set_postfix_str(
                    f"{algo_name} | {fun_name} | run {run_id+1} | {obj_name}"
                )



            # 2) handle futures yang selesai
            for fut in done:
                fun_name, run_id, algo_name, obj_name = futures.pop(fut)
                slot_id = future_slot.pop(fut, None)

                payload = fut.result()

                save_job_checkpoint(
                    payload["fun_name"], payload["run_id"], payload["algo_name"], payload["obj_name"],
                    {
                        "seed": seed_for_run(payload["fun_name"], payload["run_id"]),
                        "best_fitness": payload["best_fitness"],
                        "best_solution": payload["best_solution"],
                        "curve": payload["curve"],
                    }
                )

                # OPSI B: realtime merged plot
                if is_run_complete(fun_name, run_id, list(algorithms.keys()), list(objectives.keys())):
                    for obj_name in objectives.keys():
                        algo_curves = load_run_curves(fun_name, run_id, list(algorithms.keys()), obj_name)
                        generate_convergence_plot(
                            algo_curves,
                            title=f"Convergence [{obj_name.upper()}] {fun_name.upper()} Run {run_id+1}",
                            filename=f"output/convergence_{fun_name}_{obj_name}_run_{run_id+1}_ALL.png"
                        )


                overall.update(1)

                # slot selesai → tandai idle dulu
                if slot_id is not None:
                    sb = slot_bars[slot_id]
                    sb.n = 100
                    sb.refresh()
                    slot_current.pop(slot_id, None)

                    # submit job baru ke slot itu (kalau ada)
                    ok = submit_next_job(slot_id)
                    if not ok:
                        sb.set_postfix_str("idle")
                        sb.refresh()

        # rapihin bar
        overall.close()
        for sb in slot_bars:
            sb.close()


    # -----------------------------
    # Final aggregation + reports
    # -----------------------------
    for fun_name, _fitness_fn in fitness_functions:
        stats = load_all_job_checkpoints(
            fun_name,
            list(algorithms.keys()),
            list(objectives.keys())
        )
        generate_statistical_report(
            stats,
            fun_name,
            filename=f"output/{fun_name}_statistical_report.txt"
        )
    print("Done. Checkpoints, plots, and reports are in ./checkpoint_jobs and ./output")

if __name__ == "__main__":
    main()
