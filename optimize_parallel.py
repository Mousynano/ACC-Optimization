# Optimize_parallel.py
import argparse
from collections import defaultdict, deque
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import hashlib
import json
import os
# Prevent nested BLAS/OpenMP oversubscription inside ProcessPool workers.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import time
from multiprocessing import Manager

from tqdm import tqdm

from benchmarks.acc_cost_v2 import (
    BENCHMARKS_ACC_V2,
    development_cost_fitness as development_cost_fitness_fj,
    load_cost_profile as load_cost_profile_fj,
)
from benchmarks.acc_cost_fje_v2 import (
    BENCHMARKS_ACC_FJE_V2,
    development_cost_fitness as development_cost_fitness_fje,
    load_cost_profile as load_cost_profile_fje,
)
from core.scenarios_v2 import load_scenario_suite
from benchmarks.sisken_prastiyanto import BENCHMARKS_ACC as LEGACY_BENCH

from optimizers.ga import run_ga
from optimizers.fvrade import run_fvrade
from optimizers.rnegde import run_rnegde
from optimizers.nshade import run_nshade
from optimizers.de_nps import run_de_nps
from optimizers.de import run_de

from core.config import load_manifest
from core.utils import iae, ise, itae, itse
from core.job_worker import run_job
from core.checkpoint_jobs import save_job_checkpoint, job_done, load_all_job_checkpoints
from core.campaign_eta import (
    RuntimeModel,
    finish_timestamp,
    format_duration,
    schedule_makespan_seconds,
)


# -----------------------------
# Registry
# -----------------------------
# All six algorithms from the manuscript contract remain registered.
ALGORITHM_REGISTRY = {
    "ga": run_ga,
    "de": run_de,
    "fvrade": run_fvrade,
    "rnegde": run_rnegde,
    "nshade": run_nshade,
    "denps": run_de_nps,
}

LEGACY_OBJECTIVES = {
    "iae": iae,
    "ise": ise,
    "itae": itae,
    "itse": itse,
}

# -----------------------------
# Versioned experiment configuration
# -----------------------------
MANIFEST = load_manifest()
BENCHMARK_KEY = MANIFEST["benchmark"]
if BENCHMARK_KEY == "acc_cost_sahib_fje_v2":
    fitness_functions = [(BENCHMARK_KEY, BENCHMARKS_ACC_FJE_V2[BENCHMARK_KEY])]
    objectives = {"proposed_cost_fje": None}
    development_cost_fitness = development_cost_fitness_fje
    load_cost_profile = load_cost_profile_fje
    ACTIVE_COST_FRAMEWORK = True
elif BENCHMARK_KEY == "acc_cost_sahib_v2":
    fitness_functions = [(BENCHMARK_KEY, BENCHMARKS_ACC_V2[BENCHMARK_KEY])]
    objectives = {"proposed_cost": None}
    development_cost_fitness = development_cost_fitness_fj
    load_cost_profile = load_cost_profile_fj
    ACTIVE_COST_FRAMEWORK = True
else:
    if BENCHMARK_KEY not in LEGACY_BENCH:
        raise RuntimeError(f"Unknown benchmark key: {BENCHMARK_KEY}")
    fitness_functions = [(BENCHMARK_KEY, LEGACY_BENCH[BENCHMARK_KEY])]
    objectives = dict(LEGACY_OBJECTIVES)
    development_cost_fitness = None
    load_cost_profile = None
    ACTIVE_COST_FRAMEWORK = False

N_RUNS = int(MANIFEST["optimization"]["independent_runs"])
max_iter = int(MANIFEST["optimization"]["max_iterations"])
population_size = int(MANIFEST["optimization"]["population_size"])
BASE_SEED = int(MANIFEST["optimization"]["base_seed"])
MIN_PARAMS = list(MANIFEST["controller"]["min_bounds"])
MAX_PARAMS = list(MANIFEST["controller"]["max_bounds"])

configured_algorithms = list(MANIFEST["optimization"]["algorithms"])
unknown = [name for name in configured_algorithms if name not in ALGORITHM_REGISTRY]
if unknown:
    raise RuntimeError(f"Manifest references unknown algorithms: {unknown}")
algorithms = {name: ALGORITHM_REGISTRY[name] for name in configured_algorithms}
configured_objectives = list(MANIFEST["optimization"].get("objectives", objectives))
if set(configured_objectives) != set(objectives):
    raise RuntimeError(
        f"Manifest objectives {configured_objectives} do not match active registry {list(objectives)}"
    )

# Default conservatively to avoid 25 Python/Numba workers turning a laptop into
# oversubscribing a typical laptop. Override explicitly on a workstation/server.
_default_workers = max(1, min(6, (os.cpu_count() or 2) - 1))
MAX_CONCURRENT_ALGOS = int(os.getenv("MAX_CONCURRENT_ALGOS", str(_default_workers)))
if MAX_CONCURRENT_ALGOS < 1:
    raise RuntimeError("MAX_CONCURRENT_ALGOS must be >= 1")

def seed_for_run(fun_name: str, run_id: int, base_seed: int = BASE_SEED) -> int:
    s = f"{base_seed}:{fun_name}:{run_id}".encode("utf-8")
    digest = hashlib.md5(s).hexdigest()  # stable across machines/runs
    return int(digest[:8], 16)


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


def expected_job_count() -> int:
    """All jobs in the configured campaign, including already checkpointed jobs."""
    return len(fitness_functions) * N_RUNS * len(algorithms) * len(objectives)


def runtime_key(fun_name: str, algo_name: str, obj_name: str):
    return (fun_name, algo_name, obj_name)


def load_runtime_model() -> RuntimeModel:
    """Learn existing runtime evidence from completed checkpoints.

    This is intentionally separate from tqdm ETA.  Historical durations are learned
    per benchmark/algorithm/objective so e.g. DE-NPS is not assumed to cost the same
    as GA merely because they share the same outer iteration limit.
    """
    model = RuntimeModel(historical_weight=0.70)
    for fun_name, _fitness_fn in fitness_functions:
        stats = load_all_job_checkpoints(
            fun_name,
            list(algorithms.keys()),
            list(objectives.keys()),
        )
        for obj_name in objectives.keys():
            for algo_name in algorithms.keys():
                model.extend_completed(
                    runtime_key(fun_name, algo_name, obj_name),
                    stats[obj_name][algo_name].get("durations", []),
                )
    return model


def write_finished_campaign_snapshot(total_jobs: int):
    payload = {
        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
        "jobs_total": total_jobs,
        "jobs_completed": total_jobs,
        "jobs_running": 0,
        "jobs_queued": 0,
        "estimated_remaining_seconds": 0.0,
        "estimated_remaining": "0m",
        "estimated_finish_local": finish_timestamp(0.0),
        "eta_state": "COMPLETE",
        "forecast_model": "algorithm/objective historical median + live projection + parallel slot simulation",
        "concurrency": MAX_CONCURRENT_ALGOS,
    }
    with open("output/campaign_eta.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def main():
    ensure_outdir()

    parser = argparse.ArgumentParser(description="Run the ACC evolutionary benchmark.")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a tiny real-ACC integration test for every optimizer and exit",
    )
    args = parser.parse_args()

    if args.smoke_test:
        fun_name, fitness_fn = next(iter(fitness_functions))
        if ACTIVE_COST_FRAMEWORK:
            # Smoke tests intentionally use development-only scenarios and equal
            # temporary weights. They never depend on or freeze research weights.
            fitness_fn = lambda params, _obj: development_cost_fitness(params)
        print(f"Smoke test benchmark={fun_name}")
        failures = []
        for obj_name, obj_fn in objectives.items():
            print(f" objective={obj_name}")
            for algo_name, algo_runner in algorithms.items():
                try:
                    payload = run_job(
                        fun_name,
                        0,
                        algo_name,
                        obj_name,
                        obj_fn,
                        algo_runner,
                        fitness_fn,
                        MIN_PARAMS,
                        MAX_PARAMS,
                        8,
                        2,
                        123,
                        None,
                        None,
                        None,
                    )
                    sol = payload["best_solution"]
                    fit = float(payload["best_fitness"])
                    curve = payload["curve"]
                    fe = payload.get("function_evaluations")
                    valid_bounds = all(
                        lo <= float(x) <= hi
                        for x, lo, hi in zip(sol, MIN_PARAMS, MAX_PARAMS)
                    )
                    ok = len(sol) == 3 and len(curve) == 2 and valid_bounds and fit == fit
                    print(
                        f"  {algo_name:8s} ok={ok} fitness={fit:.6g} "
                        f"FE={fe} gains={sol}"
                    )
                    if not ok:
                        failures.append(f"{obj_name}/{algo_name}")
                except Exception as exc:
                    failures.append(f"{obj_name}/{algo_name}")
                    print(f"  {algo_name:8s} FAILED: {exc}")
        if failures:
            raise SystemExit(f"Smoke test failed: {', '.join(failures)}")
        print("All optimizer/objective smoke tests passed.")
        return

    if ACTIVE_COST_FRAMEWORK:
        # Fail once, before spawning worker processes, if calibration artifacts are
        # not frozen or were derived from a different scenario-manifest version.
        profile = load_cost_profile(require_frozen=True)
        suite = load_scenario_suite(
            MANIFEST["cost_framework"]["scenario_manifest_path"],
            development=False,
            require_frozen=True,
        )
        if profile.scenario_manifest_sha256 != suite.manifest_sha256:
            raise RuntimeError(
                "Frozen cost profile does not match the frozen scenario manifest. "
                "Re-run the calibration script declared by the active manifest; do not mix framework versions."
            )
        print(
            f"{MANIFEST['framework_version']} preflight OK | frozen scenario/cost profile | weights="
            f"{profile.weights.tolist()}"
        )

    jobs = make_jobs()
    total_jobs = expected_job_count()
    completed_at_start = total_jobs - len(jobs)
    runtime_model = load_runtime_model()

    if not jobs:
        print("All jobs already checkpointed. Refreshing campaign status only...")
        write_finished_campaign_snapshot(total_jobs)
    else:
        print(
            f"Campaign jobs: {total_jobs} total | {completed_at_start} completed | "
            f"{len(jobs)} pending (concurrency={MAX_CONCURRENT_ALGOS})"
        )

        manager = Manager()
        progress = manager.dict()  # (fun, run, algo, obj) -> iteration
        status = manager.dict()
        current_obj_dict = manager.dict()

        with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_ALGOS) as ex:
            futures = {}          # future -> (fun, run, algo, obj)
            future_slot = {}      # future -> slot index
            slot_current = {}     # slot -> (fun, run, algo, obj)
            slot_started_at = {}  # slot -> monotonic timestamp
            pending_queue = deque(jobs)

            # Dedicated stakeholder-facing completion forecast.  It is deliberately
            # independent from tqdm's generic ETA and learns runtime by algorithm/objective.
            campaign_line = tqdm(
                total=0,
                desc="Campaign ETA | warming up",
                position=0,
                bar_format="{desc}",
                leave=True,
            )

            overall = tqdm(
                total=total_jobs,
                initial=completed_at_start,
                desc="All Jobs",
                ncols=100,
                position=1,
            )

            slot_bars = []
            for i in range(MAX_CONCURRENT_ALGOS):
                bar = tqdm(
                    total=max_iter,
                    desc=f"Slot {i+1}",
                    ncols=100,
                    position=i + 2,
                    leave=False,
                )
                bar.set_postfix_str("idle")
                slot_bars.append(bar)

            def submit_next_job(slot_id: int) -> bool:
                try:
                    fun_name, fitness_fn, run_id, algo_name, obj_name, obj_fn = pending_queue.popleft()
                except IndexError:
                    return False

                seed = seed_for_run(fun_name, run_id)
                fut = ex.submit(
                    run_job,
                    fun_name,
                    run_id,
                    algo_name,
                    obj_name,
                    obj_fn,
                    algorithms[algo_name],
                    fitness_fn,
                    MIN_PARAMS,
                    MAX_PARAMS,
                    population_size,
                    max_iter,
                    seed,
                    progress,
                    status,
                    current_obj_dict,
                )

                futures[fut] = (fun_name, run_id, algo_name, obj_name)
                future_slot[fut] = slot_id
                slot_current[slot_id] = (fun_name, run_id, algo_name, obj_name)
                slot_started_at[slot_id] = time.monotonic()

                sb = slot_bars[slot_id]
                sb.n = 0
                sb.total = max_iter
                sb.set_postfix_str(
                    f"{algo_name} | {fun_name} | run {run_id + 1} | {obj_name}"
                )
                sb.refresh()
                return True

            for slot_id in range(MAX_CONCURRENT_ALGOS):
                if not submit_next_job(slot_id):
                    break

            last_eta_refresh = 0.0
            last_eta_json_write = 0.0
            ETA_REFRESH_SECONDS = 1.0
            ETA_JSON_SECONDS = 5.0
            MIN_LIVE_PROGRESS_STEPS = 5

            def refresh_campaign_eta(force: bool = False):
                nonlocal last_eta_refresh, last_eta_json_write
                now_mono = time.monotonic()
                if not force and (now_mono - last_eta_refresh) < ETA_REFRESH_SECONDS:
                    return
                last_eta_refresh = now_mono

                live_total = defaultdict(list)
                active_elapsed = {}
                active_fraction = {}

                for slot_id, meta in list(slot_current.items()):
                    fun_name, run_id, algo_name, obj_name = meta
                    step = int(progress.get((fun_name, run_id, algo_name, obj_name), 0))
                    step = max(0, min(step, max_iter))
                    elapsed = max(
                        0.0,
                        now_mono - slot_started_at.get(slot_id, now_mono),
                    )
                    fraction = step / float(max_iter) if max_iter else 0.0
                    active_elapsed[slot_id] = elapsed
                    active_fraction[slot_id] = fraction

                    if step >= MIN_LIVE_PROGRESS_STEPS and fraction > 0.0:
                        live_total[runtime_key(fun_name, algo_name, obj_name)].append(
                            elapsed / fraction
                        )

                active_remaining = []
                active_missing_prediction = False
                for slot_id, meta in list(slot_current.items()):
                    fun_name, _run_id, algo_name, obj_name = meta
                    key = runtime_key(fun_name, algo_name, obj_name)
                    elapsed = active_elapsed.get(slot_id, 0.0)
                    fraction = active_fraction.get(slot_id, 0.0)
                    current_projection = (
                        elapsed / fraction if fraction > 0.0 else None
                    )

                    predicted_total = runtime_model.predict_total_seconds(key, live_total)
                    if predicted_total is None:
                        predicted_total = current_projection
                    if predicted_total is None:
                        active_missing_prediction = True
                        continue

                    pace_remaining = 0.0
                    if current_projection is not None:
                        pace_remaining = max(0.0, current_projection - elapsed)

                    # Historical jobs may have been faster than a currently throttled one.
                    # Keep a small guard from the live pace so ETA cannot collapse absurdly.
                    remaining = max(
                        0.0,
                        predicted_total - elapsed,
                        0.25 * pace_remaining,
                    )
                    active_remaining.append(remaining)

                pending_durations = []
                pending_missing_prediction = False
                for (
                    fun_name,
                    _fitness_fn,
                    _run_id,
                    algo_name,
                    obj_name,
                    _obj_fn,
                ) in pending_queue:
                    predicted = runtime_model.predict_total_seconds(
                        runtime_key(fun_name, algo_name, obj_name),
                        live_total,
                    )
                    if predicted is None:
                        pending_missing_prediction = True
                        break
                    pending_durations.append(predicted)

                remaining_seconds = None
                if not active_missing_prediction and not pending_missing_prediction:
                    remaining_seconds = schedule_makespan_seconds(
                        active_remaining,
                        pending_durations,
                        MAX_CONCURRENT_ALGOS,
                    )

                completed_now = total_jobs - len(pending_queue) - len(futures)
                running_now = len(futures)
                queued_now = len(pending_queue)
                eta_text = format_duration(remaining_seconds)
                finish_text = finish_timestamp(remaining_seconds)
                live_count = sum(len(values) for values in live_total.values())
                history_count = runtime_model.historical_count()

                if not futures and not pending_queue:
                    eta_state = "COMPLETE"
                elif history_count >= len(algorithms):
                    eta_state = "STABLE"
                elif live_count > 0 or history_count > 0:
                    eta_state = "LIVE"
                else:
                    eta_state = "WARMING UP"

                campaign_line.set_description_str(
                    f"Campaign ETA | done {completed_now}/{total_jobs} | "
                    f"running {running_now} | queued {queued_now} | "
                    f"remaining {eta_text} | finish {finish_text} | {eta_state}"
                )
                campaign_line.refresh()

                if force or (now_mono - last_eta_json_write) >= ETA_JSON_SECONDS:
                    last_eta_json_write = now_mono
                    payload = {
                        "timestamp_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                        "jobs_total": total_jobs,
                        "jobs_completed": completed_now,
                        "jobs_running": running_now,
                        "jobs_queued": queued_now,
                        "estimated_remaining_seconds": remaining_seconds,
                        "estimated_remaining": eta_text,
                        "estimated_finish_local": finish_text,
                        "eta_state": eta_state,
                        "forecast_model": "algorithm/objective historical median + live projection + parallel slot simulation",
                        "historical_runtime_samples": history_count,
                        "live_runtime_projections": live_count,
                        "concurrency": MAX_CONCURRENT_ALGOS,
                    }
                    tmp_path = "output/campaign_eta.json.tmp"
                    final_path = "output/campaign_eta.json"
                    with open(tmp_path, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh, indent=2)
                    os.replace(tmp_path, final_path)

            while futures:
                done, _ = concurrent.futures.wait(
                    futures,
                    timeout=0.2,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )

                for slot_id, meta in list(slot_current.items()):
                    fun_name, run_id, algo_name, obj_name = meta
                    step = int(progress.get((fun_name, run_id, algo_name, obj_name), 0))
                    step = max(0, min(step, max_iter))
                    sb = slot_bars[slot_id]
                    delta = step - sb.n
                    if delta > 0:
                        sb.update(delta)
                    sb.set_postfix_str(
                        f"{algo_name} | {fun_name} | run {run_id + 1} | {obj_name}"
                    )

                refresh_campaign_eta()

                for fut in done:
                    _fun_name, _run_id, _algo_name, _obj_name = futures.pop(fut)
                    slot_id = future_slot.pop(fut, None)
                    payload = fut.result()

                    runtime_model.add_completed(
                        runtime_key(
                            payload["fun_name"],
                            payload["algo_name"],
                            payload["obj_name"],
                        ),
                        payload.get("duration"),
                    )

                    save_job_checkpoint(
                        payload["fun_name"],
                        payload["run_id"],
                        payload["algo_name"],
                        payload["obj_name"],
                        {
                            "seed": seed_for_run(payload["fun_name"], payload["run_id"]),
                            "best_fitness": payload["best_fitness"],
                            "best_solution": payload["best_solution"],
                            "curve": payload["curve"],
                            "duration": payload.get("duration"),
                            "function_evaluations": payload.get("function_evaluations"),
                        },
                    )

                    overall.update(1)

                    if slot_id is not None:
                        sb = slot_bars[slot_id]
                        sb.n = max_iter
                        sb.refresh()
                        slot_current.pop(slot_id, None)
                        slot_started_at.pop(slot_id, None)

                        if not submit_next_job(slot_id):
                            sb.set_postfix_str("idle")
                            sb.refresh()

            refresh_campaign_eta(force=True)
            campaign_line.close()
            overall.close()
            for sb in slot_bars:
                sb.close()

    print(
        "Done. Raw run artifacts are in ./checkpoint_jobs. "
        "Run analysis.ipynb for statistical, convergence, time-response, and robustness outputs."
    )


if __name__ == "__main__":
    main()
