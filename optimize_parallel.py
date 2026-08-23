# Optimize_parallel.py
import os

import concurrent.futures
import argparse

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager
from tqdm import tqdm

# from benchmarks.sisken_prastiyanto_wltc import (
#     use_wltc_scenarios,
#     acc_fitness_func,
#     acc_constraint_evaluator,
# )

from benchmarks.simple_funcs import BENCHMARKS as SIMPLE_BENCH

from core.scenarios import use_wltc_scenarios
from benchmarks.benchmark import acc_fopid_fitness_func, acc_constraint_evaluator

from optimizers.aoa import run_aoa
from optimizers.iaoa import run_iaoa
from optimizers.apo import run_apo
from optimizers.aos import run_aos
from optimizers.koa import run_koa
from optimizers.cao import run_cao
from optimizers.tso import run_tso
from optimizers.loa_v2 import run_loa_v2
from optimizers.loa_v3 import run_loa_v3
from optimizers.pso import run_pso
from optimizers.ska import run_ska
from optimizers.hoa import run_hoa
from optimizers.kma import run_kma
from optimizers.gwo import run_gwo
from optimizers.dbo import run_dbo
from optimizers.reo import run_reo
from optimizers.rime import run_rime
from optimizers.sa_da_rime import run_sa_da_rime
from optimizers.csa import run_csa
from optimizers.acsa import run_acsa
from optimizers.adbo import run_adbo
from optimizers.agwo import run_agwo
from optimizers.akoa import run_akoa
from optimizers.atso import run_atso
from optimizers.areo import run_areo

from core.utils import iae, ise, itae, itse, cappiello_iae, cappiello_ise, cappiello_itae, cappiello_itse
from core.report import generate_statistical_report, generate_convergence_plot, generate_time_plots
from core.checkpoint_jobs import save_job_checkpoint, load_all_job_checkpoints, create_placeholder_checkpoints
from core.run_completion import is_run_complete, load_run_curves
from core.seeding import seed_for_run
from core.job_worker import make_jobs
from core.progress_manager import ProgressManager
from core.scheduler import JobScheduler

from configs.run_configs import OptimizationConfig

# -----------------------------
# Registry
# -----------------------------
algorithms = {
    # "TSO": run_tso
    # "AOA": run_aoa,
    # "IAOA": run_iaoa,
    # "APO": run_apo,
    # # "AOS": run_aos,
    # # "KOA": run_koa,
    # # "CAO": run_cao,
    # # "LOA_V3": run_loa_v3,
    # # "LOA_V2": run_loa_v2,
    # # "PSO": run_pso,
    # # "SKA": run_ska,
    # # "HOA": run_hoa,
    # # "KMA": run_kma,
    # "RIME": run_rime,
    # "GWO": run_gwo,
    # # "DBO": run_dbo,
    # # "REO": run_reo,
    # "SADARIME": run_sa_da_rime
    "CSA": run_csa,
    "ACSA": run_acsa,
    "ADBO": run_adbo,
    "AGWO": run_agwo,
    "AKOA": run_akoa,
    "ATSO": run_atso,
    "AREO": run_areo,
}

objectives = {
    # "iae": iae,
    "ise": ise,
    # "itae": itae,
    # "itse": itse,
    # "iae": cappiello_iae,
    # "ise": cappiello_ise,
    # "itae": cappiello_itae,
    # "itse": cappiello_itse
}

# -----------------------------
# Config
# -----------------------------

# For ACC benchmarking
# fitness_functions = [("acc", acc_fitness_func)]

# For latest ACC benchmarking
# fitness_functions = [("acc_fopid", acc_fopid_fitness_func)]

# For mathematical benchmarking
fitness_functions = [
    ("rastrigin", SIMPLE_BENCH["rastrigin"]),
    ("ackley", SIMPLE_BENCH["ackley"]),
    ("sphere", SIMPLE_BENCH["sphere"]),
    ("rosenbrock", SIMPLE_BENCH["rosenbrock"]),
]

config = OptimizationConfig()

seeds = [i + 1 for i in range(config.n_runs)]


# This is obselete since benchmark.py already handles each controller rqeuirements
# # This is used in Classical ACC
# config.min_params = [-10, -10, -10]
# config.max_params = [10, 10, 10]

# # This is for FOPID params
config.min_params = [0.0, 0.0, 0.0, 0.10, 0.00, 0.8,  5.0]
config.max_params = [5.0, 2.0, 2.0, 2.00, 2.00, 2.5, 40.0]

# # And this is for augmented FOPID params (with Kvrel)
# # config.min_params = [0.0, 0.0, 0.0, 0.10, 0.00, 0.8,  5.0, 0.0]
# # config.max_params = [5.0, 2.0, 2.0, 2.00, 2.00, 2.5, 40.0, 5.0]

use_wltc_scenarios("wltc_class3b.csv", dt=1/60, mode="train")

# Use this code below to check whether or not the wltc cases have been applied or not
# from acc_refactor import benchmark
# print("Active scenarios seen by benchmark:")
# print([scenario["name"] for scenario in benchmark.SCENARIOS])

def main():
    os.makedirs("output", exist_ok=True)

    parser = argparse.ArgumentParser(description="Run optimizations or create placeholder checkpoints.")
    parser.add_argument("--create-checkpoints-only", action="store_true", help="Create checkpoint .pkl files for all jobs without running optimizations")
    args = parser.parse_args()

    manager = Manager()
    progress = manager.dict()  # key: (fun,run,algo,obj) -> iter
    status = manager.dict()    # optional
    current_obj_dict = manager.dict()  # NEW: (fun, run, algo) -> obj_name

    jobs = make_jobs(fitness_functions, algorithms, objectives, config.n_runs)
    if not jobs:
        print("All jobs already checkpointed. Generating final reports only...")
    else:
        print(f"Total pending jobs: {len(jobs)} (concurrency={config.max_concurrent_jobs})")

        # If user asked to only create checkpoint files, do that and skip running algos
        if args.create_checkpoints_only:
            print("Creating placeholder checkpoint files for pending jobs...")
            create_placeholder_checkpoints(jobs, config.min_params, config.max_params)
        else:
            # progress bar overall
            overall = tqdm(total=len(jobs), desc="All Jobs", ncols=100)

    with ProcessPoolExecutor(max_workers=config.max_concurrent_jobs) as ex:
        progress_ui = ProgressManager(
            config.max_concurrent_jobs,
            config.max_iter
        )

        scheduler = JobScheduler(
            jobs,
            algorithms,
            objectives,
            config,
            progress,
            status,
            current_obj_dict,
            progress_ui,
        )

        overall = tqdm(
            total=len(jobs),
            desc="All Jobs",
            ncols=100,
            position=0
        )

        with ProcessPoolExecutor(
            max_workers=config.max_concurrent_jobs
        ) as ex:

            free_slots = list(
                range(config.max_concurrent_jobs)
            )

            # initial submit
            for _ in range(config.max_concurrent_jobs):

                if not free_slots:
                    break

                slot_id = free_slots.pop(0)

                ok = scheduler.submit_next_job(
                    slot_id,
                    ex
                )

                if not ok:
                    free_slots.append(slot_id)
                    break

            while scheduler.futures:

                done, _ = concurrent.futures.wait(
                    scheduler.futures,
                    timeout=0.2,
                    return_when=concurrent.futures.FIRST_COMPLETED
                )

                # update progress bar
                for slot_id, job in list(
                    scheduler.slot_current.items()
                ):

                    step = progress.get(
                        (
                            job.fun_name,
                            job.run_id,
                            job.algo_name,
                            job.obj_name,
                        ),
                        0,
                    )

                    step = max(
                        0,
                        min(step, config.max_iter)
                    )

                    progress_ui.update_slot(
                        slot_id,
                        step
                    )

                # handle completed jobs
                for fut in done:

                    job = scheduler.futures.pop(fut)

                    slot_id = scheduler.future_slot.pop(
                        fut,
                        None
                    )

                    payload = fut.result()

                    save_job_checkpoint(
                        payload["fun_name"],
                        payload["run_id"],
                        payload["algo_name"],
                        payload["obj_name"],
                        {
                            "seed": seed_for_run(
                                payload["fun_name"],
                                payload["run_id"]
                            ),
                            "best_fitness": payload["best_fitness"],
                            "best_solution": payload["best_solution"],
                            "curve": payload["curve"],
                            "duration": payload.get(
                                "duration"
                            ),
                        }
                    )

                    if is_run_complete(
                        job.fun_name,
                        job.run_id,
                        list(algorithms.keys()),
                        list(objectives.keys())
                    ):
                        for obj_name in objectives.keys():

                            algo_curves = load_run_curves(
                                job.fun_name,
                                job.run_id,
                                list(algorithms.keys()),
                                obj_name
                            )

                            generate_convergence_plot(
                                algo_curves,
                                title=(
                                    f"Convergence "
                                    f"[{obj_name.upper()}] "
                                    f"{job.fun_name.upper()} "
                                    f"Run {job.run_id+1}"
                                ),
                                filename=(
                                    f"output/convergence_"
                                    f"{job.fun_name}_"
                                    f"{obj_name}_"
                                    f"run_{job.run_id+1}_ALL.png"
                                )
                            )

                    overall.update(1)

                    if slot_id is not None:

                        progress_ui.complete_slot(
                            slot_id
                        )

                        scheduler.slot_current.pop(
                            slot_id,
                            None
                        )

                        ok = scheduler.submit_next_job(
                            slot_id,
                            ex
                        )

                        if not ok:
                            progress_ui.set_idle(
                                slot_id
                            )

        overall.close()
        progress_ui.close()

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
