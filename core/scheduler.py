from core.seeding import seed_for_run
from core.job_worker import run_job


class JobScheduler:

    def __init__(
        self,
        jobs,
        algorithms,
        objectives,
        config,
        progress,
        status,
        current_obj_dict,
        progress_ui,
    ):
        self.jobs = jobs
        self.algorithms = algorithms
        self.objectives = objectives
        self.config = config

        self.progress = progress
        self.status = status
        self.current_obj_dict = current_obj_dict

        self.progress_ui = progress_ui

        self.futures = {}
        self.future_slot = {}
        self.slot_current = {}

        self.job_iter = iter(jobs)

    def submit_next_job(
        self,
        slot_id,
        executor,
    ):
        try:
            job = next(self.job_iter)

            seed = seed_for_run(
                job.fun_name,
                job.run_id
            )

            fut = executor.submit(
                run_job,
                job.fun_name,
                job.run_id,
                job.algo_name,
                job.obj_name,
                job.obj_fn,
                self.algorithms[job.algo_name],
                job.fitness_fn,
                self.config.min_params,
                self.config.max_params,
                self.config.population_size,
                self.config.max_iter,
                seed,
                self.progress,
                self.status,
                self.current_obj_dict,
            )

        except StopIteration:
            return False

        self.futures[fut] = job
        self.future_slot[fut] = slot_id
        self.slot_current[slot_id] = job
        
        self.progress_ui.assign_job(
            slot_id,
            job
        )

        # sb = self.progress_ui.get_bar(slot_id)

        # sb.n = 0
        # sb.total = self.config.max_iter

        # sb.set_postfix_str(
        #     f"{job.algo_name} | "
        #     f"{job.fun_name} | "
        #     f"run {job.run_id + 1} | "
        #     f"{job.obj_name}"
        # )

        # sb.refresh()

        return True