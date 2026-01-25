import time
from multiprocessing import Process
from core.objective_worker import objective_worker

def algo_worker(
    algo_name,
    algo_runner,
    fitness_fn,
    objectives,
    min_params,
    max_params,
    population_size,
    max_iter,
    progress,
    sub_progress,
    status
):

    # init progress
    for obj_name in objectives:
        sub_progress[(algo_name, obj_name)] = 0

    progress[algo_name] = 0

    processes = []
    for obj_name, obj_fn in objectives.items():
        p = Process(
            target=objective_worker,
            args=(
                algo_name,
                obj_name,
                algo_runner,
                fitness_fn,
                obj_fn,
                min_params,
                max_params,
                population_size,
                max_iter,
                sub_progress,
                status,
            ),
        )
        p.start()
        processes.append(p)

    total = len(objectives)
    finished_prev = -1

    while True:
        finished = sum(
            1 for obj_name in objectives
            if sub_progress[(algo_name, obj_name)] >= max_iter
        )

        progress[algo_name] = finished

        if finished != finished_prev:
            finished_prev = finished

        if finished >= total:
            break

        time.sleep(0.05)

    for p in processes:
        p.join()

