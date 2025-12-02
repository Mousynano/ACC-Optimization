import time
from multiprocessing import Process, Manager
from core.objective_worker import objective_worker

def algo_worker(algo_name, algo_runner, fitness_fn, objectives,
                min_params, max_params, population_size, max_iter,
                progress, sub_progress, status):

    # Init data structures
    progress[algo_name] = 0
    sub_progress[algo_name] = {obj: 0 for obj in objectives.keys()}
    status[algo_name] = {}

    processes = []
    for obj_name, obj_fn in objectives.items():
        p = Process(
            target=objective_worker,
            args=(
                algo_name, obj_name, algo_runner, fitness_fn, obj_fn,
                min_params, max_params, population_size, max_iter,
                sub_progress, status[algo_name]
            )
        )

        p.start()
        processes.append(p)

    # Monitor until all objectives finish
    finished = 0
    while finished < len(objectives):
        finished = sum(1 for obj in objectives if sub_progress[algo_name][obj] >= max_iter)
        progress[algo_name] = finished
        time.sleep(0.05)

    for p in processes:
        p.join()
