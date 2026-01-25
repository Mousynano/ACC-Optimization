import inspect

def objective_worker(
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
    status
):

    key = (algo_name, obj_name)

    def cb(step):
        sub_progress[key] = step

    def fitness_wrapper(params):
        if len(inspect.signature(fitness_fn).parameters) == 2:
            return fitness_fn(params, obj_fn)
        return fitness_fn(params)


    sol, best_fit, curve = algo_runner(
        func=fitness_wrapper,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        verbose=False,
        progress_callback=cb,
    )

    status[key] = {
        "solution": sol,
        "fitness": best_fit,
        "curve": curve,
    }

    sub_progress[key] = max_iter
