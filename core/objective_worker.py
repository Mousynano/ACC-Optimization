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
    status_for_algo
):

    # Callback → update ONE objective bar
    def cb(step):
        sub_progress[algo_name][obj_name] = step

    # Wrap the fitness function to embed objective
    def fitness_wrapper(params):
        return fitness_fn(params, obj_fn)   # <- your function from objectives dict

    # Run the algorithm for this objective only
    sol, best_fit, curve = algo_runner(
        func=fitness_wrapper,        # <─ use wrapper
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        verbose=False,
        progress_callback=cb         # <─ key line
    )

    status_for_algo[obj_name] = {
        "solution": sol,
        "fitness": best_fit,
        "curve": curve,
    }

    sub_progress[algo_name][obj_name] = max_iter
