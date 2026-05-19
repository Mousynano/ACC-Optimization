from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # tetap untuk printing dan return result
from tqdm import tqdm


class Wolf:
    def __init__(self, min_params, max_params, dimensions=3):
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.position = self.initialize(dimensions)
        self.value = xp.inf

    def initialize(self, x):
        return xp.array([
            self.min_params[i] + (self.max_params[i] - self.min_params[i]) * np.random.rand()
            for i in range(x)
        ])

    def move_to(self, new_position):
        self.position = xp.clip(new_position, self.min_params, self.max_params)


class GreyWolfOptimizer:
    def __init__(self, fitness_function, min_params, max_params, maximize=False,
                 n_wolves=100, max_iteration=300, progress_callback=None):

        self.fitness_function = fitness_function
        self.n_wolves = n_wolves
        self.max_iteration = max_iteration
        self.dimensions = len(min_params)

        self.wolves = [
            Wolf(min_params, max_params, self.dimensions)
            for _ in range(n_wolves)
        ]

        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)

        self.maximize = maximize
        self.best_value = -xp.inf if maximize else xp.inf
        self.best_position = xp.zeros(self.dimensions)

        self.alpha_position = xp.zeros(self.dimensions)
        self.beta_position = xp.zeros(self.dimensions)
        self.delta_position = xp.zeros(self.dimensions)

        self.alpha_value = -xp.inf if maximize else xp.inf
        self.beta_value = -xp.inf if maximize else xp.inf
        self.delta_value = -xp.inf if maximize else xp.inf

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

        self.progress_callback = progress_callback

    def _better(self, a, b):
        return a > b if self.maximize else a < b

    def _evaluate_wolves(self):
        self.alpha_value = -xp.inf if self.maximize else xp.inf
        self.beta_value = -xp.inf if self.maximize else xp.inf
        self.delta_value = -xp.inf if self.maximize else xp.inf

        for wolf in self.wolves:
            pos_cpu = to_cpu(wolf.position)
            value_cpu = self.fitness_function(pos_cpu)
            value = xp.asarray(value_cpu)
            wolf.value = value

            if self._better(value, self.alpha_value):
                self.delta_value = self.beta_value
                self.delta_position = self.beta_position.copy()

                self.beta_value = self.alpha_value
                self.beta_position = self.alpha_position.copy()

                self.alpha_value = value
                self.alpha_position = wolf.position.copy()

            elif self._better(value, self.beta_value):
                self.delta_value = self.beta_value
                self.delta_position = self.beta_position.copy()

                self.beta_value = value
                self.beta_position = wolf.position.copy()

            elif self._better(value, self.delta_value):
                self.delta_value = value
                self.delta_position = wolf.position.copy()

            if self._better(value, self.best_value):
                self.best_value = value
                self.best_position = wolf.position.copy()

    def _move_wolves(self, iteration):
        a = 2.0 - 2.0 * (iteration / max(1, self.max_iteration - 1))

        for wolf in self.wolves:
            r1 = xp.asarray(np.random.rand(self.dimensions))
            r2 = xp.asarray(np.random.rand(self.dimensions))
            A1 = 2 * a * r1 - a
            C1 = 2 * r2
            D_alpha = xp.abs(C1 * self.alpha_position - wolf.position)
            X1 = self.alpha_position - A1 * D_alpha

            r1 = xp.asarray(np.random.rand(self.dimensions))
            r2 = xp.asarray(np.random.rand(self.dimensions))
            A2 = 2 * a * r1 - a
            C2 = 2 * r2
            D_beta = xp.abs(C2 * self.beta_position - wolf.position)
            X2 = self.beta_position - A2 * D_beta

            r1 = xp.asarray(np.random.rand(self.dimensions))
            r2 = xp.asarray(np.random.rand(self.dimensions))
            A3 = 2 * a * r1 - a
            C3 = 2 * r2
            D_delta = xp.abs(C3 * self.delta_position - wolf.position)
            X3 = self.delta_position - A3 * D_delta

            new_position = (X1 + X2 + X3) / 3.0
            wolf.move_to(new_position)

    def _save_history(self):
        self.history["best_fitness"].append(float(to_cpu(self.best_value)))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def run(self, verbose=True):
        self._evaluate_wolves()
        self._save_history()

        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with GWO",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            self._move_wolves(iteration)
            self._evaluate_wolves()
            self._save_history()

            iterator.set_postfix({
                "Best": f"{float(to_cpu(self.best_value)):.6f}"
            })

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(to_cpu(self.best_value))


def run_gwo(func, min_params, max_params, population_size,
            max_iter=100, verbose=False, progress_callback=None, seed=None):

    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = GreyWolfOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_wolves=population_size,
        max_iteration=max_iter,
        maximize=False,
        progress_callback=progress_callback,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
