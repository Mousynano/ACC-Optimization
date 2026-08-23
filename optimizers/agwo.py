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


class ArchivedGreyWolfOptimizer:
    def __init__(self, fitness_function, min_params, max_params, maximize=False,
                 n_wolves=100, max_iteration=300, progress_callback=None, archive_size=4):

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

        # External archive: keeps the best solutions found so far.
        self.archive_size = max(1, int(archive_size))
        self.archive_positions = []
        self.archive_fitness = []

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


    def _fitness_as_float(self, value):
        return float(to_cpu(value))

    def _sort_archive_items(self, items):
        items = list(items)
        items.sort(key=lambda item: self._fitness_as_float(item[1]), reverse=self.maximize)
        return items

    def _sync_archive(self):
        """Update external archive with current population and keep only top archive_size."""
        items = [(agent.position.copy(), agent.value) for agent in self.wolves]
        items.extend([
            (pos.copy(), fit)
            for pos, fit in zip(self.archive_positions, self.archive_fitness)
        ])

        items = self._sort_archive_items(items)
        keep = min(self.archive_size, len(items))
        items = items[:keep]

        self.archive_positions = [pos.copy() for pos, _ in items]
        self.archive_fitness = [fit for _, fit in items]

        if items:
            self.best_position = self.archive_positions[0].copy()
            self.best_value = self.archive_fitness[0]

    def _inject_archive(self):
        """Elitist archiving: inject archived best solutions into the worst agents."""
        if not self.archive_positions:
            return

        ordered = list(range(len(self.wolves)))
        ordered.sort(
            key=lambda idx: self._fitness_as_float(self.wolves[idx].value),
            reverse=not self.maximize
        )

        for rank, idx in enumerate(ordered[:len(self.archive_positions)]):
            self.wolves[idx].position = self.archive_positions[rank].copy()
            self.wolves[idx].value = self.archive_fitness[rank]
            if hasattr(self.wolves[idx], "previous_position"):
                self.wolves[idx].previous_position = self.archive_positions[rank].copy()

    def _save_history(self):
        self.history["best_fitness"].append(float(to_cpu(self.best_value)))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def run(self, verbose=True):
        self._evaluate_wolves()
        self._sync_archive()
        self._inject_archive()
        self._save_history()

        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with AGWO",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            self._move_wolves(iteration)
            self._evaluate_wolves()
            self._sync_archive()
            self._inject_archive()
            self._save_history()

            iterator.set_postfix({
                "Best": f"{float(to_cpu(self.best_value)):.6f}"
            })

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(to_cpu(self.best_value))


def run_agwo(func, min_params, max_params, population_size,
            max_iter=100, verbose=False, progress_callback=None, seed=None, archive_size=4):

    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = ArchivedGreyWolfOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_wolves=population_size,
        max_iteration=max_iter,
        maximize=False,
        progress_callback=progress_callback,
        archive_size=archive_size,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
