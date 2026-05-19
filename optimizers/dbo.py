from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # tetap untuk printing dan return result
from tqdm import tqdm
import math


class DungBeetle:
    def __init__(self, min_params, max_params, dimensions=3):
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.position = self.initialize(dimensions)
        self.previous_position = self.position.copy()
        self.value = xp.inf

    def initialize(self, x):
        return xp.array([
            self.min_params[i] + (self.max_params[i] - self.min_params[i]) * np.random.rand()
            for i in range(x)
        ])

    def move_to(self, new_position):
        self.position = xp.clip(new_position, self.min_params, self.max_params)


class DungBeetleOptimizer:
    def __init__(self, fitness_function, min_params, max_params, maximize=False,
                 n_beetles=100, max_iteration=300,
                 k=0.1, b=0.3, S=0.5, progress_callback=None):

        self.fitness_function = fitness_function
        self.n_beetles = n_beetles
        self.max_iteration = max_iteration
        self.dimensions = len(min_params)

        self.beetles = [
            DungBeetle(min_params, max_params, self.dimensions)
            for _ in range(n_beetles)
        ]

        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)

        self.maximize = maximize
        self.best_value = -xp.inf if maximize else xp.inf
        self.best_position = xp.zeros(self.dimensions)
        self.worst_value = -xp.inf if maximize else xp.inf
        self.worst_position = xp.zeros(self.dimensions)

        self.k = k
        self.b = b
        self.S = S

        self.n_roll = max(1, int(round(0.20 * n_beetles)))
        self.n_brood = max(1, int(round(0.20 * n_beetles)))
        self.n_small = max(1, int(round(0.25 * n_beetles)))
        rest = n_beetles - self.n_roll - self.n_brood - self.n_small
        self.n_thief = max(1, rest)

        total = self.n_roll + self.n_brood + self.n_small + self.n_thief
        if total > n_beetles:
            self.n_thief -= (total - n_beetles)

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

        self.progress_callback = progress_callback

    def _better(self, a, b):
        return a > b if self.maximize else a < b

    def _fitness(self, position):
        value_cpu = self.fitness_function(to_cpu(position))
        return xp.asarray(value_cpu)

    def _evaluate_beetles(self):
        self.worst_value = -xp.inf if not self.maximize else xp.inf

        for beetle in self.beetles:
            beetle.value = self._fitness(beetle.position)

            if self._better(beetle.value, self.best_value):
                self.best_value = beetle.value
                self.best_position = beetle.position.copy()

            if (self.maximize and beetle.value < self.worst_value) or \
               ((not self.maximize) and beetle.value > self.worst_value):
                self.worst_value = beetle.value
                self.worst_position = beetle.position.copy()

    def _dynamic_bounds(self, center, R):
        lower = xp.maximum(center * (1 - R), self.min_params)
        upper = xp.minimum(center * (1 + R), self.max_params)
        return lower, upper

    def _accept(self, beetle, candidate):
        candidate = xp.clip(candidate, beetle.min_params, beetle.max_params)
        old_value = self._fitness(beetle.position)
        new_value = self._fitness(candidate)
        beetle.previous_position = beetle.position.copy()
        if self._better(new_value, old_value):
            beetle.move_to(candidate)
            beetle.value = new_value

    def _move_beetles(self, iteration):
        Xb = self.best_position.copy()
        Xstar = self.best_position.copy()
        Xw = self.worst_position.copy()
        R = 1 - (iteration + 1) / max(1, self.max_iteration)

        roll_end = self.n_roll
        brood_end = roll_end + self.n_brood
        small_end = brood_end + self.n_small

        for idx, beetle in enumerate(self.beetles):
            if idx < roll_end:
                if np.random.rand() < 0.9:
                    alpha = 1 if np.random.rand() < 0.5 else -1
                    delta_x = xp.abs(beetle.position - Xw)
                    candidate = beetle.position + alpha * self.k * beetle.previous_position + self.b * delta_x
                else:
                    theta = np.random.uniform(0, math.pi)
                    if abs(theta) < 1e-12 or abs(theta - math.pi / 2) < 1e-12 or abs(theta - math.pi) < 1e-12:
                        candidate = beetle.position.copy()
                    else:
                        candidate = beetle.position + math.tan(theta) * xp.abs(beetle.position - beetle.previous_position)
                self._accept(beetle, candidate)
                continue

            if idx < brood_end:
                Lb_star, Ub_star = self._dynamic_bounds(Xstar, R)
                b1 = xp.asarray(np.random.rand(self.dimensions))
                b2 = xp.asarray(np.random.rand(self.dimensions))
                candidate = Xstar + b1 * (beetle.position - Lb_star) + b2 * (beetle.position - Ub_star)
                self._accept(beetle, candidate)
                continue

            if idx < small_end:
                Lbb, Ubb = self._dynamic_bounds(Xb, R)
                C1 = np.random.normal()
                C2 = xp.asarray(np.random.rand(self.dimensions))
                candidate = beetle.position + C1 * (beetle.position - Lbb) + C2 * (beetle.position - Ubb)
                self._accept(beetle, candidate)
                continue

            g = xp.asarray(np.random.normal(size=self.dimensions))
            candidate = Xb + self.S * g * (xp.abs(beetle.position - Xstar) + xp.abs(beetle.position - Xb))
            self._accept(beetle, candidate)

    def _save_history(self):
        self.history["best_fitness"].append(float(to_cpu(self.best_value)))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def run(self, verbose=True):
        self._evaluate_beetles()
        self._save_history()

        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with DBO",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            self._move_beetles(iteration)
            self._evaluate_beetles()
            self._save_history()

            iterator.set_postfix({
                "Best": f"{float(to_cpu(self.best_value)):.6f}"
            })

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(to_cpu(self.best_value))


def run_dbo(func, min_params, max_params, population_size,
            max_iter=100, verbose=False, progress_callback=None, seed=None):

    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = DungBeetleOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_beetles=population_size,
        max_iteration=max_iter,
        maximize=False,
        progress_callback=progress_callback,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
