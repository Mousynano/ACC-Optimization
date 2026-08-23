from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # tetap untuk printing dan return result
import random as pyrandom
from random import random
from tqdm import tqdm


class ArithmeticOptimizationAlgorithm:
    """
    Arithmetic Optimization Algorithm (AOA).

    Paper basis:
    - Abualigah et al., The Arithmetic Optimization Algorithm, 2021.

    Design follows the project PSO wrapper style:
    - xp / to_cpu backend
    - class-based optimizer
    - history["best_fitness"] and history["best_position"]
    - run_arithmetic_aoa(...) wrapper returning best_params, best_fitness, history
    """

    def __init__(self, fitness_function, min_params, max_params, maximize=False,
                 n_agents=100, max_iteration=300,
                 alpha=5.0, mu=0.5, moa_min=0.2, moa_max=0.9,
                 eps=1e-12, tolerance=1e-6, progress_callback=None):

        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)
        self.dimensions = len(min_params)

        self.n_agents = n_agents
        self.max_iteration = max_iteration
        self.maximize = maximize

        self.alpha = float(alpha)
        self.mu = float(mu)
        self.moa_min = float(moa_min)
        self.moa_max = float(moa_max)
        self.eps = float(eps)
        self.tolerance = tolerance
        self.progress_callback = progress_callback

        self.positions = self._initialize_population()
        self.fitness = [float("-inf") if maximize else float("inf") for _ in range(n_agents)]

        self.best_value = float("-inf") if maximize else float("inf")
        self.best_position = xp.zeros(self.dimensions)

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

        self._evaluate_population()
        self._save_history()

    def _initialize_population(self):
        positions = []
        for _ in range(self.n_agents):
            pos = self.min_params + (self.max_params - self.min_params) * xp.random.random(self.dimensions)
            positions.append(pos)
        return xp.array(positions)

    def _is_better(self, value, reference):
        return value > reference if self.maximize else value < reference

    def _evaluate_position(self, position):
        pos_cpu = to_cpu(position)
        value_cpu = self.fitness_function(pos_cpu)
        return float(value_cpu)

    def _evaluate_population(self):
        for i in range(self.n_agents):
            value = self._evaluate_position(self.positions[i])
            self.fitness[i] = value
            if self._is_better(value, self.best_value):
                self.best_value = value
                self.best_position = self.positions[i].copy()

    def _effective_l(self):
        # Original AOA uses L = (UB - LB) * mu + LB.
        # For symmetric bounds this may become exactly zero; fallback avoids a frozen search.
        original_l = (self.max_params - self.min_params) * self.mu + self.min_params
        fallback_l = (self.max_params - self.min_params) * self.mu
        return xp.where(xp.abs(original_l) < self.eps, fallback_l, original_l)

    def _move_agents(self, iteration):
        t = iteration + 1
        moa = self.moa_min + t * ((self.moa_max - self.moa_min) / max(self.max_iteration, 1))
        mop = 1.0 - ((t ** (1.0 / self.alpha)) / (max(self.max_iteration, 1) ** (1.0 / self.alpha)))
        l_vector = self._effective_l()

        new_positions = self.positions.copy()

        for i in range(self.n_agents):
            candidate = self.positions[i].copy()
            for j in range(self.dimensions):
                r1, r2, r3 = random(), random(), random()
                if r1 > moa:
                    # Exploration: division or multiplication around the best solution.
                    if r2 < 0.5:
                        candidate[j] = self.best_position[j] / (mop + self.eps) * l_vector[j]
                    else:
                        candidate[j] = self.best_position[j] * mop * l_vector[j]
                else:
                    # Exploitation: subtraction or addition around the best solution.
                    if r3 < 0.5:
                        candidate[j] = self.best_position[j] - mop * l_vector[j]
                    else:
                        candidate[j] = self.best_position[j] + mop * l_vector[j]

            new_positions[i] = xp.clip(candidate, self.min_params, self.max_params)

        self.positions = new_positions

    def _save_history(self):
        self.history["best_fitness"].append(float(self.best_value))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def _run_iter(self, iteration):
        self._move_agents(iteration)
        self._evaluate_population()
        self._save_history()

    def run(self, verbose=True):
        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with Arithmetic AOA",
            ncols=100,
            disable=not verbose
        )

        for iteration in iterator:
            self._run_iter(iteration)
            iterator.set_postfix({"Best": f"{float(self.best_value):.6f}"})

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(self.best_value)


def run_arithmetic_aoa(func, min_params, max_params, population_size,
                       max_iter=100, verbose=False, progress_callback=None,
                       seed=None, **kwargs):
    if seed is not None:
        pyrandom.seed(seed)
        np.random.seed(seed)
        try:
            xp.random.seed(seed)
        except Exception:
            pass

    model = ArithmeticOptimizationAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_agents=population_size,
        max_iteration=max_iter,
        maximize=False,
        progress_callback=progress_callback,
        **kwargs
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]


# Optional alias. Use carefully if another AOA file already exists in your project.
def run_aoa(func, min_params, max_params, population_size,
            max_iter=100, verbose=False, progress_callback=None, seed=None, **kwargs):
    return run_arithmetic_aoa(
        func, min_params, max_params, population_size,
        max_iter=max_iter, verbose=verbose,
        progress_callback=progress_callback, seed=seed, **kwargs
    )
