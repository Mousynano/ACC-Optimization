from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # tetap untuk printing dan return result
import random as pyrandom
from random import random
from tqdm import tqdm


class ChannaArgusOptimizer:
    """
    Channa Argus Optimizer (CAO).

    Paper basis:
    - Fang et al., Channa argus optimizer for solving numerical optimization
      and engineering problems, Scientific Reports, 2025.

    This implementation follows the paper's two update modes:
    - hunting / exploration strategy: Eq. (2)
    - escaping / exploitation strategy: Eq. (6) with K(t)

    Default mode is minimization for controller/objective tuning.
    """

    def __init__(self, fitness_function, min_params, max_params, maximize=False,
                 n_arguses=100, max_iteration=300,
                 exploration_probability="linear", greedy=True,
                 eps=1e-12, tolerance=1e-6, progress_callback=None):

        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)
        self.dimensions = len(min_params)

        self.n_arguses = int(n_arguses)
        self.max_iteration = int(max_iteration)
        self.maximize = maximize
        self.exploration_probability = exploration_probability
        self.greedy = bool(greedy)
        self.eps = float(eps)
        self.tolerance = tolerance
        self.progress_callback = progress_callback

        self.positions = self._initialize_population()
        self.fitness = [float("-inf") if maximize else float("inf") for _ in range(self.n_arguses)]

        self.best_value = float("-inf") if maximize else float("inf")
        self.best_position = xp.zeros(self.dimensions)

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

        self._evaluate_population()
        self._sort_population()
        self._save_history()

    def _initialize_population(self):
        return xp.array([
            self.min_params + (self.max_params - self.min_params) * xp.random.random(self.dimensions)
            for _ in range(self.n_arguses)
        ])

    def _is_better(self, value, reference):
        return value > reference if self.maximize else value < reference

    def _evaluate_position(self, position):
        pos_cpu = to_cpu(position)
        value_cpu = self.fitness_function(pos_cpu)
        return float(value_cpu)

    def _evaluate_population(self):
        for i in range(self.n_arguses):
            value = self._evaluate_position(self.positions[i])
            self.fitness[i] = value
            if self._is_better(value, self.best_value):
                self.best_value = value
                self.best_position = self.positions[i].copy()

    def _sort_population(self):
        fit_np = np.asarray(self.fitness, dtype=float)
        order = np.argsort(fit_np)
        if self.maximize:
            order = order[::-1]
        self.positions = self.positions[xp.array(order, dtype=int)]
        self.fitness = [self.fitness[int(i)] for i in order]
        if self._is_better(float(self.fitness[0]), self.best_value):
            self.best_value = float(self.fitness[0])
            self.best_position = self.positions[0].copy()

    def _centroid(self):
        return xp.mean(self.positions, axis=0)

    def _leader_centroid(self):
        n_leaders = max(1, int(np.ceil(self.n_arguses / 2.0)))
        return xp.mean(self.positions[:n_leaders], axis=0)

    def _select_elite_s(self):
        best = self.positions[0]
        second = self.positions[1] if self.n_arguses > 1 else self.positions[0]
        leader_center = self._leader_centroid()
        pick = int(np.random.randint(0, 3))
        if pick == 0:
            return best.copy()
        if pick == 1:
            return second.copy()
        return leader_center.copy()

    def _phase_probability(self, iteration):
        t_ratio = (iteration + 1) / max(self.max_iteration, 1)
        if self.exploration_probability == "linear":
            return max(0.0, 1.0 - t_ratio)
        if self.exploration_probability == "cosine":
            return 0.5 * (1.0 + np.cos(np.pi * t_ratio))
        if isinstance(self.exploration_probability, (float, int)):
            return float(np.clip(self.exploration_probability, 0.0, 1.0))
        return max(0.0, 1.0 - t_ratio)

    def _hunting_update(self, i):
        # Eq. (2): Zi(t+1) = S(t) + P(t) * (r*(G-Zi) + (1-r)*(Z-Zi))
        Zi = self.positions[i]
        G = self.best_position
        Z = self._centroid()
        S = self._select_elite_s()
        P_vec = xp.asarray(np.random.normal(0.0, 1.0, size=self.dimensions), dtype=float)
        r = random()
        return S + P_vec * (r * (G - Zi) + (1.0 - r) * (Z - Zi))

    def _escaping_update(self, i, iteration):
        # Eq. (6): Zi(t+1) = K(t)*G(t) + P(t)*(r*(G-Zi)+(1-r)*(Z-Zi))
        # Eq. (7): K(t)=0.1*(exp(t/maxIter)-1)
        Zi = self.positions[i]
        G = self.best_position
        Z = self._centroid()
        P_vec = xp.asarray(np.random.normal(0.0, 1.0, size=self.dimensions), dtype=float)
        r = random()
        K = 0.1 * (float(np.exp((iteration + 1) / max(self.max_iteration, 1))) - 1.0)
        return K * G + P_vec * (r * (G - Zi) + (1.0 - r) * (Z - Zi))

    def _move_arguses(self, iteration):
        self._sort_population()
        p = self._phase_probability(iteration)

        new_positions = self.positions.copy()
        new_fitness = list(self.fitness)

        for i in range(self.n_arguses):
            if random() < p:
                candidate = self._hunting_update(i)
            else:
                candidate = self._escaping_update(i, iteration)

            candidate = xp.clip(candidate, self.min_params, self.max_params)
            cand_fit = self._evaluate_position(candidate)

            if (not self.greedy) or self._is_better(cand_fit, float(self.fitness[i])):
                new_positions[i] = candidate
                new_fitness[i] = cand_fit

            if self._is_better(cand_fit, self.best_value):
                self.best_value = cand_fit
                self.best_position = candidate.copy()

        self.positions = new_positions
        self.fitness = new_fitness
        self._sort_population()

    def _save_history(self):
        self.history["best_fitness"].append(float(self.best_value))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def _run_iter(self, iteration):
        self._move_arguses(iteration)
        self._save_history()

    def run(self, verbose=True):
        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with Channa CAO",
            ncols=100,
            disable=not verbose
        )

        for iteration in iterator:
            self._run_iter(iteration)
            iterator.set_postfix({"Best": f"{float(self.best_value):.6f}"})

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(self.best_value)


def run_channa_cao(func, min_params, max_params, population_size,
                   max_iter=100, verbose=False, progress_callback=None,
                   seed=None, **kwargs):
    if seed is not None:
        pyrandom.seed(seed)
        np.random.seed(seed)
        try:
            xp.random.seed(seed)
        except Exception:
            pass

    model = ChannaArgusOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_arguses=population_size,
        max_iteration=max_iter,
        maximize=False,
        progress_callback=progress_callback,
        **kwargs
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]


# Optional alias. Use carefully if another CAO file already exists in your project.
def run_cao(func, min_params, max_params, population_size,
            max_iter=100, verbose=False, progress_callback=None, seed=None, **kwargs):
    return run_channa_cao(
        func, min_params, max_params, population_size,
        max_iter=max_iter, verbose=verbose,
        progress_callback=progress_callback, seed=seed, **kwargs
    )
