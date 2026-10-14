from core.xp import xp, to_cpu
import numpy as np
from tqdm import tqdm


class RNEGDEOptimizer:
    """Random Neighbor Elite Guided Differential Evolution (RNEGDE).

    Implements the paper's DE/current-to-rnbest/1 mutation, JADE_Sort-style
    adaptive F/CR, adaptive Cauchy-distributed neighbor size, binomial crossover,
    and external archive while exposing the project's standard optimizer API.
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        population_size=100,
        max_iter=30,
        maximize=False,
        c=0.1,
        progress_callback=None,
        seed=None,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)
        self.population_size = int(population_size)
        self.max_iter = int(max_iter)
        self.dim = len(min_params)
        self.maximize = maximize
        self.c = float(c)
        self.progress_callback = progress_callback
        self.seed = seed

        self.mu_f = 0.5
        self.mu_cr = 0.5
        self.mu_ns = None
        self.archive = []

        self.history = {"best_fitness": [], "best_position": []}
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            try:
                xp.random.seed(seed)
            except Exception:
                pass

    def _better(self, a, b):
        return a > b if self.maximize else a < b

    def _evaluate(self, position):
        return float(self.fitness_function(to_cpu(position)))

    def _initialize_population(self):
        low = np.asarray(to_cpu(self.min_params), dtype=float)
        high = np.asarray(to_cpu(self.max_params), dtype=float)
        pop = self._rng.uniform(low=low, high=high, size=(self.population_size, self.dim))
        return xp.asarray(pop)

    def _sample_f(self):
        # Paper: regenerate F if outside [0,1].
        for _ in range(1000):
            f = self.mu_f + 0.1 * self._rng.standard_cauchy()
            if 0.0 <= f <= 1.0:
                return float(f)
        return float(np.clip(self.mu_f, 0.0, 1.0))

    def _sample_cr(self):
        # Paper: regenerate CR if outside [0,1].
        for _ in range(1000):
            cr = self._rng.normal(self.mu_cr, 0.1)
            if 0.0 <= cr <= 1.0:
                return float(cr)
        return float(np.clip(self.mu_cr, 0.0, 1.0))

    def _sample_ns(self):
        lower = 2
        upper = max(lower, int(np.floor(0.3 * self.population_size)))
        for _ in range(2000):
            ns = int(np.rint(self.mu_ns + self._rng.standard_cauchy()))
            if lower <= ns <= upper:
                return ns
        return int(np.clip(round(self.mu_ns), lower, upper))

    def _pick_population_index(self, excluded):
        choices = [i for i in range(self.population_size) if i not in excluded]
        if not choices:
            raise RuntimeError("Population too small for RNEGDE mutation")
        return int(self._rng.choice(choices))

    def _pick_union_vector(self, population, excluded_population_indices):
        entries = []
        for i in range(self.population_size):
            if i not in excluded_population_indices:
                entries.append((population[i], float("nan"), "population", i))
        for i, (vec, fit) in enumerate(self.archive):
            entries.append((vec, fit, "archive", i))
        if not entries:
            raise RuntimeError("No P union A candidate available")
        return entries[int(self._rng.integers(0, len(entries)))]

    def _lehmer_mean(self, values):
        values = np.asarray(values, dtype=float)
        if values.size == 0 or values.sum() <= 0:
            return None
        return float(np.sum(values * values) / np.sum(values))

    def _binomial_crossover(self, target, mutant, cr):
        j_rand = int(self._rng.integers(0, self.dim))
        mask = self._rng.random(self.dim) <= cr
        mask[j_rand] = True
        trial = xp.where(xp.asarray(mask), mutant, target)
        return xp.clip(trial, self.min_params, self.max_params)

    def _save_history(self, population, fitness):
        fit = np.asarray(to_cpu(fitness), dtype=float)
        best_idx = int(np.argmax(fit) if self.maximize else np.argmin(fit))
        self.history["best_fitness"].append(float(fit[best_idx]))
        self.history["best_position"].append(to_cpu(population[best_idx].copy()))

    def run(self, verbose=True):
        population = self._initialize_population()
        fitness = xp.asarray([self._evaluate(population[i]) for i in range(self.population_size)])
        self.mu_ns = float(self._rng.uniform(0.1 * self.population_size, 0.3 * self.population_size))
        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with RNEGDE",
            ncols=100,
            disable=not verbose,
        )

        for generation in iterator:
            fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
            order = np.argsort(fit_cpu)
            if self.maximize:
                order = order[::-1]
            rank_of = np.empty(self.population_size, dtype=int)
            rank_of[order] = np.arange(self.population_size)

            f_values = np.asarray([self._sample_f() for _ in range(self.population_size)])
            cr_sorted = np.sort(np.asarray([self._sample_cr() for _ in range(self.population_size)]))
            ns_values = np.asarray([self._sample_ns() for _ in range(self.population_size)], dtype=int)

            new_population = population.copy()
            new_fitness = fitness.copy()
            successful_f, successful_cr, successful_ns = [], [], []

            for i in range(self.population_size):
                f_i = float(f_values[i])
                cr_i = float(cr_sorted[rank_of[i]])
                ns_i = int(ns_values[i])

                available = [j for j in range(self.population_size) if j != i]
                n_take = min(ns_i, len(available))
                neighbor_indices = list(self._rng.choice(available, size=n_take, replace=False))
                region = neighbor_indices + [i]
                if self.maximize:
                    guide_idx = max(region, key=lambda j: fit_cpu[j])
                else:
                    guide_idx = min(region, key=lambda j: fit_cpu[j])

                r1_idx = self._pick_population_index({i})
                r1 = population[r1_idx]

                r2, _r2_fit, source, r2_idx = self._pick_union_vector(
                    population,
                    excluded_population_indices={i, r1_idx},
                )
                if source == "population" and r2_idx == r1_idx:
                    # Defensive guard, although excluded above.
                    r2, _r2_fit, source, r2_idx = self._pick_union_vector(
                        population,
                        excluded_population_indices={i, r1_idx},
                    )

                target = population[i]
                guide = population[guide_idx]
                mutant = target + f_i * (guide - target) + f_i * (r1 - r2)
                mutant = xp.clip(mutant, self.min_params, self.max_params)
                trial = self._binomial_crossover(target, mutant, cr_i)

                trial_fit = self._evaluate(trial)
                target_fit = fit_cpu[i]
                if self._better(trial_fit, target_fit):
                    parent_entry = (target.copy(), float(target_fit))
                    if len(self.archive) < self.population_size:
                        self.archive.append(parent_entry)
                    else:
                        self.archive[int(self._rng.integers(0, len(self.archive)))] = parent_entry

                    new_population[i] = trial
                    new_fitness[i] = trial_fit
                    successful_f.append(f_i)
                    successful_cr.append(cr_i)
                    successful_ns.append(ns_i)

            population = new_population
            fitness = new_fitness

            if successful_f:
                lm = self._lehmer_mean(successful_f)
                self.mu_f = (1.0 - self.c) * self.mu_f + self.c * lm
            if successful_cr:
                self.mu_cr = (1.0 - self.c) * self.mu_cr + self.c * float(np.mean(successful_cr))
            if successful_ns:
                lm_ns = self._lehmer_mean(successful_ns)
                self.mu_ns = (1.0 - self.c) * self.mu_ns + self.c * lm_ns

            self._save_history(population, fitness)
            iterator.set_postfix({"Best": f"{self.history['best_fitness'][-1]:.6f}"})
            if self.progress_callback is not None:
                self.progress_callback(generation + 1)

        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        best_idx = int(np.argmax(fit_cpu) if self.maximize else np.argmin(fit_cpu))
        return to_cpu(population[best_idx]), float(fit_cpu[best_idx])


def run_rnegde(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=30,
    verbose=False,
    progress_callback=None,
    seed=None,
):
    model = RNEGDEOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        maximize=False,
        progress_callback=progress_callback,
        seed=seed,
    )
    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
