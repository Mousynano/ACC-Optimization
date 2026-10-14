from core.xp import xp, to_cpu
import numpy as np
from tqdm import tqdm


class NeighborhoodSHADEOptimizer:
    """Neighborhood-based SHADE (N-SHADE).

    Implements the paper's fixed-population SHADE variant with:
      - DE/current-to-pbest/1 mutation
      - external archive
      - binomial crossover
      - spatial neighborhoods of round(sqrt(NP)) using Manhattan distance
      - F/CR adaptation from successful neighbors, weighted by fitness improvement

    The optimizer follows the active benchmark optimizer contract.
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        population_size=100,
        max_iter=30,
        maximize=False,
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
        self.progress_callback = progress_callback
        self.seed = seed

        self.archive = []
        self.neighborhood_size = max(1, int(round(np.sqrt(self.population_size))))

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

    @staticmethod
    def _weighted_arithmetic(values, improvements):
        values = np.asarray(values, dtype=float)
        improvements = np.asarray(improvements, dtype=float)
        if values.size == 0:
            return None
        total = improvements.sum()
        weights = np.full(values.size, 1.0 / values.size) if total <= 0 else improvements / total
        return float(np.sum(weights * values))

    @staticmethod
    def _weighted_lehmer(values, improvements):
        values = np.asarray(values, dtype=float)
        improvements = np.asarray(improvements, dtype=float)
        if values.size == 0:
            return None
        total = improvements.sum()
        weights = np.full(values.size, 1.0 / values.size) if total <= 0 else improvements / total
        denom = np.sum(weights * values)
        if denom <= 0:
            return None
        return float(np.sum(weights * values * values) / denom)

    def _neighbor_parameter(self, i, population_cpu, success_mask, success_f, success_cr, success_delta):
        # Paper uses L1/Manhattan spatial neighborhoods of round(sqrt(NP)).
        successful_indices = np.flatnonzero(success_mask)
        if successful_indices.size == 0:
            return float(self._rng.uniform(0.5, 1.0)), float(self._rng.uniform(0.1, 1.0))

        distances = np.sum(
            np.abs(population_cpu[successful_indices] - population_cpu[i]),
            axis=1,
        )
        take = min(self.neighborhood_size, successful_indices.size)
        nearest_local = np.argsort(distances)[:take]
        neighbors = successful_indices[nearest_local]

        f = self._weighted_lehmer(success_f[neighbors], success_delta[neighbors])
        cr = self._weighted_arithmetic(success_cr[neighbors], success_delta[neighbors])

        if f is None or not np.isfinite(f):
            f = float(self._rng.uniform(0.5, 1.0))
        if cr is None or not np.isfinite(cr):
            cr = float(self._rng.uniform(0.1, 1.0))

        return float(np.clip(f, 1e-12, 1.0)), float(np.clip(cr, 0.0, 1.0))

    def _binomial_crossover(self, target, mutant, cr):
        j_rand = int(self._rng.integers(0, self.dim))
        mask = self._rng.random(self.dim) <= cr
        mask[j_rand] = True
        trial = xp.where(xp.asarray(mask), mutant, target)
        return xp.clip(trial, self.min_params, self.max_params)

    def _pick_r1(self, excluded):
        candidates = [i for i in range(self.population_size) if i not in excluded]
        if not candidates:
            raise RuntimeError("Population too small for N-SHADE mutation")
        return int(self._rng.choice(candidates))

    def _pick_r2(self, population, excluded_population_indices):
        candidates = []
        for i in range(self.population_size):
            if i not in excluded_population_indices:
                candidates.append(population[i])
        candidates.extend(self.archive)
        if not candidates:
            raise RuntimeError("No candidate available from P union A")
        return candidates[int(self._rng.integers(0, len(candidates)))]

    def _save_history(self, population, fitness):
        fit = np.asarray(to_cpu(fitness), dtype=float)
        best_idx = int(np.argmax(fit) if self.maximize else np.argmin(fit))
        self.history["best_fitness"].append(float(fit[best_idx]))
        self.history["best_position"].append(to_cpu(population[best_idx].copy()))

    def run(self, verbose=True):
        population = self._initialize_population()
        fitness = xp.asarray([self._evaluate(population[i]) for i in range(self.population_size)])

        # Per-individual successful parameter records used to create the next
        # generation's local spatial adaptation neighborhoods.
        success_mask = np.zeros(self.population_size, dtype=bool)
        success_f = np.full(self.population_size, np.nan)
        success_cr = np.full(self.population_size, np.nan)
        success_delta = np.zeros(self.population_size, dtype=float)

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with N-SHADE",
            ncols=100,
            disable=not verbose,
        )

        for generation in iterator:
            pop_cpu = np.asarray(to_cpu(population), dtype=float)
            fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
            order = np.argsort(fit_cpu)
            if self.maximize:
                order = order[::-1]

            new_population = population.copy()
            new_fitness = fitness.copy()
            next_success_mask = np.zeros(self.population_size, dtype=bool)
            next_success_f = np.full(self.population_size, np.nan)
            next_success_cr = np.full(self.population_size, np.nan)
            next_success_delta = np.zeros(self.population_size, dtype=float)

            for i in range(self.population_size):
                f_i, cr_i = self._neighbor_parameter(
                    i, pop_cpu, success_mask, success_f, success_cr, success_delta
                )

                p_min = min(1.0, 2.0 / self.population_size)
                p = float(self._rng.uniform(p_min, 0.2)) if p_min < 0.2 else p_min
                top_count = max(2, int(np.ceil(p * self.population_size)))
                top_count = min(top_count, self.population_size)
                pbest_idx = int(self._rng.choice(order[:top_count]))

                r1_idx = self._pick_r1({i, pbest_idx})
                r2 = self._pick_r2(population, {i, pbest_idx, r1_idx})

                target = population[i]
                mutant = target + f_i * (population[pbest_idx] - target) + f_i * (population[r1_idx] - r2)
                mutant = xp.clip(mutant, self.min_params, self.max_params)
                trial = self._binomial_crossover(target, mutant, cr_i)
                trial_fit = self._evaluate(trial)
                target_fit = fit_cpu[i]

                # Paper uses <= for minimization selection; equality is accepted.
                accepted = (trial_fit >= target_fit) if self.maximize else (trial_fit <= target_fit)
                if accepted:
                    self.archive.append(target.copy())
                    if len(self.archive) > self.population_size:
                        del self.archive[int(self._rng.integers(0, len(self.archive)))]

                    new_population[i] = trial
                    new_fitness[i] = trial_fit
                    next_success_mask[i] = True
                    next_success_f[i] = f_i
                    next_success_cr[i] = cr_i
                    next_success_delta[i] = abs(target_fit - trial_fit)

            population = new_population
            fitness = new_fitness
            success_mask = next_success_mask
            success_f = next_success_f
            success_cr = next_success_cr
            success_delta = next_success_delta

            self._save_history(population, fitness)
            iterator.set_postfix({"Best": f"{self.history['best_fitness'][-1]:.6f}"})
            if self.progress_callback is not None:
                self.progress_callback(generation + 1)

        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        best_idx = int(np.argmax(fit_cpu) if self.maximize else np.argmin(fit_cpu))
        return to_cpu(population[best_idx]), float(fit_cpu[best_idx])


def run_nshade(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=30,
    verbose=False,
    progress_callback=None,
    seed=None,
):
    model = NeighborhoodSHADEOptimizer(
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
