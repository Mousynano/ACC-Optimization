from core.xp import xp, to_cpu
import numpy as np
from tqdm import tqdm


class DENPSOptimizer:
    """Diversity enhancement-based Differential Evolution with NPS (DE-NPS).

    Core paper mechanisms implemented:
      1. Bi-stage F adaptation (tripuls in early stage, Cauchy later) and
         SHADE-style historical memories for F/CR.
      2. Logarithmic-spiral perturbation embedded in binomial crossover.
      3. Covariance/PCA-guided intervention for stagnant individuals.

    Compatibility note
    ------------------
    The paper additionally uses linear population-size reduction. The project's
    benchmark contract fixes ``population_size`` across optimizers, so the wrapper
    defaults to ``population_reduction=False``. Set it to True when reproducing the
    paper-style population reduction rather than the fixed-population ACC benchmark.

    The paper invokes MATLAB ``tripuls`` without redefining its mathematical form.
    ``_tripuls`` below uses the default MATLAB triangular-pulse convention
    (unit height, width 1, centered at zero).
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        population_size=100,
        max_iter=30,
        maximize=False,
        memory_size=4,
        p_best_rate=0.2,
        tau=0.1,
        spiral_b=1.0,
        population_reduction=False,
        min_population_size=4,
        progress_callback=None,
        seed=None,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)
        self.initial_population_size = int(population_size)
        self.population_size = int(population_size)
        self.max_iter = int(max_iter)
        self.dim = len(min_params)
        self.maximize = maximize
        self.memory_size = int(memory_size)
        self.p_best_rate = float(p_best_rate)
        self.tau = float(tau)
        self.spiral_b = float(spiral_b)
        self.population_reduction = bool(population_reduction)
        self.min_population_size = int(min_population_size)
        self.progress_callback = progress_callback
        self.seed = seed

        self.m_f = np.full(self.memory_size, 0.5, dtype=float)
        self.m_cr = np.full(self.memory_size, 0.5, dtype=float)
        self.memory_index = 0
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
        pop = self._rng.uniform(low=low, high=high, size=(self.initial_population_size, self.dim))
        return xp.asarray(pop)

    @staticmethod
    def _tripuls(t):
        # MATLAB tripuls(t) default: unit-height triangular pulse of width 1.
        t = np.asarray(t, dtype=float)
        return np.maximum(1.0 - 2.0 * np.abs(t), 0.0)

    def _sample_f(self, mu_f, progress):
        if progress < 0.2:
            r = float(self._rng.random())
            f = mu_f + 0.1 * r * float(self._tripuls(r * mu_f))
            return float(np.clip(f, 1e-12, 1.0))

        # Later stage: Cauchy(mu_F, 0.1), regenerate if non-positive, truncate >1.
        for _ in range(1000):
            f = mu_f + 0.1 * self._rng.standard_cauchy()
            if f > 0:
                return float(min(f, 1.0))
        return float(np.clip(mu_f, 1e-12, 1.0))

    def _sample_cr(self, mu_cr):
        if not np.isfinite(mu_cr):
            return 0.0
        return float(np.clip(self._rng.normal(mu_cr, 0.1), 0.0, 1.0))

    def _pick_population_index(self, n, excluded):
        choices = [i for i in range(n) if i not in excluded]
        if not choices:
            raise RuntimeError("Population too small for DE-NPS mutation")
        return int(self._rng.choice(choices))

    def _pick_union_vector(self, population, excluded_population_indices):
        n = len(population)
        candidates = []
        for i in range(n):
            if i not in excluded_population_indices:
                candidates.append(population[i])
        candidates.extend(self.archive)
        if not candidates:
            raise RuntimeError("No P union A candidate available")
        return candidates[int(self._rng.integers(0, len(candidates)))]

    def _log_spiral_crossover(self, target, mutant, cr, progress):
        j_rand = int(self._rng.integers(0, self.dim))
        trial = target.copy()

        # Paper: a linearly increases from -1 to 0 as nfes/nfesmax grows.
        a = -1.0 + float(progress)
        # Algorithm 1: l = (a - 1) * rand + 1.
        l = (a - 1.0) * float(self._rng.random()) + 1.0

        for j in range(self.dim):
            if j == j_rand or self._rng.random() < cr:
                trial[j] = mutant[j]
            else:
                if self._rng.random() > self.tau:
                    trial[j] = target[j]
                else:
                    distance = target[j] - mutant[j]
                    spiral = distance * np.exp(l * self.spiral_b) * np.cos(l * 2.0 * np.pi)
                    trial[j] = spiral + target[j]

        return xp.clip(trial, self.min_params, self.max_params)

    @staticmethod
    def _weighted_lehmer(values, improvements):
        values = np.asarray(values, dtype=float)
        improvements = np.asarray(improvements, dtype=float)
        if values.size == 0:
            return None
        s = improvements.sum()
        weights = np.full(values.size, 1.0 / values.size) if s <= 0 else improvements / s
        denom = np.sum(weights * values)
        if denom <= 0:
            return None
        return float(np.sum(weights * values * values) / denom)

    def _update_memory(self, successful_f, successful_cr, improvements):
        if not successful_f:
            return
        mf = self._weighted_lehmer(successful_f, improvements)
        mcr = self._weighted_lehmer(successful_cr, improvements)
        if mf is not None:
            self.m_f[self.memory_index] = mf
        if mcr is not None:
            self.m_cr[self.memory_index] = mcr
        self.memory_index = (self.memory_index + 1) % self.memory_size

    def _pca_population(self, population, fitness):
        pop_cpu = np.asarray(to_cpu(population), dtype=float)
        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        order = np.argsort(fit_cpu)
        if self.maximize:
            order = order[::-1]
        top_count = max(2, int(np.ceil(0.2 * len(pop_cpu))))
        elite = pop_cpu[order[:top_count]]

        centered = elite - np.mean(elite, axis=0, keepdims=True)
        if elite.shape[0] <= 1 or np.allclose(centered, 0.0):
            return elite.copy(), pop_cpu[order[0]].copy()

        cov = np.cov(centered, rowvar=False)
        cov = np.atleast_2d(cov)
        try:
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
            transformed = centered @ eigvecs + np.mean(elite, axis=0, keepdims=True)
        except np.linalg.LinAlgError:
            transformed = elite.copy()
        return transformed, pop_cpu[order[0]].copy()

    def _stagnation_candidate(self, population, fitness, index, severe):
        pca_pop, best = self._pca_population(population, fitness)
        current = np.asarray(to_cpu(population[index]), dtype=float)
        if len(pca_pop) < 2:
            return xp.asarray(current)

        i1, i2 = self._rng.choice(len(pca_pop), size=2, replace=False)
        r = float(self._rng.random())

        if severe:
            # Eq. (11): random-search equation around PCA representation.
            base = pca_pop[index % len(pca_pop)]
            candidate = base - r * (pca_pop[i1] - best)
        else:
            # Eq. (12): best-based search equation.
            candidate = best + r * (pca_pop[i1] - pca_pop[i2])

        low = np.asarray(to_cpu(self.min_params), dtype=float)
        high = np.asarray(to_cpu(self.max_params), dtype=float)
        return xp.asarray(np.clip(candidate, low, high))

    def _reduce_population(self, population, fitness, stagnation, generation):
        if not self.population_reduction:
            return population, fitness, stagnation

        progress = (generation + 1) / max(1, self.max_iter)
        target_size = int(round(
            self.initial_population_size
            + progress * (self.min_population_size - self.initial_population_size)
        ))
        target_size = max(self.min_population_size, min(target_size, len(population)))
        if target_size >= len(population):
            return population, fitness, stagnation

        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        order = np.argsort(fit_cpu)
        if self.maximize:
            order = order[::-1]
        keep = order[:target_size]
        population = population[keep]
        fitness = fitness[keep]
        stagnation = stagnation[keep]
        self.population_size = target_size

        # Paper archive capacity follows current population size.
        if len(self.archive) > target_size:
            chosen = self._rng.choice(len(self.archive), size=target_size, replace=False)
            self.archive = [self.archive[int(i)] for i in chosen]
        return population, fitness, stagnation

    def _save_history(self, population, fitness):
        fit = np.asarray(to_cpu(fitness), dtype=float)
        best_idx = int(np.argmax(fit) if self.maximize else np.argmin(fit))
        self.history["best_fitness"].append(float(fit[best_idx]))
        self.history["best_position"].append(to_cpu(population[best_idx].copy()))

    def run(self, verbose=True):
        population = self._initialize_population()
        fitness = xp.asarray([self._evaluate(population[i]) for i in range(len(population))])
        stagnation = np.zeros(len(population), dtype=int)
        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with DE-NPS",
            ncols=100,
            disable=not verbose,
        )

        for generation in iterator:
            n = len(population)
            fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
            order = np.argsort(fit_cpu)
            if self.maximize:
                order = order[::-1]
            progress = generation / max(1, self.max_iter - 1)

            new_population = population.copy()
            new_fitness = fitness.copy()
            successful_f, successful_cr, improvements = [], [], []

            for i in range(n):
                mem_idx = int(self._rng.integers(0, self.memory_size))
                f_i = self._sample_f(self.m_f[mem_idx], progress)
                cr_i = self._sample_cr(self.m_cr[mem_idx])

                p_count = max(2, int(np.ceil(self.p_best_rate * n)))
                p_count = min(p_count, n)
                pbest_idx = int(self._rng.choice(order[:p_count]))
                r1_idx = self._pick_population_index(n, {i, pbest_idx})
                r2 = self._pick_union_vector(population, {i, pbest_idx, r1_idx})

                target = population[i]
                mutant = target + f_i * (population[pbest_idx] - target) + f_i * (population[r1_idx] - r2)
                mutant = xp.clip(mutant, self.min_params, self.max_params)
                trial = self._log_spiral_crossover(target, mutant, cr_i, progress)
                trial_fit = self._evaluate(trial)
                target_fit = fit_cpu[i]

                accepted = self._better(trial_fit, target_fit)
                if accepted:
                    self.archive.append(target.copy())
                    if len(self.archive) > n:
                        del self.archive[int(self._rng.integers(0, len(self.archive)))]
                    new_population[i] = trial
                    new_fitness[i] = trial_fit
                    successful_f.append(f_i)
                    successful_cr.append(cr_i)
                    improvements.append(abs(target_fit - trial_fit))
                    stagnation[i] = 0
                else:
                    stagnation[i] += 1

            population = new_population
            fitness = new_fitness
            self._update_memory(successful_f, successful_cr, improvements)

            # Conservative implementation of Sec. 3.3: intervene when the paper's
            # stagnation threshold (count > 2D) is crossed, rather than spending an
            # extra objective evaluation on every individual every generation.
            fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
            for i in range(len(population)):
                if stagnation[i] > 2 * self.dim:
                    candidate = self._stagnation_candidate(
                        population,
                        fitness,
                        i,
                        severe=True,
                    )
                    candidate_fit = self._evaluate(candidate)
                    current_fit = float(to_cpu(fitness[i]))
                    if self._better(candidate_fit, current_fit):
                        population[i] = candidate
                        fitness[i] = candidate_fit
                    stagnation[i] = 0

            population, fitness, stagnation = self._reduce_population(
                population, fitness, stagnation, generation
            )

            self._save_history(population, fitness)
            iterator.set_postfix({"Best": f"{self.history['best_fitness'][-1]:.6f}"})
            if self.progress_callback is not None:
                self.progress_callback(generation + 1)

        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        best_idx = int(np.argmax(fit_cpu) if self.maximize else np.argmin(fit_cpu))
        return to_cpu(population[best_idx]), float(fit_cpu[best_idx])


def run_de_nps(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=30,
    verbose=False,
    progress_callback=None,
    seed=None,
):
    # Fixed population is intentional for compatibility with the ACC benchmark's
    # same-population/same-iteration comparison contract. To reproduce the paper's
    # linear population reduction, instantiate DENPSOptimizer directly with
    # population_reduction=True.
    model = DENPSOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        maximize=False,
        population_reduction=False,
        progress_callback=progress_callback,
        seed=seed,
    )
    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
