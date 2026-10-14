from core.xp import xp, to_cpu
import numpy as np
from tqdm import tqdm


class FVRADEOptimizer:
    """Function Value Ranking Aware Differential Evolution (FVRADE).

    Paper mechanism implemented here:
      - DE/current-to-rwrand/1 mutation
      - exponential rank-based roulette selection with sigma=0.1
      - JADE_Sort-style adaptive F/CR
      - binomial crossover
      - external archive with capacity NP

    The public wrapper follows the active benchmark optimizer contract.
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        population_size=100,
        max_iter=30,
        maximize=False,
        sigma=0.1,
        beta=0.1,
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
        self.sigma = float(sigma)
        self.beta = float(beta)
        self.progress_callback = progress_callback
        self.seed = seed

        self.mu_cr = 0.5
        self.mu_f = 0.5
        self.archive = []

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

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
        pop = self._rng.uniform(
            low=np.asarray(to_cpu(self.min_params), dtype=float),
            high=np.asarray(to_cpu(self.max_params), dtype=float),
            size=(self.population_size, self.dim),
        )
        return xp.asarray(pop)

    def _rank_probabilities(self, fitness):
        # Paper ranks from best (1) to worst (NP). Probabilities depend only on rank.
        fitness_cpu = np.asarray(to_cpu(fitness), dtype=float)
        order = np.argsort(fitness_cpu)
        if self.maximize:
            order = order[::-1]

        rank = np.empty(self.population_size, dtype=int)
        rank[order] = np.arange(1, self.population_size + 1)

        n = float(self.population_size)
        sigma = max(self.sigma, 1e-12)
        weights = (1.0 / (sigma * n * np.sqrt(2.0 * np.pi))) * np.exp(
            -((rank - 1.0) ** 2) / (2.0 * sigma * sigma * n * n)
        )
        weights_sum = weights.sum()
        if not np.isfinite(weights_sum) or weights_sum <= 0:
            return np.full(self.population_size, 1.0 / self.population_size)
        return weights / weights_sum

    def _sample_f(self):
        # JADE: Cauchy(mu_F, 0.1), regenerate until positive, truncate at 1.
        for _ in range(1000):
            f = self.mu_f + 0.1 * self._rng.standard_cauchy()
            if f > 0:
                return float(min(f, 1.0))
        return float(np.clip(self.mu_f, 1e-8, 1.0))

    def _sample_cr_set(self):
        cr = self._rng.normal(self.mu_cr, 0.1, size=self.population_size)
        return np.clip(cr, 0.0, 1.0)

    def _pick_distinct_index(self, excluded):
        candidates = [i for i in range(self.population_size) if i not in excluded]
        if not candidates:
            raise RuntimeError("Population too small for FVRADE mutation")
        return int(self._rng.choice(candidates))

    def _sample_union_vector(self, population, excluded_population_indices):
        candidates = []
        candidate_meta = []

        for i in range(self.population_size):
            if i not in excluded_population_indices:
                candidates.append(population[i])
                candidate_meta.append(("population", i))

        for i, vec in enumerate(self.archive):
            candidates.append(vec)
            candidate_meta.append(("archive", i))

        if not candidates:
            raise RuntimeError("No candidate available from P union A")

        idx = int(self._rng.integers(0, len(candidates)))
        return candidates[idx], candidate_meta[idx]

    def _binomial_crossover(self, target, mutant, cr):
        j_rand = int(self._rng.integers(0, self.dim))
        mask = self._rng.random(self.dim) <= cr
        mask[j_rand] = True
        trial = xp.where(xp.asarray(mask), mutant, target)
        return xp.clip(trial, self.min_params, self.max_params)

    def _update_memory(self, successful_f, successful_cr):
        if successful_cr:
            self.mu_cr = (1.0 - self.beta) * self.mu_cr + self.beta * float(np.mean(successful_cr))
        if successful_f:
            sf = np.asarray(successful_f, dtype=float)
            denom = sf.sum()
            if denom > 0:
                lehmer = float(np.sum(sf * sf) / denom)
                self.mu_f = (1.0 - self.beta) * self.mu_f + self.beta * lehmer

    def _save_history(self, population, fitness):
        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        best_idx = int(np.argmax(fit_cpu) if self.maximize else np.argmin(fit_cpu))
        self.history["best_fitness"].append(float(fit_cpu[best_idx]))
        self.history["best_position"].append(to_cpu(population[best_idx].copy()))

    def run(self, verbose=True):
        population = self._initialize_population()
        fitness = xp.asarray([self._evaluate(population[i]) for i in range(self.population_size)])
        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with FVRADE",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
            rank_order = np.argsort(fit_cpu)
            if self.maximize:
                rank_order = rank_order[::-1]
            rank_of = np.empty(self.population_size, dtype=int)
            rank_of[rank_order] = np.arange(self.population_size)

            probabilities = self._rank_probabilities(fitness)
            cr_sorted = np.sort(self._sample_cr_set())
            f_values = np.asarray([self._sample_f() for _ in range(self.population_size)])

            new_population = population.copy()
            new_fitness = fitness.copy()
            successful_f = []
            successful_cr = []

            for i in range(self.population_size):
                cr_i = float(cr_sorted[rank_of[i]])
                f_i = float(f_values[i])

                guide_idx = int(self._rng.choice(self.population_size, p=probabilities))
                # Paper excludes target/guiding individual for the other random parents.
                if guide_idx == i and self.population_size > 1:
                    allowed = [j for j in range(self.population_size) if j != i]
                    guide_idx = int(self._rng.choice(allowed, p=np.asarray(probabilities)[allowed] / np.asarray(probabilities)[allowed].sum()))

                r1_idx = self._pick_distinct_index({i, guide_idx})
                r1 = population[r1_idx]
                r1_fit = float(to_cpu(fitness[r1_idx]))

                r2, r2_meta = self._sample_union_vector(
                    population,
                    excluded_population_indices={i, guide_idx, r1_idx},
                )
                if r2_meta[0] == "population":
                    r2_fit = float(to_cpu(fitness[r2_meta[1]]))
                else:
                    # Archive stores (position, fitness) tuples internally.
                    # Backward compatibility for any vector-only archive entry.
                    entry = self.archive[r2_meta[1]]
                    if isinstance(entry, tuple):
                        r2, r2_fit = entry
                    else:
                        r2_fit = self._evaluate(r2)

                # If archive entries are tuples, unwrap any vector selected above.
                if isinstance(r2, tuple):
                    r2, r2_fit = r2

                # Better random vector is r1, worse one is r2 (paper's directional difference).
                if self._better(r2_fit, r1_fit):
                    r1, r2 = r2, r1
                    r1_fit, r2_fit = r2_fit, r1_fit

                target = population[i]
                guide = population[guide_idx]
                mutant = target + f_i * (guide - target) + f_i * (r1 - r2)
                mutant = xp.clip(mutant, self.min_params, self.max_params)
                trial = self._binomial_crossover(target, mutant, cr_i)
                trial_fit = self._evaluate(trial)
                target_fit = float(to_cpu(fitness[i]))

                if self._better(trial_fit, target_fit):
                    # Paper archive keeps obsolete parents, capacity NP.
                    parent_entry = (target.copy(), target_fit)
                    if len(self.archive) < self.population_size:
                        self.archive.append(parent_entry)
                    else:
                        aidx = int(self._rng.integers(0, len(self.archive)))
                        old_entry = self.archive[aidx]
                        old_fit = old_entry[1] if isinstance(old_entry, tuple) else self._evaluate(old_entry)
                        # Algorithm 2 retains the better between the random archive member and parent.
                        if self._better(target_fit, old_fit):
                            self.archive[aidx] = parent_entry

                    new_population[i] = trial
                    new_fitness[i] = trial_fit
                    successful_f.append(f_i)
                    successful_cr.append(cr_i)

            population = new_population
            fitness = new_fitness
            self._update_memory(successful_f, successful_cr)
            self._save_history(population, fitness)

            best = self.history["best_fitness"][-1]
            iterator.set_postfix({"Best": f"{best:.6f}"})
            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        best_idx = int(np.argmax(fit_cpu) if self.maximize else np.argmin(fit_cpu))
        return to_cpu(population[best_idx]), float(fit_cpu[best_idx])


def run_fvrade(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=30,
    verbose=False,
    progress_callback=None,
    seed=None,
):
    model = FVRADEOptimizer(
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
