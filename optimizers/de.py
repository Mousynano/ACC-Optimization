from core.xp import xp, to_cpu
import numpy as np
from tqdm import tqdm


class DifferentialEvolutionOptimizer:
    """Classic Differential Evolution using DE/rand/1/bin.

    Pipeline contract used by the active benchmark:
      - decision vectors use the shared NumPy backend
      - fitness evaluation is performed on CPU via to_cpu(...)
      - bound handling uses clipping
      - minimization/maximization is supported internally
      - history stores best fitness and best position once per generation
      - progress_callback receives the completed generation count

    Mutation:
        v_i = x_r1 + F * (x_r2 - x_r3)

    Binomial crossover:
        u_ij = v_ij  if rand_j <= CR or j == j_rand
               x_ij  otherwise

    Greedy selection:
        u_i replaces x_i if it is at least as good as x_i.
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        population_size=100,
        max_iter=30,
        mutation_factor=0.8,
        crossover_rate=0.9,
        maximize=False,
        progress_callback=None,
        seed=None,
        initial_population=None,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)

        self.population_size = int(population_size)
        self.max_iter = int(max_iter)
        self.dim = len(min_params)

        self.mutation_factor = float(mutation_factor)
        self.crossover_rate = float(crossover_rate)
        self.maximize = maximize
        self.progress_callback = progress_callback
        self.seed = seed
        self.initial_population = None
        if initial_population is not None:
            arr = np.asarray(initial_population, dtype=float)
            if arr.shape != (self.population_size, self.dim):
                raise ValueError("initial_population must have shape (population_size, dim)")
            if not np.isfinite(arr).all() or np.any(arr < np.asarray(min_params)) or np.any(arr > np.asarray(max_params)):
                raise ValueError("initial_population must be finite and inside the bounds")
            self.initial_population = arr.copy()

        if self.population_size < 4:
            raise ValueError(
                "DE/rand/1/bin requires population_size >= 4 "
                "(one target plus three distinct donor indices)."
            )

        if self.dim < 1:
            raise ValueError("At least one decision variable is required.")

        if not (0.0 <= self.crossover_rate <= 1.0):
            raise ValueError("crossover_rate (CR) must be in [0, 1].")

        if self.mutation_factor <= 0.0:
            raise ValueError("mutation_factor (F) must be > 0.")

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

        # Use a local NumPy RNG for deterministic index/mask generation.
        self._rng = np.random.default_rng(seed)

        # Keep the shared backend RNG aligned as well.
        if seed is not None:
            try:
                xp.random.seed(seed)
            except Exception:
                pass

    def _evaluate(self, position):
        """Evaluate one candidate on CPU, matching the ACC fitness pipeline."""
        return float(self.fitness_function(to_cpu(position)))

    def _initialize_population(self):
        """Uniform initialization inside parameter bounds."""
        if self.initial_population is not None:
            return xp.asarray(self.initial_population.copy())
        low = np.asarray(to_cpu(self.min_params), dtype=float)
        high = np.asarray(to_cpu(self.max_params), dtype=float)

        population = self._rng.uniform(
            low=low,
            high=high,
            size=(self.population_size, self.dim),
        )
        return xp.asarray(population)

    def _pick_donors(self, target_idx):
        """Pick r1, r2, r3, all distinct and different from target_idx."""
        candidates = np.delete(np.arange(self.population_size), target_idx)
        r1, r2, r3 = self._rng.choice(candidates, size=3, replace=False)
        return int(r1), int(r2), int(r3)

    def _mutate(self, population, target_idx):
        """Classic DE/rand/1 mutation."""
        r1, r2, r3 = self._pick_donors(target_idx)

        mutant = (
            population[r1]
            + self.mutation_factor * (population[r2] - population[r3])
        )

        # Same bound-control convention as the existing optimizers.
        return xp.clip(mutant, self.min_params, self.max_params)

    def _binomial_crossover(self, target, mutant):
        """DE binomial crossover with mandatory j_rand."""
        j_rand = int(self._rng.integers(0, self.dim))
        mask = self._rng.random(self.dim) <= self.crossover_rate

        # Guarantee at least one component comes from the mutant vector.
        mask[j_rand] = True

        trial = xp.where(xp.asarray(mask), mutant, target)
        return xp.clip(trial, self.min_params, self.max_params)

    def _accepted(self, trial_fitness, target_fitness):
        """Greedy one-to-one DE selection; equality is accepted."""
        if self.maximize:
            return trial_fitness >= target_fitness
        return trial_fitness <= target_fitness

    def _best_index(self, fitness):
        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        if self.maximize:
            return int(np.argmax(fit_cpu))
        return int(np.argmin(fit_cpu))

    def _save_history(self, population, fitness):
        best_idx = self._best_index(fitness)
        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)

        self.history["best_fitness"].append(float(fit_cpu[best_idx]))
        self.history["best_position"].append(
            to_cpu(population[best_idx].copy())
        )

    def run(self, verbose=True):
        # ---------------------------------------------------------------
        # 1) Initialize population and evaluate initial fitness.
        # ---------------------------------------------------------------
        population = self._initialize_population()
        fitness = xp.asarray([
            self._evaluate(population[i])
            for i in range(self.population_size)
        ])

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with DE",
            ncols=100,
            disable=not verbose,
        )

        for generation in iterator:
            # Use generation-synchronous replacement:
            # all donor vectors come from the current population, not from
            # partially updated individuals within the same generation.
            new_population = population.copy()
            new_fitness = fitness.copy()

            fit_cpu = np.asarray(to_cpu(fitness), dtype=float)

            # -----------------------------------------------------------
            # 2) Mutation -> crossover -> evaluation -> greedy selection
            #    for every target vector.
            # -----------------------------------------------------------
            for i in range(self.population_size):
                target = population[i]

                mutant = self._mutate(population, i)
                trial = self._binomial_crossover(target, mutant)
                trial_fitness = self._evaluate(trial)

                if self._accepted(trial_fitness, fit_cpu[i]):
                    new_population[i] = trial
                    new_fitness[i] = trial_fitness

            population = new_population
            fitness = new_fitness

            # -----------------------------------------------------------
            # 3) Track best solution once per completed generation.
            # -----------------------------------------------------------
            self._save_history(population, fitness)

            iterator.set_postfix({
                "Best": f"{self.history['best_fitness'][-1]:.6f}"
            })

            if self.progress_callback is not None:
                self.progress_callback(generation + 1)

        # ---------------------------------------------------------------
        # 4) Return the final best candidate using the shared contract.
        # ---------------------------------------------------------------
        best_idx = self._best_index(fitness)
        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)

        return (
            to_cpu(population[best_idx]),
            float(fit_cpu[best_idx]),
        )


def run_de(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=30,
    verbose=False,
    progress_callback=None,
    seed=None,
    mutation_factor=0.8,
    crossover_rate=0.9,
    initial_population=None,
):
    """Run DE using the shared optimizer-runner contract."""

    model = DifferentialEvolutionOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        mutation_factor=mutation_factor,
        crossover_rate=crossover_rate,
        maximize=False,  # ACC objective is an error metric -> minimization
        progress_callback=progress_callback,
        seed=seed,
        initial_population=initial_population,
    )

    best_params, best_fitness = model.run(verbose=verbose)

    return (
        best_params,
        best_fitness,
        model.history["best_fitness"],
    )
