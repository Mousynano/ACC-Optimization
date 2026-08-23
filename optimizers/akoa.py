try:
    from core.xp import xp, to_cpu, GPU_AVAILABLE
except Exception:  # fallback for standalone testing outside the project
    import numpy as xp
    GPU_AVAILABLE = False

    def to_cpu(x):
        return xp.asarray(x)

import numpy as np
from tqdm import tqdm


class ArchivedKeplerOptimizationAlgorithm:
    """
    Kepler Optimization Algorithm (KOA) draft implementation.

    Style follows the existing optimizer wrappers in this project:
    - class based optimizer
    - xp backend for CPU/GPU arrays
    - CPU-safe objective evaluation
    - history["best_fitness"] and history["best_position"]
    - wrapper function run_koa(...)

    Notes:
    - Default mode is minimization because ACC objective functions are errors.
    - max_fes is optional. If provided, it limits total objective calls.
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        planets=100,
        max_iter=300,
        max_fes=None,
        maximize=False,
        mu0=0.1,
        gamma=15.0,
        cycles=3,
        eps=1e-12,
        progress_callback=None,
        archive_size=4,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)
        self.planets = int(planets)
        self.max_iter = int(max_iter)
        self.max_fes = None if max_fes is None else int(max_fes)
        self.maximize = maximize
        self.dim = len(min_params)

        self.mu0 = float(mu0)
        self.gamma = float(gamma)
        self.cycles = int(cycles)
        self.eps = float(eps)

        self.history = {"best_fitness": [], "best_position": []}
        self.progress_callback = progress_callback
        self.fe_count = 0

        # External archive: keeps the best planets found so far.
        self.archive_size = max(1, int(archive_size))
        self.archive_positions = None
        self.archive_fitness = None

        self.population = None
        self.fitness = None
        self.eccentricity = None
        self.period = None
        self.best_position = xp.zeros(self.dim)
        self.best_value = -float("inf") if maximize else float("inf")

    def initialize_positions(self):
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.planets, self.dim),
        )

    def evaluate(self, pos_xp):
        pos_cpu = to_cpu(pos_xp)
        value = float(self.fitness_function(pos_cpu))
        self.fe_count += 1
        return value

    def evaluate_population(self, population):
        values = []
        for i in range(population.shape[0]):
            values.append(self.evaluate(population[i]))
        return xp.array(values, dtype=float)

    def is_better(self, value_a, value_b):
        return value_a > value_b if self.maximize else value_a < value_b

    def _update_global_best(self):
        if self.maximize:
            idx = int(xp.argmax(self.fitness))
        else:
            idx = int(xp.argmin(self.fitness))

        value = float(self.fitness[idx])
        if self.is_better(value, self.best_value):
            self.best_value = value
            self.best_position = self.population[idx].copy()

    def _mass_values(self):
        fit = self.fitness

        if self.maximize:
            best = float(xp.max(fit))
            worst = float(xp.min(fit))
            quality = fit - worst + self.eps
        else:
            best = float(xp.min(fit))
            worst = float(xp.max(fit))
            quality = worst - fit + self.eps

        denom = float(xp.sum(quality)) + self.eps
        masses = quality / denom

        if self.maximize:
            best_idx = int(xp.argmax(fit))
        else:
            best_idx = int(xp.argmin(fit))

        sun_mass = masses[best_idx] * xp.random.uniform(0.0, 1.0)
        return masses, sun_mass

    def _normalized_distance(self, distances):
        d_min = float(xp.min(distances))
        d_max = float(xp.max(distances))
        if abs(d_max - d_min) <= self.eps:
            return xp.zeros_like(distances)
        return (distances - d_min) / (d_max - d_min + self.eps)

    def _cyclic_a2(self, iteration):
        cycle_len = max(1.0, self.max_iter / max(self.cycles, 1))
        return -1.0 - ((iteration % cycle_len) / cycle_len)

    def _make_velocity(self, i, iteration, masses, sun_mass, distances, r_norm, mu_t):
        Xi = self.population[i]
        sun = self.best_position
        lb = self.min_params
        ub = self.max_params
        span = ub - lb

        # Random solutions Xa and Xb from current population.
        idx_a = int(xp.random.randint(0, self.planets))
        idx_b = int(xp.random.randint(0, self.planets))
        Xa = self.population[idx_a]
        Xb = self.population[idx_b]

        r3 = float(xp.random.uniform(0.0, 1.0))
        r4 = float(xp.random.uniform(0.0, 1.0))
        r5 = xp.random.uniform(0.0, 1.0, size=self.dim)
        r6 = xp.random.uniform(0.0, 1.0, size=self.dim)

        U = xp.where(r5 <= r6, 0.0, 1.0)
        U1 = xp.where(r5 <= r4, 0.0, 1.0)
        U2 = 0.0 if r3 <= r4 else 1.0
        F = 1.0 if r4 <= 0.5 else -1.0

        Ri = float(distances[i]) + self.eps
        mi = float(masses[i]) + self.eps
        Ti = float(self.period[i]) + self.eps

        # Kepler third-law style semi-major axis.
        a_i = r3 * ((Ti * Ti * mu_t * (float(sun_mass) + mi) / (4.0 * np.pi * np.pi + self.eps)) ** (1.0 / 3.0))
        a_i = abs(float(a_i)) + self.eps

        inside = abs(mu_t * (float(sun_mass) + mi) * (2.0 / Ri - 1.0 / a_i))
        L = float(np.sqrt(max(inside, 0.0)))

        M_scalar = r3 * (1.0 - r4) + r4
        ell = U * M_scalar * L
        M_vec = r3 * (1.0 - r5) + r5
        ell_bar = (1.0 - U) * M_vec * L

        if float(r_norm[i]) <= 0.5:
            V = (
                ell * (2.0 * r4 * Xi - Xb)
                + ell_bar * (Xa - Xb)
                + (1.0 - r_norm[i]) * F * U1 * r5 * span
            )
        else:
            V = (
                r4 * L * (Xa - Xi)
                + (1.0 - r_norm[i]) * F * U2 * r5 * (r3 * ub - lb)
            )

        return V, F, U, U1, Xa, Xb, mi, Ri

    def _run_one_iteration(self, iteration):
        self._update_global_best()
        masses, sun_mass = self._mass_values()
        sun = self.best_position

        distances = xp.sqrt(xp.sum((self.population - sun.reshape(1, -1)) ** 2, axis=1))
        r_norm = self._normalized_distance(distances)
        mu_t = self.mu0 * float(np.exp(-self.gamma * (iteration + 1) / max(self.max_iter, 1)))

        new_population = self.population.copy()
        new_fitness = self.fitness.copy()

        a2 = self._cyclic_a2(iteration)

        for i in range(self.planets):
            if self.max_fes is not None and self.fe_count >= self.max_fes:
                break

            V, F, U, U1, Xa, Xb, mi, Ri = self._make_velocity(
                i, iteration, masses, sun_mass, distances, r_norm, mu_t
            )

            Xi = self.population[i]
            grav = (
                float(self.eccentricity[i])
                * mu_t
                * float(sun_mass)
                * mi
                / (Ri * Ri + self.eps)
                + float(xp.random.uniform(0.0, 1.0))
            )

            if float(xp.random.uniform(0.0, 1.0)) < 0.5:
                candidate = Xi + F * V + (grav + abs(float(np.random.normal()))) * U * (sun - Xi)
            else:
                r4 = float(xp.random.uniform(0.0, 1.0))
                eta = (a2 - 1.0) * r4 + 1.0
                h = 1.0 / float(np.exp(eta * float(np.random.normal())))
                center = (Xi + sun + Xa) / 3.0
                candidate = Xi * U1 + (1.0 - U1) * (center + h * (center - Xb))

            candidate = xp.clip(candidate, self.min_params, self.max_params)
            cand_fit = self.evaluate(candidate)

            if self.is_better(cand_fit, float(self.fitness[i])):
                new_population[i] = candidate
                new_fitness[i] = cand_fit

        self.population = new_population
        self.fitness = new_fitness
        self._update_global_best()

    def _sort_by_fitness(self, population, fitness):
        idx = xp.argsort(fitness)
        if self.maximize:
            idx = idx[::-1]
        return population[idx], fitness[idx]

    def _update_archive(self):
        population = self.population
        fitness = self.fitness

        if self.archive_positions is not None:
            population = xp.concatenate([population, self.archive_positions], axis=0)
            fitness = xp.concatenate([fitness, self.archive_fitness], axis=0)

        population, fitness = self._sort_by_fitness(population, fitness)
        keep = min(self.archive_size, int(population.shape[0]))

        self.archive_positions = population[:keep].copy()
        self.archive_fitness = fitness[:keep].copy()

        self.best_position = self.archive_positions[0].copy()
        self.best_value = float(self.archive_fitness[0])

    def _inject_archive(self):
        if self.archive_positions is None:
            return

        # Replace the worst planets with archived elites, preserving population size.
        ordered = xp.argsort(self.fitness)
        if not self.maximize:
            ordered = ordered[::-1]

        n = min(int(self.archive_positions.shape[0]), int(self.population.shape[0]))
        for rank in range(n):
            idx = int(ordered[rank])
            self.population[idx] = self.archive_positions[rank].copy()
            self.fitness[idx] = self.archive_fitness[rank]

    def _save_history(self):
        self.history["best_fitness"].append(float(self.best_value))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def run(self, verbose=True):
        self.population = self.initialize_positions()
        self.eccentricity = xp.random.uniform(0.0, 1.0, size=self.planets)
        self.period = xp.abs(xp.random.normal(0.0, 1.0, size=self.planets)) + self.eps
        self.fitness = self.evaluate_population(self.population)
        self._update_global_best()
        self._update_archive()
        self._inject_archive()

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with AKOA",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            if self.max_fes is not None and self.fe_count >= self.max_fes:
                break

            self._run_one_iteration(iteration)
            self._update_archive()
            self._inject_archive()
            self._save_history()

            iterator.set_postfix({"Best": f"{float(self.best_value):.6f}"})

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(self.best_value)


def run_akoa(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=300,
    verbose=False,
    progress_callback=None,
    seed=None,
    max_fes=None,
    archive_size=4,
):
    if seed is not None:
        np.random.seed(seed)
        try:
            xp.random.seed(seed)
        except Exception:
            pass

    model = ArchivedKeplerOptimizationAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        planets=population_size,
        max_iter=max_iter,
        max_fes=max_fes,
        maximize=False,
        progress_callback=progress_callback,
        archive_size=archive_size,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
