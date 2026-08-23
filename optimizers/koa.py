try:
    from core.xp import xp, to_cpu, GPU_AVAILABLE
except Exception:  # fallback for standalone testing outside the project
    import numpy as xp
    GPU_AVAILABLE = False

    def to_cpu(x):
        return xp.asarray(x)

import numpy as np
from tqdm import tqdm


class KeplerOptimizationAlgorithm:
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

    def _save_history(self):
        self.history["best_fitness"].append(float(self.best_value))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def run(self, verbose=True):
        self.population = self.initialize_positions()
        self.eccentricity = xp.random.uniform(0.0, 1.0, size=self.planets)
        self.period = xp.abs(xp.random.normal(0.0, 1.0, size=self.planets)) + self.eps
        self.fitness = self.evaluate_population(self.population)
        self._update_global_best()

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with KOA",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            if self.max_fes is not None and self.fe_count >= self.max_fes:
                break

            self._run_one_iteration(iteration)
            self._save_history()

            iterator.set_postfix({"Best": f"{float(self.best_value):.6f}"})

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(self.best_value)


def run_koa(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=300,
    verbose=False,
    progress_callback=None,
    seed=None,
    max_fes=None,
):
    if seed is not None:
        np.random.seed(seed)
        try:
            xp.random.seed(seed)
        except Exception:
            pass

    model = KeplerOptimizationAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        planets=population_size,
        max_iter=max_iter,
        max_fes=max_fes,
        maximize=False,
        progress_callback=progress_callback,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
