try:
    from core.xp import xp, to_cpu, GPU_AVAILABLE
except Exception:  # fallback for standalone testing outside the project
    import numpy as xp
    GPU_AVAILABLE = False

    def to_cpu(x):
        return xp.asarray(x)

import numpy as np
from tqdm import tqdm


class AtomicOrbitalSearch:
    """
    Atomic Orbital Search (AOS) draft implementation.

    This follows the project optimizer style:
    - xp backend arrays
    - CPU-safe objective evaluation
    - class + run_aos wrapper
    - history["best_fitness"] and history["best_position"]

    Default mode is minimization.
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        electrons=100,
        max_iter=300,
        max_fes=None,
        maximize=False,
        n_layers=None,
        photon_rate=None,
        random_walk_scale=0.01,
        eps=1e-12,
        progress_callback=None,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)
        self.electrons = int(electrons)
        self.max_iter = int(max_iter)
        self.max_fes = None if max_fes is None else int(max_fes)
        self.maximize = maximize
        self.dim = len(min_params)
        self.n_layers = n_layers
        self.photon_rate = photon_rate
        self.random_walk_scale = float(random_walk_scale)
        self.eps = float(eps)
        self.progress_callback = progress_callback

        self.history = {"best_fitness": [], "best_position": []}
        self.fe_count = 0

        self.population = None
        self.fitness = None
        self.best_position = xp.zeros(self.dim)
        self.best_value = -float("inf") if maximize else float("inf")

    def initialize_positions(self):
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.electrons, self.dim),
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

    def _sort_population(self):
        order = xp.argsort(self.fitness)
        if self.maximize:
            order = order[::-1]
        self.population = self.population[order]
        self.fitness = self.fitness[order]

    def _update_global_best(self):
        idx = 0
        value = float(self.fitness[idx])
        if self.is_better(value, self.best_value):
            self.best_value = value
            self.best_position = self.population[idx].copy()

    def _layer_indices(self):
        n_layers = self.n_layers
        if n_layers is None:
            n_layers = max(2, int(round(np.sqrt(self.electrons))))
        n_layers = max(1, min(int(n_layers), self.electrons))

        # Sorted candidates are divided into concentric layers. Better candidates
        # are assigned to inner layers.
        indices = np.arange(self.electrons)
        return np.array_split(indices, n_layers)

    def _run_one_iteration(self):
        self._sort_population()
        self._update_global_best()

        atom_binding_state = xp.mean(self.population, axis=0)
        atom_binding_energy = float(xp.mean(self.fitness))
        lowest_energy = self.population[0].copy()

        new_population = self.population.copy()
        new_fitness = self.fitness.copy()
        span = self.max_params - self.min_params

        layers = self._layer_indices()

        for layer_number, idx_group in enumerate(layers, start=1):
            if len(idx_group) == 0:
                continue

            idx_xp = xp.array(idx_group, dtype=int)
            layer_pop = self.population[idx_xp]
            layer_fit = self.fitness[idx_xp]

            layer_binding_state = xp.mean(layer_pop, axis=0)
            layer_binding_energy = float(xp.mean(layer_fit))

            if self.maximize:
                local_best_idx = int(xp.argmax(layer_fit))
            else:
                local_best_idx = int(xp.argmin(layer_fit))
            local_lowest_energy = layer_pop[local_best_idx].copy()

            PR_layer = float(xp.random.uniform(0.0, 1.0)) if self.photon_rate is None else float(self.photon_rate)

            for raw_idx in idx_group:
                if self.max_fes is not None and self.fe_count >= self.max_fes:
                    break

                i = int(raw_idx)
                Xi = self.population[i]
                Ei = float(self.fitness[i])
                phi = float(xp.random.uniform(0.0, 1.0))
                alpha = xp.random.uniform(0.0, 1.0, size=self.dim)
                beta = xp.random.uniform(0.0, 1.0, size=self.dim)
                gamma = xp.random.uniform(0.0, 1.0, size=self.dim)

                if phi >= PR_layer:
                    # Photon action: emission or absorption.
                    worse_than_layer_mean = Ei >= layer_binding_energy if not self.maximize else Ei <= layer_binding_energy

                    if worse_than_layer_mean:
                        # Emission: move with respect to global lowest energy and atom binding state.
                        candidate = Xi + (alpha * (beta * lowest_energy - gamma * atom_binding_state)) / float(layer_number)
                    else:
                        # Absorption: move with respect to layer lowest energy and layer binding state.
                        candidate = Xi + alpha * (beta * local_lowest_energy - gamma * layer_binding_state)
                else:
                    # Other interactions: small random walk.
                    ri = xp.random.uniform(-1.0, 1.0, size=self.dim) * span * self.random_walk_scale
                    candidate = Xi + ri

                candidate = xp.clip(candidate, self.min_params, self.max_params)
                cand_fit = self.evaluate(candidate)

                if self.is_better(cand_fit, float(self.fitness[i])):
                    new_population[i] = candidate
                    new_fitness[i] = cand_fit

        self.population = new_population
        self.fitness = new_fitness
        self._sort_population()
        self._update_global_best()

    def _save_history(self):
        self.history["best_fitness"].append(float(self.best_value))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def run(self, verbose=True):
        self.population = self.initialize_positions()
        self.fitness = self.evaluate_population(self.population)
        self._sort_population()
        self._update_global_best()

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with AOS",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            if self.max_fes is not None and self.fe_count >= self.max_fes:
                break

            self._run_one_iteration()
            self._save_history()

            iterator.set_postfix({"Best": f"{float(self.best_value):.6f}"})

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(self.best_value)


def run_aos(
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

    model = AtomicOrbitalSearch(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        electrons=population_size,
        max_iter=max_iter,
        max_fes=max_fes,
        maximize=False,
        progress_callback=progress_callback,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
