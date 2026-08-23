try:
    from core.xp import xp, to_cpu, GPU_AVAILABLE
except Exception:  # fallback for standalone testing outside the project
    import numpy as xp
    GPU_AVAILABLE = False

    def to_cpu(x):
        return xp.asarray(x)

import numpy as np
from tqdm import tqdm


class ArtificialProtozoaOptimizer:
    """
    Artificial Protozoa Optimizer (APO) draft implementation.

    Implements the four main behaviors from the paper:
    - autotrophic foraging
    - heterotrophic foraging
    - dormancy
    - reproduction

    Default mode is minimization for ACC/objective-error tuning.
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        protozoa=100,
        max_iter=300,
        max_fes=None,
        maximize=False,
        npairs=1,
        pf_max=0.1,
        eps=1e-12,
        progress_callback=None,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)
        self.protozoa = int(protozoa)
        self.max_iter = int(max_iter)
        self.max_fes = None if max_fes is None else int(max_fes)
        self.maximize = maximize
        self.dim = len(min_params)
        self.npairs = max(1, int(npairs))
        self.pf_max = float(pf_max)
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
            size=(self.protozoa, self.dim),
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
        value = float(self.fitness[0])
        if self.is_better(value, self.best_value):
            self.best_value = value
            self.best_position = self.population[0].copy()

    def _mapping_vector(self, count):
        count = int(max(1, min(self.dim, count)))
        idx = np.random.choice(self.dim, size=count, replace=False)
        mask = xp.zeros(self.dim)
        for j in idx:
            mask[int(j)] = 1.0
        return mask

    def _autotroph(self, rank_i, f_factor, mf):
        Xi = self.population[rank_i]
        j = int(xp.random.randint(0, self.protozoa))
        Xj = self.population[j]

        interaction = xp.zeros(self.dim)
        max_pair = min(self.npairs, max(1, (self.protozoa - 1) // 2))

        for _ in range(max_pair):
            if rank_i <= 0:
                k_minus = 0
            else:
                k_minus = int(xp.random.randint(0, rank_i + 1))

            if rank_i >= self.protozoa - 1:
                k_plus = self.protozoa - 1
            else:
                k_plus = int(xp.random.randint(rank_i, self.protozoa))

            fit_minus = abs(float(self.fitness[k_minus]))
            fit_plus = abs(float(self.fitness[k_plus]))
            wa = float(np.exp(-abs(fit_minus / (fit_plus + self.eps))))
            interaction = interaction + wa * (self.population[k_minus] - self.population[k_plus])

        interaction = interaction / float(max_pair)
        return Xi + f_factor * (Xj - Xi + interaction) * mf

    def _heterotroph(self, rank_i, iteration, f_factor, mf):
        Xi = self.population[rank_i]
        decay = 1.0 - (iteration + 1) / max(self.max_iter, 1)
        direction = 1.0 if float(xp.random.uniform(0.0, 1.0)) < 0.5 else -1.0
        Xnear = (1.0 + direction * xp.random.uniform(0.0, 1.0, size=self.dim) * decay) * Xi

        interaction = xp.zeros(self.dim)
        max_pair = min(self.npairs, max(1, (self.protozoa - 1) // 2))

        for k in range(1, max_pair + 1):
            idx_minus = max(0, rank_i - k)
            idx_plus = min(self.protozoa - 1, rank_i + k)

            fit_minus = abs(float(self.fitness[idx_minus]))
            fit_plus = abs(float(self.fitness[idx_plus]))
            wh = float(np.exp(-abs(fit_minus / (fit_plus + self.eps))))
            interaction = interaction + wh * (self.population[idx_minus] - self.population[idx_plus])

        interaction = interaction / float(max_pair)
        return Xi + f_factor * (Xnear - Xi + interaction) * mf

    def _dormancy(self):
        return self.min_params + xp.random.uniform(0.0, 1.0, size=self.dim) * (self.max_params - self.min_params)

    def _reproduction(self, rank_i, mr):
        Xi = self.population[rank_i]
        direction = 1.0 if float(xp.random.uniform(0.0, 1.0)) < 0.5 else -1.0
        random_protozoan = self.min_params + xp.random.uniform(0.0, 1.0, size=self.dim) * (self.max_params - self.min_params)
        return Xi + direction * float(xp.random.uniform(0.0, 1.0)) * random_protozoan * mr

    def _run_one_iteration(self, iteration):
        self._sort_population()
        self._update_global_best()

        pf = self.pf_max * float(xp.random.uniform(0.0, 1.0))
        n_dr = int(np.ceil(self.protozoa * pf))
        if n_dr > 0:
            dr_indices = set(np.random.choice(self.protozoa, size=n_dr, replace=False).tolist())
        else:
            dr_indices = set()

        p_ah = 0.5 * (1.0 + float(np.cos((iteration + 1) / max(self.max_iter, 1) * np.pi)))
        f_factor = float(xp.random.uniform(0.0, 1.0)) * (1.0 + float(np.cos((iteration + 1) / max(self.max_iter, 1) * np.pi)))

        new_population = self.population.copy()
        new_fitness = self.fitness.copy()

        for i in range(self.protozoa):
            if self.max_fes is not None and self.fe_count >= self.max_fes:
                break

            rank_ratio = (i + 1) / float(self.protozoa)

            if i in dr_indices:
                p_dr = 0.5 * (1.0 + float(np.cos((1.0 - rank_ratio) * np.pi)))
                if p_dr > float(xp.random.uniform(0.0, 1.0)):
                    candidate = self._dormancy()
                else:
                    m_count = int(np.ceil(self.dim * float(xp.random.uniform(0.0, 1.0))))
                    mr = self._mapping_vector(m_count)
                    candidate = self._reproduction(i, mr)
            else:
                m_count = int(np.ceil(self.dim * rank_ratio))
                mf = self._mapping_vector(m_count)

                if p_ah > float(xp.random.uniform(0.0, 1.0)):
                    candidate = self._autotroph(i, f_factor, mf)
                else:
                    candidate = self._heterotroph(i, iteration, f_factor, mf)

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
            desc="Optimizing with APO",
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


def run_apo(
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

    model = ArtificialProtozoaOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        protozoa=population_size,
        max_iter=max_iter,
        max_fes=max_fes,
        maximize=False,
        progress_callback=progress_callback,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
