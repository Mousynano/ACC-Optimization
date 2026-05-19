"""
RIME Iteration Version (Python)
--------------------------------
Conversion of the MATLAB RIME iteration-based version into the same design
style as the provided PSO/HOA Python optimizers.

Original MATLAB loop criterion:
    while it <= Max_iter

This Python version is intended for minimization by default, which matches the
original RIME benchmark code and typical ACC objective/error minimization.
"""

try:
    from core.xp import xp, to_cpu, GPU_AVAILABLE
except Exception:  # fallback agar file tetap bisa dites standalone dengan NumPy
    import numpy as xp
    GPU_AVAILABLE = False

    def to_cpu(arr):
        return arr.get() if hasattr(arr, "get") else arr

# from core.seeding import seed_everything
from tqdm import tqdm


class RIMEOptimizationAlgorithm:
    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        population_size=100,
        max_iter=300,
        maximize=False,
        w=5,
        progress_callback=None,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.population_size = population_size
        self.max_iter = max_iter
        self.maximize = maximize
        self.w = w  # Soft-rime parameter W in the MATLAB code
        self.dim = len(min_params)
        self.progress_callback = progress_callback

        self.best_position = xp.zeros(self.dim)
        self.best_fitness = -xp.inf if maximize else xp.inf

        # History disimpan di CPU agar aman untuk plotting/serialisasi.
        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

    def initialize_positions(self):
        """Equivalent to MATLAB initialization(SearchAgents_no, dim, ub, lb)."""
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.population_size, self.dim),
        )

    def evaluate(self, pos_xp):
        """
        Evaluasi satu kandidat.
        Fitness ACC/objective biasanya berjalan di CPU, maka posisi dikonversi
        ke CPU sebelum dipanggil.
        """
        pos_cpu = to_cpu(pos_xp)
        return float(self.fitness_function(pos_cpu))

    def _is_better(self, value, reference):
        return value > reference if self.maximize else value < reference

    @staticmethod
    def _matlab_round_positive(x):
        """MATLAB round untuk nilai positif: halves round away from zero."""
        return xp.floor(x + 0.5)

    def _normalize_rime_rates(self, rime_rates):
        """
        MATLAB normr(Rime_rates) untuk vektor 1xN ≈ v / ||v||_2.
        Pada objective ACC umumnya fitness non-negatif, sehingga hasilnya bisa
        dipakai sebagai probabilitas hard-rime.
        """
        norm_value = xp.linalg.norm(rime_rates)
        if float(to_cpu(norm_value)) == 0.0:
            return xp.zeros_like(rime_rates)
        normalized = rime_rates / norm_value
        # Probabilitas harus berada pada [0, 1]. Clip menjaga stabilitas jika
        # objective menghasilkan nilai negatif/aneh.
        return xp.clip(normalized, 0.0, 1.0)

    def _evaluate_initial_population(self, population):
        fitness = xp.array([
            self.evaluate(population[i])
            for i in range(self.population_size)
        ])

        if self.maximize:
            best_idx = int(fitness.argmax())
        else:
            best_idx = int(fitness.argmin())

        self.best_fitness = float(fitness[best_idx])
        self.best_position = population[best_idx].copy()

        return fitness

    def _soft_hard_rime_update(self, population, fitness, iteration):
        """Vectorized equivalent of the nested i-j update loop in MATLAB RIME."""
        # RimeFactor = (rand-0.5)*2*cos((pi*it/(Max_iter/10))) *
        #              (1-round(it*W/Max_iter)/W)
        random_scalar = float(to_cpu(xp.random.random()))
        decay = 1.0 - float(to_cpu(self._matlab_round_positive(iteration * self.w / self.max_iter))) / self.w
        rime_factor = (
            (random_scalar - 0.5)
            * 2.0
            * xp.cos(xp.pi * iteration / (self.max_iter / 10.0))
            * decay
        )

        # E = sqrt(it / Max_iter)
        e_factor = xp.sqrt(iteration / self.max_iter)

        new_population = population.copy()
        best_row = self.best_position.reshape(1, -1)

        # Soft-rime search strategy, Eq. (3)
        random_soft = xp.random.random(size=(self.population_size, self.dim))
        random_positions = (
            (self.max_params - self.min_params).reshape(1, -1)
            * xp.random.random(size=(self.population_size, self.dim))
            + self.min_params.reshape(1, -1)
        )
        soft_candidates = best_row + rime_factor * random_positions
        new_population = xp.where(random_soft < e_factor, soft_candidates, new_population)

        # Hard-rime puncture mechanism, Eq. (7)
        normalized_rates = self._normalize_rime_rates(fitness).reshape(-1, 1)
        random_hard = xp.random.random(size=(self.population_size, self.dim))
        new_population = xp.where(random_hard < normalized_rates, best_row, new_population)

        # Boundary absorption
        new_population = xp.clip(new_population, self.min_params, self.max_params)

        return new_population

    def _greedy_selection(self, population, new_population, fitness):
        new_fitness_list = [
            self.evaluate(new_population[i])
            for i in range(self.population_size)
        ]
        new_fitness = xp.array(new_fitness_list)

        if self.maximize:
            better_mask = new_fitness > fitness
        else:
            better_mask = new_fitness < fitness

        better_mask_col = better_mask.reshape(-1, 1)
        population = xp.where(better_mask_col, new_population, population)
        fitness = xp.where(better_mask, new_fitness, fitness)

        return population, fitness

    def _update_global_best(self, population, fitness):
        if self.maximize:
            best_idx = int(fitness.argmax())
        else:
            best_idx = int(fitness.argmin())

        candidate_fitness = float(fitness[best_idx])
        if self._is_better(candidate_fitness, self.best_fitness):
            self.best_fitness = candidate_fitness
            self.best_position = population[best_idx].copy()

    def _save_history(self):
        self.history["best_fitness"].append(float(self.best_fitness))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def run(self, verbose=True):
        population = self.initialize_positions()
        fitness = self._evaluate_initial_population(population)

        iterator = tqdm(
            range(1, self.max_iter + 1),
            desc="Optimizing with RIME",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            new_population = self._soft_hard_rime_update(population, fitness, iteration)
            population, fitness = self._greedy_selection(population, new_population, fitness)
            self._update_global_best(population, fitness)
            self._save_history()

            iterator.set_postfix({"Best": f"{self.best_fitness:.6f}"})

            if self.progress_callback is not None:
                self.progress_callback(iteration)

        return to_cpu(self.best_position), float(self.best_fitness)


def run_rime(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=300,
    verbose=False,
    progress_callback=None,
    seed=None,
):
    """
    Wrapper kompatibel dengan pola run_pso/run_hoa.
    Return:
        best_params, best_fitness, history_best_fitness
    """

    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = RIMEOptimizationAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        maximize=False,  # ACC/error objective -> minimization
        progress_callback=progress_callback,
    )

    best_params, best_fitness = model.run(verbose=verbose)

    return (
        best_params,
        best_fitness,
        model.history["best_fitness"],
    )
