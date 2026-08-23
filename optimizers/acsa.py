from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # untuk return / plotting / operasi aman di CPU
from random import random, randint
from tqdm import tqdm


class ArchivedCrowSearchAlgorithm:
    def __init__(self, fitness_function, min_params, max_params,
                 pop_size=100, max_iter=300, ap=0.1, fl=2.0,
                 archive_size=4, maximize=True, progress_callback=None):
        self.fitness_function = fitness_function

        # backend-agnostic (numpy/cupy)
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)

        self.dim = len(min_params)
        self.pop_size = pop_size
        self.max_iter = max_iter

        # ACSA parameter: sama seperti CSA + arsip solusi terbaik
        self.ap = ap
        self.fl = fl
        self.archive_size = archive_size
        self.maximize = maximize

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }
        self.progress_callback = progress_callback

        self.crows = None
        self.memory = None
        self.f_values = None
        self.fit_memory = None

        self.best_position = None
        self.best_fitness = None

    # -------------------------
    #  Util fungsi kecil
    # -------------------------
    def __is_better(self, new_val, old_val):
        if old_val is None:
            return True
        return (new_val > old_val) if self.maximize else (new_val < old_val)

    def __evaluate(self, position):
        """Evaluasi fitness: objective/simulasi tetap di CPU."""
        return float(self.fitness_function(to_cpu(position)))

    def __sort_by_fitness(self, positions, fitness):
        sort_indices = xp.argsort(fitness)
        if self.maximize:
            sort_indices = sort_indices[::-1]
        return positions[sort_indices], fitness[sort_indices]

    def __random_position(self, size=None):
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=size if size is not None else self.dim
        )

    def __initialize_position(self):
        return self.__random_position(size=(self.pop_size, self.dim))

    def __initialize_iter(self):
        self.crows = self.__initialize_position()
        self.f_values = xp.array([
            self.__evaluate(self.crows[i])
            for i in range(self.pop_size)
        ])

        # mem = x, fit_mem = ft
        self.memory = self.crows.copy()
        self.fit_memory = self.f_values.copy()

        # archive awal: ambil solusi terbaik sebanyak archive_size
        self.__archive_best_rows()
        self.best_position = self.memory[0].copy()
        self.best_fitness = self.fit_memory[0]

    def __archive_best_rows(self):
        """
        Implementasi ide ACSA:
        sort fit_mem, mem, x, ft dari terbaik ke terburuk,
        lalu simpan hanya baris terbaik sesuai archive_size.
        """
        all_positions = xp.concatenate([self.memory, self.crows], axis=0)
        all_fitness = xp.concatenate([self.fit_memory, self.f_values], axis=0)

        all_positions, all_fitness = self.__sort_by_fitness(all_positions, all_fitness)

        keep = min(self.archive_size, int(all_positions.shape[0]))
        self.memory = all_positions[:keep].copy()
        self.fit_memory = all_fitness[:keep].copy()

        # populasi aktif ikut diperkecil mengikuti archive, sesuai pseudo-code ACSA
        self.crows = self.memory.copy()
        self.f_values = self.fit_memory.copy()
        self.pop_size = keep

    def __update_global_best(self):
        if self.__is_better(self.fit_memory[0], self.best_fitness):
            self.best_fitness = self.fit_memory[0]
            self.best_position = self.memory[0].copy()

    # -------------------------
    #  ACSA movement
    # -------------------------
    def __choose_crow_to_follow(self, i):
        if self.pop_size <= 1:
            return 0

        j = randint(0, self.pop_size - 1)
        while j == i:
            j = randint(0, self.pop_size - 1)
        return j

    def __generate_new_position(self, i):
        j = self.__choose_crow_to_follow(i)

        if random() >= self.ap:
            r = random()
            new_pos = self.crows[i] + r * self.fl * (self.memory[j] - self.crows[i])
        else:
            new_pos = self.__random_position()

        return xp.clip(new_pos, self.min_params, self.max_params)

    def __move_crows(self):
        return xp.array([
            self.__generate_new_position(i)
            for i in range(self.pop_size)
        ])

    def __save_history(self):
        self.history["best_fitness"].append(float(to_cpu(self.best_fitness)))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    # -------------------------
    #  Main optimization loop
    # -------------------------
    def run(self, verbose=True):
        self.__initialize_iter()

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with ACSA",
            ncols=100,
            disable=not verbose
        )

        for t in iterator:
            new_crows = self.__move_crows()
            new_fitness = xp.array([
                self.__evaluate(new_crows[i])
                for i in range(self.pop_size)
            ])

            self.crows = new_crows
            self.f_values = new_fitness

            # Memory update for archived flock + retain top archive_size rows
            self.__archive_best_rows()
            self.__update_global_best()
            self.__save_history()

            if self.progress_callback is not None:
                self.progress_callback(t + 1)

            iterator.set_postfix({"Best": f"{float(to_cpu(self.best_fitness)):.6f}"})

        return to_cpu(self.best_position), float(to_cpu(self.best_fitness))


def run_acsa(func,
             min_params,
             max_params,
             population_size,
             max_iter=100,
             verbose=False,
             progress_callback=None,
             seed=None):
    """
    Wrapper kompatibel dengan runner optimasi lain.
    Minimization by default (maximize=False).
    """

    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = ArchivedCrowSearchAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        pop_size=population_size,
        max_iter=max_iter,
        archive_size=4,
        maximize=False,
        progress_callback=progress_callback
    )

    best_params, best_fitness = model.run(verbose=verbose)

    return best_params, best_fitness, model.history["best_fitness"]
