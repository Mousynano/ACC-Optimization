from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # untuk return / plotting / operasi aman di CPU
from random import random, seed
from tqdm import tqdm


class StochasticKomodoAlgorithm:
    def __init__(self, fitness_function, min_params, max_params,
                 pop_size=100, max_iter=300, g1=0.35, g2=0.7,
                 w1=0.5, w2=0.5, rs=0.01, nC=5,
                 maximize=True, progress_callback=None):

        self.fitness_function = fitness_function

        # backend-agnostic (numpy/cupy)
        self.min_params = np.array(min_params)
        self.max_params = np.array(max_params)

        self.dim = len(min_params)
        self.pop_size = pop_size
        self.max_iter = max_iter

        self.g1, self.g2 = g1, g2
        self.w1, self.w2 = w1, w2
        self.rs = rs
        self.nC = nC
        self.maximize = maximize

        # history disimpan di CPU (supaya gampang di-plot / diserialisasi)
        self.history = {
            "best_fitness": [],
            "best_position": []
        }

        self.progress_callback = progress_callback

        # cache fitness global
        self.f_values = None          # fitness tiap individu (xp array)
        self.f_best = None            # float (CPU)
        self.k_best = None            # posisi terbaik (xp array)

    # -------------------------
    #  Util fungsi kecil
    # -------------------------
    def _is_better(self, new_val, old_val):
        """Cek apakah new_val lebih baik dari old_val sesuai mode."""
        if old_val is None:
            return True
        return (new_val > old_val) if self.maximize else (new_val < old_val)

    def _update_global_best(self, k, f):
        if self._is_better(f, self.f_best):
            self.f_best = f
            self.k_best = np.copy(k)

    # -------------------------
    #  Inisialisasi & evaluasi
    # -------------------------
    def initialize(self):
        """Inisialisasi populasi dengan backend GPU/CPU."""
        K = np.zeros((self.pop_size, self.dim))

        for i in range(self.pop_size):
            # tetap pakai random() biar deterministic antar backend
            K[i] = np.asarray([
                self.min_params[d] + (self.max_params[d] - self.min_params[d]) * random()
                for d in range(self.dim)
            ])

        return K

    def evaluate(self, position):
        """Evaluasi fitness: wajib CPU (simulasi ACC, dsb)."""
        return float(self.fitness_function(to_cpu(position)))

    # -------------------------
    #  Perilaku komodo
    # -------------------------
    def big_male_move(self, idx, K, f_vals):
        """
        Perilaku big male (eksploitasi):
        - pakai fitness yang sudah di-cache (f_vals), tidak re-evaluate.
        """
        f_i = float(to_cpu(f_vals[idx]))

        if self.maximize:
            better_idx = np.where(f_vals > f_i)[0]
        else:
            better_idx = np.where(f_vals < f_i)[0]

        if better_idx.size == 0:
            return K[idx]  # tidak ada yang lebih baik, tetap

        cb_i = np.mean(K[better_idx], axis=0)
        new_k = self.w1 * K[idx] + (1 - self.w1) * cb_i

        return np.clip(new_k, self.min_params, self.max_params)
    
    def __big_male_step(self, i, K):
        new_k = self.big_male_move(i, K, self.f_values)
        new_f = self.evaluate(new_k)

        old_f = float(to_cpu(self.f_values[i]))
        if self._is_better(new_f, old_f):
            K[i] = new_k
            self.f_values[i] = new_f
            self._update_global_best(new_k, new_f)

    def __female_step(self, i, K):
        C = self.generate_candidates(K[i])
        fC = np.asarray([self.evaluate(c) for c in C], dtype=float)

        idx = int(np.argmax(fC)) if self.maximize else int(np.argmin(fC))
        c_best = C[idx]
        f_best = float(to_cpu(fC[idx]))

        old_f = float(to_cpu(self.f_values[i]))
        if self._is_better(f_best, old_f):
            K[i] = c_best
            self.f_values[i] = f_best
            self._update_global_best(c_best, f_best)

    def __small_male_step(self, i, K):
        new_k = self.small_male_move(K[i], self.k_best)
        new_f = self.evaluate(new_k)

        old_f = float(to_cpu(self.f_values[i]))
        if self._is_better(new_f, old_f):
            K[i] = new_k
            self.f_values[i] = new_f
            self._update_global_best(new_k, new_f)

    def __sync_global_best(self, K):
        idx = int(np.argmax(self.f_values)) if self.maximize else int(np.argmin(self.f_values))
        f = float(to_cpu(self.f_values[idx]))
        if self._is_better(f, self.f_best):
            self.f_best = f
            self.k_best = np.copy(K[idx])

    def generate_candidates(self, k_i):
        """
        Generate kandidat betina (parthenogenesis/mutation):
        full vectorized di GPU/CPU backend.
        """
        noise = np.random.uniform(-0.5, 0.5, (self.nC, self.dim))
        scaled = noise * self.rs * (self.max_params - self.min_params)
        C = k_i + scaled
        return np.clip(C, self.min_params, self.max_params)

    def small_male_move(self, k_i, k_best):
        """Perilaku small male (mlipir mendekati best)."""
        new_k = self.w2 * k_i + (1 - self.w2) * k_best
        return np.clip(new_k, self.min_params, self.max_params)

    # -------------------------
    #  Main optimization loop
    # -------------------------
    def run(self, verbose=True):
        # init populasi
        K = self.initialize()

        # initial fitness untuk setiap individu (CPU sekali)
        f_vals_list = [self.evaluate(K[i]) for i in range(self.pop_size)]
        self.f_values = np.asarray(f_vals_list, dtype=float)

        # best awal dari populasi
        if self.maximize:
            best_idx = int(np.argmax(self.f_values))
        else:
            best_idx = int(np.argmin(self.f_values))

        self.k_best = np.copy(K[best_idx])
        self.f_best = float(to_cpu(self.f_values[best_idx]))

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with SKA",
            ncols=100,
            disable=not verbose
        )

        for t in iterator:
            # loop tiap komodo
            for i in range(self.pop_size):
                r = random()
                if r < self.g1:
                    # move_type = 'big'
                    self.__big_male_step(i, K)
                elif r < self.g2:
                    # move_type = 'female'
                    self.__female_step(i, K)
                else:
                    # move_type = 'small'
                    self.__small_male_step(i, K)

                self.__sync_global_best(K)

            # simpan history (CPU)
            self.history["best_fitness"].append(self.f_best)
            self.history["best_position"].append(to_cpu(self.k_best.copy()))

            if self.progress_callback is not None:
                self.progress_callback(t + 1)

            iterator.set_postfix({"Best": f"{self.f_best:.6f}"})

        # return dalam bentuk CPU
        return to_cpu(self.k_best), self.f_best


def run_ska(func,
            min_params,
            max_params,
            population_size,
            max_iter=100,
            verbose=False,
            progress_callback=None,
            seed=None):
    """
    Wrapper biar kompatibel sama objective_worker kamu.
    Minimization by default (maximize=False).
    """
    
    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = StochasticKomodoAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        pop_size=population_size,
        max_iter=max_iter,
        maximize=False,
        progress_callback=progress_callback
    )

    best_params, best_fitness = model.run(verbose=verbose)

    return best_params, best_fitness, model.history["best_fitness"]
