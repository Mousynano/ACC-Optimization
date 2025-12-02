import numpy as np
from random import random, uniform
from tqdm import tqdm
import matplotlib.pyplot as plt

# ===============================
#  Stochastic Komodo Algorithm (SKA)
# ===============================
class StochasticKomodoAlgorithm:
    def __init__(self, fitness_function, min_params, max_params,
                 pop_size=30, max_iter=100, g1=0.35, g2=0.7,
                 w1=0.5, w2=0.5, rs=0.01, nC=5, maximize=True, progress_callback=None):
        self.fitness_function = fitness_function
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

        # Tracking history
        self.history = {
            "best_fitness": [],
            "best_position": []
        }

        self.progress_callback = progress_callback

    # Inisialisasi populasi awal
    def initialize(self):
        return np.array([
            np.array([uniform(self.min_params[d], self.max_params[d])
                      for d in range(self.dim)])
            for _ in range(self.pop_size)
        ])

    # Evaluasi fitness
    def evaluate(self, position):
        return self.fitness_function(position)

    # Update solusi terbaik global
    def global_update(self, k, k_best):
        if k_best is None:
            return k
        f_k, f_best = self.evaluate(k), self.evaluate(k_best)
        if (self.maximize and f_k > f_best) or (not self.maximize and f_k < f_best):
            return np.copy(k)
        return k_best

    # Perilaku big male
    def big_male_move(self, k_i, K, f_values):
        better_idx = np.where(f_values > self.evaluate(k_i))[0] if self.maximize else np.where(f_values < self.evaluate(k_i))[0]
        if len(better_idx) == 0:
            return k_i
        cb_i = np.mean(K[better_idx], axis=0)
        new_k = self.w1 * k_i + (1 - self.w1) * cb_i
        return np.clip(new_k, self.min_params, self.max_params)

    # Perilaku female (parthenogenesis)
    def generate_candidates(self, k_i):
        candidates = []
        for _ in range(self.nC):
            c = k_i + np.random.uniform(-0.5, 0.5, self.dim) * self.rs * (self.max_params - self.min_params)
            c = np.clip(c, self.min_params, self.max_params)
            candidates.append(c)
        return np.array(candidates)

    # Perilaku small male
    def small_male_move(self, k_i, k_best):
        new_k = self.w2 * k_i + (1 - self.w2) * k_best
        return np.clip(new_k, self.min_params, self.max_params)

    # Jalankan algoritma utama
    def run(self, verbose=True):
        K = self.initialize()
        f_values = np.array([self.evaluate(k) for k in K])
        best_idx = np.argmax(f_values) if self.maximize else np.argmin(f_values)
        k_best = np.copy(K[best_idx])
        best_value = f_values[best_idx]

        iterator = tqdm(range(self.max_iter), desc="Optimizing with SKA", ncols=100, disable=not verbose)

        for t in iterator:
            for i in range(self.pop_size):
                r = random()
                if r < self.g1:
                    s = 'big_male'
                elif r < self.g2:
                    s = 'female'
                else:
                    s = 'small_male'

                if s == 'big_male':
                    K[i] = self.big_male_move(K[i], K, f_values)
                    k_best = self.global_update(K[i], k_best)

                elif s == 'female':
                    C = self.generate_candidates(K[i])
                    fC = np.array([self.evaluate(c) for c in C])
                    idx = np.argmax(fC) if self.maximize else np.argmin(fC)
                    c_best = C[idx]
                    if (self.maximize and fC[idx] > self.evaluate(K[i])) or (not self.maximize and fC[idx] < self.evaluate(K[i])):
                        K[i] = c_best
                        k_best = self.global_update(K[i], k_best)

                else:
                    K[i] = self.small_male_move(K[i], k_best)

            f_values = np.array([self.evaluate(k) for k in K])
            best_idx = np.argmax(f_values) if self.maximize else np.argmin(f_values)
            best_value = f_values[best_idx]
            k_best = self.global_update(K[best_idx], k_best)

            # Simpan hasil tiap iterasi
            self.history["best_fitness"].append(best_value)
            self.history["best_position"].append(k_best.copy())

            if self.progress_callback is not None:
                self.progress_callback(t + 1)

            iterator.set_postfix({"Best": f"{best_value:.6f}"})

        return k_best, best_value

def run_ska(
    func,
    min_params,
    max_params,
    population_size,
    max_iter=100,
    verbose=False,
    progress_callback=None
):
    model = StochasticKomodoAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        pop_size=population_size,
        max_iter=max_iter,
        maximize=False,
        progress_callback=progress_callback
    )

    best_params, best_fitness = model.run(
        verbose=verbose,
    )

    return best_params, best_fitness, model.history["best_fitness"]
