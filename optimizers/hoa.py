import numpy as np
from random import uniform
from tqdm import tqdm
import matplotlib.pyplot as plt

class HikingOptimizationAlgorithm:
    def __init__(self, fitness_function, min_params, max_params,
                 hikers=30, max_iter=100, maximize=True, progress_callback=None):  # default: maximize
        self.fitness_function = fitness_function
        self.min_params = np.array(min_params)
        self.max_params = np.array(max_params)
        self.dim = len(min_params)
        self.hikers = hikers
        self.max_iter = max_iter
        self.maximize = maximize

        # History tracking
        self.history = {"best_fitness": [], "best_position": []}

        self.progress_callback = progress_callback

    # Tobler’s Hiking Function (Eq. 1)
    def toblers_velocity(self, slope):
        return 6 * np.exp(-3.5 * np.abs(slope + 0.05))

    # Compute slope (Eq. 2)
    def slope(self, theta_deg):
        return np.tan(np.deg2rad(theta_deg))

    # Initialize positions (Eq. 5)
    def initialize_positions(self):
        return np.array([
            [uniform(self.min_params[d], self.max_params[d]) for d in range(self.dim)]
            for _ in range(self.hikers)
        ])

    def evaluate(self, pos):
        return self.fitness_function(pos)

    def run(self, verbose=True):
        beta = self.initialize_positions()
        fitness = np.array([self.evaluate(b) for b in beta])

        # best = maksimum
        best_idx = np.argmax(fitness)
        beta_best = np.copy(beta[best_idx])
        f_best = fitness[best_idx]

        iterator = tqdm(range(self.max_iter), desc="Optimizing with HOA", ncols=100, disable=not verbose)

        for t in iterator:
            for i in range(self.hikers):
                # random elevation angle [0, 50°]
                theta_i = uniform(0, 50)
                slope_i = self.slope(theta_i)

                # Eq. (1) Tobler’s hiking velocity
                Wi_prev = self.toblers_velocity(slope_i)

                # random sweep factor α ∈ [1, 3], and random γ ∈ [0, 1]
                alpha_i = uniform(1, 3)
                gamma_i = uniform(0, 1)

                # Eq. (3) updated velocity
                Wi = Wi_prev + gamma_i * (beta_best - alpha_i * beta[i])

                # Eq. (4) position update
                beta_new = beta[i] + Wi

                # bound control
                beta_new = np.clip(beta_new, self.min_params, self.max_params)

                # evaluate new fitness
                f_new = self.evaluate(beta_new)

                # keep better fitness (maximize)
                if f_new > fitness[i]:
                    beta[i] = beta_new
                    fitness[i] = f_new

            # Update best hiker (maximize)
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > f_best:
                beta_best = np.copy(beta[best_idx])
                f_best = fitness[best_idx]

            # Save iteration data
            self.history["best_fitness"].append(f_best)
            self.history["best_position"].append(beta_best.copy())
            iterator.set_postfix({"Best": f"{f_best:.6f}"})

            if self.progress_callback is not None:
                self.progress_callback(t + 1)

        return beta_best, f_best

def run_hoa(
    func,
    min_params,
    max_params,
    population_size,
    max_iter=100,
    verbose=False,
    progress_callback=None
):
    model = HikingOptimizationAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        hikers=population_size,
        max_iter=max_iter,
        maximize=False,
        progress_callback=progress_callback
    )

    best_params, best_fitness = model.run(
        verbose=verbose,
    )

    return best_params, best_fitness, model.history["best_fitness"]
