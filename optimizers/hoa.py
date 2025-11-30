import numpy as np
from random import uniform
from tqdm import tqdm
import matplotlib.pyplot as plt

class HikingOptimizationAlgorithm:
    def __init__(self, fitness_function, obj_function, min_params, max_params,
                 hikers=30, max_iter=100, maximize=True):  # default: maximize
        self.fitness_function = fitness_function
        self.obj_function = obj_function
        self.min_params = np.array(min_params)
        self.max_params = np.array(max_params)
        self.dim = len(min_params)
        self.hikers = hikers
        self.max_iter = max_iter
        self.maximize = maximize

        # History tracking
        self.history = {"best_fitness": [], "best_position": []}

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
        return self.fitness_function(pos, self.obj_function)

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

        return beta_best, f_best

# === Example: Test HOA on Rastrigin Function ===
def rastrigin(x):
    A = 10
    value = A * len(x) + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
    return -value  # balik supaya semakin besar fitness, semakin baik

def run_hoa(func, min_params, max_params, population_size, max_iter=100, verbose=False):
    """
    Runner for Hiking Optimization Algorithm (HOA)
    """

    iae_model = HikingOptimizationAlgorithm(
        obj_function="iae",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        hikers=population_size,
        max_iter=max_iter,
        maximize=False,
    )

    ise_model = HikingOptimizationAlgorithm(
        obj_function="ise",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        hikers=population_size,
        max_iter=max_iter,
        maximize=False,
    )

    itae_model = HikingOptimizationAlgorithm(
        obj_function="itae",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        hikers=population_size,
        max_iter=max_iter,
        maximize=False,
    )

    itse_model = HikingOptimizationAlgorithm(
        obj_function="itse",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        hikers=population_size,
        max_iter=max_iter,
        maximize=False,
    )

    iae_best_params, iae_best_fitness = iae_model.run(verbose=verbose)
    ise_best_params, ise_best_fitness = ise_model.run(verbose=verbose)
    itae_best_params, itae_best_fitness = itae_model.run(verbose=verbose)
    itse_best_params, itse_best_fitness = itse_model.run(verbose=verbose)

    best_params = {
        "iae": iae_best_params, 
        "ise": ise_best_params, 
        "itae": itae_best_params, 
        "itse": itse_best_params
    }
    best_fitness = {
        "iae": iae_best_fitness, 
        "ise": ise_best_fitness, 
        "itae": itae_best_fitness, 
        "itse": itse_best_fitness
    }
    curve = {
        "iae": iae_model.history.get("best_fitness", []),
        "ise": ise_model.history.get("best_fitness", []),
        "itae": itae_model.history.get("best_fitness", []),
        "itse": itse_model.history.get("best_fitness", []),
    }

    return best_params, best_fitness, curve
