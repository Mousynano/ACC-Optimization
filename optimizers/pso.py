import numpy as np
import matplotlib.pyplot as plt
from random import random
from tqdm import tqdm


class Particle:
    def __init__(self, min_params, max_params, dimensions=3):
        self.min_params = np.array(min_params)
        self.max_params = np.array(max_params)
        self.position = self.initialize(dimensions)
        self.velocity = np.zeros(dimensions)
        self.best_position = self.position.copy()
        self.best_value = -10000

    def initialize(self, x):
        return np.array([
            self.min_params[i] + (self.max_params[i] - self.min_params[i]) * random()
            for i in range(x)
        ])

    def move(self):
        self.position = np.clip(
            self.position + self.velocity,
            self.min_params,
            self.max_params
        )

class ParticleSwarmOptimization:
    def __init__(self, fitness_function, obj_function, min_params, max_params, maximize=False,
                 n_particles=30, max_iteration=100,
                 w=0.7, c1=1.4, c2=1.2, tolerance=1e-6):
        self.fitness_function = fitness_function
        self.obj_function = obj_function
        self.n_particles = n_particles
        self.max_iteration = max_iteration
        self.particles = [Particle(min_params, max_params, len(min_params))
                          for _ in range(n_particles)]
        self.best_value = -10000
        self.best_position = np.zeros(len(min_params))
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tolerance = tolerance
        self.maximize = maximize

        # === Tracking History ===
        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

    def _evaluate_particles(self):
        for p in self.particles:
            value = self.fitness_function(p.position, self.obj_function)

            # update personal best
            if (self.maximize and value > p.best_value) or \
               ((not self.maximize) and value < p.best_value):
                p.best_value = value
                p.best_position = p.position.copy()

            # update global best
            if (self.maximize and value > self.best_value) or \
               ((not self.maximize) and value < self.best_value):
                self.best_value = value
                self.best_position = p.position.copy()

    def _move_particles(self):
        for particle in self.particles:
            r1, r2 = random(), random()
            cognitive = self.c1 * r1 * (particle.best_position - particle.position)
            social = self.c2 * r2 * (self.best_position - particle.position)
            particle.velocity = self.w * particle.velocity + cognitive + social
            particle.move()

    def _save_history(self):
        self.history["best_fitness"].append(self.best_value)
        self.history["best_position"].append(self.best_position.copy())

    def _run_iter(self):
        self._move_particles()
        self._evaluate_particles()
        self._save_history()

    def run(self, verbose=True):
        # Inisialisasi progress bar
        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with PSO",
            ncols=100,
            disable=not verbose
        )

        for iteration in iterator:
            self._run_iter()

            # Update progress bar display
            iterator.set_postfix({"Best": f"{self.best_value:.6f}, Pos: {np.round(self.best_position, 4)}"})

            # Kriteria berhenti opsional
            if abs(self.best_value) < self.tolerance:
                iterator.set_postfix({"Status": "Converged"})
                break

        return self.best_position, self.best_value

def run_pso(func, min_params, max_params, population_size, max_iter=100, verbose=False):
    """
    Runner for Particle Swarm Optimization (PSO)
    """

    iae_model = ParticleSwarmOptimization(
        obj_function="iae",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_particles=population_size,
        max_iteration=max_iter,
    )

    ise_model = ParticleSwarmOptimization(
        obj_function="ise",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_particles=population_size,
        max_iteration=max_iter,
    )

    itae_model = ParticleSwarmOptimization(
        obj_function="itae",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_particles=population_size,
        max_iteration=max_iter,
    )

    itse_model = ParticleSwarmOptimization(
        obj_function="itse",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_particles=population_size,
        max_iteration=max_iter,
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
