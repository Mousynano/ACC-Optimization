from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # tetap untuk printing dan return result
from random import random
from tqdm import tqdm


class Particle:
    def __init__(self, min_params, max_params, maximize, dimensions=3):
        # simpan batas dalam backend xp
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)

        # posisi dan velocity GPU/CPU
        self.position = self.initialize(dimensions)
        self.velocity = xp.zeros(dimensions)

        # best pos/value
        self.best_position = self.position.copy()
        self.best_value = -xp.inf if maximize else xp.inf

    def initialize(self, x):
        return xp.array([
            self.min_params[i] + (self.max_params[i] - self.min_params[i]) * random()
            for i in range(x)
        ])

    def move(self):
        self.position = xp.clip(
            self.position + self.velocity,
            self.min_params,
            self.max_params
        )


class ParticleSwarmOptimization:
    def __init__(self, fitness_function, min_params, max_params, maximize=False,
                 n_particles=100, max_iteration=300,
                 w=0.3, c1=2.0, c2=2.0, tolerance=1e-6, progress_callback=None):

        self.fitness_function = fitness_function
        self.n_particles = n_particles
        self.max_iteration = max_iteration

        # spawn particles in xp backend (CPU/GPU)
        self.particles = [
            Particle(min_params, max_params, maximize, len(min_params))
            for _ in range(n_particles)
        ]

        self.maximize = maximize
        self.best_value = -xp.inf if maximize else xp.inf
        self.best_position = xp.zeros(len(min_params))

        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.tolerance = tolerance

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

        self.progress_callback = progress_callback

    def _evaluate_particles(self):
        for p in self.particles:
            # convert position to CPU→fitness→XP safe path
            pos_cpu = to_cpu(p.position)
            value_cpu = self.fitness_function(pos_cpu)
            value = xp.asarray(value_cpu)  # ensure GPU-friendly scalar

            # personal best
            if (self.maximize and value > p.best_value) or \
               ((not self.maximize) and value < p.best_value):
                p.best_value = value
                p.best_position = p.position.copy()

            # global best
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
        # store CPU values for safety & plotting
        self.history["best_fitness"].append(float(to_cpu(self.best_value)))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def _run_iter(self):
        self._move_particles()
        self._evaluate_particles()
        self._save_history()

    def run(self, verbose=True):

        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with PSO",
            ncols=100,
            disable=not verbose
        )

        for iteration in iterator:
            self._run_iter()

            # CPU-safe display
            iterator.set_postfix({
                "Best": f"{float(to_cpu(self.best_value)):.6f}"
            })

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        # return CPU arrays
        return to_cpu(self.best_position), float(to_cpu(self.best_value))


def run_pso(func, min_params, max_params, population_size,
            max_iter=100, verbose=False, progress_callback=None, seed=None):
        
        # if seed is not None:
        #     seed_everything(seed, xp=xp)

        model = ParticleSwarmOptimization(
            fitness_function=func,
            min_params=min_params,
            max_params=max_params,
            n_particles=population_size,
            max_iteration=max_iter,
            maximize=False,
            progress_callback=progress_callback
        )

        best_params, best_fitness = model.run(verbose=verbose)

        return best_params, best_fitness, model.history["best_fitness"]

