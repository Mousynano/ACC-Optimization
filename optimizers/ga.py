from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np
from random import uniform
from tqdm import tqdm
from core.utils import lerp


class GeneticAlgorithm:
    def __init__(self, fitness_function, min_params, max_params,
                 population_size=100, max_iter=300,
                 mutation_rate=0.4, crossover_rate=0.7,
                 maximize=True, progress_callback=None):

        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.dim = len(min_params)

        self.population_size = population_size
        self.max_iter = max_iter
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.maximize = maximize

        self.history = {"best_fitness": [], "best_position": []}
        self.progress_callback = progress_callback

    # ----------------------------------------------------
    # GPU-friendly population initialization
    # ----------------------------------------------------
    def _initialize_population(self):
        pop = xp.zeros((self.population_size, self.dim))
        for i in range(self.population_size):
            pop[i] = xp.array([
                self.min_params[d] + (self.max_params[d] - self.min_params[d]) * uniform(0, 1)
                for d in range(self.dim)
            ])
        return pop

    # ----------------------------------------------------
    # Crossover
    # ----------------------------------------------------
    def _crossover(self, p1, p2):
        r1, r2 = uniform(0, 1), uniform(0, 1)

        c1, c2 = p1.copy(), p2.copy()

        if r1 < self.crossover_rate and r2 < self.crossover_rate:
            c1 = xp.array([p2[0], p1[1], p2[2]])
            c2 = xp.array([p1[0], p2[1], p2[2]])
        else:
            if self.crossover_rate > r1:
                c1 = xp.array([p2[0], p2[1], p1[2]])
                c2 = xp.array([p2[0], p1[1], p2[2]])
            elif self.crossover_rate > r2:
                c1 = xp.array([p1[0], p2[1], p2[2]])
                c2 = xp.array([p2[0], p2[1], p1[2]])

        return c1, c2

    # ----------------------------------------------------
    # Mutation (GPU arrays but CPU lerp)
    # ----------------------------------------------------
    def _mutation(self, c1, c2):
        if uniform(0, 1) < self.mutation_rate:
            c1 = xp.array([
                lerp(float(c1[0]), (uniform(0, 1) * 2 - 1), self.mutation_rate),
                lerp(float(c1[1]), (uniform(0, 1) * 2 - 1), self.mutation_rate),
                lerp(float(c1[2]), (uniform(0, 1) * 2 - 1), self.mutation_rate)
            ])

            c2 = xp.array([
                lerp(float(c2[0]), (uniform(0, 1) * 2 - 1), self.mutation_rate),
                lerp(float(c2[1]), (uniform(0, 1) * 2 - 1), self.mutation_rate),
                lerp(float(c2[2]), (uniform(0, 1) * 2 - 1), self.mutation_rate)
            ])

        return c1, c2

    # ----------------------------------------------------
    # Main GA loop
    # ----------------------------------------------------
    def run(self, verbose=True):
        population = self._initialize_population()

        iterator = tqdm(range(self.max_iter), desc="Optimizing with GA",
                        ncols=100, disable=not verbose)

        for t in iterator:

            # Evaluate fitness (CPU)
            fitness = xp.array([
                self.fitness_function(to_cpu(ind))
                for ind in population
            ])

            # GPU sorting
            sorted_idx = xp.argsort(fitness)
            if self.maximize:
                sorted_idx = sorted_idx[::-1]

            best_idx = int(sorted_idx[0])
            second_idx = int(sorted_idx[1])

            best_params = population[best_idx].copy()
            best_fit = float(to_cpu(fitness[best_idx]))

            # Crossover & mutation
            child1, child2 = self._crossover(best_params, population[second_idx])
            child1, child2 = self._mutation(child1, child2)

            # Replace worst individuals
            population[int(sorted_idx[-1])] = child1
            population[int(sorted_idx[-2])] = child2

            # Save history on CPU
            self.history["best_fitness"].append(best_fit)
            self.history["best_position"].append(to_cpu(best_params.copy()))

            if self.progress_callback is not None:
                self.progress_callback(t + 1)

            iterator.set_postfix({"Best": f"{best_fit:.6f}"})

        return to_cpu(best_params), best_fit


def run_ga(func, min_params, max_params, population_size, max_iter=100,
           verbose=False, progress_callback=None, seed=None):
    
    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = GeneticAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        maximize=False,
        progress_callback=progress_callback
    )

    best_params, best_fit = model.run(verbose=verbose)

    return best_params, best_fit, model.history["best_fitness"]
