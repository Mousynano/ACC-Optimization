import numpy as np
from random import uniform
from tqdm import tqdm
from core.utils import lerp

class GeneticAlgorithm:
    def __init__(self, fitness_function, min_params, max_params,
                 population_size=30, max_iter=100, mutation_rate=0.4, crossover_rate=0.7, maximize=True, progress_callback=None):  # default: maximize
        self.fitness_function = fitness_function
        self.min_params = np.array(min_params)
        self.max_params = np.array(max_params)
        self.dim = len(min_params)
        self.population_size = population_size
        self.max_iter = max_iter
        self.maximize = maximize

        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        # self.mutated_offsprings = mutated_offsprings

        # History tracking
        self.history = {"best_fitness": [], "best_position": []}
        self.progress_callback = progress_callback

    def __crossover(self, parent1, parent2):
        rand1, rand2 = uniform(0, 1), uniform(0, 1)
        child1, child2 = parent1.copy(), parent2.copy()
        if rand1 < self.crossover_rate and rand2 < self.crossover_rate:
            child1 = [parent2[0], parent1[1], parent2[2]]
            child2 = [parent1[0], parent2[1], parent2[2]]
        else:
            if self.crossover_rate > rand1:
                child1 = [parent2[0], parent2[1], parent1[2]]
                child2 = [parent2[0], parent1[1], parent2[2]]
            elif self.crossover_rate > rand2:
                child1 = [parent1[0], parent2[1], parent2[2]]
                child2 = [parent2[0], parent2[1], parent1[2]]
            else:
                child1, child2 = parent1.copy(), parent2.copy()

        return np.array(child1), np.array(child2)
    
    def __mutation(self, child1, child2):
        if self.mutation_rate > uniform(0, 1):
            child1 = np.array([
                lerp(child1[0], (uniform(0, 1) * 2 - 1), self.mutation_rate),
                lerp(child1[1], (uniform(0, 1) * 2 - 1), self.mutation_rate),
                lerp(child1[2], (uniform(0, 1) * 2 - 1), self.mutation_rate),
            ])
            child2 = np.array([
                lerp(child2[0], (uniform(0, 1) * 2 - 1), self.mutation_rate),
                lerp(child2[1], (uniform(0, 1) * 2 - 1), self.mutation_rate),
                lerp(child2[2], (uniform(0, 1) * 2 - 1), self.mutation_rate),
            ])
        return child1, child2

    def run(self, verbose=True, progress_callback=None):
        """
        Runs GA for max_iter iterations.
        progress_callback(step:int) is called on every iteration.
        """

        population = np.array([
            np.array([uniform(self.min_params[i], self.max_params[i]) for i in range(self.dim)])
            for _ in range(self.population_size)
        ])

        iterator = tqdm(range(self.max_iter), desc="Optimizing with GA", disable=not verbose) \
                if verbose else range(self.max_iter)

        for t in iterator:

            # Compute fitness
            fitness = np.array([
                self.fitness_function(ind)  # <─ IMPORTANT: objective already embedded in wrapper
                for ind in population
            ])

            sorted_idx = np.argsort(fitness)[::-1] if self.maximize else np.argsort(fitness)

            best_idx = sorted_idx[0]
            second_best_idx = sorted_idx[1]

            best_params = population[best_idx]
            best_fit = fitness[best_idx]

            # Crossover & mutation
            child1, child2 = self.__crossover(best_params, population[second_best_idx])
            child1, child2 = self.__mutation(child1, child2)

            # Replace worst individuals
            population[sorted_idx[-1]] = child1
            population[sorted_idx[-2]] = child2

            # Save history
            self.history["best_fitness"].append(best_fit)
            self.history["best_position"].append(best_params.copy())

            # Progress callback
            if self.progress_callback is not None:
                self.progress_callback(t + 1)

        return best_params, best_fit

    
def run_ga(func, min_params, max_params, population_size, max_iter=100, verbose=False, progress_callback=None):
    model = GeneticAlgorithm(
        fitness_function=func,    # <─ only one objective
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        maximize=False,
        progress_callback=progress_callback
    )

    best_params, best_fit = model.run(
        verbose=verbose,
    )

    return (
        best_params,
        best_fit,
        model.history["best_fitness"]
    )