import numpy as np
from random import uniform
from tqdm import tqdm
import matplotlib.pyplot as plt
from core.utils import lerp

class GeneticAlgorithm:
    def __init__(self, fitness_function, obj_function, min_params, max_params,
                 population_size=30, max_iter=100, mutation_rate=0.4, crossover_rate=0.7, maximize=True):  # default: maximize
        self.fitness_function = fitness_function
        self.obj_function = obj_function
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

    def run(self, verbose=True):
        population = np.array([np.array([uniform(self.min_params[i], self.max_params[i]) for i in range(self.dim)]) for _ in range(self.population_size)])

        iterator = tqdm(range(self.max_iter), desc="Optimizing with GA", ncols=100, disable=not verbose)

        for t in iterator:
            fitness = np.array([self.fitness_function(ind, self.obj_function) for ind in population])
            sorted_indices = np.argsort(fitness)[::-1] if self.maximize else np.argsort(fitness)

            # Elitism 
            best_idx = sorted_indices[0]
            second_best_idx = sorted_indices[1]
            best_params = population[best_idx]
            second_best_params = population[second_best_idx]
            
            # Crossover & Mutation
            child1, child2 = self.__crossover(best_params, second_best_params)
            child1, child2 = self.__mutation(child1, child2)

            # Update population
            worst_idx = sorted_indices[-1]
            second_worst_idx = sorted_indices[-2]
            population[worst_idx] = child1
            population[second_worst_idx] = child2

            # Track best fitness and params
            best_fitness = fitness[best_idx]
            self.history["best_fitness"].append(best_fitness)
            self.history["best_position"].append(best_params)
        
        return best_params, best_fitness
    
def run_ga(func, min_params, max_params, population_size, max_iter=100, verbose=False):
    iae_model = GeneticAlgorithm(
        obj_function="iae",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        mutation_rate=0.3,
        crossover_rate=0.7,
        maximize=True
    )

    ise_model = GeneticAlgorithm(
        obj_function="ise",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        mutation_rate=0.3,
        crossover_rate=0.7,
        maximize=True
    )
    itae_model = GeneticAlgorithm(
        obj_function="itae",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        mutation_rate=0.3,
        crossover_rate=0.7,
        maximize=True
    )
    itse_model = GeneticAlgorithm(
        obj_function="itse",
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        mutation_rate=0.3,
        crossover_rate=0.7,
        maximize=True
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