from core.xp import xp, to_cpu, GPU_AVAILABLE
from tqdm import tqdm
from core.seeding import seed_everything

class TunaSwarmOptimization:
    def __init__(self, fitness_function, min_params, max_params,
                 pop_size=100, max_iter=300, a=0.7,z=0.05, maximize=True, progress_callback=None):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.dim = len(min_params)
        self.pop_size = pop_size
        self.maximize = maximize
        self.t_max = max_iter

        self.a = a
        self.z = z

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }
        self.progress_callback = progress_callback

    def __initialize_position(self):
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.pop_size, self.dim)
        )
    
    def __initialize_iter(self, tunas):
        fitness = xp.array([
            self.__evaluate(tunas[i])
            for i in range(self.pop_size)
        ])

        tunas_old, fitness_old = self.__sort_by_fitness(tunas, fitness)
        return tunas_old, fitness_old
    
    def __prepare_params(self, t):
        b = xp.random.uniform(0, 1)

        a1 = self.a + (1 - self.a) * t / self.t_max
        a2 = (1 - self.a) - (1 - self.a) * t / self.t_max

        # bagian `l` perlu di verify pemdasnya
        l = xp.exp(3 * xp.cos(((self.t_max + 1 / t) - 1) * xp.pi))

        beta = xp.exp(b * l) * xp.cos(2 * xp.pi * b)

        p = xp.power(1 - t / self.t_max, t / self.t_max)

        return a1, a2, beta, p

    def __random_position(self):
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=self.dim
        )

    def __random_tuna(self, population):
        idx = xp.random.randint(self.pop_size)
        return population[idx]

    def __spiral_foraging(self, a1, a2, beta, tuna, best_tuna=None, random_tuna=None, previous_tuna=None):
        reference = best_tuna if best_tuna is not None else random_tuna
        res = a1 * (reference + beta * xp.abs(reference - tuna))
        if previous_tuna is not None: return res + a2 * previous_tuna
        else: return res + a2 * tuna

    def __parabolic_foraging(self, tuna, best_tuna, p):
        rand = xp.random.uniform(0, 1)
        tf = xp.random.uniform(0, 1)
        if rand < 0.5: return best_tuna + rand * (best_tuna - tuna) + tf * xp.power(p, 2) * (best_tuna - tuna)
        else: return tf * xp.power(p, 2) * tuna
    
    def __evaluate(self, position):
        return float(self.fitness_function(to_cpu(position)))
    
    def __sort_by_fitness(self, tunas, f_values):
        sort_indices = xp.argsort(f_values)
        if self.maximize:
            sort_indices = sort_indices[::-1]

        return tunas[sort_indices], f_values[sort_indices]

    def run(self, verbose=True):
        tunas_old = self.__initialize_position()
        fitness = None

        iterator = tqdm(
            range(self.t_max),
            desc="Optimizing with TSO",
            ncols=100,
            disable=not verbose
        )

        for t in iterator:
            tunas_old, fitness = self.__initialize_iter(tunas_old)
            tunas_new = xp.array(tunas_old.copy())

            best_tuna = tunas_old[0]
            f_best = fitness[0]

            a1, a2, beta, p = self.__prepare_params(t + 1)

            for j in range(self.pop_size):
                if xp.random.uniform(0, 1) < self.z:
                    tunas_new[j] = self.__random_position()
                else:
                    if (xp.random.uniform(0, 1) < 0.5):
                        if ((t + 1) / self.t_max < xp.random.uniform(0, 1)):
                            random_tuna = self.__random_tuna(tunas_old)          
                            tunas_new[j] = self.__spiral_foraging(a1, a2, beta, tunas_old[j], None, random_tuna, tunas_old[j-1] if j > 0 else None)
                        else:
                            tunas_new[j] = self.__spiral_foraging(a1, a2, beta, tunas_old[j], best_tuna, None, tunas_old[j-1] if j > 0 else None)
                    else:
                        tunas_new[j] = self.__parabolic_foraging(tunas_old[j], best_tuna, p)
                tunas_new[j] = xp.clip(tunas_new[j], self.min_params, self.max_params)

            fitness = xp.array([
                self.__evaluate(tunas_new[i])
                for i in range(self.pop_size)
            ])
            tunas_old, fitness = self.__sort_by_fitness(tunas_new, fitness)

            self.history["best_fitness"].append(fitness[0])
            self.history["best_position"].append(to_cpu(tunas_old[0]))

            if self.progress_callback is not None:
                self.progress_callback(t + 1)

            iterator.set_postfix({"Best": f"{fitness[0]}"})

        return to_cpu(tunas_old[0]), fitness[0]

def run_tso(func,
            min_params,
            max_params,
            population_size,
            max_iter=100,
            verbose=False,
            progress_callback=None,
            seed=None):
    model = TunaSwarmOptimization(
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