from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # tetap untuk printing dan return result
from tqdm import tqdm
import math


class RippleAgent:
    def __init__(self, min_params, max_params, dimensions=3):
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.position = self.initialize(dimensions)
        self.value = xp.inf
        self.F = 0.5
        self.CR = 0.9

    def initialize(self, x):
        return xp.array([
            self.min_params[i] + (self.max_params[i] - self.min_params[i]) * np.random.rand()
            for i in range(x)
        ])

    def move_to(self, new_position):
        self.position = xp.clip(new_position, self.min_params, self.max_params)


class RippleEvolutionOptimizer:
    def __init__(self, fitness_function, min_params, max_params, maximize=False,
                 n_agents=100, max_iteration=300,
                 eta0=0.6, tau0=0.6, A0=0.2, delta=0.995,
                 omega=math.pi, sigma=0.05,
                 p=0.1, rho=0.2,
                 tau_F=0.1, tau_CR=0.1,
                 F_min=0.1, F_max=0.9,
                 drift_p0=0.2, levy_alpha=1.5, levy_scale=0.01,
                 progress_callback=None):

        self.fitness_function = fitness_function
        self.n_agents = n_agents
        self.max_iteration = max_iteration
        self.dimensions = len(min_params)

        self.agents = [
            RippleAgent(min_params, max_params, self.dimensions)
            for _ in range(n_agents)
        ]

        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.span = self.max_params - self.min_params

        self.maximize = maximize
        self.best_value = -xp.inf if maximize else xp.inf
        self.best_position = xp.zeros(self.dimensions)

        self.eta0 = eta0
        self.tau0 = tau0
        self.A0 = A0
        self.delta = delta
        self.omega = omega
        self.sigma = sigma
        self.p = p
        self.rho = rho
        self.tau_F = tau_F
        self.tau_CR = tau_CR
        self.F_min = F_min
        self.F_max = F_max
        self.drift_p0 = drift_p0
        self.levy_alpha = levy_alpha
        self.levy_scale = levy_scale

        self.history = {
            "best_fitness": [],
            "best_position": [],
        }

        self.progress_callback = progress_callback

    def _better(self, a, b):
        return a > b if self.maximize else a < b

    def _fitness(self, position):
        value_cpu = self.fitness_function(to_cpu(position))
        return xp.asarray(value_cpu)

    def _evaluate_population(self):
        for agent in self.agents:
            agent.value = self._fitness(agent.position)
            if self._better(agent.value, self.best_value):
                self.best_value = agent.value
                self.best_position = agent.position.copy()

    def _sorted_indices(self, values):
        order = np.argsort([float(to_cpu(v)) for v in values])
        if self.maximize:
            order = order[::-1]
        return order.tolist()

    def _levy(self):
        beta = self.levy_alpha
        sigma_u = (
            math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
            / (math.gamma((1 + beta) / 2) * beta * (2 ** ((beta - 1) / 2)))
        ) ** (1 / beta)
        u = np.random.normal(0, sigma_u, self.dimensions)
        v = np.random.normal(0, 1, self.dimensions)
        step = u / (np.abs(v) ** (1 / beta))
        return xp.asarray(step)

    def _reflect_bounds(self, position):
        pos = position.copy()
        for _ in range(2):
            pos = xp.where(pos < self.min_params, 2 * self.min_params - pos, pos)
            pos = xp.where(pos > self.max_params, 2 * self.max_params - pos, pos)
        return xp.clip(pos, self.min_params, self.max_params)

    def _swell(self, iteration):
        A_t = self.A0 * (self.delta ** iteration)
        phi = xp.asarray(np.random.uniform(0, 2 * math.pi, self.dimensions))
        return (A_t * self.sigma) * xp.sin(self.omega * (iteration / max(1, self.max_iteration)) + phi) * self.span

    def _run_iter(self, iteration):
        positions = [agent.position.copy() for agent in self.agents]
        values = [agent.value for agent in self.agents]
        F_values = [agent.F for agent in self.agents]
        CR_values = [agent.CR for agent in self.agents]

        sorted_idx = self._sorted_indices(values)
        rank_map = {idx: rank for rank, idx in enumerate(sorted_idx)}

        k = max(1, int(math.ceil(self.p * self.n_agents)))
        m = max(1, int(math.ceil(self.rho * self.n_agents)))
        elite_idx = sorted_idx[:m]
        pbest_pool = sorted_idx[:k]
        elite_mean = xp.mean(xp.stack([positions[idx] for idx in elite_idx], axis=0), axis=0)
        global_best = positions[sorted_idx[0]].copy()
        tide = self.tau0 * (iteration / max(1, self.max_iteration))
        drift_prob = self.drift_p0 * (1 - (iteration / max(1, self.max_iteration)))

        for i, agent in enumerate(self.agents):
            Fi = F_values[i]
            CRi = CR_values[i]

            if np.random.rand() < self.tau_F:
                Fi = np.random.uniform(self.F_min, self.F_max)
            if np.random.rand() < self.tau_CR:
                CRi = np.random.uniform(0.0, 1.0)

            rankshare = rank_map[i] / max(1, self.n_agents - 1)
            undertow = self.eta0 * (1 - rankshare)
            swell = self._swell(iteration)

            pbest_idx = int(np.random.choice(pbest_pool))
            candidates = [idx for idx in range(self.n_agents) if idx not in {i, pbest_idx}]
            if len(candidates) < 2:
                candidates = [idx for idx in range(self.n_agents) if idx != i]
            r1_idx, r2_idx = np.random.choice(candidates, 2, replace=False)

            current = positions[i]
            mutant = (
                current
                + Fi * (positions[pbest_idx] - current)
                + Fi * (positions[r1_idx] - positions[r2_idx])
                + undertow * (global_best - current)
                + tide * (elite_mean - current)
                + swell
            )

            trial = current.copy()
            jrand = np.random.randint(0, self.dimensions)
            for j in range(self.dimensions):
                if np.random.rand() < CRi or j == jrand:
                    trial[j] = mutant[j]

            if np.random.rand() < drift_prob:
                trial = trial + self.levy_scale * self._levy() * self.span

            trial = self._reflect_bounds(trial)
            trial_value = self._fitness(trial)

            if self._better(trial_value, values[i]):
                agent.move_to(trial)
                agent.value = trial_value
            agent.F = Fi
            agent.CR = CRi

    def _save_history(self):
        self.history["best_fitness"].append(float(to_cpu(self.best_value)))
        self.history["best_position"].append(to_cpu(self.best_position.copy()))

    def run(self, verbose=True):
        self._evaluate_population()
        self._save_history()

        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with REO",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            self._run_iter(iteration)
            self._evaluate_population()
            self._save_history()

            iterator.set_postfix({
                "Best": f"{float(to_cpu(self.best_value)):.6f}"
            })

            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return to_cpu(self.best_position), float(to_cpu(self.best_value))


def run_reo(func, min_params, max_params, population_size,
            max_iter=100, verbose=False, progress_callback=None, seed=None):

    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = RippleEvolutionOptimizer(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_agents=population_size,
        max_iteration=max_iter,
        maximize=False,
        progress_callback=progress_callback,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
