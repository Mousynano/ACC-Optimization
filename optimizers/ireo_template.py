from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
import numpy as np  # tetap untuk printing dan return result
from tqdm import tqdm
import math
from reo import RippleEvolutionOptimizer


class IREOTemplate(RippleEvolutionOptimizer):
    """
    Template awal untuk IREO.
    Saat ini isinya REO + acceptance rule untuk menerima solusi yang sedikit lebih buruk
    di iterasi awal. Ini sengaja dibuat sebagai starting point, bukan final IREO paper.
    """

    def __init__(self, *args, acceptance_T0=1.0, acceptance_floor=1e-8, **kwargs):
        super().__init__(*args, **kwargs)
        self.acceptance_T0 = acceptance_T0
        self.acceptance_floor = acceptance_floor

    def _temperature(self, iteration):
        frac = 1.0 - (iteration / max(1, self.max_iteration))
        return max(self.acceptance_floor, self.acceptance_T0 * frac)

    def _accept_worse(self, current_value, trial_value, iteration):
        gap = float(to_cpu(trial_value - current_value))
        if self.maximize:
            gap = -gap
        T = self._temperature(iteration)
        prob = math.exp(-max(0.0, gap) / max(T, 1e-12))
        return np.random.rand() < prob

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

            if self._better(trial_value, values[i]) or self._accept_worse(values[i], trial_value, iteration):
                agent.move_to(trial)
                agent.value = trial_value
            agent.F = Fi
            agent.CR = CRi

    def run(self, verbose=True):
        iterator_name = "Optimizing with IREO-template"
        self._evaluate_population()
        self._save_history()

        iterator = tqdm(
            range(self.max_iteration),
            desc=iterator_name,
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


def run_ireo_template(func, min_params, max_params, population_size,
                      max_iter=100, verbose=False, progress_callback=None, seed=None):

    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = IREOTemplate(
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
