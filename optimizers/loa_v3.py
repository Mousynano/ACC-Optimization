from core.xp import xp, to_cpu, GPU_AVAILABLE
import math
from tqdm import tqdm


class LeleOptimizationAlgorithm:
    def __init__(self, fitness_function, min_params, max_params,
                 population_size=30, max_iter=100,
                 maximize=False, progress_callback=None):

        # Core setup
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.dim = len(min_params)

        self.pop_size = population_size
        self.max_iter = max_iter
        self.maximize = maximize
        self.progress_callback = progress_callback

        # --- Hyperparameters V2/V3 ---
        self.alpha_max = 0.9
        self.alpha_min = 0.1
        self.levy_beta = 1.5

        # --- V3 additions ---
        self.k_elite = max(2, self.pop_size // 10)

        self.temp0 = 1.0
        self.temp_min = 1e-3
        self.patience = 15
        self.stagnation = 0
        self.restart_frac = 0.2

        self.de_F = 0.5
        self.de_CR = 0.7
        self.repulsion_gamma_max = 0.2

        # State
        self.X = None
        self.fitness = None
        self.best_pos = None
        self.best_score = -xp.inf if maximize else xp.inf

        # Personal best memory
        self.pbest_pos = None
        self.pbest_score = None

        # History tracking
        self.history = {"best_fitness": [], "best_position": []}

    def _initialize_population(self):
        self.X = xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.pop_size, self.dim)
        )

        fitness_list = []
        for i in range(self.pop_size):
            fitness_list.append(self.fitness_function(to_cpu(self.X[i])))
        self.fitness = xp.array(fitness_list)

        # init pbest AFTER X and fitness exist
        self.pbest_pos = self.X.copy()
        self.pbest_score = self.fitness.copy()

        self._update_global_best()

    def _update_global_best(self):
        if self.maximize:
            best_idx = xp.argmax(self.fitness)
            if self.fitness[best_idx] > self.best_score:
                self.best_score = self.fitness[best_idx]
                self.best_pos = self.X[best_idx].copy()
        else:
            best_idx = xp.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_score:
                self.best_score = self.fitness[best_idx]
                self.best_pos = self.X[best_idx].copy()

    def _levy_flight(self, shape):
        beta = self.levy_beta
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2)
        sigma_u = (num / den) ** (1 / beta)

        u = xp.random.normal(0, sigma_u, size=shape)
        v = xp.random.normal(0, 1, size=shape)
        step = u / (xp.abs(v) ** (1 / beta))
        return step

    def run(self, verbose=True):
        self._initialize_population()

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with LOA v3",
            ncols=100,
            disable=not verbose
        )

        eps = 1e-12

        for t in iterator:
            progress = t / self.max_iter
            current_alpha = self.alpha_max - progress * (self.alpha_max - self.alpha_min)

            # Temperature schedule (SA-lite)
            T = max(self.temp_min, self.temp0 * (1.0 - progress))

            # Sorting: best first
            if self.maximize:
                sorted_idx = xp.argsort(-self.fitness)  # descending
            else:
                sorted_idx = xp.argsort(self.fitness)   # ascending

            # Hungry = worst half
            cut = int(0.5 * self.pop_size)
            hungry_ids = sorted_idx[cut:]
            hungry_mask = xp.zeros(self.pop_size, dtype=bool)
            hungry_mask[hungry_ids] = True
            hungry_mask_col = hungry_mask.reshape(-1, 1)

            # Multi-food elite
            elite_ids = sorted_idx[:self.k_elite]
            elites = self.X[elite_ids]
            elite_pick = xp.random.randint(0, self.k_elite, size=self.pop_size)
            food = elites[elite_pick]

            # Repulsion (anti-crowding)
            nbr_idx = xp.random.randint(0, self.pop_size, size=self.pop_size)
            X_nbr = self.X[nbr_idx]
            diff = self.X - X_nbr
            norm = xp.sqrt(xp.sum(diff**2, axis=1, keepdims=True)) + eps
            repulse_dir = diff / norm
            gamma = self.repulsion_gamma_max * (1.0 - progress)

            r1 = xp.random.random((self.pop_size, 1))
            r2 = xp.random.random((self.pop_size, 1))
            noise = xp.random.normal(0, 1, size=(self.pop_size, self.dim))

            # Hungry: exploit (food + pbest + repulsion + small noise)
            move_hungry = (
                self.X
                + current_alpha * r1 * (food - self.X)
                + 0.4 * current_alpha * r2 * (self.pbest_pos - self.X)
                + gamma * repulse_dir
                + (0.01 * (1.0 - progress)) * noise
            )

            # Satiated: explore via DE or Levy-around-food
            r1i = xp.random.randint(0, self.pop_size, size=self.pop_size)
            r2i = xp.random.randint(0, self.pop_size, size=self.pop_size)
            r3i = xp.random.randint(0, self.pop_size, size=self.pop_size)

            mutant = self.X[r1i] + self.de_F * (self.X[r2i] - self.X[r3i])
            cross = xp.random.random((self.pop_size, self.dim)) < self.de_CR
            de_trial = xp.where(cross, mutant, self.X)

            levy_step = self._levy_flight((self.pop_size, self.dim))
            levy_weight = 0.5 * (1.0 - progress)
            levy_trial = self.X + levy_weight * levy_step * (food - self.X)

            p_de = 0.6
            use_de = (xp.random.random(self.pop_size) < p_de).reshape(-1, 1)
            move_satiated = xp.where(use_de, de_trial, levy_trial) + gamma * repulse_dir

            # Combine
            X_new = xp.where(hungry_mask_col, move_hungry, move_satiated)

            # Reflect + clip bounds
            X_new = xp.where(X_new < self.min_params, 2 * self.min_params - X_new, X_new)
            X_new = xp.where(X_new > self.max_params, 2 * self.max_params - X_new, X_new)
            X_new = xp.clip(X_new, self.min_params, self.max_params)

            # Evaluate fitness (CPU calls)
            fitness_new_list = [self.fitness_function(to_cpu(X_new[i])) for i in range(self.pop_size)]
            fitness_new = xp.array(fitness_new_list)

            # # Acceptance
            # delta = fitness_new - self.fitness

            # if self.maximize:
            #     # better if delta > 0
            #     accept_satiated = (delta > 0) | (xp.random.random(self.pop_size) < xp.exp(delta / (T + eps)))
            #     accept = xp.where(hungry_mask, (delta > 0), accept_satiated)
            # else:
            #     # better if delta < 0
            #     accept_satiated = (delta < 0) | (xp.random.random(self.pop_size) < xp.exp(-delta / (T + eps)))
            #     accept = xp.where(hungry_mask, (delta < 0), accept_satiated)

            delta = fitness_new - self.fitness
            u = xp.random.random(self.pop_size)
            logu = xp.log(u + eps)  # log(u) selalu <= 0

            if not self.maximize:
                # minimization: accept worse (delta > 0) jika log(u) < -delta/T
                accept_satiated = (delta < 0) | ((delta > 0) & (logu < (-delta / (T + eps))))
                accept = xp.where(hungry_mask, (delta < 0), accept_satiated)
            else:
                # maximization: accept worse (delta < 0) jika log(u) < delta/T  (delta/T negatif)
                accept_satiated = (delta > 0) | ((delta < 0) & (logu < (delta / (T + eps))))
                accept = xp.where(hungry_mask, (delta > 0), accept_satiated)

            accept_col = accept.reshape(-1, 1)
            self.X = xp.where(accept_col, X_new, self.X)
            self.fitness = xp.where(accept, fitness_new, self.fitness)

            # Update pbest
            if self.maximize:
                better_pbest = self.fitness > self.pbest_score
            else:
                better_pbest = self.fitness < self.pbest_score

            better_col = better_pbest.reshape(-1, 1)
            self.pbest_pos = xp.where(better_col, self.X, self.pbest_pos)
            self.pbest_score = xp.where(better_pbest, self.fitness, self.pbest_score)

            # Update global best + stagnation
            prev_best = float(to_cpu(self.best_score))
            self._update_global_best()
            new_best = float(to_cpu(self.best_score))

            improved = (new_best > prev_best) if self.maximize else (new_best < prev_best)
            self.stagnation = 0 if improved else (self.stagnation + 1)

            # Partial restart if stagnating
            if self.stagnation >= self.patience:
                n_reset = max(1, int(self.restart_frac * self.pop_size))
                worst_ids = sorted_idx[-n_reset:]

                randX = xp.random.uniform(self.min_params, self.max_params, size=(n_reset, self.dim))
                oppX = self.min_params + self.max_params - randX

                # Evaluate both and pick better
                rand_fit = xp.array([self.fitness_function(to_cpu(randX[i])) for i in range(n_reset)])
                opp_fit = xp.array([self.fitness_function(to_cpu(oppX[i])) for i in range(n_reset)])

                if self.maximize:
                    choose_opp = opp_fit > rand_fit
                else:
                    choose_opp = opp_fit < rand_fit

                choose_opp_col = choose_opp.reshape(-1, 1)
                newX = xp.where(choose_opp_col, oppX, randX)
                newFit = xp.where(choose_opp, opp_fit, rand_fit)

                self.X[worst_ids] = newX
                self.fitness[worst_ids] = newFit
                self.pbest_pos[worst_ids] = newX
                self.pbest_score[worst_ids] = newFit

                self.stagnation = 0
                self._update_global_best()

            # History
            self.history["best_fitness"].append(float(to_cpu(self.best_score)))
            self.history["best_position"].append(to_cpu(self.best_pos.copy()))

            if self.progress_callback:
                self.progress_callback(t + 1)

            iterator.set_postfix({"Best": f"{float(to_cpu(self.best_score)):.6f}"})

        return to_cpu(self.best_pos), float(to_cpu(self.best_score))


def run_loa_v3(func, min_params, max_params, population_size,
               max_iter=100, verbose=False, progress_callback=None, seed=None):

    # Optional seeding
    if seed is not None:
        try:
            xp.random.seed(seed)
        except Exception:
            pass

    model = LeleOptimizationAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        maximize=False,
        progress_callback=progress_callback
    )

    best_params, best_fitness = model.run(verbose=verbose)
    return best_params, best_fitness, model.history["best_fitness"]
