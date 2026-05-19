from core.xp import xp, to_cpu, GPU_AVAILABLE
import numpy as np
import math
from tqdm import tqdm

class LeleOptimizationAlgorithm:
    def __init__(self, fitness_function, min_params, max_params,
                 population_size=30, max_iter=100,
                 maximize=False, progress_callback=None):

        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.dim = len(min_params)
        
        self.pop_size = population_size
        self.max_iter = max_iter
        self.maximize = maximize

        # --- Hyperparameters V2 (Adaptive) ---
        self.alpha_max = 0.9  # Kecepatan awal (agresif)
        self.alpha_min = 0.1  # Kecepatan akhir (presisi)
        self.beta_min = 0.2   # Exploration factor min
        self.levy_beta = 1.5  # Levy exponent

        # History tracking
        self.history = {
            "best_fitness": [],
            "best_position": []
        }
        self.progress_callback = progress_callback

        # State
        self.X = None
        self.fitness = None
        self.best_pos = None
        self.best_score = -xp.inf if maximize else xp.inf

    def _initialize_population(self):
        self.X = xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.pop_size, self.dim)
        )
        
        # Hitung fitness awal loop CPU
        fitness_list = []
        for i in range(self.pop_size):
            val_cpu = self.fitness_function(to_cpu(self.X[i]))
            fitness_list.append(val_cpu)
        self.fitness = xp.array(fitness_list)
        
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
        # Sigma scalar calculation (CPU math)
        num = math.gamma(1 + beta) * math.sin(math.pi * beta / 2)
        den = math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)
        sigma_u = (num / den)**(1 / beta)
        
        u = xp.random.normal(0, sigma_u, size=shape)
        v = xp.random.normal(0, 1, size=shape)
        step = u / (xp.abs(v)**(1 / beta))
        return step

    def run(self, verbose=True):
        self._initialize_population()
        
        iterator = tqdm(range(self.max_iter), desc="Optimizing with LOA v2", ncols=100, disable=not verbose)

        for t in iterator:
            # --- 1. Adaptive Parameters (Time-Varying) ---
            # Alpha mengecil seiring waktu (Linear Decay) agar konvergensi halus
            # Rumus: alpha_max - (t/max_iter) * (alpha_max - alpha_min)
            progress = t / self.max_iter
            current_alpha = self.alpha_max - progress * (self.alpha_max - self.alpha_min)
            
            # Faktor eksplorasi juga mengecil
            current_explore = 1.0 - progress 
            
            avg_fitness = xp.mean(self.fitness)
            
            # --- 2. Determine State (Hungry vs Satiated) ---
            if self.maximize:
                is_hungry = self.fitness < avg_fitness
            else:
                is_hungry = self.fitness > avg_fitness # Minimization: Fitness besar = Lapar
            
            hungry_mask = is_hungry.reshape(-1, 1)

            # Generate Random Numbers untuk Stochastic movement
            r1 = xp.random.random((self.pop_size, 1))
            r2 = xp.random.random((self.pop_size, 1)) # Faktor noise tambahan

            # --- 3. FASE LAPAR (Exploitation + Local Search) ---
            # Dulu: X + 0.8 * (Best - X)  <- Kaku
            # Sekarang: X + adaptive_alpha * (Best - X) + wiggle
            # Kita tambahkan sedikit "Interaction" ke random neighbor agar tidak numpuk di satu titik
            random_indices = xp.random.randint(0, self.pop_size, self.pop_size)
            X_neighbor = self.X[random_indices]
            
            diff_to_food = self.best_pos - self.X
            diff_to_neighbor = X_neighbor - self.X
            
            # Gerakan utama ke Food, gerakan sekunder ke Neighbor (Social Learning)
            move_hungry = self.X + \
                          current_alpha * r1 * diff_to_food + \
                          (0.1 * current_alpha) * r2 * diff_to_neighbor

            # --- 4. FASE KENYANG (Exploration / Levy) ---
            # Dulu: X + 0.5 * Levy * (X - Best) <- Terlalu liar di akhir
            # Sekarang: X + (decaying_factor * Levy) * (X - Best)
            levy_step = self._levy_flight((self.pop_size, self.dim))
            
            # Levy weight mengecil drastis mendekati akhir iterasi
            levy_weight = 0.5 * current_explore 
            
            move_satiated = self.X + levy_weight * levy_step * (self.X - self.best_pos)
            
            # Gabungkan posisi
            X_new = xp.where(hungry_mask, move_hungry, move_satiated)
            
            # Boundary & Greedy Selection
            X_new = xp.clip(X_new, self.min_params, self.max_params)
            
            fitness_new_list = []
            for i in range(self.pop_size):
                fitness_new_list.append(self.fitness_function(to_cpu(X_new[i])))
            fitness_new = xp.array(fitness_new_list)
            
            if self.maximize:
                update_mask = fitness_new > self.fitness
            else:
                update_mask = fitness_new < self.fitness
                
            update_mask_col = update_mask.reshape(-1, 1)
            self.X = xp.where(update_mask_col, X_new, self.X)
            self.fitness = xp.where(update_mask, fitness_new, self.fitness)
            
            self._update_global_best()
            
            # History recording
            self.history["best_fitness"].append(float(to_cpu(self.best_score)))
            self.history["best_position"].append(to_cpu(self.best_pos.copy()))
            
            if self.progress_callback:
                self.progress_callback(t + 1)
            
            iterator.set_postfix({"Best": f"{float(to_cpu(self.best_score)):.6f}"})

        return to_cpu(self.best_pos), float(to_cpu(self.best_score))

def run_loa_v2(func, min_params, max_params, population_size,
            max_iter=100, verbose=False, progress_callback=None, seed=None):
    
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