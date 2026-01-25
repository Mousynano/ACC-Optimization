from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything
from tqdm import tqdm

class HikingOptimizationAlgorithm:
    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        hikers=100,
        max_iter=300,
        maximize=True,
        progress_callback=None
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.array(min_params)
        self.max_params = xp.array(max_params)
        self.dim = len(min_params)
        self.hikers = hikers
        self.max_iter = max_iter
        self.maximize = maximize

        # History tracking (di-CPU biar gampang dipakai/di-plot)
        self.history = {"best_fitness": [], "best_position": []}
        self.progress_callback = progress_callback

    # Tobler’s Hiking Function (Eq. 1) – sudah vectorized
    def toblers_velocity(self, slope):
        # slope: xp.array shape (hikers,)
        return 6 * xp.exp(-3.5 * xp.abs(slope + 0.05))

    # Compute slope (Eq. 2) – vectorized
    def slope(self, theta_deg):
        # theta_deg: xp.array shape (hikers,)
        return xp.tan(xp.deg2rad(theta_deg))

    # Initialize positions (Eq. 5) – vectorized
    def initialize_positions(self):
        # result: (hikers, dim)
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.hikers, self.dim),
        )

    def evaluate(self, pos_xp):
        """
        Hitung fitness untuk satu posisi.
        pos_xp: xp.array (dim,)
        """
        pos_cpu = to_cpu(pos_xp)
        return float(self.fitness_function(pos_cpu))

    def __prepare_params(self, beta_best):
        theta = xp.random.uniform(0.0, 50.0, size=self.hikers)       # (hikers,)
        slope_i = self.slope(theta)                                  # (hikers,)
        Wi_prev = self.toblers_velocity(slope_i)                     # (hikers,)

        # α_i ∈ [1, 3], γ_i ∈ [0, 1]  --> dibentuk (hikers, 1) biar broadcast ke dim
        alpha_i = xp.random.uniform(1.0, 3.0, size=(self.hikers, 1))  # (hikers,1)
        gamma_i = xp.random.uniform(0.0, 1.0, size=(self.hikers, 1))  # (hikers,1)

        # Bentuk Wi_prev ke (hikers,1) biar gampang di-broadcast ke dim
        Wi_prev_col = Wi_prev.reshape(-1, 1)                          # (hikers,1)

        # Ambil beta_best sekarang (dim,) → (1, dim) buat broadcast
        beta_best_row = beta_best.reshape(1, -1)                      # (1,dim)

        return alpha_i, gamma_i, Wi_prev_col, beta_best_row

    def __update_position_and_velocity(self, alpha_i, gamma_i, Wi_prev_col, beta_best_row, beta):
        direction = beta_best_row - alpha_i * beta                    # (hikers, dim)
        Wi = Wi_prev_col + gamma_i * direction                        # (hikers, dim)

        # β_new = β_i + Wi
        beta_new = beta + Wi                                          # (hikers, dim)

        # Bound control
        beta_new = xp.clip(beta_new, self.min_params, self.max_params)

        return beta_new

    def __evaluate_population(self, beta, beta_new, fitness):
        fitness_new_list = [
            self.evaluate(beta_new[i])
            for i in range(self.hikers)
        ]
        fitness_new = xp.array(fitness_new_list)

        # Pilih solusi yang lebih baik per-hiker
        if self.maximize:
            better_mask = fitness_new > fitness   # (hikers,)
        else:
            better_mask = fitness_new < fitness   # (hikers,)

        # Update per-hiker (vectorized pakai xp.where)
        better_mask_col = better_mask.reshape(-1, 1)  # (hikers,1) buat posisi
        beta = xp.where(better_mask_col, beta_new, beta)
        fitness = xp.where(better_mask, fitness_new, fitness)

        return beta, fitness

    def __update_global_best(self, beta, fitness, best_idx, f_best, beta_best):
        if self.maximize:
            best_idx = int(fitness.argmax())
            if fitness[best_idx] > f_best:
                f_best = float(fitness[best_idx])
                beta_best = beta[best_idx].copy()
        else:
            best_idx = int(fitness.argmin())
            if fitness[best_idx] < f_best:
                f_best = float(fitness[best_idx])
                beta_best = beta[best_idx].copy()

        return best_idx, f_best, beta_best

    def run(self, verbose=True):
        # --- Inisialisasi posisi & fitness awal ------------------------------
        beta = self.initialize_positions()  # (hikers, dim) di GPU/CPU (xp)

        # Fitness awal (loop per hiker masih perlu karena fitness ACC di CPU)
        fitness = xp.array([
            self.evaluate(beta[i])
            for i in range(self.hikers)
        ])

        # Pilih best awal
        if self.maximize:
            best_idx = int(fitness.argmax())
        else:
            best_idx = int(fitness.argmin())

        beta_best = beta[best_idx].copy()
        f_best = float(fitness[best_idx])

        iterator = tqdm(
            range(self.max_iter),
            desc="Optimizing with HOA",
            ncols=100,
            disable=not verbose,
        )

        for t in iterator:
            # ================================================================
            # 1) Generate random parameter untuk SEMUA hikers (vectorized)
            # ================================================================
            # θ_i ∈ [0, 50°] untuk semua hikers
            alpha_i, gamma_i, Wi_prev_col, beta_best_row = self.__prepare_params(beta_best)

            # ================================================================
            # 2) Update velocity & posisi SEKALIGUS (vectorized)
            #    Eq. (3) & Eq. (4)
            # ================================================================
            # Wi = Wi_prev + γ_i * (β_best − α_i * β_i)
            beta_new = self.__update_position_and_velocity(alpha_i, gamma_i, Wi_prev_col, beta_best_row, beta)

            # ================================================================
            # 3) Evaluasi fitness baru (CPU), lalu greedy selection (vectorized)
            # ================================================================
            beta, fitness = self.__evaluate_population(beta, beta_new, fitness)

            # ================================================================
            # 4) Update global best (β_best, f_best)
            # ================================================================
            best_idx, f_best, beta_best = self.__update_global_best(beta, fitness, best_idx, f_best, beta_best)

            # Simpan history (di CPU)
            self.history["best_fitness"].append(float(f_best))
            self.history["best_position"].append(to_cpu(beta_best.copy()))

            iterator.set_postfix({"Best": f"{f_best:.6f}"})

            # Progress callback (kalau dipakai multiprocess)
            if self.progress_callback is not None:
                self.progress_callback(t + 1)

        return to_cpu(beta_best), float(f_best)


def run_hoa(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=300,
    verbose=False,
    progress_callback=None,
    seed=None
):
    """
    Wrapper biar tetap kompatibel sama objective_worker kamu.
    """

    # if seed is not None:
    #     seed_everything(seed, xp=xp)

    model = HikingOptimizationAlgorithm(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        hikers=population_size,
        max_iter=max_iter,
        maximize=False,           # ACC pakai fungsi error → minimization
        progress_callback=progress_callback,
    )

    best_params, best_fitness = model.run(verbose=verbose)

    return (
        best_params,
        best_fitness,
        model.history["best_fitness"],
    )