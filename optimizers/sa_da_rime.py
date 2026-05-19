"""
SA-DA-RIME-ACC Draft Optimizer (Python)
----------------------------------------
Safety-Aware Diversity-Adaptive RIME for lightweight offline tuning of
FOPID Adaptive Cruise Control (ACC).

Design style follows the provided pso.py / hoa.py pattern:
- class-based optimizer
- xp/to_cpu backend support
- tqdm progress bar
- history dictionary for plotting
- wrapper function run_sa_da_rime(...)

This file is intentionally a research draft. The core algorithmic hooks are
explicit and easy to ablate:
1) chaotic opposition-based initialization
2) adaptive exploration-exploitation control
3) feasible elite archive
4) buoyancy-centering operator
5) partial restart / diversity repair
6) lightweight final-stage local search

Default mode is minimization, which matches ACC error/objective optimization.
"""

from __future__ import annotations

try:
    from core.xp import xp, to_cpu, GPU_AVAILABLE
except Exception:  # standalone fallback
    import numpy as xp
    GPU_AVAILABLE = False

    def to_cpu(arr):
        return arr.get() if hasattr(arr, "get") else arr

import math
import numpy as np
from tqdm import tqdm


class SafetyAwareDiversityAdaptiveRIME:
    """
    SA-DA-RIME optimizer draft.

    Parameters
    ----------
    fitness_function : callable
        Objective function. It receives a CPU numpy array and returns a float.
        For ACC, this should be the complete objective J to minimize.

    constraint_evaluator : callable, optional
        Optional safety/feasibility checker. It receives a CPU numpy array and
        returns one of:
            - None: treated as feasible
            - bool: True feasible, False infeasible
            - tuple/list: (feasible: bool, violation: float)
            - dict with optional keys:
                feasible / is_feasible : bool
                violation / safety_violation / penalty : float
                collision : bool
        If omitted, all candidates are treated as feasible.

    max_iter : int
        Iteration budget. Used when max_fes is None.

    max_fes : int, optional
        Function-evaluation budget. If provided, the optimizer stops when this
        objective-call budget is reached. This is recommended for fair
        metaheuristic benchmarking.
    """

    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        population_size=50,
        max_iter=300,
        max_fes=None,
        maximize=False,
        w=5,
        constraint_evaluator=None,
        archive_size=12,
        diversity_threshold=0.06,
        stagnation_patience=15,
        restart_fraction=0.20,
        buoyancy_fraction=0.35,
        safety_violation_threshold=0.25,
        use_chaotic_opposition=True,
        use_adaptive_schedule=True,
        use_archive=True,
        use_buoyancy=True,
        use_restart=True,
        use_local_search=True,
        local_search_start=0.80,
        local_search_top_k=3,
        local_search_steps=2,
        local_search_scale=0.025,
        strict_budget=True,
        progress_callback=None,
    ):
        self.fitness_function = fitness_function
        self.constraint_evaluator = constraint_evaluator

        self.min_params = xp.array(min_params, dtype=float)
        self.max_params = xp.array(max_params, dtype=float)
        self.range_params = self.max_params - self.min_params
        self.dim = len(min_params)

        self.population_size = int(population_size)
        self.max_iter = int(max_iter)
        self.max_fes = None if max_fes is None else int(max_fes)
        self.maximize = bool(maximize)
        self.w = w
        self.strict_budget = strict_budget
        self.progress_callback = progress_callback

        # SA-DA-RIME controls
        self.archive_size = int(archive_size)
        self.diversity_threshold = float(diversity_threshold)
        self.stagnation_patience = int(stagnation_patience)
        self.restart_fraction = float(restart_fraction)
        self.buoyancy_fraction = float(buoyancy_fraction)
        self.safety_violation_threshold = float(safety_violation_threshold)

        self.use_chaotic_opposition = bool(use_chaotic_opposition)
        self.use_adaptive_schedule = bool(use_adaptive_schedule)
        self.use_archive = bool(use_archive)
        self.use_buoyancy = bool(use_buoyancy)
        self.use_restart = bool(use_restart)
        self.use_local_search = bool(use_local_search)

        self.local_search_start = float(local_search_start)
        self.local_search_top_k = int(local_search_top_k)
        self.local_search_steps = int(local_search_steps)
        self.local_search_scale = float(local_search_scale)

        # State
        self.best_position = xp.zeros(self.dim)
        self.best_fitness = -xp.inf if maximize else xp.inf
        self.best_feasible = False
        self.best_violation = xp.inf
        self.fes = 0
        self.iteration = 0
        self.stagnation_counter = 0
        self._last_best_fitness = self.best_fitness

        # Feasible elite archive: list of dicts {"position", "fitness", "violation"}
        self.archive = []

        # History stored on CPU for plotting and serialization.
        self.history = {
            "best_fitness": [],
            "best_position": [],
            "diversity": [],
            "safety_violation_rate": [],
            "archive_size": [],
            "fes": [],
            "stagnation_counter": [],
        }

    # ------------------------------------------------------------------
    # Basic utilities
    # ------------------------------------------------------------------
    def _budget_available(self):
        if self.max_fes is None:
            return True
        return self.fes < self.max_fes

    def _progress(self):
        if self.max_fes is not None:
            return min(1.0, max(1, self.fes) / max(1, self.max_fes))
        return min(1.0, max(1, self.iteration) / max(1, self.max_iter))

    def _is_better_value(self, value, reference):
        return value > reference if self.maximize else value < reference

    def _is_better_candidate(self, fit_a, feas_a, viol_a, fit_b, feas_b, viol_b):
        """
        Feasibility-aware comparison.
        Feasible candidate wins against infeasible candidate. If both have the
        same feasibility status, use fitness. If both are infeasible and fitness
        is tied-ish, prefer lower violation.
        """
        if feas_a and not feas_b:
            return True
        if feas_b and not feas_a:
            return False

        if feas_a and feas_b:
            return self._is_better_value(fit_a, fit_b)

        # both infeasible: reduce violation first, then fitness
        if viol_a < viol_b - 1e-12:
            return True
        if viol_b < viol_a - 1e-12:
            return False
        return self._is_better_value(fit_a, fit_b)

    @staticmethod
    def _matlab_round_positive(x):
        return xp.floor(x + 0.5)

    def _to_float(self, value):
        return float(to_cpu(value))

    def _clip(self, position):
        return xp.clip(position, self.min_params, self.max_params)

    # ------------------------------------------------------------------
    # Evaluation and constraints
    # ------------------------------------------------------------------
    def evaluate(self, pos_xp):
        """Evaluate objective and count one function evaluation."""
        if self.strict_budget and not self._budget_available():
            return None

        pos_cpu = np.asarray(to_cpu(pos_xp), dtype=float)
        value = float(self.fitness_function(pos_cpu))
        self.fes += 1

        if self.progress_callback is not None:
            # Use FEs as callback signal when an FE budget exists; otherwise
            # iteration callback is also sent in run().
            if self.max_fes is not None:
                self.progress_callback(self.fes)

        return value

    def _parse_constraint_result(self, result):
        if result is None:
            return True, 0.0

        if isinstance(result, (bool, np.bool_)):
            return bool(result), 0.0 if bool(result) else 1.0

        if isinstance(result, (tuple, list)) and len(result) >= 2:
            feasible = bool(result[0])
            violation = float(result[1])
            if not feasible and violation <= 0.0:
                violation = 1.0
            return feasible, max(0.0, violation)

        if isinstance(result, dict):
            violation = result.get("safety_violation", None)
            if violation is None:
                violation = result.get("violation", None)
            if violation is None:
                violation = result.get("penalty", 0.0)
            violation = max(0.0, float(violation))

            collision = bool(result.get("collision", False))
            feasible = result.get("feasible", None)
            if feasible is None:
                feasible = result.get("is_feasible", None)
            if feasible is None:
                feasible = (violation <= 0.0) and (not collision)
            feasible = bool(feasible) and (not collision)

            if collision:
                violation = max(violation, 1.0)
            if not feasible and violation <= 0.0:
                violation = 1.0
            return feasible, violation

        # Numeric result is treated as total violation.
        try:
            violation = max(0.0, float(result))
            return violation <= 0.0, violation
        except Exception:
            return True, 0.0

    def check_constraints(self, pos_xp):
        if self.constraint_evaluator is None:
            return True, 0.0
        pos_cpu = np.asarray(to_cpu(pos_xp), dtype=float)
        result = self.constraint_evaluator(pos_cpu)
        return self._parse_constraint_result(result)

    def _evaluate_candidate(self, pos_xp):
        value = self.evaluate(pos_xp)
        if value is None:
            return None
        feasible, violation = self.check_constraints(pos_xp)
        return value, feasible, violation

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _chaotic_positions(self):
        """Logistic chaotic map initialization in the normalized [0, 1] space."""
        u = xp.random.random(size=(self.population_size, self.dim))
        # A few logistic-map steps improve dispersion without being expensive.
        for _ in range(3):
            u = 4.0 * u * (1.0 - u)
        return self.min_params + u * self.range_params

    def _random_positions(self):
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.population_size, self.dim),
        )

    def initialize_population(self):
        if self.use_chaotic_opposition:
            population = self._chaotic_positions()
            opposition = self.min_params + self.max_params - population
            opposition = self._clip(opposition)
        else:
            population = self._random_positions()
            opposition = None

        fitness = xp.full(self.population_size, xp.inf if not self.maximize else -xp.inf)
        feasible = xp.zeros(self.population_size, dtype=bool)
        violation = xp.ones(self.population_size) * xp.inf

        for i in range(self.population_size):
            if not self._budget_available() and self.strict_budget:
                break

            result_x = self._evaluate_candidate(population[i])
            if result_x is None:
                break
            fit_x, feas_x, viol_x = result_x

            chosen_pos = population[i].copy()
            chosen_fit, chosen_feas, chosen_viol = fit_x, feas_x, viol_x

            if opposition is not None and self._budget_available():
                result_opp = self._evaluate_candidate(opposition[i])
                if result_opp is not None:
                    fit_o, feas_o, viol_o = result_opp
                    if self._is_better_candidate(fit_o, feas_o, viol_o, chosen_fit, chosen_feas, chosen_viol):
                        chosen_pos = opposition[i].copy()
                        chosen_fit, chosen_feas, chosen_viol = fit_o, feas_o, viol_o

            population[i] = chosen_pos
            fitness[i] = chosen_fit
            feasible[i] = chosen_feas
            violation[i] = chosen_viol

            self._maybe_update_global_best(chosen_pos, chosen_fit, chosen_feas, chosen_viol)

        self._update_archive(population, fitness, feasible, violation)
        return population, fitness, feasible, violation

    # ------------------------------------------------------------------
    # Archive and global best
    # ------------------------------------------------------------------
    def _maybe_update_global_best(self, position, fitness, feasible, violation):
        if self._is_better_candidate(
            fitness,
            feasible,
            violation,
            self.best_fitness,
            self.best_feasible,
            self.best_violation,
        ):
            self.best_position = position.copy()
            self.best_fitness = float(fitness)
            self.best_feasible = bool(feasible)
            self.best_violation = float(violation)
            self.stagnation_counter = 0
        else:
            # stagnation is updated generation-wise, not candidate-wise
            pass

    def _sort_archive(self):
        if self.maximize:
            self.archive.sort(key=lambda item: item["fitness"], reverse=True)
        else:
            self.archive.sort(key=lambda item: item["fitness"])

    def _update_archive(self, population, fitness, feasible, violation):
        if not self.use_archive:
            return

        pop_cpu = to_cpu(population)
        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        feas_cpu = np.asarray(to_cpu(feasible), dtype=bool)
        viol_cpu = np.asarray(to_cpu(violation), dtype=float)

        for i in range(len(pop_cpu)):
            if not feas_cpu[i]:
                continue
            if not np.isfinite(fit_cpu[i]):
                continue

            pos = xp.array(pop_cpu[i], dtype=float)

            # Simple duplicate guard based on near-equal position.
            duplicate = False
            for item in self.archive:
                if float(to_cpu(xp.linalg.norm(item["position"] - pos))) < 1e-10:
                    duplicate = True
                    break
            if duplicate:
                continue

            self.archive.append({
                "position": pos.copy(),
                "fitness": float(fit_cpu[i]),
                "violation": float(viol_cpu[i]),
            })

        self._sort_archive()
        self.archive = self.archive[: self.archive_size]

    def _archive_centroid(self):
        if len(self.archive) == 0:
            return self.best_position.copy()
        positions = xp.stack([item["position"] for item in self.archive], axis=0)
        return xp.mean(positions, axis=0)

    def _sample_archive_position(self):
        if len(self.archive) == 0:
            return self.best_position.copy()
        idx = np.random.randint(0, len(self.archive))
        return self.archive[idx]["position"].copy()

    # ------------------------------------------------------------------
    # RIME core and SA-DA enhancements
    # ------------------------------------------------------------------
    def _normalize_rime_rates(self, fitness, violation):
        rates = fitness.copy()

        # In minimization, worse fitness should have higher puncture probability.
        # If constraints exist, infeasible/worse-violation candidates receive
        # stronger pressure to copy dimensions from elite solutions.
        if not self.maximize:
            rates = rates - xp.min(rates)
        else:
            rates = xp.max(rates) - rates

        rates = rates + violation
        rates = xp.nan_to_num(rates, nan=0.0, posinf=0.0, neginf=0.0)
        rates = xp.maximum(rates, 0.0)

        norm_value = xp.linalg.norm(rates)
        if self._to_float(norm_value) <= 1e-30:
            return xp.zeros_like(rates)
        return xp.clip(rates / norm_value, 0.0, 1.0)

    def _population_diversity(self, population):
        if self.population_size <= 1:
            return 0.0
        normalized = (population - self.min_params) / (self.range_params + 1e-12)
        diversity = xp.mean(xp.std(normalized, axis=0))
        return float(to_cpu(diversity))

    def _safety_violation_rate(self, feasible):
        feasible_cpu = np.asarray(to_cpu(feasible), dtype=bool)
        return float(1.0 - np.mean(feasible_cpu))

    def _adaptive_factors(self, diversity, violation_rate):
        progress = self._progress()
        base_e = math.sqrt(progress)

        if not self.use_adaptive_schedule:
            return base_e, 1.0

        low_div = max(0.0, (self.diversity_threshold - diversity) / max(self.diversity_threshold, 1e-12))
        stagnated = 1.0 if self.stagnation_counter >= self.stagnation_patience else 0.0

        # Higher soft probability = more exploration / repair movement.
        soft_probability = base_e + 0.25 * low_div + 0.20 * stagnated + 0.15 * violation_rate
        soft_probability = float(np.clip(soft_probability, 0.0, 1.0))

        # Hard puncture becomes stronger as the search matures, but is reduced
        # when diversity is too low to prevent every candidate from collapsing
        # into one point too early.
        puncture_gain = 0.75 + 0.50 * progress - 0.25 * low_div
        puncture_gain = float(np.clip(puncture_gain, 0.50, 1.50))

        return soft_probability, puncture_gain

    def _rime_update(self, population, fitness, feasible, violation, diversity, violation_rate):
        progress = self._progress()

        random_scalar = float(to_cpu(xp.random.random()))
        denominator = self.max_fes if self.max_fes is not None else self.max_iter
        current = self.fes if self.max_fes is not None else self.iteration
        current = max(1, current)

        decay = 1.0 - float(to_cpu(self._matlab_round_positive(current * self.w / max(1, denominator)))) / self.w
        rime_factor = (
            (random_scalar - 0.5)
            * 2.0
            * math.cos(math.pi * current / max(1.0, denominator / 10.0))
            * decay
        )

        soft_probability, puncture_gain = self._adaptive_factors(diversity, violation_rate)

        elite_target = self._archive_centroid() if (self.use_archive and len(self.archive) > 0) else self.best_position
        elite_row = elite_target.reshape(1, -1)

        new_population = population.copy()

        # Soft-rime exploration around archive-guided elite target.
        random_soft = xp.random.random(size=(self.population_size, self.dim))
        random_positions = self.min_params + xp.random.random(size=(self.population_size, self.dim)) * self.range_params
        soft_candidates = elite_row + rime_factor * random_positions
        new_population = xp.where(random_soft < soft_probability, soft_candidates, new_population)

        # Hard-rime puncture: candidates with poor fitness/violation copy some
        # dimensions from archive-guided elite target.
        normalized_rates = self._normalize_rime_rates(fitness, violation).reshape(-1, 1)
        hard_probability = xp.clip(normalized_rates * puncture_gain, 0.0, 1.0)
        random_hard = xp.random.random(size=(self.population_size, self.dim))
        new_population = xp.where(random_hard < hard_probability, elite_row, new_population)

        return self._clip(new_population)

    def _worst_indices(self, fitness, violation, fraction):
        n = max(1, int(math.ceil(self.population_size * fraction)))
        fit_cpu = np.asarray(to_cpu(fitness), dtype=float)
        viol_cpu = np.asarray(to_cpu(violation), dtype=float)

        # Safety violations increase badness for both min/max cases.
        if self.maximize:
            badness = -fit_cpu + viol_cpu
        else:
            badness = fit_cpu + viol_cpu
        badness = np.nan_to_num(badness, nan=np.inf, posinf=np.inf, neginf=-np.inf)
        return np.argsort(badness)[-n:]

    def _apply_buoyancy_centering(self, candidate_population, population, fitness, feasible, violation, diversity, violation_rate):
        if not self.use_buoyancy:
            return candidate_population

        low_diversity = diversity < self.diversity_threshold
        unsafe_population = violation_rate >= self.safety_violation_threshold
        stagnated = self.stagnation_counter >= self.stagnation_patience

        if not (low_diversity or unsafe_population or stagnated):
            return candidate_population

        centroid = self._archive_centroid()
        progress = self._progress()
        low_div_score = max(0.0, (self.diversity_threshold - diversity) / max(self.diversity_threshold, 1e-12))

        alpha = 0.15 + 0.55 * violation_rate + 0.30 * low_div_score
        alpha = float(np.clip(alpha, 0.10, 0.90))

        beta = self.local_search_scale * (1.0 - progress) + 0.005
        beta = float(np.clip(beta, 0.005, 0.08))

        idxs = self._worst_indices(fitness, violation, self.buoyancy_fraction)
        for idx in idxs:
            x_i = candidate_population[idx]
            x_j = self._sample_archive_position()
            r = xp.random.uniform(-1.0, 1.0, size=self.dim)

            lifted = x_i + alpha * (centroid - x_i) + beta * r * (x_i - x_j)
            candidate_population[idx] = self._clip(lifted)

        return candidate_population

    # ------------------------------------------------------------------
    # Selection, restart, local search
    # ------------------------------------------------------------------
    def _greedy_selection(self, population, candidate_population, fitness, feasible, violation):
        for i in range(self.population_size):
            if self.strict_budget and not self._budget_available():
                break

            result = self._evaluate_candidate(candidate_population[i])
            if result is None:
                break
            new_fit, new_feas, new_viol = result

            old_fit = float(fitness[i])
            old_feas = bool(feasible[i])
            old_viol = float(violation[i])

            if self._is_better_candidate(new_fit, new_feas, new_viol, old_fit, old_feas, old_viol):
                population[i] = candidate_population[i]
                fitness[i] = new_fit
                feasible[i] = new_feas
                violation[i] = new_viol

                self._maybe_update_global_best(candidate_population[i], new_fit, new_feas, new_viol)

        self._update_archive(population, fitness, feasible, violation)
        return population, fitness, feasible, violation

    def _restart_repair(self, population, fitness, feasible, violation, diversity):
        if not self.use_restart:
            return population, fitness, feasible, violation

        trigger = (
            diversity < self.diversity_threshold
            or self.stagnation_counter >= self.stagnation_patience
        )
        if not trigger:
            return population, fitness, feasible, violation

        idxs = self._worst_indices(fitness, violation, self.restart_fraction)
        centroid = self._archive_centroid()

        for idx in idxs:
            if self.strict_budget and not self._budget_available():
                break

            mode = np.random.rand()
            if mode < 0.35:
                # Opposition point of current candidate
                new_pos = self.min_params + self.max_params - population[idx]
            elif mode < 0.70 and len(self.archive) > 0:
                # Archive-centered perturbation
                scale = 0.10 * (1.0 - self._progress()) + 0.02
                perturb = xp.random.normal(0.0, scale, size=self.dim) * self.range_params
                new_pos = centroid + perturb
            else:
                # Random immigrant
                new_pos = self.min_params + xp.random.random(size=self.dim) * self.range_params

            new_pos = self._clip(new_pos)
            result = self._evaluate_candidate(new_pos)
            if result is None:
                break
            new_fit, new_feas, new_viol = result

            # Restart is allowed to replace the worst candidate if it is better
            # in a feasibility-aware sense.
            if self._is_better_candidate(new_fit, new_feas, new_viol, float(fitness[idx]), bool(feasible[idx]), float(violation[idx])):
                population[idx] = new_pos
                fitness[idx] = new_fit
                feasible[idx] = new_feas
                violation[idx] = new_viol
                self._maybe_update_global_best(new_pos, new_fit, new_feas, new_viol)

        self._update_archive(population, fitness, feasible, violation)
        return population, fitness, feasible, violation

    def _local_search(self, population, fitness, feasible, violation):
        if not self.use_local_search:
            return population, fitness, feasible, violation
        if self._progress() < self.local_search_start:
            return population, fitness, feasible, violation
        if len(self.archive) == 0:
            return population, fitness, feasible, violation

        top_k = min(self.local_search_top_k, len(self.archive))
        step_scale = self.local_search_scale * (1.0 - self._progress()) + 0.002

        for k in range(top_k):
            base = self.archive[k]["position"].copy()
            base_fit = self.archive[k]["fitness"]
            base_feas = True
            base_viol = self.archive[k]["violation"]

            current = base.copy()
            current_fit = base_fit
            current_feas = base_feas
            current_viol = base_viol

            for _ in range(self.local_search_steps):
                if self.strict_budget and not self._budget_available():
                    break

                perturb = xp.random.normal(0.0, step_scale, size=self.dim) * self.range_params
                candidate = self._clip(current + perturb)
                result = self._evaluate_candidate(candidate)
                if result is None:
                    break
                cand_fit, cand_feas, cand_viol = result

                if self._is_better_candidate(cand_fit, cand_feas, cand_viol, current_fit, current_feas, current_viol):
                    current = candidate
                    current_fit = cand_fit
                    current_feas = cand_feas
                    current_viol = cand_viol
                    self._maybe_update_global_best(candidate, cand_fit, cand_feas, cand_viol)

            # Inject local refinement into the nearest/worst slot if useful.
            if self._is_better_candidate(current_fit, current_feas, current_viol, base_fit, base_feas, base_viol):
                worst_idx = self._worst_indices(fitness, violation, 1.0 / self.population_size)[0]
                if self._is_better_candidate(current_fit, current_feas, current_viol, float(fitness[worst_idx]), bool(feasible[worst_idx]), float(violation[worst_idx])):
                    population[worst_idx] = current
                    fitness[worst_idx] = current_fit
                    feasible[worst_idx] = current_feas
                    violation[worst_idx] = current_viol

        self._update_archive(population, fitness, feasible, violation)
        return population, fitness, feasible, violation

    # ------------------------------------------------------------------
    # History and main loop
    # ------------------------------------------------------------------
    def _update_stagnation(self):
        improved = self._is_better_value(self.best_fitness, self._last_best_fitness)
        if improved:
            self.stagnation_counter = 0
            self._last_best_fitness = self.best_fitness
        else:
            self.stagnation_counter += 1

    def _save_history(self, diversity, violation_rate):
        self.history["best_fitness"].append(float(self.best_fitness))
        self.history["best_position"].append(np.asarray(to_cpu(self.best_position.copy()), dtype=float))
        self.history["diversity"].append(float(diversity))
        self.history["safety_violation_rate"].append(float(violation_rate))
        self.history["archive_size"].append(int(len(self.archive)))
        self.history["fes"].append(int(self.fes))
        self.history["stagnation_counter"].append(int(self.stagnation_counter))

    def run(self, verbose=True):
        population, fitness, feasible, violation = self.initialize_population()

        if self.max_fes is not None:
            iterator = tqdm(
                total=self.max_fes,
                desc="Optimizing with SA-DA-RIME",
                ncols=100,
                disable=not verbose,
            )
            last_fes = self.fes
        else:
            iterator = tqdm(
                range(1, self.max_iter + 1),
                desc="Optimizing with SA-DA-RIME",
                ncols=100,
                disable=not verbose,
            )
            last_fes = 0

        try:
            while True:
                if self.max_fes is not None:
                    if not self._budget_available():
                        break
                    self.iteration += 1
                else:
                    if self.iteration >= self.max_iter:
                        break
                    self.iteration += 1

                diversity = self._population_diversity(population)
                violation_rate = self._safety_violation_rate(feasible)

                candidate_population = self._rime_update(
                    population,
                    fitness,
                    feasible,
                    violation,
                    diversity,
                    violation_rate,
                )
                candidate_population = self._apply_buoyancy_centering(
                    candidate_population,
                    population,
                    fitness,
                    feasible,
                    violation,
                    diversity,
                    violation_rate,
                )

                population, fitness, feasible, violation = self._greedy_selection(
                    population,
                    candidate_population,
                    fitness,
                    feasible,
                    violation,
                )

                diversity = self._population_diversity(population)
                population, fitness, feasible, violation = self._restart_repair(
                    population,
                    fitness,
                    feasible,
                    violation,
                    diversity,
                )

                population, fitness, feasible, violation = self._local_search(
                    population,
                    fitness,
                    feasible,
                    violation,
                )

                # Refresh metrics after repair/local-search.
                diversity = self._population_diversity(population)
                violation_rate = self._safety_violation_rate(feasible)
                self._update_archive(population, fitness, feasible, violation)
                self._update_stagnation()
                self._save_history(diversity, violation_rate)

                if verbose:
                    if self.max_fes is not None:
                        iterator.update(max(0, self.fes - last_fes))
                        last_fes = self.fes
                    else:
                        # tqdm over range is advanced by for-loop normally, but here
                        # we use a manual while loop; update one iteration manually.
                        iterator.update(1)
                    iterator.set_postfix({
                        "Best": f"{float(self.best_fitness):.6f}",
                        "Div": f"{diversity:.4f}",
                        "Unsafe": f"{violation_rate:.2f}",
                        "FEs": self.fes,
                    })

                if self.progress_callback is not None and self.max_fes is None:
                    self.progress_callback(self.iteration)

                if self.max_fes is None and self.iteration >= self.max_iter:
                    break
        finally:
            if hasattr(iterator, "close"):
                iterator.close()

        return np.asarray(to_cpu(self.best_position), dtype=float), float(self.best_fitness)


# ----------------------------------------------------------------------
# Public wrapper, compatible with run_pso / run_hoa style.
# ----------------------------------------------------------------------
def run_sa_da_rime(
    func,
    min_params,
    max_params,
    population_size=50,
    max_iter=300,
    max_fes=None,
    verbose=False,
    progress_callback=None,
    seed=None,
    constraint_evaluator=None,
    return_full_history=False,
    **kwargs,
):
    """
    Wrapper for ACC/objective_worker compatibility.

    Returns
    -------
    best_params : numpy.ndarray
    best_fitness : float
    history : list[float] or dict
        By default, returns history["best_fitness"] to match pso.py/hoa.py.
        Set return_full_history=True to obtain all diagnostic curves.
    """
    if seed is not None:
        np.random.seed(seed)
        try:
            xp.random.seed(seed)
        except Exception:
            pass

    model = SafetyAwareDiversityAdaptiveRIME(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        population_size=population_size,
        max_iter=max_iter,
        max_fes=max_fes,
        maximize=False,
        constraint_evaluator=constraint_evaluator,
        progress_callback=progress_callback,
        **kwargs,
    )

    best_params, best_fitness = model.run(verbose=verbose)
    history = model.history if return_full_history else model.history["best_fitness"]
    return best_params, best_fitness, history


# Optional short alias.
def run_sadarime(*args, **kwargs):
    return run_sa_da_rime(*args, **kwargs)


if __name__ == "__main__":
    # Smoke test: Sphere function.
    def sphere(x):
        return float(np.sum(x ** 2))

    best_x, best_y, hist = run_sa_da_rime(
        sphere,
        min_params=[-5, -5, -5, -5, -5],
        max_params=[5, 5, 5, 5, 5],
        population_size=20,
        max_iter=50,
        verbose=True,
        seed=42,
    )
    print("Best fitness:", best_y)
    print("Best position:", best_x)
