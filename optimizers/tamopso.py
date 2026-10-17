"""TAMOPSO: Two-stage Archive Maintenance MOPSO.

Reconstructed from Zhang et al., Scientific Reports (2026),
"A multi-objective particle swarm optimization algorithm with two-stage
archive maintenance and auxiliary archive guidance".

Implemented paper mechanisms
----------------------------
1. Standard PSO movement (paper Eqs. 2-3).
2. Two-stage external archive maintenance (Eqs. 4-13, Algorithm 2):
   adaptive grids -> high-density cell -> DDI + minimum-angle ranking.
3. Bounded auxiliary archive and pbest update (Eqs. 14-20, Algorithm 3).
4. Stagnation detector and archive-guided reconstruction (Eqs. 21-23,
   Algorithm 4), including top-30% crowding-distance guide candidates.

Reproduction notes
------------------
- The paper says gbest is selected from Arc but does not provide a separate
  leader-selection formula.  This implementation samples it uniformly from Arc.
- The text says 50 grid lines span the expanded [fmin-ep, fmax+ep] range.
  The printed Eq. (5) is geometrically inconsistent with those stated endpoints;
  here the grid is built directly with linspace(fmin-ep, fmax+ep, 50).
- TAMOPSO is presented/tested as a minimization optimizer. ``maximize`` is kept
  for API symmetry, but faithful use is ``maximize=False``.
"""

from __future__ import annotations

import math
import numpy as np
from tqdm import tqdm

from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything

from mopso_common import (
    best_per_objective,
    crowding_distance,
    dominates,
    evaluate_population,
    evaluate_position,
    non_dominated_union,
    representative_index,
    seed_backends,
    to_minimization,
    unique_solution_indices,
)


def _rank_with_ties(values, smaller_is_better=True, atol=1e-12):
    """Rank 1 = best; numerically tied values receive the same rank."""
    values = np.asarray(values, dtype=float)
    order = np.argsort(values if smaller_is_better else -values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    rank = 1
    ranks[order[0]] = rank
    for k in range(1, len(order)):
        prev = values[order[k - 1]]
        cur = values[order[k]]
        if not np.isclose(cur, prev, atol=atol, rtol=0.0):
            rank = k + 1
        ranks[order[k]] = rank
    return ranks


class TAMOPSO:
    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        n_particles=200,
        max_iteration=50,
        archive_size=None,
        maximize=False,
        w=0.4,
        c1=2.0,
        c2=2.0,
        numgrid=50,
        stagnation_threshold=3,
        auxiliary_size=None,
        progress_callback=None,
        seed=None,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.asarray(min_params, dtype=float)
        self.max_params = xp.asarray(max_params, dtype=float)
        self.dim = len(min_params)
        self.n_particles = int(n_particles)
        self.max_iteration = int(max_iteration)
        self.archive_size = int(archive_size or n_particles)
        self.auxiliary_size = int(auxiliary_size or self.archive_size)
        self.maximize = maximize

        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.numgrid = int(numgrid)
        self.stagnation_threshold = int(stagnation_threshold)
        self.progress_callback = progress_callback
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        seed_backends(seed)

        self.positions = self._initialize_positions()
        self.velocities = xp.zeros((self.n_particles, self.dim), dtype=float)
        self.objectives = None

        self.pbest_positions = self.positions.copy()
        self.pbest_objectives = None
        self.stagnation = np.zeros(self.n_particles, dtype=int)

        self.archive_positions = np.empty((0, self.dim), dtype=float)
        self.archive_objectives = None

        self.aux_positions = np.empty((0, self.dim), dtype=float)
        self.aux_objectives = None
        self.aux_age = np.empty(0, dtype=np.int64)
        self._age_counter = 0

        self.history = {
            "best_fitness": [],
            "best_position": [],
            "archive_size": [],
            "auxiliary_size": [],
            "stagnant_particles": [],
        }

    # ------------------------------------------------------------------
    # Initialization / evaluation
    # ------------------------------------------------------------------
    def _initialize_positions(self):
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.n_particles, self.dim),
        )

    def _evaluate(self, positions):
        return evaluate_population(self.fitness_function, positions)

    # ------------------------------------------------------------------
    # Basic PSO movement (paper Eqs. 2-3)
    # ------------------------------------------------------------------
    def _select_global_guide(self):
        if len(self.archive_positions) == 0:
            idx = int(self.rng.integers(self.n_particles))
            return self.positions[idx].copy()
        idx = int(self.rng.integers(len(self.archive_positions)))
        return xp.asarray(self.archive_positions[idx])

    def _move_particles(self):
        for i in range(self.n_particles):
            gbest = self._select_global_guide()
            r1 = float(self.rng.random())
            r2 = float(self.rng.random())
            self.velocities[i] = (
                self.w * self.velocities[i]
                + self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
                + self.c2 * r2 * (gbest - self.positions[i])
            )
            self.positions[i] = xp.clip(
                self.positions[i] + self.velocities[i],
                self.min_params,
                self.max_params,
            )

    # ------------------------------------------------------------------
    # Two-stage archive maintenance (paper Eqs. 4-13, Algorithm 2)
    # ------------------------------------------------------------------
    def _grid_cells(self, objectives_min, iteration):
        f = np.asarray(objectives_min, dtype=float)
        n, m = f.shape
        coords = np.zeros((n, m), dtype=int)
        ratio = float(iteration) / max(float(self.max_iteration), 1.0)

        for j in range(m):
            vals = f[:, j]
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            sigma = float(np.std(vals))
            ep = 0.05 * (1.0 + ratio) * sigma  # Eq. (4)

            if np.isclose(lo, hi, atol=1e-15, rtol=0.0):
                coords[:, j] = 0
                continue

            # The surrounding paper text specifies expanded endpoints and 50
            # equidistant grid lines.  Building those endpoints directly avoids
            # the apparent sign typo in printed Eq. (5).
            lines = np.linspace(lo - ep, hi + ep, self.numgrid)
            # Eqs. (6)-(7): assign to the closest grid line.  For equally spaced
            # lines this is equivalent to nearest-line indexing.
            coords[:, j] = np.argmin(np.abs(vals[:, None] - lines[None, :]), axis=1)

        return [tuple(row.tolist()) for row in coords]

    @staticmethod
    def _minimum_angles(region_objectives):
        f = np.asarray(region_objectives, dtype=float)
        n = len(f)
        if n <= 1:
            return np.full(n, np.pi, dtype=float)
        norms = np.linalg.norm(f, axis=1)
        theta_min = np.full(n, np.inf, dtype=float)
        for i in range(n):
            for j in range(i + 1, n):
                denom = norms[i] * norms[j]
                if denom <= 1e-15:
                    theta = 0.0
                else:
                    cosine = np.clip(abs(np.dot(f[i], f[j])) / denom, -1.0, 1.0)
                    theta = float(np.arccos(cosine))
                theta_min[i] = min(theta_min[i], theta)
                theta_min[j] = min(theta_min[j], theta)
        theta_min[np.isinf(theta_min)] = np.pi
        return theta_min

    @staticmethod
    def _ddi(objectives_min, region_indices):
        f = np.asarray(objectives_min, dtype=float)
        reg = f[np.asarray(region_indices, dtype=int)]
        z_min = np.min(f, axis=0)
        z_max = np.max(f, axis=0)
        d1 = np.linalg.norm(reg - z_min, axis=1)  # Eq. (10)
        d2 = np.linalg.norm(reg - z_max, axis=1)  # Eq. (11)
        # Eq. (12)
        return np.maximum(d1 - np.min(d1), np.max(d2) - d2)

    @staticmethod
    def _local_crowding(region_objectives):
        return crowding_distance(np.asarray(region_objectives, dtype=float))

    def _choose_deletion(self, objectives_min, region_indices, iteration):
        idx = np.asarray(region_indices, dtype=int)
        region_f = objectives_min[idx]

        ddi = self._ddi(objectives_min, idx)
        theta = self._minimum_angles(region_f)
        crank = _rank_with_ties(ddi, smaller_is_better=True)
        drank = _rank_with_ties(theta, smaller_is_better=False)

        ratio = float(iteration) / max(float(self.max_iteration), 1.0)
        score = (1.0 - ratio) * crank + ratio * drank  # Eq. (13)
        worst = np.max(score)
        candidates = np.flatnonzero(np.isclose(score, worst, atol=1e-12, rtol=0.0))

        if len(candidates) > 1:
            if ratio < 0.5:
                # Early stage: convergence tie-break first.
                val = np.max(ddi[candidates])
                candidates = candidates[np.isclose(ddi[candidates], val, atol=1e-12, rtol=0.0)]
                if len(candidates) > 1:
                    val = np.min(theta[candidates])
                    candidates = candidates[np.isclose(theta[candidates], val, atol=1e-12, rtol=0.0)]
            else:
                # Later stage: diversity tie-break first.
                val = np.min(theta[candidates])
                candidates = candidates[np.isclose(theta[candidates], val, atol=1e-12, rtol=0.0)]
                if len(candidates) > 1:
                    val = np.max(ddi[candidates])
                    candidates = candidates[np.isclose(ddi[candidates], val, atol=1e-12, rtol=0.0)]

        if len(candidates) > 1:
            cd = self._local_crowding(region_f)
            finite_cd = np.where(np.isfinite(cd[candidates]), cd[candidates], np.inf)
            val = np.min(finite_cd)
            if np.isfinite(val):
                candidates = candidates[np.isclose(finite_cd, val, atol=1e-12, rtol=0.0)]

        local_remove = int(self.rng.choice(candidates)) if len(candidates) > 1 else int(candidates[0])
        return int(idx[local_remove])

    def _update_external_archive(self, positions, objectives, iteration):
        x_new = np.asarray(to_cpu(positions), dtype=float)
        f_new = np.asarray(objectives, dtype=float)

        if self.archive_objectives is None or len(self.archive_positions) == 0:
            x_all, f_all = x_new, f_new
        else:
            x_all = np.vstack([self.archive_positions, x_new])
            f_all = np.vstack([self.archive_objectives, f_new])

        x_arc, f_arc = non_dominated_union(x_all, f_all, self.maximize)
        deleted_x, deleted_f = [], []

        while len(x_arc) > self.archive_size:
            f_min = to_minimization(f_arc, self.maximize)
            cells = self._grid_cells(f_min, iteration)
            counts = {}
            for k, cell in enumerate(cells):
                counts.setdefault(cell, []).append(k)
            # Deterministic tie between equally dense cells: first encountered.
            dense_cell = max(counts.keys(), key=lambda c: len(counts[c]))
            region_idx = counts[dense_cell]
            remove = self._choose_deletion(f_min, region_idx, iteration)
            deleted_x.append(x_arc[remove].copy())
            deleted_f.append(f_arc[remove].copy())
            x_arc = np.delete(x_arc, remove, axis=0)
            f_arc = np.delete(f_arc, remove, axis=0)

        self.archive_positions = x_arc
        self.archive_objectives = f_arc

        if deleted_x:
            return np.vstack(deleted_x), np.vstack(deleted_f)
        n_obj = f_new.shape[1]
        return np.empty((0, self.dim)), np.empty((0, n_obj))

    # ------------------------------------------------------------------
    # Bounded auxiliary archive + pbest (paper Eqs. 14-20, Algorithm 3)
    # ------------------------------------------------------------------
    def _update_auxiliary_archive(self, deleted_positions, deleted_objectives):
        if len(deleted_positions) > 0:
            new_age = np.arange(
                self._age_counter,
                self._age_counter + len(deleted_positions),
                dtype=np.int64,
            )
            self._age_counter += len(deleted_positions)

            if self.aux_objectives is None or len(self.aux_positions) == 0:
                x = np.asarray(deleted_positions, dtype=float)
                f = np.asarray(deleted_objectives, dtype=float)
                age = new_age
            else:
                x = np.vstack([self.aux_positions, deleted_positions])
                f = np.vstack([self.aux_objectives, deleted_objectives])
                age = np.concatenate([self.aux_age, new_age])

            idx = unique_solution_indices(x, f)
            x, f, age = x[idx], f[idx], age[idx]
        else:
            if self.aux_objectives is None:
                return
            x, f, age = self.aux_positions, self.aux_objectives, self.aux_age

        while len(x) > self.auxiliary_size:
            cd = crowding_distance(f)
            finite = np.where(np.isfinite(cd), cd, np.inf)
            min_cd = np.min(finite)
            if np.isinf(min_cd):
                tied = np.arange(len(x))
            else:
                tied = np.flatnonzero(np.isclose(finite, min_cd, atol=1e-12, rtol=0.0))
            # Algorithm 3: if crowding-distance tie exists, remove the older solution.
            remove = int(tied[np.argmin(age[tied])])
            x = np.delete(x, remove, axis=0)
            f = np.delete(f, remove, axis=0)
            age = np.delete(age, remove)

        self.aux_positions = x
        self.aux_objectives = f
        self.aux_age = age

    def _gaussian_noise(self):
        span = np.asarray(to_cpu(self.max_params - self.min_params), dtype=float)
        sigma = 0.01 * span  # Eq. (19)
        return self.rng.normal(0.0, sigma, size=self.dim)

    def _random_aux_position(self):
        idx = int(self.rng.integers(len(self.aux_positions)))
        return self.aux_positions[idx].copy()

    def _update_pbest(self, iteration):
        cur_min = to_minimization(self.objectives, self.maximize)
        pbest_min = to_minimization(self.pbest_objectives, self.maximize)
        pool = np.vstack([cur_min, pbest_min])
        f_lo = np.min(pool, axis=0)
        f_hi = np.max(pool, axis=0)
        denom = (f_hi - f_lo) + 1e-12

        min_b = np.asarray(to_cpu(self.min_params), dtype=float)
        max_b = np.asarray(to_cpu(self.max_params), dtype=float)
        ratio = float(iteration) / max(float(self.max_iteration), 1.0)

        for i in range(self.n_particles):
            x = np.asarray(to_cpu(self.positions[i]), dtype=float)
            pprev = np.asarray(to_cpu(self.pbest_positions[i]), dtype=float)
            aux_available = len(self.aux_positions) > 0

            if dominates(cur_min[i], pbest_min[i]):
                if aux_available:
                    c = self._random_aux_position()
                    improvement = (pbest_min[i] - cur_min[i]) / denom  # Eq. (17), minimization form
                    lam = 0.5 * (1.0 + np.tanh(np.mean(improvement)))  # Eq. (18)
                    beta = float(self.rng.random()) * (1.0 - ratio)    # Eq. (16)
                    # Eq. (15): third term follows the described improvement
                    # direction from previous pbest -> new position.
                    pnew = lam * x + (1.0 - lam) * c + beta * (x - pprev)
                else:
                    pnew = x.copy()  # Eq. (14)
            else:
                noise = self._gaussian_noise()
                if aux_available:
                    c = self._random_aux_position()
                    r1 = float(self.rng.random())
                    r2 = float(self.rng.random())
                    pnew = pprev + r1 * (c - pprev) + r2 * (c - x) + noise  # Eq. (20)
                else:
                    pnew = pprev + noise  # Eq. (19)

            pnew = np.clip(pnew, min_b, max_b)
            self.pbest_positions[i] = xp.asarray(pnew)
            self.pbest_objectives[i] = evaluate_position(
                self.fitness_function,
                self.pbest_positions[i],
            )

    # ------------------------------------------------------------------
    # Stagnation detector + reconstruction (paper Eqs. 21-23, Algorithm 4)
    # ------------------------------------------------------------------
    def _update_stagnation_and_reconstruct(self, old_positions, old_objectives):
        old_min = to_minimization(old_objectives, self.maximize)
        new_min = to_minimization(self.objectives, self.maximize)

        for i in range(self.n_particles):
            if dominates(new_min[i], old_min[i]):
                self.stagnation[i] = 0
            else:
                self.stagnation[i] += 1

        if len(self.archive_positions) == 0:
            return

        stagnant_idx = np.flatnonzero(self.stagnation >= self.stagnation_threshold)
        if len(stagnant_idx) == 0:
            return

        cd = crowding_distance(self.archive_objectives)
        # Descending crowding: boundary points (inf) naturally appear first.
        order = np.argsort(-cd, kind="mergesort")
        k = max(1, int(math.ceil(0.30 * len(order))))
        candidate_idx = order[:k]

        min_b = np.asarray(to_cpu(self.min_params), dtype=float)
        max_b = np.asarray(to_cpu(self.max_params), dtype=float)

        for i in stagnant_idx:
            if len(candidate_idx) >= 2:
                guides = self.rng.choice(candidate_idx, size=2, replace=False)
            else:
                guides = np.array([candidate_idx[0], candidate_idx[0]], dtype=int)

            x = np.asarray(to_cpu(self.positions[i]), dtype=float)
            g1 = self.archive_positions[int(guides[0])]
            g2 = self.archive_positions[int(guides[1])]
            xnew = (x + g1 + g2) / 3.0  # Eq. (22)
            xnew = np.clip(xnew, min_b, max_b)  # Eq. (23)
            self.positions[i] = xp.asarray(xnew)
            self.objectives[i] = evaluate_position(self.fitness_function, self.positions[i])
            self.stagnation[i] = 0

    # ------------------------------------------------------------------
    # History / run
    # ------------------------------------------------------------------
    def _save_history(self):
        if len(self.archive_positions) == 0:
            return
        rep = representative_index(self.archive_objectives, self.maximize)
        self.history["best_fitness"].append(
            best_per_objective(self.archive_objectives, self.maximize).copy()
        )
        self.history["best_position"].append(self.archive_positions[rep].copy())
        self.history["archive_size"].append(int(len(self.archive_positions)))
        self.history["auxiliary_size"].append(int(len(self.aux_positions)))
        self.history["stagnant_particles"].append(
            int(np.sum(self.stagnation >= self.stagnation_threshold))
        )

    def run(self, verbose=True):
        self.objectives = self._evaluate(self.positions)
        self.pbest_objectives = self.objectives.copy()

        # Algorithm 1 initialization: Arc starts from non-dominated population.
        self._update_external_archive(self.positions, self.objectives, iteration=0)
        iterator = tqdm(
            range(1, self.max_iteration + 1),
            desc="Optimizing with TAMOPSO",
            ncols=100,
            disable=not verbose,
        )

        for t in iterator:
            old_positions = self.positions.copy()
            old_objectives = self.objectives.copy()

            # Algorithm 1: select gbest, move, boundary correction, evaluate.
            self._move_particles()
            self.objectives = self._evaluate(self.positions)

            # Eq. (21) + Algorithm 4.
            self._update_stagnation_and_reconstruct(old_positions, old_objectives)

            # Add current ND candidates, apply two-stage archive maintenance.
            sdel_x, sdel_f = self._update_external_archive(
                self.positions,
                self.objectives,
                iteration=t,
            )

            # Algorithm 3: bounded historical archive then pbest update.
            self._update_auxiliary_archive(sdel_x, sdel_f)
            self._update_pbest(iteration=t)
            self._save_history()

            iterator.set_postfix({
                "Arc": len(self.archive_positions),
                "Aux": len(self.aux_positions),
            })

            if self.progress_callback is not None:
                self.progress_callback(t)

        return self.archive_positions.copy(), self.archive_objectives.copy()


def run_tamopso(
    func,
    min_params,
    max_params,
    population_size=200,
    max_iter=50,
    verbose=False,
    progress_callback=None,
    seed=None,
    maximize=False,
    archive_size=None,
    auxiliary_size=None,
    w=0.4,
    c1=2.0,
    c2=2.0,
    numgrid=50,
    stagnation_threshold=3,
):
    """Wrapper following the same overall style as the supplied run_pso/run_hoa.

    Returns the Pareto archive, not a fabricated scalar 'best' solution.
    """
    model = TAMOPSO(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_particles=population_size,
        max_iteration=max_iter,
        archive_size=archive_size,
        auxiliary_size=auxiliary_size,
        maximize=maximize,
        w=w,
        c1=c1,
        c2=c2,
        numgrid=numgrid,
        stagnation_threshold=stagnation_threshold,
        progress_callback=progress_callback,
        seed=seed,
    )

    pareto_params, pareto_objectives = model.run(verbose=verbose)
    return pareto_params, pareto_objectives, model.history["best_fitness"]
