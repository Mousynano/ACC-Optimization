"""MOPSO/DMS: decomposition + multi-selection multi-objective PSO.

Reconstructed from Ma, Dai, Xue & Peng (2025), CMC 82(1), 997-1026,
"A Multi-Objective Particle Swarm Optimization Algorithm Based on
Decomposition and Multi-Selection Strategy".

Implemented paper mechanisms
----------------------------
- decomposition by uniformly distributed direction vectors (Eq. 5);
- decomposition-based evolving-population update (Algorithm 1);
- Pareto + decomposition external archive update (Algorithm 2);
- NDP probability distribution for discontinuous PF handling (Eqs. 10-11,
  Algorithm 3);
- multi-selection of gbest/pbest with PBI (Eqs. 7-9, Algorithm 4);
- adaptive inertia weight (Eqs. 12-13);
- PSO position/velocity update (Eqs. 2-3) + polynomial mutation
  (Eqs. 14-17);
- optional final IGD comparison between POP and E when a true/reference PF
  is supplied (Algorithm 5).

Reproduction notes
------------------
The paper defines T (neighbor size), TCount, w_min, and a uniform direction-vector
set as inputs/concepts but does not report numeric values for T, TCount, or w_min,
nor a generator for the direction vectors.  Defaults here are therefore explicit
implementation choices: neighbor_size=10, t_count=5, w_min=0.4; w_max=0.9 follows
the reported initial inertia.  All can be overridden.

The paper's PBI equations are implemented as printed.  The pbest 'largest angle'
rule is operationalized as the angle in decision space between candidate-x_i and
gbest-x_i because the paper does not specify the coordinate space for that angle.
"""

from __future__ import annotations

import itertools
import math
from math import comb

import numpy as np
from tqdm import tqdm

from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything

from mopso_common import (
    angle_between,
    best_per_objective,
    dominates,
    evaluate_population,
    igd,
    non_dominated_mask,
    non_dominated_union,
    representative_index,
    seed_backends,
    to_minimization,
    unique_solution_indices,
)


def _simplex_lattice(n_points: int, n_objectives: int) -> np.ndarray:
    """Deterministic simplex-lattice direction vectors, subsampled if necessary."""
    n_points = int(n_points)
    m = int(n_objectives)
    if m == 1:
        return np.ones((n_points, 1), dtype=float)
    if m == 2:
        a = np.linspace(0.0, 1.0, n_points)
        return np.column_stack([a, 1.0 - a])

    h = 1
    while comb(h + m - 1, m - 1) < n_points:
        h += 1

    def compositions(total, parts, prefix=()):
        if parts == 1:
            yield prefix + (total,)
            return
        for v in range(total + 1):
            yield from compositions(total - v, parts - 1, prefix + (v,))

    points = np.asarray(list(compositions(h, m)), dtype=float) / float(h)
    if len(points) <= n_points:
        return points

    # Farthest-point subsampling for a reasonably even deterministic subset.
    selected = [0]
    min_dist = np.linalg.norm(points - points[0], axis=1)
    while len(selected) < n_points:
        idx = int(np.argmax(min_dist))
        selected.append(idx)
        min_dist = np.minimum(min_dist, np.linalg.norm(points - points[idx], axis=1))
    return points[np.asarray(selected, dtype=int)]


class MOPSODMS:
    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        n_particles=100,
        max_iteration=300,
        archive_size=None,
        maximize=False,
        neighbor_size=10,
        t_count=5,
        theta=5.0,
        w_max=0.9,
        w_min=0.4,
        c1=2.0,
        c2=2.0,
        mutation_probability=None,
        eta_m=20.0,
        reference_front=None,
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
        self.maximize = maximize

        self.neighbor_size = int(max(1, min(neighbor_size, self.n_particles)))
        self.t_count = int(t_count)
        self.theta = float(theta)
        self.w_max = float(w_max)
        self.w_min = float(w_min)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.mutation_probability = (
            1.0 / self.dim if mutation_probability is None else float(mutation_probability)
        )
        self.eta_m = float(eta_m)
        self.reference_front = None if reference_front is None else np.asarray(reference_front, dtype=float)
        self.progress_callback = progress_callback
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        seed_backends(seed)

        self.positions = self._initialize_positions()
        self.velocities = xp.zeros((self.n_particles, self.dim), dtype=float)
        self.objectives = None

        self.directions = None
        self.neighbors = None
        self.count = np.zeros(self.n_particles, dtype=int)
        self.probability = np.full(self.n_particles, 1.0 / self.n_particles, dtype=float)

        self.archive_positions = np.empty((0, self.dim), dtype=float)
        self.archive_objectives = None

        self.z = None
        self.z_star = None

        self.history = {
            "best_fitness": [],
            "best_position": [],
            "archive_size": [],
            "counts": [],
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

    def _initialize_directions(self, n_objectives):
        self.directions = _simplex_lattice(self.n_particles, n_objectives)
        # Neighborhood Bi: T nearest direction vectors to gamma_i.
        dist = np.linalg.norm(
            self.directions[:, None, :] - self.directions[None, :, :],
            axis=2,
        )
        self.neighbors = np.argsort(dist, axis=1, kind="mergesort")[:, : self.neighbor_size]

    # ------------------------------------------------------------------
    # Decomposition (paper Eq. 5)
    # ------------------------------------------------------------------
    def _classify(self, objectives_min, z=None):
        f = np.asarray(objectives_min, dtype=float)
        z = np.min(f, axis=0) if z is None else np.asarray(z, dtype=float)
        shifted = f - z

        f_norm = np.linalg.norm(shifted, axis=1, keepdims=True)
        g_norm = np.linalg.norm(self.directions, axis=1, keepdims=True).T
        denom = f_norm * g_norm
        sim = shifted @ self.directions.T
        sim = np.divide(sim, denom, out=np.zeros_like(sim), where=denom > 1e-15)

        # For an ideal-point vector (zero norm), all directions are equivalent;
        # choose direction 0 deterministically.
        zero = np.flatnonzero(f_norm[:, 0] <= 1e-15)
        if len(zero):
            sim[zero, :] = -np.inf
            sim[zero, 0] = 1.0
        return np.argmax(sim, axis=1), sim

    def _population_update(self, positions, objectives, velocities):
        x = np.asarray(positions, dtype=float)
        f_raw = np.asarray(objectives, dtype=float)
        v = np.asarray(velocities, dtype=float)
        f_min = to_minimization(f_raw, self.maximize)
        assignment, sim = self._classify(f_min, self.z)
        global_nd = non_dominated_mask(f_min)
        global_nd_idx = np.flatnonzero(global_nd)

        chosen = []
        for i in range(self.n_particles):
            members = np.flatnonzero(assignment == i)
            if len(members) == 0:
                pool = global_nd_idx if len(global_nd_idx) else np.arange(len(x))
            elif len(members) == 1:
                chosen.append(int(members[0]))
                continue
            else:
                local_mask = non_dominated_mask(f_min[members])
                pool = members[local_mask]
                if len(pool) == 0:
                    pool = members

            # Smallest angle <=> largest cosine similarity to gamma_i.
            best = int(pool[np.argmax(sim[pool, i])])
            chosen.append(best)

        chosen = np.asarray(chosen, dtype=int)
        return x[chosen], f_raw[chosen], v[chosen]

    # ------------------------------------------------------------------
    # External archive update (paper Algorithm 2)
    # ------------------------------------------------------------------
    def _archive_update(self, candidate_positions, candidate_objectives):
        x_new = np.asarray(candidate_positions, dtype=float)
        f_new = np.asarray(candidate_objectives, dtype=float)

        if self.archive_objectives is None or len(self.archive_positions) == 0:
            x_all, f_all = x_new, f_new
        else:
            x_all = np.vstack([self.archive_positions, x_new])
            f_all = np.vstack([self.archive_objectives, f_new])

        x_nd, f_nd = non_dominated_union(x_all, f_all, self.maximize)
        if len(x_nd) <= self.archive_size:
            self.archive_positions, self.archive_objectives = x_nd, f_nd
            return

        f_min = to_minimization(f_nd, self.maximize)
        assignment, sim = self._classify(f_min, self.z)

        selected = []
        for i in range(self.n_particles):
            members = np.flatnonzero(assignment == i)
            if len(members) == 0:
                continue
            best = int(members[np.argmax(sim[members, i])])
            if best not in selected:
                selected.append(best)

        remaining = [i for i in range(len(x_nd)) if i not in selected]

        # Algorithm 2 line 11: fill with the solutions farthest from E.
        while len(selected) < self.archive_size and remaining:
            if not selected:
                pick = remaining.pop(0)
                selected.append(pick)
                continue
            sel_f = f_min[np.asarray(selected, dtype=int)]
            rem_idx = np.asarray(remaining, dtype=int)
            distances = np.linalg.norm(f_min[rem_idx, None, :] - sel_f[None, :, :], axis=2)
            min_distance = np.min(distances, axis=1)
            k = int(np.argmax(min_distance))
            selected.append(int(rem_idx[k]))
            remaining.pop(k)

        selected = np.asarray(selected[: self.archive_size], dtype=int)
        self.archive_positions = x_nd[selected]
        self.archive_objectives = f_nd[selected]

    # ------------------------------------------------------------------
    # Multi-selection (paper Eqs. 7-11, Algorithms 3-4)
    # ------------------------------------------------------------------
    def _pbi(self, objective_min, direction):
        lam = np.asarray(direction, dtype=float)
        f = np.asarray(objective_min, dtype=float)
        norm_lam = max(np.linalg.norm(lam), 1e-15)
        d1 = abs(np.dot(self.z - f, lam)) / norm_lam  # Eq. (8), as printed
        d2 = np.linalg.norm(f - (self.z - d1 * lam))  # Eq. (9), as printed
        return float(d1 + self.theta * d2)

    def _update_counts_and_probabilities(self):
        f_min = to_minimization(self.objectives, self.maximize)
        nd = non_dominated_mask(f_min)
        has_nd = nd.copy()  # particle i is representative of subspace Yi

        self.count = np.where(has_nd, 0, self.count + 1)

        n_i = has_nd.astype(float)
        total_nd = np.sum(n_i)
        if total_nd <= 0:
            ndp = np.zeros(self.n_particles, dtype=float)
        else:
            ndp = n_i / total_nd  # Eq. (11)

        ndp = ndp.copy()
        ndp[self.count >= self.t_count] = 1.0
        weights = 1.0 - ndp
        total = np.sum(weights)
        if total <= 1e-15:
            eligible = self.count < self.t_count
            if np.any(eligible):
                weights = eligible.astype(float)
            else:
                weights = np.ones(self.n_particles, dtype=float)
            total = np.sum(weights)
        self.probability = weights / total  # Eq. (10)
        return has_nd

    def _guides_for_subspace(self, i, has_nd):
        neigh = np.asarray(self.neighbors[i], dtype=int)
        f_min = to_minimization(self.objectives, self.maximize)

        if not has_nd[i]:
            pbi_values = np.asarray([
                self._pbi(f_min[j], self.directions[i]) for j in neigh
            ])
            g_idx = int(neigh[np.argmin(pbi_values)])

            xi = np.asarray(to_cpu(self.positions[i]), dtype=float)
            g = np.asarray(to_cpu(self.positions[g_idx]), dtype=float)
            angles = []
            for j in neigh:
                cand = np.asarray(to_cpu(self.positions[int(j)]), dtype=float)
                angles.append(angle_between(cand - xi, g - xi))
            p_idx = int(neigh[int(np.argmax(angles))])
        else:
            if len(neigh) >= 2:
                pair = self.rng.choice(neigh, size=2, replace=False)
                g_idx, p_idx = int(pair[0]), int(pair[1])
            else:
                g_idx = p_idx = int(neigh[0])

        return self.positions[p_idx].copy(), self.positions[g_idx].copy()

    def _roulette_subspace(self):
        return int(self.rng.choice(self.n_particles, p=self.probability))

    # ------------------------------------------------------------------
    # Adaptive inertia (paper Eqs. 12-13)
    # ------------------------------------------------------------------
    def _inertia_weight(self, particle_idx, iteration):
        f_min = to_minimization(self.objectives, self.maximize)
        denom = self.z_star - self.z
        denom = np.where(np.abs(denom) > 1e-15, denom, 1.0)
        r_i = float(np.mean((f_min[particle_idx] - self.z) / denom))  # Eq. (13)
        w = self.w_max - (
            (self.w_max - self.w_min)
            * (float(iteration) / max(float(self.max_iteration), 1.0))
            * np.exp(-r_i)
        )  # Eq. (12)
        return float(w)

    # ------------------------------------------------------------------
    # PSO move + polynomial mutation (paper Eqs. 2-3, 14-17)
    # ------------------------------------------------------------------
    def _polynomial_mutation(self, position):
        x = np.asarray(position, dtype=float).copy()
        lb = np.asarray(to_cpu(self.min_params), dtype=float)
        ub = np.asarray(to_cpu(self.max_params), dtype=float)

        for k in range(self.dim):
            if self.rng.random() > self.mutation_probability:
                continue
            if ub[k] - lb[k] <= 1e-15:
                x[k] = lb[k]
                continue

            u = float(self.rng.random())
            delta1 = (x[k] - lb[k]) / (ub[k] - lb[k])  # Eq. (16)
            delta2 = (ub[k] - x[k]) / (ub[k] - lb[k])  # Eq. (17)
            mut_pow = 1.0 / (self.eta_m + 1.0)

            if u <= 0.5:
                val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (self.eta_m + 1.0)
                delta = val ** mut_pow - 1.0
            else:
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - delta2) ** (self.eta_m + 1.0)
                delta = 1.0 - val ** mut_pow

            x[k] = x[k] + delta * (ub[k] - lb[k])  # Eq. (14)

        return np.clip(x, lb, ub)

    def _generate_offspring(self, iteration, has_nd):
        # Parent snapshot is required because Algorithm 5 unions old POP with new particles.
        parent_positions = np.asarray(to_cpu(self.positions), dtype=float)
        parent_velocities = np.asarray(to_cpu(self.velocities), dtype=float)

        work_x = self.positions.copy()
        work_v = self.velocities.copy()

        for i in range(self.n_particles):
            if self.count[i] < self.t_count:
                target = i
            else:
                target = self._roulette_subspace()

            # The prose around Algorithm 4 implies guides correspond to the
            # actually selected subspace when a discontinuous subspace is rerouted.
            pbest, gbest = self._guides_for_subspace(target, has_nd)
            w_i = self._inertia_weight(target, iteration)
            r1 = float(self.rng.random())
            r2 = float(self.rng.random())

            v_new = (
                w_i * work_v[target]
                + self.c1 * r1 * (pbest - work_x[target])
                + self.c2 * r2 * (gbest - work_x[target])
            )
            x_new = xp.clip(
                work_x[target] + v_new,
                self.min_params,
                self.max_params,
            )
            x_new_cpu = self._polynomial_mutation(to_cpu(x_new))
            work_v[target] = v_new
            work_x[target] = xp.asarray(x_new_cpu)

        offspring_positions = np.asarray(to_cpu(work_x), dtype=float)
        offspring_velocities = np.asarray(to_cpu(work_v), dtype=float)
        offspring_objectives = self._evaluate(work_x)

        candidate_positions = np.vstack([parent_positions, offspring_positions])
        candidate_velocities = np.vstack([parent_velocities, offspring_velocities])
        candidate_objectives = np.vstack([self.objectives, offspring_objectives])
        return candidate_positions, candidate_objectives, candidate_velocities

    # ------------------------------------------------------------------
    # Reference points / history / run
    # ------------------------------------------------------------------
    def _update_reference_points(self, objective_pool):
        f_min = to_minimization(np.asarray(objective_pool, dtype=float), self.maximize)
        cur_min = np.min(f_min, axis=0)
        cur_max = np.max(f_min, axis=0)
        if self.z is None:
            self.z = cur_min.copy()
            self.z_star = cur_max.copy()
        else:
            self.z = np.minimum(self.z, cur_min)
            self.z_star = np.maximum(self.z_star, cur_max)

    def _save_history(self):
        if self.archive_objectives is None or len(self.archive_positions) == 0:
            return
        rep = representative_index(self.archive_objectives, self.maximize)
        self.history["best_fitness"].append(
            best_per_objective(self.archive_objectives, self.maximize).copy()
        )
        self.history["best_position"].append(self.archive_positions[rep].copy())
        self.history["archive_size"].append(int(len(self.archive_positions)))
        self.history["counts"].append(self.count.copy())

    def run(self, verbose=True):
        self.objectives = self._evaluate(self.positions)
        n_obj = self.objectives.shape[1]
        self._initialize_directions(n_obj)
        self._update_reference_points(self.objectives)

        # Establish one representative particle per subspace before selection.
        x0 = np.asarray(to_cpu(self.positions), dtype=float)
        v0 = np.asarray(to_cpu(self.velocities), dtype=float)
        x0, f0, v0 = self._population_update(x0, self.objectives, v0)
        self.positions = xp.asarray(x0)
        self.objectives = f0
        self.velocities = xp.asarray(v0)
        self._archive_update(x0, f0)
        iterator = tqdm(
            range(1, self.max_iteration + 1),
            desc="Optimizing with MOPSO/DMS",
            ncols=100,
            disable=not verbose,
        )

        for t in iterator:
            has_nd = self._update_counts_and_probabilities()
            cand_x, cand_f, cand_v = self._generate_offspring(t, has_nd)

            # Algorithm 5 updates Z/Z* from E U POP before environmental update.
            if self.archive_objectives is None:
                ref_pool = cand_f
            else:
                ref_pool = np.vstack([self.archive_objectives, cand_f])
            self._update_reference_points(ref_pool)

            self._archive_update(cand_x, cand_f)
            pop_x, pop_f, pop_v = self._population_update(cand_x, cand_f, cand_v)
            self.positions = xp.asarray(pop_x)
            self.objectives = pop_f
            self.velocities = xp.asarray(pop_v)
            self._save_history()

            iterator.set_postfix({"Archive": len(self.archive_positions)})
            if self.progress_callback is not None:
                self.progress_callback(t)

        pop_x = np.asarray(to_cpu(self.positions), dtype=float)
        pop_f = np.asarray(self.objectives, dtype=float)

        # Algorithm 5 uses IGD(POP) vs IGD(E), which requires a known PF S*.
        if self.reference_front is not None:
            pop_igd = igd(pop_f, self.reference_front)
            arc_igd = igd(self.archive_objectives, self.reference_front)
            if pop_igd < arc_igd:
                return pop_x, pop_f

        # Practical generic fallback when the true PF is unknown.
        return self.archive_positions.copy(), self.archive_objectives.copy()


def run_mopso_dms(
    func,
    min_params,
    max_params,
    population_size=100,
    max_iter=300,
    verbose=False,
    progress_callback=None,
    seed=None,
    maximize=False,
    archive_size=None,
    neighbor_size=10,
    t_count=5,
    theta=5.0,
    w_max=0.9,
    w_min=0.4,
    c1=2.0,
    c2=2.0,
    mutation_probability=None,
    eta_m=20.0,
    reference_front=None,
):
    """Wrapper matching the supplied optimizer modules' high-level structure."""
    model = MOPSODMS(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_particles=population_size,
        max_iteration=max_iter,
        archive_size=archive_size,
        maximize=maximize,
        neighbor_size=neighbor_size,
        t_count=t_count,
        theta=theta,
        w_max=w_max,
        w_min=w_min,
        c1=c1,
        c2=c2,
        mutation_probability=mutation_probability,
        eta_m=eta_m,
        reference_front=reference_front,
        progress_callback=progress_callback,
        seed=seed,
    )

    pareto_params, pareto_objectives = model.run(verbose=verbose)
    return pareto_params, pareto_objectives, model.history["best_fitness"]
