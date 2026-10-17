"""MOPSO reconstructed from Suleiman et al. (2025), JKSUES 37:52.

Paper-specific pieces implemented:
- decision vector X = [Power, Bandwidth, Distance] is represented generically by bounds;
- standard PSO velocity/position update (Eqs. 29-30);
- Pareto non-dominated sorting and bounded external archive;
- defaults from Table 1: w=0.9, c1=c2=0.2, archive size=100, 50 iterations.

Important reproduction note
---------------------------
The paper states that non-dominated solutions are stored in an archive, but it does
not specify how an overflowing archive is pruned or how one gbest is chosen from a
multi-objective archive.  This implementation uses crowding-distance pruning and
uniform random leader selection.  Those two details are explicitly implementation
choices, not claims about undocumented paper code.

The paper's Table 1 also lists both "Number of particles = 50" and
"Population size = 500" without explaining a distinction.  Here ``n_particles``
is the actual swarm size and defaults to 50.
"""

from __future__ import annotations

import numpy as np
from tqdm import tqdm

from core.xp import xp, to_cpu, GPU_AVAILABLE
# from core.seeding import seed_everything

from mopso_common import (
    best_per_objective,
    dominates,
    evaluate_population,
    non_dominated_union,
    representative_index,
    seed_backends,
    to_minimization,
    truncate_by_crowding,
)


class SmartGridMOPSO:
    def __init__(
        self,
        fitness_function,
        min_params,
        max_params,
        n_particles=50,
        max_iteration=50,
        archive_size=100,
        maximize=True,
        w=0.9,
        c1=0.2,
        c2=0.2,
        progress_callback=None,
        seed=None,
    ):
        self.fitness_function = fitness_function
        self.min_params = xp.asarray(min_params, dtype=float)
        self.max_params = xp.asarray(max_params, dtype=float)
        self.dim = len(min_params)
        self.n_particles = int(n_particles)
        self.max_iteration = int(max_iteration)
        self.archive_size = int(archive_size)
        self.maximize = maximize

        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)
        self.progress_callback = progress_callback
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        seed_backends(seed)

        self.positions = self._initialize_positions()
        self.velocities = self._initialize_velocities()
        self.objectives = None
        self.pbest_positions = self.positions.copy()
        self.pbest_objectives = None

        # Archive is kept on CPU: dominance/crowding are branch-heavy and small.
        self.archive_positions = np.empty((0, self.dim), dtype=float)
        self.archive_objectives = None

        # Same structural idea as the user's PSO/HOA history, but "best_fitness"
        # is a vector because a MOP has no single global-best scalar.
        self.history = {
            "best_fitness": [],
            "best_position": [],
            "archive_size": [],
        }

    def _initialize_positions(self):
        return xp.random.uniform(
            low=self.min_params,
            high=self.max_params,
            size=(self.n_particles, self.dim),
        )

    def _initialize_velocities(self):
        # The paper says velocity is randomly initialized but gives no range.
        # Use the decision-space span as the natural scale.
        span = self.max_params - self.min_params
        return xp.random.uniform(-1.0, 1.0, size=(self.n_particles, self.dim)) * span

    def _evaluate(self, positions):
        return evaluate_population(self.fitness_function, positions)

    def _update_archive(self, positions, objectives):
        x_new = np.asarray(to_cpu(positions), dtype=float)
        f_new = np.asarray(objectives, dtype=float)

        if self.archive_objectives is None or len(self.archive_positions) == 0:
            x_all, f_all = x_new, f_new
        else:
            x_all = np.vstack([self.archive_positions, x_new])
            f_all = np.vstack([self.archive_objectives, f_new])

        x_nd, f_nd = non_dominated_union(x_all, f_all, self.maximize)
        if len(x_nd) > self.archive_size:
            x_nd, f_nd = truncate_by_crowding(x_nd, f_nd, self.archive_size)

        self.archive_positions = x_nd
        self.archive_objectives = f_nd

    def _select_leader(self):
        if len(self.archive_positions) == 0:
            idx = int(self.rng.integers(self.n_particles))
            return self.positions[idx].copy()
        idx = int(self.rng.integers(len(self.archive_positions)))
        return xp.asarray(self.archive_positions[idx])

    def _move_particles(self):
        for i in range(self.n_particles):
            gbest = self._select_leader()
            r1 = float(self.rng.random())
            r2 = float(self.rng.random())

            cognitive = self.c1 * r1 * (self.pbest_positions[i] - self.positions[i])
            social = self.c2 * r2 * (gbest - self.positions[i])
            self.velocities[i] = self.w * self.velocities[i] + cognitive + social
            self.positions[i] = xp.clip(
                self.positions[i] + self.velocities[i],
                self.min_params,
                self.max_params,
            )

    def _update_pbest(self, new_objectives):
        current_min = to_minimization(new_objectives, self.maximize)
        pbest_min = to_minimization(self.pbest_objectives, self.maximize)

        for i in range(self.n_particles):
            if dominates(current_min[i], pbest_min[i]):
                take_current = True
            elif dominates(pbest_min[i], current_min[i]):
                take_current = False
            else:
                # Standard neutral handling for mutually non-dominating alternatives.
                take_current = bool(self.rng.random() < 0.5)

            if take_current:
                self.pbest_positions[i] = self.positions[i].copy()
                self.pbest_objectives[i] = new_objectives[i].copy()

    def _save_history(self):
        if len(self.archive_positions) == 0:
            return
        rep = representative_index(self.archive_objectives, self.maximize)
        self.history["best_fitness"].append(
            best_per_objective(self.archive_objectives, self.maximize).copy()
        )
        self.history["best_position"].append(self.archive_positions[rep].copy())
        self.history["archive_size"].append(int(len(self.archive_positions)))

    def run(self, verbose=True):
        # Initial objective evaluation and Pareto archive.
        self.objectives = self._evaluate(self.positions)
        self.pbest_objectives = self.objectives.copy()
        self._update_archive(self.positions, self.objectives)
        iterator = tqdm(
            range(self.max_iteration),
            desc="Optimizing with SmartGrid-MOPSO",
            ncols=100,
            disable=not verbose,
        )

        for iteration in iterator:
            self._move_particles()
            self.objectives = self._evaluate(self.positions)
            self._update_pbest(self.objectives)
            self._update_archive(self.positions, self.objectives)
            self._save_history()

            iterator.set_postfix({"Archive": len(self.archive_positions)})
            if self.progress_callback is not None:
                self.progress_callback(iteration + 1)

        return self.archive_positions.copy(), self.archive_objectives.copy()


def run_smartgrid_mopso(
    func,
    min_params,
    max_params,
    population_size=50,
    max_iter=50,
    verbose=False,
    progress_callback=None,
    seed=None,
    maximize=True,
    archive_size=100,
    w=0.9,
    c1=0.2,
    c2=0.2,
):
    """Wrapper with the same overall call style as ``run_pso`` / ``run_hoa``.

    Returns
    -------
    pareto_params : ndarray, shape (n_archive, n_variables)
    pareto_objectives : ndarray, shape (n_archive, n_objectives)
    history_best_fitness : list[ndarray]
        Component-wise ideal objective vector at each saved iteration.
    """
    model = SmartGridMOPSO(
        fitness_function=func,
        min_params=min_params,
        max_params=max_params,
        n_particles=population_size,
        max_iteration=max_iter,
        archive_size=archive_size,
        maximize=maximize,
        w=w,
        c1=c1,
        c2=c2,
        progress_callback=progress_callback,
        seed=seed,
    )

    pareto_params, pareto_objectives = model.run(verbose=verbose)
    return pareto_params, pareto_objectives, model.history["best_fitness"]
