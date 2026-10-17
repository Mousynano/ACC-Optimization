"""Pareto-dominance MOPSO with an external archive and adaptive objective grid.

The implementation follows the Coello-style design at the level required by the
v2 calibration task: Pareto dominance, an external nondominated repository,
grid-biased leader selection, crowded-cell deletion, and mutation. It is used
only to approximate the calibration Pareto front, not as a competitor in the
final six-algorithm benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

import numpy as np


@dataclass
class MOEval:
    values: np.ndarray
    feasible: bool = True
    violation: float = 0.0


@dataclass
class ArchiveItem:
    position: np.ndarray
    evaluation: MOEval


def _coerce_eval(result) -> MOEval:
    if isinstance(result, MOEval):
        return result
    if isinstance(result, tuple) and len(result) == 3:
        values, feasible, violation = result
        return MOEval(np.asarray(values, dtype=float), bool(feasible), float(violation))
    return MOEval(np.asarray(result, dtype=float), True, 0.0)


def constraint_dominates(a: MOEval, b: MOEval) -> bool:
    if a.feasible and not b.feasible:
        return True
    if b.feasible and not a.feasible:
        return False
    if not a.feasible and not b.feasible:
        return a.violation < b.violation
    av, bv = a.values, b.values
    return bool(np.all(av <= bv) and np.any(av < bv))


def nondominated(items: Sequence[ArchiveItem]) -> List[ArchiveItem]:
    out = []
    for i, item in enumerate(items):
        dominated = False
        for j, other in enumerate(items):
            if i != j and constraint_dominates(other.evaluation, item.evaluation):
                dominated = True
                break
        if not dominated:
            out.append(item)
    # Exact/near-exact duplicates add no information to Sahib means.
    unique = []
    for item in out:
        if not any(
            np.allclose(item.position, u.position, rtol=0.0, atol=1e-12)
            and np.allclose(item.evaluation.values, u.evaluation.values, rtol=0.0, atol=1e-12)
            for u in unique
        ):
            unique.append(item)
    return unique


def _grid_cells(archive: Sequence[ArchiveItem], n_grid: int, inflation: float):
    vals = np.asarray([x.evaluation.values for x in archive], dtype=float)
    mins = vals.min(axis=0)
    maxs = vals.max(axis=0)
    span = maxs - mins
    span[span <= 1e-15] = 1.0
    mins = mins - inflation * span
    maxs = maxs + inflation * span
    width = (maxs - mins) / float(n_grid)
    width[width <= 1e-15] = 1.0
    cells = []
    for row in vals:
        idx = np.floor((row - mins) / width).astype(int)
        idx = np.clip(idx, 0, n_grid - 1)
        cells.append(tuple(int(x) for x in idx))
    return cells


def _roulette(rng, weights):
    w = np.asarray(weights, dtype=float)
    total = float(w.sum())
    if not np.isfinite(total) or total <= 0.0:
        return int(rng.integers(0, len(w)))
    r = rng.random() * total
    c = 0.0
    for i, value in enumerate(w):
        c += float(value)
        if r <= c:
            return i
    return len(w) - 1


def _select_leader(rng, archive, n_grid, inflation, pressure):
    cells = _grid_cells(archive, n_grid, inflation)
    counts = {}
    for c in cells:
        counts[c] = counts.get(c, 0) + 1
    unique_cells = list(counts)
    cell_weights = [1.0 / (counts[c] ** pressure) for c in unique_cells]
    chosen_cell = unique_cells[_roulette(rng, cell_weights)]
    candidates = [i for i, c in enumerate(cells) if c == chosen_cell]
    return archive[int(rng.choice(candidates))]


def _truncate_archive(rng, archive, max_size, n_grid, inflation, pressure):
    archive = list(archive)
    while len(archive) > max_size:
        cells = _grid_cells(archive, n_grid, inflation)
        counts = {}
        for c in cells:
            counts[c] = counts.get(c, 0) + 1
        unique_cells = list(counts)
        # Prefer deleting from crowded cells.
        weights = [counts[c] ** pressure for c in unique_cells]
        chosen_cell = unique_cells[_roulette(rng, weights)]
        candidates = [i for i, c in enumerate(cells) if c == chosen_cell]
        del archive[int(rng.choice(candidates))]
    return archive


def run_mopso(
    objective_func: Callable,
    min_params,
    max_params,
    population_size=100,
    max_iter=100,
    archive_size=100,
    grid_divisions=30,
    grid_inflation=0.1,
    inertia=0.5,
    inertia_damping=0.99,
    c1=1.0,
    c2=2.0,
    leader_pressure=2.0,
    deletion_pressure=2.0,
    mutation_rate=0.1,
    seed=None,
    progress_callback=None,
):
    rng = np.random.default_rng(seed)
    lo = np.asarray(min_params, dtype=float)
    hi = np.asarray(max_params, dtype=float)
    span = hi - lo
    n = int(population_size)
    dim = lo.size
    if n < 2 or dim < 1 or np.any(span <= 0.0):
        raise ValueError("invalid MOPSO population or bounds")

    pos = rng.uniform(lo, hi, size=(n, dim))
    vel = np.zeros((n, dim), dtype=float)
    evals = [_coerce_eval(objective_func(x)) for x in pos]
    pbest_pos = pos.copy()
    pbest_eval = [MOEval(e.values.copy(), e.feasible, e.violation) for e in evals]
    archive = nondominated([ArchiveItem(pos[i].copy(), evals[i]) for i in range(n)])
    archive = _truncate_archive(
        rng, archive, int(archive_size), int(grid_divisions), float(grid_inflation), float(deletion_pressure)
    )

    w = float(inertia)
    for it in range(int(max_iter)):
        if not archive:
            raise RuntimeError("MOPSO archive became empty")
        for i in range(n):
            leader = _select_leader(
                rng, archive, int(grid_divisions), float(grid_inflation), float(leader_pressure)
            )
            r1 = rng.random(dim)
            r2 = rng.random(dim)
            vel[i] = (
                w * vel[i]
                + float(c1) * r1 * (pbest_pos[i] - pos[i])
                + float(c2) * r2 * (leader.position - pos[i])
            )
            vel[i] = np.clip(vel[i], -span, span)
            pos[i] = np.clip(pos[i] + vel[i], lo, hi)

            # Coello-style exploratory mutation, decaying through the run.
            pm = float(mutation_rate) * max(0.0, 1.0 - it / max(float(max_iter), 1.0))
            if rng.random() < pm:
                j = int(rng.integers(0, dim))
                radius = span[j] * max(0.02, 0.5 * (1.0 - it / max(float(max_iter), 1.0)))
                pos[i, j] = np.clip(pos[i, j] + rng.uniform(-radius, radius), lo[j], hi[j])

            new_eval = _coerce_eval(objective_func(pos[i]))
            evals[i] = new_eval
            old_eval = pbest_eval[i]
            if constraint_dominates(new_eval, old_eval):
                pbest_pos[i] = pos[i].copy()
                pbest_eval[i] = MOEval(new_eval.values.copy(), new_eval.feasible, new_eval.violation)
            elif not constraint_dominates(old_eval, new_eval) and rng.random() < 0.5:
                pbest_pos[i] = pos[i].copy()
                pbest_eval[i] = MOEval(new_eval.values.copy(), new_eval.feasible, new_eval.violation)

        candidates = archive + [ArchiveItem(pos[i].copy(), evals[i]) for i in range(n)]
        archive = nondominated(candidates)
        archive = _truncate_archive(
            rng, archive, int(archive_size), int(grid_divisions), float(grid_inflation), float(deletion_pressure)
        )
        w *= float(inertia_damping)
        if progress_callback is not None:
            progress_callback(it + 1)

    return archive


def merge_reference_front(fronts: Sequence[Sequence[ArchiveItem]]) -> List[ArchiveItem]:
    merged = []
    for front in fronts:
        merged.extend(front)
    return nondominated(merged)
