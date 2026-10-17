"""Shared utilities for the MOPSO implementations in this bundle.

The project code supplied by the user keeps particle state on ``xp`` (NumPy/CuPy)
while evaluating the objective on CPU.  These helpers follow the same pattern.
Multi-objective bookkeeping is intentionally performed with NumPy because archive
sizes are modest and dominance / crowding operations are branch-heavy.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from core.xp import xp, to_cpu


def as_cpu_1d(value) -> np.ndarray:
    arr = np.asarray(to_cpu(value), dtype=float).reshape(-1)
    return arr


def evaluate_position(fitness_function, position_xp) -> np.ndarray:
    """Evaluate one particle on CPU and return a 1-D objective vector."""
    position_cpu = np.asarray(to_cpu(position_xp), dtype=float)
    value = np.asarray(fitness_function(position_cpu), dtype=float).reshape(-1)
    if value.size == 0:
        raise ValueError("fitness_function must return at least one objective value")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"fitness_function returned a non-finite objective vector: {value}")
    return value


def evaluate_population(fitness_function, positions_xp) -> np.ndarray:
    values = [evaluate_position(fitness_function, positions_xp[i]) for i in range(len(positions_xp))]
    widths = {v.size for v in values}
    if len(widths) != 1:
        raise ValueError("fitness_function returned objective vectors with inconsistent lengths")
    return np.vstack(values)


def objective_signs(maximize, n_objectives: int) -> np.ndarray:
    """Return multipliers that convert all objectives to minimization form."""
    if isinstance(maximize, (bool, np.bool_)):
        flags = np.full(n_objectives, bool(maximize), dtype=bool)
    else:
        flags = np.asarray(list(maximize), dtype=bool).reshape(-1)
        if flags.size != n_objectives:
            raise ValueError(
                f"maximize has {flags.size} entries but objective function returns {n_objectives} objectives"
            )
    # maximize objective f <=> minimize -f
    return np.where(flags, -1.0, 1.0)


def to_minimization(raw_objectives: np.ndarray, maximize) -> np.ndarray:
    raw = np.asarray(raw_objectives, dtype=float)
    one_dim = raw.ndim == 1
    if one_dim:
        raw = raw.reshape(1, -1)
    signs = objective_signs(maximize, raw.shape[1])
    out = raw * signs
    return out[0] if one_dim else out


def dominates(a_min: np.ndarray, b_min: np.ndarray, atol: float = 1e-12) -> bool:
    """Pareto dominance for minimization-form objective vectors."""
    a = np.asarray(a_min, dtype=float)
    b = np.asarray(b_min, dtype=float)
    return bool(np.all(a <= b + atol) and np.any(a < b - atol))


def non_dominated_mask(objectives_min: np.ndarray, atol: float = 1e-12) -> np.ndarray:
    f = np.asarray(objectives_min, dtype=float)
    n = len(f)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if dominates(f[j], f[i], atol=atol):
                keep[i] = False
                break
    return keep


def crowding_distance(objectives: np.ndarray) -> np.ndarray:
    """NSGA-II style crowding distance. Larger means more isolated / diverse."""
    f = np.asarray(objectives, dtype=float)
    n = len(f)
    if n == 0:
        return np.empty(0, dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)

    m = f.shape[1]
    d = np.zeros(n, dtype=float)
    for j in range(m):
        order = np.argsort(f[:, j], kind="mergesort")
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        lo = f[order[0], j]
        hi = f[order[-1], j]
        span = hi - lo
        if abs(span) <= 1e-15:
            continue
        for k in range(1, n - 1):
            if np.isinf(d[order[k]]):
                continue
            d[order[k]] += (f[order[k + 1], j] - f[order[k - 1], j]) / span
    return d


def unique_solution_indices(
    positions: np.ndarray,
    objectives: np.ndarray,
    decimals: int = 12,
) -> np.ndarray:
    """Keep first occurrence of duplicate decision+objective pairs."""
    x = np.asarray(positions, dtype=float)
    f = np.asarray(objectives, dtype=float)
    if len(x) == 0:
        return np.empty(0, dtype=int)
    z = np.hstack([x, f])
    z = np.round(z, decimals=decimals)
    _, idx = np.unique(z, axis=0, return_index=True)
    return np.sort(idx)


def non_dominated_union(
    positions: np.ndarray,
    raw_objectives: np.ndarray,
    maximize,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(positions, dtype=float)
    f = np.asarray(raw_objectives, dtype=float)
    if len(x) == 0:
        return x.reshape(0, x.shape[-1] if x.ndim == 2 else 0), f.reshape(0, f.shape[-1] if f.ndim == 2 else 0)
    idx = unique_solution_indices(x, f)
    x = x[idx]
    f = f[idx]
    mask = non_dominated_mask(to_minimization(f, maximize))
    return x[mask], f[mask]


def truncate_by_crowding(
    positions: np.ndarray,
    raw_objectives: np.ndarray,
    capacity: int,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(positions, dtype=float).copy()
    f = np.asarray(raw_objectives, dtype=float).copy()
    capacity = int(capacity)
    while len(x) > capacity:
        cd = crowding_distance(f)
        # Remove one of the most crowded points. Stable argmin keeps behavior deterministic.
        finite = np.where(np.isfinite(cd), cd, np.inf)
        remove = int(np.argmin(finite))
        if np.isinf(finite[remove]):
            # Degenerate archive (e.g. identical objective values): remove the last point.
            remove = len(x) - 1
        x = np.delete(x, remove, axis=0)
        f = np.delete(f, remove, axis=0)
    return x, f


def normalized_ideal_distance(objectives_min: np.ndarray) -> np.ndarray:
    """Distance to the ideal point after per-objective normalization."""
    f = np.asarray(objectives_min, dtype=float)
    if len(f) == 0:
        return np.empty(0, dtype=float)
    lo = np.min(f, axis=0)
    hi = np.max(f, axis=0)
    span = np.where((hi - lo) > 1e-15, hi - lo, 1.0)
    z = (f - lo) / span
    return np.linalg.norm(z, axis=1)


def representative_index(raw_objectives: np.ndarray, maximize) -> int:
    """Deterministic compromise representative; not a replacement for the Pareto set."""
    fmin = to_minimization(raw_objectives, maximize)
    return int(np.argmin(normalized_ideal_distance(fmin)))


def best_per_objective(raw_objectives: np.ndarray, maximize) -> np.ndarray:
    raw = np.asarray(raw_objectives, dtype=float)
    flags = np.asarray(objective_signs(maximize, raw.shape[1]) < 0, dtype=bool)
    out = np.empty(raw.shape[1], dtype=float)
    for j, is_max in enumerate(flags):
        out[j] = np.max(raw[:, j]) if is_max else np.min(raw[:, j])
    return out


def igd(solution_objectives: np.ndarray, reference_front: np.ndarray) -> float:
    """Inverted generational distance: mean distance from PF reference points to set S."""
    s = np.asarray(solution_objectives, dtype=float)
    ref = np.asarray(reference_front, dtype=float)
    if len(s) == 0:
        return float("inf")
    distances = np.linalg.norm(ref[:, None, :] - s[None, :, :], axis=2)
    return float(np.mean(np.min(distances, axis=1)))


def angle_between(a: np.ndarray, b: np.ndarray, eps: float = 1e-15) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= eps or nb <= eps:
        return 0.0
    c = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return float(np.arccos(c))


def seed_backends(seed):
    if seed is None:
        return
    np.random.seed(seed)
    try:
        xp.random.seed(seed)
    except Exception:
        pass
