"""Compact single-objective PSO used only for v2 cost-profile verification.

This is intentionally separate from the six algorithms in the final benchmark.
Its default configuration is supplied by the v2 manifest and mirrors the
Sahib-Ahmed verification protocol as closely as practical for this codebase.
"""

from __future__ import annotations

import numpy as np


def run_pso(
    func,
    min_params,
    max_params,
    population_size=30,
    max_iter=50,
    c1=2.0,
    c2=2.0,
    inertia_start=0.9,
    inertia_end=0.2,
    verbose=False,
    progress_callback=None,
    seed=None,
):
    rng = np.random.default_rng(seed)
    lo = np.asarray(min_params, dtype=float)
    hi = np.asarray(max_params, dtype=float)
    dim = lo.size
    if dim < 1 or hi.shape != lo.shape or np.any(hi <= lo):
        raise ValueError("invalid PSO bounds")
    n = int(population_size)
    if n < 2:
        raise ValueError("PSO population_size must be >= 2")

    span = hi - lo
    pos = rng.uniform(lo, hi, size=(n, dim))
    vel = rng.uniform(-0.5 * span, 0.5 * span, size=(n, dim))
    fit = np.asarray([float(func(x)) for x in pos], dtype=float)
    pbest_pos = pos.copy()
    pbest_fit = fit.copy()
    g_idx = int(np.argmin(fit))
    g_pos = pos[g_idx].copy()
    g_fit = float(fit[g_idx])
    curve = []

    for it in range(int(max_iter)):
        if max_iter <= 1:
            inertia = float(inertia_end)
        else:
            frac = it / float(max_iter - 1)
            inertia = float(inertia_start + frac * (inertia_end - inertia_start))
        r1 = rng.random(size=(n, dim))
        r2 = rng.random(size=(n, dim))
        vel = (
            inertia * vel
            + float(c1) * r1 * (pbest_pos - pos)
            + float(c2) * r2 * (g_pos - pos)
        )
        # Proportional adaptation of Sahib-style velocity clamp: half the range.
        vel = np.clip(vel, -0.5 * span, 0.5 * span)
        pos = np.clip(pos + vel, lo, hi)
        fit = np.asarray([float(func(x)) for x in pos], dtype=float)

        improved = fit < pbest_fit
        pbest_pos[improved] = pos[improved]
        pbest_fit[improved] = fit[improved]
        idx = int(np.argmin(pbest_fit))
        if float(pbest_fit[idx]) < g_fit:
            g_fit = float(pbest_fit[idx])
            g_pos = pbest_pos[idx].copy()
        curve.append(g_fit)
        if progress_callback is not None:
            progress_callback(it + 1)

    return g_pos.tolist(), g_fit, curve
