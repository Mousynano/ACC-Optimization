#!/usr/bin/env python3
"""Plot cruise-control and car-following motion from one ACC checkpoint.

Place this script in the repository root, or run it with --project-root pointing
at the active ACC repository.

Outputs (two figures per case):
  1) cruise_motion_overview_<tag>.png
  2) cruise_motion_diagnostic_<tag>.png
  3) car_following_motion_overview_<tag>.png
  4) car_following_motion_diagnostic_<tag>.png

The overview figures follow the manuscript-style motion plots. The diagnostic
figures expose commanded acceleration, realized acceleration, and jerk so that
sample-to-sample chatter is easy to inspect.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path, help="Checkpoint .pkl file")
    p.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing benchmarks/ and core/ (default: cwd)",
    )
    p.add_argument(
        "--objective",
        choices=("iae", "ise", "itae", "itse"),
        default=None,
        help="Objective used for the checkpoint. Inferred from filename if omitted.",
    )
    p.add_argument("--out-dir", type=Path, default=Path("motion_plots"))
    p.add_argument("--zoom-start", type=float, default=10.0)
    p.add_argument("--zoom-end", type=float, default=18.0)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def infer_objective(path: Path) -> str:
    name = path.stem.lower()
    # Longest names first so ITAE/ITSE do not get confused with IAE/ISE.
    for obj in ("itae", "itse", "iae", "ise"):
        if name.endswith(f"_{obj}") or f"_{obj}_" in name:
            return obj
    raise ValueError(
        "Could not infer objective from checkpoint filename. "
        "Pass --objective iae|ise|itae|itse explicitly."
    )


def load_checkpoint(path: Path) -> dict:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict) or "best_solution" not in data:
        raise ValueError("Checkpoint must be a dict containing 'best_solution'.")
    gains = np.asarray(data["best_solution"], dtype=float)
    if gains.shape != (3,):
        raise ValueError(f"Expected [k_v, k_p, k_d], got shape {gains.shape}")
    return data


def zoom_mask(time: np.ndarray, start: float, end: float) -> np.ndarray:
    if end <= start:
        raise ValueError("--zoom-end must be greater than --zoom-start")
    mask = (time >= start) & (time <= end)
    if not np.any(mask):
        raise ValueError(
            f"Zoom interval [{start}, {end}] s does not overlap simulation time."
        )
    return mask


def save_cruise_overview(hist: dict, out: Path, title: str, dpi: int) -> None:
    t = np.asarray(hist["time"], dtype=float)
    v = np.asarray(hist["v_ego"], dtype=float)
    v_ref = np.asarray(hist["v_ref"], dtype=float)
    a = np.asarray(hist["a_ego"], dtype=float)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t, v_ref, linestyle="--", label="Reference")
    axes[0].plot(t, v, label="Ego")
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].set_title(title)

    axes[1].plot(t, a, label="Ego acceleration")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Acceleration (m/s²)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_cruise_diagnostic(
    hist: dict,
    out: Path,
    title: str,
    dt: float,
    zoom_start: float,
    zoom_end: float,
    dpi: int,
) -> None:
    t = np.asarray(hist["time"], dtype=float)
    a_cmd = np.asarray(hist["a_cmd"], dtype=float)
    a = np.asarray(hist["a_ego"], dtype=float)
    jerk = np.concatenate(([0.0], np.diff(a) / dt))
    mask = zoom_mask(t, zoom_start, zoom_end)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t[mask], a_cmd[mask], label="Commanded acceleration")
    axes[0].plot(t[mask], a[mask], label="Realized acceleration")
    axes[0].axhline(0.0, linewidth=0.8)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title(title)

    axes[1].plot(t[mask], jerk[mask], label="Jerk")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Jerk (m/s³)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_following_overview(hist: dict, out: Path, title: str, dpi: int) -> None:
    t = np.asarray(hist["time"], dtype=float)
    dist = np.asarray(hist["dist"], dtype=float)
    d_safe = np.asarray(hist["d_safe"], dtype=float)
    v_ego = np.asarray(hist["v_ego"], dtype=float)
    v_lead = np.asarray(hist["v_lead"], dtype=float)
    a_ego = np.asarray(hist["a_ego"], dtype=float)
    a_lead = np.asarray(hist["a_lead"], dtype=float)

    # Same three-panel motion structure used by the manuscript.
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(t, dist, label="Actual separation")
    axes[0].plot(t, d_safe, linestyle=":", label="Desired CTG spacing")
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title(title)

    axes[1].plot(t, v_lead, linestyle="--", label="Lead")
    axes[1].plot(t, v_ego, label="Ego")
    axes[1].set_ylabel("Velocity (m/s)")

    axes[2].plot(t, a_lead, linestyle="--", label="Lead")
    axes[2].plot(t, a_ego, label="Ego")
    axes[2].axhline(0.0, linewidth=0.8)
    axes[2].set_ylabel("Acceleration (m/s²)")
    axes[2].set_xlabel("Time (s)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_following_diagnostic(
    hist: dict,
    out: Path,
    title: str,
    dt: float,
    zoom_start: float,
    zoom_end: float,
    dpi: int,
) -> None:
    t = np.asarray(hist["time"], dtype=float)
    a_cmd = np.asarray(hist["a_cmd"], dtype=float)
    a = np.asarray(hist["a_ego"], dtype=float)
    jerk = np.concatenate(([0.0], np.diff(a) / dt))
    mask = zoom_mask(t, zoom_start, zoom_end)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    axes[0].plot(t[mask], a_cmd[mask], label="Commanded acceleration")
    axes[0].plot(t[mask], a[mask], label="Realized acceleration")
    axes[0].axhline(0.0, linewidth=0.8)
    axes[0].set_ylabel("Acceleration (m/s²)")
    axes[0].set_title(title)

    axes[1].plot(t[mask], jerk[mask], label="Jerk")
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Jerk (m/s³)")

    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root))

    # Import only after repository root is on sys.path.
    from benchmarks.sisken_prastiyanto import (  # noqa: E402
        SIMULATION_DT,
        lean_simulate_system,
    )
    from core.fast_acc import fast_cruise_history  # noqa: E402
    from core.utils import iae, ise, itae, itse  # noqa: E402

    objective_funcs = {
        "iae": iae,
        "ise": ise,
        "itae": itae,
        "itse": itse,
    }

    checkpoint = args.checkpoint.resolve()
    ckpt = load_checkpoint(checkpoint)
    gains = np.asarray(ckpt["best_solution"], dtype=float)
    objective = args.objective or infer_objective(checkpoint)
    obj_fun = objective_funcs[objective]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = checkpoint.stem

    cruise = fast_cruise_history(gains)
    _, following = lean_simulate_system(gains, obj_fun, return_history=True)

    cruise_overview = args.out_dir / f"cruise_motion_overview_{tag}.png"
    cruise_diag = args.out_dir / f"cruise_motion_diagnostic_{tag}.png"
    following_overview = args.out_dir / f"car_following_motion_overview_{tag}.png"
    following_diag = args.out_dir / f"car_following_motion_diagnostic_{tag}.png"

    gains_label = (
        f"k_v={gains[0]:.4f}, k_p={gains[1]:.4f}, k_d={gains[2]:.4f}"
    )

    save_cruise_overview(
        cruise,
        cruise_overview,
        f"Cruise-control motion — {objective.upper()}\n{gains_label}",
        args.dpi,
    )
    save_cruise_diagnostic(
        cruise,
        cruise_diag,
        f"Cruise-control acceleration/jerk — {objective.upper()}\n{gains_label}",
        SIMULATION_DT,
        args.zoom_start,
        args.zoom_end,
        args.dpi,
    )
    save_following_overview(
        following,
        following_overview,
        f"Car-following motion profile — {objective.upper()}\n{gains_label}",
        args.dpi,
    )
    save_following_diagnostic(
        following,
        following_diag,
        f"Car-following acceleration/jerk — {objective.upper()}\n{gains_label}",
        SIMULATION_DT,
        args.zoom_start,
        args.zoom_end,
        args.dpi,
    )

    a_follow = np.asarray(following["a_ego"], dtype=float)
    jerk_follow = np.concatenate(([0.0], np.diff(a_follow) / SIMULATION_DT))
    a_cruise = np.asarray(cruise["a_ego"], dtype=float)
    jerk_cruise = np.concatenate(([0.0], np.diff(a_cruise) / SIMULATION_DT))

    print(f"Checkpoint : {checkpoint}")
    print(f"Objective  : {objective.upper()}")
    print(f"Best value : {ckpt.get('best_fitness', float('nan'))}")
    print(f"Gains      : {gains}")
    print(f"Cruise max |jerk|       : {np.max(np.abs(jerk_cruise)):.6f} m/s^3")
    print(f"Car-follow max |jerk|   : {np.max(np.abs(jerk_follow)):.6f} m/s^3")
    print("Saved:")
    for path in (cruise_overview, cruise_diag, following_overview, following_diag):
        print(f"  {path}")


if __name__ == "__main__":
    main()
