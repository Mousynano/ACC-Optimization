#!/usr/bin/env python3
"""Plot active-framework cruise/following motion for one checkpoint.

Normal use requires frozen scenarios. ``--development`` renders the explicitly
marked smoke-test scenarios and is not a research result.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from core.config import load_manifest
from core.scenarios_v2 import load_scenario_suite

_ACTIVE_MANIFEST = load_manifest()
if _ACTIVE_MANIFEST["benchmark"] == "acc_cost_sahib_fje_v2":
    from benchmarks.acc_cost_fje_v2 import simulate_cruise_history, simulate_following_history
else:
    from benchmarks.acc_cost_v2 import simulate_cruise_history, simulate_following_history


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--development", action="store_true")
    p.add_argument("--out-dir", type=Path, default=Path("motion_plots_v2"))
    p.add_argument("--dpi", type=int, default=250)
    return p.parse_args()


def load_gains(path):
    with path.open("rb") as fh:
        data=pickle.load(fh)
    gains=np.asarray(data["best_solution"], dtype=float)
    if gains.shape != (3,):
        raise ValueError("ACC checkpoint best_solution must be [k_ve,k_de,k_vrel]")
    return data, gains


def main():
    args=parse_args()
    manifest=load_manifest()
    suite=load_scenario_suite(
        manifest["cost_framework"]["scenario_manifest_path"],
        development=args.development,
        require_frozen=not args.development,
    )
    data,gains=load_gains(args.checkpoint)
    args.out_dir.mkdir(parents=True,exist_ok=True)
    tag=args.checkpoint.stem

    cruise=suite.cruise[0]
    hist=simulate_cruise_history(gains,cruise)
    t=np.asarray(hist["time"]); v=np.asarray(hist["v_ego"]); vs=np.asarray(hist["v_set"])
    a=np.asarray(hist["a_ego"]); j=np.asarray(hist["jerk"])
    fig,ax=plt.subplots(3,1,figsize=(10,9),sharex=True)
    ax[0].plot(t,vs,"--",label="User set speed"); ax[0].plot(t,v,label="Ego")
    ax[1].plot(t,a,label="Realized acceleration")
    ax[2].plot(t,j,label="Realized jerk")
    ax[0].set_ylabel("Speed (m/s)"); ax[1].set_ylabel("Acceleration (m/s²)"); ax[2].set_ylabel("Jerk (m/s³)"); ax[2].set_xlabel("Time (s)")
    ax[0].set_title(f"{manifest['framework_version']} cruise | k_ve={gains[0]:.4g}, k_de={gains[1]:.4g}, k_vrel={gains[2]:.4g}")
    for x in ax: x.grid(alpha=.25); x.legend()
    fig.tight_layout(); fig.savefig(args.out_dir/f"cruise_v2_{tag}.png",dpi=args.dpi,bbox_inches="tight"); plt.close(fig)

    follow=suite.following[0]
    hist=simulate_following_history(gains,follow)
    t=np.asarray(hist["time"]); ve=np.asarray(hist["v_ego"]); vl=np.asarray(hist["v_lead"])
    d=np.asarray(hist["dist"]); ds=np.asarray(hist["d_safe"]); a=np.asarray(hist["a_ego"]); j=np.asarray(hist["jerk"])
    fig,ax=plt.subplots(4,1,figsize=(10,11),sharex=True)
    ax[0].plot(t,d,label="Actual spacing"); ax[0].plot(t,ds,":",label="CTG desired spacing")
    ax[1].plot(t,vl,"--",label="Lead"); ax[1].plot(t,ve,label="Ego")
    ax[2].plot(t,a,label="Ego acceleration"); ax[3].plot(t,j,label="Ego jerk")
    ax[0].set_ylabel("Spacing (m)"); ax[1].set_ylabel("Speed (m/s)"); ax[2].set_ylabel("Acceleration (m/s²)"); ax[3].set_ylabel("Jerk (m/s³)"); ax[3].set_xlabel("Time (s)")
    ax[0].set_title(f"{manifest['framework_version']} following | {follow.scenario_id}")
    for x in ax: x.grid(alpha=.25); x.legend()
    fig.tight_layout(); fig.savefig(args.out_dir/f"following_v2_{tag}.png",dpi=args.dpi,bbox_inches="tight"); plt.close(fig)

    print("Wrote:", args.out_dir/f"cruise_v2_{tag}.png")
    print("Wrote:", args.out_dir/f"following_v2_{tag}.png")


if __name__ == "__main__":
    main()
