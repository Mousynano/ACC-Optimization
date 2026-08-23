import matplotlib.pyplot as plt
import numpy as np

from benchmarks.sisken_prastiyanto_wltc import (
    use_wltc_scenarios,
    acc_fitness_func,
    acc_constraint_evaluator,
    build_wltc_scenarios,
    acc_single_scenario_history,
)

from benchmarks.benchmark import acc_fopid_fitness_func, acc_constraint_evaluator

from core.utils import itse, cappiello_ise

def plot_simulation(history):
    time = history["time"]
    v_ego = history["v_ego"]
    v_lead = history["v_lead"]
    dist = history["dist"]
    a_ego = history["a_ego"]
    car_force = history["car_force"]

    # Hitung safe distance untuk plot
    Tg = 1.2
    d_def = 20
    v_ego_arr = np.array(v_ego)
    d_safe = d_def + Tg * v_ego_arr

    # Konversi mode ACC menjadi angka untuk plotting

    plt.figure(figsize=(14, 10))

    # ================= SPEED PLOT =================
    plt.subplot(4, 1, 1)
    plt.plot(time, v_ego, label="Ego Speed", linewidth=2)
    plt.plot(time, v_lead, label="Lead Speed", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Ego vs Lead Velocity")
    plt.legend()
    plt.grid(True)

    # ================= DISTANCE PLOT =================
    plt.subplot(4, 1, 2)
    plt.plot(time, dist, label="Actual Distance", linewidth=2)
    plt.plot(time, d_safe, label="Safe Distance", linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Distance (m)")
    plt.title("Distance Keeping (Actual vs Safe)")
    plt.legend()
    plt.grid(True)

    # ================= ACCELERATION + MODE =================
    plt.subplot(4, 1, 3)
    plt.plot(time, a_ego, label="Acceleration (m/s²)", linewidth=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (m/s²)")
    plt.title("Ego Acceleration + ACC Mode")
    plt.legend()
    plt.grid(True)

    # ================= CAR FORCE PLOT =================
    plt.subplot(4, 1, 4)
    plt.plot(time, car_force, label="Car Force (N)", color='orange', linewidth=1.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Car Force (N)")
    plt.title("Car Force")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    scenarios = build_wltc_scenarios(
        csv_path="wltc_class3b.csv",
        dt=1/60,
        mode="train"
    )

    params = [ 4.92235536, 1.41968051, 0.08955723, 0.14250939, 1.81305403, 0.81684026,40.        ]

    obj_val, history = acc_single_scenario_history(
        params=params,
        obj_function=itse,
        scenario=scenarios[0]
    )

    print("Objective value:", obj_val)
    print("TTC min:", history.get("ttc_min"))
    print("Collision:", history.get("collision"))

    plot_simulation(history)