import math

from core.utils import sin_angle, itse
from core.physics import VehiclePhysics
from concurrent.futures import ThreadPoolExecutor

SCENARIOS = [
    {"theta_angle": 0,  "vehicle_mass": 1500, "rolling_resistance": 0.06},
    # {"theta_angle": 0,  "vehicle_mass": 1500*1.1, "rolling_resistance": 0.06},
    # {"theta_angle": 0,  "vehicle_mass": 1500*0.9, "rolling_resistance": 0.06},
    # {"theta_angle": 0,  "vehicle_mass": 1500, "rolling_resistance": 0.06*1.2},

    # {"theta_angle": 10, "vehicle_mass": 1500, "rolling_resistance": 0.06},
    # {"theta_angle": 10, "vehicle_mass": 1500*1.1, "rolling_resistance": 0.06},
    # {"theta_angle": 10, "vehicle_mass": 1500*0.9, "rolling_resistance": 0.06},
    # {"theta_angle": 10, "vehicle_mass": 1500, "rolling_resistance": 0.06*1.2},

    # {"theta_angle": -10, "vehicle_mass": 1500, "rolling_resistance": 0.06},
    # {"theta_angle": -10, "vehicle_mass": 1500*1.1, "rolling_resistance": 0.06},
    # {"theta_angle": -10, "vehicle_mass": 1500*0.9, "rolling_resistance": 0.06},
    # {"theta_angle": -10, "vehicle_mass": 1500, "rolling_resistance": 0.06*1.2},
]

def compute_ttc(d_actual, v_ego, v_lead, eps=1e-9):
    if d_actual <= 0:
        return 0.0
    dv = v_ego - v_lead
    if dv <= eps:
        return math.inf
    return d_actual / dv

def error_synthesizer(derr, verr, c_nrg, time, obj_fun):
    w1_pos = 20.6537       
    w2_vel = 1.74043       
    w3_nrg = 0.074934    

    try:
        J = obj_fun(w1_pos, derr, time) + obj_fun(w2_vel, verr, time) #+ obj_fun(w3_nrg, c_nrg, time)
    except TypeError:
        J = obj_fun(w1_pos, derr) + obj_fun(w2_vel, verr)# + obj_fun(w3_nrg, c_nrg)
    return J

def lean_simulate_system(params, obj_fun, return_history=False, ttc_threshold=1.5,
                         sim_time=60, theta_angle=0, vehicle_mass=1500, 
                         rolling_resistance=0.06
                         ):
    """
    obj_fun is now a FUNCTION, not a string.
    It must accept (err, time) or (err) depending on the metric.
    """
    physics = VehiclePhysics(theta=theta_angle, M=vehicle_mass, Croll=rolling_resistance)

    Kve, Kvrel, Kde       = params
    theta                   = 0
    vset                    = 25
    xego, vego, aego        = 0, 22, 0
    xlead, vlead, alead     = 60, 22, 0

    ddef, Tg                = 20, 1.2
    Ts                      = 1 / 60
    simulation_steps        = int(sim_time / Ts)

    obj_val                 = 0.0
    TET                     = 0.0
    TIT                     = 0.0
    ttc_min                 = math.inf
    obj_val                 = 0

    history = {
        "time": [], "v_ego": [], "v_lead": [], "x_ego": [], "x_lead": [],
        "car_force": [], "dist": [], "a_ego": [], "a_lead": [], "err": [], 
        "ttc": [], "obj_val": [], "ttc_min": [], "TET": [], "TIT": [], "d_safe": []
    }

    for i in range(simulation_steps):
        t = i * Ts

        # Compute errors
        dactual             = xlead - xego
        dsafe               = ddef + Tg * vego

        vrel                = vlead - vego
        derr                = dsafe - dactual
        verr                = vset - vego

        # Controller
        u_v                 = Kve * verr
        u_x                 = vrel * Kvrel + Kde * derr

        # LOGIKA SWITCHING YANG LEBIH LOGIS
        u                   = min(u_x, u_v)

        # Update ego car
        vego, aego, Ft      = physics.step(vego, u, Ts)
        xego                += vego * Ts

        # # Update lead car
        theta               = (theta + 30 * Ts) % 360
        alead               = sin_angle(theta)
        vlead               += alead * Ts
        xlead               += vlead * Ts

        c_nrg               = max(0, Ft)
        err                 = error_synthesizer(derr, verr, 0, t, obj_fun)
        obj_val             += err

        # TTC Calculation
        ttc = compute_ttc(dactual, vego, vlead)
        if ttc < ttc_min:
            ttc_min = ttc
        if ttc < ttc_threshold:
            TET += Ts
            TIT += (ttc_threshold - ttc) * Ts

        if return_history:
            history["time"].append(t)
            history["v_ego"].append(vego)
            history["v_lead"].append(vlead)
            history["x_ego"].append(xego)
            history["x_lead"].append(xlead)
            history["car_force"].append(Ft)
            history["dist"].append(dactual)
            history["d_safe"].append(dsafe)
            history["a_ego"].append(aego)
            history["a_lead"].append(alead)
            history["err"].append(err)
            history["ttc"].append(ttc)

        history["TET"] = TET
        history["TIT"] = TIT
        history["ttc_min"] = ttc_min if ttc_min != math.inf else None
        

    return [obj_val, history] if return_history else obj_val

def acc_single_scenario(params, obj_function, scenario):
    return lean_simulate_system(
        params,
        obj_function,
        theta_angle=scenario["theta_angle"],
        vehicle_mass=scenario["vehicle_mass"],
        rolling_resistance=scenario["rolling_resistance"]
    )

def acc_fitness_func(params, obj_function):
    # THREAD POOL: sangat ringan overheadnya
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(acc_single_scenario, params, obj_function, scen)
            for scen in SCENARIOS
        ]

        # kembalikan jumlah total objective dari 12 kondisi
        return sum(f.result() for f in futures)

BENCHMARKS_ACC = {
    "acc": acc_fitness_func,
}