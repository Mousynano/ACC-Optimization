from core.utils import sin_angle
from core.physics import VehiclePhysics

def update_step_response_data(error, car):
    if isinstance(car, dict):
        step_response_result = car['step_response_result']
        time = car['time']
    elif isinstance(car, Car):
        step_response_result = car.step_response_result
        time = car.time
    rise_time = find_rise_time(time, error)
    settling_time = find_settling_time(time, error)

    step_response_result['riseTime'].append(rise_time)
    step_response_result['settlingTime'].append(settling_time)

    overshoot = find_overshoot(error)
    step_response_result['overshoot'].append(overshoot)
    step_response_result['overshootPercentage'].append(
        (overshoot / error[1]) * 100) if error[1] > 0 else 0
    return step_response_result

def find_rise_time(time, error, start_percentile=0.9, end_percentile=0.1):
    start_index = next(
        (i for i, x in enumerate(error) if x <= error[1] * start_percentile),
        None)
    if start_index is None:
        return 0

    end_index = next((i for i, e in enumerate(error)
                      if e >= error[1] * end_percentile and i > start_index),
                     None)
    if end_index is not None and end_index > start_index:
        start_value = time[start_index]
        end_value = time[end_index]
        result = end_value - start_value
        return result
    else:
        return 0

def find_settling_time(time, error):
    error_arr = [abs(e) for e in error]
    settled_index = None
    max_error = max(error_arr)
    if max_error == 0:
        return 0

    for i in range(1, len(error_arr)):
        settled_index = i if error_arr[i] / max_error * 100 >= 0.02 else -1
    return time[settled_index] - time[1]

def find_overshoot(error):
    errorArr = [abs(e) for e in error]
    valleys = []
    overshoot = None

    for i in range(1, len(errorArr)):
        if len(valleys) == 2 or i == len(errorArr) - 1:
            break
        elif (errorArr[i] < errorArr[i - 1] and errorArr[i] < errorArr[i + 1]):
            if i + 2 < len(errorArr):
                if (errorArr[i] < errorArr[i - 1] and errorArr[i] < errorArr[i + 2]):
                    valleys.append(i)
                    continue
            valleys.append(i)
            continue

    if len(valleys) == 2:
        overshoot = max(errorArr[valleys[0]:valleys[1] + 1])

    if overshoot is None:
        overshoot = 0
    return overshoot


SCENARIOS = [
    {"theta_angle": 0,  "vehicle_mass": 1500, "rolling_resistance": 0.06},
    {"theta_angle": 0,  "vehicle_mass": 1500*1.1, "rolling_resistance": 0.06},
    {"theta_angle": 0,  "vehicle_mass": 1500*0.9, "rolling_resistance": 0.06},
    {"theta_angle": 0,  "vehicle_mass": 1500, "rolling_resistance": 0.06*1.2},

    {"theta_angle": 10, "vehicle_mass": 1500, "rolling_resistance": 0.06},
    {"theta_angle": 10, "vehicle_mass": 1500*1.1, "rolling_resistance": 0.06},
    {"theta_angle": 10, "vehicle_mass": 1500*0.9, "rolling_resistance": 0.06},
    {"theta_angle": 10, "vehicle_mass": 1500, "rolling_resistance": 0.06*1.2},

    {"theta_angle": -10, "vehicle_mass": 1500, "rolling_resistance": 0.06},
    {"theta_angle": -10, "vehicle_mass": 1500*1.1, "rolling_resistance": 0.06},
    {"theta_angle": -10, "vehicle_mass": 1500*0.9, "rolling_resistance": 0.06},
    {"theta_angle": -10, "vehicle_mass": 1500, "rolling_resistance": 0.06*1.2},
]


def lean_simulate_system(params, obj_function, return_history=False, 
                         sim_time=60, theta_angle=0, vehicle_mass=1500, 
                         rolling_resistance=0.06):
    """
    obj_function is now a FUNCTION, not a string.
    It must accept (err, time) or (err) depending on the metric.
    """
    physics = VehiclePhysics(theta=theta_angle, M=vehicle_mass, Croll=rolling_resistance)

    verr_gain, vx_gain, xerr_gain = params
    theta = 0
    vset = 25
    xego, vego, aego = 0, 22, 0
    xlead, vlead, alead = 60, 22, 0
    ddef, Tg = 20, 1.2
    Ts = 1 / 60
    obj_val = 0
    simulation_steps = int(sim_time / Ts)

    history = {
        "time": [], "v_ego": [], "v_lead": [], "x_ego": [], "x_lead": [],
        "car_force": [], "dist": [], "a_ego": [], "err": [], "mode": []
    }

    for i in range(simulation_steps):
        t = i * Ts

        # Compute errors
        dactual = xlead - xego
        dsafe   = ddef + Tg * vego
        derr    = dsafe - dactual
        verr    = vset - vego
        err     = derr + verr

        # Controller
        u_v = verr_gain * verr
        u_x = (vlead - vego) * vx_gain + xerr_gain * (dsafe - dactual)
        u = min(u_v, u_x)

        # OBJECTIVE FUNCTION: direct call
        try:
            obj_val += obj_function(err, t)   # ITAE / ITSE require time
        except TypeError:
            obj_val += obj_function(err)      # IAE / ISE have no time param

        # Update ego car
        vego, aego, Fd = physics.step(vego, u, Ts)
        xego += vego * Ts

        # Update lead car
        theta = (theta + 30 * Ts) % 360
        alead = sin_angle(theta)
        vlead += alead * Ts
        xlead += vlead * Ts

        if return_history:
            history["time"].append(t)
            history["v_ego"].append(vego)
            history["v_lead"].append(vlead)
            history["x_ego"].append(xego)
            history["x_lead"].append(xlead)
            history["car_force"].append(Fd)
            history["dist"].append(dactual)
            history["a_ego"].append(aego)
            history["err"].append(err)
            history["mode"].append("N/A")

    return [obj_val, history] if return_history else obj_val

from concurrent.futures import ThreadPoolExecutor

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
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [
            pool.submit(acc_single_scenario, params, obj_function, scen)
            for scen in SCENARIOS
        ]

        # kembalikan jumlah total objective dari 12 kondisi
        return sum(f.result() for f in futures)

BENCHMARKS_ACC = {
    "acc": acc_fitness_func,
}