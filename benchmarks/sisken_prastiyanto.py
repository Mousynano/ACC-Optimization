from core.utils import sin_angle, itse
from core.physics import VehiclePhysics
from concurrent.futures import ThreadPoolExecutor
import numpy as np

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

def ego_sensing(d_actual_ground_truth, v_lead, SENSOR_MAX_RANGE, NOISE_DIST_STD, NOISE_VEL_STD):
    has_target = d_actual_ground_truth < SENSOR_MAX_RANGE

    if has_target:
        has_target = True

        noise_d = 0
        # noise_d = np.random.normal(0, NOISE_DIST_STD)
        d_sensed = d_actual_ground_truth + noise_d

        noise_v = 0
        # noise_v = np.random.normal(0, NOISE_VEL_STD)
        v_lead_sensed = v_lead + noise_v

        d_sensed = max(0.1, d_sensed)

    else:
        has_target = False
        d_sensed = 1000.0
        v_lead_sensed = 0.0

    return has_target, d_sensed, v_lead_sensed

        # if has_target:
        #     # MODE: CAR FOLLOWING (ACC)
        #     # Mobil Ego "melihat" mobil depan dan mencoba menjaga jarak
            
        #     # Hitung Desired Safe Distance (Constant Time Gap)
        #     d_safe = d_def + (1.2 * v_ego) 
            
        #     # Error metrics berdasarkan data SENSOR (bukan data asli)
        #     err_v = v_lead_sensed - v_ego
        #     err_d = d_sensed - d_safe # Positive jika terlalu jauh, negative jika terlalu dekat
            
        #     # Control Law (Simple P-Controller switching)
        #     u_v = Kve * (v_ref - v_ego) # Tetap ingin cepat jika aman
        #     u_d = err_v * Kvrel + Kxe * err_d # Adjustment based on gap
            
        #     u = min(u_v, u_d) # Pilih aksi yang paling "aman" (min acceleration)
        #     current_mode = "ACC (Tracking)"
            
        #     # Cost Calculation Component (Cappiello Definition) [cite: 220, 222]
        #     # e_pos = d_sensed - d_def (Simplified from paper notation)
        #     c_pos = (d_sensed - d_def)**2 
        #     c_vel = (v_lead_sensed - v_ego)**2
            
        # else:
        #     # MODE: FREE FLOW (Cruise Control)
        #     # Sensor buta (jarak > 25m), mobil hanya menjaga kecepatan set
            
        #     err_v = v_ref - v_ego
        #     u = Kve * err_v
        #     current_mode = "Free Flow"
            
        #     # Jika tidak ada target, penalti jarak dianggap 0 untuk metric simulasi
        #     # Tapi penalti kecepatan dihitung terhadap v_ref
        #     c_pos = 0 
        #     c_vel = (v_ref - v_ego)**2

def control_law(Kve, Kvrel, Kxe, v_ref, v_ego, v_lead, v_lead_sensed, d_sensed, d_actual_ground_truth, d_def, has_target):
    if has_target:
        # MODE: CAR FOLLOWING (ACC)
        # Mobil Ego "melihat" mobil depan dan mencoba menjaga jarak
        Tg = 1.2  # Time gap constant
        
        # Hitung Desired Safe Distance (Constant Time Gap)
        d_safe = d_def + (Tg * v_ego) 
        
        # Error metrics berdasarkan data SENSOR (bukan data asli)
        err_v = v_ref - v_ego
        err_d = d_sensed - d_safe # Positive jika terlalu jauh, negative jika terlalu dekat
        err_vrel = v_lead_sensed - v_ego
        
        # Control Law (Simple P-Controller switching)
        u_v = err_v * Kve
        u_d = err_vrel * Kvrel + Kxe * err_d # Adjustment based on gap
        
        u = min(u_v, u_d) # Pilih aksi yang paling "aman" (min acceleration)
        current_mode = "ACC (Tracking)"
        
        # Cost Calculation Component (Cappiello Definition) [cite: 220, 222]
        c_pos = d_actual_ground_truth - d_safe
        c_vel = v_ref - v_ego
        
    else:
        # MODE: FREE FLOW (Cruise Control)
        # Sensor buta (jarak > 25m), mobil hanya menjaga kecepatan set
        
        err_v = v_ref - v_ego
        u = Kve * err_v
        current_mode = "Free Flow"
        
        # Jika tidak ada target, penalti jarak dianggap 0 untuk metric simulasi
        # Tapi penalti kecepatan dihitung terhadap v_ref
        c_pos = 0
        c_vel = v_ref - v_ego
        (v_ref - v_ego)**2

    return u, current_mode, c_pos, c_vel, err_v

def error_synthesizer(derr, verr, c_nrg, time, obj_fun):
    w1_pos = 20.6537       
    w2_vel = 1.74043       
    w3_nrg = 0.074934    

    try:
        J = obj_fun(w1_pos, derr, time) + obj_fun(w2_vel, verr, time) + obj_fun(w3_nrg, c_nrg, time)
    except TypeError:
        J = obj_fun(w1_pos, derr) + obj_fun(w2_vel, verr) + obj_fun(w3_nrg, c_nrg)
    return J

def cappiello_original(derr, verr, c_nrg, time):
    w1_pos = 20.6537       
    w2_vel = 1.74043       
    w3_nrg = 0.074934      

    J = (w1_pos * derr**2) + (w2_vel * verr**2) # + (w3_nrg * c_nrg) the energy term is omitted for now
    return J


# def lean_simulate_system_noisy(params, obj_fun, sim_time=60, theta_angle=0, return_history=False, 
#                                vehicle_mass=1500, rolling_resistance=0.06):
    
#     # # --- 1. SETUP & KONSTANTA ---
#     # # Fisika Kendaraan (Ground Truth)
#     physics = VehiclePhysics(theta=theta_angle, M=vehicle_mass, Croll=rolling_resistance)

    # # Controller Gains (dari input algoritma optimasi)
    # Kve, Kvrel, Kxe = params
    
    # # Kondisi Awal
    # theta = 0
    # v_ref = 25.0        # Target speed saat jalan kosong (m/s)
    # x_ego, v_ego, a_ego = 0, 22.0, 0
    # x_lead, v_lead = 60.0, 22.0 # Mulai di jarak 60m (di luar jangkauan sensor)
    
    # # --- KONFIGURASI SENSOR (NOISE & LIMIT) ---
    # SENSOR_MAX_RANGE = 25.0   # Sensor buta jika jarak >= 25m
    # NOISE_DIST_STD   = 0.5    # Deviasi standar noise jarak (meter)
    # NOISE_VEL_STD    = 0.2    # Deviasi standar noise kecepatan (m/s)

    # # --- KONSTANTA CAPPIELLO ET AL. ---
    # # Bobot optimal dari paper Cappiello [cite: 264]
    # w1_pos = 20.6537       
    # w2_vel = 1.74043       
    # w3_nrg = 0.074934      
    # d_def = 20.0          # Desired distance [cite: 262]
    
    # Ts = 1 / 60

    # simulation_steps = int(sim_time / Ts)
    # total_cost_val = 0

    # history = {
    #     "time": [], "v_ego": [], "v_lead": [], "v_lead_sensed": [], "x_ego": [], "x_lead": [],
    #     "car_force": [], "a_ego": [], "mode": [], "cost_instant": [],
    #     "dist": [], "dist_reading": []
    # } if return_history else None

    # # --- 2. SIMULATION LOOP ---
    # for i in range(simulation_steps):
    #     t = i * Ts

    #     # --- A. UPDATE DUNIA NYATA (GROUND TRUTH) ---
    #     # Gerakan mobil depan (Lead Vehicle)
    #     theta = (theta + 30 * Ts) % 360
    #     a_lead = np.sin(np.deg2rad(theta)) 
    #     v_lead += a_lead * Ts
    #     x_lead += v_lead * Ts

    #     # Jarak Asli (Fisika)
    #     d_actual_ground_truth = x_lead - x_ego
        
    #     # --- B. MODEL SENSOR (NOISE & LIMIT) ---
    #     has_target, d_sensed, v_lead_sensed = ego_sensing(d_actual_ground_truth, v_lead, SENSOR_MAX_RANGE, NOISE_DIST_STD, NOISE_VEL_STD)

    #     u, current_mode, c_pos, c_vel, err_v = control_law(
    #         Kve, Kvrel, Kxe, v_ref, v_ego, v_lead, v_lead_sensed, d_sensed, d_actual_ground_truth, d_def, has_target
    #     )

    #     # --- D. UPDATE FISIKA EGO ---
    #     v_ego, a_ego, Fd = physics.step(v_ego, u, Ts)
    #     x_ego += v_ego * Ts
        
    #     c_nrg = max(0, Fd * v_ego)  # Hanya hitung saat akselerasi

    #     inst_cost = error_synthesizer(derr, verr, c_nrg, t, obj_fun)
    #     total_cost_val += inst_cost

    #     if return_history:
    #         history["time"].append(t)
    #         history["v_ego"].append(v_ego)
    #         history["v_lead"].append(v_lead)
    #         history["v_lead_sensed"].append(v_lead_sensed if has_target else 0)
    #         history["x_ego"].append(x_ego)
    #         history["x_lead"].append(x_lead)
    #         history["car_force"].append(Fd)
    #         history["a_ego"].append(a_ego)
    #         history["mode"].append(current_mode)
    #         history["cost_instant"].append(inst_cost)
    #         history["dist"].append(d_actual_ground_truth)
    #         history["dist_reading"].append(d_sensed if has_target else 30) 

    # return [total_cost_val, history] if return_history else total_cost_val



def lean_simulate_system(params, obj_fun, return_history=False, 
                         sim_time=60, theta_angle=0, vehicle_mass=1500, 
                         rolling_resistance=0.06):
    """
    obj_fun is now a FUNCTION, not a string.
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
        "car_force": [], "dist": [], "a_ego": [], "err": []
    }

    for i in range(simulation_steps):
        t = i * Ts

        # Compute errors
        dactual = xlead - xego
        dsafe   = ddef + Tg * vego

        vrel    = vlead - vego
        derr    = dsafe - dactual
        verr    = vset - vego

        # Controller
        u_v = verr_gain * verr
        u_x = vrel * vx_gain + xerr_gain * derr

        # LOGIKA SWITCHING YANG LEBIH LOGIS
        u = min(u_x, u_v)

        # Update ego car
        vego, aego, Ft = physics.step(vego, u, Ts)
        xego += vego * Ts

        # # Update lead car
        # theta = (theta + 30 * Ts) % 360
        # alead = sin_angle(theta)
        # vlead += alead * Ts
        # xlead += vlead * Ts


        # --- Parameter Gaussian ---
        mu = 0.0           # Mean (Rata-rata): 0 berarti kadang gas, kadang rem seimbang
        sigma = 0.5        # Std Dev (Simpangan Baku): Seberapa "kasar" perubahannya
                           # Sigma kecil = menyetir halus, Sigma besar = menyetir agresif

        # --- Update lead car ---
        
        # 1. Generate percepatan acak (Gaussian Distribution)
        alead = np.random.normal(loc=mu, scale=sigma)
        
        # 2. (Opsional tapi PENTING) Batasi nilai percepatan agar masuk akal (Clamping)
        #    Misal: Max pengereman -5 m/s^2, Max gas +3 m/s^2
        alead = np.clip(alead, -5.0, 3.0)

        # 3. Integrasi Euler (Sama seperti sebelumnya)
        vlead += alead * Ts
        
        # 4. (PENTING) Pastikan mobil tidak mundur (kecepatan negatif)
        if vlead < 0:
            vlead = 0
            alead = 0 # Reset akselerasi jika berhenti
            
        xlead += vlead * Ts
        
        # Catatan: Variabel 'theta' tidak lagi digunakan di sini



        c_nrg   = max(0, Ft)
        err     = error_synthesizer(derr, verr, 0, t, obj_fun)
        obj_val += err

        if return_history:
            history["time"].append(t)
            history["v_ego"].append(vego)
            history["v_lead"].append(vlead)
            history["x_ego"].append(xego)
            history["x_lead"].append(xlead)
            history["car_force"].append(Ft)
            history["dist"].append(dactual)
            history["a_ego"].append(aego)
            history["err"].append(err)

    return [obj_val, history] if return_history else obj_val

# def load_vlead_aided_simulation("")

from core.car import Car, EgoCar, LeadCar

def load_lead_data_from_csv(filepath):
    import pandas as pd

    data = pd.read_csv(filepath)
    time = data['time'].to_numpy()
    v_lead = data['v_lead'].to_numpy()
    a_lead = data['a_lead'].to_numpy()

    return time, v_lead, a_lead

def new_simulate_system(params, obj_fun, return_history=False, 
                        sim_time=60, theta_angle=0, vehicle_mass=1500, 
                        rolling_resistance=0.06):
    ego_physics_conf = {
        "M": vehicle_mass,
        "Cd": 0.36,
        "A": 2.42,
        "rho": 1.225,
        "Croll": rolling_resistance,
        "g": 9.81,  
        "theta": theta_angle
    }

    ego_controller_conf = {
        "Kve": params[0],
        "Kvrel": params[1],
        "Kxe": params[2],
        "v_set": 25.0,
        "d_def": 20.0,
        "t_gap": 1.2
    }

    ego = EgoCar(pos=0.0, vel=22.0, physics_config=ego_physics_conf, controller_config=ego_controller_conf)

    lead = LeadCar(pos=60.0, vel=22.0)
    # lead_car_movement = load_lead_data_from_csv("data/lead_car_prastiyanto.csv")

    Ts = 1 / 60
    simulation_steps = int(sim_time / Ts)

    history = {
        "time": [], "v_ego": [], "v_lead": [], "x_ego": [], "x_lead": [],
        "dist": [], "d_safe": [], "car_force": [], "a_ego": [], "a_lead": [], "err": []
    } if return_history else None

    for i in range(simulation_steps):
        t = i * Ts

        # Update Lead Car
        lead.update_prastiyanto(Ts)

        # Update Ego Car
        u, Ft = ego.step(lead, Ts)

        # Compute errors
        derr = (ego.controller.d_safe + ego.controller.t_gap * ego.vel) - ego.real_dist
        verr = ego.controller.v_set - ego.vel
        err = cappiello_original(derr, verr, 0, t)

        # Update history
        if return_history:
            history["time"].append(t)
            history["a_ego"].append(ego.accel)
            history["a_lead"].append(lead.accel)
            history["v_ego"].append(ego.vel)
            history["v_lead"].append(lead.vel)
            history["x_ego"].append(ego.pos)
            history["x_lead"].append(lead.pos)
            history["dist"].append(ego.real_dist)
            history["d_safe"].append(ego.controller.d_safe)
            history["car_force"].append(Ft)
            history["err"].append(err)

    return [err, history] if return_history else err


def lean_simulate_system_multirate(params, obj_fun, return_history=False, 
                                   sim_time=60, theta_angle=0, vehicle_mass=1500, 
                                   rolling_resistance=0.06):
    
    physics = VehiclePhysics(theta=theta_angle, M=vehicle_mass, Croll=rolling_resistance)
    verr_gain, vx_gain, xerr_gain = params

    # --- FREKUENSI ---
    FPS_PHYSICS = 60  # Fisika dunia nyata (halus)
    FPS_SENSOR  = 20  # Refresh rate sensor/kontroler (agak patah-patah)
    
    Ts_physics = 1 / FPS_PHYSICS
    # Rasio update: 60 / 20 = 3. Artinya kontroler update tiap 3 frame fisika.
    UPDATE_RATIO = int(FPS_PHYSICS / FPS_SENSOR) 

    # Init State
    vset = 25
    xego, vego, aego = 0, 22, 0
    xlead, vlead = 60, 22
    ddef, Tg = 20, 1.2
    
    obj_val = 0
    simulation_steps = int(sim_time / Ts_physics)
    
    # Variabel untuk Zero-Order Hold (Nilai yang "ditahan")
    u_held = 0.0 
    
    # Init Lead Car dynamics
    alead = 0
    mu, sigma = 0.0, 0.5

    history = {
        "time": [], "v_ego": [], "v_lead": [], "x_ego": [], "x_lead": [],
        "dist": [], "u_applied": [], "err": [] # u_applied = yang masuk ke fisika
    }

    for i in range(simulation_steps):
        t = i * Ts_physics

        # --- 1. UPDATE DUNIA NYATA (FISIKA) - Jalan tiap step (60Hz) ---
        
        # Lead Car bergerak (Stochastic)
        # Kita update alead tiap step biar gerakannya smooth tapi random
        alead_noise = np.random.normal(mu, sigma)
        # Smoothing factor biar ga jittery banget (Low Pass Filter sederhana)
        alead = 0.9 * alead + 0.1 * alead_noise 
        alead = np.clip(alead, -5.0, 3.0)
        
        vlead += alead * Ts_physics
        if vlead < 0: vlead, alead = 0, 0
        xlead += vlead * Ts_physics

        # Ground Truth States
        dactual_ground_truth = xlead - xego
        dsafe_ground_truth   = ddef + Tg * vego

        # --- 2. UPDATE SENSOR & KONTROL (Hanya tiap ratio step - 20Hz) ---
        if i % UPDATE_RATIO == 0:
            # Di sini kita "Membaca Sensor"
            # Kamu bisa tambahkan noise sensor di sini kalau mau lebih realistis
            # d_sensed = dactual_ground_truth + np.random.normal(0, 0.5) 
            
            # Hitung Error berdasarkan pembacaan sensor
            d_err = dsafe_ground_truth - dactual_ground_truth
            v_err = vset - vego
            v_rel = vlead - vego
            
            # Hitung Kontroler
            u_v = verr_gain * v_err
            u_x = v_rel * vx_gain + xerr_gain * d_err
            
            # Logic Switching
            u_calculated = min(u_x, u_v)
            
            # UPDATE ZOH: Simpan nilai u ini untuk dipakai frame-frame selanjutnya
            u_held = u_calculated
            
            # Hitung Cost (biasanya cost dihitung saat update kontrol)
            # Energi dihitung dari u_held
            err_val = error_synthesizer(d_err, v_err, u_held, t, obj_fun)
            obj_val += err_val

        # --- 3. APPLY CONTROL TO PHYSICS ---
        # Fisika menggunakan 'u_held' (nilai terakhir yang dihitung kontroler)
        # Meskipun kontroler tidak update di frame ini, pedal gas tetap terinjak!
        
        vego, aego, Ft = physics.step(vego, u_held, Ts_physics)
        xego += vego * Ts_physics

        # --- 4. HISTORY LOGGING ---
        if return_history:
            history["time"].append(t)
            history["v_ego"].append(vego)
            history["v_lead"].append(vlead)
            history["x_ego"].append(xego)
            history["x_lead"].append(xlead)
            history["dist"].append(dactual_ground_truth)
            history["u_applied"].append(u_held)
            # history["err"] mungkin perlu handling khusus karena update-nya jarang
            # tapi untuk plotting, append nilai terakhir oke.
    
    return [obj_val, history] if return_history else obj_val


def balls(params, obj_fun, return_history=False, 
                         sim_time=60, theta_angle=0, vehicle_mass=1500, 
                         rolling_resistance=0.06):
    """
    obj_fun is now a FUNCTION, not a string.
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
        "car_force": [], "dist": [], "a_ego": [], "err": []
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
        # u = min(u_v, u_x)
        if dactual < dsafe:
            u = min(u_v, u_x)
        else:
            u = u_x

        # OBJECTIVE FUNCTION: direct call
        try:
            obj_val += obj_fun(err, t)   # ITAE / ITSE require time
        except TypeError:
            obj_val += obj_fun(err)      # IAE / ISE have no time param
        

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

    # return [obj_val, history] if return_history else obj_val
    return [obj_val, history] if return_history else obj_val

def lean_simulate_system_legacy(params, obj_function, return_history=False, 
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
        "car_force": [], "dist": [], "a_ego": [], "err": []
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

    # return [obj_val, history] if return_history else obj_val
    return history if return_history else obj_val


def acc_single_scenario(params, obj_function, scenario):
    return new_simulate_system(
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