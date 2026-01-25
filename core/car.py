
# class Car:
#     def __init__(self, idx, params):
#         self.idx = idx
#         self.params = params
#         self.adc_error_history = [0]
#         self.avc_error_history = [0]
#         self.time_adc = [0]
#         self.time_avc = [0]
#         self.avc_error = 0
#         self.adc_error = 0
#         self.sum_error_Avc = 0
#         self.sum_error_Adc = 0
#         self.pid_avc = 0
#         self.pid_adc = 0
#         self.vset = 25
#         self.xego = 0
#         self.vego = 22
#         self.aego = 0
#         self.pid = 0
#         self.ff = 0
#         self.time = [0]
#         self.fitness = 10000
#         self.step_response_result = {
#             'riseTime': [],
#             'settlingTime': [],
#             'overshoot': [],
#             'overshootPercentage': []
#         }

#     def _to_dict(self):
#         return {
#             'idx': self.idx,
#             'params': self.params,
#             'fitness': self.fitness,
#             'step_response_result': self.step_response_result
#         }

from core.physics import VehiclePhysicsV2
from core.control import ACCController
import numpy as np

# Base Class (Cuma data holder)
class Car:
    def __init__(self, pos=0.0, vel=0.0):
        self.pos = pos
        self.vel = vel
        self.accel = 0.0

# --- LEAD CAR (Kinematic / Data Driven) ---
class LeadCar(Car):
    def __init__(self, pos=60.0, vel=0.0):
        super().__init__(pos=pos, vel=vel)
        self.theta = 0.0 
    def update(self, target_vel, dt):
        """
        Lead Car bergerak sesuai data (misal dari CSV UDDS).
        Tidak punya fisika, cuma kinematic.
        """
        self.accel = (target_vel - self.vel) / dt # Hitung akselerasi cuma buat data
        self.vel = target_vel
        self.pos += self.vel * dt

    def update_prastiyanto(self, dt):
        self.theta = (self.theta + 30 * dt) % 360
        self.accel = np.sin(np.deg2rad(self.theta)) 
        self.vel += self.accel * dt
        self.pos += self.vel * dt


# --- EGO CAR (Dynamic / Physics Driven) ---
class EgoCar(Car):
    def __init__(self, pos=0.0, vel=0.0, physics_config={}, controller_config={}):
        super().__init__(pos, vel)
        
        # COMPOSITION: EgoCar MEMILIKI Physics dan Controller
        self.physics = VehiclePhysicsV2(**physics_config)
        self.controller = ACCController(**controller_config)
        
        # Sensor readings
        self.sensed_dist = 0.0
        self.sensed_vrel = 0.0

        # Real readings
        self.real_dist = 0.0
        self.real_vrel = 0.0

    def step(self, lead_car, dt, noise_level=0.0):
        """
        Satu langkah simulasi lengkap:
        Sense -> Think -> Act -> Physics Update
        """
        
        # 1. SENSE (Baca Sensor + Noise)
        self.real_dist = lead_car.pos - self.pos
        self.real_vrel = lead_car.vel - self.vel
        
        # Tambah noise sensor (Gaussian)
        self.sensed_dist = self.real_dist + np.random.normal(0, noise_level)
        self.sensed_vrel = self.real_vrel + np.random.normal(0, noise_level * 0.1)
        
        # 2. THINK (Controller menghitung u)
        u_cmd = self.controller.compute_command(
            ego_v=self.vel,
            lead_v=self.vel + self.sensed_vrel, # v_lead = v_ego + v_rel
            lead_dist=self.sensed_dist,
            dt=dt
        )
        
        # 3. ACT & PHYSICS UPDATE
        # Masukkan u_cmd ke mesin fisika
        self.vel, self.accel, force = self.physics.step(self.vel, u_cmd, dt)
        
        # Update posisi (Integral kecepatan)
        self.pos += self.vel * dt
        
        return u_cmd, force