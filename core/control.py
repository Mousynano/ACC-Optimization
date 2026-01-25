import numpy as np
class ACCController:
    def __init__(self, Kve, Kvrel, Kxe, v_set=25.0, d_def=20.0, t_gap=1.2):
        self.Kve = Kve
        self.Kvrel = Kvrel
        self.Kxe = Kxe
        
        # Setpoints
        self.v_set = v_set    # Target kecepatan cruising
        self.d_safe_base = d_def
        self.t_gap = t_gap
        self.d_safe = d_def 
        
        # Memory untuk PID (Integral & Derivative)
        self.prev_error = 0
        self.integral_sum = 0
        
    def compute_command(self, ego_v, lead_v, lead_dist, dt):
        # 1. Tentukan Jarak Aman Dinamis (d_safe + time_gap * speed)
        self.d_safe = self.d_safe_base + (self.t_gap * ego_v)
        
        # 2. Hitung Error
        err_dist = lead_dist - self.d_safe   # Jarak: Positif kalau terlalu jauh, Negatif kalau terlalu dekat
        err_vel  = self.v_set - ego_v              # Kecepatan: Target - Aktual
        
        # 3. Controller
        u_v = self.Kve * err_vel
        u_x = self.Kvrel * err_vel + self.Kxe * err_dist

        # # Model Prastiyanto
        # if lead_dist > self.d_safe:
        #     # MODE: FOLLOWING (Within Sensor Range)
        #     # Kita ingin ngejar kecepatan setpoint sambil jaga jarak aman
        #     u = min(u_v, u_x)  # Pilih aksi yang paling "aman" (min acceleration)
        #     current_mode = "ACC (Tracking)"
        # else:
        #     # MODE: CRUISE (No Target Detected)
        #     # Kita hanya ingin ngejar kecepatan setpoint
        #     u = u_v
        #     current_mode = "Cruise Control"

        u = min(u_v, u_x)  # Pilih aksi yang paling "aman" (min acceleration)

        # Clamp output ke range [-1, 1]
        return np.clip(u, -1.0, 1.0)