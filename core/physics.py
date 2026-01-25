import numpy as np
from math import sin, cos, radians

class VehiclePhysics:
    def __init__(self, 
                 M=1741,                # vehicle mass (kg)
                 Cd=0.36,               # aerodynamic drag coefficient
                 A=2.42,                # frontal area (m^2)
                 rho=1.225,             # air density 20°C
                 Croll=0.015,           # rolling resistance coefficient
                 g=9.81, 
                 theta=0.0):            # slope angle in degrees

        self.M = M
        self.Cd = Cd
        self.A = A
        self.rho = rho
        self.Croll = Croll
        self.g = g
        self.theta = np.radians(theta)  # convert deg → rad

    # ======== Forces ========
    def aerodynamic_drag(self, v):
        return 0.5 * self.rho * self.Cd * self.A * v**2

    def rolling_resistance(self):
        return self.Croll * self.M * self.g * cos(self.theta)

    def gravitational_force(self):
        return self.M * self.g * sin(self.theta)

    # ======== Integration Step ========
    def step(self, v, u, dt):
        """
        u = accelerator command (interpreted as engine force ratio 0–1)
        F_t = traction force = u * F_max (assumed ~4000–6000 N)
        """

        Fmax = 6000  # approximated max traction from typical ICE mid-size car
        Ft = np.clip(u, -1, 1) * Fmax

        Fa = self.aerodynamic_drag(v)
        Frr = self.rolling_resistance()
        Fg  = self.gravitational_force()

        # total opposing forces
        F_resist = Fa + Frr + Fg

        # dv/dt = (Ft – F_resist) / M
        dv = (Ft - F_resist) / self.M

        # velocity integration
        v_new = v + dv * dt
        v_new = max(v_new, 0)  # no backward movement

        return v_new, dv, Ft


class VehiclePhysicsV2:
    def __init__(self, M=1500, Cd=0.36, A=2.42, rho=1.225, Croll=0.015, g=9.81, theta=0.0):
        self.M = M
        self.Cd = Cd
        self.A = A
        self.rho = rho
        self.Croll = Croll
        self.g = g
        self.theta = radians(theta) # Simpan dalam radian
        
        # Max Force (N) untuk normalisasi input u [-1, 1]
        self.F_max = 6000.0 

    def set_slope(self, theta_deg):
        """Update kemiringan jalan secara dinamis (misal saat nanjak)"""
        self.theta = radians(theta_deg)

    def step(self, v, u, dt):
        """
        v : current speed (m/s)
        u : control signal [-1, 1] (Gas/Rem)
        dt: delta time (s)
        """
        # 1. Convert Control Signal to Force
        Ft = np.clip(u, -1.0, 1.0) * self.F_max
        
        # 2. Calculate Resistances
        Fa = 0.5 * self.rho * self.Cd * self.A * (v**2) # Drag Udara
        Fr = self.Croll * self.M * self.g * cos(self.theta) # Gesekan Ban
        Fg = self.M * self.g * sin(self.theta) # Gravitasi (Nanjak/Turun)
        
        F_resist = Fa + Fr + Fg
        
        # 3. Newton's 2nd Law (F = ma -> a = F/m)
        F_net = Ft - F_resist
        accel = F_net / self.M
        
        # 4. Integrate Velocity
        v_new = v + accel * dt
        
        # Constraint: Tidak boleh mundur
        if v_new < 0:
            v_new = 0
            accel = 0
            
        return v_new, accel, Ft