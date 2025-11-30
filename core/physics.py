import math
import numpy as np
from core.utils import sin_angle, cos_angle

# ====================== PHYSICS (Rajamani, Izci, Ekinci, etc.) ======================
class VehiclePhysics:
    def __init__(self, M=1741, Da=0.6, Croll=0.06, g=9.81, theta=0.0):
        self.M = M
        self.Da = Da
        self.Croll = Croll
        self.g = g
        self.theta = theta

    def aerodynamic_drag(self, v):
        # linearized drag: F_a = Da * v
        return self.Da * v

    def rolling_resistance(self):
        # F_rr = Croll * M * g
        return self.Croll * self.M * self.g * cos_angle(self.theta)

    def gravitational_force(self):
        # F_g = M * g * sin(theta)
        return self.M * self.g * sin_angle(self.theta)

    def step(self, v, u, dt):
        """
        u = acceleration command from PID (unit: m/s^2)
        Assumption: Fd / M = u  →   Fd = u * M
        """
        
        Fd = u * self.M
        Fa = self.aerodynamic_drag(v)
        Frr = self.rolling_resistance()
        Fg = self.gravitational_force()
        Ftot = Fd - (Fa + Frr + Fg)
        # print(f"Debug: Fd={Fd}, Fa={Fa}, Frr={Frr}, Fg={Fg}, Ftot={Ftot}, u={u}, v={v}, dt={dt}, M={self.M}")

        # Newton: M dv/dt = Fd - (Fa + Frr + Fg)
        dv = Ftot / self.M

        dv = max(-3.0, min(dv, 2.0)) # constraint: perubahan kecepatan maksimal ±5 m/s²

        v_new = v + dv * dt

        # constraint: ACC tidak mengizinkan mundur
        if v_new < 0.0:
            v_new = 0.0
            dv = 0.0

        return v_new, dv, Fd