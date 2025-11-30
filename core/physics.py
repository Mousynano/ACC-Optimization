import numpy as np
from math import sin, cos

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
