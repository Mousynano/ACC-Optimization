"""Longitudinal ego-vehicle plant used by the ACC benchmark."""

from math import cos, sin
import numpy as np


class VehiclePhysics:
    def __init__(
        self,
        M=1500.0,
        Cd=0.36,
        A=2.42,
        rho=1.225,
        Croll=0.015,
        g=9.81,
        theta=0.0,
    ):
        self.M = float(M)
        self.Cd = float(Cd)
        self.A = float(A)
        self.rho = float(rho)
        self.Croll = float(Croll)
        self.g = float(g)
        self.theta = np.radians(theta)

    def aerodynamic_drag(self, v):
        return 0.5 * self.rho * self.Cd * self.A * v**2

    def rolling_resistance(self):
        return self.Croll * self.M * self.g * cos(self.theta)

    def gravitational_force(self):
        return self.M * self.g * sin(self.theta)

    def step_acceleration_command(self, v, a_cmd, dt, Fmax=6000.0):
        """Advance one plant step from a desired longitudinal acceleration."""
        resistance = (
            self.aerodynamic_drag(v)
            + self.rolling_resistance()
            + self.gravitational_force()
        )
        required_force = self.M * float(a_cmd) + resistance
        traction_force = float(np.clip(required_force, -Fmax, Fmax))
        acceleration = (traction_force - resistance) / self.M
        v_new = max(float(v) + acceleration * float(dt), 0.0)
        return v_new, acceleration, traction_force
