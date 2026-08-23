"""Shared data structures for the refactored ACC simulation stack.

The main design goal is to keep the simulator independent from the controller.
A controller receives an ``ACCState`` and returns a ``ControllerOutput``.
The simulator does not need to know whether the controller is classical ACC,
PID, FOPID, MPC, or another controller family.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ACCState:
    """Information available to the controller at one simulation step.

    Sign conventions used by the provided controllers:
    - ``d_actual = x_lead - x_ego``.
    - ``d_ref = standstill_distance + time_gap * v_ego``.
    - ``spacing_error_far_positive = d_actual - d_ref``.
      Positive means the ego vehicle is farther than the desired gap and may
      accelerate. Negative means the ego vehicle is too close and should brake.
    - ``spacing_error_close_positive = d_ref - d_actual``.
      This is the convention used by the legacy/classical ACC controller.
    """

    t: float
    dt: float
    x_ego: float
    v_ego: float
    a_ego: float
    x_lead: float
    v_lead: float
    a_lead: float
    v_set: float
    d_actual: float
    d_ref: float
    d_safe: float

    @property
    def relative_velocity(self) -> float:
        """Lead velocity minus ego velocity, matching the legacy script."""
        return self.v_lead - self.v_ego

    @property
    def speed_error(self) -> float:
        """Set-speed error. Positive means the ego is below the set speed."""
        return self.v_set - self.v_ego

    @property
    def spacing_error_close_positive(self) -> float:
        """Legacy convention: positive means ego is too close."""
        return self.d_ref - self.d_actual

    @property
    def spacing_error_far_positive(self) -> float:
        """FOPID convention: positive means ego is farther than desired."""
        return self.d_actual - self.d_ref


@dataclass
class ControllerOutput:
    """Controller command returned to the vehicle physics model."""

    u: float
    mode: str
    error: float
    d_ref: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SimulationConfig:
    """Configuration shared by all ACC simulations."""

    dt: float = 1.0 / 60.0
    sim_time: float = 60.0
    v_set: float = 25.0
    time_gap: float = 1.2
    standstill_distance: float = 20.0
    ttc_threshold: float = 1.5
    control_limit: Optional[float] = None
    acceleration_limit: Optional[float] = None
    jerk_limit: Optional[float] = None
    min_gap_margin: float = 0.0
    eps: float = 1e-9
