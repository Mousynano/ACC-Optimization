"""Classical ACC controller extracted from the legacy monolithic script.

Original legacy logic:
    vrel = vlead - vego
    derr = dsafe - dactual
    verr = vset - vego
    u_v = Kve * verr
    u_x = Kvrel * vrel + Kde * derr
    u = min(u_x, u_v)

The implementation below preserves that control law but wraps it in a clean
controller interface so it can be swapped with FOPID without changing the
vehicle model, lead-car model, objective, or optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from controllers.base import BaseACCController, require_param_count
from core.types import ACCState, ControllerOutput


@dataclass
class ClassicalACCController(BaseACCController):
    """Legacy/classical ACC controller.

    Parameters
    ----------
    Kve:
        Speed-error gain applied to ``v_set - v_ego``.
    Kvrel:
        Relative-velocity gain applied to ``v_lead - v_ego``.
    Kde:
        Distance-error gain applied to ``d_ref - d_actual``.
    time_gap:
        Constant-time-headway parameter used to compute desired gap.
    standstill_distance:
        Desired gap at zero ego speed.
    """

    Kve: float
    Kvrel: float
    Kde: float
    time_gap: float = 1.2
    standstill_distance: float = 20.0
    name: str = "classical_acc"

    @classmethod
    def from_params(
        cls,
        params: Sequence[float],
        default_time_gap: float = 1.2,
        default_standstill_distance: float = 20.0,
    ) -> "ClassicalACCController":
        """Build the controller from an optimizer vector.

        Supported optimizer vectors:
        - ``[Kve, Kvrel, Kde]``
        - ``[Kve, Kvrel, Kde, time_gap, standstill_distance]``
        """
        require_param_count(params, valid_counts=(3, 5), controller_name=cls.__name__)
        Kve, Kvrel, Kde = map(float, params[:3])
        if len(params) >= 5:
            time_gap = float(params[3])
            standstill_distance = float(params[4])
        else:
            time_gap = float(default_time_gap)
            standstill_distance = float(default_standstill_distance)
        return cls(Kve=Kve, Kvrel=Kvrel, Kde=Kde, time_gap=time_gap, standstill_distance=standstill_distance)

    def desired_gap(self, v_ego: float) -> float:
        """Constant-time-headway spacing policy."""
        return self.standstill_distance + self.time_gap * v_ego

    def step(self, state: ACCState) -> ControllerOutput:
        d_ref = self.desired_gap(state.v_ego)
        derr = d_ref - state.d_actual  # legacy sign: positive means too close
        verr = state.v_set - state.v_ego
        vrel = state.v_lead - state.v_ego

        u_speed = self.Kve * verr
        u_spacing = self.Kvrel * vrel + self.Kde * derr
        u = min(u_spacing, u_speed)

        # This part is kinda different since we usually compute both
        # errors regardless of which control command is active
        if u_spacing <= u_speed:
            mode = "spacing"
            active_error = derr
        else:
            mode = "speed"
            active_error = verr

        return ControllerOutput(
            u=float(u),
            mode=mode,
            error=float(active_error),
            d_ref=float(d_ref),
            diagnostics={
                "u_speed": float(u_speed),
                "u_spacing": float(u_spacing),
                "speed_error": float(verr),
                "spacing_error_close_positive": float(derr),
                "relative_velocity": float(vrel),
                "time_gap": float(self.time_gap),
                "standstill_distance": float(self.standstill_distance),
            },
        )
