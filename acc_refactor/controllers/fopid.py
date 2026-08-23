"""Fractional-order PID controller for ACC.

This module implements a practical discrete FOPID controller using a finite
memory Grunwald-Letnikov approximation. The controller is intentionally
self-contained and NumPy-only so it can run inside the existing metaheuristic
pipeline without SciPy/control-toolbox dependencies.

Continuous ideal form:
    u(t) = Kp e(t) + Ki D^{-lambda} e(t) + Kd D^{mu} e(t)

Discrete finite-memory approximation:
    D^alpha e[k] ~= dt^{-alpha} sum_{j=0}^{M} c_j(alpha) e[k-j]

where ``alpha = -lambda`` for the fractional integral and ``alpha = mu`` for
the fractional derivative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from acc_refactor.controllers.base import BaseACCController, require_param_count
from acc_refactor.types import ACCState, ControllerOutput


def _gl_coefficients(alpha: float, memory_size: int) -> np.ndarray:
    """Return Grunwald-Letnikov coefficients for derivative order ``alpha``.

    ``alpha`` can be positive for a fractional derivative or negative for a
    fractional integral. The recurrence avoids scipy.special and is stable
    enough for controller prototyping with modest memory sizes.
    """
    if memory_size < 1:
        raise ValueError("memory_size must be at least 1")

    coeffs = np.empty(memory_size, dtype=float)
    coeffs[0] = 1.0
    for j in range(1, memory_size):
        coeffs[j] = coeffs[j - 1] * (1.0 - (alpha + 1.0) / float(j))
    return coeffs


@dataclass
class FOPIDACCController(BaseACCController):
    """FOPID controller for ACC spacing/speed control.

    Parameter vector support
    ------------------------
    The optimizer may pass one of these vectors:

    1. ``[Kp, Ki, Kd, lambda_, mu]``
       Uses default time gap and standstill distance.

    2. ``[Kp, Ki, Kd, lambda_, mu, time_gap, standstill_distance]``
       Optimizes the FOPID gains and the spacing policy.

    3. ``[Kp, Ki, Kd, lambda_, mu, time_gap, standstill_distance, Kvrel]``
       Adds relative-velocity damping. This is useful for ACC because a pure
       spacing-error FOPID can oscillate when the lead vehicle changes speed.

    Error used by the FOPID block
    -----------------------------
    The controller uses an intuitive acceleration sign convention:
    - positive error -> ego is allowed/encouraged to accelerate;
    - negative error -> ego should decelerate.

    In spacing mode, ``e = d_actual - d_ref``.
    In speed mode, ``e = v_set - v_ego``.
    """

    Kp: float
    Ki: float
    Kd: float
    lambda_: float
    mu: float
    time_gap: float = 1.2
    standstill_distance: float = 20.0
    Kvrel: float = 0.0
    memory_size: int = 300
    output_limit: Optional[float] = None
    switch_margin: float = 0.0
    derivative_filter: float = 1.0
    name: str = "fopid_acc"
    _errors: List[float] = field(default_factory=list, init=False, repr=False)
    _int_coeffs: np.ndarray = field(default=None, init=False, repr=False)
    _der_coeffs: np.ndarray = field(default=None, init=False, repr=False)
    _prev_derivative_term: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.lambda_ <= 2.0):
            raise ValueError("lambda_ should be in (0, 2] for a practical FOPID search range")
        if not (0.0 <= self.mu <= 2.0):
            raise ValueError("mu should be in [0, 2] for a practical FOPID search range")
        if not (0.0 < self.derivative_filter <= 1.0):
            raise ValueError("derivative_filter must be in (0, 1]")
        self._int_coeffs = _gl_coefficients(alpha=-float(self.lambda_), memory_size=int(self.memory_size))
        self._der_coeffs = _gl_coefficients(alpha=float(self.mu), memory_size=int(self.memory_size))

    @classmethod
    def from_params(
        cls,
        params: Sequence[float],
        default_time_gap: float = 1.2,
        default_standstill_distance: float = 20.0,
        memory_size: int = 300,
        output_limit: Optional[float] = None,
        derivative_filter: float = 1.0,
    ) -> "FOPIDACCController":
        """Build the controller from an optimizer vector."""
        require_param_count(params, valid_counts=(5, 7, 8), controller_name=cls.__name__)
        Kp, Ki, Kd, lambda_, mu = map(float, params[:5])
        if len(params) >= 7:
            time_gap = float(params[5])
            standstill_distance = float(params[6])
        else:
            time_gap = float(default_time_gap)
            standstill_distance = float(default_standstill_distance)
        Kvrel = float(params[7]) if len(params) >= 8 else 0.0
        return cls(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            lambda_=lambda_,
            mu=mu,
            time_gap=time_gap,
            standstill_distance=standstill_distance,
            Kvrel=Kvrel,
            memory_size=memory_size,
            output_limit=output_limit,
            derivative_filter=derivative_filter,
        )

    def reset(self) -> None:
        self._errors.clear()
        self._prev_derivative_term = 0.0

    def desired_gap(self, v_ego: float) -> float:
        return self.standstill_distance + self.time_gap * v_ego

    def _fractional_terms(self, error: float, dt: float) -> tuple[float, float]:
        """Update memory and return fractional integral and derivative terms."""
        self._errors.insert(0, float(error))
        if len(self._errors) > self.memory_size:
            del self._errors[self.memory_size:]

        e = np.asarray(self._errors, dtype=float)
        n = len(e)

        integral = (dt ** self.lambda_) * float(np.dot(self._int_coeffs[:n], e))
        derivative_raw = (dt ** (-self.mu)) * float(np.dot(self._der_coeffs[:n], e)) if self.mu > 0 else float(error)

        derivative = (
            self.derivative_filter * derivative_raw
            + (1.0 - self.derivative_filter) * self._prev_derivative_term
        )
        self._prev_derivative_term = derivative
        return integral, derivative

    def _select_error(self, state: ACCState, d_ref: float) -> tuple[str, float]:
        """Choose spacing mode when the lead vehicle constrains ego motion."""
        lead_is_relevant = state.d_actual <= d_ref + self.switch_margin or state.v_lead < state.v_ego
        if lead_is_relevant:
            # Positive when ego is too far; negative when ego is too close.
            return "spacing", state.d_actual - d_ref
        return "speed", state.v_set - state.v_ego

    def step(self, state: ACCState) -> ControllerOutput:
        d_ref = self.desired_gap(state.v_ego)
        mode, error = self._select_error(state, d_ref)
        frac_integral, frac_derivative = self._fractional_terms(error, state.dt)

        u_fopid = self.Kp * error + self.Ki * frac_integral + self.Kd * frac_derivative

        # Optional ACC-specific damping. With the sign convention used here,
        # v_lead - v_ego < 0 means the ego is faster than the lead car, so this
        # term becomes negative and helps braking when Kvrel > 0.
        u_rel = self.Kvrel * (state.v_lead - state.v_ego)
        u = u_fopid + u_rel

        if self.output_limit is not None:
            limit = abs(float(self.output_limit))
            u = float(np.clip(u, -limit, limit))

        return ControllerOutput(
            u=float(u),
            mode=mode,
            error=float(error),
            d_ref=float(d_ref),
            diagnostics={
                "u_fopid": float(u_fopid),
                "u_relative_velocity": float(u_rel),
                "fractional_integral": float(frac_integral),
                "fractional_derivative": float(frac_derivative),
                "Kp": float(self.Kp),
                "Ki": float(self.Ki),
                "Kd": float(self.Kd),
                "lambda": float(self.lambda_),
                "mu": float(self.mu),
                "time_gap": float(self.time_gap),
                "standstill_distance": float(self.standstill_distance),
                "Kvrel": float(self.Kvrel),
            },
        )
