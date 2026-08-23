"""Pure fractional-order PID controller for adaptive cruise control (ACC).

This module provides two controller classes with deliberately separated roles:

1. FOPIDACCController
   The main controller used in the proposed method. It is a pure FOPID-ACC
   controller driven by spacing/speed error. The optimizer vector is either:

       [Kp, Ki, Kd, lambda_, mu]
       [Kp, Ki, Kd, lambda_, mu, time_gap, standstill_distance]

   No relative-velocity damping gain is included in this main class. This keeps
   the proposed architecture aligned with the canonical FOPID form:

       u(t) = Kp e(t) + Ki D^(-lambda) e(t) + Kd D^(mu) e(t)

2. AugmentedFOPIDACCController
   Optional ablation/comparison variant. It adds a classical ACC-style relative
   velocity damping term to the pure FOPID output:

       u(t) = u_FOPID(t) + Kvrel * (v_lead(t) - v_ego(t))

   Use this only when you explicitly want to report an augmented-controller
   ablation. Do not use it as the primary FOPID formulation.

Discrete fractional approximation
----------------------------------
The fractional integral and derivative are implemented using a finite-memory
Grunwald-Letnikov approximation:

    D^alpha e[k] ~= dt^(-alpha) * sum_{j=0}^{M-1} c_j(alpha) e[k-j]

where alpha = -lambda for the fractional integral and alpha = mu for the
fractional derivative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from controllers.base import BaseACCController, require_param_count
from core.types import ACCState, ControllerOutput


__all__ = [
    "FOPIDACCController",
    "AugmentedFOPIDACCController",
]


def _gl_coefficients(alpha: float, memory_size: int) -> np.ndarray:
    """Return finite-memory Grunwald-Letnikov coefficients.

    Parameters
    ----------
    alpha:
        Fractional order. Positive values approximate derivatives; negative
        values approximate integrals.
    memory_size:
        Number of past error samples retained in the approximation window.

    Notes
    -----
    The recurrence avoids SciPy dependencies and is adequate for lightweight
    controller prototyping inside the existing metaheuristic pipeline.
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
    """Pure FOPID controller for ACC spacing/speed control.

    Parameter vector support
    ------------------------
    The optimizer may pass one of these vectors:

    1. ``[Kp, Ki, Kd, lambda_, mu]``
       Uses default time gap and standstill distance.

    2. ``[Kp, Ki, Kd, lambda_, mu, time_gap, standstill_distance]``
       Optimizes the FOPID gains and the ACC spacing policy together.

    Error convention
    ----------------
    The controller uses an acceleration-oriented sign convention:

    - positive error -> ego vehicle may accelerate;
    - negative error -> ego vehicle should decelerate.

    In spacing mode:
        ``e = d_actual - d_ref``

    In speed mode:
        ``e = v_set - v_ego``

    The default ACC behavior is spacing-prioritized. Speed mode is used only
    when the lead vehicle is not constraining ego motion.
    """

    Kp: float
    Ki: float
    Kd: float
    lambda_: float
    mu: float
    time_gap: float = 1.2
    standstill_distance: float = 20.0
    memory_size: int = 300
    output_limit: Optional[float] = None
    switch_margin: float = 0.0
    derivative_filter: float = 1.0
    name: str = "pure_fopid_acc"

    _errors: List[float] = field(default_factory=list, init=False, repr=False)
    _int_coeffs: np.ndarray = field(default=None, init=False, repr=False)
    _der_coeffs: np.ndarray = field(default=None, init=False, repr=False)
    _prev_derivative_term: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        if not (0.0 < self.lambda_ <= 2.0):
            raise ValueError("lambda_ should be in (0, 2] for a practical FOPID search range")
        if not (0.0 <= self.mu <= 2.0):
            raise ValueError("mu should be in [0, 2] for a practical FOPID search range")
        if self.time_gap <= 0.0:
            raise ValueError("time_gap must be positive")
        if self.standstill_distance < 0.0:
            raise ValueError("standstill_distance must be non-negative")
        if not (0.0 < self.derivative_filter <= 1.0):
            raise ValueError("derivative_filter must be in (0, 1]")

        self.memory_size = int(self.memory_size)
        self._int_coeffs = _gl_coefficients(alpha=-float(self.lambda_), memory_size=self.memory_size)
        self._der_coeffs = _gl_coefficients(alpha=float(self.mu), memory_size=self.memory_size)

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
        """Build a pure FOPID-ACC controller from an optimizer vector."""
        require_param_count(params, valid_counts=(5, 7), controller_name=cls.__name__)

        Kp, Ki, Kd, lambda_, mu = map(float, params[:5])
        if len(params) == 7:
            time_gap = float(params[5])
            standstill_distance = float(params[6])
        else:
            time_gap = float(default_time_gap)
            standstill_distance = float(default_standstill_distance)

        return cls(
            Kp=Kp,
            Ki=Ki,
            Kd=Kd,
            lambda_=lambda_,
            mu=mu,
            time_gap=time_gap,
            standstill_distance=standstill_distance,
            memory_size=memory_size,
            output_limit=output_limit,
            derivative_filter=derivative_filter,
        )

    def reset(self) -> None:
        """Clear controller memory before starting a new simulation."""
        self._errors.clear()
        self._prev_derivative_term = 0.0

    def desired_gap(self, v_ego: float) -> float:
        """Return constant-time-headway desired gap."""
        return self.standstill_distance + self.time_gap * float(v_ego)

    def _fractional_terms(self, error: float, dt: float) -> Tuple[float, float]:
        """Update finite memory and return fractional integral/derivative terms."""
        if dt <= 0.0:
            raise ValueError("state.dt must be positive")

        self._errors.insert(0, float(error))
        if len(self._errors) > self.memory_size:
            del self._errors[self.memory_size:]

        e = np.asarray(self._errors, dtype=float)
        n = len(e)

        # alpha = -lambda -> dt^lambda for the fractional integral.
        fractional_integral = (dt ** self.lambda_) * float(np.dot(self._int_coeffs[:n], e))

        # alpha = mu -> dt^(-mu) for the fractional derivative. If mu = 0,
        # the derivative branch reduces to the current error contribution.
        if self.mu > 0.0:
            derivative_raw = (dt ** (-self.mu)) * float(np.dot(self._der_coeffs[:n], e))
        else:
            derivative_raw = float(error)

        fractional_derivative = (
            self.derivative_filter * derivative_raw
            + (1.0 - self.derivative_filter) * self._prev_derivative_term
        )
        self._prev_derivative_term = fractional_derivative

        return fractional_integral, fractional_derivative

    def _select_error(self, state: ACCState, d_ref: float) -> Tuple[str, float]:
        """Select spacing or speed error for the FOPID block.

        Spacing mode is selected when the lead vehicle constrains the ego
        vehicle, either because the actual gap is close to/below the desired
        gap or because the lead vehicle is slower than the ego vehicle.
        """
        lead_is_relevant = state.d_actual <= d_ref + self.switch_margin or state.v_lead < state.v_ego
        if lead_is_relevant:
            return "spacing", float(state.d_actual - d_ref)
        return "speed", float(state.v_set - state.v_ego)

    def _apply_output_limit(self, u: float) -> float:
        """Clamp controller output when an acceleration/actuation limit is set."""
        if self.output_limit is None:
            return float(u)
        limit = abs(float(self.output_limit))
        return float(np.clip(u, -limit, limit))

    def _raw_fopid_components(self, state: ACCState) -> dict:
        """Compute the unclamped pure FOPID output and diagnostics."""
        d_ref = self.desired_gap(state.v_ego)
        mode, error = self._select_error(state, d_ref)
        frac_integral, frac_derivative = self._fractional_terms(error, state.dt)

        u_p = self.Kp * error
        u_i = self.Ki * frac_integral
        u_d = self.Kd * frac_derivative
        u_fopid = u_p + u_i + u_d

        return {
            "d_ref": float(d_ref),
            "mode": mode,
            "error": float(error),
            "fractional_integral": float(frac_integral),
            "fractional_derivative": float(frac_derivative),
            "u_p": float(u_p),
            "u_i": float(u_i),
            "u_d": float(u_d),
            "u_fopid": float(u_fopid),
        }

    def step(self, state: ACCState) -> ControllerOutput:
        """Compute the pure FOPID commanded acceleration/control signal."""
        comp = self._raw_fopid_components(state)
        u = self._apply_output_limit(comp["u_fopid"])

        return ControllerOutput(
            u=float(u),
            mode=comp["mode"],
            error=comp["error"],
            d_ref=comp["d_ref"],
            diagnostics={
                "controller_architecture": "pure_fopid_acc",
                "u_raw": float(comp["u_fopid"]),
                "u_fopid": float(comp["u_fopid"]),
                "u_p": float(comp["u_p"]),
                "u_i": float(comp["u_i"]),
                "u_d": float(comp["u_d"]),
                "fractional_integral": float(comp["fractional_integral"]),
                "fractional_derivative": float(comp["fractional_derivative"]),
                "Kp": float(self.Kp),
                "Ki": float(self.Ki),
                "Kd": float(self.Kd),
                "lambda": float(self.lambda_),
                "mu": float(self.mu),
                "time_gap": float(self.time_gap),
                "standstill_distance": float(self.standstill_distance),
            },
        )


@dataclass
class AugmentedFOPIDACCController(FOPIDACCController):
    """Optional FOPID-ACC variant with relative-velocity damping.

    This class is intentionally separated from ``FOPIDACCController`` so the
    main proposed controller remains a pure FOPID formulation. Use this class
    only for ablation studies or secondary comparisons.

    Optimizer vector:
        [Kp, Ki, Kd, lambda_, mu, time_gap, standstill_distance, Kvrel]
    """

    Kvrel: float = 0.0
    name: str = "augmented_fopid_acc"

    @classmethod
    def from_params(
        cls,
        params: Sequence[float],
        default_time_gap: float = 1.2,
        default_standstill_distance: float = 20.0,
        memory_size: int = 300,
        output_limit: Optional[float] = None,
        derivative_filter: float = 1.0,
    ) -> "AugmentedFOPIDACCController":
        """Build the optional augmented FOPID-ACC controller."""
        require_param_count(params, valid_counts=(8,), controller_name=cls.__name__)

        Kp, Ki, Kd, lambda_, mu = map(float, params[:5])
        time_gap = float(params[5])
        standstill_distance = float(params[6])
        Kvrel = float(params[7])

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

    def step(self, state: ACCState) -> ControllerOutput:
        """Compute FOPID output plus relative-velocity damping."""
        comp = self._raw_fopid_components(state)

        # With the sign convention used here, v_lead - v_ego < 0 means the ego
        # is faster than the lead vehicle, so a positive Kvrel contributes a
        # negative/braking command.
        u_relative_velocity = self.Kvrel * (state.v_lead - state.v_ego)
        u_raw = comp["u_fopid"] + u_relative_velocity
        u = self._apply_output_limit(u_raw)

        return ControllerOutput(
            u=float(u),
            mode=comp["mode"],
            error=comp["error"],
            d_ref=comp["d_ref"],
            diagnostics={
                "controller_architecture": "augmented_fopid_acc",
                "u_raw": float(u_raw),
                "u_fopid": float(comp["u_fopid"]),
                "u_relative_velocity": float(u_relative_velocity),
                "u_p": float(comp["u_p"]),
                "u_i": float(comp["u_i"]),
                "u_d": float(comp["u_d"]),
                "fractional_integral": float(comp["fractional_integral"]),
                "fractional_derivative": float(comp["fractional_derivative"]),
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
