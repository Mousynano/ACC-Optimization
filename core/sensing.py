"""Sensor abstractions for the ACC benchmark.

The simulator owns the ground-truth lead-vehicle states. The ego controller must not
read those states directly. Instead, ground-truth longitudinal separation is passed
through an explicit range-sensor model. Relative velocity is then estimated from the
finite difference of successive *sensed* range measurements.

The nominal benchmark keeps sensor bias and random noise at zero so previously
validated optimization results remain reproducible. Noise can be enabled explicitly
for sensing-robustness studies without changing the controller interface.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional


# Constants for the deterministic, sample-indexed pseudo-random stream used by the
# sensor model. A custom stream is used so the readable Python simulator and the
# Numba backend can generate exactly the same Gaussian samples for a given seed.
_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX_MUL2 = 0x94D049BB133111EB
_STREAM_SALT = 0xD1B54A32D192ED03
_FLOAT53_INV = 1.0 / (1 << 53)
_TWO_PI = 2.0 * math.pi


def _splitmix64_hash(seed: int, sample_index: int, stream: int) -> int:
    """Return one deterministic 64-bit hash for a sensor sample.

    The sample index, rather than mutable global RNG state, determines the draw.
    This makes a noise realization reproducible regardless of unrelated random calls
    elsewhere in an optimizer.
    """
    x = (
        (int(seed) & _MASK64)
        ^ (((int(sample_index) + 1) * _SPLITMIX_GAMMA) & _MASK64)
        ^ (((int(stream) + 1) * _STREAM_SALT) & _MASK64)
    ) & _MASK64
    z = (x + _SPLITMIX_GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * _SPLITMIX_MUL1) & _MASK64
    z = ((z ^ (z >> 27)) * _SPLITMIX_MUL2) & _MASK64
    return (z ^ (z >> 31)) & _MASK64


def _uniform_open01(seed: int, sample_index: int, stream: int) -> float:
    # Use the high 53 bits to create a double-precision variate strictly inside
    # (0, 1), avoiding log(0) in the Box--Muller transform.
    value = _splitmix64_hash(seed, sample_index, stream)
    return ((value >> 11) + 0.5) * _FLOAT53_INV


def deterministic_standard_normal(seed: int, sample_index: int) -> float:
    """Return a reproducible N(0, 1) sample for ``(seed, sample_index)``."""
    u1 = max(_uniform_open01(seed, sample_index, 0), 1e-15)
    u2 = _uniform_open01(seed, sample_index, 1)
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(_TWO_PI * u2)


@dataclass(frozen=True)
class RangeSensorConfig:
    """Configuration of the range-sensing boundary.

    Parameters
    ----------
    noise_std_m:
        Standard deviation of additive zero-mean Gaussian range noise in metres.
        ``0.0`` reproduces the nominal deterministic benchmark.
    bias_m:
        Constant additive range bias in metres.
    min_distance_m:
        Lower clamp applied after bias/noise injection. A nonnegative default keeps
        the controller from receiving an impossible negative range.
    seed:
        Seed for the deterministic sample-indexed noise stream. The same seed and
        sampling schedule produce the same measurement sequence in Python and Numba.
    """

    noise_std_m: float = 0.0
    bias_m: float = 0.0
    min_distance_m: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        if self.noise_std_m < 0.0:
            raise ValueError("noise_std_m must be nonnegative")
        if not math.isfinite(self.noise_std_m):
            raise ValueError("noise_std_m must be finite")
        if not math.isfinite(self.bias_m):
            raise ValueError("bias_m must be finite")
        if not math.isfinite(self.min_distance_m):
            raise ValueError("min_distance_m must be finite")


@dataclass(frozen=True)
class RangeSensorReading:
    """One ego-observable sensor sample plus diagnostic metadata."""

    distance: float
    relative_velocity: float
    rate_valid: bool
    injected_noise: float = 0.0
    measurement_error: float = 0.0


class RangeOnlySensor:
    """Range-only sensor with finite-difference range-rate estimation.

    Measurement model
    -----------------
    ``d_sensed = max(min_distance, d_true + bias + noise)``

    with ``noise ~ N(0, noise_std_m^2)`` when noise is enabled.

    Sign convention for relative velocity
    -------------------------------------
    ``v_rel = d(d_sensed) / dt ~= v_lead - v_ego``

    Therefore:
        ``v_rel > 0`` -> gap is opening
        ``v_rel < 0`` -> gap is closing

    The first sample has no previous range measurement, so its range rate is marked
    invalid and returned as zero instead of using ground-truth lead velocity.
    """

    def __init__(self, dt: float, config: Optional[RangeSensorConfig] = None):
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.dt = float(dt)
        self.config = config if config is not None else RangeSensorConfig()
        self._prev_distance: Optional[float] = None
        self._sample_index = 0

    def reset(self) -> None:
        self._prev_distance = None
        self._sample_index = 0

    def sense(self, distance_ground_truth: float) -> RangeSensorReading:
        distance_ground_truth = float(distance_ground_truth)
        if not math.isfinite(distance_ground_truth):
            raise ValueError("distance_ground_truth must be finite")

        injected_noise = 0.0
        if self.config.noise_std_m > 0.0:
            injected_noise = (
                self.config.noise_std_m
                * deterministic_standard_normal(self.config.seed, self._sample_index)
            )

        raw_measurement = (
            distance_ground_truth
            + self.config.bias_m
            + injected_noise
        )
        distance = max(self.config.min_distance_m, raw_measurement)

        if self._prev_distance is None:
            relative_velocity = 0.0
            rate_valid = False
        else:
            relative_velocity = (distance - self._prev_distance) / self.dt
            rate_valid = True

        self._prev_distance = distance
        self._sample_index += 1

        return RangeSensorReading(
            distance=distance,
            relative_velocity=relative_velocity,
            rate_valid=rate_valid,
            injected_noise=injected_noise,
            measurement_error=distance - distance_ground_truth,
        )
