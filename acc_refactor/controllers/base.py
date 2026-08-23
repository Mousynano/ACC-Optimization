"""Controller interface used by the refactored ACC simulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence

from acc_refactor.types import ACCState, ControllerOutput


class BaseACCController(ABC):
    """Base class for ACC controllers.

    A controller is intentionally small:
    - ``reset`` clears internal memory before each scenario.
    - ``step`` maps the current ACC state to one control command.

    The simulator only depends on this interface, not on controller details.
    """

    name: str = "base"

    def reset(self) -> None:
        """Reset internal memory/state before simulating a new scenario."""

    @abstractmethod
    def step(self, state: ACCState) -> ControllerOutput:
        """Return one control command for the current ACC state."""


def require_param_count(params: Sequence[float], valid_counts: Sequence[int], controller_name: str) -> None:
    """Raise a clear error when an optimizer passes an invalid parameter vector."""
    if len(params) not in valid_counts:
        valid = ", ".join(str(v) for v in valid_counts)
        raise ValueError(
            f"{controller_name} expects parameter vector length {valid}; "
            f"received {len(params)} values: {list(params)}"
        )
