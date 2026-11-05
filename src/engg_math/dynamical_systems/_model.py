"""Model coercion shared by the analysis functions."""

from __future__ import annotations

from typing import TypeAlias

from engg_math.continuous_analysis import StateSpace, TransferFunction

from ._errors import DomainError

System: TypeAlias = StateSpace | TransferFunction


def as_state_space(system: System) -> StateSpace:
    """Return ``system`` as a state-space model."""
    if isinstance(system, StateSpace):
        return system
    if isinstance(system, TransferFunction):
        return system.to_state_space()
    raise DomainError(f"expected StateSpace or TransferFunction, got {type(system).__name__}")
