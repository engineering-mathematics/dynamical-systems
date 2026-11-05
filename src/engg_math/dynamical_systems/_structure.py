"""Kalman controllability and observability tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from engg_math.linear_systems import DenseMatrix, rank

from ._model import as_state_space

if TYPE_CHECKING:
    from ._model import System
    from ._types import Array


def controllability_matrix(system: System) -> Array:
    """Return ``[B, AB, ..., A^(n-1) B]`` (n x n*m)."""
    ss = as_state_space(system)
    blocks = [np.asarray(ss.b)]
    for _ in range(ss.n_states - 1):
        blocks.append(ss.a @ blocks[-1])
    return np.asfortranarray(np.hstack(blocks))


def observability_matrix(system: System) -> Array:
    """Return ``[C; CA; ...; C A^(n-1)]`` (n*p x n)."""
    ss = as_state_space(system)
    blocks = [np.asarray(ss.c)]
    for _ in range(ss.n_states - 1):
        blocks.append(blocks[-1] @ ss.a)
    return np.asfortranarray(np.vstack(blocks))


def is_controllable(system: System, tol: float | None = None) -> bool:
    """Whether the controllability matrix has full row rank.

    The Kalman rank test is ill-conditioned as the order grows; ``tol`` is the SVD rank tolerance
    (default: the NumPy machine-precision rule).
    """
    ss = as_state_space(system)
    return rank(DenseMatrix(controllability_matrix(ss)), tol) == ss.n_states


def is_observable(system: System, tol: float | None = None) -> bool:
    """Whether the observability matrix has full column rank. See `is_controllable` on ``tol``."""
    ss = as_state_space(system)
    return rank(DenseMatrix(observability_matrix(ss)), tol) == ss.n_states
