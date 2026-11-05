"""Controllability and observability."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from engg_math.continuous_analysis import StateSpace, TransferFunction
from engg_math.dynamical_systems import (
    DomainError,
    controllability_matrix,
    is_controllable,
    is_observable,
    observability_matrix,
)


def test_controllability_matrix_columns() -> None:
    """REQ-DY-001: columns are B, AB, A^2 B."""
    a = [[0.0, 1.0], [-2.0, -3.0]]
    ss = StateSpace(a, [[0.0], [1.0]], [[1.0, 0.0]], [[0.0]])
    np.testing.assert_allclose(controllability_matrix(ss), [[0.0, 1.0], [1.0, -3.0]], atol=1e-15)


def test_observability_matrix_rows() -> None:
    """REQ-DY-002: rows are C, CA."""
    ss = StateSpace([[0.0, 1.0], [-2.0, -3.0]], [[0.0], [1.0]], [[1.0, 0.0]], [[0.0]])
    np.testing.assert_allclose(observability_matrix(ss), [[1.0, 0.0], [0.0, 1.0]], atol=1e-15)


def test_uncontrollable_system_detected() -> None:
    """REQ-DY-003: a decoupled state with no input is uncontrollable."""
    ss = StateSpace(np.diag([-1.0, -2.0]), [[1.0], [0.0]], [[1.0, 1.0]], [[0.0]])
    assert not is_controllable(ss)
    assert is_observable(ss)


def test_unobservable_system_detected() -> None:
    """REQ-DY-004: a state that never reaches the output is unobservable."""
    ss = StateSpace(np.diag([-1.0, -2.0]), [[1.0], [1.0]], [[1.0, 0.0]], [[0.0]])
    assert is_controllable(ss)
    assert not is_observable(ss)


def test_transfer_function_realisation_is_minimal() -> None:
    """REQ-DY-005: the canonical realisation of a coprime TF is controllable and observable."""
    g = TransferFunction([1.0, 3.0], [1.0, 3.0, 2.0, 5.0])
    assert is_controllable(g)
    assert is_observable(g)


def test_pole_zero_cancellation_loses_observability() -> None:
    """REQ-DY-006: (s+1)/((s+1)(s+2)) has a hidden mode."""
    g = TransferFunction([1.0, 1.0], [1.0, 3.0, 2.0])
    assert not (is_controllable(g) and is_observable(g))


def test_rejects_unsupported_model() -> None:
    """REQ-DY-007: arbitrary objects are rejected."""
    with pytest.raises(DomainError):
        is_controllable("not a system")  # type: ignore[arg-type]


@settings(max_examples=40, deadline=None)
@given(st.lists(st.integers(1, 8), min_size=2, max_size=4, unique=True))
def test_distinct_real_poles_with_dense_input_are_controllable(poles: list[int]) -> None:
    """REQ-DY-008: diagonal A, distinct integer poles, all-ones B and C: fully reachable."""
    n = len(poles)
    ss = StateSpace(np.diag([-float(p) for p in poles]), np.ones((n, 1)), np.ones((1, n)), [[0.0]])
    assert is_controllable(ss, tol=1e-9)
    assert is_observable(ss, tol=1e-9)
