"""Time-domain responses."""

from __future__ import annotations

import numpy as np
import pytest

from engg_math.continuous_analysis import Domain, StateSpace, TransferFunction
from engg_math.core import ShapeError
from engg_math.dynamical_systems import (
    DomainError,
    dc_gain,
    impulse_response,
    simulate,
    step_response,
)

DOMAIN = Domain(0.0, 5.0, 101)


def test_first_order_step_matches_closed_form() -> None:
    """REQ-DY-010: 1/(s+1) step is 1 - exp(-t)."""
    r = step_response(TransferFunction([1.0], [1.0, 1.0]), DOMAIN)
    np.testing.assert_allclose(r.y[:, 0], 1.0 - np.exp(-r.t), rtol=1e-6, atol=1e-8)


def test_first_order_impulse_matches_closed_form() -> None:
    """REQ-DY-011: 1/(s+1) impulse is exp(-t)."""
    r = impulse_response(TransferFunction([1.0], [1.0, 1.0]), DOMAIN)
    np.testing.assert_allclose(r.y[:, 0], np.exp(-r.t), rtol=1e-9)


def test_feedthrough_appears_in_step_output() -> None:
    """REQ-DY-012: y = C x + D u includes D."""
    ss = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[2.0]])
    r = step_response(ss, DOMAIN)
    assert r.y[0, 0] == pytest.approx(2.0)
    assert r.y[-1, 0] == pytest.approx(3.0, rel=1e-2)


def test_free_response_from_initial_state() -> None:
    """REQ-DY-013: autonomous motion is x0 * exp(-t)."""
    ss = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
    r = simulate(ss, [2.0], DOMAIN)
    np.testing.assert_allclose(r.y[:, 0], 2.0 * np.exp(-r.t), rtol=1e-9)


def test_time_varying_input() -> None:
    """REQ-DY-014: callable input is sampled on the grid."""
    ss = StateSpace([[0.0]], [[1.0]], [[1.0]], [[0.0]])
    r = simulate(ss, [0.0], DOMAIN, u=lambda t: [1.0 if t >= 0 else 0.0])
    np.testing.assert_allclose(r.y[:, 0], r.t, rtol=1e-6, atol=1e-8)


def test_multi_input_selects_channel() -> None:
    """REQ-DY-015: input_index drives one channel only."""
    ss = StateSpace([[-1.0]], [[1.0, 10.0]], [[1.0]], [[0.0, 0.0]])
    first = step_response(ss, DOMAIN, 0).y[-1, 0]
    second = step_response(ss, DOMAIN, 1).y[-1, 0]
    assert second == pytest.approx(10.0 * first, rel=1e-6)


def test_dc_gain_matches_transfer_function() -> None:
    """REQ-DY-016: dc gain agrees with G(0)."""
    g = TransferFunction([2.0], [1.0, 4.0])
    np.testing.assert_allclose(dc_gain(g), [[g.dc_gain()]])


def test_dc_gain_singular_a_raises() -> None:
    """REQ-DY-017: an integrator has no dc gain."""
    with pytest.raises(DomainError):
        dc_gain(TransferFunction([1.0], [1.0, 0.0]))


def test_invalid_input_index_raises() -> None:
    """REQ-DY-018: input_index outside range is rejected."""
    g = TransferFunction([1.0], [1.0, 1.0])
    with pytest.raises(DomainError):
        step_response(g, DOMAIN, 1)
    with pytest.raises(DomainError):
        impulse_response(g, DOMAIN, -1)


def test_wrong_input_shape_raises() -> None:
    """REQ-DY-019: input dimension must match B."""
    ss = StateSpace([[-1.0]], [[1.0]], [[1.0]], [[0.0]])
    with pytest.raises(ShapeError):
        simulate(ss, [0.0], DOMAIN, u=[1.0, 2.0])
