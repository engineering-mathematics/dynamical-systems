"""Frequency response and pole modes."""

from __future__ import annotations

import numpy as np
import pytest

from engg_math.continuous_analysis import StateSpace, TransferFunction
from engg_math.dynamical_systems import (
    DomainError,
    bode,
    frequency_response,
    pole_modes,
)


def test_frequency_response_matches_transfer_function() -> None:
    """REQ-DY-020: H(jw) equals G(jw) evaluated directly."""
    g = TransferFunction([1.0, 2.0], [1.0, 3.0, 5.0])
    w = np.logspace(-2, 2, 25)
    np.testing.assert_allclose(frequency_response(g, w).h[:, 0, 0], g(1j * w), rtol=1e-10)


def test_first_order_bode_corner() -> None:
    """REQ-DY-021: 1/(s+1) is -3.01 dB and -45 deg at w = 1."""
    b = bode(TransferFunction([1.0], [1.0, 1.0]), [1.0])
    assert b.magnitude_db[0] == pytest.approx(-10 * np.log10(2.0))
    assert b.phase_deg[0] == pytest.approx(-45.0)


def test_phase_is_unwrapped() -> None:
    """REQ-DY-022: a triple lag runs past -180 degrees."""
    g = TransferFunction([1.0], np.poly([-1.0, -1.0, -1.0]))
    b = bode(g, np.logspace(-2, 3, 400))
    assert b.phase_deg[-1] < -180.0
    assert np.all(np.diff(b.phase_deg) < 1.0)


def test_zero_gain_gives_minus_infinity_db() -> None:
    """REQ-DY-023: exact zero magnitude maps to -inf without warnings."""
    b = bode(TransferFunction([1.0, 0.0], [1.0, 1.0]), [0.0])
    assert b.magnitude_db[0] == -np.inf


def test_bode_rejects_mimo() -> None:
    """REQ-DY-024: bode is SISO only."""
    ss = StateSpace([[-1.0]], [[1.0, 1.0]], [[1.0]], [[0.0, 0.0]])
    with pytest.raises(DomainError):
        bode(ss, [1.0])


def test_frequency_on_pole_raises() -> None:
    """REQ-DY-025: an imaginary-axis pole makes H(jw) undefined."""
    g = TransferFunction([1.0], [1.0, 0.0, 1.0])
    with pytest.raises(DomainError):
        frequency_response(g, [1.0])


@pytest.mark.parametrize("w", [[], [[1.0]], [np.nan], [np.inf]])
def test_invalid_frequency_grid_raises(w: list[object]) -> None:
    """REQ-DY-026: w must be a finite non-empty 1-D array."""
    with pytest.raises(DomainError):
        frequency_response(TransferFunction([1.0], [1.0, 1.0]), w)  # type: ignore[arg-type]


def test_mimo_frequency_response_shape() -> None:
    """REQ-DY-027: h is (N, outputs, inputs)."""
    ss = StateSpace(-np.eye(2), np.eye(2), np.eye(2), np.zeros((2, 2)))
    assert frequency_response(ss, [0.5, 1.0, 2.0]).h.shape == (3, 2, 2)


def test_pole_modes_underdamped_pair() -> None:
    """REQ-DY-028: s^2 + 2 zeta wn s + wn^2 recovers wn and zeta."""
    wn, zeta = 3.0, 0.2
    m = pole_modes(TransferFunction([1.0], [1.0, 2 * zeta * wn, wn**2]))
    np.testing.assert_allclose(m.natural_frequency, [wn, wn])
    np.testing.assert_allclose(m.damping_ratio, [zeta, zeta])


def test_pole_at_origin_has_nan_damping() -> None:
    """REQ-DY-029: damping ratio is undefined at the origin."""
    m = pole_modes(TransferFunction([1.0], [1.0, 0.0]))
    assert m.natural_frequency[0] == 0.0
    assert np.isnan(m.damping_ratio[0])
