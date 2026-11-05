"""Reference-solution tests against scipy.signal."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import signal as _signal

from engg_math.continuous_analysis import Domain, TransferFunction
from engg_math.dynamical_systems import bode, impulse_response, step_response

signal: Any = _signal  # scipy-stubs differ between releases

NUM, DEN = [4.0, 2.0], [1.0, 2.5, 6.0, 4.0]
DOMAIN = Domain(0.0, 8.0, 161)


def test_step_matches_scipy() -> None:
    """REQ-DY-030: step response agrees with scipy.signal.step."""
    t, y = signal.step(signal.lti(NUM, DEN), T=DOMAIN.grid())
    np.testing.assert_allclose(
        step_response(TransferFunction(NUM, DEN), DOMAIN).y[:, 0], y, rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(t, DOMAIN.grid())


def test_impulse_matches_scipy() -> None:
    """REQ-DY-031: impulse response agrees with scipy.signal.impulse."""
    _, y = signal.impulse(signal.lti(NUM, DEN), T=DOMAIN.grid())
    np.testing.assert_allclose(
        impulse_response(TransferFunction(NUM, DEN), DOMAIN).y[:, 0], y, rtol=1e-6, atol=1e-8
    )


def test_bode_matches_scipy() -> None:
    """REQ-DY-032: magnitude and phase agree with scipy.signal.bode."""
    w = np.logspace(-2, 2, 60)
    _, mag, phase = signal.bode(signal.lti(NUM, DEN), w)
    result = bode(TransferFunction(NUM, DEN), w)
    np.testing.assert_allclose(result.magnitude_db, mag, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(result.phase_deg, phase, rtol=1e-8, atol=1e-6)
