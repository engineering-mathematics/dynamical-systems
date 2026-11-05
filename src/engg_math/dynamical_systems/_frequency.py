"""Frequency-domain analysis and pole summary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._errors import DomainError
from ._model import as_state_space

if TYPE_CHECKING:
    import numpy.typing as npt

    from ._model import System
    from ._types import ComplexArray, FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class FrequencyResponse:
    """``h[k]`` is the p x m matrix ``H(j w[k])``; ``w`` in rad/s."""

    w: FloatArray
    h: ComplexArray


@dataclass(frozen=True, slots=True, eq=False)
class Bode:
    """SISO magnitude in dB and unwrapped phase in degrees at ``w`` (rad/s)."""

    w: FloatArray
    magnitude_db: FloatArray
    phase_deg: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class PoleModes:
    """Pole ``poles[k]`` with natural frequency ``|p|`` and damping ratio ``-Re(p)/|p|``.

    The damping ratio is NaN for a pole at the origin.
    """

    poles: ComplexArray
    natural_frequency: FloatArray
    damping_ratio: FloatArray


def _frequencies(w: npt.ArrayLike) -> FloatArray:
    grid = np.asarray(w, dtype=float)
    if grid.ndim != 1 or grid.size == 0 or not np.all(np.isfinite(grid)):
        raise DomainError("w must be a non-empty 1-D array of finite frequencies")
    return grid


def frequency_response(system: System, w: npt.ArrayLike) -> FrequencyResponse:
    """Evaluate ``H(jw) = C (jwI - A)^-1 B + D`` by solving, never inverting."""
    ss = as_state_space(system)
    grid = _frequencies(w)
    eye = np.eye(ss.n_states)
    h = np.empty((grid.size, ss.n_outputs, ss.n_inputs), dtype=complex)
    for k, freq in enumerate(grid):
        try:
            x = np.linalg.solve(1j * freq * eye - ss.a, ss.b.astype(complex))
        except np.linalg.LinAlgError as exc:
            raise DomainError(f"w={freq} lies on a pole; H(jw) is undefined") from exc
        h[k] = ss.c @ x + ss.d
    return FrequencyResponse(grid, h)


def bode(system: System, w: npt.ArrayLike) -> Bode:
    """Magnitude (dB) and unwrapped phase (degrees) of a SISO system."""
    ss = as_state_space(system)
    if ss.n_inputs != 1 or ss.n_outputs != 1:
        raise DomainError(f"bode needs a SISO system, got {ss.n_inputs} in / {ss.n_outputs} out")
    resp = frequency_response(ss, w)
    h = resp.h[:, 0, 0]
    with np.errstate(divide="ignore"):  # an exact zero gives -inf dB
        magnitude = 20.0 * np.log10(np.abs(h))
    phase = np.degrees(np.unwrap(np.angle(h)))
    return Bode(resp.w, magnitude, phase)


def pole_modes(system: System) -> PoleModes:
    """Natural frequency and damping ratio of each pole."""
    poles = np.asarray(as_state_space(system).poles(), dtype=complex)
    wn = np.abs(poles)
    zeta = np.full(poles.shape, np.nan)
    nonzero = wn > 0.0
    zeta[nonzero] = -poles.real[nonzero] / wn[nonzero]
    return PoleModes(poles, wn, zeta)
