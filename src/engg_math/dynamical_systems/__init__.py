"""Time response, frequency response and structural analysis of LTI systems."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from ._errors import DomainError
from ._frequency import Bode, FrequencyResponse, PoleModes, bode, frequency_response, pole_modes
from ._structure import controllability_matrix, is_controllable, is_observable, observability_matrix
from ._time import Response, dc_gain, impulse_response, simulate, step_response

try:
    __version__ = version("engg-math-dynamical-systems")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "Bode",
    "DomainError",
    "FrequencyResponse",
    "PoleModes",
    "Response",
    "__version__",
    "bode",
    "controllability_matrix",
    "dc_gain",
    "frequency_response",
    "impulse_response",
    "is_controllable",
    "is_observable",
    "observability_matrix",
    "pole_modes",
    "simulate",
    "step_response",
]
