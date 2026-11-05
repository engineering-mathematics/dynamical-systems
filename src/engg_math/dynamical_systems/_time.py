"""Time-domain responses of LTI systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from engg_math.continuous_analysis import LinearODE

from ._errors import DomainError
from ._model import as_state_space

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt

    from engg_math.continuous_analysis import Domain

    from ._model import System
    from ._types import Array, FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class Response:
    """Sampled response: ``y[k]`` and ``x[k]`` are the output and state at ``t[k]``."""

    t: FloatArray
    y: FloatArray
    x: FloatArray


# Shapes are validated by LinearODE.solve before this runs.
def _input_samples(
    u: Callable[[float], npt.ArrayLike] | npt.ArrayLike, t: FloatArray
) -> FloatArray:
    rows = [
        np.atleast_1d(np.asarray(u(float(time)) if callable(u) else u, dtype=float)) for time in t
    ]
    samples: FloatArray = np.vstack(rows)
    return samples


def simulate(
    system: System,
    x0: npt.ArrayLike,
    domain: Domain,
    u: Callable[[float], npt.ArrayLike] | npt.ArrayLike | None = None,
) -> Response:
    """Simulate ``x' = A x + B u``, ``y = C x + D u`` from ``x0`` on ``domain``.

    ``u`` is a constant vector or a function of time; ``None`` means zero input.
    """
    ss = as_state_space(system)
    traj = LinearODE(ss.a, ss.b).solve(x0, domain, u)
    y: FloatArray = traj.x @ ss.c.T
    if u is not None:
        y = y + _input_samples(u, traj.t) @ ss.d.T
    return Response(traj.t, y, traj.x)


def _check_input_index(n_inputs: int, input_index: int) -> None:
    if not 0 <= input_index < n_inputs:
        raise DomainError(f"input_index must be in [0, {n_inputs}), got {input_index}")


def step_response(system: System, domain: Domain, input_index: int = 0) -> Response:
    """Response to a unit step on one input, from rest."""
    ss = as_state_space(system)
    _check_input_index(ss.n_inputs, input_index)
    u = np.zeros(ss.n_inputs)
    u[input_index] = 1.0
    return simulate(ss, np.zeros(ss.n_states), domain, u)


def impulse_response(system: System, domain: Domain, input_index: int = 0) -> Response:
    """Response to a unit impulse on one input: free motion from ``x0 = B[:, input_index]``.

    The direct-feedthrough term ``D * delta(t)`` is not representable on a grid and is omitted.
    """
    ss = as_state_space(system)
    _check_input_index(ss.n_inputs, input_index)
    return simulate(ss, np.asarray(ss.b)[:, input_index], domain)


def dc_gain(system: System) -> Array:
    """Steady-state gain ``D - C A^-1 B`` (p x m), by solve. Needs ``A`` nonsingular."""
    ss = as_state_space(system)
    try:
        x = np.linalg.solve(ss.a, ss.b)
    except np.linalg.LinAlgError as exc:
        raise DomainError("dc gain is undefined: A is singular (pole at the origin)") from exc
    gain: Array = ss.d - ss.c @ x
    return gain
