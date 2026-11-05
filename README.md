# engg-math-dynamical-systems

Time response, frequency response, pole analysis, controllability and observability of LTI systems
for `engg_math`. Accepts `StateSpace` or `TransferFunction` from `engg-math-continuous-analysis`.

```bash
pip install engg-math-dynamical-systems
```

```python
from engg_math.continuous_analysis import Domain, TransferFunction
from engg_math.dynamical_systems import bode, is_controllable, step_response

g = TransferFunction([1.0], [1.0, 2.0, 1.0])  # 1 / (s + 1)^2
step_response(g, Domain(0.0, 10.0, 201)).y[-1]  # ~1.0
bode(g, [0.1, 1.0, 10.0]).magnitude_db
is_controllable(g)  # True
```

## API

| Area | Names |
|---|---|
| Time domain | `simulate`, `step_response`, `impulse_response`, `Response`, `dc_gain` |
| Frequency domain | `frequency_response`, `bode`, `FrequencyResponse`, `Bode` |
| Poles | `pole_modes`, `PoleModes` (natural frequency, damping ratio) |
| Structure | `controllability_matrix`, `observability_matrix`, `is_controllable`, `is_observable` |
| Errors | `DomainError` (subclass of `EnggMathError`) |

## Conventions

- Arrays are indexed `[time, channel]`: `Response.y` is `(N, outputs)`, `Response.x` is `(N, states)`.
- `H(jw)` and `dc_gain` use linear solves, never inversion. A frequency on a pole, or a singular `A` for
  `dc_gain`, raises `DomainError`.
- The impulse response is the free motion from `x0 = B[:, j]`; the `D delta(t)` term is omitted.
- Rank tests use SVD rank from `engg-math-linear-systems`. The Kalman matrices are ill-conditioned for
  high orders; pass `tol` explicitly when it matters.
- Forced responses use the adaptive DOP853 solver of `LinearODE`.

## Scope of 0.1

The legacy scipy `lti` wrapper and plotext plotters are not included; plotting returns with `engg-math-viz`.

## Development

```bash
uv sync
uv run pytest && uv run ruff check && uv run mypy && uv run lint-imports
```

Licensed under Apache-2.0.
