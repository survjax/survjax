"""Two-level nested parameter isolation (spec 7.1/7.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL
from jax.typing import ArrayLike

import survjax
from survjax._likelihood_parameters import stop_invalid_gradients


@survjax.register_family
@dataclass(frozen=True)
class NestedSurvival(survjax.SurvivalDistribution):
    base: survjax.SurvivalDistribution = survjax.distribution()
    log_multiplier: jax.Array = survjax.param(constraint=survjax.constraints.real)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        return jnp.exp(self.log_multiplier) * self.base.log_survival(t)

    def reference_time(self) -> jax.Array:
        return self.base.reference_time()


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "outer,inner",
    [(None, None), (jax.jacfwd, None), (jax.jacrev, None)]
    + [(a, b) for a in (jax.jacfwd, jax.jacrev) for b in (jax.jacfwd, jax.jacrev)],
    ids=[
        "value",
        "fwd",
        "rev",
        "fwd-over-fwd",
        "fwd-over-rev",
        "rev-over-fwd",
        "rev-over-rev",
    ],
)
def test_two_nested_levels_preserve_valid_rows(outer, inner, compiled, x64_enabled):
    """Spec 7.1/7.3: a deepest NaN cannot reach either shared wrapper parameter."""
    keep = jnp.array([0, 2])
    time = jnp.array([1.0, 2.0, 3.0])
    parameters = jnp.array([0.5, np.nan, 2.0, 0.2, 0.3])

    def together(p):
        baseline = survjax.Weibull(shape=1.5, scale=p[:3])
        middle = NestedSurvival(base=baseline, log_multiplier=p[3])
        dist = NestedSurvival(base=middle, log_multiplier=p[4])
        prepared = stop_invalid_gradients(dist, dist.parameter_issues().invalid_entries)
        return prepared.log_survival(time)[keep]

    def isolated_formula(p):
        # Weibull log S=-(t/scale)**1.5, multiplied by exp(a)*exp(b).
        # Only valid scales enter this formula, so the bad scale is unused.
        return -jnp.exp(p[3] + p[4]) * (time[keep] / p[keep]) ** 1.5

    actual, expected = together, isolated_formula
    if inner is not None:
        actual, expected = inner(actual), inner(expected)
    if outer is not None:
        actual, expected = outer(actual), outer(expected)
    if compiled:
        actual = jax.jit(actual)
    value = actual(parameters)
    assert np.isfinite(value).all()
    np.testing.assert_allclose(
        value, expected(parameters), atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
