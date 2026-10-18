"""Base-owned boundaries of interior-only primitives (spec 4.5/7.2, rev 27)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL
from jax.typing import ArrayLike

import survjax


@survjax.register_family
@dataclass(frozen=True)
class InteriorOnly(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param()

    def log_survival(self, t: ArrayLike) -> jax.Array:
        return -self.rate * t * (t / t)

    def reference_time(self) -> jax.Array:
        return jnp.ones_like(self.rate)


@survjax.register_family
@dataclass(frozen=True)
class LogLogistic(InteriorOnly):
    def log_survival(self, t: ArrayLike) -> jax.Array:
        return -jax.nn.softplus(self.rate * jnp.log(t))


@pytest.mark.parametrize("family", [InteriorOnly, LogLogistic])
@pytest.mark.parametrize(
    "name,zero",
    [
        ("log_survival", 0),
        ("survival", 1),
        ("log_cdf", -np.inf),
        ("cumulative_hazard", 0),
    ],
)
def test_base_boundary_values_and_parameter_gradients(family, name, zero):
    """Spec 4.5: zero-time constants and proper finite infinity limits have zero gradients."""
    # S(0)=1; proper S(inf)=0. No derivative of an infinite infinity value is required.
    cases = [(0.0, zero)]
    if name in ("survival", "log_cdf"):
        cases.append((jnp.inf, 0.0))
    for time, expected in cases:

        def value(rate, time=time):
            return getattr(family(rate=rate), name)(time)

        assert value(2.0) == expected
        for derivative in (jax.jacfwd(value), jax.jacrev(value)):
            np.testing.assert_array_equal(derivative(2.0), 0.0)
            np.testing.assert_array_equal(
                jax.jit(jax.vmap(derivative))(jnp.array([2.0, 3.0])), [0.0, 0.0]
            )


@pytest.mark.parametrize("family", [InteriorOnly, LogLogistic])
def test_mixed_grid_finite_derivative_isolated_from_boundaries(family, x64_enabled):
    """Spec 5/7.2: an interior gradient survives zero/infinity slots under debug_nans."""
    # At t=2: d(-r*t)/dr=-2; d[-log(1+t**r)]/dr=-4*log(2)/5 at r=2.
    expected = -2.0 if family is InteriorOnly else -4.0 * np.log(2.0) / 5.0
    times = jnp.array([0.0, 2.0, jnp.inf])

    def value(rate):
        return family(rate=rate).log_survival_grid(times)[1]

    with jax.debug_nans(True):
        for derivative in (jax.jacfwd(value), jax.jacrev(value)):
            np.testing.assert_allclose(
                jax.jit(derivative)(2.0),
                expected,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
        np.testing.assert_array_equal(
            family(rate=[2.0, 3.0]).survival_grid(times)[[0, 2], :], [[1, 1], [0, 0]]
        )
