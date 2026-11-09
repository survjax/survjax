"""Truncation integration: spec 2, 4.2 and 7.1–7.3."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_mixed_truncated_values_and_transformations(family, x64_enabled):
    """Spec 2/7.2: exact, right, left, partial interval and full coverage."""
    data = survjax.SurvivalData.from_bounds(
        [2.0, 2.0, 0.0, 0.5, 1.0],
        [2.0, np.inf, 2.0, 2.0, np.inf],
        exact=[True, False, False, False, False],
        left_trunc=1.0,
        right_trunc=3.0,
    )
    if family == "exponential":
        dist = survjax.Exponential(rate=0.5)
        # H=t/2; Z=S(1)-S(3); f(2)=.5*exp(-1).
        h1, h2, h3, log_f = 0.5, 1.0, 1.5, np.log(0.5) - 1
    else:
        dist = survjax.Weibull(shape=2.0, scale=2.0)
        # H=t^2/4; f(2)=(2*2/4)*exp(-1).
        h1, h2, h3, log_f = 0.25, 1.0, 2.25, -1.0
    z = np.exp(-h1) - np.exp(-h3)
    expected = [
        log_f - np.log(z),
        np.log((np.exp(-h2) - np.exp(-h3)) / z),
        np.log((np.exp(-h1) - np.exp(-h2)) / z),
        np.log((np.exp(-h1) - np.exp(-h2)) / z),
        0.0,
    ]
    with jax.debug_nans(True):
        actual = dist.log_likelihood(data)
    for value in (actual, jax.jit(lambda d: d.log_likelihood(data))(dist)):
        np.testing.assert_allclose(
            value, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
    draws = jax.tree.map(lambda p: jnp.stack([p, p]), dist)
    values = jax.jit(jax.vmap(lambda d: d.log_likelihood(data)))(draws)
    np.testing.assert_allclose(
        values, [expected, expected], atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


def test_delayed_entry_values_gradient_and_hessian(x64_enabled):
    """Spec 2/7.3: memorylessness gives exact log(r)-r(t-L), right -r(c-L)."""
    data = survjax.SurvivalData([3.0, 4.0, 2.0], [True, False, False], left_trunc=2.0)

    def rows(rate):
        return survjax.Exponential(rate=rate).log_likelihood(data)

    rate = jnp.asarray(0.5)
    np.testing.assert_allclose(
        rows(rate),
        [np.log(0.5) - 0.5, -1, 0],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(first(rows))(rate),
            [1, -2, 0],
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        for second in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                second(first(rows))(rate),
                [-4, 0, 0],
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )


def test_invalid_intersections_remain_nan():
    """Spec 2/7.1: right before entry and empty intervals are invalid data."""
    data = survjax.SurvivalData.from_bounds(
        [0.5, 0.0, 1.0],
        [np.inf, 1.0, 3.0],
        left_trunc=1.0,
        right_trunc=3.0,
    )
    for fn in (
        lambda d: d.log_likelihood(data),
        jax.jit(lambda d: d.log_likelihood(data)),
    ):
        values = fn(survjax.Exponential(rate=0.5))
        assert np.isnan(values[:2]).all()
        assert values[2] == 0


@survjax.register_family
@dataclass(frozen=True)
class ConditionalProbe(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def reference_time(self):
        return jnp.ones_like(self.rate)

    def log_prob(self, t):
        return -self.rate * t

    def log_interval_prob(self, lower, upper):
        return -self.rate * lower

    def log_conditional_interval_prob(self, lower, upper, left_trunc, right_trunc):
        # A dispatch probe also rejects an inactive empty set on exact rows.
        valid = (lower >= left_trunc) & (upper <= right_trunc) & (upper > lower)
        return jnp.where(valid, -self.rate * (lower - left_trunc), jnp.nan)


def test_conditional_hook_receives_intersection_and_safe_exact_window():
    """Spec 4.2/7.2: evaluator calls the hook without family-type dispatch."""
    data = survjax.SurvivalData.from_bounds(
        [2.0, 0.5, 2.0],
        [2.0, 2.0, 4.0],
        exact=[True, False, False],
        left_trunc=1.0,
        right_trunc=3.0,
    )
    with jax.debug_nans(True):
        values = ConditionalProbe(rate=1.0).log_likelihood(data)
    np.testing.assert_array_equal(values, [-1, 0, -1])
