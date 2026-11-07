"""Generic interval hooks, using analytically known exponential probabilities."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import (
    ATOL,
    DERIVATIVE_ATOL,
    DERIVATIVE_RTOL,
    LIKELIHOOD_GRAD_EPS,
    RTOL,
)
from jax.test_util import check_grads

from survjax import Exponential


def test_interval_values_and_transformations(x64_enabled):
    """Spec 4.2/7.3: tail-selected differences agree with exponential algebra."""
    a = jnp.array([0.0, 0.5, 2.0, 10.0])
    b = jnp.array([1.0, 1.5, 3.0, jnp.inf])
    # S(a)-S(b)=exp(-2a)*(1-exp(-2(b-a))).
    expected = -2 * np.asarray(a) + np.log(
        -np.expm1(-2 * (np.asarray(b) - np.asarray(a)))
    )
    f = Exponential(rate=2.0).log_interval_prob
    for result in (f(a, b), jax.jit(f)(a, b), jax.vmap(f)(a, b)):
        np.testing.assert_allclose(
            result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_interval_endpoint_policy():
    """Spec 7.1: equal endpoints are zero mass; reversed endpoints invalid."""
    f = Exponential(rate=1.0).log_interval_prob
    result = jax.jit(f)(
        jnp.array([1.0, 2.0, -1.0, jnp.inf]), jnp.array([1.0, 1.0, 2.0, jnp.inf])
    )
    assert np.isneginf(result[0]) and np.isneginf(result[3])
    assert np.isnan(result[1:3]).all()


def test_conditional_intersections(x64_enabled):
    """Spec 7.1: partial, empty and full intersections with two-sided window."""
    f = Exponential(rate=1.0).log_conditional_interval_prob
    a, b = jnp.array([0.0, 3.0, 0.0]), jnp.array([2.0, 4.0, 4.0])
    # Window (1,3]; partial numerator (1,2]. Cancel exp(-1).
    expected = [np.log(-np.expm1(-1.0)) - np.log(-np.expm1(-2.0)), -np.inf, 0.0]
    for result in (f(a, b, 1.0, 3.0), jax.jit(f)(a, b, 1.0, 3.0)):
        np.testing.assert_allclose(
            result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
        assert result[2] == 0
    assert np.isnan(f(0.0, 4.0, 2.0, 1.0))


@pytest.mark.parametrize("conditional", [False, True])
def test_interval_parameter_derivatives(conditional, x64_enabled):
    """Spec 7.3: generic hooks differentiate twice on smooth interiors."""

    def f(rate):
        dist = Exponential(rate=rate)
        if conditional:
            return dist.log_conditional_interval_prob(0.5, 1.5, 0.2, 2.0)
        return dist.log_interval_prob(0.5, 1.5)

    check_grads(
        jax.jit(f),
        (jnp.asarray(1.2),),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
    )


def test_lower_tail_and_finite_far_tail(x64_enabled):
    """Spec 4.2/7.2: both difference branches and narrow intervals retain mass."""
    a = jnp.array([0.01, 1e-10, 1000.0])
    b = jnp.array([0.02, 1.001e-10, 1001.0])
    # Evaluate the analytic exponential formula in float64 at represented inputs.
    lower, upper = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    expected = -lower + np.log(-np.expm1(-(upper - lower)))
    result = jax.jit(Exponential(rate=1.0).log_interval_prob)(a, b)
    np.testing.assert_allclose(
        result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


def test_invalid_parameters_and_nan_endpoints():
    """Spec 7.1: invalid calculations must not be relabelled zero mass."""
    result = Exponential(rate=jnp.array([jnp.nan, -1.0])).log_interval_prob(0.5, 1.0)
    assert np.isnan(result).all()
    assert np.isnan(Exponential(rate=1.0).log_interval_prob(jnp.nan, 1.0))
    assert np.isnan(Exponential(rate=1.0).log_interval_prob(0.0, jnp.nan))


def test_bounded_support_mass_taxonomy():
    """Spec 7.1: zero numerator differs from zero conditioning mass."""
    from dataclasses import dataclass

    from survjax import SurvivalDistribution, param, register_family

    @register_family
    @dataclass(frozen=True)
    class Uniform(SurvivalDistribution):
        endpoint: jax.Array = param()

        def reference_time(self):
            return self.endpoint / 2

        def log_survival(self, t):
            inside = t < self.endpoint
            safe = jnp.where(inside, t / self.endpoint, 0.0)
            return jnp.where(inside, jnp.log1p(-safe), -jnp.inf)

    dist = Uniform(endpoint=1.0)
    assert np.isneginf(dist.log_interval_prob(2.0, 3.0))
    assert np.isneginf(dist.log_conditional_interval_prob(2.0, 3.0, 0.0, 4.0))
    assert np.isnan(dist.log_conditional_interval_prob(0.0, 4.0, 2.0, 3.0))


def test_conditional_window_boundaries_and_infinity(x64_enabled):
    """Spec 7.1: exact window coverage is zero, including infinite windows."""
    dist = Exponential(rate=1.2)
    for a, b, window_lower, window_upper in (
        (1.0, 3.0, 1.0, 3.0),
        (1.0, 4.0, 1.0, 3.0),
        (0.0, 3.0, 1.0, 3.0),
        (1.0, jnp.inf, 1.0, jnp.inf),
    ):
        assert (
            dist.log_conditional_interval_prob(a, b, window_lower, window_upper) == 0.0
        )
    # Memorylessness: log P(T>.5 | T>.2)=-1.2*(.5-.2).
    result = dist.log_conditional_interval_prob(0.5, jnp.inf, 0.2, jnp.inf)
    np.testing.assert_allclose(
        result, -0.36, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
