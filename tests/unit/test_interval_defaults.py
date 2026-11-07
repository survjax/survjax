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
