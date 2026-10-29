"""Likelihood values and parameter derivatives at valid boundaries (spec 7/12)."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL, TAIL_ATOL, TAIL_RTOL

import survjax


def test_exponential_exact_at_smallest_normal_time(x64_enabled):
    """Spec 3.3/7.1/7.3: exact positive tiny time has finite value and derivatives."""
    dtype = np.float64 if x64_enabled else np.float32
    time, rate = jnp.asarray(np.finfo(dtype).tiny), jnp.asarray(0.5)
    with mp.workdps(100):
        t, r = mp.mpf(float(time)), mp.mpf(float(rate))
        expected = [float(mp.log(r) - r * t), float(1 / r - t)]

    def value(r):
        return survjax.Exponential(rate=r).log_likelihood(time=time, event=True)

    for differentiate in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            [value(rate), differentiate(value)(rate)],
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.5])
def test_weibull_exact_at_smallest_normal_time(shape, x64_enabled):
    """Spec 3.3/4.1/7.3: log-domain evaluation avoids a subnormal t/b ratio."""
    dtype = np.float64 if x64_enabled else np.float32
    time, parameters = jnp.asarray(np.finfo(dtype).tiny), jnp.array([shape, 2.0])
    with mp.workdps(100):
        k, b = (mp.mpf(float(p)) for p in parameters)
        log_ratio = mp.log(mp.mpf(float(time))) - mp.log(b)
        cumulative_hazard = mp.exp(k * log_ratio)
        # log f=log k-log b+(k-1)*log(t/b)-H; differentiate in k and b.
        expected = [
            float(mp.log(k) - mp.log(b) + (k - 1) * log_ratio - cumulative_hazard),
            float(1 / k + (1 - cumulative_hazard) * log_ratio),
            float(k * (cumulative_hazard - 1) / b),
        ]

    def value(p):
        return survjax.Weibull(shape=p[0], scale=p[1]).log_likelihood(
            time=time, event=True
        )

    for differentiate in (jax.jacfwd, jax.jacrev):
        actual = [value(parameters), *differentiate(value)(parameters)]
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


@pytest.mark.parametrize("family", [survjax.Exponential, survjax.Weibull])
def test_tiny_right_censoring_keeps_nonzero_mass(family, x64_enabled):
    """Spec 7.1/12.1: normal tiny H gives log S=-H, never rounded to zero."""
    dtype = np.float64 if x64_enabled else np.float32
    tiny = np.finfo(dtype).tiny
    # H=t for rate 1; H=sqrt(t) for Weibull k=.5,b=1. Both are normal.
    dist = (
        family(rate=1.0)
        if family is survjax.Exponential
        else family(shape=0.5, scale=1.0)
    )
    expected = -tiny if family is survjax.Exponential else -np.sqrt(tiny)
    actual = dist.log_likelihood(time=jnp.asarray(tiny), event=False)
    assert actual < 0
    np.testing.assert_allclose(
        actual, expected, atol=TAIL_ATOL[x64_enabled], rtol=TAIL_RTOL[x64_enabled]
    )


@pytest.mark.parametrize("shape", [0.5, 1.0, 2.5])
def test_zero_censoring_keeps_zero_derivatives_beside_an_event(shape, x64_enabled):
    """Spec 4.5/7.2: inactive density at zero must not affect either valid row."""
    parameters = jnp.array([shape, 2.0])
    data = survjax.SurvivalData.from_bounds(
        [0.0, 1.0], [np.inf, 1.0], exact=[False, True]
    )

    def rows(p):
        return survjax.Weibull(shape=p[0], scale=p[1]).log_likelihood(data)

    # At t=1,b=2: H=(1/2)**k; c=0 contributes the constant log S(0)=0.
    cumulative_hazard, log_ratio = 0.5**shape, np.log(0.5)
    expected_value = (
        np.log(shape) - np.log(2) + (shape - 1) * log_ratio - cumulative_hazard
    )
    expected_gradient = [
        1 / shape + (1 - cumulative_hazard) * log_ratio,
        shape * (cumulative_hazard - 1) / 2,
    ]
    values = rows(parameters)
    assert values[0] == 0
    np.testing.assert_allclose(
        values[1], expected_value, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    for differentiate in (jax.jacfwd, jax.jacrev):
        jacobian = differentiate(rows)(parameters)
        np.testing.assert_array_equal(jacobian[0], 0)
        np.testing.assert_allclose(
            jacobian[1],
            expected_gradient,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
