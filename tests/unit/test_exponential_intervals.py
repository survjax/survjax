"""Width-based exponential interval probabilities (spec 4.2 and 7)."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
from impl_tolerances import (
    ATOL,
    DERIVATIVE_ATOL,
    DERIVATIVE_RTOL,
    LIKELIHOOD_GRAD_EPS,
    RTOL,
)
from jax.test_util import check_grads

from survjax import Exponential


def test_interval_values_against_high_precision(x64_enabled):
    """Spec 7.2: narrow, tail and underflowing masses stay in log space."""
    tiny = 1e-200 if x64_enabled else 1e-30
    lower = jnp.array([1.0, 1000.0, tiny, 0.0])
    upper = jnp.array(
        [1.0 + (1e-12 if x64_enabled else 1e-5), 1001.0, 2 * tiny, jnp.inf]
    )
    rates = jnp.array([1.2, 1.0, tiny, 2.0])
    # Independent arbitrary-precision exponential interval formula, evaluated
    # at the actual represented inputs. expm1 retains the tiny width product.
    with mp.workdps(80):
        expected = []
        for a, b, r in zip(
            np.asarray(lower), np.asarray(upper), np.asarray(rates), strict=True
        ):
            a, b, r = (mp.mpf(float(x)) for x in (a, b, r))
            expected.append(float(-r * a + mp.log(-mp.expm1(-r * (b - a)))))

    def f(rate, a, b):
        return Exponential(rate=rate).log_interval_prob(a, b)

    for result in (
        f(rates, lower, upper),
        jax.jit(f)(rates, lower, upper),
        jax.vmap(f)(rates, lower, upper),
    ):
        np.testing.assert_allclose(
            result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_narrow_interval_parameter_derivatives(x64_enabled):
    """Spec 7.3: width-based derivatives avoid cancellation through order two."""
    upper = float(jnp.asarray(1.0 + (1e-12 if x64_enabled else 1e-5)))
    rate = jnp.asarray(1.2)

    def f(r):
        return Exponential(rate=r).log_interval_prob(1.0, upper)

    with mp.workdps(80):
        width = mp.mpf(upper) - 1
        r = mp.mpf(float(rate))

        def formula(x):
            return -x + mp.log(-mp.expm1(-x * width))

        gradient = float(mp.diff(formula, r))
        hessian = float(mp.diff(formula, r, 2))
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(first(f))(rate),
            gradient,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        for second in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                jax.jit(second(first(f)))(rate),
                hessian,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )


def test_interval_endpoint_and_parameter_policy():
    """Spec 7.1: equal endpoints have zero mass; invalid inputs give NaN."""
    f = jax.jit(Exponential(rate=1.0).log_interval_prob)
    result = f(
        jnp.array([1.0, jnp.inf, 2.0, -1.0, jnp.nan, 0.0]),
        jnp.array([1.0, jnp.inf, 1.0, 2.0, 2.0, jnp.nan]),
    )
    assert np.isneginf(result[:2]).all()
    assert np.isnan(result[2:]).all()
    assert np.isnan(
        Exponential(rate=jnp.array([0.0, -1.0, jnp.inf, jnp.nan])).log_interval_prob(
            0.0, 1.0
        )
    ).all()


def test_boundary_batch_has_safe_parameter_gradients(x64_enabled):
    """Spec 7.2/7.3: empty and infinite endpoints do not poison active AD."""

    def f(rate):
        values = Exponential(rate=rate).log_interval_prob(
            jnp.array([0.0, 2.0, 1.0, jnp.inf]),
            jnp.array([jnp.inf, jnp.inf, 1.0, jnp.inf]),
        )
        return jnp.where(jnp.array([True, True, False, False]), values, 0.0).sum()

    # Full support contributes 0; right tail contributes -2r; empties ignored.
    with jax.debug_nans(True):
        for transform, expected in (
            (lambda g: g, -2.4),
            (jax.grad, -2.0),
            (jax.hessian, 0.0),
        ):
            np.testing.assert_allclose(
                transform(f)(jnp.asarray(1.2)),
                expected,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )


def test_interval_batch_broadcast(x64_enabled):
    """Spec 4.2: endpoint and parameter batches use trailing-axis broadcasting."""
    result = Exponential(rate=jnp.array([1.0, 2.0])).log_interval_prob(
        jnp.array([[0.0], [1.0]]), jnp.inf
    )
    # Infinite upper endpoints reduce to log S(a)=-r*a.
    np.testing.assert_allclose(
        result,
        [[0.0, 0.0], [-1.0, -2.0]],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


def test_interior_interval_finite_difference_gradients(x64_enabled):
    """Spec 7.3: first and second AD derivatives match finite differences."""

    def f(rate):
        return Exponential(rate=rate).log_interval_prob(0.5, 1.5)

    check_grads(
        jax.jit(f),
        (jnp.asarray(1.2),),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
    )
