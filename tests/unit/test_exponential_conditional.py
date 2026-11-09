"""Exponential conditional intervals, spec 4.2 and 7.1–7.3."""

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


def test_far_tail_conditional_value_and_derivatives(x64_enabled):
    """Spec 4.2/7.3: cancel common tail hazard before values and derivatives."""
    entry = float(2**40 if x64_enabled else 2**20)
    rate = jnp.asarray(1.2)

    def f(r):
        return Exponential(rate=r).log_conditional_interval_prob(
            entry + 0.25, entry + 0.75, entry, entry + 1.0
        )

    # Memorylessness removes entry. Independent high-precision survival ratio.
    with mp.workdps(80):

        def formula(r):
            return mp.log((mp.exp(-r / 4) - mp.exp(-3 * r / 4)) / (1 - mp.exp(-r)))

        r = mp.mpf(float(rate))
        expected = [float(mp.diff(formula, r, n)) for n in range(3)]
    for result in (f(rate), jax.jit(f)(rate), jax.vmap(f)(jnp.stack([rate, rate]))):
        np.testing.assert_allclose(
            result, expected[0], atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(first(f))(rate),
            expected[1],
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        for second in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                jax.jit(second(first(f)))(rate),
                expected[2],
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
    check_grads(
        jax.jit(f),
        (rate,),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
    )


def test_tiny_conditional_window_retains_log_mass(x64_enabled):
    """Spec 7.1/7.2: a tiny positive denominator is not treated as zero."""
    tiny = float(jnp.asarray(1e-200 if x64_enabled else 1e-30))
    # Rate and window width product underflows; the conditional ratio tends
    # to one half. mpmath evaluates expm1 without forming linear underflow.
    with mp.workdps(80):
        x = mp.mpf(tiny) ** 2
        expected = float(mp.log(mp.expm1(-x) / mp.expm1(-2 * x)))
    result = Exponential(rate=tiny).log_conditional_interval_prob(
        0.0, tiny, 0.0, 2 * tiny
    )
    np.testing.assert_allclose(
        result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


def test_conditional_intersection_boundaries_and_broadcast(x64_enabled):
    """Spec 7.1: full coverage, partial overlap and empty intersections."""
    rates = jnp.array([1.0, 2.0])
    lower = jnp.array([[0.0], [1.0], [2.0], [3.0], [jnp.inf]])
    upper = jnp.array([[4.0], [3.0], [jnp.inf], [4.0], [jnp.inf]])
    result = jax.jit(Exponential(rate=rates).log_conditional_interval_prob)(
        lower, upper, 1.0, 3.0
    )
    # Partial interval (2,3] over window (1,3], factored by exp(-r).
    partial = (
        -np.asarray(rates)
        + np.log(-np.expm1(-np.asarray(rates)))
        - np.log(-np.expm1(-2 * np.asarray(rates)))
    )
    expected = np.array(
        [[0.0, 0.0], [0.0, 0.0], partial, [-np.inf, -np.inf], [-np.inf, -np.inf]]
    )
    np.testing.assert_allclose(
        result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    assert np.equal(result[:2], 0.0).all()


def test_conditional_invalid_inputs_are_nan():
    """Spec 7.1: no invalid window or rate is repaired into a probability."""
    f = jax.jit(Exponential(rate=1.0).log_conditional_interval_prob)
    for args in (
        (0.0, 1.0, 1.0, 1.0),
        (0.0, 1.0, 2.0, 1.0),
        (0.0, 1.0, -1.0, 2.0),
        (0.0, 1.0, jnp.inf, jnp.inf),
        (0.0, 1.0, 0.0, jnp.nan),
        (jnp.nan, 1.0, 0.0, 2.0),
        (0.0, jnp.nan, 0.0, 2.0),
        (-1.0, 1.0, 0.0, 2.0),
        (2.0, 1.0, 0.0, 3.0),
    ):
        assert np.isnan(f(*args))
    assert np.isnan(
        Exponential(
            rate=jnp.array([0.0, -1.0, jnp.inf, jnp.nan])
        ).log_conditional_interval_prob(0.0, 3.0, 1.0, 2.0)
    ).all()


def test_conditional_infinite_window_safe_gradients(x64_enabled):
    """Spec 7.2/7.3: boundary rows do not poison a smooth right-tail objective."""

    def f(rate):
        values = Exponential(rate=rate).log_conditional_interval_prob(
            jnp.array([0.0, 2.0, 3.0, jnp.inf]),
            jnp.array([jnp.inf, jnp.inf, 3.0, jnp.inf]),
            1.0,
            jnp.inf,
        )
        return jnp.where(jnp.array([True, True, False, False]), values, 0.0).sum()

    # Full window contributes zero; memorylessness gives -r*(2-1)=-r.
    with jax.debug_nans(True):
        for transform, expected in (
            (lambda g: g, -1.2),
            (jax.grad, -1.0),
            (jax.hessian, 0.0),
        ):
            np.testing.assert_allclose(
                transform(f)(jnp.asarray(1.2)),
                expected,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
