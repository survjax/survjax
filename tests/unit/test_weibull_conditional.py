"""Conditional Weibull probabilities from independent survival mathematics."""

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

from survjax import Exponential, Weibull


def test_far_tail_conditional_derivatives(x64_enabled):
    """Spec 4.2/7.3: cancel the common tail hazard before all AD modes."""
    entry = float(2**40 if x64_enabled else 2**20)
    parameters = jnp.array([1.5, 2.0])

    def f(p):
        return Weibull(shape=p[0], scale=p[1]).log_conditional_interval_prob(
            entry + 0.25, entry + 0.75, entry, entry + 1.0
        )

    with mp.workdps(100):
        k, s = (mp.mpf(float(v)) for v in parameters)

        def formula(k, s):
            h_l, h_a, h_b, h_u = (
                (mp.mpf(t) / s) ** k
                for t in (entry, entry + 0.25, entry + 0.75, entry + 1.0)
            )
            return (
                -(h_a - h_l)
                + mp.log(-mp.expm1(-(h_b - h_a)))
                - mp.log(-mp.expm1(-(h_u - h_l)))
            )

        expected = float(formula(k, s))
        gradient = [float(mp.diff(formula, (k, s), v)) for v in ((1, 0), (0, 1))]
        hessian = [
            [float(mp.diff(formula, (k, s), v)) for v in row]
            for row in (((2, 0), (1, 1)), ((1, 1), (0, 2)))
        ]
    for result in (
        f(parameters),
        jax.jit(f)(parameters),
        jax.vmap(f)(jnp.stack([parameters, parameters])),
    ):
        np.testing.assert_allclose(
            result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(first(f))(parameters),
            gradient,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        for second in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                jax.jit(second(first(f)))(parameters),
                hessian,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )


def test_tiny_window_and_zero_entry(x64_enabled):
    """Spec 7.1/7.2: underflowing positive truncation mass remains usable."""
    tiny = float(jnp.asarray(1e-200 if x64_enabled else 1e-30))
    # k=2, scale=1, window (0,2t], numerator (0,t]; tends to one quarter.
    with mp.workdps(100):
        h = mp.mpf(tiny) ** 2
        expected = float(mp.log(mp.expm1(-h) / mp.expm1(-4 * h)))
    result = Weibull(shape=2.0, scale=1.0).log_conditional_interval_prob(
        0.0, tiny, 0.0, 2 * tiny
    )
    np.testing.assert_allclose(
        result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


def test_shape_one_conditional_identity_and_boundaries(x64_enabled):
    """Spec 4.1/7.1: exponential identity includes finite and infinite windows."""
    lower = [[0.0], [1.0], [2.0], [3.0], [jnp.inf]]
    upper = [[4.0], [3.0], [jnp.inf], [4.0], [jnp.inf]]
    scales = jnp.array([1.0, 2.0])
    for window_upper in (3.0, jnp.inf):
        expected = Exponential(rate=1 / scales).log_conditional_interval_prob(
            lower, upper, 1.0, window_upper
        )
        result = Weibull(shape=1.0, scale=scales).log_conditional_interval_prob(
            lower, upper, 1.0, window_upper
        )
        np.testing.assert_allclose(
            result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
    assert (
        Weibull(shape=1.3, scale=2.1).log_conditional_interval_prob(1.0, 3.0, 1.0, 3.0)
        == 0.0
    )


def test_conditional_invalid_inputs():
    """Spec 7.1: invalid windows, endpoints and parameters are never repaired."""
    f = jax.jit(Weibull(shape=1.3, scale=2.1).log_conditional_interval_prob)
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
    for k, s in (
        (0.0, 1.0),
        (-1.0, 1.0),
        (jnp.inf, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, jnp.nan),
    ):
        assert np.isnan(
            Weibull(shape=k, scale=s).log_conditional_interval_prob(0.0, 4.0, 1.0, 3.0)
        )


def test_mixed_finite_window_safe_gradients(x64_enabled):
    """Spec 7.2/7.3: finite-window full/empty rows do not poison derivatives."""

    def f(p):
        values = Weibull(shape=p[0], scale=p[1]).log_conditional_interval_prob(
            jnp.array([0.0, 2.0, 4.0, jnp.inf]),
            jnp.array([5.0, 3.0, 5.0, jnp.inf]),
            1.0,
            4.0,
        )
        return jnp.where(jnp.array([True, True, False, False]), values, 0.0).sum()

    # Only (2,3] over (1,4] contributes; compute reference from survival powers.
    def analytical(p):
        k, s = p
        return jnp.log(
            (jnp.exp(-((2 / s) ** k)) - jnp.exp(-((3 / s) ** k)))
            / (jnp.exp(-((1 / s) ** k)) - jnp.exp(-((4 / s) ** k)))
        )

    p = jnp.array([1.3, 2.1])
    with jax.debug_nans(True):
        for transform in (lambda g: g, jax.grad, jax.hessian):
            np.testing.assert_allclose(
                transform(f)(p),
                transform(analytical)(p),
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
    check_grads(
        jax.jit(f),
        (p,),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
    )
