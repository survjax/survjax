"""Weibull interval values and derivatives from independent mathematics."""

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


def test_weibull_interval_high_precision_values(x64_enabled):
    """Spec 7.2: narrow powers, tiny mass, separated endpoints and tails."""
    tiny = 1e-200 if x64_enabled else 1e-30
    narrow = 1e-12 if x64_enabled else 1e-5
    tiny_shape = 1e-300 if x64_enabled else 1e-35
    a = jnp.array([1.0, tiny, tiny, 10.0, 0.5, 3.0, 1.0])
    b = jnp.array(
        [1.0 + narrow, 2 * tiny, 1 / tiny, 10.1, 3.0, 3.0 + narrow, 1.0 + narrow]
    )
    shape = jnp.array([1.3, 2.0, 0.01, 3.0, 0.5, 1.3, tiny_shape])
    scale = jnp.array([2.1, 1.0, 1.0, 1.0, 2.0, 2.1, 1.0])
    # 400 digits resolve the last row's O(1e-312) difference of powers.
    with mp.workdps(400):
        expected = []
        for row in zip(
            np.asarray(a),
            np.asarray(b),
            np.asarray(shape),
            np.asarray(scale),
            strict=True,
        ):
            lo, hi, k, s = (mp.mpf(float(v)) for v in row)
            h_lo, h_hi = (lo / s) ** k, (hi / s) ** k
            expected.append(float(-h_lo + mp.log(-mp.expm1(-(h_hi - h_lo)))))

    def f(k, s, lo, hi):
        return Weibull(shape=k, scale=s).log_interval_prob(lo, hi)

    for result in (
        f(shape, scale, a, b),
        jax.jit(f)(shape, scale, a, b),
        jax.vmap(f)(shape, scale, a, b),
    ):
        np.testing.assert_allclose(
            result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_weibull_narrow_interval_gradients_and_hessians(x64_enabled):
    """Spec 7.3: natural-parameter derivatives through order two, all AD modes."""
    upper = float(jnp.asarray(1.0 + (1e-12 if x64_enabled else 1e-5)))
    parameters = jnp.array([1.3, 2.1])

    def f(p):
        return Weibull(shape=p[0], scale=p[1]).log_interval_prob(1.0, upper)

    with mp.workdps(100):
        k, s = (mp.mpf(float(v)) for v in parameters)

        def formula(k, s):
            return -((1 / s) ** k) + mp.log(
                -mp.expm1(-((mp.mpf(upper) / s) ** k - (1 / s) ** k))
            )

        gradient = np.array(
            [float(mp.diff(formula, (k, s), orders)) for orders in ((1, 0), (0, 1))]
        )
        hessian = np.array(
            [
                [float(mp.diff(formula, (k, s), orders)) for orders in row]
                for row in (((2, 0), (1, 1)), ((1, 1), (0, 2)))
            ]
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
    check_grads(
        jax.jit(f),
        (parameters,),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
    )


def test_shape_one_interval_identity_and_broadcast(x64_enabled):
    """Spec 4.1/4.2: k=1 is exponential; endpoint/batch axes broadcast."""
    lower = jnp.array([[0.0], [0.5], [1.0]])
    upper = jnp.array([[1.0], [1.5], [jnp.inf]])
    scales = jnp.array([1.0, 2.0])
    # Analytical identity S(t)=exp(-t/scale) at k=1, independently implemented.
    expected = Exponential(rate=1 / scales).log_interval_prob(lower, upper)
    result = Weibull(shape=1.0, scale=scales).log_interval_prob(
        lower.tolist(), upper.tolist()
    )
    np.testing.assert_allclose(
        result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


def test_weibull_interval_endpoint_policy():
    """Spec 7.1: empty intervals differ from invalid endpoints and parameters."""
    result = jax.jit(Weibull(shape=2.0, scale=1.0).log_interval_prob)(
        jnp.array([0.0, 1.0, jnp.inf, -1.0, 2.0, jnp.nan, 0.0]),
        jnp.array([0.0, 1.0, jnp.inf, 2.0, 1.0, 2.0, jnp.nan]),
    )
    assert np.isneginf(result[:3]).all()
    assert np.isnan(result[3:]).all()
    for shape, scale in (
        (0.0, 1.0),
        (-1.0, 1.0),
        (jnp.inf, 1.0),
        (1.0, 0.0),
        (1.0, -1.0),
        (1.0, jnp.nan),
    ):
        assert np.isnan(Weibull(shape=shape, scale=scale).log_interval_prob(0.0, 1.0))


def test_weibull_boundary_batch_safe_gradients(x64_enabled):
    """Spec 7.2: full, right and empty intervals have safe shape/scale tangents."""

    def f(p):
        values = Weibull(shape=p[0], scale=p[1]).log_interval_prob(
            jnp.array([0.0, 2.0, 1.0, jnp.inf]),
            jnp.array([jnp.inf, jnp.inf, 1.0, jnp.inf]),
        )
        return jnp.where(jnp.array([True, True, False, False]), values, 0.0).sum()

    # Only right tail contributes: -(2/scale)**shape, at shape=1,scale=2.
    # Gradient=(0,1/2); Hessian=((0,1/2),(1/2,-1/2)).
    with jax.debug_nans(True):
        p = jnp.array([1.0, 2.0])
        for result, expected in (
            (f(p), -1.0),
            (jax.grad(f)(p), [0.0, 0.5]),
            (jax.hessian(f)(p), [[0.0, 0.5], [0.5, -0.5]]),
        ):
            np.testing.assert_allclose(
                result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
            )
