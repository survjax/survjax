"""Weibull staged RMST values, boundaries and derivative smoke (spec 8.7)."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import (
    ATOL,
    DERIVATIVE_ATOL,
    DERIVATIVE_RTOL,
    LIKELIHOOD_GRAD_EPS,
    RTOL,
    TAIL_ATOL,
    TAIL_RTOL,
)
from jax.test_util import check_grads

from survjax import Exponential, Weibull


def integral(k, b, tau):
    # Independent quadrature of survival, split across decades and at t=b.
    ratio = tau / b
    points = (
        [0]
        + [mp.mpf(x) for x in (0.01, 0.1, 1, 10, 100, 1000, 10000) if x < ratio]
        + [ratio]
    )
    return b * mp.quad(lambda u: mp.exp(-(u**k)), points)


def test_weibull_rmst_values_against_quadrature(x64_enabled):
    """Spec 8.7/12.1: full value range, scale changes, eager/jit/vmap."""
    shapes = [0.2, 0.5, 1.0, 2.0, 10.0, 50.0]
    ratios = [0.01, 0.1, 0.5, 1.0, 10.0, 100.0, 1e5]
    with mp.workdps(60):
        expected = np.array(
            [[float(integral(mp.mpf(k), 1, mp.mpf(t))) for k in shapes] for t in ratios]
        )
    for scale in (1e-4, 1.0, 1e4):
        dist = Weibull(shape=jnp.array(shapes), scale=scale)
        cutoff = jnp.array(ratios)[:, None] * scale
        for value in (
            dist.rmst(cutoff),
            jax.jit(dist.rmst)(cutoff),
            jax.jit(jax.vmap(dist.rmst))(cutoff),
        ):
            np.testing.assert_allclose(
                value,
                expected * scale,
                atol=TAIL_ATOL[x64_enabled],
                rtol=TAIL_RTOL[x64_enabled],
            )


def test_weibull_rmst_power_tails(x64_enabled):
    """Spec 8.7: x underflow gives tau, x overflow gives b*Gamma(1+1/k)."""
    dist = Weibull(shape=50.0, scale=2.0)
    # x=1e-400 / 1e400 is outside either dtype; RMST stays representable.
    cutoff = jnp.array([2e-8, 2e8])
    with mp.workdps(60):
        expected = [
            float(integral(mp.mpf(50), 2, mp.mpf("0.00000002"))),
            float(2 * mp.gamma(mp.mpf("1.02"))),
        ]
    with jax.debug_nans(True):
        value = jax.jit(dist.rmst)(cutoff)
    np.testing.assert_allclose(
        value, expected, atol=TAIL_ATOL[x64_enabled], rtol=TAIL_RTOL[x64_enabled]
    )


def test_weibull_shape_one_rmst_and_cutoff_grid(x64_enabled):
    """Spec 5/8.7: k=1 is exponential; explicit cutoff axes give (m,S,2)."""
    scale = jnp.array([[0.5, 1.0], [2.0, 4.0], [8.0, 16.0]])
    dist = Weibull(shape=1.0, scale=scale)
    cutoff = jnp.array([0.0, 0.25, 5.0])[:, None, None]
    expected = -np.expm1(-np.asarray(cutoff) / np.asarray(scale)) * np.asarray(scale)
    assert dist.rmst(1.0).shape == (3, 2)
    assert dist.rmst(cutoff).shape == (3, 3, 2)
    for result in (dist.rmst(cutoff), Exponential(rate=1 / scale).rmst(cutoff)):
        np.testing.assert_allclose(
            result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_weibull_rmst_invalid_values_and_shapes(x64_enabled):
    """Spec 8.7: parameter issues override zero; invalid cutoff values are NaN."""
    single = Weibull(shape=jnp.float32(1), scale=jnp.float32(2))
    assert single.rmst(1).dtype == jnp.float32
    if x64_enabled:  # A float64 cutoff is meaningful only in the x64 process.
        wide = Weibull(shape=jnp.float32(0.25), scale=jnp.float32(1e38))
        # The mean is b*Gamma(5)=24*b, outside float32 but inside float64.
        np.testing.assert_allclose(
            wide.rmst(jnp.float64(1e50)), 24 * float(wide.scale), rtol=RTOL[True]
        )
    for field in ("shape", "scale"):
        parameters = {
            "shape": 2.0,
            "scale": 1.0,
            field: jnp.array([1.0, 0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf]),
        }
        dist = Weibull(**parameters)
        for cutoff in (0.0, 1.0, -1.0, jnp.nan, jnp.inf, -jnp.inf):
            result = jax.jit(dist.rmst)(cutoff)
            assert np.isnan(result[1:]).all()
            assert np.isnan(result[0]) == (not np.isfinite(cutoff) or cutoff < 0)
        with pytest.raises(TypeError):
            dist.rmst(jnp.ones(3))

        def fn(p, parameters=parameters, field=field):
            return Weibull(**{**parameters, field: p}).rmst(1.0)[0]

        for transform in (jax.jacfwd, jax.jacrev):
            gradient = transform(fn)(parameters[field])
            assert np.isfinite(gradient).all()
            np.testing.assert_array_equal(gradient[1:], 0)


def test_weibull_rmst_zero_derivatives():
    """Spec 8.7: zero has zero parameter derivatives and cutoff derivative 1."""

    def fn(x):
        return Weibull(shape=x[0], scale=x[1]).rmst(x[2])

    point = jnp.array([1.0, 2.0, 0.0])
    with jax.debug_nans(True):
        assert jax.jit(fn)(point) == 0
        for transform in (jax.jacfwd, jax.jacrev):
            np.testing.assert_array_equal(jax.jit(transform(fn))(point), [0, 0, 1])


def test_weibull_rmst_derivative_smoke(x64_enabled):
    """Spec 8.7: all first directions; supported Hessian; shape Hessian raises."""

    def fn(x):
        return Weibull(shape=x[0], scale=x[1]).rmst(x[2])

    point = jnp.array([2.0, 1.25, 0.75])
    with mp.workdps(60):
        expected = [
            float(mp.diff(integral, (2, 1.25, 0.75), order))
            for order in ((1, 0, 0), (0, 1, 0), (0, 0, 1))
        ]
        bb, bt, tt = [
            float(mp.diff(integral, (2, 1.25, 0.75), order))
            for order in ((0, 2, 0), (0, 1, 1), (0, 0, 2))
        ]
        expected_hessian = [[bb, bt], [bt, tt]]
    for transform in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(transform(fn))(point),
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )

    def supported(x):
        return Weibull(shape=2.0, scale=x[0]).rmst(x[1])

    np.testing.assert_allclose(
        jax.jit(jax.hessian(supported))(point[1:]),
        expected_hessian,
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    check_grads(
        supported,
        (point[1:],),
        order=2,
        modes=("fwd", "rev"),
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )
    with pytest.raises(NotImplementedError):
        jax.hessian(lambda k: Weibull(shape=k, scale=1.25).rmst(0.75))(2.0)


def test_weibull_rmst_invalid_cutoff_tangent_isolation(x64_enabled):
    """Spec 7.2/8.7: masked invalid cutoffs contribute zero to tangents."""
    cutoff = jnp.array([0.0, 1.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])

    def fn(t):
        values = Weibull(shape=1.0, scale=2.0).rmst(t)
        return jnp.where(jnp.isfinite(t) & (t >= 0), values, 0.0).sum()

    # At k=1, d/dtau=exp(-tau/2), including derivative 1 at the origin.
    for transform in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(transform(fn))(cutoff),
            [1, np.exp(-0.5), 0, 0, 0, 0],
            rtol=RTOL[x64_enabled],
        )
