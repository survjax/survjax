"""Staged RMST surface and independent exponential evidence (spec 8.7)."""

from dataclasses import dataclass
from functools import partial

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

import survjax


@survjax.register_family
@dataclass(frozen=True)
class Custom(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param()

    def log_survival(self, t):
        return -self.rate * t

    def reference_time(self):
        return jnp.ones_like(self.rate)


def test_staged_summary_surface():
    """Spec 8.7: unsupported methods fail by type at call and trace time."""
    custom = Custom(rate=2.0)
    for call in (custom.rmst, jax.jit(custom.rmst)):
        with pytest.raises(NotImplementedError):
            call(0.0)
    for dist in (
        custom,
        survjax.Exponential(rate=2.0),
        survjax.Weibull(shape=1, scale=2),
    ):
        for name, args in (("mean", ()), ("median", ()), ("quantile", (0.5,))):
            for call in (getattr(dist, name), jax.jit(getattr(dist, name))):
                with pytest.raises(NotImplementedError):
                    call(*args)
        for name in "sample inverse_cumulative_hazard rmst_grid rmst_with_info".split():
            assert not hasattr(dist, name)
    with pytest.raises(TypeError):
        survjax.Exponential(rate=1).rmst(1, options=None)


@pytest.mark.parametrize("bad", [True, 1j, "1", None, np.array([1], dtype=object)])
def test_rmst_rejects_nonreal_cutoffs(bad):
    """Spec 4.5/8.7: static dtype misuse raises even under tracing."""

    def call():
        return survjax.Exponential(rate=1).rmst(bad)

    with pytest.raises(TypeError):
        call()
    with pytest.raises(TypeError):
        jax.jit(call)()


def test_rmst_rejects_incompatible_shapes():
    """Spec 5/8.7: cutoff axes follow trailing broadcasting only."""
    for call in (
        survjax.Exponential(rate=[1, 2]).rmst,
        jax.jit(survjax.Exponential(rate=[1, 2]).rmst),
    ):
        with pytest.raises(TypeError):
            call(jnp.ones(3))


def test_rmst_mpmath_values_and_scale_changes(x64_enabled):
    """Spec 8.7/12.1: independent integral over r*tau in [0.01, 1e5]."""
    ratios = jnp.array([0.01, 0.1, 1.0, 10.0, 1e5])
    rates = jnp.array([[0.5, 1.0], [2.0, 4.0], [8.0, 16.0]])
    compare = partial(
        np.testing.assert_allclose,
        atol=TAIL_ATOL[x64_enabled],
        rtol=TAIL_RTOL[x64_enabled],
    )
    for scale in (1e-4, 1.0, 1e4):
        rate = rates / scale
        tau = ratios[:, None, None] / rate
        with mp.workdps(60):
            # Substitute u=r*t: integral_0^tau exp(-r*t)dt = integral_0^(r*tau) exp(-u)du/r.
            integrals = np.array(
                [float(mp.quad(lambda u: mp.exp(-u), [0, float(x)])) for x in ratios]
            )
        expected = integrals[:, None, None] / np.asarray(rate)

        def fn(t, rate=rate):
            return survjax.Exponential(rate=rate).rmst(t)

        for result in (fn(tau), jax.jit(fn)(tau), jax.jit(jax.vmap(fn))(tau)):
            assert result.shape == (5, 3, 2)
            compare(result, expected)
        # Explicit common cutoff grid gives (m,S,arms), also when m equals S.
        common = ratios[:3, None, None]
        expected = -np.expm1(-np.asarray(rate) * np.asarray(common)) / np.asarray(rate)
        np.testing.assert_allclose(
            fn(common), expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_rmst_zero_value_and_derivatives():
    """Spec 8.7: RMST(0)=0, d/dr=0, d/dtau=S(0)=1."""

    def fn(r, t):
        return survjax.Exponential(rate=r).rmst(t)

    with jax.debug_nans(True):
        assert jax.jit(fn)(2.0, 0.0) == 0
        for derivative in (jax.jacfwd, jax.jacrev):
            assert jax.jit(derivative(fn, 0))(2.0, 0.0) == 0
            assert jax.jit(derivative(fn, 1))(2.0, 0.0) == 1
            assert derivative(derivative(fn, 0), 0)(2.0, 0.0) == 0


def test_rmst_invalid_entries_are_nan():
    """Spec 8.7: invalid parameters override the zero-cutoff boundary too."""
    rate = jnp.array([2.0, 0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])

    def fn(t):
        return survjax.Exponential(rate=rate).rmst(t)

    for tau in (0.0, 1.0, -1.0, jnp.nan, jnp.inf, -jnp.inf):
        for out in (fn(tau), jax.jit(fn)(tau), jax.vmap(fn)(jnp.array([tau]))[0]):
            assert np.isnan(out[1:]).all()
            assert np.isnan(out[0]) == (not np.isfinite(tau) or tau < 0)


@pytest.mark.parametrize("shared", [True, False])
def test_rmst_invalid_entries_preserve_valid_derivatives(shared, x64_enabled):
    """Spec 7.2/8.7: invalid cutoffs/parameters cannot poison other entries."""
    rate = (
        jnp.array(2.0)
        if shared
        else jnp.array([2.0, 0.0, -1.0, jnp.nan, jnp.inf, -jnp.inf])
    )
    cutoff = (
        jnp.array([1.0, -1.0, jnp.nan, jnp.inf, -jnp.inf, 0.0])
        if shared
        else jnp.ones(6)
    )

    def fn(r):
        return survjax.Exponential(rate=r).rmst(cutoff)[0]

    # Differentiating integral exp(-r*t)dt gives ((r+1)exp(-r)-1)/r**2 at tau=1.
    compare = partial(np.testing.assert_allclose, rtol=RTOL[x64_enabled])
    first = (3 * np.exp(-2) - 1) / 4
    second = (2 - 10 * np.exp(-2)) / 8
    for derivative in (jax.jacfwd, jax.jacrev):
        expected = first if shared else [first, 0, 0, 0, 0, 0]
        compare(jax.jit(derivative(fn))(rate), expected)
        hessian = derivative(derivative(fn))(rate)
        target = second if shared else np.diag([second, 0, 0, 0, 0, 0])
        compare(hessian, target)


def test_rmst_derivatives_against_independent_integral(x64_enabled):
    """Spec 8.7/12.1: both modes, full Hessian, third/fourth derivatives."""

    def fn(x):
        return survjax.Exponential(rate=x[0]).rmst(x[1])

    point = jnp.array([1.25, 0.75])
    compare = partial(np.testing.assert_allclose, rtol=RTOL[x64_enabled])
    with mp.workdps(60):

        def integral(r, t):
            return mp.quad(lambda u: mp.exp(-r * u), [0, t])

        expected_gradient = [
            float(mp.diff(integral, (1.25, 0.75), order)) for order in ((1, 0), (0, 1))
        ]
        expected_hessian = [
            [
                float(mp.diff(integral, (1.25, 0.75), (2 - i - j, i + j)))
                for j in range(2)
            ]
            for i in range(2)
        ]
        for transform in (jax.jacfwd, jax.jacrev):
            compare(jax.jit(transform(fn))(point), expected_gradient)
            compare(jax.jit(transform(transform(fn)))(point), expected_hessian)

            def derivative(r):
                return survjax.Exponential(rate=r).rmst(0.75)

            for order in range(1, 5):
                derivative = transform(derivative)
                expected = float(mp.diff(lambda r: integral(r, 0.75), 1.25, order))
                compare(jax.jit(derivative)(1.25), expected)
    compare(jax.grad(fn)(point)[1], np.exp(-1.25 * 0.75))
    check_grads(
        fn,
        (point,),
        order=2,
        modes=("fwd", "rev"),
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )
