"""Feature `summary.mean`: mean derivatives (spec 4.4, 8.4, 8.7, 8.8, 12.1; rev 51).

8.4: "Summaries are differentiable; nothing raises"; generic mean tangent
"`∫ ∂θ S·dθ dt` ... on a **static** node set ... Forward, reverse and
higher-order derivatives all work"; "Entries that failed or did not converge
get NaN tangents, never zeros. ... known-infinite means get NaN or infinite
tangents." 4.4: PH closures differentiate "by the chain rule". 8.7 isolation
of valid entries remains as value policy (8.8).

Tolerances (12.1 log units): ``|theta (got - want)| <= scale * |value|``,
higher orders multiplied by every variable once per order; ``scale`` is the
12.1 ``rtol`` for built-in closed forms and ``generic_rtol()`` for generic
quadrature (Q-058). References: mpmath ``diff`` of `b*Γ(1+1/k)` at 60 digits.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
import summary_reference as sref
from helpers import (
    X64,
    assert_close,
    assert_generic_log_derivative,
    assert_log_parameter_derivative,
    f,
    generic_rtol,
    ph,
    rmst_rtol,
    rounded,
)
from jax.test_util import check_grads
from summary_cases import HESSIANS, MEAN, MODES, assert_hessian, grad, valid_subset_grad

import survjax

requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")


def _power(k, b):
    return fam.power_hazard()(shape=k, scale=b)


# =============================================================== built-in closed forms


@pytest.mark.parametrize("mode", MODES)
@MEAN
def test_exponential_mean_derivative(mode):
    """spec 8.4, rev 51: `d(1/r)/dr = -1/r**2`."""
    (rr,) = rounded(0.7)
    g = grad(lambda r: survjax.Exponential(rate=r).mean(), 0, mode)(f(rr))
    assert_log_parameter_derivative(g, -1 / rr**2, rr, 1 / rr, "dm/dr")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [0.2, 0.5, 2.5, 50.0])
@MEAN
def test_weibull_mean_derivatives(k, mode):
    """spec 8.4, rev 51: `d/dk` and `d/db` of `b*Γ(1+1/k)` against mpmath."""
    kk, bb = rounded(k, 2.0)
    m = sref.weibull_mean(kk, bb)
    gk, gb = grad(lambda k_, b_: survjax.Weibull(shape=k_, scale=b_).mean(), (0, 1), mode)(f(kk), f(bb))
    assert_log_parameter_derivative(gk, sref.weibull_mean_derivative(kk, bb, (1, 0)), kk, m, "dm/dk")
    assert_log_parameter_derivative(gb, sref.weibull_mean_derivative(kk, bb, (0, 1)), bb, m, "dm/db")


@pytest.mark.parametrize("mode", list(HESSIANS))
@MEAN
def test_weibull_mean_hessian(mode):
    """spec 8.4, rev 51: "higher-order derivatives": the `(k, b)` mean Hessian, all four compositions, both dtypes."""
    kk, bb = rounded(2.5, 2.0)
    got = HESSIANS[mode](lambda x: survjax.Weibull(shape=x[0], scale=x[1]).mean())(f([kk, bb]))
    assert_hessian(
        got, (kk, bb), lambda o: sref.weibull_mean_derivative(kk, bb, o), sref.weibull_mean(kk, bb), rmst_rtol(), "mean"
    )


@MEAN
def test_ph_weibull_mean_gradient_equals_the_equivalent_weibull():
    """spec 4.4, rev 51: the PH-closure mean differentiates by the chain rule through `Weibull(k, b0*exp(-eta/k))`."""

    def via_ph(x):
        return ph(survjax.Weibull(shape=x[0], scale=x[1]), x[2]).mean()

    def via_equiv(x):
        return survjax.Weibull(shape=x[0], scale=x[1] * jnp.exp(-x[2] / x[0])).mean()

    x = f([1.5, 2.0, 0.4])
    assert_close(jax.grad(via_ph)(x), jax.grad(via_equiv)(x))
    assert_close(jax.jacfwd(via_ph)(x), jax.jacfwd(via_equiv)(x))


# =============================================================== generic paths


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", [0.5, 2.5, 10.0])
@MEAN
def test_generic_mean_first_derivatives(k, mode):
    """spec 8.4, rev 51: generic mean tangents against mpmath `diff` of `b*Γ(1+1/k)`."""
    kk, bb = rounded(k, 2.0)
    m = sref.weibull_mean(kk, bb)
    gk, gb = grad(lambda k_, b_: _power(k_, b_).mean(), (0, 1), mode)(f(kk), f(bb))
    assert_generic_log_derivative(gk, sref.weibull_mean_derivative(kk, bb, (1, 0)), kk, m, "dm/dk")
    assert_generic_log_derivative(gb, sref.weibull_mean_derivative(kk, bb, (0, 1)), bb, m, "dm/db")


@pytest.mark.parametrize("mode", list(HESSIANS))
@MEAN
def test_generic_mean_hessian(mode):
    """spec 8.4, rev 51: generic `(k, b)` mean Hessian, both dtypes."""
    kk, bb = rounded(2.5, 2.0)
    got = HESSIANS[mode](lambda x: _power(x[0], x[1]).mean())(f([kk, bb]))
    assert_hessian(
        got, (kk, bb), lambda o: sref.weibull_mean_derivative(kk, bb, o), sref.weibull_mean(kk, bb), generic_rtol(), "mean"
    )


@requires_x64
@pytest.mark.parametrize(
    "fn",
    [
        lambda r, t: fam.const_hazard()(rate=r).mean() * t,
        lambda k, b: _power(k, b).mean(),
        lambda k, b: survjax.Weibull(shape=k, scale=b).mean(),
    ],
    ids=["custom-mean", "power-mean", "weibull-mean"],
)
@MEAN
def test_check_grads_order_two_mean(fn):
    """spec 8.8/12.1, rev 51: `check_grads(order=2, modes=("fwd", "rev"))` on interior generic and built-in means."""
    check_grads(fn, (f([0.7, 1.3]), f([1.4, 1.6])), order=2, modes=("fwd", "rev"))


@MEAN
def test_generic_mean_gradient_agrees_under_jit_and_vmap():
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of generic mean gradients."""

    def one(k):
        return _power(k, f(2.0)).mean()

    ks = f([0.7, 1.5, 2.5])
    want = np.array([np.asarray(jax.grad(one)(k)) for k in ks])
    assert_close(jax.vmap(jax.grad(one))(ks), want)
    assert_close(jax.jit(jax.vmap(jax.grad(one)))(ks), want)


# =============================================================== isolation and non-finite tangents


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("generic", [False, True], ids=["builtin", "generic"])
@MEAN
def test_mean_isolates_failing_entries_with_shared_scale(generic, mode):
    """spec 8.7/8.8, rev 51: a failing entry (`shape=-1`: a declared violation for Weibull; for the undeclared custom family `S -> 1`, so its mean fails) leaves the shared-scale mean gradient of the others unchanged."""
    make = _power if generic else (lambda k, b: survjax.Weibull(shape=k, scale=b))
    g_full, g_alone = valid_subset_grad(
        lambda b: make(f([1.5, -1.0, 2.5]), b).mean(),
        lambda b: make(f([1.5, 2.5]), b).mean(),
        f(2.0),
        mode,
    )
    assert np.isfinite(np.asarray(g_full)).all()
    assert_close(g_full, g_alone)


@MEAN
def test_improper_mean_has_a_non_finite_tangent():
    """spec 8.4, rev 51: an improper family's mean is known infinite ("NaN or infinite tangents") or unresolved ("NaN tangents"); either way never finite."""
    _, tangent = jax.jvp(lambda r: fam.mixture_cure()(cure=f(0.3), rate=r).mean(), (f(1.0),), (f(1.0),))
    assert not np.isfinite(np.asarray(tangent))
