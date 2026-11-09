"""Feature `summary.quantile`: quantile, median and inverse cumulative hazard derivatives (spec 4.4, 8.1, 8.4, 8.7, 8.8, 12.1; rev 51).

8.4: "Summaries are differentiable; nothing raises." "**Inverse cumulative
hazard and quantiles** (implicit function theorem): `dt = (∂θ log S(t)·dθ +
de) / h(t)` ... `dq/dp = 1/f(q)`." "Entries that failed or did not converge
get NaN tangents, never zeros. Endpoint quantiles ... get NaN or infinite
tangents." 4.4: PH closures differentiate "by the chain rule". 8.7 isolation
of valid entries remains as value policy (8.8).

Tolerances (12.1 log units): ``|theta (got - want)| <= scale * |value|``
(``p`` and ``e`` included), higher orders multiplied by every variable once
per order; ``scale`` is the 12.1 ``rtol`` for built-in closed forms and
``generic_rtol()`` for the generic solver (Q-058). References: hand
derivatives of the 8.1/8.2 closed forms, and mpmath ``diff`` at 60 digits
(``summary_reference``).
"""

import math

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
from summary_cases import HESSIANS, MODES, QUANTILE, assert_hessian, grad, valid_subset_grad

import survjax

requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")


def _power(k, b):
    return fam.power_hazard()(shape=k, scale=b)


def _weibull_inverse_derivatives(k, b, e):
    """`t = b*e**(1/k)`: `dt/dk = -t log(e)/k**2`, `dt/db = t/b`, `dt/de = t/(k e) = 1/h(t)` (8.4)."""
    t = b * e ** (1 / k)
    return t, (-t * math.log(e) / k**2, t / b, t / (k * e))


# =============================================================== built-in closed forms


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("p", [0.01, 0.5, 0.99])
@QUANTILE
def test_exponential_quantile_derivatives(p, mode):
    """spec 8.4, rev 51: `q = -log1p(-p)/r`: `dq/dr = -q/r`, `dq/dp = 1/f(q) = 1/(r(1-p))`."""
    rr, pp = rounded(0.7, p)
    q = -math.log1p(-pp) / rr
    gr, gp = grad(lambda r, p_: survjax.Exponential(rate=r).quantile(p_), (0, 1), mode)(f(rr), f(pp))
    assert_log_parameter_derivative(gr, -q / rr, rr, q, "dq/dr")
    assert_log_parameter_derivative(gp, 1 / (rr * (1 - pp)), pp, q, "dq/dp")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("p", [0.01, 0.5, 0.99])
@pytest.mark.parametrize("k", [0.5, 2.5, 10.0])
@QUANTILE
def test_weibull_quantile_derivatives(k, p, mode):
    """spec 8.4, rev 51: Weibull quantile derivatives in `k`, `b`, `p` against mpmath `diff` of `b*(-log1p(-p))**(1/k)`."""
    kk, bb, pp = rounded(k, 2.0, p)
    q = sref.weibull_quantile(kk, bb, pp)
    grads = grad(lambda k_, b_, p_: survjax.Weibull(shape=k_, scale=b_).quantile(p_), (0, 1, 2), mode)(
        f(kk), f(bb), f(pp)
    )
    for g, x, orders, name in zip(grads, (kk, bb, pp), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "kbp", strict=True):
        assert_log_parameter_derivative(g, sref.weibull_quantile_derivative(kk, bb, pp, orders), x, q, f"d/d{name}")


@pytest.mark.parametrize("mode", list(HESSIANS))
@QUANTILE
def test_weibull_quantile_hessian(mode):
    """spec 8.4, rev 51: the `(k, b, p)` quantile Hessian, all four compositions, both dtypes."""
    kk, bb, pp = rounded(2.5, 2.0, 0.3)
    got = HESSIANS[mode](lambda x: survjax.Weibull(shape=x[0], scale=x[1]).quantile(x[2]))(f([kk, bb, pp]))
    assert_hessian(
        got,
        (kk, bb, pp),
        lambda o: sref.weibull_quantile_derivative(kk, bb, pp, o),
        sref.weibull_quantile(kk, bb, pp),
        rmst_rtol(),
        "quantile",
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("e", [0.1, 1.0, 5.0])
@QUANTILE
def test_weibull_inverse_cumulative_hazard_derivatives(e, mode):
    """spec 8.4, rev 51: derivatives of `b*e**(1/k)` in `k`, `b` and `e` (`dt/de = 1/h(t)`)."""
    kk, bb, ee = rounded(2.5, 2.0, e)
    t, wants = _weibull_inverse_derivatives(kk, bb, ee)
    grads = grad(lambda k_, b_, e_: survjax.Weibull(shape=k_, scale=b_).inverse_cumulative_hazard(e_), (0, 1, 2), mode)(
        f(kk), f(bb), f(ee)
    )
    for g, want, x, name in zip(grads, wants, (kk, bb, ee), "kbe", strict=True):
        assert_log_parameter_derivative(g, want, x, t, f"d/d{name}")


@pytest.mark.parametrize("mode", MODES)
@QUANTILE
def test_median_derivatives_equal_the_half_quantile_derivatives(mode):
    """spec 8.1/8.4, rev 51: `median() = quantile(0.5)`, so their parameter derivatives agree."""
    x = f([2.5, 2.0])
    via_median = grad(lambda y: survjax.Weibull(shape=y[0], scale=y[1]).median(), 0, mode)(x)
    via_quantile = grad(lambda y: survjax.Weibull(shape=y[0], scale=y[1]).quantile(f(0.5)), 0, mode)(x)
    assert_close(via_median, via_quantile)


@QUANTILE
def test_ph_weibull_quantile_gradient_equals_the_equivalent_weibull():
    """spec 4.4, rev 51: PH-closure quantile and inverse cumulative hazard differentiate by the chain rule through `Weibull(k, b0*exp(-eta/k))`."""

    def via_ph(x):
        d = ph(survjax.Weibull(shape=x[0], scale=x[1]), x[2])
        return jnp.stack([d.quantile(f(0.3)), d.inverse_cumulative_hazard(f(0.7))])

    def via_equiv(x):
        d = survjax.Weibull(shape=x[0], scale=x[1] * jnp.exp(-x[2] / x[0]))
        return jnp.stack([d.quantile(f(0.3)), d.inverse_cumulative_hazard(f(0.7))])

    x = f([1.5, 2.0, 0.4])
    assert_close(jax.jacrev(via_ph)(x), jax.jacrev(via_equiv)(x))
    assert_close(jax.jacfwd(via_ph)(x), jax.jacfwd(via_equiv)(x))


# =============================================================== generic paths


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("p", [0.01, 0.5, 0.99])
@pytest.mark.parametrize("k", [0.5, 2.5])
@QUANTILE
def test_generic_quantile_first_derivatives(k, p, mode):
    """spec 8.4, rev 51: implicit-function quantile tangents `dt = (∂θ log S·dθ + de)/h(t)`, `dq/dp = 1/f(q)`, against mpmath."""
    kk, bb, pp = rounded(k, 2.0, p)
    q = sref.weibull_quantile(kk, bb, pp)
    grads = grad(lambda k_, b_, p_: _power(k_, b_).quantile(p_), (0, 1, 2), mode)(f(kk), f(bb), f(pp))
    for g, x, orders, name in zip(grads, (kk, bb, pp), ((1, 0, 0), (0, 1, 0), (0, 0, 1)), "kbp", strict=True):
        assert_generic_log_derivative(g, sref.weibull_quantile_derivative(kk, bb, pp, orders), x, q, f"d/d{name}")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("e", [0.1, 1.0, 5.0])
@QUANTILE
def test_generic_inverse_cumulative_hazard_derivatives(e, mode):
    """spec 8.4, rev 51: implicit-function tangents of the generic `H⁻¹(e)` for `H = (t/b)**k`, against the hand derivatives of `b*e**(1/k)`."""
    kk, bb, ee = rounded(2.5, 2.0, e)
    t, wants = _weibull_inverse_derivatives(kk, bb, ee)
    grads = grad(lambda k_, b_, e_: _power(k_, b_).inverse_cumulative_hazard(e_), (0, 1, 2), mode)(f(kk), f(bb), f(ee))
    for g, want, x, name in zip(grads, wants, (kk, bb, ee), "kbe", strict=True):
        assert_generic_log_derivative(g, want, x, t, f"d/d{name}")


@pytest.mark.parametrize("mode", list(HESSIANS))
@QUANTILE
def test_generic_quantile_hessian(mode):
    """spec 8.4, rev 51: generic `(k, b, p)` quantile Hessian, both dtypes."""
    kk, bb, pp = rounded(2.5, 2.0, 0.3)
    got = HESSIANS[mode](lambda x: _power(x[0], x[1]).quantile(x[2]))(f([kk, bb, pp]))
    assert_hessian(
        got,
        (kk, bb, pp),
        lambda o: sref.weibull_quantile_derivative(kk, bb, pp, o),
        sref.weibull_quantile(kk, bb, pp),
        generic_rtol(),
        "quantile",
    )


@requires_x64
@pytest.mark.parametrize(
    "fn",
    [
        lambda r, p: fam.const_hazard()(rate=r).quantile(p),
        lambda k, p: _power(k, f(2.0)).quantile(p),
        lambda k, e: _power(k, f(2.0)).inverse_cumulative_hazard(e),
        lambda r, p: survjax.Exponential(rate=r).quantile(p),
        lambda k, p: survjax.Weibull(shape=k, scale=f(2.0)).quantile(p),
        lambda k, e: survjax.Weibull(shape=k, scale=f(2.0)).inverse_cumulative_hazard(e),
        lambda e, p: ph(survjax.Weibull(shape=f(1.5), scale=f(2.0)), e).quantile(p),
        lambda e, p: ph(_power(f(1.5), f(2.0)), e).quantile(p),
    ],
    ids=[
        "custom-quantile",
        "power-quantile",
        "power-inverse",
        "exponential-quantile",
        "weibull-quantile",
        "weibull-inverse",
        "ph-weibull-quantile",
        "ph-custom-quantile",
    ],
)
@QUANTILE
def test_check_grads_order_two_quantile(fn):
    """spec 8.8/12.1, rev 51: `check_grads(order=2, modes=("fwd", "rev"))` on interior quantile and inverse-cumulative-hazard cases."""
    check_grads(fn, (f([0.7, 1.3]), f([0.4, 0.6])), order=2, modes=("fwd", "rev"))


@QUANTILE
def test_generic_quantile_gradient_agrees_under_jit_and_vmap():
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of generic quantile gradients."""

    def one(k):
        return _power(k, f(2.0)).quantile(f(0.4))

    ks = f([0.7, 1.5, 2.5])
    want = np.array([np.asarray(jax.grad(one)(k)) for k in ks])
    assert_close(jax.vmap(jax.grad(one))(ks), want)
    assert_close(jax.jit(jax.vmap(jax.grad(one)))(ks), want)


# =============================================================== isolation and endpoint tangents


@pytest.mark.parametrize("mode", MODES)
@QUANTILE
def test_generic_quantile_isolates_invalid_probabilities(mode):
    """spec 8.7/8.8, rev 51: invalid `p` entries leave the shared-shape gradient of valid quantiles unchanged."""
    g_full, g_alone = valid_subset_grad(
        lambda k: _power(k, f(2.0)).quantile(f([0.3, 1.5, 0.6])),
        lambda k: _power(k, f(2.0)).quantile(f([0.3, 0.6])),
        f(1.5),
        mode,
    )
    assert np.isfinite(np.asarray(g_full)).all()
    assert_close(g_full, g_alone)


@pytest.mark.parametrize("generic", [False, True], ids=["builtin", "generic"])
@QUANTILE
def test_upper_endpoint_quantile_has_a_non_finite_tangent(generic):
    """spec 8.4, rev 51: "Endpoint quantiles ... get NaN or infinite tangents": `quantile(1) = +inf` in the rate direction (Q-062)."""
    make = fam.const_hazard() if generic else survjax.Exponential
    _, tangent = jax.jvp(lambda r: make(rate=r).quantile(f(1.0)), (f(0.5),), (f(1.0),))
    assert not np.isfinite(np.asarray(tangent))
