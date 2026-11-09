"""Feature `summary.rmst`: RMST derivatives (spec 8.4, 8.7, 8.8, 12.1; rev 51).

8.8: once `rmst` is enabled, section 8.7's "derivative restrictions and the
exception to section 8.4" no longer apply; "Their absence, rejection and
derivative-exclusion checks for that method are replaced by correctness
checks"; "Section 8.7's tail accuracy ... remain[s] in force". 8.4:
"Summaries are differentiable; nothing raises"; generic tangent "`∫ ∂θ S·dθ
dt` (plus `S(tau)·dtau` for RMST), evaluated on a **static** node set ...
Forward, reverse and higher-order derivatives all work. At `tau=0`, RMST first
derivatives follow from this tangent (parameter derivatives 0, `tau`
derivative `S(0)=1`); higher-order derivatives there involve the density at 0
and carry no guarantee"; Weibull RMST is "the closed-form value plus the
quadrature tangent via `custom_jvp`" with full Hessians; "Entries that failed
or did not converge get NaN tangents, never zeros." 8.7 isolation ("NaN
entries do not give NaN derivatives to other entries, including through shared
parameters") remains as value policy.

Tolerances (12.1 log units): ``|theta (got - want)| <= scale * |value|`` per
variable (``tau`` included), higher orders multiplied by every variable once
per order. ``scale`` is the 12.1 ``rtol`` for Weibull RMST (staged accuracy)
and ``generic_rtol()`` for generic quadrature (Q-058). References: mpmath
``diff`` of the closed form at 60 digits and mpmath quadrature of
``dS/dtheta`` (``rmst_reference``). No higher-order derivative is tested at
``tau=0``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rmst_reference as rref
import spec_families as fam
from helpers import (
    X64,
    assert_close,
    assert_cutoff_derivative,
    assert_generic_log_derivative,
    f,
    generic_rtol,
    ph,
    rmst_rtol,
    rounded,
)
from jax.test_util import check_grads
from summary_cases import HESSIANS, MODES, RMST, RMST_INFO, assert_hessian, grad, valid_subset_grad

import survjax

requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")


def _power(k, b):
    return fam.power_hazard()(shape=k, scale=b)


# =============================================================== Weibull RMST shape Hessians


SHAPE_CASES = {
    "interior": (2.5, 2.0, 3.0),
    "heavy": (0.5, 2.0, 20.0),
    "underflow-tail": (50.0, 2.0, 0.02),
    "overflow-tail": (10.0, 2.0, 2e5),
}


@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize("case", list(SHAPE_CASES))
@RMST
def test_weibull_rmst_shape_second_derivative_is_correct(case, mode):
    """spec 8.4/8.8, rev 51: rejection replaced by correctness: `d2/dk2` is correct and never raises, interior and both power tails."""
    kk, bb, tt = rounded(*SHAPE_CASES[case])
    value = rref.weibull(kk, bb, tt)[0]
    want = rref.weibull_closed_form_derivative(kk, bb, tt, (2, 0, 0))
    got = float(HESSIANS[mode](lambda x: survjax.Weibull(shape=x[0], scale=f(bb)).rmst(f(tt)))(f([kk]))[0, 0])
    assert np.isfinite(got)
    assert kk**2 * abs(got - want) <= rmst_rtol() * abs(value), (got, want)


@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize("case", list(SHAPE_CASES))
@RMST
def test_weibull_rmst_full_hessian_is_correct(case, mode):
    """spec 8.4/8.8, rev 51: the `(k, b, tau)` Hessian ("full Hessians"), keeping the 8.7 tail accuracy (12.1 RMST log units)."""
    kk, bb, tt = rounded(*SHAPE_CASES[case])
    value = rref.weibull(kk, bb, tt)[0]
    got = HESSIANS[mode](lambda x: survjax.Weibull(shape=x[0], scale=x[1]).rmst(x[2]))(f([kk, bb, tt]))
    assert_hessian(
        got, (kk, bb, tt), lambda o: rref.weibull_closed_form_derivative(kk, bb, tt, o), value, rmst_rtol(), "weibull"
    )


@pytest.mark.parametrize("mode", list(HESSIANS))
@RMST
def test_ph_weibull_rmst_hessian_with_shape_is_correct(mode):
    """spec 4.4/8.4/8.8, rev 51: PH-Weibull RMST full `(k, b0, eta, tau)` Hessian through the closure."""
    kk, bb, ee, tt = rounded(2.5, 2.0, -0.7, 3.0)
    value = rref.ph_weibull(kk, bb, ee, tt)[0]
    got = HESSIANS[mode](lambda x: ph(survjax.Weibull(shape=x[0], scale=x[1]), x[2]).rmst(x[3]))(f([kk, bb, ee, tt]))
    assert_hessian(
        got,
        (kk, bb, 1.0, tt),
        lambda o: rref.ph_weibull_closed_form_derivative(kk, bb, ee, tt, o),
        value,
        rmst_rtol(),
        "ph-weibull",
    )


@requires_x64
@RMST
def test_check_grads_order_two_weibull_rmst_with_shape():
    """spec 8.8/12.1, rev 51: the derivative exclusion of the shape direction is replaced: `check_grads(order=2)` in `(k, b, tau)`."""
    check_grads(
        lambda k, b, t: survjax.Weibull(shape=k, scale=b).rmst(t),
        (f([0.7, 2.5]), f([1.5, 2.0]), f([2.0, 3.0])),
        order=2,
        modes=("fwd", "rev"),
    )


@requires_x64
@RMST
def test_check_grads_order_two_ph_weibull_rmst_with_shape():
    """spec 8.8/12.1, rev 51: `check_grads(order=2)` for PH-Weibull RMST in `(k, b0, eta, tau)`."""
    check_grads(
        lambda k, b, e, t: ph(survjax.Weibull(shape=k, scale=b), e).rmst(t),
        (f([0.7, 2.5]), f([1.5, 2.0]), f([0.5, -0.7]), f([2.0, 3.0])),
        order=2,
        modes=("fwd", "rev"),
    )


# =============================================================== tau = 0


ZERO_CUTOFF = {
    "exponential": (lambda r, t: survjax.Exponential(rate=r).rmst(t), (0.7,)),
    "weibull": (lambda k, b, t: survjax.Weibull(shape=k, scale=b).rmst(t), (2.5, 2.0)),
    "ph-weibull": (lambda k, b, e, t: ph(survjax.Weibull(shape=k, scale=b), e).rmst(t), (2.5, 2.0, -0.3)),
    "custom": (lambda r, t: fam.const_hazard()(rate=r).rmst(t), (0.5,)),
    "ph-custom": (lambda k, b, e, t: ph(_power(k, b), e).rmst(t), (1.5, 2.0, 0.3)),
}


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", list(ZERO_CUTOFF))
@RMST
def test_rmst_first_derivatives_at_zero_cutoff(family, mode):
    """spec 8.4, rev 51: "At `tau=0`, RMST first derivatives follow from this tangent (parameter derivatives 0, `tau` derivative `S(0)=1`)", on every kind of family."""
    fn, params = ZERO_CUTOFF[family]
    n = len(params)
    grads = grad(fn, tuple(range(n + 1)), mode)(*(f(x) for x in params), f(0.0))
    for i in range(n):
        assert float(grads[i]) == 0.0, f"parameter {i}: {grads[i]}"
    assert_cutoff_derivative(grads[n], 1.0)


# =============================================================== generic paths


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("k", "ratio"), [(0.5, 1.0), (2.5, 1.5), (2.5, 100.0), (10.0, 1.0)])
@RMST
def test_generic_rmst_first_derivatives_match_quadrature(k, ratio, mode):
    """spec 8.4, rev 51: static-node quadrature tangent `∫ dS/dtheta dt` plus `S(tau) dtau` against mpmath, forward and reverse."""
    kk, bb, tt = rounded(k, 2.0, ratio * 2.0)
    value, dk, db, dtau = rref.weibull(kk, bb, tt)
    gk, gb, gt = grad(lambda k_, b_, t_: _power(k_, b_).rmst(t_), (0, 1, 2), mode)(f(kk), f(bb), f(tt))
    assert_generic_log_derivative(gk, dk, kk, value, "d/dk")
    assert_generic_log_derivative(gb, db, bb, value, "d/db")
    np.testing.assert_allclose(float(gt), dtau, rtol=0, atol=generic_rtol(), err_msg="d/dtau = S(tau)")


@pytest.mark.parametrize("mode", list(HESSIANS))
@RMST
def test_generic_rmst_hessian(mode):
    """spec 8.4, rev 51: "Forward, reverse and higher-order derivatives all work": generic `(k, b, tau)` RMST Hessian, both dtypes."""
    kk, bb, tt = rounded(2.5, 2.0, 3.0)
    value = rref.weibull(kk, bb, tt)[0]
    got = HESSIANS[mode](lambda x: _power(x[0], x[1]).rmst(x[2]))(f([kk, bb, tt]))
    assert_hessian(
        got, (kk, bb, tt), lambda o: rref.weibull_closed_form_derivative(kk, bb, tt, o), value, generic_rtol(), "rmst"
    )


@pytest.mark.parametrize("mode", MODES)
@RMST
def test_ph_rmst_without_closure_derivatives_match_the_physical_law(mode):
    """spec 4.4/8.4, rev 51: PH of a custom baseline: `(k, b0, eta, tau)` RMST derivatives against mpmath quadrature."""
    kk, bb, ee, tt = rounded(1.5, 2.0, 0.3, 3.0)
    value, dk, db, de, dtau = rref.ph_weibull(kk, bb, ee, tt)
    grads = grad(lambda k_, b_, e_, t_: ph(_power(k_, b_), e_).rmst(t_), (0, 1, 2, 3), mode)(
        f(kk), f(bb), f(ee), f(tt)
    )
    assert_generic_log_derivative(grads[0], dk, kk, value, "d/dk")
    assert_generic_log_derivative(grads[1], db, bb, value, "d/db0")
    assert_generic_log_derivative(grads[2], de, 1.0, value, "d/deta")
    np.testing.assert_allclose(float(grads[3]), dtau, rtol=0, atol=generic_rtol(), err_msg="d/dtau")


@requires_x64
@pytest.mark.parametrize(
    "fn",
    [
        lambda r, t: fam.const_hazard()(rate=r).rmst(t),
        lambda k, t: _power(k, f(2.0)).rmst(t),
        lambda e, t: ph(_power(f(1.5), f(2.0)), e).rmst(t),
    ],
    ids=["custom-rmst", "power-rmst", "ph-custom-rmst"],
)
@RMST
def test_check_grads_order_two_generic_rmst(fn):
    """spec 8.8/12.1, rev 51: `check_grads(order=2, modes=("fwd", "rev"))` on interior generic RMST cases."""
    check_grads(fn, (f([0.7, 1.3]), f([0.4, 0.6])), order=2, modes=("fwd", "rev"))


@RMST
def test_generic_rmst_gradient_agrees_under_jit_and_vmap():
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of generic RMST gradients."""

    def one(k):
        return _power(k, f(2.0)).rmst(f(3.0))

    ks = f([0.7, 1.5, 2.5])
    want = np.array([np.asarray(jax.grad(one)(k)) for k in ks])
    assert_close(jax.vmap(jax.grad(one))(ks), want)
    assert_close(jax.jit(jax.vmap(jax.grad(one)))(ks), want)


# =============================================================== isolation and failed-entry tangents


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("bad", [-1.0, np.nan, np.inf], ids=["negative", "nan", "posinf"])
@RMST
def test_generic_rmst_isolates_invalid_cutoffs_with_shared_rate(bad, mode):
    """spec 8.7, rev 51: "NaN entries do not give NaN derivatives to other entries, including through shared parameters" on the generic path."""
    g_full, g_alone = valid_subset_grad(
        lambda r: fam.const_hazard()(rate=r).rmst(f([1.0, bad, 2.0])),
        lambda r: fam.const_hazard()(rate=r).rmst(f([1.0, 2.0])),
        f(0.5),
        mode,
    )
    assert np.isfinite(np.asarray(g_full)).all()
    assert_close(g_full, g_alone)


@pytest.mark.parametrize("mode", MODES)
@RMST
def test_valid_entries_isolated_from_a_non_finite_failure(mode):
    """spec 8.3/8.7, rev 51: a non-finite-arithmetic entry does not contaminate the shared-parameter gradient of valid entries."""
    g_full, g_alone = valid_subset_grad(
        lambda r: fam.nan_beyond()(rate=r, cut=f(1.0)).rmst(f([0.5, 2.0, 0.9])),
        lambda r: fam.nan_beyond()(rate=r, cut=f(1.0)).rmst(f([0.5, 0.9])),
        f(1.0),
        mode,
    )
    assert np.isfinite(np.asarray(g_full)).all()
    assert_close(g_full, g_alone)


@pytest.mark.parametrize("mode", list(HESSIANS))
@RMST
def test_weibull_rmst_hessian_isolation_with_shared_parameters(mode):
    """spec 8.7/8.8, rev 51: second-order isolation now covers Weibull (shape Hessians are supported): shared `(k, b)` beside a NaN cutoff."""
    index = jnp.asarray([0, 2])

    def full(x):
        return jnp.sum(survjax.Weibull(shape=x[0], scale=x[1]).rmst(f([1.0, np.nan, 3.0]))[index])

    def alone(x):
        return jnp.sum(survjax.Weibull(shape=x[0], scale=x[1]).rmst(f([1.0, 3.0])))

    x = f([2.5, 2.0])
    got = HESSIANS[mode](full)(x)
    assert np.isfinite(np.asarray(got)).all()
    assert_close(got, HESSIANS[mode](alone)(x))


@RMST_INFO
def test_non_converged_entry_has_a_nan_tangent():
    """spec 8.4, rev 51: "Entries that failed or did not converge get NaN tangents, never zeros" (work exhaustion)."""
    opts = survjax.SummaryOptions(rtol=f(1e-30), atol=f(1e-30), max_levels=1)
    _, tangent = jax.jvp(lambda r: fam.const_hazard()(rate=r).rmst(f(3.0), options=opts), (f(0.5),), (f(1.0),))
    assert np.isnan(np.asarray(tangent))


@RMST
def test_non_finite_failure_entry_has_a_nan_tangent():
    """spec 8.4, rev 51: the failed (non-finite arithmetic) entry's own tangent is NaN; the valid entries' are finite."""
    _, tangent = jax.jvp(
        lambda r: fam.nan_beyond()(rate=r, cut=f(1.0)).rmst(f([0.5, 2.0, 0.9])), (f(1.0),), (f(1.0),)
    )
    tangent = np.asarray(tangent)
    assert np.isnan(tangent[1])
    assert np.isfinite(tangent[[0, 2]]).all()
