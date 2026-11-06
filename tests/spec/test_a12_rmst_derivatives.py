"""A12: staged RMST derivatives (spec 8.4, 8.7, 12.1; rev 44).

8.7: "First derivatives in all parameters and `tau`, in forward and reverse
mode ... `d rmst / d tau = S(tau)`. At `tau=0`, parameter derivatives are 0
and the `tau` derivative is `S(0)=1`." "Exponential RMST supports derivatives
of all orders. For Weibull RMST, and therefore PH-Weibull, second- and
higher-order derivatives that involve the shape `k` are **not supported**:
they raise `NotImplementedError` ... Some differentiation orders, or tail
branches that do not use `igamma`, may instead return the correct value; they
never return a wrong value. Derivatives that do not involve `k` (for example
in `b`, `eta` or `tau`) are supported to all orders on finite, valid smooth
interior inputs (`tau>0`)." No higher-order Weibull guarantee is tested at
``tau=0``.

First-derivative references are mpmath quadrature of ``dS/dtheta``
(``rmst_reference``); higher orders use mpmath ``diff`` of the closed forms at
60 digits. Tolerances (12.1, rev 44): ``theta * d/dtheta`` within
``rtol * |rmst|`` for ``k``, ``b``/``b0`` and ``r``, ``d/deta`` within
``rtol * |rmst|``, ``d/dtau`` within ``rtol``; higher orders "in the same log
units: the derivative is multiplied by each variable it is taken with respect
to (`tau` included, once per order)", scale ``|rmst|``. Shape higher
derivatives (12.1, rev 45) raise or are correct, never wrong.
"""

import itertools

import jax
import jax.numpy as jnp
import mpmath
import numpy as np
import pytest
import rmst_reference as ref
from helpers import (
    X64,
    assert_cutoff_derivative,
    assert_log_parameter_derivative,
    f,
    ph,
    rmst_rtol,
    rounded,
)
from jax.test_util import check_grads

import survjax

requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")

K_GRID = (0.2, 0.5, 1.0, 2.5, 10.0, 50.0)
RATIOS = (1e-2, 1e-1, 1.0, 10.0, 1e3, 1e5)
B = 2.0
MODES = ("rev", "fwd")


def _grad(fn, argnums, mode):
    return (jax.grad if mode == "rev" else jax.jacfwd)(fn, argnums=argnums)


def _weibull_rmst(k, b, tau):
    return survjax.Weibull(shape=k, scale=b).rmst(tau)


def _ph_weibull_rmst(k, b0, eta, tau):
    return ph(survjax.Weibull(shape=k, scale=b0), eta).rmst(tau)


def _exponential_rmst(r, tau):
    return survjax.Exponential(rate=r).rmst(tau)


def _ph_exponential_rmst(r, eta, tau):
    return ph(survjax.Exponential(rate=r), eta).rmst(tau)


def assert_higher_derivative(actual, expected, variables, orders, rmst, err_msg=""):
    """``prod(x_i**n_i) * (actual - expected)`` within ``rtol * |rmst|`` (spec 12.1, rev 44)."""
    actual = float(np.asarray(actual, dtype=np.float64))
    assert np.isfinite(actual), f"{err_msg}: non-finite {actual!r}"
    weight = float(np.prod([abs(x) ** n for x, n in zip(variables, orders, strict=True)]))
    err = weight * abs(actual - expected)
    assert err <= rmst_rtol() * abs(rmst), f"{err_msg}: {actual} vs {expected} (weighted error {err})"


# --------------------------------------------------------------- first derivatives: Weibull grid


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("k", K_GRID)
def test_weibull_first_derivatives_match_quadrature(k, ratio, mode):
    """spec 8.7/12.1, rev 44: `d/dk`, `d/db`, `d/dtau` against mpmath quadrature over the full range, forward and reverse, both dtypes."""
    kk, bb, tt = rounded(k, B, ratio * B)
    value, dk, db, dtau = ref.weibull(kk, bb, tt)
    gk, gb, gt = _grad(_weibull_rmst, (0, 1, 2), mode)(f(kk), f(bb), f(tt))
    assert_log_parameter_derivative(gk, dk, kk, value, "d/dk")
    assert_log_parameter_derivative(gb, db, bb, value, "d/db")
    assert_cutoff_derivative(gt, dtau, "d/dtau")


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("k", "ratio"),
    [(50.0, 1e-2), (50.0, 1e-1), (50.0, 10.0), (50.0, 1e5), (10.0, 1e5)],
    ids=["underflow-1e-100", "underflow-1e-50", "overflow-1e50", "overflow-1e250", "overflow-k10"],
)
def test_weibull_first_derivatives_are_finite_in_the_power_tails(k, ratio, mode):
    """spec 8.7, rev 44: the closed form "gives ... NaN gradients in both tails"; the release gives correct first derivatives there."""
    kk, bb, tt = rounded(k, B, ratio * B)
    value, dk, db, dtau = ref.weibull(kk, bb, tt)
    grads = _grad(_weibull_rmst, (0, 1, 2), mode)(f(kk), f(bb), f(tt))
    assert all(np.isfinite(np.asarray(g)) for g in grads)
    assert_log_parameter_derivative(grads[0], dk, kk, value, "d/dk")
    assert_log_parameter_derivative(grads[1], db, bb, value, "d/db")
    assert_cutoff_derivative(grads[2], dtau, "d/dtau")


@pytest.mark.step("A12")
@pytest.mark.parametrize(("k", "b", "tau"), [(0.7, 1.5, 2.0), (2.5, 2.0, 3.0), (10.0, 1.0, 0.95)])
def test_cutoff_derivative_is_the_survival_at_the_cutoff(k, b, tau):
    """spec 8.7, rev 44: "`d rmst / d tau = S(tau)`"; Weibull `S(tau) = exp(-(tau/b)**k)`, also per entry."""
    kk, bb, tt = rounded(k, b, tau)
    g = jax.grad(_weibull_rmst, argnums=2)(f(kk), f(bb), f(tt))
    assert_cutoff_derivative(g, np.exp(-((tt / bb) ** kk)))
    taus = f([0.5 * tau, tau, 2.0 * tau])
    jac = jax.jacrev(lambda t: _weibull_rmst(f(kk), f(bb), t))(taus)
    assert_cutoff_derivative(jac, np.diag(np.exp(-((np.asarray(taus, np.float64) / bb) ** kk))))


# --------------------------------------------------------------- first derivatives: Exponential and PH


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("ratio", RATIOS)
@pytest.mark.parametrize("r", [0.25, 3.0])
def test_exponential_first_derivatives_match_reference(r, ratio, mode):
    """spec 8.7, rev 44: `d/dr` and `d/dtau = exp(-r tau)` (mpmath), forward and reverse."""
    rr, tt = rounded(r, ratio / r)
    value, dr, dtau = ref.exponential(rr, tt)
    gr, gt = _grad(_exponential_rmst, (0, 1), mode)(f(rr), f(tt))
    assert_log_parameter_derivative(gr, dr, rr, value, "d/dr")
    assert_cutoff_derivative(gt, dtau, "d/dtau")


PH_CASES = [
    (0.5, 2.0, -1.5, 0.5),
    (0.5, 2.0, 0.8, 50.0),
    (1.0, 2.0, 0.3, 2.0),
    (2.5, 2.0, -0.7, 3.0),
    (2.5, 2.0, 1.2, 0.05),
    (10.0, 2.0, 0.5, 1.9),
    (50.0, 2.0, -2.0, 1e4),
]


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", PH_CASES, ids=str)
def test_ph_weibull_first_derivatives_match_physical_law_quadrature(case, mode):
    """spec 4.4/8.7, rev 44: derivatives in `k`, `b0`, `eta` (predictor) and `tau` against quadrature of `exp(-exp(eta)(t/b0)**k)`."""
    k, b0, eta, tau = rounded(*case)
    value, dk, db0, deta, dtau = ref.ph_weibull(k, b0, eta, tau)
    gk, gb, ge, gt = _grad(_ph_weibull_rmst, (0, 1, 2, 3), mode)(f(k), f(b0), f(eta), f(tau))
    assert_log_parameter_derivative(gk, dk, k, value, "d/dk")
    assert_log_parameter_derivative(gb, db0, b0, value, "d/db0")
    assert_log_parameter_derivative(ge, deta, 1.0, value, "d/deta")
    assert_cutoff_derivative(gt, dtau, "d/dtau")


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(("r", "eta", "tau"), [(0.5, -1.0, 2.0), (2.0, 0.7, 0.01), (0.1, 2.0, 1e3)])
def test_ph_exponential_first_derivatives_are_hand_derived(r, eta, tau, mode):
    """spec 4.4/8.7, rev 44: with `q = r e**eta`, `d/dr = e**eta dR/dq`, `d/deta = q dR/dq`, `d/dtau = exp(-q tau)`."""
    rr, ee, tt = rounded(r, eta, tau)
    q = rr * np.exp(ee)
    value, dq, dtau = ref.exponential(q, tt)
    gr, ge, gt = _grad(_ph_exponential_rmst, (0, 1, 2), mode)(f(rr), f(ee), f(tt))
    assert_log_parameter_derivative(gr, np.exp(ee) * dq, rr, value, "d/dr")
    assert_log_parameter_derivative(ge, q * dq, 1.0, value, "d/deta")
    assert_cutoff_derivative(gt, dtau, "d/dtau")


@pytest.mark.step("A12")
@pytest.mark.parametrize("case", PH_CASES, ids=str)
def test_ph_weibull_first_derivatives_equal_equivalent_weibull(case):
    """spec 4.4, rev 44: the PH identity for delegated summaries in first derivatives (chain rule through `b0 exp(-eta/k)`)."""
    args = tuple(f(x) for x in case)

    def eq(k, b0, eta, tau):
        return survjax.Weibull(shape=k, scale=b0 * jnp.exp(-eta / k)).rmst(tau)

    value = float(np.asarray(_ph_weibull_rmst(*args)))
    got = jax.grad(_ph_weibull_rmst, argnums=(0, 1, 2, 3))(*args)
    want = jax.grad(eq, argnums=(0, 1, 2, 3))(*args)
    scales = rounded(case[0], case[1], 1.0)
    for g, w, s, name in zip(got[:3], want[:3], scales, ("k", "b0", "eta"), strict=True):
        assert_log_parameter_derivative(g, np.asarray(w), s, value, f"d/d{name}")
    assert_cutoff_derivative(got[3], np.asarray(want[3]))


# --------------------------------------------------------------- tau = 0


TAU0_FAMILIES = {
    "exponential": (_exponential_rmst, (0.7,)),
    "weibull-k0.5": (_weibull_rmst, (0.5, 2.0)),
    "weibull-k1": (_weibull_rmst, (1.0, 2.0)),
    "weibull-k2.5": (_weibull_rmst, (2.5, 2.0)),
    "ph-exponential": (_ph_exponential_rmst, (0.7, 0.4)),
    "ph-weibull": (_ph_weibull_rmst, (1.5, 2.0, -0.6)),
}


@pytest.mark.step("A12")
@pytest.mark.parametrize("family", list(TAU0_FAMILIES))
def test_rmst_at_zero_is_zero(family):
    """spec 4.5/8.7, rev 41: "`rmst(0)` | `0`"."""
    fn, params = TAU0_FAMILIES[family]
    np.testing.assert_array_equal(np.asarray(fn(*(f(p) for p in params), f(0.0))), 0.0)


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("family", list(TAU0_FAMILIES))
def test_first_derivatives_at_zero_cutoff(family, mode):
    """spec 8.7, rev 44: "At `tau=0`, parameter derivatives are 0 and the `tau` derivative is `S(0)=1`", forward and reverse."""
    fn, params = TAU0_FAMILIES[family]
    n = len(params)
    grads = _grad(fn, tuple(range(n + 1)), mode)(*(f(p) for p in params), f(0.0))
    for g in grads[:n]:
        np.testing.assert_array_equal(np.asarray(g), 0.0)
    assert_cutoff_derivative(grads[n], 1.0)


# --------------------------------------------------------------- Exponential: all orders


def _mp_ph_exponential(r, eta, tau):
    q = r * mpmath.exp(eta)
    return -mpmath.expm1(-q * tau) / q


HESSIANS = {
    "rev-over-rev": lambda fn: jax.jacrev(jax.jacrev(fn)),
    "fwd-over-rev": lambda fn: jax.jacfwd(jax.jacrev(fn)),
    "fwd-over-fwd": lambda fn: jax.jacfwd(jax.jacfwd(fn)),
    "rev-over-fwd": lambda fn: jax.jacrev(jax.jacfwd(fn)),
}


def _assert_hessian(got, variables, ref_fn, value, err_msg):
    n = len(variables)
    got = np.asarray(got, dtype=np.float64)
    assert got.shape == (n, n)
    for i, j in itertools.product(range(n), repeat=2):
        orders = [0] * n
        orders[i] += 1
        orders[j] += 1
        assert_higher_derivative(got[i, j], ref_fn(tuple(orders)), variables, orders, value, f"{err_msg}[{i},{j}]")


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize(("r", "tau"), [(0.5, 3.0), (2.0, 0.4), (0.05, 100.0)])
def test_exponential_hessian_in_rate_and_cutoff(r, tau, mode):
    """spec 8.7/12.1, rev 44: "finite Exponential Hessians"; full `(r, tau)` Hessian against mpmath `diff` of `-expm1(-r tau)/r`."""
    rr, tt = rounded(r, tau)
    got = HESSIANS[mode](lambda x: _exponential_rmst(x[0], x[1]))(f([rr, tt]))
    value = ref.exponential(rr, tt)[0]
    _assert_hessian(got, (rr, tt), lambda o: ref.exponential_derivative(rr, tt, o), value, "hessian")


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", list(HESSIANS))
def test_ph_exponential_hessian_in_rate_predictor_and_cutoff(mode):
    """spec 8.7, rev 44: PH-Exponential `(r, eta, tau)` Hessian against mpmath `diff` of `-expm1(-r e**eta tau)/(r e**eta)`."""
    rr, ee, tt = rounded(0.6, 0.4, 2.0)
    got = HESSIANS[mode](lambda x: _ph_exponential_rmst(x[0], x[1], x[2]))(f([rr, ee, tt]))
    with mpmath.workdps(40):
        value = float(_mp_ph_exponential(mpmath.mpf(rr), mpmath.mpf(ee), mpmath.mpf(tt)))

    def ref_fn(orders):
        with mpmath.workdps(60):
            return float(mpmath.diff(_mp_ph_exponential, (mpmath.mpf(rr), mpmath.mpf(ee), mpmath.mpf(tt)), orders))

    _assert_hessian(got, (rr, 1.0, tt), ref_fn, value, "ph-exp hessian")


@pytest.mark.step("A12")
@pytest.mark.parametrize("orders", [(3, 0), (0, 3), (2, 1), (1, 2)], ids=["rrr", "ttt", "rrt", "rtt"])
def test_exponential_third_derivatives(orders):
    """spec 8.7, rev 44: "Exponential RMST supports derivatives of all orders"; e.g. `d3/dtau3 = r**2 exp(-r tau)`."""
    rr, tt = rounded(0.5, 3.0)
    fn = _exponential_rmst
    for argnum in [0] * orders[0] + [1] * orders[1]:
        fn = jax.grad(fn, argnums=argnum)
    got = fn(f(rr), f(tt))
    want = ref.exponential_derivative(rr, tt, orders)
    if orders == (0, 3):
        assert want == pytest.approx(rr**2 * np.exp(-rr * tt), rel=1e-12)
    assert_higher_derivative(got, want, (rr, tt), orders, ref.exponential(rr, tt)[0])


@pytest.mark.step("A12")
@requires_x64
def test_check_grads_order_two_exponential_rmst():
    """spec 12.1, rev 41: `check_grads(order=2, modes=("fwd","rev"))` in `(r, tau)` on interior inputs."""
    check_grads(_exponential_rmst, (f([0.5, 2.0]), f([3.0, 0.4])), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A12")
@requires_x64
def test_check_grads_order_two_ph_exponential_rmst():
    """spec 12.1, rev 41: `check_grads(order=2)` in `(r, eta, tau)`."""
    check_grads(_ph_exponential_rmst, (f([0.5, 2.0]), f([0.3, -0.4]), f([3.0, 0.4])), order=2, modes=("fwd", "rev"))


# --------------------------------------------------------------- Weibull: derivatives not involving k


WEIBULL_INTERIOR = [(2.5, 2.0, 3.0), (0.7, 1.5, 2.0), (10.0, 2.0, 1.9)]


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize(("k", "b", "tau"), WEIBULL_INTERIOR)
def test_weibull_hessian_in_scale_and_cutoff(k, b, tau, mode):
    """spec 8.7/12.1, rev 44: "Weibull Hessians in `b` and `tau` on valid smooth interior inputs", fixed `k`, all mode compositions."""
    kk, bb, tt = rounded(k, b, tau)
    got = HESSIANS[mode](lambda x: _weibull_rmst(f(kk), x[0], x[1]))(f([bb, tt]))
    value = ref.weibull(kk, bb, tt)[0]
    _assert_hessian(
        got, (bb, tt), lambda o: ref.weibull_closed_form_derivative(kk, bb, tt, (0, *o)), value, "weibull (b, tau)"
    )


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize(("k", "b0", "eta", "tau"), [(2.5, 2.0, -0.7, 3.0), (0.7, 1.5, 0.5, 2.0)])
def test_ph_weibull_hessian_in_scale_predictor_and_cutoff(k, b0, eta, tau, mode):
    """spec 8.7, rev 44: derivatives "in `b`, `eta` or `tau`" of PH-Weibull to second order, fixed `k`."""
    kk, bb, ee, tt = rounded(k, b0, eta, tau)
    got = HESSIANS[mode](lambda x: _ph_weibull_rmst(f(kk), x[0], x[1], x[2]))(f([bb, ee, tt]))
    value = ref.ph_weibull(kk, bb, ee, tt)[0]
    _assert_hessian(
        got,
        (bb, 1.0, tt),
        lambda o: ref.ph_weibull_closed_form_derivative(kk, bb, ee, tt, (0, *o)),
        value,
        "ph-weibull (b0, eta, tau)",
    )


@pytest.mark.step("A12")
@pytest.mark.parametrize(("k", "b", "tau"), WEIBULL_INTERIOR)
def test_weibull_third_cutoff_derivative(k, b, tau):
    """spec 8.7, rev 44: non-shape derivatives "to all orders": `d3/dtau3 rmst = S''(tau)`."""
    kk, bb, tt = rounded(k, b, tau)
    fn = jax.grad(jax.grad(jax.grad(lambda t: _weibull_rmst(f(kk), f(bb), t))))
    want = ref.weibull_closed_form_derivative(kk, bb, tt, (0, 0, 3))
    assert_higher_derivative(fn(f(tt)), want, (tt,), (3,), ref.weibull(kk, bb, tt)[0])


@pytest.mark.step("A12")
@requires_x64
def test_check_grads_order_two_weibull_rmst_without_shape():
    """spec 12.1, rev 41: `check_grads(order=2)` "excluding the Weibull RMST shape direction": `(b, tau)` at fixed `k`."""
    k = f([0.7, 2.5])
    check_grads(lambda b, t: _weibull_rmst(k, b, t), (f([1.5, 2.0]), f([2.0, 3.0])), order=2, modes=("fwd", "rev"))


@pytest.mark.step("A12")
@requires_x64
def test_check_grads_order_two_ph_weibull_rmst_without_shape():
    """spec 12.1, rev 41: `check_grads(order=2)` in `(b0, eta, tau)` at fixed `k`."""
    k = f([0.7, 2.5])
    check_grads(
        lambda b0, eta, t: _ph_weibull_rmst(k, b0, eta, t),
        (f([1.5, 2.0]), f([0.5, -0.7]), f([2.0, 3.0])),
        order=2,
        modes=("fwd", "rev"),
    )


# --------------------------------------------------------------- Weibull shape: raise or be correct


SHAPE_CASES = {
    "interior": (2.5, 2.0, 3.0),
    "heavy": (0.5, 2.0, 20.0),
    "underflow-tail": (50.0, 2.0, 0.02),
    "overflow-tail": (10.0, 2.0, 2e5),
}


def _raises_or_matches(compute, check):
    """8.7/12.1 (rev 45): a shape higher derivative raises `NotImplementedError` or returns the correct value, never a wrong one."""
    try:
        got = compute()
    except NotImplementedError:
        return
    check(np.asarray(got, dtype=np.float64))


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize("case", list(SHAPE_CASES))
def test_weibull_shape_hessian_raises_or_is_correct(case, mode):
    """spec 8.7/12.1, rev 45: Weibull Hessians in `k` "either raising `NotImplementedError` or returning the correct value, never a wrong value"."""
    kk, bb, tt = rounded(*SHAPE_CASES[case])
    value = ref.weibull(kk, bb, tt)[0]
    want = ref.weibull_closed_form_derivative(kk, bb, tt, (2, 0, 0))
    _raises_or_matches(
        lambda: HESSIANS[mode](lambda x: _weibull_rmst(x[0], f(bb), f(tt)))(f([kk]))[0, 0],
        lambda got: assert_higher_derivative(got, want, (kk,), (2,), value, "d2/dk2"),
    )


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", list(HESSIANS))
@pytest.mark.parametrize("case", list(SHAPE_CASES))
def test_weibull_full_hessian_raises_or_is_correct(case, mode):
    """spec 8.7/12.1, rev 45: the `(k, b, tau)` Hessian involves `k`; it raises or every entry is correct."""
    kk, bb, tt = rounded(*SHAPE_CASES[case])
    value = ref.weibull(kk, bb, tt)[0]
    _raises_or_matches(
        lambda: HESSIANS[mode](lambda x: _weibull_rmst(x[0], x[1], x[2]))(f([kk, bb, tt])),
        lambda got: _assert_hessian(
            got, (kk, bb, tt), lambda o: ref.weibull_closed_form_derivative(kk, bb, tt, o), value, "weibull"
        ),
    )


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", list(HESSIANS))
def test_ph_weibull_shape_hessian_raises_or_is_correct(mode):
    """spec 8.7/12.1, rev 45: "alone and through PH": the `(k, eta)` Hessian raises or is correct."""
    kk, bb, ee, tt = rounded(2.5, 2.0, -0.7, 3.0)
    value = ref.ph_weibull(kk, bb, ee, tt)[0]
    _raises_or_matches(
        lambda: HESSIANS[mode](lambda x: _ph_weibull_rmst(x[0], f(bb), x[1], f(tt)))(f([kk, ee])),
        lambda got: _assert_hessian(
            got,
            (kk, 1.0),
            lambda o: ref.ph_weibull_closed_form_derivative(kk, bb, ee, tt, (o[0], 0, o[1], 0)),
            value,
            "ph-weibull (k, eta)",
        ),
    )
