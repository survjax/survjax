"""observation.truncation: first and second parameter derivatives under truncation (spec 7.1-7.3, 12.1; rev 51).

Same requirements as ``test_observation_censoring_derivatives.py`` (7.3, 12.1),
now with ``log Z`` in every row. Exponential, hand-derived: for a set
``(lo, hi]`` write ``g(lo, hi) = log(e**(-r lo) - e**(-r hi)) = -r lo + log(-expm1(-r (hi - lo)))``;
then with ``w = hi - lo``

- ``dg/dr = -lo + w/expm1(r w)``, ``d2g/dr2 = -w**2 e**(r w)/expm1(r w)**2``
  (from ``d/dr log(-expm1(-r w)) = w/expm1(r w)``);
- ``hi = +inf``: ``g = -r lo``, ``dg/dr = -lo``, ``d2g/dr2 = 0``.

A row is ``log f(t) - g(L, U)`` (exact; ``d/dr log f = 1/r - t``,
``d2 = -1/r**2``) or ``g(lo, hi) - g(L, U)`` (censored, ``lo = max(lower, L)``,
``hi = min(upper, U)``). Weibull derivatives come from ``observation_reference``
(``mpmath.diff`` at 50 digits). 12.1 tolerances as in A9; ``check_grads`` in
float64 only.
"""

import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as ref
import pytest
from helpers import INF, NAN, X64, assert_close, f, flag_data
from jax.test_util import check_grads

import survjax

TRUNC = pytest.mark.feature("observation.truncation")
BOTH = pytest.mark.feature("observation.censoring", "observation.truncation")
requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")

L0, U0 = 0.5, 3.0


def dg(r, lo, hi):
    """``(dg/dr, d2g/dr2)`` of ``g(lo, hi)`` (module docstring)."""
    if np.isinf(hi):
        return -lo, 0.0
    w = hi - lo
    e = np.expm1(r * w)
    return -lo + w / e, -(w**2) * np.exp(r * w) / e**2


def _r(x):
    return float(np.asarray(f(x)))


# Exact/right rows in (0.5, 3]: (kind, t) -> (lower, upper, exact) for the reference.
ER_KINDS = ["exact", "right", "exact", "right"]
ER_TIME = [1.0, 1.0, 2.5, 2.0]
ER_TABLE = [(1.0, 1.0, True), (1.0, INF, False), (2.5, 2.5, True), (2.0, INF, False)]

ALL_KINDS = ER_KINDS + ["left", "interval", "interval", "interval"]
ALL_TIME = ER_TIME + [2.0, 1.0, 0.2, 1.0]
ALL_TIME2 = [NAN] * 5 + [2.0, 1.5, 4.0]
ALL_TABLE = ER_TABLE + [(0.0, 2.0, False), (1.0, 2.0, False), (0.2, 1.5, False), (1.0, 4.0, False)]


def _er_data(lt=L0, rt=U0):
    n = len(ER_KINDS)
    return flag_data(ER_KINDS, ER_TIME, left_trunc=[lt] * n, right_trunc=[rt] * n)


def _all_data(lt=L0, rt=U0):
    n = len(ALL_KINDS)
    return flag_data(ALL_KINDS, ALL_TIME, ALL_TIME2, left_trunc=[lt] * n, right_trunc=[rt] * n)


def _exp_hand(r, table, lt, rt):
    """Summed ``(gradient, Hessian)`` of Exponential rows in ``(lt, rt]``."""
    gz, hz = dg(r, lt, rt)
    grad = hess = 0.0
    for lo, hi, exact in table:
        if exact:
            gr, hr = 1.0 / r - lo, -1.0 / r**2
        else:
            gr, hr = dg(r, max(lo, lt), min(hi, rt))
        grad += gr - gz
        hess += hr - hz
    return grad, hess


def _weibull_reference(table, k, s, lt, rt):
    g, h = np.zeros(2), np.zeros((2, 2))
    for lo, hi, ex in table:
        g = g + ref.gradient("weibull", (k, s), lo, hi, exact=ex, left=lt, right=rt)
        h = h + ref.hessian("weibull", (k, s), lo, hi, exact=ex, left=lt, right=rt)
    return g, h


# ----------------------------------------------------------------- Exponential


WINDOWS = {"left": (L0, INF), "right": (0.0, U0), "two-sided": (L0, U0)}


@TRUNC
@pytest.mark.parametrize("window", list(WINDOWS))
def test_exponential_exact_right_derivatives_are_hand_derived(window):
    """spec 7.3, rev 51: gradient and Hessian of summed truncated exact/right rows in a shared rate (rev, fwd, three Hessian routes)."""
    lt, rt = WINDOWS[window]
    data = _er_data(lt, rt)

    def loss(rate):
        return jnp.sum(survjax.Exponential(rate=rate).log_likelihood(data))

    r = jnp.asarray(f(0.7))
    want_g, want_h = _exp_hand(_r(0.7), ER_TABLE, lt, rt)
    assert_close(jax.grad(loss)(r), want_g)
    assert_close(jax.jacfwd(loss)(r), want_g)
    for fn in (jax.hessian(loss), jax.grad(jax.grad(loss)), jax.jacfwd(jax.jacfwd(loss))):
        assert_close(fn(r), want_h)


@BOTH
@pytest.mark.parametrize("window", list(WINDOWS))
def test_exponential_full_matrix_derivatives_are_hand_derived(window):
    """spec 7.3, rev 51: the same for all four kinds, including partial overlaps."""
    lt, rt = WINDOWS[window]
    data = _all_data(lt, rt)

    def loss(rate):
        return jnp.sum(survjax.Exponential(rate=rate).log_likelihood(data))

    r = jnp.asarray(f(0.7))
    table = [(_r(lo), _r(hi), ex) for lo, hi, ex in ALL_TABLE]
    want_g, want_h = _exp_hand(_r(0.7), table, lt, rt)
    assert_close(jax.grad(loss)(r), want_g)
    assert_close(jax.hessian(loss)(r), want_h)


@TRUNC
def test_exponential_per_row_jacobian_with_per_row_windows():
    """spec 5 and 7.3, rev 51: per-row rates and windows: the Jacobian is diagonal with each row's hand derivative."""
    data = flag_data(["exact", "right", "exact"], [1.0, 1.0, 2.5], left_trunc=[0.0, L0, L0], right_trunc=[U0, INF, U0])
    rates = jnp.asarray(f([0.5, 0.8, 2.0]))
    r = [_r(x) for x in (0.5, 0.8, 2.0)]
    expected = np.diag(
        [
            1 / r[0] - 1.0 - dg(r[0], 0.0, U0)[0],
            dg(r[1], 1.0, INF)[0] - dg(r[1], L0, INF)[0],
            1 / r[2] - 2.5 - dg(r[2], L0, U0)[0],
        ]
    )
    for jac in (jax.jacrev, jax.jacfwd):
        out = jac(lambda x: survjax.Exponential(rate=x).log_likelihood(data))(rates)
        assert_close(np.asarray(out), expected, err_msg=jac.__name__)


# ------------------------------------------------------------------- Weibull


@TRUNC
@pytest.mark.parametrize("k", [0.6, 1.0, 2.5])
def test_weibull_exact_right_hessian_matches_mpmath(k):
    """spec 7.3, rev 51: `(k, b)` gradient and Hessian of summed truncated exact/right rows, `(0.5, 3]`."""

    def loss(p):
        return jnp.sum(survjax.Weibull(shape=p[0], scale=p[1]).log_likelihood(_er_data()))

    p = jnp.asarray(f([k, 2.0]))
    want_g, want_h = _weibull_reference(ER_TABLE, k, 2.0, L0, U0)
    assert_close(np.asarray(jax.grad(loss)(p)), want_g)
    assert_close(np.asarray(jax.hessian(loss)(p)), want_h)


@BOTH
@pytest.mark.parametrize("k", [0.6, 1.0, 2.5])
def test_weibull_full_matrix_hessian_matches_mpmath(k):
    """spec 7.3, rev 51: the same for all four kinds, with partial overlaps."""

    def loss(p):
        return jnp.sum(survjax.Weibull(shape=p[0], scale=p[1]).log_likelihood(_all_data()))

    p = jnp.asarray(f([k, 2.0]))
    want_g, want_h = _weibull_reference(ALL_TABLE, k, 2.0, L0, U0)
    assert_close(np.asarray(jax.grad(loss)(p)), want_g)
    assert_close(np.asarray(jax.hessian(loss)(p)), want_h)


@BOTH
def test_forward_and_reverse_hessians_agree_with_mpmath():
    """spec 7.3, rev 51: all four `jacfwd`/`jacrev` compositions equal the mpmath Hessian on the full truncated matrix."""

    def loss(p):
        return jnp.sum(survjax.Weibull(shape=p[0], scale=p[1]).log_likelihood(_all_data()))

    p = jnp.asarray(f([1.4, 2.2]))
    _, want = _weibull_reference(ALL_TABLE, 1.4, 2.2, L0, U0)
    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            assert_close(np.asarray(outer(inner(loss))(p)), want, err_msg=f"{outer.__name__}({inner.__name__})")


@TRUNC
def test_hessian_under_jit_and_vmap():
    """spec 7.3 and 12.1, rev 51: `jit` of the Hessian equals eager; `jit(vmap)` over draws equals per-draw mpmath Hessians."""

    def loss(p):
        return jnp.sum(survjax.Weibull(shape=p[0], scale=p[1]).log_likelihood(_er_data()))

    hess = jax.hessian(loss)
    p = jnp.asarray(f([1.4, 2.2]))
    assert_close(np.asarray(jax.jit(hess)(p)), np.asarray(hess(p)))
    draws = jnp.asarray(f([[0.8, 1.5], [1.4, 2.2], [2.5, 3.0]]))
    out = np.asarray(jax.jit(jax.vmap(hess))(draws))
    want = np.stack([_weibull_reference(ER_TABLE, float(k), float(s), L0, U0)[1] for k, s in np.asarray(draws)])
    assert_close(out, want)


@TRUNC
def test_gradient_under_vmap_over_datasets_with_different_windows():
    """spec 7.3 and 12.1, rev 51: `vmap` over datasets whose windows differ: each summed gradient is its hand value."""
    data = survjax.SurvivalData(
        f([[1.0, 2.0], [1.0, 2.0]]),
        jnp.asarray([[True, False], [True, False]]),
        left_trunc=f([[0.0, 0.0], [L0, L0]]),
        right_trunc=f([[U0, U0], [INF, INF]]),
    )

    def grad_one(d):
        return jax.grad(lambda r: jnp.sum(survjax.Exponential(rate=r).log_likelihood(d)))(jnp.asarray(f(0.7)))

    table = [(1.0, 1.0, True), (2.0, INF, False)]
    want = [_exp_hand(_r(0.7), table, 0.0, U0)[0], _exp_hand(_r(0.7), table, L0, INF)[0]]
    assert_close(np.asarray(jax.vmap(grad_one)(data)), want)


# ----------------------------------------------------------------- check_grads


@TRUNC
@requires_x64
@pytest.mark.parametrize("window", list(WINDOWS))
def test_check_grads_weibull_exact_right(window):
    """spec 7.3 and 12.1, rev 51: `check_grads(order=2, modes=("fwd","rev"))` on truncated exact/right rows."""
    data = _er_data(*WINDOWS[window])

    def loss(k, s):
        return jnp.sum(survjax.Weibull(shape=k, scale=s).log_likelihood(data))

    check_grads(loss, (f(1.4), f(2.2)), order=2, modes=("fwd", "rev"))


@BOTH
@requires_x64
@pytest.mark.parametrize("window", list(WINDOWS))
def test_check_grads_weibull_full_matrix_per_row(window):
    """spec 7.3 and 12.1, rev 51: `check_grads` on the per-row contribution vector of the full truncated matrix."""
    data = _all_data(*WINDOWS[window])
    n = len(ALL_KINDS)

    def contributions(k, s):
        return survjax.Weibull(shape=k, scale=s).log_likelihood(data)

    check_grads(
        contributions,
        (f(np.linspace(0.6, 2.5, n)), f(np.linspace(1.2, 3.0, n))),
        order=2,
        modes=("fwd", "rev"),
    )


@TRUNC
@requires_x64
def test_check_grads_exponential_and_from_log_under_jit():
    """spec 4.1, 7.3 and 12.1, rev 51: `check_grads` of jitted Exponential and `Weibull.from_log` truncated objectives."""
    data = _er_data()
    check_grads(
        jax.jit(lambda r: jnp.sum(survjax.Exponential(rate=r).log_likelihood(data))), (f(0.7),), order=2, modes=("fwd", "rev")
    )
    check_grads(
        lambda u, v: jnp.sum(survjax.Weibull.from_log(log_shape=u, log_scale=v).log_likelihood(data)),
        (f(0.3), f(0.8)),
        order=2,
        modes=("fwd", "rev"),
    )


# ------------------------------------------------------------------ isolation


@TRUNC
@pytest.mark.parametrize("bad", [-1.0, 0.0, NAN, INF], ids=["negative", "zero", "nan", "posinf"])
def test_rows_without_parameter_issues_keep_derivatives(bad):
    """spec 7.1 and 7.3, rev 51: per-row rates with row 1 invalid: rows 0 and 2 keep their hand gradients and Hessians, both modes.

    Rows in ``(0.5, 3]``: exact at ``1``, right at ``1`` (invalid rate),
    exact at ``2.5``; rates ``[0.7, bad, 1.3]``.
    """
    data = flag_data(["exact", "right", "exact"], [1.0, 1.0, 2.5], left_trunc=[L0] * 3, right_trunc=[U0] * 3)
    rates = jnp.asarray(f([0.7, bad, 1.3]))
    out = np.asarray(survjax.Exponential(rate=rates).log_likelihood(data))
    assert np.isnan(out[1])
    for i, (t, r) in ((0, (1.0, 0.7)), (2, (2.5, 1.3))):
        rr = _r(r)
        gz, hz = dg(rr, L0, U0)
        want_g = np.zeros(3)
        want_g[i] = 1 / rr - t - gz
        want_h = np.zeros((3, 3))
        want_h[i, i] = -1 / rr**2 - hz

        def row(x, i=i):
            return survjax.Exponential(rate=x).log_likelihood(data)[i]

        for jac in (jax.jacrev, jax.jacfwd):
            assert_close(np.asarray(jac(row)(rates)), want_g, err_msg=f"row {i} {jac.__name__}")
        for hess in (jax.hessian, lambda fn: jax.jacrev(jax.jacrev(fn)), lambda fn: jax.jacfwd(jax.jacfwd(fn))):
            assert_close(np.asarray(hess(row)(rates)), want_h, err_msg=f"row {i}")


@TRUNC
def test_uninformative_row_has_zero_derivatives():
    """spec 2, 7.1 and 7.3, rev 51: a right row at entry contributes exactly zero for every rate, so its derivatives are zero."""
    data = flag_data(["right"], [L0], left_trunc=[L0], right_trunc=[U0])

    def loss(r):
        return jnp.sum(survjax.Exponential(rate=r).log_likelihood(data))

    r = jnp.asarray(f(0.7))
    np.testing.assert_array_equal(np.asarray(jax.grad(loss)(r)), 0.0)
    np.testing.assert_array_equal(np.asarray(jax.hessian(loss)(r)), 0.0)
