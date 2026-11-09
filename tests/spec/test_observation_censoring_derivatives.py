"""observation.censoring: first and second parameter derivatives of left/interval rows (spec 7.1-7.3, 12.1; rev 51).

7.3: "Built-in likelihoods require first and second parameter derivatives on
finite, valid, smooth interior cases, composed with `jit` and `vmap`, in
forward and reverse modes, including Hessians"; 12.1: `check_grads(order=2,
modes=("fwd","rev"))` on interior cases and reference-input gradient
isolation; 7.1/7.3: isolation of rows without parameter issues to second
order.

Exponential, hand-derived (``H = r t``):

- left at ``c``: ``l = log(1 - e**(-r c))``; ``dl/dr = c e**(-r c)/(1 - e**(-r c)) = c/expm1(r c)``;
  ``d2l/dr2 = d/dr [c (e**(r c) - 1)**-1] = -c**2 e**(r c)/expm1(r c)**2``.
- interval ``(a, b]``, ``w = b - a``: ``l = -r a + log(1 - e**(-r w))``;
  ``dl/dr = -a + w/expm1(r w)``; ``d2l/dr2 = -w**2 e**(r w)/expm1(r w)**2``.

Weibull derivatives in ``(k, b)`` come from ``observation_reference``
(``mpmath.diff`` at 50 digits of the spec 2 table). The entries compared are
of order ``1e-2`` to ``10`` and come from a few elementary operations, so the
12.1 tolerances apply, as in A9. ``check_grads`` uses second-order finite
differences and runs in float64 only.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as ref
import pytest
import spec_families as fam
from helpers import NAN, X64, assert_close, f, flag_data
from jax.test_util import check_grads

import survjax

FEATURE = pytest.mark.feature("observation.censoring")
requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")

# Rows (lower, upper, exact) for the reference, and the matching flag data.
KINDS = ["exact", "right", "left", "interval", "left", "interval"]
TIME = [1.0, 2.0, 1.5, 0.5, 4.0, 2.0]
TIME2 = [NAN, NAN, NAN, 2.5, NAN, 2.2]
TABLE = [
    (1.0, 1.0, True),
    (2.0, np.inf, False),
    (0.0, 1.5, False),
    (0.5, 2.5, False),
    (0.0, 4.0, False),
    (2.0, 2.2, False),
]


def _data():
    return flag_data(KINDS, TIME, TIME2)


def _weibull_sum(k, s):
    return jnp.sum(survjax.Weibull(shape=k, scale=s).log_likelihood(_data()))


def _weibull_reference_sum(k, s):
    g, h = np.zeros(2), np.zeros((2, 2))
    for lo, hi, ex in TABLE:
        g = g + ref.gradient("weibull", (k, s), lo, hi, exact=ex)
        h = h + ref.hessian("weibull", (k, s), lo, hi, exact=ex)
    return g, h


def _r(x):
    return float(np.asarray(f(x)))


def exp_left_derivs(r, c):
    e = np.expm1(r * c)
    return c / e, -(c**2) * np.exp(r * c) / e**2


def exp_interval_derivs(r, a, b_):
    w = b_ - a
    e = np.expm1(r * w)
    return -a + w / e, -(w**2) * np.exp(r * w) / e**2


# --------------------------------------------------------------- Exponential, hand-derived


@FEATURE
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_exponential_per_row_jacobian_is_hand_derived(mode):
    """spec 7.3, rev 51: `jacrev`/`jacfwd` of left/interval contributions in per-row rates are diagonal with `c/expm1(r c)` and `-a + w/expm1(r w)`."""
    data = flag_data(["left", "interval", "left"], [2.0, 1.0, 0.3], [NAN, 3.0, NAN])
    rates = jnp.asarray(f([0.5, 0.8, 2.0]))
    jac = jax.jacrev if mode == "rev" else jax.jacfwd
    out = np.asarray(jac(lambda r: survjax.Exponential(rate=r).log_likelihood(data))(rates))
    r = [_r(x) for x in (0.5, 0.8, 2.0)]
    expected = np.diag(
        [exp_left_derivs(r[0], 2.0)[0], exp_interval_derivs(r[1], 1.0, 3.0)[0], exp_left_derivs(r[2], _r(0.3))[0]]
    )
    assert_close(out, expected)


@FEATURE
def test_exponential_left_and_interval_second_derivatives_are_hand_derived():
    """spec 7.3, rev 51: `d2/dr2` of a left row is `-c**2 e**(r c)/expm1(r c)**2`, of an interval row `-w**2 e**(r w)/expm1(r w)**2`."""
    r = jnp.asarray(f(0.7))
    for kinds, time, time2, expected in (
        (["left"], [2.0], None, exp_left_derivs(_r(0.7), 2.0)[1]),
        (["interval"], [1.0], [3.5], exp_interval_derivs(_r(0.7), 1.0, 3.5)[1]),
    ):
        data = flag_data(kinds, time, time2)

        def loss(rate, data=data):
            return jnp.sum(survjax.Exponential(rate=rate).log_likelihood(data))

        assert_close(jax.hessian(loss)(r), expected, err_msg=kinds[0])
        assert_close(jax.grad(jax.grad(loss))(r), expected, err_msg=kinds[0])
        assert_close(jax.jacfwd(jax.jacfwd(loss))(r), expected, err_msg=kinds[0])


@FEATURE
def test_exponential_shared_rate_mixed_batch_derivatives_are_hand_derived():
    """spec 7.3, rev 51: summed mixed exact/right/left/interval rows with one rate: gradient and Hessian are the sums of the per-kind formulas.

    Exact ``t``: ``1/r - t`` and ``-1/r**2``; right ``c``: ``-c`` and ``0``.
    """
    data = _data()
    r = _r(0.8)
    g = (1 / r - 1.0) + (-2.0) + exp_left_derivs(r, 1.5)[0] + exp_interval_derivs(r, 0.5, 2.5)[0]
    g += exp_left_derivs(r, 4.0)[0] + exp_interval_derivs(r, 2.0, _r(2.2))[0]
    h = -1 / r**2 + exp_left_derivs(r, 1.5)[1] + exp_interval_derivs(r, 0.5, 2.5)[1]
    h += exp_left_derivs(r, 4.0)[1] + exp_interval_derivs(r, 2.0, _r(2.2))[1]

    def loss(rate):
        return jnp.sum(survjax.Exponential(rate=rate).log_likelihood(data))

    rate = jnp.asarray(f(0.8))
    assert_close(jax.grad(loss)(rate), g)
    assert_close(jax.hessian(loss)(rate), h)


# --------------------------------------------------------------- Weibull, mpmath


@FEATURE
@pytest.mark.parametrize("k", [0.6, 1.0, 2.5])
def test_weibull_left_and_interval_gradients_match_mpmath(k):
    """spec 7.3, rev 51: `(d/dk, d/db)` of a left row at `1.5` and an interval row `(0.5, 2.5]` (Weibull `b=2.2`)."""
    data = flag_data(["left", "interval"], [1.5, 0.5], [NAN, 2.5])

    def contributions(kk, s):
        return survjax.Weibull(shape=kk, scale=s).log_likelihood(data)

    jac = jax.jacrev(contributions, argnums=(0, 1))(jnp.asarray(f(k)), jnp.asarray(f(2.2)))
    actual = np.stack([np.asarray(jac[0]), np.asarray(jac[1])], axis=-1)
    expected = np.stack(
        [
            ref.gradient("weibull", (k, 2.2), 0.0, 1.5),
            ref.gradient("weibull", (k, 2.2), 0.5, 2.5),
        ]
    )
    assert_close(actual, expected)


@FEATURE
@pytest.mark.parametrize("k", [0.6, 1.0, 2.5])
def test_weibull_mixed_batch_hessian_matches_mpmath(k):
    """spec 7.3, rev 51: full `(k, b)` Hessian of the summed mixed batch, including the cross term, for `k<1`, `k=1`, `k>1`."""
    hess = jax.hessian(_weibull_sum, argnums=(0, 1))(jnp.asarray(f(k)), jnp.asarray(f(2.2)))
    actual = np.array([[hess[0][0], hess[0][1]], [hess[1][0], hess[1][1]]], dtype=float)
    _, expected = _weibull_reference_sum(k, 2.2)
    assert_close(actual, expected)


@FEATURE
def test_weibull_mixed_batch_gradient_matches_mpmath():
    """spec 7.3, rev 51: gradient of the summed mixed batch in `(k, b)`."""
    grad = jax.grad(_weibull_sum, argnums=(0, 1))(jnp.asarray(f(1.4)), jnp.asarray(f(2.2)))
    expected, _ = _weibull_reference_sum(1.4, 2.2)
    assert_close(np.asarray(grad), expected)


@FEATURE
def test_forward_and_reverse_hessians_agree_with_mpmath():
    """spec 7.3, rev 51: `jacfwd(jacrev)`, `jacrev(jacfwd)`, `jacrev(jacrev)` and `jacfwd(jacfwd)` all equal the mpmath Hessian."""
    params = jnp.asarray(f([1.4, 2.2]))

    def loss(p):
        return _weibull_sum(p[0], p[1])

    _, expected = _weibull_reference_sum(1.4, 2.2)
    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            assert_close(np.asarray(outer(inner(loss))(params)), expected, err_msg=f"{outer.__name__}({inner.__name__})")


@FEATURE
def test_hessian_under_jit_matches_eager():
    """spec 7.3 and 12.1, rev 51: the Hessian composes with `jit`."""
    hess = jax.hessian(lambda p: _weibull_sum(p[0], p[1]))
    params = jnp.asarray(f([1.4, 2.2]))
    assert_close(np.asarray(jax.jit(hess)(params)), np.asarray(hess(params)))


@FEATURE
def test_hessian_under_vmap_over_parameter_draws():
    """spec 7.3 and 12.1, rev 51: `vmap` of the Hessian over draws equals per-draw mpmath Hessians."""
    params = jnp.asarray(f([[0.8, 1.5], [1.4, 2.2], [2.5, 3.0]]))
    actual = np.asarray(jax.jit(jax.vmap(jax.hessian(lambda p: _weibull_sum(p[0], p[1]))))(params))
    expected = np.stack([_weibull_reference_sum(float(k), float(s))[1] for k, s in np.asarray(params)])
    assert_close(actual, expected)


@FEATURE
def test_gradient_under_vmap_over_datasets():
    """spec 7.3 and 12.1, rev 51: `vmap` over datasets of the summed gradient: per dataset `c/expm1(r c)` (left) plus `-a + w/expm1(r w)` (interval)."""
    data = survjax.SurvivalData(
        f([[1.5, 0.5], [0.5, 3.0]]),
        left_censored=jnp.asarray([[True, False], [False, True]]),
        interval_censored=jnp.asarray([[False, True], [True, False]]),
        time2=f([[NAN, 2.5], [1.0, NAN]]),
    )

    def grad_one(d):
        return jax.grad(lambda r: jnp.sum(survjax.Exponential(rate=r).log_likelihood(d)))(jnp.asarray(f(0.8)))

    r = _r(0.8)
    expected = [
        exp_left_derivs(r, 1.5)[0] + exp_interval_derivs(r, 0.5, 2.5)[0],
        exp_interval_derivs(r, 0.5, 1.0)[0] + exp_left_derivs(r, 3.0)[0],
    ]
    assert_close(np.asarray(jax.vmap(grad_one)(data)), expected)


# ------------------------------------------------------------------- check_grads


@FEATURE
@requires_x64
def test_check_grads_exponential_per_row_contributions():
    """spec 7.3 and 12.1, rev 51: `check_grads(order=2, modes=("fwd","rev"))` on the mixed contribution vector, per-row rates."""

    def contributions(r):
        return survjax.Exponential(rate=r).log_likelihood(_data())

    check_grads(contributions, (f([0.5, 0.7, 1.1, 1.6, 2.0, 0.9]),), order=2, modes=("fwd", "rev"))


@FEATURE
@requires_x64
@pytest.mark.parametrize("k", [0.6, 1.0, 1.4, 3.0])
def test_check_grads_weibull_shared_parameters(k):
    """spec 7.3 and 12.1, rev 51: `check_grads` in `(k, b)` on the summed mixed batch."""
    check_grads(_weibull_sum, (f(k), f(2.2)), order=2, modes=("fwd", "rev"))


@FEATURE
@requires_x64
def test_check_grads_weibull_per_row_contributions():
    """spec 7.3 and 12.1, rev 51: `check_grads` on the contribution vector with per-row shapes and scales."""

    def contributions(k, s):
        return survjax.Weibull(shape=k, scale=s).log_likelihood(_data())

    check_grads(
        contributions,
        (f([0.6, 0.9, 1.3, 2.0, 3.0, 1.1]), f([1.0, 1.5, 2.2, 3.0, 0.7, 2.0])),
        order=2,
        modes=("fwd", "rev"),
    )


@FEATURE
@requires_x64
def test_check_grads_weibull_from_log():
    """spec 4.1 and 7.3, rev 51: `check_grads` through `Weibull.from_log` on the mixed batch."""

    def loss(u, v):
        return jnp.sum(survjax.Weibull.from_log(log_shape=u, log_scale=v).log_likelihood(_data()))

    check_grads(loss, (f(0.3), f(0.8)), order=2, modes=("fwd", "rev"))


@FEATURE
@requires_x64
def test_check_grads_under_jit_and_vmap():
    """spec 7.3 and 12.1, rev 51: `check_grads` of the jitted sum and of a `vmap` over draws."""
    check_grads(jax.jit(_weibull_sum), (f(1.4), f(2.2)), order=2, modes=("fwd", "rev"))
    check_grads(jax.vmap(_weibull_sum), (f([0.7, 1.4, 2.5]), f([1.5, 2.2, 3.0])), order=2, modes=("fwd", "rev"))


@FEATURE
@requires_x64
def test_check_grads_custom_family_on_generic_defaults():
    """spec 4.2 and 7.3, rev 51: "The custom-family recipe is verified where its primitives are twice differentiable" (`power_hazard`)."""

    def loss(k, s):
        return jnp.sum(fam.power_hazard()(shape=k, scale=s).log_likelihood(_data()))

    check_grads(loss, (f(1.4), f(2.2)), order=2, modes=("fwd", "rev"))


@FEATURE
def test_custom_family_hessian_matches_mpmath():
    """spec 4.2 and 7.3, rev 51: `power_hazard` on generic defaults has the Weibull Hessian on the mixed batch."""

    def loss(p):
        return jnp.sum(fam.power_hazard()(shape=p[0], scale=p[1]).log_likelihood(_data()))

    _, expected = _weibull_reference_sum(1.4, 2.2)
    assert_close(np.asarray(jax.hessian(loss)(jnp.asarray(f([1.4, 2.2])))), expected)


# ------------------------------------------------------------------- isolation


@FEATURE
@pytest.mark.parametrize("bad", [-1.0, 0.0, NAN, np.inf], ids=["negative", "zero", "nan", "posinf"])
def test_rows_without_parameter_issues_keep_derivatives(bad):
    """spec 7.1 and 7.3, rev 51: with per-row scales and row 1 invalid, left/interval rows 0 and 2 keep their mpmath gradients and Hessians, both modes.

    Rows: left at ``1.5``, interval ``(0.5, 2.5]`` (invalid scale), interval
    ``(2.0, 2.2]``; shared ``k = 1.4``, scales ``[2.2, bad, 3.0]``. The
    contribution of row ``i`` depends only on its own scale, so the expected
    Jacobian rows are zero except their own entry.
    """
    data = flag_data(["left", "interval", "interval"], [1.5, 0.5, 2.0], [NAN, 2.5, 2.2])

    def row(i):
        def fn(scales):
            return survjax.Weibull(shape=f(1.4), scale=scales).log_likelihood(data)[i]

        return fn

    scales = jnp.asarray(f([2.2, bad, 3.0]))
    out = np.asarray(survjax.Weibull(shape=f(1.4), scale=scales).log_likelihood(data))
    assert np.isnan(out[1])
    for i, (lo, hi, s) in ((0, (0.0, 1.5, 2.2)), (2, (2.0, 2.2, 3.0))):
        g_ref = ref.gradient("weibull", (1.4, s), lo, hi)[1]
        h_ref = ref.hessian("weibull", (1.4, s), lo, hi)[1, 1]
        want_g = np.zeros(3)
        want_g[i] = g_ref
        want_h = np.zeros((3, 3))
        want_h[i, i] = h_ref
        for jac in (jax.jacrev, jax.jacfwd):
            assert_close(np.asarray(jac(row(i))(scales)), want_g, err_msg=f"row {i} {jac.__name__}")
        for hess in (jax.hessian, lambda fn: jax.jacrev(jax.jacrev(fn)), lambda fn: jax.jacfwd(jax.jacfwd(fn))):
            assert_close(np.asarray(hess(row(i))(scales)), want_h, err_msg=f"row {i}")


@functools.cache
def _steep_reference():
    """Exponential-like family whose ``reference_time()`` has an infinite derivative at ``rate = 1`` (as in A9).

    ``S(t) = exp(-rate t)``; ``reference_time() = 1 + sqrt(rate - 1)``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class SteepReferenceCensoring(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            t = jnp.asarray(t)
            ts = jnp.where(t < 0, 0.0, t)
            return jnp.where(t < 0, 0.0, -self.rate * ts)

        def reference_time(self):
            return 1.0 + jnp.sqrt(self.rate - 1.0)

    return SteepReferenceCensoring


@FEATURE
def test_reference_input_does_not_reach_censored_row_gradients():
    """spec 7.2 and 12.1, rev 51: "For **censored** rows, the inactive density piece uses `reference_time()`", with `stop_gradient`.

    Rows: left at ``2`` and interval ``(1, 3]``, ``r = 1``. Hand values:
    ``dl/dr = 2/expm1(2) + (-1 + 2/expm1(2)) = -1 + 4/expm1(2)`` and
    ``d2l/dr2 = 2 * (-4 e**2/expm1(2)**2)``. A leak of the infinite
    reference derivative would give ``0 * inf = NaN``.
    """
    data = flag_data(["left", "interval"], [2.0, 1.0], [NAN, 3.0])

    def loss(rate):
        return jnp.sum(_steep_reference()(rate=rate).log_likelihood(data))

    rate = jnp.asarray(f(1.0))
    e = np.expm1(2.0)
    assert_close(jax.grad(loss)(rate), -1.0 + 4.0 / e)
    assert_close(jax.jacfwd(loss)(rate), -1.0 + 4.0 / e)
    assert_close(jax.hessian(loss)(rate), -8.0 * np.exp(2.0) / e**2)
    assert_close(jax.jit(jax.grad(loss))(rate), -1.0 + 4.0 / e)
