"""Feature `family.loglogistic`: parameter derivatives of direct methods and the likelihood (spec 4.1, 7.1-7.3, 12.1; rev 55).

7.3: first and second parameter derivatives on finite, valid, smooth interior
cases, with `jit` and `vmap`, forward and reverse modes, Hessians; isolation
of rows without parameter issues to second order. 12.1: `check_grads(order=2,
modes=("fwd","rev"))` on interior cases. 4.1: `from_log` gradients equal the
ordinary constructor's (Q-069).

Hand first derivatives (``L = log(t/alpha)``, ``u = beta L``, ``s = sigmoid(u)
= w/(1+w)``): right ``log S = -softplus(u)``: ``d/dbeta = -L s``,
``d/dalpha = beta s/alpha``; exact ``log f``: ``d/dbeta = 1/beta + L - 2 L s``,
``d/dalpha = -(beta/alpha)(1 - 2 s)``. References are ``mpmath.diff`` at 50
digits (``observation_reference``, ``family_reference``); interior cases have
derivatives of order ``1e-2`` to ``10``, so the 12.1 tolerances apply.
``check_grads`` runs in float64 only.
"""

import family_reference as fr
import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as oref
import pytest
from helpers import INF, NAN, X64, assert_close, b, f, flag_data, loglogistic_from_log
from jax.test_util import check_grads

import survjax

LOGLOGISTIC = pytest.mark.feature("family.loglogistic")
WITH_CENSORING = pytest.mark.feature("family.loglogistic", "observation.censoring")
WITH_TRUNCATION = pytest.mark.feature("family.loglogistic", "observation.truncation")

requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")

BETA, ALPHA = 1.6, 2.0
TIME = [0.3, 1.5, 4.0, 0.8, 2.5]
EVENT = [True, True, False, False, True]
MODES = {"rev": jax.grad, "fwd": jax.jacfwd}
HESSIANS = {
    "rev-over-rev": lambda fn: jax.jacrev(jax.jacrev(fn)),
    "fwd-over-rev": lambda fn: jax.jacfwd(jax.jacrev(fn)),
    "fwd-over-fwd": lambda fn: jax.jacfwd(jax.jacfwd(fn)),
    "rev-over-fwd": lambda fn: jax.jacrev(jax.jacfwd(fn)),
}


def _data():
    return survjax.SurvivalData(f(TIME), b(EVENT))


def _total(x, data=None):
    """Summed likelihood in ``x = (shape, scale)``."""
    data = _data() if data is None else data
    return jnp.sum(survjax.LogLogistic(shape=x[0], scale=x[1]).log_likelihood(data))


def _orders(i, j):
    orders = [0, 0]
    orders[i] += 1
    orders[j] += 1
    return orders


def _reference_gradient(beta=BETA):
    return sum(
        oref.gradient("loglogistic", (beta, ALPHA), t, None if e else INF, exact=e) for t, e in zip(TIME, EVENT, strict=True)
    )


def _reference_hessian(beta=BETA):
    return sum(
        oref.hessian("loglogistic", (beta, ALPHA), t, None if e else INF, exact=e) for t, e in zip(TIME, EVENT, strict=True)
    )


# =============================================================== direct methods


@LOGLOGISTIC
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("method", fr.LOG_METHODS)
@pytest.mark.parametrize("t", [0.2, 1.5, 6.0])
def test_direct_method_gradients_match_mpmath(t, method, mode):
    """spec 4.1/7.3, rev 55: `d/dshape` and `d/dscale` of `log S`, `log F`, `log f` and `log h` at interior times, both modes."""
    grads = MODES[mode](lambda k, a: getattr(survjax.LogLogistic(shape=k, scale=a), method)(f(t)), argnums=(0, 1))(
        f(BETA), f(ALPHA)
    )
    for i, g in enumerate(grads):
        orders = [0, 0]
        orders[i] = 1
        assert_close(g, fr.direct_derivative("loglogistic", (BETA, ALPHA), t, method, orders), err_msg=f"d{method}/dθ{i}")


@LOGLOGISTIC
@pytest.mark.parametrize("method", ["log_survival", "log_prob"])
def test_direct_method_hessians_match_mpmath(method):
    """spec 7.3, rev 55: the `(shape, scale)` Hessian of `log S` and `log f` at `t = 2.5`, reverse over forward."""
    got = np.asarray(
        jax.jacrev(jax.jacfwd(lambda x: getattr(survjax.LogLogistic(shape=x[0], scale=x[1]), method)(f(2.5))))(f([BETA, ALPHA]))
    )
    want = [[fr.direct_derivative("loglogistic", (BETA, ALPHA), 2.5, method, _orders(i, j)) for j in range(2)] for i in range(2)]
    assert_close(got, want)


@requires_x64
@LOGLOGISTIC
@pytest.mark.parametrize("method", fr.DIRECT_METHODS)
def test_check_grads_order_two_direct_methods(method):
    """spec 12.1, rev 55: `check_grads(order=2, modes=("fwd", "rev"))` on interior direct-method cases (`beta` below and above 1)."""
    check_grads(
        lambda k, a: getattr(survjax.LogLogistic(shape=k, scale=a), method)(f([0.4, 2.0, 7.0])),
        (f([0.6, 1.0, 2.5]), f([0.5, 1.5, 3.0])),
        order=2,
        modes=("fwd", "rev"),
    )


# =============================================================== likelihood


@LOGLOGISTIC
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("beta", [0.7, BETA])
def test_likelihood_gradient_matches_mpmath(beta, mode):
    """spec 7.3, rev 55: gradient of the summed exact/right likelihood in `(shape, scale)`, both modes, `beta` below and above 1."""
    assert_close(MODES[mode](_total)(f([beta, ALPHA])), _reference_gradient(beta))


@LOGLOGISTIC
@pytest.mark.parametrize("mode", list(HESSIANS))
def test_likelihood_hessian_matches_mpmath(mode):
    """spec 7.3, rev 55: the full `(shape, scale)` Hessian of the summed likelihood, all four forward/reverse compositions."""
    assert_close(HESSIANS[mode](_total)(f([BETA, ALPHA])), _reference_hessian())


@LOGLOGISTIC
def test_per_row_parameters_give_a_diagonal_hessian():
    """spec 5/7.3, rev 55: per-row `scale` with a shared `shape`: `d2/dalpha_i dalpha_j` is diagonal, each entry the row's own mpmath value."""
    alphas = f([0.5, 1.0, 2.0, 3.0, 5.0])
    hess = np.asarray(
        jax.hessian(lambda a: jnp.sum(survjax.LogLogistic(shape=f(BETA), scale=a).log_likelihood(_data())))(alphas)
    )
    want = np.diag(
        [
            oref.hessian("loglogistic", (BETA, float(a)), t, None if e else INF, exact=e)[1, 1]
            for a, t, e in zip(alphas, TIME, EVENT, strict=True)
        ]
    )
    assert_close(hess, want)


@LOGLOGISTIC
def test_likelihood_gradient_agrees_under_jit_and_vmap():
    """spec 7.3/12.1, rev 55: gradients agree eagerly, under `jit` and under `vmap` over parameter draws; `jit` Hessian."""
    xs = f([[0.7, 1.0], [BETA, ALPHA], [3.0, 0.8]])
    want = np.stack([np.asarray(jax.grad(_total)(x)) for x in xs])
    assert_close(jax.vmap(jax.grad(_total))(xs), want)
    assert_close(jax.jit(jax.vmap(jax.grad(_total)))(xs), want)
    assert_close(jax.jit(jax.hessian(_total))(xs[1]), jax.hessian(_total)(xs[1]))


@requires_x64
@LOGLOGISTIC
def test_check_grads_order_two_likelihood():
    """spec 12.1, rev 55: `check_grads(order=2, modes=("fwd", "rev"))` on the summed likelihood, shared and per-row parameters, and under `jit`."""
    check_grads(_total, (f([BETA, ALPHA]),), order=2, modes=("fwd", "rev"))
    check_grads(
        lambda k, a: jnp.sum(survjax.LogLogistic(shape=k, scale=a).log_likelihood(_data())),
        (f([0.6, 0.9, 1.3, 2.0, 3.0]), f([0.5, 1.0, 2.0, 3.0, 5.0])),
        order=2,
        modes=("fwd", "rev"),
    )
    check_grads(jax.jit(_total), (f([BETA, ALPHA]),), order=2, modes=("fwd", "rev"))


@LOGLOGISTIC
@pytest.mark.parametrize("mode", list(MODES))
def test_from_log_gradient_follows_the_chain_rule(mode):
    """spec 4.1/7.3, rev 55: in `(log_shape, log_scale)`, each derivative is the natural one times the parameter (Q-069)."""

    def via_log(y):
        return jnp.sum(loglogistic_from_log(y[0], y[1]).log_likelihood(_data()))

    g = np.asarray(MODES[mode](via_log)(f([np.log(BETA), np.log(ALPHA)])))
    want = _reference_gradient()
    assert_close(g, [float(f(BETA)) * want[0], float(f(ALPHA)) * want[1]])


# =============================================================== isolation of rows without parameter issues


@LOGLOGISTIC
@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF], ids=["zero", "negative", "nan", "posinf"])
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_rows_without_parameter_issues_keep_their_derivatives(mode, bad):
    """spec 7.1/7.3, rev 55: with per-row parameters, a row whose `scale` violates `positive` leaves the other rows' first and second `shape` derivatives equal to those computed alone."""
    scale_full = jnp.asarray(f([ALPHA, bad, 1.2]))
    data_full = survjax.SurvivalData(f([1.5, 2.0, 2.5]), b([True, True, False]))
    data_alone = survjax.SurvivalData(f([1.5, 2.5]), b([True, False]))

    def full(k):
        out = survjax.LogLogistic(shape=k, scale=scale_full).log_likelihood(data_full)
        return out[0] + out[2]

    def alone(k):
        return jnp.sum(survjax.LogLogistic(shape=k[jnp.asarray([0, 2])], scale=f([ALPHA, 1.2])).log_likelihood(data_alone))

    ks = jnp.asarray(f([BETA, BETA, 0.8]))
    first = MODES[mode]
    second = HESSIANS["fwd-over-rev" if mode == "rev" else "fwd-over-fwd"]
    g_full = np.asarray(first(full)(ks))
    g_alone = np.asarray(first(alone)(ks))
    assert np.isfinite(g_full[[0, 2]]).all()
    assert_close(g_full[[0, 2]], g_alone[[0, 2]])
    h_full = np.asarray(second(full)(ks))
    h_alone = np.asarray(second(alone)(ks))
    assert_close(h_full[np.ix_([0, 2], [0, 2])], h_alone[np.ix_([0, 2], [0, 2])])


# =============================================================== Slice B rows


@WITH_CENSORING
@pytest.mark.parametrize("mode", list(MODES))
def test_left_and_interval_gradients_match_mpmath(mode):
    """spec 7.3, rev 55: gradients of left and interval rows in `(shape, scale)` against mpmath, both modes."""
    data = flag_data(["left", "interval"], [0.9, 0.5], [NAN, 3.0])
    got = np.asarray(MODES[mode](lambda x: _total(x, data))(f([BETA, ALPHA])))
    want = oref.gradient("loglogistic", (BETA, ALPHA), 0.0, 0.9) + oref.gradient("loglogistic", (BETA, ALPHA), 0.5, 3.0)
    assert_close(got, want)


@WITH_CENSORING
def test_left_and_interval_hessian_matches_mpmath():
    """spec 7.3, rev 55: the Hessian of left and interval rows against mpmath."""
    data = flag_data(["left", "interval"], [0.9, 0.5], [NAN, 3.0])
    got = jax.hessian(lambda x: _total(x, data))(f([BETA, ALPHA]))
    want = oref.hessian("loglogistic", (BETA, ALPHA), 0.0, 0.9) + oref.hessian("loglogistic", (BETA, ALPHA), 0.5, 3.0)
    assert_close(got, want)


@WITH_TRUNCATION
@pytest.mark.parametrize("mode", list(MODES))
def test_truncated_row_gradients_include_log_z(mode):
    """spec 2/7.3, rev 55: gradients of truncated exact and right rows, including `-log Z`, against mpmath."""
    data = survjax.SurvivalData(f([1.5, 2.5]), b([True, False]), left_trunc=f(0.5), right_trunc=f(6.0))
    got = np.asarray(MODES[mode](lambda x: _total(x, data))(f([BETA, ALPHA])))
    want = oref.gradient("loglogistic", (BETA, ALPHA), 1.5, exact=True, left=0.5, right=6.0) + oref.gradient(
        "loglogistic", (BETA, ALPHA), 2.5, INF, left=0.5, right=6.0
    )
    assert_close(got, want)
