"""Feature `family.lognormal`: parameter derivatives of direct methods and the likelihood (spec 4.1, 7.1-7.3, 12.1; rev 55).

7.3: "Built-in likelihoods require first and second parameter derivatives on
finite, valid, smooth interior cases, composed with `jit` and `vmap`, in
forward and reverse modes, including Hessians"; isolation of rows without
parameter issues "to second order, in forward and reverse modes". 12.1
(Transformations): eager/`jit`/`vmap` agreement; `check_grads(order=2,
modes=("fwd","rev"))` on interior cases. 4.1: `from_log` "parameter gradients
equal those of the ordinary constructor wherever `exp` is finite" (Q-069).

Hand first derivatives (``z = (log t - mu)/sigma``, inverse Mills ratio
``lam = phi(z)/Phi(-z)``): exact ``log f``: ``d/dmu = z/sigma``, ``d/dsigma =
(z**2 - 1)/sigma``; right ``log S = log Phi(-z)``: ``d/dmu = lam/sigma``,
``d/dsigma = lam z/sigma``. References are ``mpmath.diff`` at 50 digits of the
same expressions (``observation_reference``, ``family_reference``); the cases
are interior with derivatives of order ``1e-2`` to ``10``, so the 12.1
tolerances apply (as at A9). ``check_grads`` uses second-order finite
differences and runs in float64 only.
"""

import family_reference as fr
import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as oref
import pytest
from helpers import NAN, INF, X64, assert_close, b, f, flag_data, lognormal_from_log
from jax.test_util import check_grads

import survjax

LOGNORMAL = pytest.mark.feature("family.lognormal")
WITH_CENSORING = pytest.mark.feature("family.lognormal", "observation.censoring")
WITH_TRUNCATION = pytest.mark.feature("family.lognormal", "observation.truncation")

requires_x64 = pytest.mark.skipif(not X64, reason="second-order finite differences need float64")

MU, SIGMA = 0.4, 0.9
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
    """Summed likelihood in ``x = (meanlog, sdlog)``."""
    data = _data() if data is None else data
    return jnp.sum(survjax.LogNormal(meanlog=x[0], sdlog=x[1]).log_likelihood(data))


def _reference_gradient():
    return sum(
        oref.gradient("lognormal", (MU, SIGMA), t, None if e else INF, exact=e) for t, e in zip(TIME, EVENT, strict=True)
    )


def _orders(i, j):
    """Derivative orders in ``(meanlog, sdlog)`` for the Hessian entry ``[i, j]``."""
    orders = [0, 0]
    orders[i] += 1
    orders[j] += 1
    return orders


def _reference_hessian():
    return sum(
        oref.hessian("lognormal", (MU, SIGMA), t, None if e else INF, exact=e) for t, e in zip(TIME, EVENT, strict=True)
    )


# =============================================================== direct methods


@LOGNORMAL
@pytest.mark.parametrize("mode", list(MODES))
@pytest.mark.parametrize("method", fr.LOG_METHODS)
@pytest.mark.parametrize("t", [0.2, 1.5, 6.0])
def test_direct_method_gradients_match_mpmath(t, method, mode):
    """spec 4.1/7.3, rev 55: `d/dmeanlog` and `d/dsdlog` of `log S`, `log F`, `log f` and `log h` at interior times, both modes."""
    grads = MODES[mode](lambda m, s: getattr(survjax.LogNormal(meanlog=m, sdlog=s), method)(f(t)), argnums=(0, 1))(
        f(MU), f(SIGMA)
    )
    for i, g in enumerate(grads):
        orders = [0, 0]
        orders[i] = 1
        assert_close(g, fr.direct_derivative("lognormal", (MU, SIGMA), t, method, orders), err_msg=f"d{method}/dθ{i}")


@LOGNORMAL
@pytest.mark.parametrize("method", ["log_survival", "log_prob"])
def test_direct_method_hessians_match_mpmath(method):
    """spec 7.3, rev 55: the `(meanlog, sdlog)` Hessian of `log S` and `log f` at `t = 2.5`, reverse over forward."""
    got = np.asarray(
        jax.jacrev(jax.jacfwd(lambda x: getattr(survjax.LogNormal(meanlog=x[0], sdlog=x[1]), method)(f(2.5))))(f([MU, SIGMA]))
    )
    want = [[fr.direct_derivative("lognormal", (MU, SIGMA), 2.5, method, _orders(i, j)) for j in range(2)] for i in range(2)]
    assert_close(got, want)


@requires_x64
@LOGNORMAL
@pytest.mark.parametrize("method", fr.DIRECT_METHODS)
def test_check_grads_order_two_direct_methods(method):
    """spec 12.1, rev 55: `check_grads(order=2, modes=("fwd", "rev"))` on interior direct-method cases."""
    check_grads(
        lambda m, s: getattr(survjax.LogNormal(meanlog=m, sdlog=s), method)(f([0.4, 2.0, 7.0])),
        (f([0.0, 0.5, 1.0]), f([0.6, 1.0, 1.4])),
        order=2,
        modes=("fwd", "rev"),
    )


# =============================================================== likelihood


@LOGNORMAL
@pytest.mark.parametrize("mode", list(MODES))
def test_likelihood_gradient_matches_mpmath(mode):
    """spec 7.3, rev 55: gradient of the summed exact/right likelihood in `(meanlog, sdlog)`, both modes."""
    assert_close(MODES[mode](_total)(f([MU, SIGMA])), _reference_gradient())


@LOGNORMAL
@pytest.mark.parametrize("mode", list(HESSIANS))
def test_likelihood_hessian_matches_mpmath(mode):
    """spec 7.3, rev 55: the full `(meanlog, sdlog)` Hessian of the summed likelihood, all four forward/reverse compositions."""
    assert_close(HESSIANS[mode](_total)(f([MU, SIGMA])), _reference_hessian())


@LOGNORMAL
def test_per_row_parameters_give_a_block_diagonal_hessian():
    """spec 5/7.3, rev 55: per-row `meanlog` with a shared `sdlog`: `d2/dmu_i dmu_j` is diagonal, each entry the row's own mpmath value."""
    mus = f([0.0, 0.3, 0.6, 0.9, 1.2])
    hess = np.asarray(
        jax.hessian(lambda m: jnp.sum(survjax.LogNormal(meanlog=m, sdlog=f(SIGMA)).log_likelihood(_data())))(mus)
    )
    want = np.diag(
        [
            oref.hessian("lognormal", (float(m), SIGMA), t, None if e else INF, exact=e)[0, 0]
            for m, t, e in zip(mus, TIME, EVENT, strict=True)
        ]
    )
    assert_close(hess, want)


@LOGNORMAL
def test_likelihood_gradient_agrees_under_jit_and_vmap():
    """spec 7.3/12.1, rev 55: gradients agree eagerly, under `jit` and under `vmap` over parameter draws."""
    xs = f([[0.0, 0.6], [0.4, 0.9], [1.0, 1.5]])
    want = np.stack([np.asarray(jax.grad(_total)(x)) for x in xs])
    assert_close(jax.vmap(jax.grad(_total))(xs), want)
    assert_close(jax.jit(jax.vmap(jax.grad(_total)))(xs), want)
    assert_close(jax.jit(jax.hessian(_total))(xs[1]), jax.hessian(_total)(xs[1]))


@requires_x64
@LOGNORMAL
def test_check_grads_order_two_likelihood():
    """spec 12.1, rev 55: `check_grads(order=2, modes=("fwd", "rev"))` on the summed likelihood, shared and per-row parameters, and under `jit`."""
    check_grads(_total, (f([MU, SIGMA]),), order=2, modes=("fwd", "rev"))
    check_grads(
        lambda m, s: jnp.sum(survjax.LogNormal(meanlog=m, sdlog=s).log_likelihood(_data())),
        (f([0.0, 0.3, 0.6, 0.9, 1.2]), f([0.5, 0.7, 0.9, 1.1, 1.3])),
        order=2,
        modes=("fwd", "rev"),
    )
    check_grads(jax.jit(_total), (f([MU, SIGMA]),), order=2, modes=("fwd", "rev"))


@LOGNORMAL
@pytest.mark.parametrize("mode", list(MODES))
def test_from_log_gradient_follows_the_chain_rule(mode):
    """spec 4.1/7.3, rev 55: in `(meanlog, log_sdlog)`, `d/dlog_sdlog = sdlog d/dsdlog` (Q-069)."""

    def via_log(y):
        return jnp.sum(lognormal_from_log(y[0], y[1]).log_likelihood(_data()))

    g = np.asarray(MODES[mode](via_log)(f([MU, np.log(SIGMA)])))
    want = _reference_gradient()
    assert_close(g, [want[0], float(f(SIGMA)) * want[1]])


# =============================================================== isolation of rows without parameter issues


@LOGNORMAL
@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF], ids=["zero", "negative", "nan", "posinf"])
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_rows_without_parameter_issues_keep_their_derivatives(mode, bad):
    """spec 7.1/7.3, rev 55: with per-row parameters, a row whose `sdlog` violates `positive` leaves the other rows' first and second derivatives equal to those computed alone."""
    sig_full = jnp.asarray(f([SIGMA, bad, 1.2]))
    data_full = survjax.SurvivalData(f([1.5, 2.0, 2.5]), b([True, True, False]))
    data_alone = survjax.SurvivalData(f([1.5, 2.5]), b([True, False]))

    def full(m):
        out = survjax.LogNormal(meanlog=m, sdlog=sig_full).log_likelihood(data_full)
        return out[0] + out[2]

    def alone(m):
        return jnp.sum(survjax.LogNormal(meanlog=m[jnp.asarray([0, 2])], sdlog=f([SIGMA, 1.2])).log_likelihood(data_alone))

    mus = jnp.asarray(f([MU, MU, 0.1]))
    first = MODES[mode]
    second = HESSIANS["fwd-over-rev" if mode == "rev" else "fwd-over-fwd"]
    g_full = np.asarray(first(full)(mus))
    g_alone = np.asarray(first(alone)(mus))
    assert np.isfinite(g_full[[0, 2]]).all()
    assert_close(g_full[[0, 2]], g_alone[[0, 2]])
    h_full = np.asarray(second(full)(mus))
    h_alone = np.asarray(second(alone)(mus))
    assert_close(h_full[np.ix_([0, 2], [0, 2])], h_alone[np.ix_([0, 2], [0, 2])])


# =============================================================== Slice B rows


@WITH_CENSORING
@pytest.mark.parametrize("mode", list(MODES))
def test_left_and_interval_gradients_match_mpmath(mode):
    """spec 7.3, rev 55: gradients of left and interval rows in `(meanlog, sdlog)` against mpmath, both modes."""
    data = flag_data(["left", "interval"], [0.9, 0.5], [NAN, 3.0])
    got = np.asarray(MODES[mode](lambda x: _total(x, data))(f([MU, SIGMA])))
    want = oref.gradient("lognormal", (MU, SIGMA), 0.0, 0.9) + oref.gradient("lognormal", (MU, SIGMA), 0.5, 3.0)
    assert_close(got, want)


@WITH_CENSORING
def test_left_and_interval_hessian_matches_mpmath():
    """spec 7.3, rev 55: the Hessian of left and interval rows against mpmath."""
    data = flag_data(["left", "interval"], [0.9, 0.5], [NAN, 3.0])
    got = jax.hessian(lambda x: _total(x, data))(f([MU, SIGMA]))
    want = oref.hessian("lognormal", (MU, SIGMA), 0.0, 0.9) + oref.hessian("lognormal", (MU, SIGMA), 0.5, 3.0)
    assert_close(got, want)


@WITH_TRUNCATION
@pytest.mark.parametrize("mode", list(MODES))
def test_truncated_row_gradients_include_log_z(mode):
    """spec 2/7.3, rev 55: gradients of truncated exact and right rows, including `-log Z`, against mpmath."""
    data = survjax.SurvivalData(f([1.5, 2.5]), b([True, False]), left_trunc=f(0.5), right_trunc=f(6.0))
    got = np.asarray(MODES[mode](lambda x: _total(x, data))(f([MU, SIGMA])))
    want = oref.gradient("lognormal", (MU, SIGMA), 1.5, exact=True, left=0.5, right=6.0) + oref.gradient(
        "lognormal", (MU, SIGMA), 2.5, INF, left=0.5, right=6.0
    )
    assert_close(got, want)
