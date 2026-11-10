"""Feature `family.loglogistic`: the likelihood on `LogLogistic` (spec 2, 4.4, 5, 6.2, 7.1-7.3, 12.1, 12.2; rev 55).

As for `family.lognormal` (12.2, rev 55): exact and right rows without
truncation always; left/interval rows with `observation.censoring`;
truncation with `observation.truncation`; until then those rows are NaN (7.1),
tested with `until(...)`. 7.1: issue rows are NaN and triaged; 4.4: PH over a
log-logistic baseline (no closure) keeps the PH formulas.

Hand values (spec 2, ``u = beta log(t/alpha)``): an exact row at ``t =
alpha`` is ``log beta - log alpha - 2 log 2``; a right row there is ``-log
2``. References: ``observation_reference`` (log-logistic branch) and
``family_reference.direct`` for tail rows.
"""

import math

import family_reference as fr
import jax
import numpy as np
import observation_reference as oref
import pytest
from helpers import (
    INF,
    NAN,
    assert_close,
    assert_tail_close,
    b,
    f,
    flag_data,
    ll,
    nan_causes,
    ph,
    unsupported_nan,
)

import survjax

LOGLOGISTIC = pytest.mark.feature("family.loglogistic")
WITH_CENSORING = pytest.mark.feature("family.loglogistic", "observation.censoring")
WITH_TRUNCATION = pytest.mark.feature("family.loglogistic", "observation.truncation")
WITH_BOTH = pytest.mark.feature("family.loglogistic", "observation.censoring", "observation.truncation")

BETA, ALPHA = 1.6, 2.0
TIME = [0.3, 1.5, 4.0, 0.8, 2.5, 12.0]
EVENT = [True, True, True, False, False, False]


def _dist(beta=BETA, alpha=ALPHA):
    return survjax.LogLogistic(shape=f(beta), scale=f(alpha))


def _exact_right_reference(beta=BETA, alpha=ALPHA, time=TIME, event=EVENT):
    return [
        oref.contribution("loglogistic", (beta, alpha), t, None if e else INF, exact=e)
        for t, e in zip(time, event, strict=True)
    ]


# =============================================================== exact and right rows


@LOGLOGISTIC
@pytest.mark.parametrize("beta", [0.5, 1.0, BETA, 5.0])
def test_exact_and_right_rows_match_mpmath(beta):
    """spec 2/7.1, rev 55: exact rows `log f(t)` and right rows `log S(c)`, for `beta` below, at and above 1."""
    out = ll(_dist(beta), survjax.SurvivalData(f(TIME), b(EVENT)))
    assert_close(out, _exact_right_reference(beta=beta))


@LOGLOGISTIC
def test_rows_at_the_scale_are_hand_derived():
    """spec 2, rev 55: at `t = alpha`, an exact row is `log beta - log alpha - 2 log 2` and a right row `-log 2`."""
    t = f(ALPHA)
    out = ll(_dist(), survjax.SurvivalData(np.stack([t, t]), b([True, False])))
    assert_close(out, [math.log(float(f(BETA))) - math.log(ALPHA) - 2 * math.log(2), -math.log(2)])


@LOGLOGISTIC
def test_right_censored_at_zero_is_exactly_zero():
    """spec 2/4.5/7.1, rev 55: `c = 0`: `log S(0) = 0` from the base, exactly."""
    out = ll(_dist(), survjax.SurvivalData(f([0.0, 1.0]), b([False, True])))
    assert np.asarray(out)[0] == 0.0


@LOGLOGISTIC
def test_raw_form_equals_survival_data_form():
    """spec 7.1, rev 55: `dist.log_likelihood(time=..., event=...)` equals the `SurvivalData` form."""
    d = _dist()
    assert_close(ll(d, time=f(TIME), event=b(EVENT)), ll(d, survjax.SurvivalData(f(TIME), b(EVENT))))


@LOGLOGISTIC
def test_draw_column_against_observations():
    """spec 5, rev 55: data `(N,)` against a distribution `(D, 1)` gives `(D, N)`."""
    betas = [0.8, 2.4]
    out = ll(survjax.LogLogistic(shape=f(betas)[:, None], scale=f(ALPHA)), survjax.SurvivalData(f(TIME), b(EVENT)))
    assert out.shape == (2, len(TIME))
    for i, beta in enumerate(betas):
        assert_close(out[i], _exact_right_reference(beta=beta))


@LOGLOGISTIC
def test_per_row_parameters_pair_with_rows():
    """spec 5, rev 55: data `(N,)` against a distribution `(N,)` pairs rows."""
    alphas = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    out = ll(survjax.LogLogistic(shape=f(BETA), scale=f(alphas)), survjax.SurvivalData(f(TIME), b(EVENT)))
    want = [_exact_right_reference(alpha=a, time=[t], event=[e])[0] for a, t, e in zip(alphas, TIME, EVENT, strict=True)]
    assert_close(out, want)


@LOGLOGISTIC
@pytest.mark.parametrize("u", [-60.0, -20.0, 20.0, 60.0])
def test_rows_in_both_tails(u):
    """spec 4.1/7.1/12.1, rev 55: exact and right rows far in both tails (`log S = -softplus(u)` tiny for `u ≪ 0`, about `-u` for `u ≫ 0`), against mpmath with zero absolute tolerance."""
    t = fr.loglogistic_time(BETA, ALPHA, u)
    out = ll(_dist(), survjax.SurvivalData(f([t, t]), b([True, False])))
    want_f, rtol_f = fr.direct("loglogistic", (BETA, ALPHA), t, "log_prob")
    want_s, rtol_s = fr.direct("loglogistic", (BETA, ALPHA), t, "log_survival")
    assert_tail_close(out[0], want_f, rtol_f)
    assert_tail_close(out[1], want_s, rtol_s)


# =============================================================== diagnostics


@LOGLOGISTIC
def test_invalid_data_rows_are_nan_and_attributed_to_data():
    """spec 6.2/7.1, rev 55: invalid exact times (negative, NaN) give NaN rows attributed to data; valid rows keep their values."""
    data = survjax.SurvivalData(f([1.5, -1.0, NAN, 2.5]), b([True, True, True, False]))
    out = ll(_dist(), data)
    np.testing.assert_array_equal(np.isnan(out), [False, True, True, False])
    bad_data, bad_params, other = nan_causes(_dist(), data, out)
    np.testing.assert_array_equal(bad_data, [False, True, True, False])
    assert not bad_params.any() and not other.any()
    assert_close(out[[0, 3]], _exact_right_reference(time=[1.5, 2.5], event=[True, False]))


@LOGLOGISTIC
@pytest.mark.parametrize(
    "beta, alpha",
    [(0.0, ALPHA), (-1.0, ALPHA), (NAN, ALPHA), (BETA, 0.0), (BETA, INF)],
    ids=["shape-zero", "shape-negative", "shape-nan", "scale-zero", "scale-inf"],
)
def test_parameter_issue_rows_are_nan_and_attributed_to_parameters(beta, alpha):
    """spec 4.7/7.1, rev 55: rows marked by `parameter_issues().invalid_entries` are NaN; the others keep their values."""
    d = survjax.LogLogistic(shape=f([BETA, beta, BETA]), scale=f([ALPHA, alpha, ALPHA]))
    data = survjax.SurvivalData(f([1.5, 1.5, 2.5]), b([True, False, False]))
    out = ll(d, data)
    np.testing.assert_array_equal(np.isnan(out), [False, True, False])
    _, bad_params, other = nan_causes(d, data, out)
    np.testing.assert_array_equal(bad_params, [False, True, False])
    assert not other.any()
    assert_close(out[[0, 2]], _exact_right_reference(time=[1.5, 2.5], event=[True, False]))


@LOGLOGISTIC
def test_valid_batch_has_no_nan_and_no_triage_mask():
    """spec 7.1, rev 55: valid data and parameters give finite rows and empty triage masks, also for `beta < 1`."""
    for beta in (0.5, BETA):
        data = survjax.SurvivalData(f(TIME), b(EVENT))
        out = ll(_dist(beta), data)
        assert np.isfinite(out).all()
        for mask in nan_causes(_dist(beta), data, out):
            assert not mask.any()


# =============================================================== transformations


@LOGLOGISTIC
def test_likelihood_agrees_under_jit():
    """spec 7.3/12.1, rev 55: eager and `jit` agree, with prepared data and with traced construction."""
    d, data = _dist(), survjax.SurvivalData(f(TIME), b(EVENT))
    want = ll(d, data)
    assert_close(jax.jit(lambda dd, da: dd.log_likelihood(da))(d, data), want)
    assert_close(jax.jit(lambda t, e: _dist().log_likelihood(survjax.SurvivalData(t, e)))(f(TIME), b(EVENT)), want)


@LOGLOGISTIC
def test_likelihood_under_vmap_over_parameter_draws():
    """spec 5/7.3/12.1, rev 55: `vmap` over draws of `(shape, scale)` equals the `(D, 1)` batch."""
    betas, alphas = f([0.6, 1.6, 3.0]), f([1.0, 2.0, 0.7])
    data = survjax.SurvivalData(f(TIME), b(EVENT))
    mapped = jax.vmap(lambda k, a: survjax.LogLogistic(shape=k, scale=a).log_likelihood(data))(betas, alphas)
    assert_close(mapped, ll(survjax.LogLogistic(shape=betas[:, None], scale=alphas[:, None]), data))


@LOGLOGISTIC
@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_likelihood_under_debug_nans(beta):
    """spec 7.2/12.1, rev 55: a valid mixed batch (tiny, interior, upper-tail and zero times) raises no NaN eagerly under `jax_debug_nans`, in values or the gradient."""
    tail = fr.loglogistic_time(beta, ALPHA, 15.0)
    data = survjax.SurvivalData(f([1.5, 1e-30, tail, 0.0, 1e-15, 2.0, tail]), b([True, True, True, False, False, False, False]))

    def total(k, a):
        out = survjax.LogLogistic(shape=k, scale=a).log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, argnums=(0, 1), has_aux=True)(f(beta), f(ALPHA))
    values = np.asarray(values)
    assert np.isfinite(values).all()
    assert values[3] == 0.0
    assert np.isfinite(np.asarray(grads)).all()


@LOGLOGISTIC
def test_ph_wrapper_of_loglogistic_keeps_the_ph_formulas():
    """spec 4.4, rev 55: PH over `LogLogistic` (no closure) keeps the PH formulas: exact `log h0 + eta + exp(eta) log S0`, right `exp(eta) log S0`."""
    eta = -0.4
    out = ll(ph(_dist(), f(eta)), survjax.SurvivalData(f([1.5, 2.5]), b([True, False])))
    (e,) = oref.rounded(eta)
    log_h0, _ = fr.direct("loglogistic", (BETA, ALPHA), 1.5, "log_hazard")
    log_s0_exact, _ = fr.direct("loglogistic", (BETA, ALPHA), 1.5, "log_survival")
    log_s0_right, _ = fr.direct("loglogistic", (BETA, ALPHA), 2.5, "log_survival")
    assert_close(out, [log_h0 + e + math.exp(e) * log_s0_exact, math.exp(e) * log_s0_right])


# =============================================================== unsupported modes until Slice B


@pytest.mark.until("observation.censoring")
@LOGLOGISTIC
def test_left_and_interval_rows_are_nan_until_censoring_is_enabled():
    """spec 7.1/12.2, rev 55: left and interval rows are NaN while `observation.censoring` is not enabled; exact and right rows keep their values; the recipe reports them as unsupported."""
    data = flag_data(["exact", "left", "interval", "right"], [1.5, 2.0, 0.5, 2.5], [NAN, NAN, 3.0, NAN])
    out = ll(_dist(), data)
    np.testing.assert_array_equal(np.isnan(out), [False, True, True, False])
    np.testing.assert_array_equal(unsupported_nan(_dist(), data, out), [False, True, True, False])
    assert_close(out[[0, 3]], _exact_right_reference(time=[1.5, 2.5], event=[True, False]))


@pytest.mark.until("observation.truncation")
@LOGLOGISTIC
def test_truncated_rows_are_nan_until_truncation_is_enabled():
    """spec 7.1/12.2, rev 55: truncated rows are NaN while `observation.truncation` is not enabled; untruncated rows keep their values."""
    data = survjax.SurvivalData(f([1.5, 1.5, 2.5]), b([True, True, False]), right_trunc=f([INF, 6.0, INF]))
    out = ll(_dist(), data)
    np.testing.assert_array_equal(np.isnan(out), [False, True, False])
    np.testing.assert_array_equal(unsupported_nan(_dist(), data, out), [False, True, False])


# =============================================================== left and interval rows (Slice B)


@WITH_CENSORING
def test_left_and_interval_rows_match_mpmath():
    """spec 2/7.1, rev 55: left `log F(c)` and interval `log(S(a) - S(b))` beside exact and right rows."""
    kinds = ["exact", "left", "interval", "right", "left", "interval"]
    time, time2 = [1.5, 0.6, 0.5, 2.5, 7.0, 3.0], [NAN, NAN, 3.0, NAN, NAN, 3.2]
    out = ll(_dist(), flag_data(kinds, time, time2))
    want = [
        oref.contribution("loglogistic", (BETA, ALPHA), 1.5, exact=True),
        oref.contribution("loglogistic", (BETA, ALPHA), 0.0, 0.6),
        oref.contribution("loglogistic", (BETA, ALPHA), 0.5, 3.0),
        oref.contribution("loglogistic", (BETA, ALPHA), 2.5, INF),
        oref.contribution("loglogistic", (BETA, ALPHA), 0.0, 7.0),
        oref.contribution("loglogistic", (BETA, ALPHA), 3.0, 3.2),
    ]
    assert_close(out, want)


@WITH_CENSORING
@pytest.mark.parametrize("u", [-30.0, 30.0])
def test_left_rows_in_both_tails(u):
    """spec 4.1/7.1/12.1, rev 55: `log F = -softplus(-u)` on left rows: about `u` for `u ≪ 0`, tiny and nonzero for `u ≫ 0`."""
    c = fr.loglogistic_time(BETA, ALPHA, u)
    out = ll(_dist(), flag_data(["left"], [c]))
    want, rtol = fr.direct("loglogistic", (BETA, ALPHA), c, "log_cdf")
    assert out[0] < 0.0
    assert_tail_close(out[0], want, rtol)


@WITH_CENSORING
def test_interval_rows_are_additive():
    """spec 2/12.1, rev 55: `logaddexp` of `(0, a]` and `(a, b]` equals the left row at `b`."""
    out = ll(_dist(), flag_data(["left", "interval", "left"], [0.7, 0.7, 2.2], [NAN, 2.2, NAN]))
    assert_close(np.logaddexp(out[0], out[1]), out[2])


# =============================================================== truncation (Slice B)


@WITH_TRUNCATION
@pytest.mark.parametrize("left, right", [(0.5, INF), (0.0, 6.0), (0.5, 6.0)], ids=["left", "right", "two-sided"])
def test_truncated_exact_and_right_rows_match_mpmath(left, right):
    """spec 2/7.1, rev 55: exact and right rows minus `log Z` under left, right and two-sided truncation."""
    data = survjax.SurvivalData(f([1.5, 2.5]), b([True, False]), left_trunc=f(left), right_trunc=f(right))
    out = ll(_dist(), data)
    want = [
        oref.contribution("loglogistic", (BETA, ALPHA), 1.5, exact=True, left=left, right=right),
        oref.contribution("loglogistic", (BETA, ALPHA), 2.5, INF, left=left, right=right),
    ]
    assert_close(out, want)


@WITH_BOTH
def test_truncated_interval_row_with_partial_overlap():
    """spec 2/7.1, rev 55: `(0.3, 2]` under the window `(0.5, 3]` contributes `log P(0.5<T<=2) - log P(0.5<T<=3)`."""
    out = ll(_dist(), flag_data(["interval"], [0.3], [2.0], left_trunc=[0.5], right_trunc=[3.0]))
    assert_close(out, [oref.contribution("loglogistic", (BETA, ALPHA), 0.3, 2.0, left=0.5, right=3.0)])
