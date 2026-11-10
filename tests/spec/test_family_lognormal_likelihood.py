"""Feature `family.lognormal`: the likelihood on `LogNormal` (spec 2, 3.2, 5, 6.2, 7.1-7.3, 12.1, 12.2; rev 55).

12.2 (rev 55): each brought-forward family is enabled under the full contract
of section 7 for "the likelihood modes implemented at the time", gated by the
12.1 rows that apply to a built-in family. Exact and right-censored rows
without truncation are always implemented; left/interval rows need
`observation.censoring` and truncation needs `observation.truncation` (Slice
B features), so those tests list both features. Until a Slice B feature is
enabled, its rows stay NaN (7.1: "Observation mode not implemented in the
current slice | NaN on that row"), tested with `until(...)`.

Spec 2 contributions (untruncated): exact ``log f(t)``; right ``log S(c)``;
left ``log F(c)``; interval ``log(S(a) - S(b))``; with truncation each minus
``log Z = log P(L < T <= U)``. 7.1: rows in `data.issues.invalid_rows` and
`dist.parameter_issues().invalid_entries` are NaN; the triage recipe
attributes them. 4.4: PH over a baseline without a closure keeps the PH
formulas (`log S = exp(eta) log S0`, `log f = log h0 + eta + exp(eta) log
S0`); "Log-normal and log-logistic baselines have no PH closure".

References: ``observation_reference`` (mpmath, 50 digits, log-normal branch)
for rows; ``family_reference.direct`` for tail rows, with its condition-number
tolerance; hand values where stated.
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

LOGNORMAL = pytest.mark.feature("family.lognormal")
WITH_CENSORING = pytest.mark.feature("family.lognormal", "observation.censoring")
WITH_TRUNCATION = pytest.mark.feature("family.lognormal", "observation.truncation")
WITH_BOTH = pytest.mark.feature("family.lognormal", "observation.censoring", "observation.truncation")

MU, SIGMA = 0.4, 0.9
TIME = [0.3, 1.5, 4.0, 0.8, 2.5, 12.0]
EVENT = [True, True, True, False, False, False]


def _dist(mu=MU, sigma=SIGMA):
    return survjax.LogNormal(meanlog=f(mu), sdlog=f(sigma))


def _exact_right_reference(mu=MU, sigma=SIGMA, time=TIME, event=EVENT):
    return [
        oref.contribution("lognormal", (mu, sigma), t, None if e else INF, exact=e)
        for t, e in zip(time, event, strict=True)
    ]


# =============================================================== exact and right rows


@LOGNORMAL
def test_exact_and_right_rows_match_mpmath():
    """spec 2/7.1, rev 55: exact rows `log f(t)` and right rows `log S(c)` of a mixed batch."""
    out = ll(_dist(), survjax.SurvivalData(f(TIME), b(EVENT)))
    assert_close(out, _exact_right_reference())


@LOGNORMAL
def test_exact_row_at_the_median_time_is_hand_derived():
    """spec 2, rev 55: at `t = exp(mu)`, `log f = -mu - log sigma - log(2π)/2` and a right row is `log S = -log 2`."""
    t = np.exp(f(MU))
    out = ll(_dist(), survjax.SurvivalData(np.stack([t, t]), b([True, False])))
    assert_close(out, [-float(f(MU)) - math.log(float(f(SIGMA))) - 0.5 * math.log(2 * math.pi), -math.log(2)])


@LOGNORMAL
def test_right_censored_at_zero_is_exactly_zero():
    """spec 2/4.5/7.1, rev 55: `c = 0` covers `(0, +inf]`: `log S(0) = 0` from the base, exactly."""
    out = ll(_dist(), survjax.SurvivalData(f([0.0, 1.0]), b([False, True])))
    assert np.asarray(out)[0] == 0.0


@LOGNORMAL
def test_raw_form_equals_survival_data_form():
    """spec 7.1, rev 55: `dist.log_likelihood(time=..., event=...)` equals the `SurvivalData` form."""
    d = _dist()
    assert_close(ll(d, time=f(TIME), event=b(EVENT)), ll(d, survjax.SurvivalData(f(TIME), b(EVENT))))


@LOGNORMAL
def test_draw_column_against_observations():
    """spec 5, rev 55: data `(N,)` against a distribution `(D, 1)` gives `(D, N)`, row `d` from draw `d`."""
    mus = [0.0, 0.8]
    d = survjax.LogNormal(meanlog=f(mus)[:, None], sdlog=f(SIGMA))
    out = ll(d, survjax.SurvivalData(f(TIME), b(EVENT)))
    assert out.shape == (2, len(TIME))
    for i, mu in enumerate(mus):
        assert_close(out[i], _exact_right_reference(mu=mu))


@LOGNORMAL
def test_per_row_parameters_pair_with_rows():
    """spec 5, rev 55: data `(N,)` against a distribution `(N,)` pairs rows."""
    mus = [-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    d = survjax.LogNormal(meanlog=f(mus), sdlog=f(SIGMA))
    out = ll(d, survjax.SurvivalData(f(TIME), b(EVENT)))
    want = [_exact_right_reference(mu=m, time=[t], event=[e])[0] for m, t, e in zip(mus, TIME, EVENT, strict=True)]
    assert_close(out, want)


@LOGNORMAL
@pytest.mark.parametrize("x", [-10.0, -20.0, -30.0])
def test_right_row_deep_in_the_upper_tail(x):
    """spec 4.1/7.1/12.1, rev 55: a right row where `S` underflows in linear space keeps `log S = log_ndtr(x)` (`-454` at `x = -30`)."""
    c = fr.lognormal_time(MU, SIGMA, x)
    out = ll(_dist(), survjax.SurvivalData(f([c]), b([False])))
    want, rtol = fr.direct("lognormal", (MU, SIGMA), c, "log_survival")
    assert_tail_close(out[0], want, rtol)


@LOGNORMAL
@pytest.mark.parametrize("x", [-10.0, -20.0, 10.0, 20.0])
def test_exact_row_in_both_tails(x):
    """spec 7.1/12.1, rev 55: exact rows far in either tail, where `f` underflows: `log f` against mpmath (12.1: "compare log probabilities")."""
    t = fr.lognormal_time(MU, SIGMA, x)
    out = ll(_dist(), survjax.SurvivalData(f([t]), b([True])))
    want, rtol = fr.direct("lognormal", (MU, SIGMA), t, "log_prob")
    assert_tail_close(out[0], want, rtol)


@LOGNORMAL
@pytest.mark.parametrize("x", [5.0, 8.0])
def test_right_row_near_entry_keeps_relative_precision(x):
    """spec 4.1/7.1, rev 55: a right row at small `t` (`x > 0`) has a tiny `log S`, kept to relative precision (zero absolute tolerance)."""
    c = fr.lognormal_time(MU, SIGMA, x)
    out = ll(_dist(), survjax.SurvivalData(f([c]), b([False])))
    want, rtol = fr.direct("lognormal", (MU, SIGMA), c, "log_survival")
    assert_tail_close(out[0], want, rtol)


# =============================================================== diagnostics


@LOGNORMAL
def test_invalid_data_rows_are_nan_and_attributed_to_data():
    """spec 6.2/7.1, rev 55: rows in `data.issues.invalid_rows` (negative and NaN exact times) are NaN; valid rows keep their values; the recipe attributes the NaN to data."""
    data = survjax.SurvivalData(f([1.5, -1.0, NAN, 2.5]), b([True, True, True, False]))
    out = ll(_dist(), data)
    np.testing.assert_array_equal(np.isnan(out), [False, True, True, False])
    bad_data, bad_params, other = nan_causes(_dist(), data, out)
    np.testing.assert_array_equal(bad_data, [False, True, True, False])
    assert not bad_params.any() and not other.any()
    assert_close(out[[0, 3]], _exact_right_reference(time=[1.5, 2.5], event=[True, False]))


@LOGNORMAL
@pytest.mark.parametrize(
    "mu, sigma",
    [(0.0, 0.0), (0.0, -1.0), (NAN, 1.0), (INF, 1.0), (0.0, INF)],
    ids=["sdlog-zero", "sdlog-negative", "meanlog-nan", "meanlog-inf", "sdlog-inf"],
)
def test_parameter_issue_rows_are_nan_and_attributed_to_parameters(mu, sigma):
    """spec 4.7/7.1, rev 55: rows marked by `parameter_issues().invalid_entries` are NaN "even where the family's formula is finite there"; the others keep their values."""
    d = survjax.LogNormal(meanlog=f([MU, mu, MU]), sdlog=f([SIGMA, sigma, SIGMA]))
    data = survjax.SurvivalData(f([1.5, 1.5, 2.5]), b([True, False, False]))
    out = ll(d, data)
    np.testing.assert_array_equal(np.isnan(out), [False, True, False])
    _, bad_params, other = nan_causes(d, data, out)
    np.testing.assert_array_equal(bad_params, [False, True, False])
    assert not other.any()
    assert_close(out[[0, 2]], _exact_right_reference(time=[1.5, 2.5], event=[True, False]))


@LOGNORMAL
def test_valid_batch_has_no_nan_and_no_triage_mask():
    """spec 7.1, rev 55: valid data and parameters give finite rows and empty triage masks."""
    data = survjax.SurvivalData(f(TIME), b(EVENT))
    out = ll(_dist(), data)
    assert np.isfinite(out).all()
    for mask in nan_causes(_dist(), data, out):
        assert not mask.any()


# =============================================================== transformations


@LOGNORMAL
def test_likelihood_agrees_under_jit():
    """spec 7.3/12.1, rev 55: eager and `jit` agree, with prepared data and with traced construction."""
    d, data = _dist(), survjax.SurvivalData(f(TIME), b(EVENT))
    want = ll(d, data)
    assert_close(jax.jit(lambda dd, da: dd.log_likelihood(da))(d, data), want)
    assert_close(jax.jit(lambda t, e: _dist().log_likelihood(survjax.SurvivalData(t, e)))(f(TIME), b(EVENT)), want)


@LOGNORMAL
def test_likelihood_under_vmap_over_parameter_draws():
    """spec 5/7.3/12.1, rev 55: `vmap` over draws of `(meanlog, sdlog)` equals the `(D, 1)` batch."""
    mus, sigmas = f([0.0, 0.8, -0.3]), f([0.5, 0.9, 1.4])
    data = survjax.SurvivalData(f(TIME), b(EVENT))
    mapped = jax.vmap(lambda m, s: survjax.LogNormal(meanlog=m, sdlog=s).log_likelihood(data))(mus, sigmas)
    assert_close(mapped, ll(survjax.LogNormal(meanlog=mus[:, None], sdlog=sigmas[:, None]), data))


@LOGNORMAL
def test_likelihood_under_debug_nans():
    """spec 7.2/12.1, rev 55: a valid mixed batch (tiny, interior, upper-tail and zero times) raises no NaN eagerly under `jax_debug_nans`, in values or the gradient."""
    times = f([1.5, 1e-30, fr.lognormal_time(MU, SIGMA, -25.0), 0.0, 1e-15, 2.0, fr.lognormal_time(MU, SIGMA, -25.0)])
    data = survjax.SurvivalData(times, b([True, True, True, False, False, False, False]))

    def total(m, s):
        out = survjax.LogNormal(meanlog=m, sdlog=s).log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, argnums=(0, 1), has_aux=True)(f(MU), f(SIGMA))
    values = np.asarray(values)
    assert np.isfinite(values).all()
    assert values[3] == 0.0
    assert np.isfinite(np.asarray(grads)).all()


# =============================================================== PH wrapper of a log-normal baseline


@LOGNORMAL
def test_ph_wrapper_of_lognormal_keeps_the_ph_formulas():
    """spec 4.4, rev 55: PH over `LogNormal` (no closure) keeps the PH formulas: exact `log h0 + eta + exp(eta) log S0`, right `exp(eta) log S0`."""
    eta = 0.3
    d = ph(_dist(), f(eta))
    out = ll(d, survjax.SurvivalData(f([1.5, 2.5]), b([True, False])))
    (e,) = oref.rounded(eta)
    log_h0, _ = fr.direct("lognormal", (MU, SIGMA), 1.5, "log_hazard")
    log_s0_exact, _ = fr.direct("lognormal", (MU, SIGMA), 1.5, "log_survival")
    log_s0_right, _ = fr.direct("lognormal", (MU, SIGMA), 2.5, "log_survival")
    assert_close(out, [log_h0 + e + math.exp(e) * log_s0_exact, math.exp(e) * log_s0_right])


# =============================================================== unsupported modes until Slice B


@pytest.mark.until("observation.censoring")
@LOGNORMAL
def test_left_and_interval_rows_are_nan_until_censoring_is_enabled():
    """spec 7.1/12.2, rev 55: left and interval rows are NaN while `observation.censoring` is not enabled; exact and right rows keep their values; the recipe reports them as unsupported."""
    data = flag_data(["exact", "left", "interval", "right"], [1.5, 2.0, 0.5, 2.5], [NAN, NAN, 3.0, NAN])
    out = ll(_dist(), data)
    np.testing.assert_array_equal(np.isnan(out), [False, True, True, False])
    np.testing.assert_array_equal(unsupported_nan(_dist(), data, out), [False, True, True, False])
    assert_close(out[[0, 3]], _exact_right_reference(time=[1.5, 2.5], event=[True, False]))


@pytest.mark.until("observation.truncation")
@LOGNORMAL
def test_truncated_rows_are_nan_until_truncation_is_enabled():
    """spec 7.1/12.2, rev 55: rows with truncation are NaN while `observation.truncation` is not enabled; untruncated rows keep their values."""
    data = survjax.SurvivalData(f([1.5, 1.5, 2.5]), b([True, True, False]), left_trunc=f([0.0, 0.5, 0.0]))
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
        oref.contribution("lognormal", (MU, SIGMA), 1.5, exact=True),
        oref.contribution("lognormal", (MU, SIGMA), 0.0, 0.6),
        oref.contribution("lognormal", (MU, SIGMA), 0.5, 3.0),
        oref.contribution("lognormal", (MU, SIGMA), 2.5, INF),
        oref.contribution("lognormal", (MU, SIGMA), 0.0, 7.0),
        oref.contribution("lognormal", (MU, SIGMA), 3.0, 3.2),
    ]
    assert_close(out, want)


@WITH_CENSORING
def test_left_row_where_the_cdf_is_tiny():
    """spec 4.1/7.1/12.1, rev 55: a left row at `z = -20` gives `log F = log_ndtr(-20)`, about `-203`, not `-inf`."""
    c = fr.lognormal_time(MU, SIGMA, 20.0)
    out = ll(_dist(), flag_data(["left"], [c]))
    want, rtol = fr.direct("lognormal", (MU, SIGMA), c, "log_cdf")
    assert_tail_close(out[0], want, rtol)


@WITH_CENSORING
def test_left_row_where_the_cdf_is_near_one():
    """spec 4.1/12.1, rev 55: a left row deep in the upper tail (`x = -8`): `log F ≈ -S` is negative and nonzero, to relative precision."""
    c = fr.lognormal_time(MU, SIGMA, -8.0)
    out = ll(_dist(), flag_data(["left"], [c]))
    want, rtol = fr.direct("lognormal", (MU, SIGMA), c, "log_cdf")
    assert out[0] < 0.0
    assert_tail_close(out[0], want, rtol)


@WITH_CENSORING
def test_interval_rows_are_bounded_by_their_end_points():
    """spec 2/12.1, rev 55: `logaddexp` of the adjacent sets `(0, a]` and `(a, b]` equals the left row at `b` (interval additivity)."""
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
        oref.contribution("lognormal", (MU, SIGMA), 1.5, exact=True, left=left, right=right),
        oref.contribution("lognormal", (MU, SIGMA), 2.5, INF, left=left, right=right),
    ]
    assert_close(out, want)


@WITH_BOTH
def test_truncated_interval_row_with_partial_overlap():
    """spec 2/7.1, rev 55: the interval `(0.3, 2]` overlaps the window `(0.5, 3]` partially and contributes `log P(0.5<T<=2) - log P(0.5<T<=3)`."""
    data = flag_data(["interval"], [0.3], [2.0], left_trunc=[0.5], right_trunc=[3.0])
    out = ll(_dist(), data)
    want = oref.contribution("lognormal", (MU, SIGMA), 0.3, 2.0, left=0.5, right=3.0)
    assert_close(out, [want])
