"""Feature `summary.mean`: mean values (spec 4.4, 8.1, 8.3, 8.5, 8.8, 12.1; rev 51, 54).

8.1: `mean() = ∫_0^inf S(t) dt`; Exponential `1/r`, Weibull `b*Γ(1+1/k)`. "A
converged numerical mean does not prove a finite expectation ... A known
infinite mean (from a family override) is `+inf`, with its own status in
`mean_with_info`; unresolved convergence yields NaN." 8.3: tanh-sinh on
`[0, s]` plus exp-sinh on `[s, +inf)`; "verification covers heavy tails
(log-logistic `beta` near 1), bounded support and scale changes". 4.4: PH uses
the closure where provided, "otherwise the generic method". 8.8: gated by the
12.1 Summaries row.

Rev 54 (8.1; Q-060 resolved): "A family reports it by overriding
`mean_with_info`; `mean` returns the value part of `mean_with_info`, so one
override serves both. The generic mean returns `+inf` with status
`known_infinite`, without quadrature, wherever `log_survival_at_infinity() >
-inf`." Rev 56 (8.1; Q-067 settled): "The override has the signature
`mean_with_info(self, *, options=None) -> tuple[Array, SummaryInfo]` and may
ignore `options`. An analytical answer reports status `converged` (or
`known_infinite`), a NaN `error_estimate` and zero work counts." The override
tests use the custom ``spec_families.known_mean_log_logistic`` (the 8.1
log-logistic row through that override) and ``mean_with_info_sentinel``; they
test the extension contract, not the built-in `LogLogistic` (feature
`family.loglogistic`, ``test_family_loglogistic_summaries.py``). Their
overrides build a `SummaryInfo`, so they also need `summary.containers`.

References: closed forms (``summary_reference`` at 60 digits). Built-in values
use the 12.1 ``rtol`` with no absolute floor; generic numerical values use
``generic_rtol()``, 10 times it (12.1, rev 54).
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
import summary_reference as sref
from helpers import (
    INF,
    STATUS_CODES,
    assert_generic_close,
    assert_rel_close,
    assert_status,
    f,
    ph,
    rounded,
    status,
)
from summary_cases import BUILT_IN, MEAN, MEAN_INFO, families

import survjax


def _ph_law_scale(k, b0, eta):
    """PH-Weibull law equals `Weibull(k, b0*exp(-eta/k))` (spec 4.4)."""
    return b0 * math.exp(-eta / k)


# --------------------------------------------------------------- built-in closed forms


@pytest.mark.parametrize("r", [0.01, 0.7, 50.0])
@MEAN
def test_exponential_mean_is_the_reciprocal_rate(r):
    """spec 8.1, rev 51: Exponential mean `1/r`."""
    (rr,) = rounded(r)
    assert_rel_close(survjax.Exponential(rate=f(rr)).mean(), 1.0 / rr)


@pytest.mark.parametrize("k", [0.2, 0.5, 1.0, 2.5, 10.0, 50.0])
@MEAN
def test_weibull_mean_closed_form(k):
    """spec 8.1, rev 51: Weibull mean `b*Γ(1+1/k)` over the 8.7 shape range."""
    kk, bb = rounded(k, 2.0)
    assert_rel_close(survjax.Weibull(shape=f(kk), scale=f(bb)).mean(), sref.weibull_mean(kk, bb))


@MEAN
def test_built_in_mean_is_vectorized_over_the_batch():
    """spec 8.1/8.7, rev 51: the mean has `batch_shape`; each entry is its own closed form."""
    ks = rounded(0.5, 1.0, 3.0)
    got = survjax.Weibull(shape=f(ks), scale=f(2.0)).mean()
    assert np.shape(got) == (3,)
    assert_rel_close(got, [sref.weibull_mean(k, 2.0) for k in ks])


@pytest.mark.parametrize("c", [1e-6, 1e6])
@MEAN
def test_built_in_mean_scales_with_the_time_scale(c):
    """spec 8.3/12.1, rev 51: scale changes for the built-in mean."""
    kk, bb = rounded(2.5, 2.0 * c)
    assert_rel_close(survjax.Weibull(shape=f(kk), scale=f(bb)).mean(), sref.weibull_mean(kk, bb))


# --------------------------------------------------------------- generic fallbacks


@pytest.mark.parametrize("r", [0.01, 0.5, 50.0])
@MEAN
def test_generic_mean_of_a_constant_hazard(r):
    """spec 8.3/8.8, rev 51: generic mean (tanh-sinh plus exp-sinh) of a custom `S = exp(-r t)` is `1/r`."""
    (rr,) = rounded(r)
    assert_generic_close(fam.const_hazard()(rate=f(rr)).mean(), 1.0 / rr)


@pytest.mark.parametrize("k", [0.5, 1.0, 2.5, 10.0])
@MEAN
def test_generic_mean_of_a_power_hazard(k):
    """spec 8.3/8.8, rev 51: generic mean of a custom Weibull-like family is `b*Γ(1+1/k)`."""
    kk, bb = rounded(k, 2.0)
    assert_generic_close(fam.power_hazard()(shape=f(kk), scale=f(bb)).mean(), sref.weibull_mean(kk, bb))


@MEAN
def test_generic_mean_is_vectorized_over_a_batch():
    """spec 8.1/8.8, rev 51: per-entry generic means on a batched custom family."""
    rates = rounded(0.1, 1.0, 10.0)
    assert_generic_close(fam.const_hazard()(rate=f(rates)).mean(), [1.0 / r for r in rates])


@pytest.mark.parametrize("c", [1e-6, 1e6])
@MEAN
def test_generic_mean_scales_with_the_time_scale(c):
    """spec 8.3/8.5, rev 51: scale changes: with `b -> c*b` the generic mean scales by `c`."""
    kk, bb = rounded(2.5, 2.0 * c)
    assert_generic_close(fam.power_hazard()(shape=f(kk), scale=f(bb)).mean(), sref.weibull_mean(kk, bb))


@MEAN
def test_mean_is_in_the_parameter_dtype():
    """spec 3.3/8.1, rev 51: built-in and generic means are in the parameter dtype."""
    for d in (survjax.Weibull(shape=f(2.5), scale=f(2.0)), fam.const_hazard()(rate=f(0.5))):
        assert jnp.result_type(d.mean()) == jnp.result_type(f(0.0))


# --------------------------------------------------------------- PH identities


@pytest.mark.parametrize("closure", [True, False], ids=["builtin-closure", "custom-generic"])
@MEAN
def test_ph_weibull_mean_equals_the_equivalent_weibull(closure):
    """spec 4.4/8.8, rev 51: PH-Weibull mean is that of `Weibull(k, b0*exp(-eta/k))`, through the closure or the generic path."""
    kk, bb, ee = rounded(1.5, 2.0, 0.3)
    base = (survjax.Weibull if closure else fam.power_hazard())(shape=f(kk), scale=f(bb))
    want = sref.weibull_mean(kk, _ph_law_scale(kk, bb, ee))
    check = assert_rel_close if closure else assert_generic_close
    check(ph(base, f(ee)).mean(), want)


@MEAN
def test_ph_exponential_mean_equals_the_equivalent_exponential():
    """spec 4.4, rev 51: PH-Exponential mean equals that of `Exponential(r*exp(eta))`: `exp(-eta)/r`."""
    rr, ee = rounded(0.7, -0.4)
    assert_rel_close(ph(survjax.Exponential(rate=f(rr)), f(ee)).mean(), 1.0 / (rr * math.exp(ee)))


@MEAN
def test_ph_mean_with_draw_by_arm_broadcasting():
    """spec 4.4/8.7, rev 51: baseline batch `(S, 1)` with `eta` `(S, 2)`; the mean per entry has shape `(S, 2)`."""
    ks = np.asarray(f([[1.5], [2.5], [0.8]]))
    eta = np.asarray(f([[0.0, 0.5], [-0.3, 0.2], [0.1, -1.0]]))
    for make, check in ((survjax.Weibull, assert_rel_close), (fam.power_hazard(), assert_generic_close)):
        mean = ph(make(shape=f(ks), scale=f(2.0)), f(eta)).mean()
        assert np.shape(mean) == (3, 2)
        want = [
            [sref.weibull_mean(float(ks[i, 0]), _ph_law_scale(float(ks[i, 0]), 2.0, float(eta[i, j]))) for j in range(2)]
            for i in range(3)
        ]
        check(mean, want)


# --------------------------------------------------------------- bounded support, improper, heavy tails


@MEAN
def test_bounded_support_mean_is_half_the_end():
    """spec 8.3/8.8, rev 51: mean of uniform `(0, end]` is `∫_0^end (1 - t/end) dt = end/2`."""
    assert_generic_close(fam.uniform_support()(end=f(2.0)).mean(), 1.0)


@MEAN
def test_improper_family_mean_is_plus_infinity():
    """spec 8.1, rev 54: "The generic mean returns `+inf` ... wherever `log_survival_at_infinity() > -inf`" (`mixture_cure`, `S(+inf) = c = 0.3`)."""
    assert np.asarray(fam.mixture_cure()(cure=f(0.3), rate=f(1.0)).mean()) == INF


@MEAN_INFO
def test_improper_family_mean_is_known_infinite():
    """spec 8.1/8.5, rev 54: the same mean is `+inf` "with status `known_infinite`" in `mean_with_info`, and `mean()` is its value part."""
    d = fam.mixture_cure()(cure=f(0.3), rate=f(1.0))
    value, info = d.mean_with_info()
    assert np.asarray(value) == INF
    assert_status(info, "known_infinite")
    assert np.asarray(d.mean()) == INF


@MEAN_INFO
def test_improper_family_mean_uses_no_quadrature():
    """spec 8.1/8.5, rev 54: "without quadrature": no levels, solver or bracket steps ("unused counts are zero"); the error estimate is NaN, since "fewer than two estimates exist"."""
    _, info = fam.mixture_cure()(cure=f(0.3), rate=f(1.0)).mean_with_info()
    for name in ("num_levels", "num_solver_steps", "num_bracket_steps"):
        np.testing.assert_array_equal(np.asarray(getattr(info, name)), 0, err_msg=name)
    assert np.isnan(np.asarray(info.error_estimate))


@MEAN_INFO
def test_known_infinite_mean_is_decided_per_entry():
    """spec 8.1/8.5, rev 54: "wherever" `log_survival_at_infinity() > -inf`, per broadcast entry: cure `[0.3, 0]` gives `[+inf, 1/r]` with `[known_infinite, converged]`.

    With ``c = 0``, ``log S = logaddexp(log 0, log1p(0) - r t) = -r t``, a
    proper exponential law with mean ``1/r = 2`` (``r = 0.5``) from the
    generic quadrature.
    """
    (r,) = rounded(0.5)
    values, info = fam.mixture_cure()(cure=f([0.3, 0.0]), rate=f([1.0, r])).mean_with_info()
    values = np.asarray(values)
    assert values[0] == INF
    assert_generic_close(values[1], 1.0 / r)
    assert_status(info, ["known_infinite", "converged"])


@pytest.mark.parametrize("beta", [0.5, 1.0])
@MEAN_INFO
def test_log_logistic_mean_override_reports_known_infinite(beta):
    """spec 8.1, rev 56: log-logistic "known infinite for `beta<=1`", reported by a custom `mean_with_info(self, *, options=None)` override: `+inf` with `known_infinite`; "`mean` returns the value part of `mean_with_info`, so one override serves both"."""
    d = fam.known_mean_log_logistic()(alpha=f(2.0), beta=f(beta))
    value, info = d.mean_with_info()
    assert np.asarray(value) == INF
    assert_status(info, "known_infinite")
    assert np.asarray(d.mean()) == INF


@MEAN_INFO
def test_log_logistic_mean_override_closed_form_for_beta_two():
    """spec 8.1, rev 56: the same override for `beta = 2 > 1`: `alpha*(π/beta)/sin(π/beta) = alpha*π/2`, `converged`, and `mean()` equals it."""
    aa, bb = rounded(2.0, 2.0)
    d = fam.known_mean_log_logistic()(alpha=f(aa), beta=f(bb))
    value, info = d.mean_with_info()
    assert_rel_close(value, sref.log_logistic_mean(aa, bb))
    assert_status(info, "converged")
    assert_rel_close(d.mean(), sref.log_logistic_mean(aa, bb))


@MEAN_INFO
def test_log_logistic_mean_override_is_per_entry():
    """spec 8.1/8.5, rev 56: a batch `beta = [0.5, 1, 2]` keeps the override's per-entry values and statuses through `mean` and `mean_with_info`."""
    aa, bb = rounded(2.0, 2.0)
    d = fam.known_mean_log_logistic()(alpha=f(aa), beta=f([0.5, 1.0, bb]))
    values, info = d.mean_with_info()
    for got in (np.asarray(values), np.asarray(d.mean())):
        assert got.shape == (3,)
        assert (got[:2] == INF).all()
        assert_rel_close(got[2], sref.log_logistic_mean(aa, bb))
    assert_status(info, ["known_infinite", "known_infinite", "converged"])


@MEAN_INFO
def test_mean_returns_the_value_part_of_a_mean_with_info_override():
    """spec 8.1, rev 56: "`mean` returns the value part of `mean_with_info`": an override returning a sentinel (not `1/rate`) is what `mean()` returns, with and without `options` (the override "may ignore `options`")."""
    d = fam.mean_with_info_sentinel()(rate=f(0.5))
    value, info = d.mean_with_info()
    np.testing.assert_array_equal(np.asarray(value), fam.MEAN_SENTINEL)
    assert_status(info, "converged")
    np.testing.assert_array_equal(np.asarray(d.mean()), fam.MEAN_SENTINEL)
    np.testing.assert_array_equal(np.asarray(d.mean(options=survjax.SummaryOptions())), fam.MEAN_SENTINEL)


@pytest.mark.parametrize("family", BUILT_IN)
@MEAN_INFO
def test_built_in_analytical_mean_reports_no_work(family):
    """spec 8.1, rev 56: "An analytical answer reports status `converged` (or `known_infinite`), a NaN `error_estimate` and zero work counts": the Exponential and Weibull closed forms (8.1 table) and PH closures (4.4)."""
    _, info = families()[family]().mean_with_info()
    assert_status(info, "converged")
    assert np.isnan(np.asarray(info.error_estimate)).all()
    for name in ("num_levels", "num_solver_steps", "num_bracket_steps"):
        np.testing.assert_array_equal(np.asarray(getattr(info, name)), 0, err_msg=name)


@pytest.mark.parametrize("beta", [3.0, 1.5])
@MEAN
def test_heavy_tail_generic_mean(beta):
    """spec 8.3/8.8, rev 51: heavy tails: custom log-logistic mean `alpha*(π/beta)/sin(π/beta)` (8.1)."""
    aa, bb = rounded(2.0, beta)
    got = fam.softplus_log_logistic()(alpha=f(aa), beta=f(bb)).mean()
    assert_generic_close(got, sref.log_logistic_mean(aa, bb))


@MEAN_INFO
def test_heavy_tail_near_one_mean_is_correct_or_reported_unresolved():
    """spec 8.1/8.3, rev 51: "log-logistic `beta` near 1": a converged value is close to the true mean; otherwise NaN with a non-converged status; never `+inf` (the mean is finite, and only a known infinite mean is `+inf`)."""
    aa, bb = rounded(2.0, 1.05)
    value, info = fam.softplus_log_logistic()(alpha=f(aa), beta=f(bb)).mean_with_info()
    value, code = float(np.asarray(value)), int(status(info))
    assert value != INF
    assert code != STATUS_CODES["known_infinite"]
    if code == STATUS_CODES["converged"]:
        assert_generic_close(value, sref.log_logistic_mean(aa, bb))
    else:
        assert np.isnan(value)
