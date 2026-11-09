"""observation.censoring: the `log_interval_prob(lower, upper)` member (spec 4.2, 7.1, 7.2; rev 51).

4.2: "`log_interval_prob(lower, upper)` | Stable log difference with tail
selection: `log_survival` when `upper=+inf`, `log_cdf` when `lower=0`,
otherwise the better-conditioned of the two differences via `log1mexp`";
overrides "receive endpoints `0` and `+inf` ... and must handle them". 7.1:
"Direct interval probability with equal endpoints is zero; reversed endpoints
are invalid." Calling convention and the value for reversed endpoints are
guesses (Q-065): positional ``(lower, upper)`` broadcasting against
``batch_shape`` like a direct-method argument, and NaN for reversed
endpoints (direct methods "never raise on values and carry no issue masks",
4.5).

Exponential ``r = 0.5`` and Weibull ``k = 1.5``, ``b = 2``; interval values
``-H(a) + log(-expm1(-(H(b) - H(a))))`` as in
``test_observation_censoring_values.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as ref
import pytest
import spec_families as fam
from helpers import INF, assert_close, f, flag_data, ll

import survjax

FEATURE = pytest.mark.feature("observation.censoring")


def _families():
    return {
        "exponential": survjax.Exponential(rate=f(0.5)),
        "weibull": survjax.Weibull(shape=f(1.5), scale=f(2.0)),
        "custom": fam.power_hazard()(shape=f(1.5), scale=f(2.0)),
    }


REF = {"exponential": ("exponential", (0.5,)), "weibull": ("weibull", (1.5, 2.0)), "custom": ("weibull", (1.5, 2.0))}


@FEATURE
@pytest.mark.parametrize("name", list(REF))
def test_finite_interval_matches_reference(name):
    """spec 4.2, rev 51: `log_interval_prob(a, b) = log(S(a) - S(b))` for `(0.7, 2.6]` (built-in overrides and the generic default)."""
    dist = _families()[name]
    family, params = REF[name]
    value = np.asarray(dist.log_interval_prob(f(0.7), f(2.6)))
    assert_close(value, ref.contribution(family, params, 0.7, 2.6))


@FEATURE
@pytest.mark.parametrize("name", list(REF))
def test_upper_infinite_endpoint_is_log_survival(name):
    """spec 4.2, rev 51: "`log_survival` when `upper=+inf`"."""
    dist = _families()[name]
    assert_close(np.asarray(dist.log_interval_prob(f(1.3), f(INF))), np.asarray(dist.log_survival(f(1.3))))


@FEATURE
@pytest.mark.parametrize("name", list(REF))
def test_zero_lower_endpoint_is_log_cdf(name):
    """spec 4.2, rev 51: "`log_cdf` when `lower=0`"."""
    dist = _families()[name]
    assert_close(np.asarray(dist.log_interval_prob(f(0.0), f(1.3))), np.asarray(dist.log_cdf(f(1.3))))


@FEATURE
@pytest.mark.parametrize("name", list(REF))
def test_whole_line_is_log_one(name):
    """spec 2, 4.2 and 4.5, rev 51: `(0, +inf]` of a proper family: `log(S(0) - S(+inf)) = log 1 = 0`."""
    dist = _families()[name]
    assert np.asarray(dist.log_interval_prob(f(0.0), f(INF))) == 0.0


@FEATURE
@pytest.mark.parametrize("name", list(REF))
def test_equal_endpoints_give_zero_probability(name):
    """spec 7.1, rev 51: "Direct interval probability with equal endpoints is zero": log value `-inf`."""
    dist = _families()[name]
    assert np.asarray(dist.log_interval_prob(f(1.3), f(1.3))) == -INF


@FEATURE
@pytest.mark.parametrize("name", list(REF))
def test_reversed_endpoints_are_nan(name):
    """spec 4.5 and 7.1, rev 51: "reversed endpoints are invalid"; a direct method does not raise, so the value is NaN (Q-065)."""
    dist = _families()[name]
    assert np.isnan(np.asarray(dist.log_interval_prob(f(2.6), f(0.7))))


@FEATURE
def test_improper_family_interval_to_infinity_includes_the_cure_mass():
    """spec 2 and 4.2, rev 51: "Sets closed at `+inf` include `T=+inf`, so for an improper family `P(c<T<=+inf)=S(c)`".

    ``mixture_cure`` with ``c = 0.3``, ``r = 0.5``: ``log S(2) = log(0.3 + 0.7 e**-1)``.
    """
    dist = fam.mixture_cure()(cure=f(0.3), rate=f(0.5))
    cure, r = ref.rounded(0.3, 0.5)
    assert_close(np.asarray(dist.log_interval_prob(f(2.0), f(INF))), np.log(cure + (1 - cure) * np.exp(-r * 2.0)))


@FEATURE
def test_endpoints_broadcast_against_the_batch():
    """spec 5, rev 51: endpoints of shape `(m, 1)` against a `(2,)` batch give `(m, 2)` (Q-065)."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    out = np.asarray(dist.log_interval_prob(f([[0.5], [1.0]]), f([[1.0], [3.0]])))
    assert out.shape == (2, 2)
    expected = [
        [ref.contribution("exponential", (r,), lo, hi) for r in (0.5, 2.0)]
        for lo, hi in ((0.5, 1.0), (1.0, 3.0))
    ]
    assert_close(out, expected)


@FEATURE
def test_member_agrees_with_the_interval_row_contribution():
    """spec 2 and 4.2, rev 51: untruncated, an interval row contributes `log P(a<T<=b)`, the member's value."""
    dist = survjax.Weibull(shape=f(1.5), scale=f(2.0))
    row = ll(dist, flag_data(["interval"], [0.7], [2.6]))[0]
    assert_close(row, np.asarray(dist.log_interval_prob(f(0.7), f(2.6))))


@FEATURE
def test_member_under_jit_and_its_gradient():
    """spec 7.3 and 12.1, rev 51: the member under `jit`, and its rate derivative `-a + w/expm1(r w)` (Exponential, `(1, 3]`)."""

    def fn(r):
        return survjax.Exponential(rate=r).log_interval_prob(f(1.0), f(3.0))

    r = jnp.asarray(f(0.5))
    assert_close(jax.jit(fn)(r), fn(r))
    rr = float(np.asarray(f(0.5)))
    assert_close(jax.grad(fn)(r), -1.0 + 2.0 / np.expm1(2.0 * rr))
