"""A7: exact and right-censored likelihood against mpmath across scales (spec 2, 4.1, 7.1, 12.1, 12.2; rev 33).

Without truncation ``Z = 1`` (spec 2), so an exact row contributes ``log f(t)``
and a right-censored row ``log S(c)``:

- Exponential, rate ``r``: ``log r - r t``; ``-r c``.
- Weibull, shape ``k``, scale ``b``, ``z = t/b``:
  ``log k - log b + (k-1) log z - z**k``; ``-z**k``.

References come from mpmath at 50 digits, evaluated at the inputs after
rounding to the active dtype (``helpers.mp_exponential_contribution``,
``helpers.mp_weibull_contribution``). Comparisons use zero absolute
tolerance and a relative tolerance that is the 12.1 ``rtol`` unless
cancellation between terms needs more (justified in ``helpers._reference``).

The cases span rates and scales from ``1e-3`` to ``1e4``, shapes from 0.3 to
8 and contributions from about ``-1e3`` to ``+6``, every value a normal number
in both dtypes. Each case is its own test so that a failure names the case.
"""

import jax
import numpy as np
import pytest
from helpers import (
    assert_tail_close,
    b,
    f,
    ll,
    mp_exponential_contribution,
    mp_weibull_contribution,
)

import survjax

# (rate, time)
EXPONENTIAL_CASES = {
    "small_rate_long_time": (1e-3, 5e2),
    "unit_scale": (0.5, 2.0),
    "large_rate_short_time": (3.0, 1e-4),
    "very_large_rate": (1e3, 1e-2),
    "tiny_rate": (1e-6, 1e5),
    "large_hazard": (2.0, 5e2),
}

# (shape, scale, time)
WEIBULL_CASES = {
    "k0.3_large_scale": (0.3, 1e3, 5.0),
    "k0.5_tiny_time": (0.5, 2.0, 1e-6),
    "k1": (1.0, 10.0, 3.0),
    "bc_like": (1.3796517717622814, 11.422861823436245, 2.5),
    "k2.5": (2.5, 3.0, 7.0),
    "k8_past_scale": (8.0, 1.0, 1.2),
    "k8_before_scale": (8.0, 1.0, 0.5),
    "k2_tiny_scale": (2.0, 1e-3, 5e-3),
    "k2_large_scale": (2.0, 1e4, 3e4),
}


def _exp_value(rate, t, exact):
    return ll(survjax.Exponential(rate=f(rate)), survjax.SurvivalData(f(t), b(exact)))


def _weibull_value(k, s, t, exact):
    dist = survjax.Weibull(shape=f(k), scale=f(s))
    return ll(dist, survjax.SurvivalData(f(t), b(exact)))


@pytest.mark.step("A7")
@pytest.mark.parametrize("case", list(EXPONENTIAL_CASES))
def test_exponential_exact_contribution_matches_mpmath(case):
    """spec 2/7.1/12.1, rev 33: exponential exact contribution `log r - r t` against mpmath."""
    rate, t = EXPONENTIAL_CASES[case]
    expected, rtol = mp_exponential_contribution(rate, t, exact=True)
    assert_tail_close(_exp_value(rate, t, True), expected, rtol)


@pytest.mark.step("A7")
@pytest.mark.parametrize("case", list(EXPONENTIAL_CASES))
def test_exponential_right_censored_contribution_matches_mpmath(case):
    """spec 2/7.1/12.1, rev 33: exponential right-censored contribution `-r c` against mpmath."""
    rate, t = EXPONENTIAL_CASES[case]
    expected, rtol = mp_exponential_contribution(rate, t, exact=False)
    assert_tail_close(_exp_value(rate, t, False), expected, rtol)


@pytest.mark.step("A7")
@pytest.mark.parametrize("case", list(WEIBULL_CASES))
def test_weibull_exact_contribution_matches_mpmath(case):
    """spec 2/7.1/12.1, rev 33: Weibull exact contribution `log k - log b + (k-1) log z - z**k` against mpmath."""
    k, s, t = WEIBULL_CASES[case]
    expected, rtol = mp_weibull_contribution(k, s, t, exact=True)
    assert_tail_close(_weibull_value(k, s, t, True), expected, rtol)


@pytest.mark.step("A7")
@pytest.mark.parametrize("case", list(WEIBULL_CASES))
def test_weibull_right_censored_contribution_matches_mpmath(case):
    """spec 2/7.1/12.1, rev 33: Weibull right-censored contribution `-z**k` against mpmath."""
    k, s, t = WEIBULL_CASES[case]
    expected, rtol = mp_weibull_contribution(k, s, t, exact=False)
    assert_tail_close(_weibull_value(k, s, t, False), expected, rtol)


@pytest.mark.step("A7")
def test_weibull_mixed_batch_matches_mpmath_row_by_row():
    """spec 5/7.1/12.1, rev 33: per-row parameters and alternating kinds in one batch match mpmath per row.

    Rows are the Weibull cases in order, exact on even rows and right-censored
    on odd rows; per-row ``shape`` and ``scale`` give batch ``(N,)``.
    """
    cases = list(WEIBULL_CASES.values())
    exact = [i % 2 == 0 for i in range(len(cases))]
    k, s, t = (np.array([c[j] for c in cases]) for j in range(3))
    dist = survjax.Weibull(shape=f(k), scale=f(s))
    value = ll(dist, survjax.SurvivalData(f(t), b(exact)))
    assert value.shape == (len(cases),)
    for i, (case, e) in enumerate(zip(cases, exact, strict=True)):
        expected, rtol = mp_weibull_contribution(*case, exact=e)
        assert_tail_close(value[i], expected, rtol, err_msg=f"row {i}")


@pytest.mark.step("A7")
def test_exponential_mixed_batch_matches_mpmath_under_jit_and_vmap():
    """spec 7.3/12.1, rev 33: the exponential cases agree with mpmath under `jit` and under `vmap` over rows."""
    cases = list(EXPONENTIAL_CASES.values())
    exact = [i % 2 == 0 for i in range(len(cases))]
    rate, t = (np.array([c[j] for c in cases]) for j in range(2))

    def one(r, time, event):
        return survjax.Exponential(rate=r).log_likelihood(survjax.SurvivalData(time, event)
        )

    jitted = np.asarray(jax.jit(one)(f(rate), f(t), b(exact)))
    mapped = np.asarray(jax.vmap(one)(f(rate), f(t), b(exact)))
    for i, (case, e) in enumerate(zip(cases, exact, strict=True)):
        expected, rtol = mp_exponential_contribution(*case, exact=e)
        assert_tail_close(jitted[i], expected, rtol, err_msg=f"jit row {i}")
        assert_tail_close(mapped[i], expected, rtol, err_msg=f"vmap row {i}")
