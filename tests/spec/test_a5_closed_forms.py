"""A5: closed-form values of the built-in Exponential and Weibull families (spec 2, 4.1, 4.2, 8.1).

Hand derivations, from spec 4.1's survival functions and spec 2's definitions
(``H = -log S``, ``F = 1 - S``, ``f = -dS/dt``, ``h = f/S``):

Exponential, rate ``r``: ``log S = -r t``, ``H = r t``,
``log F = log1mexp(-r t)``, ``log f = log r - r t``, ``log h = log r``,
``h = r``.

Weibull, shape ``k``, scale ``b``, ``z = t/b``: ``log S = -z**k``,
``H = z**k``, ``log F = log1mexp(-z**k)``,
``log f = log k - log b + (k-1) log z - z**k``,
``log h = log k - log b + (k-1) log z``, ``h = (k/b) z**(k-1)``.

Expected values are evaluated in float64 NumPy from these formulas, at inputs
that are exactly representable or whose float32 rounding is far below the 12.1
float32 tolerances. ``log1mexp(x) = log(1 - exp(x))`` is the two-branch form of
spec 7.2 (`helpers.log1mexp`). Rev 25 (12.1): `log_cdf` values are compared
with zero absolute tolerance (`_check`), since upper-tail `log F` (Weibull
``k=3``, ``b=2``, ``t=5``: ``H = 15.6``, ``log F = -1.6e-7``) is below the
float32 ``atol``.
"""

import numpy as np
import pytest
from helpers import assert_close, assert_rel_close, f, log1mexp

import survjax


def _exp_expected(r, t):
    r, t = (np.asarray(x, np.float64) for x in (r, t))
    r, t = np.broadcast_arrays(r, t)
    return {
        "log_survival": -r * t,
        "survival": np.exp(-r * t),
        "cumulative_hazard": r * t,
        "log_cdf": log1mexp(-r * t),
        "log_prob": np.log(r) - r * t,
        "log_hazard": np.log(r),
        "hazard": r,
    }


def _weib_expected(k, b, t):
    k, b, t = (np.asarray(x, np.float64) for x in (k, b, t))
    k, b, t = np.broadcast_arrays(k, b, t)
    z = t / b
    zk = z**k
    log_h = np.log(k) - np.log(b) + (k - 1.0) * np.log(z)
    return {
        "log_survival": -zk,
        "survival": np.exp(-zk),
        "cumulative_hazard": zk,
        "log_cdf": log1mexp(-zk),
        "log_prob": log_h - zk,
        "log_hazard": log_h,
        "hazard": np.exp(log_h),
    }


DIRECT_METHODS = (
    "log_survival",
    "survival",
    "log_cdf",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)


def _check(method, actual, expected):
    """Spec 12.1, rev 25: tiny log probabilities use zero absolute tolerance."""
    if method == "log_cdf":
        assert_rel_close(actual, expected)
    else:
        assert_close(actual, expected)


# --------------------------------------------------------------- interior values


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_exponential_interior_values(method):
    """spec 4.1/8.1, rev 21: Exponential `S(t)=exp(-r*t)` and the derived direct methods, paired batch and times."""
    r, t = [0.5, 1.0, 3.0, 0.2], [2.0, 0.25, 1.5, 7.0]
    out = getattr(survjax.Exponential(rate=f(r)), method)(f(t))
    _check(method, out, _exp_expected(r, t)[method])


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_weibull_interior_values_decreasing_hazard(method):
    """spec 4.1/8.1, rev 21: Weibull `S(t)=exp(-(t/b)**k)` with `k<1`."""
    t = [0.1, 1.0, 2.0, 5.0]
    out = getattr(survjax.Weibull(shape=f(0.5), scale=f(2.0)), method)(f(t))
    _check(method, out, _weib_expected(0.5, 2.0, t)[method])


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_weibull_interior_values_constant_hazard(method):
    """spec 4.1/8.1, rev 21: Weibull with `k=1`: `log S = -t/b`, `log h = -log b`."""
    t = [0.1, 1.0, 2.0, 5.0]
    out = getattr(survjax.Weibull(shape=f(1.0), scale=f(2.0)), method)(f(t))
    _check(method, out, _weib_expected(1.0, 2.0, t)[method])


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_weibull_interior_values_increasing_hazard(method):
    """spec 4.1/8.1, rev 21: Weibull with `k>1`."""
    t = [0.1, 1.0, 2.0, 5.0]
    out = getattr(survjax.Weibull(shape=f(3.0), scale=f(2.0)), method)(f(t))
    _check(method, out, _weib_expected(3.0, 2.0, t)[method])


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_weibull_paired_batch_values(method):
    """spec 4.1/5, rev 21: batch `(N,)` against times `(N,)` pairs entries."""
    k, b, t = [0.7, 1.0, 1.8, 4.0], [0.5, 3.0, 10.0, 1.0], [0.2, 4.0, 12.0, 0.9]
    out = getattr(survjax.Weibull(shape=f(k), scale=f(b)), method)(f(t))
    _check(method, out, _weib_expected(k, b, t)[method])


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_weibull_with_shape_one_equals_exponential_with_rate_one_over_scale(method):
    """spec 4.1, rev 21: analytical identity `Weibull(shape=1, scale=b) = Exponential(rate=1/b)`."""
    t = f([0.0, 0.3, 1.0, 4.0, 25.0])
    weib = getattr(survjax.Weibull(shape=f(1.0), scale=f(4.0)), method)(t)
    expo = getattr(survjax.Exponential(rate=f(0.25)), method)(t)
    _check(method, weib, expo)


# --------------------------------------------------------------- lower tail (small t)


@pytest.mark.step("A5")
def test_exponential_small_cumulative_hazard_keeps_relative_precision():
    """spec 4.2, rev 21: `log_survival` "Must keep relative precision when `|log S|` is small"; `H = r t = 2e-30`."""
    out = survjax.Exponential(rate=f(2.0)).cumulative_hazard(f(1e-30))
    assert_rel_close(out, 2e-30)


@pytest.mark.step("A5")
def test_exponential_log_cdf_in_lower_tail():
    """spec 4.2, rev 21: `log_cdf` accurate until `|log S|` underflows; `log F = log(-expm1(-2e-30)) = log(2e-30)` to 1e-30 relative."""
    out = survjax.Exponential(rate=f(2.0)).log_cdf(f(1e-30))
    assert_close(out, np.log(2e-30))


@pytest.mark.step("A5")
def test_weibull_small_cumulative_hazard_keeps_relative_precision():
    """spec 4.2, rev 21: relative precision for small `|log S|`; `k=2`, `b=3`, `t=3e-10`: `H = (1e-10)**2 = 1e-20`."""
    out = survjax.Weibull(shape=f(2.0), scale=f(3.0)).cumulative_hazard(f(3e-10))
    assert_rel_close(out, 1e-20)


@pytest.mark.step("A5")
def test_weibull_log_cdf_in_lower_tail():
    """spec 4.2, rev 21: `log_cdf` in the lower tail; `H = 1e-20`, so `log F = log(1e-20)` to 1e-20 relative."""
    out = survjax.Weibull(shape=f(2.0), scale=f(3.0)).log_cdf(f(3e-10))
    assert_close(out, np.log(1e-20))


@pytest.mark.step("A5")
def test_weibull_log_hazard_near_zero_with_decreasing_hazard():
    """spec 4.1/4.2, rev 21: `k=0.5`, `b=1`, `t=1e-20`: `log h = log 0.5 - 0.5 log(1e-20)`, large and finite."""
    out = survjax.Weibull(shape=f(0.5), scale=f(1.0)).log_hazard(f(1e-20))
    assert_close(out, np.log(0.5) - 0.5 * np.log(1e-20))


@pytest.mark.step("A5")
def test_weibull_log_prob_near_zero_with_decreasing_hazard():
    """spec 4.1, rev 21: `k=0.5`, `b=1`, `t=1e-20`: `log f = log 0.5 + 23.03 - 1e-10`."""
    out = survjax.Weibull(shape=f(0.5), scale=f(1.0)).log_prob(f(1e-20))
    assert_close(out, np.log(0.5) - 0.5 * np.log(1e-20) - 1e-10)


# --------------------------------------------------------------- upper tail (large t)


@pytest.mark.step("A5")
def test_exponential_log_survival_in_upper_tail():
    """spec 4.1, rev 21: `log S = -r t` stays exact where `S` underflows (`r=0.3`, `t=1e7`)."""
    out = survjax.Exponential(rate=f(0.3)).log_survival(f(1e7))
    assert_close(out, -0.3 * 1e7)


@pytest.mark.step("A5")
def test_exponential_log_prob_in_upper_tail():
    """spec 4.1, rev 21: `log f = log r - r t` where `f` underflows (12.1: "compare log probabilities when probabilities underflow")."""
    out = survjax.Exponential(rate=f(0.3)).log_prob(f(1e7))
    assert_close(out, np.log(0.3) - 0.3 * 1e7)


@pytest.mark.step("A5")
def test_exponential_log_hazard_in_upper_tail_is_analytic():
    """spec 4.2, rev 21: "Built-ins override analytically; the generic `exp(log_prob - log_survival)` form cancels badly when `H` is large".

    ``r = 0.3``, ``t = 1e7``: ``H = 3e6``; ``log h = log 0.3``. In float32 the
    difference ``log f - log S`` would lose about ``ulp(3e6) = 0.25``.
    """
    out = survjax.Exponential(rate=f(0.3)).log_hazard(f(1e7))
    assert_close(out, np.log(0.3))


@pytest.mark.step("A5")
def test_exponential_hazard_in_upper_tail_is_analytic():
    """spec 4.2/8.1, rev 21: `h = r` where `S` and `f` both underflow."""
    out = survjax.Exponential(rate=f(0.3)).hazard(f(1e7))
    assert_close(out, 0.3)


@pytest.mark.step("A5")
def test_weibull_log_survival_in_upper_tail():
    """spec 4.1, rev 21: `k=2`, `b=1`, `t=1e7`: `log S = -1e14`, relative."""
    out = survjax.Weibull(shape=f(2.0), scale=f(1.0)).log_survival(f(1e7))
    assert_rel_close(out, -1e14)


@pytest.mark.step("A5")
def test_weibull_log_prob_in_upper_tail():
    """spec 4.1, rev 21: `log f = log 2 + log 1e7 - 1e14`, relative (`f` underflows)."""
    out = survjax.Weibull(shape=f(2.0), scale=f(1.0)).log_prob(f(1e7))
    assert_rel_close(out, np.log(2.0) + np.log(1e7) - 1e14)


@pytest.mark.step("A5")
def test_weibull_log_hazard_in_upper_tail_is_analytic():
    """spec 4.2, rev 21: built-in `log_hazard` is analytic, not `log_prob - log_survival`.

    ``k=2``, ``b=1``, ``t=1e7``: ``log h = log 2 + log 1e7``; ``H = 1e14``, so the
    generic difference would lose about ``ulp(1e14)`` (``0.016`` in float64,
    ``8e6`` in float32).
    """
    out = survjax.Weibull(shape=f(2.0), scale=f(1.0)).log_hazard(f(1e7))
    assert_close(out, np.log(2.0) + np.log(1e7))


@pytest.mark.step("A5")
def test_weibull_hazard_in_upper_tail_is_analytic():
    """spec 4.2/8.1, rev 21: `h = (k/b) z**(k-1) = 2e7` where `S` and `f` underflow."""
    out = survjax.Weibull(shape=f(2.0), scale=f(1.0)).hazard(f(1e7))
    assert_close(out, 2e7)


@pytest.mark.step("A5")
def test_weibull_cumulative_hazard_in_upper_tail():
    """spec 4.2/8.1, rev 21: `H = -log S = z**k = 1e14` where `S` underflows."""
    out = survjax.Weibull(shape=f(2.0), scale=f(1.0)).cumulative_hazard(f(1e7))
    assert_rel_close(out, 1e14)


@pytest.mark.step("A5")
def test_survival_underflows_to_zero_not_nan_in_upper_tail():
    """spec 8.1, rev 21: `survival` is `P(T>t)`; where it underflows it is `0`, not NaN or negative."""
    s_exp = np.asarray(survjax.Exponential(rate=f(0.3)).survival(f(1e7)))
    s_weib = np.asarray(survjax.Weibull(shape=f(2.0), scale=f(1.0)).survival(f(1e7)))
    np.testing.assert_array_equal(s_exp, 0.0)
    np.testing.assert_array_equal(s_weib, 0.0)
