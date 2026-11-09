"""observation.censoring: hand-derived left- and interval-censored contributions without truncation (spec 2, 3.1, 3.2, 4.1, 4.2, 5, 7.1; rev 51).

Spec 2: a left-censored row at ``c`` contributes ``P(L<T<=min(c,U))/Z`` and an
interval row ``(a, b]`` contributes ``P(max(a,L)<T<=min(b,U))/Z``. Without
truncation ``L=0``, ``U=+inf`` and ``Z=1``, so (``S(0)=1``):

- left at ``c``: ``log P(0<T<=c) = log F(c) = log(1 - exp(-H(c))) = log(-expm1(-H(c)))``;
- interval ``(a, b]``: ``log(S(a) - S(b)) = -H(a) + log(1 - exp(-(H(b) - H(a))))``
  ``= -H(a) + log(-expm1(-(H(b) - H(a))))``.

Exponential (4.1): ``H(t) = r t``, so left ``log(-expm1(-r c))`` and interval
``-r a + log(-expm1(-r (b - a)))`` (the 7.2 override form). Weibull:
``H(t) = (t/b)**k``. Exact rows ``log f``, right-censored rows ``-H(c)``, as in
A6. Expected values are computed in float64 NumPy from the dtype-rounded
inputs; ``test_hand_values_agree_with_mpmath`` cross-checks the hand formulas
against the mpmath table of ``observation_reference``.
"""

import numpy as np
import observation_reference as ref
import pytest
import spec_families as fam
from helpers import INF, NAN, X64, assert_close, b, f, flag_data, ll

import survjax

FEATURE = pytest.mark.feature("observation.censoring")


def exp_h(r, t):
    return r * t


def weib_h(k, s, t):
    return (t / s) ** k


def left(h_c):
    """``log F(c) = log(-expm1(-H(c)))``."""
    return np.log(-np.expm1(-h_c))


def interval(h_a, h_b):
    """``log(S(a) - S(b)) = -H(a) + log(-expm1(-(H(b) - H(a))))``."""
    return -h_a + np.log(-np.expm1(-(h_b - h_a)))


def r_(*values):
    return ref.rounded(*values)


# ------------------------------------------------------------ single kinds


@FEATURE
def test_exponential_left_censored_row_is_hand_derived():
    """spec 2 and 7.1, rev 51: Exponential `r=0.5`, left at `c=2`: `log(-expm1(-1))`."""
    data = flag_data(["left"], [2.0])
    (r, c) = r_(0.5, 2.0)
    assert_close(ll(survjax.Exponential(rate=f(0.5)), data), [left(exp_h(r, c))])


@FEATURE
def test_exponential_interval_censored_row_is_hand_derived():
    """spec 2, 7.1 and 7.2, rev 51: Exponential `r=0.5`, `(1, 3]`: `-r a + log(-expm1(-r (b-a)))` = `-0.5 + log(-expm1(-1))`."""
    data = flag_data(["interval"], [1.0], [3.0])
    r, a, c = r_(0.5, 1.0, 3.0)
    assert_close(ll(survjax.Exponential(rate=f(0.5)), data), [interval(exp_h(r, a), exp_h(r, c))])


@FEATURE
@pytest.mark.parametrize("k", [0.5, 1.0, 1.5, 3.0])
def test_weibull_left_censored_row_is_hand_derived(k):
    """spec 2 and 7.1, rev 51: Weibull `b=2`, left at `c=1.3`: `log(-expm1(-(c/b)**k))`."""
    data = flag_data(["left"], [1.3])
    kk, s, c = r_(k, 2.0, 1.3)
    assert_close(ll(survjax.Weibull(shape=f(k), scale=f(2.0)), data), [left(weib_h(kk, s, c))])


@FEATURE
@pytest.mark.parametrize("k", [0.5, 1.0, 1.5, 3.0])
def test_weibull_interval_censored_row_is_hand_derived(k):
    """spec 2 and 7.1, rev 51: Weibull `b=2`, `(0.7, 2.6]`: `-H(a) + log(-expm1(-(H(b)-H(a))))`, `H = (t/b)**k`."""
    data = flag_data(["interval"], [0.7], [2.6])
    kk, s, a, c = r_(k, 2.0, 0.7, 2.6)
    expected = interval(weib_h(kk, s, a), weib_h(kk, s, c))
    assert_close(ll(survjax.Weibull(shape=f(k), scale=f(2.0)), data), [expected])


@FEATURE
def test_hand_values_agree_with_mpmath():
    """spec 2, rev 51: the module's hand formulas equal the 50-digit mpmath evaluation of the spec 2 table (no survjax call)."""
    r, a, c = r_(0.5, 1.0, 3.0)
    np.testing.assert_allclose(left(exp_h(r, c)), ref.contribution("exponential", (0.5,), 0.0, 3.0), rtol=1e-13)
    np.testing.assert_allclose(
        interval(exp_h(r, a), exp_h(r, c)), ref.contribution("exponential", (0.5,), 1.0, 3.0), rtol=1e-13
    )
    k, s, a, c = r_(1.5, 2.0, 0.7, 2.6)
    np.testing.assert_allclose(
        interval(weib_h(k, s, a), weib_h(k, s, c)), ref.contribution("weibull", (1.5, 2.0), 0.7, 2.6), rtol=1e-12
    )


# ------------------------------------------------------------ mixed batches

KINDS = ["exact", "right", "left", "interval", "left", "interval"]
TIME = [1.0, 2.0, 1.5, 0.5, 4.0, 2.0]
TIME2 = [NAN, NAN, NAN, 2.5, NAN, 2.2]


def weibull_expected(k, s, kinds=KINDS, time=TIME, time2=TIME2):
    """Hand-derived Weibull contributions of mixed rows (module docstring)."""
    kk, ss = r_(k, s)
    out = []
    for kind, t, t2 in zip(kinds, time, time2, strict=True):
        (t,) = r_(t)
        h = weib_h(kk, ss, t)
        if kind == "exact":
            out.append(np.log(kk) - np.log(ss) + (kk - 1.0) * np.log(t / ss) - h)
        elif kind == "right":
            out.append(-h)
        elif kind == "left":
            out.append(left(h))
        else:
            (t2,) = r_(t2)
            out.append(interval(h, weib_h(kk, ss, t2)))
    return np.array(out)


@FEATURE
def test_weibull_mixed_batch_with_exact_and_right_rows():
    """spec 1, 2 and 7.1, rev 51: "All four observation types may be mixed": each row has its own kind's contribution."""
    data = flag_data(KINDS, TIME, TIME2)
    assert_close(ll(survjax.Weibull(shape=f(1.4), scale=f(2.2)), data), weibull_expected(1.4, 2.2))


@FEATURE
def test_exponential_mixed_batch_with_exact_and_right_rows():
    """spec 2 and 7.1, rev 51: Exponential `r=0.8` on the same mixed rows (Weibull `k=1`, `b=1/r`)."""
    data = flag_data(KINDS, TIME, TIME2)
    (r,) = r_(0.8)
    expected = []
    for kind, t, t2 in zip(KINDS, TIME, TIME2, strict=True):
        (t,) = r_(t)
        if kind == "exact":
            expected.append(np.log(r) - r * t)
        elif kind == "right":
            expected.append(-r * t)
        elif kind == "left":
            expected.append(left(r * t))
        else:
            (t2,) = r_(t2)
            expected.append(interval(r * t, r * t2))
    assert_close(ll(survjax.Exponential(rate=f(0.8)), data), expected)


@FEATURE
def test_bounds_form_equals_flag_form():
    """spec 3.1 and 12.1, rev 51: "flag and bounds forms normalize equivalent observations identically", so their contributions are equal.

    Bounds: exact ``(t, t]`` with the ``exact`` mask, right ``(c, +inf]``,
    left ``(0, c]``, interval ``(a, b]``.
    """
    lower = [1.0, 2.0, 0.0, 0.5, 0.0, 2.0]
    upper = [1.0, INF, 1.5, 2.5, 4.0, 2.2]
    exact = [True, False, False, False, False, False]
    bounds = survjax.SurvivalData.from_bounds(f(lower), f(upper), exact=b(exact))
    flags = flag_data(KINDS, TIME, TIME2)
    dist = survjax.Weibull(shape=f(1.4), scale=f(2.2))
    np.testing.assert_array_equal(ll(dist, bounds), ll(dist, flags))
    assert_close(ll(dist, bounds), weibull_expected(1.4, 2.2))


@FEATURE
def test_time2_is_ignored_on_left_rows():
    """spec 3.1, rev 51: "`time2` is read only on interval rows": a left row with any `time2` has the same contribution."""
    dist = survjax.Exponential(rate=f(0.5))
    a = ll(dist, flag_data(["left", "left"], [2.0, 2.0], [NAN, 0.3]))
    np.testing.assert_array_equal(a[0], a[1])


@FEATURE
def test_interval_rows_are_finite_and_negative():
    """spec 2 and 7.1, rev 51: valid untruncated censored sets with positive mass give finite log probabilities below zero."""
    data = flag_data(["left", "interval", "interval"], [0.1, 0.1, 5.0], [NAN, 0.2, 9.0])
    value = ll(survjax.Weibull(shape=f(0.8), scale=f(1.0)), data)
    assert np.isfinite(value).all()
    assert (value < 0).all()


# ------------------------------------------------------------- shapes (spec 5)


@FEATURE
def test_draw_column_against_censored_observations():
    """spec 5, rev 51: data `(N,)` against a distribution `(D, 1)` gives `(D, N)` contributions, each row hand-derived."""
    data = flag_data(["left", "interval", "exact"], [2.0, 1.0, 1.0], [NAN, 3.0, NAN])
    rates = [0.5, 2.0]
    out = ll(survjax.Exponential(rate=f([[0.5], [2.0]])), data)
    assert out.shape == (2, 3)
    expected = []
    for rate in rates:
        (r,) = r_(rate)
        expected.append([left(2.0 * r), interval(r, 3.0 * r), np.log(r) - r])
    assert_close(out, expected)


@FEATURE
def test_per_row_parameters_pair_with_censored_rows():
    """spec 5, rev 51: data `(N,)` with a distribution `(N,)` pairs rows: row `i` uses `(k_i, b_i)`."""
    data = flag_data(["left", "interval"], [1.0, 1.0], [NAN, 2.0])
    k, s = [0.7, 2.0], [1.5, 3.0]
    out = ll(survjax.Weibull(shape=f(k), scale=f(s)), data)
    k0, s0, k1, s1 = r_(k[0], s[0], k[1], s[1])
    assert_close(out, [left(weib_h(k0, s0, 1.0)), interval(weib_h(k1, s1, 1.0), weib_h(k1, s1, 2.0))])


@FEATURE
def test_two_dimensional_censored_data():
    """spec 3.2 and 5, rev 51: multidimensional observations are not flattened; contributions keep the observation shape."""
    time = [[2.0, 1.0], [0.5, 3.0]]
    data = survjax.SurvivalData(
        f(time),
        left_censored=b([[True, False], [False, True]]),
        interval_censored=b([[False, True], [True, False]]),
        time2=f([[NAN, 3.0], [1.5, NAN]]),
    )
    (r,) = r_(0.5)
    out = ll(survjax.Exponential(rate=f(0.5)), data)
    assert out.shape == (2, 2)
    assert_close(out, [[left(2.0 * r), interval(r, 3.0 * r)], [interval(0.5 * r, 1.5 * r), left(3.0 * r)]])


# ------------------------------------------------------ generic defaults (4.2)


@FEATURE
def test_custom_family_on_generic_defaults_matches_weibull():
    """spec 4.2, rev 51: `power_hazard` supplies only `log_survival`; the generic `log_cdf` and interval defaults give the Weibull values."""
    data = flag_data(KINDS, TIME, TIME2)
    dist = fam.power_hazard()(shape=f(1.4), scale=f(2.2))
    assert_close(ll(dist, data), weibull_expected(1.4, 2.2))


@FEATURE
def test_improper_family_left_and_interval_rows():
    """spec 2 and 4.2, rev 51: `mixture_cure` (`S = c + (1-c) e**(-r t)`): left `log((1-c)(1-e**(-r t)))`, interval `log((1-c)(e**(-r a)-e**(-r b)))`.

    ``F(t) = 1 - S(t) = (1-c)(1 - e**(-r t))`` and
    ``S(a) - S(b) = (1-c)(e**(-r a) - e**(-r b))``; ``c=0.3``, ``r=0.5``.
    """
    data = flag_data(["left", "interval"], [2.0, 1.0], [NAN, 3.0])
    cure, r = r_(0.3, 0.5)
    expected = [
        np.log1p(-cure) + np.log(-np.expm1(-r * 2.0)),
        np.log1p(-cure) - r * 1.0 + np.log(-np.expm1(-r * 2.0)),
    ]
    out = ll(fam.mixture_cure()(cure=f(0.3), rate=f(0.5)), data)
    assert_close(out, expected)


# ------------------------------------------------------------- zero mass (7.1)


@FEATURE
def test_interval_beyond_bounded_support_is_minus_infinity():
    """spec 7.1 and 12.1, rev 51: a valid interval with zero model mass and `Z=1` is `-inf`, not NaN.

    ``bounded_support`` (uniform on ``(0, 2]``): ``(3, 4]`` has
    ``S(3) - S(4) = 0``. ``(1, 3]`` has ``S(1) - S(3) = 1/2``, and left at
    ``3`` has ``F(3) = 1`` (log ``0``).
    """
    data = flag_data(["interval", "interval", "left"], [3.0, 1.0, 3.0], [4.0, 3.0, NAN])
    out = ll(fam.bounded_support()(end=f(2.0)), data)
    assert out[0] == -INF
    assert_close(out[1:], [np.log(0.5), 0.0])
    assert not np.asarray(data.issues.invalid_rows).any()


@FEATURE
def test_overflowing_interval_contribution_is_minus_infinity():
    """spec 7.1, rev 51: a valid row whose log contribution is below the finite range is `-inf` ("the IEEE overflow of the true value"), not NaN.

    Weibull ``k=2``, ``b=1``, interval ``(a, 2a]``: the contribution is
    ``-a**2 + log(1 - exp(-3 a**2)) ~ -a**2``. ``a = 1e200`` (float64) and
    ``a = 1e20`` (float32) make ``a**2`` exceed the dtype's maximum while
    ``a`` and ``2a`` are finite. Row 1 (left at ``1``) keeps
    ``log(-expm1(-1))``.
    """
    a = 1e200 if X64 else 1e20
    data = flag_data(["interval", "left"], [a, 1.0], [2.0 * a, NAN])
    out = ll(survjax.Weibull(shape=f(2.0), scale=f(1.0)), data)
    assert out[0] == -INF
    assert_close(out[1], left(1.0))
