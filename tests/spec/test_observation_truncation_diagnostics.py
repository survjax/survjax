"""observation.truncation: structural cases, invalid rows, `Z=0` and their diagnosis (spec 2, 3.3, 6.2, 7.1, 12.1, 12.2; rev 51).

Spec 2 structural cases: empty intersections are invalid data;
right-censored before entry (``c<L``) is ``censored_before_entry``;
right-censored at entry (``c=L``) is valid, ``uninformative`` and contributes
"exactly zero, for any valid `L` and `U`"; other censored sets covering the
whole window (left at ``c>=U``, a finite interval containing ``(L,U]``)
contribute exactly zero and are flagged ``uninformative``. 3.3: exact times
need ``L<t<=U``; ``L`` finite and nonnegative, ``L<U``. 7.1: "`Z=0` | NaN";
"Valid observation, true zero model mass or density, positive `Z` | `-inf`";
12.1: "a bounded-support family separating zero numerator (`-inf`) from zero
denominator (NaN)". The triage recipe (``helpers.nan_causes``, Q-064) puts a
``Z=0`` NaN in ``other``.

Each invalid case pairs a valid exact row 0 (Exponential ``r=0.5``, ``t=1``,
window ``(0.5, 3]``: ``log r - r t - log(e**(-r L) - e**(-r U))``) with an
invalid row 1. Masks are named only where the spec names the rule.
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, b, f, flag_data, ll, mask, nan_causes

import survjax

TRUNC = pytest.mark.feature("observation.truncation")
BOTH = pytest.mark.feature("observation.censoring", "observation.truncation")

R = 0.5
L0, U0 = 0.5, 3.0
ROW0 = float(np.log(R) - R * 1.0 - np.log(np.exp(-R * L0) - np.exp(-R * U0)))


def _dist():
    return survjax.Exponential(rate=f(R))


def _assert_row1_invalid(data):
    value = ll(_dist(), data)
    invalid = np.asarray(data.issues.invalid_rows).astype(bool)
    np.testing.assert_array_equal(invalid, [False, True])
    np.testing.assert_array_equal(np.isnan(value), invalid)
    assert_close(value[0], ROW0)
    bad_data, bad_params, other = nan_causes(_dist(), data, value)
    np.testing.assert_array_equal(bad_data, invalid)
    assert not bad_params.any()
    assert not other.any()


def _two(kind, t, t2=NAN, lt=L0, rt=U0):
    """Row 0 valid exact at 1 in ``(0.5, 3]``; row 1 of ``kind`` with its own window."""
    return flag_data(["exact", kind], [1.0, t], [NAN, t2], left_trunc=[L0, lt], right_trunc=[U0, rt])


# ------------------------------------------------------------ uninformative rows


UNINFORMATIVE_EXACT_RIGHT = {
    "right-at-entry": lambda: _two("right", L0),
    "right-at-entry-no-upper": lambda: _two("right", L0, rt=INF),
    "right-at-zero-with-right-truncation": lambda: _two("right", 0.0, lt=0.0),
    # 3.3: "`from_bounds(0, inf)` is uninformative when `L=0`", here with `U=2`.
    "bounds-zero-to-inf-without-entry": lambda: survjax.SurvivalData.from_bounds(
        f([1.0, 0.0]), f([1.0, INF]), exact=b([True, False]), left_trunc=f([L0, 0.0]), right_trunc=f([U0, 2.0])
    ),
}
UNINFORMATIVE_CENSORED = {
    "left-at-upper": lambda: _two("left", U0),
    "left-beyond-upper": lambda: _two("left", 5.0),
    "interval-equal-to-window": lambda: _two("interval", L0, U0),
    "interval-containing-window": lambda: _two("interval", 0.2, 4.0),
    "interval-to-infinity-at-entry": lambda: _two("interval", L0, INF),
}


def _assert_uninformative(data):
    value = ll(_dist(), data)
    assert value[1] == 0.0, value
    assert_close(value[0], ROW0)
    np.testing.assert_array_equal(mask(data.issues, "uninformative"), [False, True])
    assert not np.asarray(data.issues.invalid_rows).any()
    data.issues.raise_if_invalid()


@TRUNC
@pytest.mark.parametrize("case", list(UNINFORMATIVE_EXACT_RIGHT))
def test_right_censored_at_entry_is_exactly_zero_and_uninformative(case):
    """spec 2, 3.3 and 7.1, rev 51: right-censored at `c=L` "contribut[es] exactly zero, for any valid `L` and `U`", flagged `uninformative`, valid."""
    _assert_uninformative(UNINFORMATIVE_EXACT_RIGHT[case]())


@BOTH
@pytest.mark.parametrize("case", list(UNINFORMATIVE_CENSORED))
def test_censored_set_covering_the_window_is_exactly_zero_and_uninformative(case):
    """spec 2 and 7.1, rev 51: left at `c>=U`, finite intervals containing `(L,U]`, and `(L, +inf]` give exactly zero and `uninformative`."""
    _assert_uninformative(UNINFORMATIVE_CENSORED[case]())


@BOTH
def test_covering_rows_are_zero_for_extreme_parameters():
    """spec 7.1, rev 51: zero is "detected structurally rather than by arithmetic cancellation": it holds where `Z` is tiny.

    Exponential ``r = 50``, window ``(10, 12]``: ``Z ~ e**-500``; left at
    ``12`` and the interval ``(5, 20]`` cover the window and are exactly zero.
    """
    data = flag_data(["left", "interval", "right"], [12.0, 5.0, 10.0], [NAN, 20.0, NAN], left_trunc=[10.0] * 3, right_trunc=[12.0] * 3)
    value = ll(survjax.Exponential(rate=f(50.0)), data)
    np.testing.assert_array_equal(value, [0.0, 0.0, 0.0])


# --------------------------------------------------------------- invalid rows

INVALID_EXACT_RIGHT = {
    # 3.3: "Truncation membership is `L<t<=U`".
    "exact-at-entry": (lambda: _two("exact", L0), "exact_outside_truncation"),
    "exact-before-entry": (lambda: _two("exact", 0.2), "exact_outside_truncation"),
    "exact-after-upper": (lambda: _two("exact", 4.0), "exact_outside_truncation"),
    # 2: "Right-censored before entry (`c<L`): invalid data (`censored_before_entry`)".
    "right-before-entry": (lambda: _two("right", 0.2), "censored_before_entry"),
    # 2: "right-censored at `c>=U`" is an empty intersection.
    "right-at-upper": (lambda: _two("right", U0), None),
    "right-beyond-upper": (lambda: _two("right", 4.0), None),
    # 3.3: "Lower truncation must be finite and nonnegative ... `L<U` is required".
    "negative-entry": (lambda: _two("exact", 1.0, lt=-1.0), "invalid_truncation"),
    "infinite-entry": (lambda: _two("exact", 1.0, lt=INF, rt=INF), "invalid_truncation"),
    "nan-entry": (lambda: _two("exact", 1.0, lt=NAN), "invalid_truncation"),
    "nan-upper": (lambda: _two("exact", 1.0, rt=NAN), "invalid_truncation"),
    "entry-equal-upper": (lambda: _two("exact", 1.0, lt=2.0, rt=2.0), "invalid_truncation"),
    "entry-above-upper": (lambda: _two("exact", 1.0, lt=3.0, rt=2.0), "invalid_truncation"),
    # 3.3: "`from_bounds(0, inf)` is ... invalid when `L>0`".
    "bounds-zero-to-inf-after-entry": (
        lambda: survjax.SurvivalData.from_bounds(
            f([1.0, 0.0]), f([1.0, INF]), exact=b([True, False]), left_trunc=f([L0, L0]), right_trunc=f([U0, U0])
        ),
        "censored_before_entry",
    ),
}
INVALID_CENSORED = {
    # 2: "an interval ending at or before `L`" is an empty intersection.
    "interval-ending-at-entry": (lambda: _two("interval", 0.1, L0), None),
    "interval-ending-before-entry": (lambda: _two("interval", 0.1, 0.3), None),
    "interval-starting-at-upper": (lambda: _two("interval", U0, 4.0), None),
    "left-at-entry": (lambda: _two("left", L0), None),
    "left-before-entry": (lambda: _two("left", 0.3), None),
    # 3.1: an interval-flag row with `time2=+inf` and `time<L` is `censored_before_entry`.
    "interval-to-infinity-before-entry": (lambda: _two("interval", 0.2, INF), "censored_before_entry"),
}


@TRUNC
@pytest.mark.parametrize("case", list(INVALID_EXACT_RIGHT))
def test_invalid_truncated_exact_or_right_row_is_nan(case):
    """spec 2, 3.3, 6.2 and 7.1, rev 51: invalid rows are NaN, exactly `invalid_rows`, triaged to data; the valid row keeps its value."""
    build, rule = INVALID_EXACT_RIGHT[case]
    data = build()
    _assert_row1_invalid(data)
    if rule is not None:
        assert mask(data.issues, rule)[1], rule


@BOTH
@pytest.mark.parametrize("case", list(INVALID_CENSORED))
def test_invalid_truncated_censored_row_is_nan(case):
    """spec 2, 6.2 and 7.1, rev 51: empty intersections of left/interval rows with the window are invalid: NaN, triaged to data."""
    build, rule = INVALID_CENSORED[case]
    data = build()
    _assert_row1_invalid(data)
    if rule is not None:
        assert mask(data.issues, rule)[1], rule


@TRUNC
def test_invalid_truncation_under_jit_and_traced_construction():
    """spec 6.2 and 7.1, rev 51: prepared and traced construction give the same NaN row, without exceptions."""
    data = INVALID_EXACT_RIGHT["right-before-entry"][0]()
    value = np.asarray(jax.jit(lambda d: _dist().log_likelihood(d))(data))
    np.testing.assert_array_equal(np.isnan(value), [False, True])

    def run(time, event, lt, rt):
        d = survjax.SurvivalData(time, event, left_trunc=lt, right_trunc=rt)
        return _dist().log_likelihood(d), d.issues.invalid_rows

    value, invalid = jax.jit(run)(f([1.0, 0.2]), b([True, False]), f([L0, L0]), f([U0, U0]))
    np.testing.assert_array_equal(np.asarray(invalid), [False, True])
    np.testing.assert_array_equal(np.isnan(np.asarray(value)), [False, True])
    assert_close(np.asarray(value)[0], ROW0)


@TRUNC
def test_parameter_issue_on_a_truncated_row_is_nan():
    """spec 4.7 and 7.1, rev 51: a declared parameter violation makes its truncated row NaN; triaged to parameters."""
    data = flag_data(["exact", "right", "exact"], [1.0, 1.0, 1.0], left_trunc=[L0] * 3, right_trunc=[U0] * 3)
    dist = survjax.Exponential(rate=f([R, -1.0, R]))
    value = ll(dist, data)
    np.testing.assert_array_equal(np.isnan(value), [False, True, False])
    assert_close(value[[0, 2]], [ROW0, ROW0])
    bad_data, bad_params, other = nan_causes(dist, data, value)
    assert not bad_data.any()
    np.testing.assert_array_equal(bad_params, [False, True, False])
    assert not other.any()


@TRUNC
def test_valid_truncated_batch_has_no_nan_and_no_triage_mask():
    """spec 7.1, rev 51: valid truncated exact/right rows produce no NaN; every recipe mask is empty (Q-064)."""
    data = flag_data(["exact", "right", "right"], [1.0, 1.0, L0], left_trunc=[L0] * 3, right_trunc=[U0] * 3)
    value = ll(_dist(), data)
    assert not np.isnan(value).any()
    for m in nan_causes(_dist(), data, value):
        assert not m.any()


# ---------------------------------------- zero numerator versus zero denominator


@TRUNC
def test_zero_denominator_is_nan_and_triaged_as_other():
    """spec 2, 7.1 and 12.1, rev 51: `Z=0` gives NaN; the data and parameters are valid, so the recipe leaves it in `other`.

    ``bounded_support`` uniform on ``(0, 2]``; window ``(3, 5]`` has
    ``Z = S(3) - S(5) = 0``; exact at ``4`` and right at ``3.5`` are valid
    observations (``L < t <= U``, ``c >= L``).
    """
    data = flag_data(["exact", "right"], [4.0, 3.5], left_trunc=[3.0, 3.0], right_trunc=[5.0, 5.0])
    dist = fam.bounded_support()(end=f(2.0))
    value = ll(dist, data)
    assert np.isnan(value).all()
    assert bool(np.asarray(data.issues.ok))
    bad_data, bad_params, other = nan_causes(dist, data, value)
    assert not bad_data.any()
    assert not bad_params.any()
    np.testing.assert_array_equal(other, [True, True])


@BOTH
def test_bounded_support_separates_zero_numerator_from_zero_denominator():
    """spec 7.1 and 12.1, rev 51: one batch: zero numerator with positive `Z` is `-inf`; zero `Z` is NaN; a positive row is finite.

    Uniform on ``(0, 2]``, so ``S(t) = 1 - t/2`` on ``[0, 2]``:

    - row 0: interval ``(3, 4]`` in window ``(1, 5]``: numerator ``0``,
      ``Z = S(1) = 1/2``: ``-inf``;
    - row 1: interval ``(3.5, 4]`` in window ``(3, 5]``: ``Z = 0``: NaN;
    - row 2: interval ``(1, 1.5]`` in window ``(0.5, 5]``:
      ``(S(1) - S(1.5)) / S(0.5) = (1/4)/(3/4)``: ``log(1/3)``.
    """
    data = flag_data(
        ["interval", "interval", "interval"],
        [3.0, 3.5, 1.0],
        [4.0, 4.0, 1.5],
        left_trunc=[1.0, 3.0, 0.5],
        right_trunc=[5.0, 5.0, 5.0],
    )
    value = ll(fam.bounded_support()(end=f(2.0)), data)
    assert value[0] == -INF
    assert np.isnan(value[1])
    assert_close(value[2], np.log(1.0 / 3.0))
    assert bool(np.asarray(data.issues.ok))
