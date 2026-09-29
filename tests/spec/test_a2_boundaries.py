"""A2: boundary rules for times, censoring and truncation (spec 2, 3.3).

The eager constructor raises on invalid rows (spec 6.2) and accepts valid ones.
Which DataIssues field reports each rule is tested in A3.
"""

import numpy as np
import pytest
from helpers import INF, INVALID_DATA, NAN, assert_fields, b, f

import chronox_jax as cx


def _exact(t, **trunc):
    return cx.SurvivalData(f([t]), b([True]), **{k: f([v]) for k, v in trunc.items()})


def _right(c, **trunc):
    return cx.SurvivalData(f([c]), right_censored=b([True]), **{k: f([v]) for k, v in trunc.items()})


def _left(c, **trunc):
    return cx.SurvivalData(f([c]), left_censored=b([True]), **{k: f([v]) for k, v in trunc.items()})


def _interval(a, c, **trunc):
    return cx.SurvivalData(
        f([a]),
        interval_censored=b([True]),
        time2=f([c]),
        **{k: f([v]) for k, v in trunc.items()},
    )


# ------------------------------------------------------------ exact times


@pytest.mark.step("A2")
def test_exact_time_zero_is_rejected():
    """spec 3.3, rev 7: zero exact times are invalid; no epsilon is added."""
    with pytest.raises(INVALID_DATA):
        _exact(0.0)


@pytest.mark.step("A2")
def test_negative_exact_time_is_rejected():
    """spec 3.3, rev 7: negative exact times are invalid."""
    with pytest.raises(INVALID_DATA):
        _exact(-1.0)


@pytest.mark.step("A2")
def test_infinite_exact_time_is_rejected():
    """spec 3.3, rev 7: infinite exact times are invalid."""
    with pytest.raises(INVALID_DATA):
        _exact(INF)


@pytest.mark.step("A2")
def test_nan_exact_time_is_rejected():
    """spec 3.1, rev 7: missing required values are invalid; no implicit dropping."""
    with pytest.raises(INVALID_DATA):
        cx.SurvivalData(f([1.0, NAN]))


@pytest.mark.step("A2")
def test_tiny_positive_exact_time_is_kept_unchanged():
    """spec 3.3, rev 4: exact times must be strictly positive; a tiny positive time is valid and unaltered."""
    tiny = 1e-30  # a normal number in float32 and float64
    data = _exact(tiny)
    assert_fields(data, lower=f([tiny]), upper=f([tiny]))


# ------------------------------------------------------ censoring endpoints


@pytest.mark.step("A2")
def test_negative_right_censoring_time_is_rejected():
    """spec 3.3, rev 7: censoring endpoints are nonnegative."""
    with pytest.raises(INVALID_DATA):
        _right(-1.0)


@pytest.mark.step("A2")
def test_negative_left_censoring_time_is_rejected():
    """spec 3.3, rev 7: censoring endpoints are nonnegative."""
    with pytest.raises(INVALID_DATA):
        _left(-1.0)


@pytest.mark.step("A2")
def test_negative_interval_lower_endpoint_is_rejected():
    """spec 3.3, rev 7: censoring endpoints are nonnegative."""
    with pytest.raises(INVALID_DATA):
        _interval(-1.0, 2.0)


@pytest.mark.step("A2")
def test_right_censoring_at_infinity_is_rejected():
    """spec 3.3, rev 7: in the flag constructor, right censoring at +inf is invalid."""
    with pytest.raises(INVALID_DATA):
        _right(INF)


@pytest.mark.step("A2")
def test_left_censoring_at_infinity_is_rejected():
    """spec 3.3, rev 7: in the flag constructor, left censoring at +inf is invalid."""
    with pytest.raises(INVALID_DATA):
        _left(INF)


@pytest.mark.step("A2")
def test_nan_right_censoring_time_is_rejected():
    """spec 3.1, rev 7: missing required values are invalid."""
    with pytest.raises(INVALID_DATA):
        cx.SurvivalData(f([1.0, NAN]), b([True, False]))


@pytest.mark.step("A2")
def test_left_censoring_at_zero_is_rejected():
    """spec 2/7.1, rev 7: left censoring at 0 is the empty set (0, 0], which is invalid."""
    with pytest.raises(INVALID_DATA):
        _left(0.0)


@pytest.mark.step("A2")
def test_interval_with_infinite_upper_endpoint_is_accepted():
    """spec 3.3, rev 4: `upper` may be +inf."""
    assert_fields(_interval(1.0, INF), lower=[1.0], upper=[INF])


# -------------------------------------------------------------- truncation


@pytest.mark.step("A2")
def test_negative_left_truncation_is_rejected():
    """spec 3.3, rev 7: lower truncation must be nonnegative."""
    with pytest.raises(INVALID_DATA):
        _exact(1.0, left_trunc=-0.5)


@pytest.mark.step("A2")
def test_infinite_left_truncation_is_rejected():
    """spec 3.3, rev 7: lower truncation must be finite."""
    with pytest.raises(INVALID_DATA):
        _exact(1.0, left_trunc=INF)


@pytest.mark.step("A2")
def test_nan_left_truncation_is_rejected():
    """spec 3.2, rev 7: NaN truncation bounds are invalid."""
    with pytest.raises(INVALID_DATA):
        _exact(1.0, left_trunc=NAN)


@pytest.mark.step("A2")
def test_nan_right_truncation_is_rejected():
    """spec 3.2, rev 7: NaN truncation bounds are invalid."""
    with pytest.raises(INVALID_DATA):
        _exact(1.0, right_trunc=NAN)


@pytest.mark.step("A2")
def test_equal_truncation_bounds_are_rejected():
    """spec 3.3, rev 7: L<U is required."""
    with pytest.raises(INVALID_DATA):
        _right(1.0, left_trunc=2.0, right_trunc=2.0)


@pytest.mark.step("A2")
def test_reversed_truncation_bounds_are_rejected():
    """spec 3.3, rev 7: L<U is required."""
    with pytest.raises(INVALID_DATA):
        _right(1.0, left_trunc=3.0, right_trunc=2.0)


@pytest.mark.step("A2")
def test_infinite_right_truncation_is_accepted():
    """spec 3.3, rev 4: upper truncation may be +inf."""
    assert_fields(_exact(1.0, left_trunc=0.5, right_trunc=INF), right_trunc=[INF])


@pytest.mark.step("A2")
def test_explicit_zero_left_truncation_is_accepted():
    """spec 3.3, rev 4: lower truncation must be finite and nonnegative, so 0 is valid."""
    assert_fields(_exact(1.0, left_trunc=0.0), left_trunc=[0.0])


# ------------------------------------------------------- L < t <= U membership


@pytest.mark.step("A2")
def test_exact_time_equal_to_left_truncation_is_rejected():
    """spec 3.3, rev 7: truncation membership is L<t<=U, so t=L is outside."""
    with pytest.raises(INVALID_DATA):
        _exact(1.0, left_trunc=1.0)


@pytest.mark.step("A2")
def test_exact_time_below_left_truncation_is_rejected():
    """spec 3.3, rev 7: truncation membership is L<t<=U."""
    with pytest.raises(INVALID_DATA):
        _exact(0.5, left_trunc=1.0)


@pytest.mark.step("A2")
def test_exact_time_equal_to_right_truncation_is_accepted():
    """spec 3.3, rev 4: truncation membership is L<t<=U, so t=U is inside."""
    assert_fields(_exact(3.0, left_trunc=1.0, right_trunc=3.0), exact=[True], lower=[3.0])


@pytest.mark.step("A2")
def test_exact_time_above_right_truncation_is_rejected():
    """spec 3.3, rev 7: truncation membership is L<t<=U."""
    with pytest.raises(INVALID_DATA):
        _exact(3.5, right_trunc=3.0)


@pytest.mark.step("A2")
def test_exact_time_strictly_inside_window_is_accepted():
    """spec 3.3, rev 4: truncation membership is L<t<=U."""
    assert_fields(_exact(2.0, left_trunc=1.0, right_trunc=3.0), exact=[True], lower=[2.0])


# ------------------------------------------------- censored before entry


@pytest.mark.step("A2")
def test_right_censoring_before_entry_is_rejected():
    """spec 2, rev 7: right-censored before entry (c<L) is invalid (censored_before_entry)."""
    with pytest.raises(INVALID_DATA):
        _right(0.5, left_trunc=1.0)


@pytest.mark.step("A2")
def test_right_censoring_at_entry_is_accepted():
    """spec 2/3.3, rev 5: right-censored at entry (c=L>0) is valid (uninformative)."""
    assert_fields(_right(1.0, left_trunc=1.0), exact=[False], lower=[1.0], upper=[INF])


@pytest.mark.step("A2")
def test_right_censoring_at_zero_without_truncation_is_accepted():
    """spec 3.3, rev 5: `(c, +inf]` with c=L=0 is valid (uninformative)."""
    assert_fields(_right(0.0), exact=[False], lower=[0.0], upper=[INF], left_trunc=[0.0])


@pytest.mark.step("A2")
def test_right_censoring_at_entry_with_finite_right_truncation_is_accepted():
    """spec 3.3, rev 5: `(c, +inf]` with c=L is valid including finite right truncation."""
    data = _right(1.0, left_trunc=1.0, right_trunc=3.0)
    assert_fields(data, exact=[False], lower=[1.0], upper=[INF], right_trunc=[3.0])


@pytest.mark.step("A2")
def test_interval_with_infinite_time2_starting_before_entry_is_rejected():
    """spec 3.1, rev 7: an interval-flag row with `time2=+inf` and `time<L` is invalid (censored_before_entry)."""
    with pytest.raises(INVALID_DATA):
        _interval(0.5, INF, left_trunc=1.0)


@pytest.mark.step("A2")
def test_interval_with_infinite_time2_starting_at_entry_is_accepted():
    """spec 3.1, rev 5: an interval-flag row with `time2=+inf` and `time=L` is valid (uninformative)."""
    assert_fields(_interval(1.0, INF, left_trunc=1.0), exact=[False], lower=[1.0], upper=[INF])


@pytest.mark.step("A2")
def test_interval_with_infinite_time2_at_entry_with_finite_right_truncation_is_accepted():
    """spec 3.1, rev 5: `time2=+inf` with `time=L` is valid for any valid truncation window (here finite U)."""
    data = _interval(1.0, INF, left_trunc=1.0, right_trunc=3.0)
    assert_fields(data, exact=[False], lower=[1.0], upper=[INF], right_trunc=[3.0])


@pytest.mark.step("A2")
def test_bounds_zero_to_infinity_without_truncation_is_accepted():
    """spec 3.3, rev 5: `from_bounds(0, inf)` is uninformative (valid) when L=0."""
    data = cx.SurvivalData.from_bounds(f([0.0]), f([INF]))
    assert_fields(data, exact=[False], lower=[0.0], upper=[INF])


@pytest.mark.step("A2")
def test_bounds_zero_to_infinity_with_left_truncation_is_rejected():
    """spec 3.3, rev 7: `from_bounds(0, inf)` is invalid when L>0."""
    with pytest.raises(INVALID_DATA):
        cx.SurvivalData.from_bounds(f([0.0]), f([INF]), left_trunc=f([1.0]))


@pytest.mark.step("A2")
def test_bounds_right_censoring_before_entry_is_rejected():
    """spec 2, rev 7: c<L is invalid for any censored row with upper +inf, however it was constructed."""
    with pytest.raises(INVALID_DATA):
        cx.SurvivalData.from_bounds(f([0.5]), f([INF]), left_trunc=f([1.0]))


@pytest.mark.step("A2")
def test_bounds_right_censoring_at_entry_is_accepted():
    """spec 2/3.3, rev 5: `from_bounds(L, inf)` is right censoring at entry, which is valid."""
    data = cx.SurvivalData.from_bounds(f([1.0]), f([INF]), left_trunc=f([1.0]))
    assert_fields(data, exact=[False], lower=[1.0], upper=[INF], left_trunc=[1.0])


@pytest.mark.step("A2")
def test_right_censoring_after_entry_is_accepted():
    """spec 2, rev 4: partial overlap with the truncation window is allowed (L < c < U)."""
    assert_fields(_right(2.0, left_trunc=1.0, right_trunc=5.0), lower=[2.0], upper=[INF])


# ----------------------------------------------------- empty intersection


@pytest.mark.step("A2")
def test_right_censoring_at_upper_truncation_is_rejected():
    """spec 2, rev 7: empty intersection, right-censored at c>=U (here c=U), is invalid."""
    with pytest.raises(INVALID_DATA):
        _right(3.0, right_trunc=3.0)


@pytest.mark.step("A2")
def test_right_censoring_beyond_upper_truncation_is_rejected():
    """spec 2, rev 7: empty intersection, right-censored at c>=U (here c>U), is invalid."""
    with pytest.raises(INVALID_DATA):
        _right(4.0, right_trunc=3.0)


@pytest.mark.step("A2")
def test_left_censoring_at_left_truncation_is_rejected():
    """spec 2, rev 7: empty intersection, (0, c] with c=L misses (L, U], is invalid."""
    with pytest.raises(INVALID_DATA):
        _left(1.0, left_trunc=1.0)


@pytest.mark.step("A2")
def test_left_censoring_before_left_truncation_is_rejected():
    """spec 2, rev 7: empty intersection, (0, c] with c<L, is invalid."""
    with pytest.raises(INVALID_DATA):
        _left(0.5, left_trunc=1.0)


@pytest.mark.step("A2")
def test_interval_ending_at_left_truncation_is_rejected():
    """spec 2, rev 7: empty intersection, an interval ending at or before L, is invalid."""
    with pytest.raises(INVALID_DATA):
        _interval(0.5, 1.0, left_trunc=1.0)


@pytest.mark.step("A2")
def test_interval_starting_at_right_truncation_is_rejected():
    """spec 2, rev 7: empty intersection, (a, b] with a>=U, is invalid."""
    with pytest.raises(INVALID_DATA):
        _interval(3.0, 4.0, right_trunc=3.0)


# ---------------------------------------------- partial and full coverage


@pytest.mark.step("A2")
def test_left_censoring_inside_window_is_accepted():
    """spec 2, rev 4: partial overlap with the truncation window is allowed (L < c < U)."""
    assert_fields(_left(2.0, left_trunc=1.0, right_trunc=5.0), lower=[0.0], upper=[2.0])


@pytest.mark.step("A2")
def test_interval_straddling_left_truncation_is_accepted():
    """spec 2, rev 4: partial overlap with the truncation window is allowed."""
    assert_fields(_interval(0.5, 2.0, left_trunc=1.0), lower=[0.5], upper=[2.0])


@pytest.mark.step("A2")
def test_interval_straddling_right_truncation_is_accepted():
    """spec 2, rev 4: partial overlap with the truncation window is allowed."""
    assert_fields(_interval(2.0, 4.0, right_trunc=3.0), lower=[2.0], upper=[4.0])


@pytest.mark.step("A2")
def test_left_censoring_at_upper_truncation_is_accepted():
    """spec 2, rev 4: left-censored at c>=U covers the whole window and is valid (uninformative); here c=U."""
    assert_fields(_left(3.0, right_trunc=3.0), exact=[False], lower=[0.0], upper=[3.0])


@pytest.mark.step("A2")
def test_left_censoring_beyond_upper_truncation_is_accepted():
    """spec 2, rev 4: left-censored at c>=U covers the whole window and is valid (uninformative); here c>U."""
    assert_fields(_left(4.0, right_trunc=3.0), exact=[False], lower=[0.0], upper=[4.0])


@pytest.mark.step("A2")
def test_interval_containing_window_is_accepted():
    """spec 2, rev 4: an interval containing (L, U] is valid (uninformative)."""
    assert_fields(_interval(0.5, 4.0, left_trunc=1.0, right_trunc=3.0), lower=[0.5], upper=[4.0])


@pytest.mark.step("A2")
def test_interval_equal_to_window_is_accepted():
    """spec 2, rev 4: an interval containing (L, U], here equal to it, is valid (uninformative)."""
    assert_fields(_interval(1.0, 3.0, left_trunc=1.0, right_trunc=3.0), lower=[1.0], upper=[3.0])


@pytest.mark.step("A2")
def test_one_invalid_row_rejects_the_whole_eager_construction():
    """spec 3.1/6.2, rev 7: no implicit dropping; the eager constructor raises rather than skip a bad row."""
    with pytest.raises(INVALID_DATA):
        cx.SurvivalData(f([1.0, 0.0, 2.0]), b(np.array([True, True, False])))
