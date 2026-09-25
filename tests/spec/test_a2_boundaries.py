"""A2: boundary rules for times, censoring and truncation (spec 2, 3.3).

The eager constructor raises on invalid rows (spec 6.2) and accepts valid ones.
Which DataIssues field reports each rule is tested in A3.

Right censoring at 0 without truncation and ``from_bounds(0, inf)`` are not
tested: sections 2 and 3.3 disagree about them (API_ASSUMPTIONS Q-008).
"""

import numpy as np
import pytest
from helpers import INF, NAN, REJECTED, assert_fields, b, f

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
    """spec 3.3, rev 4: zero exact times are invalid; no epsilon is added."""
    with pytest.raises(REJECTED):
        _exact(0.0)


@pytest.mark.step("A2")
def test_negative_exact_time_is_rejected():
    """spec 3.3, rev 4: negative exact times are invalid."""
    with pytest.raises(REJECTED):
        _exact(-1.0)


@pytest.mark.step("A2")
def test_infinite_exact_time_is_rejected():
    """spec 3.3, rev 4: infinite exact times are invalid."""
    with pytest.raises(REJECTED):
        _exact(INF)


@pytest.mark.step("A2")
def test_nan_exact_time_is_rejected():
    """spec 3.1, rev 4: missing required values are invalid; no implicit dropping."""
    with pytest.raises(REJECTED):
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
    """spec 3.3, rev 4: censoring endpoints are nonnegative."""
    with pytest.raises(REJECTED):
        _right(-1.0)


@pytest.mark.step("A2")
def test_negative_left_censoring_time_is_rejected():
    """spec 3.3, rev 4: censoring endpoints are nonnegative."""
    with pytest.raises(REJECTED):
        _left(-1.0)


@pytest.mark.step("A2")
def test_negative_interval_lower_endpoint_is_rejected():
    """spec 3.3, rev 4: censoring endpoints are nonnegative."""
    with pytest.raises(REJECTED):
        _interval(-1.0, 2.0)


@pytest.mark.step("A2")
def test_right_censoring_at_infinity_is_rejected():
    """spec 3.3, rev 4: in the flag constructor, right censoring at +inf is invalid."""
    with pytest.raises(REJECTED):
        _right(INF)


@pytest.mark.step("A2")
def test_left_censoring_at_infinity_is_rejected():
    """spec 3.3, rev 4: in the flag constructor, left censoring at +inf is invalid."""
    with pytest.raises(REJECTED):
        _left(INF)


@pytest.mark.step("A2")
def test_nan_right_censoring_time_is_rejected():
    """spec 3.1, rev 4: missing required values are invalid."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, NAN]), b([True, False]))


@pytest.mark.step("A2")
def test_left_censoring_at_zero_is_rejected():
    """spec 2/7.1, rev 4: left censoring at 0 is the empty set (0, 0], which is invalid."""
    with pytest.raises(REJECTED):
        _left(0.0)


@pytest.mark.step("A2")
def test_interval_with_infinite_upper_endpoint_is_accepted():
    """spec 3.3, rev 4: `upper` may be +inf."""
    assert_fields(_interval(1.0, INF), lower=[1.0], upper=[INF])


# -------------------------------------------------------------- truncation


@pytest.mark.step("A2")
def test_negative_left_truncation_is_rejected():
    """spec 3.3, rev 4: lower truncation must be nonnegative."""
    with pytest.raises(REJECTED):
        _exact(1.0, left_trunc=-0.5)


@pytest.mark.step("A2")
def test_infinite_left_truncation_is_rejected():
    """spec 3.3, rev 4: lower truncation must be finite."""
    with pytest.raises(REJECTED):
        _exact(1.0, left_trunc=INF)


@pytest.mark.step("A2")
def test_nan_left_truncation_is_rejected():
    """spec 3.2, rev 4: NaN truncation bounds are invalid."""
    with pytest.raises(REJECTED):
        _exact(1.0, left_trunc=NAN)


@pytest.mark.step("A2")
def test_nan_right_truncation_is_rejected():
    """spec 3.2, rev 4: NaN truncation bounds are invalid."""
    with pytest.raises(REJECTED):
        _exact(1.0, right_trunc=NAN)


@pytest.mark.step("A2")
def test_equal_truncation_bounds_are_rejected():
    """spec 3.3, rev 4: L<U is required."""
    with pytest.raises(REJECTED):
        _right(1.0, left_trunc=2.0, right_trunc=2.0)


@pytest.mark.step("A2")
def test_reversed_truncation_bounds_are_rejected():
    """spec 3.3, rev 4: L<U is required."""
    with pytest.raises(REJECTED):
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
    """spec 3.3, rev 4: truncation membership is L<t<=U, so t=L is outside."""
    with pytest.raises(REJECTED):
        _exact(1.0, left_trunc=1.0)


@pytest.mark.step("A2")
def test_exact_time_below_left_truncation_is_rejected():
    """spec 3.3, rev 4: truncation membership is L<t<=U."""
    with pytest.raises(REJECTED):
        _exact(0.5, left_trunc=1.0)


@pytest.mark.step("A2")
def test_exact_time_equal_to_right_truncation_is_accepted():
    """spec 3.3, rev 4: truncation membership is L<t<=U, so t=U is inside."""
    assert_fields(_exact(3.0, left_trunc=1.0, right_trunc=3.0), exact=[True], lower=[3.0])


@pytest.mark.step("A2")
def test_exact_time_above_right_truncation_is_rejected():
    """spec 3.3, rev 4: truncation membership is L<t<=U."""
    with pytest.raises(REJECTED):
        _exact(3.5, right_trunc=3.0)


@pytest.mark.step("A2")
def test_exact_time_strictly_inside_window_is_accepted():
    """spec 3.3, rev 4: truncation membership is L<t<=U."""
    assert_fields(_exact(2.0, left_trunc=1.0, right_trunc=3.0), exact=[True], lower=[2.0])


# ------------------------------------------------- censored before entry


@pytest.mark.step("A2")
def test_right_censoring_before_entry_is_rejected():
    """spec 2, rev 4: right-censored at c<=L is invalid (censored_before_entry); here c<L."""
    with pytest.raises(REJECTED):
        _right(0.5, left_trunc=1.0)


@pytest.mark.step("A2")
def test_right_censoring_at_entry_is_rejected():
    """spec 2, rev 4: right-censored at c<=L is invalid (censored_before_entry); here c=L>0."""
    with pytest.raises(REJECTED):
        _right(1.0, left_trunc=1.0)


@pytest.mark.step("A2")
def test_right_censoring_after_entry_is_accepted():
    """spec 2, rev 4: partial overlap with the truncation window is allowed (L < c < U)."""
    assert_fields(_right(2.0, left_trunc=1.0, right_trunc=5.0), lower=[2.0], upper=[INF])


# ----------------------------------------------------- empty intersection


@pytest.mark.step("A2")
def test_right_censoring_at_upper_truncation_is_rejected():
    """spec 2, rev 4: empty intersection, right-censored at c>=U (here c=U), is invalid."""
    with pytest.raises(REJECTED):
        _right(3.0, right_trunc=3.0)


@pytest.mark.step("A2")
def test_right_censoring_beyond_upper_truncation_is_rejected():
    """spec 2, rev 4: empty intersection, right-censored at c>=U (here c>U), is invalid."""
    with pytest.raises(REJECTED):
        _right(4.0, right_trunc=3.0)


@pytest.mark.step("A2")
def test_left_censoring_at_left_truncation_is_rejected():
    """spec 2, rev 4: empty intersection, (0, c] with c=L misses (L, U], is invalid."""
    with pytest.raises(REJECTED):
        _left(1.0, left_trunc=1.0)


@pytest.mark.step("A2")
def test_left_censoring_before_left_truncation_is_rejected():
    """spec 2, rev 4: empty intersection, (0, c] with c<L, is invalid."""
    with pytest.raises(REJECTED):
        _left(0.5, left_trunc=1.0)


@pytest.mark.step("A2")
def test_interval_ending_at_left_truncation_is_rejected():
    """spec 2, rev 4: empty intersection, an interval ending at or before L, is invalid."""
    with pytest.raises(REJECTED):
        _interval(0.5, 1.0, left_trunc=1.0)


@pytest.mark.step("A2")
def test_interval_starting_at_right_truncation_is_rejected():
    """spec 2, rev 4: empty intersection, (a, b] with a>=U, is invalid."""
    with pytest.raises(REJECTED):
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
    """spec 3.1/6.2, rev 4: no implicit dropping; the eager constructor raises rather than skip a bad row."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 0.0, 2.0]), b(np.array([True, True, False])))
