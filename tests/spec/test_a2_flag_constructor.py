"""A2: the flag constructor ``SurvivalData(time, event, ...)`` (spec 3.1, 3.2)."""

import numpy as np
import pytest
from helpers import (
    INF,
    NAN,
    REJECTED,
    UNSUPPORTED,
    assert_fields,
    assert_same_fields,
    b,
    f,
)

import chronox_jax as cx

# ------------------------------------------------------- event indicator


@pytest.mark.step("A2")
def test_event_true_is_exact_and_false_is_right_censored():
    """spec 3.1, rev 4: `SurvivalData(time, event)` uses true for exact events and false for right censoring."""
    data = cx.SurvivalData(f([1.0, 2.0]), b([True, False]))
    assert_fields(data, exact=[True, False], lower=[1.0, 2.0], upper=[1.0, INF])


@pytest.mark.step("A2")
def test_second_positional_argument_is_event_not_censoring_flag():
    """spec 3.1, rev 4: the second positional argument is the event indicator, not a censoring flag."""
    data = cx.SurvivalData(f([1.0]), b([True]))
    assert_fields(data, exact=[True], upper=[1.0])


@pytest.mark.step("A2")
def test_integer_event_codes_zero_and_one_are_accepted():
    """spec 3.1, rev 4: indicators accept numbers exactly equal to 0 or 1."""
    data = cx.SurvivalData(f([1.0, 2.0]), np.array([1, 0], dtype=np.int32))
    assert_fields(data, exact=[True, False], upper=[1.0, INF])


@pytest.mark.step("A2")
def test_float_event_values_zero_and_one_are_accepted():
    """spec 3.1, rev 4: indicators accept numbers exactly equal to 0 or 1, including floats."""
    data = cx.SurvivalData(f([1.0, 2.0]), f([1.0, 0.0]))
    assert_fields(data, exact=[True, False], upper=[1.0, INF])


@pytest.mark.step("A2")
def test_omitting_event_and_flags_makes_every_row_exact():
    """spec 3.1, rev 4: omitting `event` and all censoring flags means every row is exact."""
    data = cx.SurvivalData(f([1.0, 2.0, 3.0]))
    assert_fields(data, exact=[True, True, True], lower=[1.0, 2.0, 3.0], upper=[1.0, 2.0, 3.0])


# ------------------------------------------------------- explicit flags


@pytest.mark.step("A2")
def test_right_censored_flag_selects_right_censoring():
    """spec 3.1, rev 4: explicit right flags select the row's kind."""
    data = cx.SurvivalData(f([1.0, 2.0]), right_censored=b([False, True]))
    assert_fields(data, exact=[True, False], lower=[1.0, 2.0], upper=[1.0, INF])


@pytest.mark.step("A2")
def test_left_censored_flag_selects_left_censoring():
    """spec 3.1, rev 4: explicit left flags select the row's kind; `time` is the censoring time."""
    data = cx.SurvivalData(f([1.0, 2.0]), left_censored=b([False, True]))
    assert_fields(data, exact=[True, False], lower=[1.0, 0.0], upper=[1.0, 2.0])


@pytest.mark.step("A2")
def test_interval_flag_uses_time_and_time2_as_endpoints():
    """spec 3.1, rev 4: `time` is an interval's lower endpoint, `time2` its upper endpoint."""
    data = cx.SurvivalData(f([1.5]), interval_censored=b([True]), time2=f([2.5]))
    assert_fields(data, exact=[False], lower=[1.5], upper=[2.5])


@pytest.mark.step("A2")
def test_all_flags_false_row_is_exact():
    """spec 3.1, rev 4: with all flags false a row is exact."""
    data = cx.SurvivalData(
        f([1.0]),
        right_censored=b([False]),
        left_censored=b([False]),
        interval_censored=b([False]),
        time2=f([NAN]),
    )
    assert_fields(data, exact=[True], lower=[1.0], upper=[1.0])


@pytest.mark.step("A2")
def test_integer_flags_zero_and_one_are_accepted():
    """spec 3.1, rev 4: indicators accept numbers exactly equal to 0 or 1 (censoring flags)."""
    data = cx.SurvivalData(f([1.0, 2.0]), left_censored=np.array([0, 1], dtype=np.int32))
    assert_fields(data, exact=[True, False], lower=[1.0, 0.0], upper=[1.0, 2.0])


@pytest.mark.step("A2")
def test_interval_with_infinite_time2_normalizes_as_right_censoring():
    """spec 3.1, rev 5: an interval-flag row with `time2=+inf` normalizes to right censoring at `c=time`."""
    interval = cx.SurvivalData(f([2.0]), interval_censored=b([True]), time2=f([INF]))
    right = cx.SurvivalData(f([2.0]), right_censored=b([True]))
    assert_same_fields(interval, right)


@pytest.mark.step("A2")
def test_interval_with_zero_time_normalizes_as_left_censoring():
    """spec 3.2, rev 4: kinds are derived views; interval (0, c] normalizes like left censoring at c."""
    interval = cx.SurvivalData(f([0.0]), interval_censored=b([True]), time2=f([3.0]))
    left = cx.SurvivalData(f([3.0]), left_censored=b([True]))
    assert_same_fields(interval, left)


@pytest.mark.step("A2")
def test_event_form_and_right_flag_form_normalize_identically():
    """spec 3.1, rev 4: event=false and right_censored=true both mean right censoring."""
    with_event = cx.SurvivalData(f([1.0, 2.0]), b([True, False]))
    with_flag = cx.SurvivalData(f([1.0, 2.0]), right_censored=b([False, True]))
    assert_same_fields(with_event, with_flag)


# -------------------------------------------------- rejected combinations


@pytest.mark.step("A2")
def test_event_with_all_false_right_flag_is_rejected():
    """spec 3.1, rev 4: `event` together with any non-None censoring flag is rejected, even if all false."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), b([True, False]), right_censored=b([False, False]))


@pytest.mark.step("A2")
def test_event_with_all_false_left_flag_is_rejected():
    """spec 3.1, rev 4: `event` together with any non-None censoring flag is rejected, even if all false."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), b([True, False]), left_censored=b([False, False]))


@pytest.mark.step("A2")
def test_event_with_all_false_interval_flag_is_rejected():
    """spec 3.1, rev 4: `event` together with any non-None censoring flag is rejected, even if all false."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(
            f([1.0, 2.0]),
            b([True, False]),
            interval_censored=b([False, False]),
            time2=f([NAN, NAN]),
        )


@pytest.mark.step("A2")
def test_right_and_left_flags_on_one_row_are_rejected():
    """spec 3.1, rev 4: at most one flag may be true per row."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), right_censored=b([False, True]), left_censored=b([False, True]))


@pytest.mark.step("A2")
def test_left_and_interval_flags_on_one_row_are_rejected():
    """spec 3.1, rev 4: at most one flag may be true per row."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(
            f([1.0]), left_censored=b([True]), interval_censored=b([True]), time2=f([2.0])
        )


# ---------------------------------------------------------- interval rows


@pytest.mark.step("A2")
def test_interval_with_equal_endpoints_is_rejected_not_made_exact():
    """spec 3.1, rev 4: kind is never inferred from equal endpoints; interval rows require time < time2."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([2.0]), interval_censored=b([True]), time2=f([2.0]))


@pytest.mark.step("A2")
def test_interval_with_reversed_endpoints_is_rejected():
    """spec 3.1, rev 4: interval rows require time < time2."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([3.0]), interval_censored=b([True]), time2=f([2.0]))


@pytest.mark.step("A2")
def test_interval_without_time2_is_rejected():
    """spec 3.1, rev 4: `time2` is required for interval rows; missing required values are invalid."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0]), interval_censored=b([True]))


@pytest.mark.step("A2")
def test_interval_with_nan_time2_is_rejected():
    """spec 3.1, rev 4: missing required values are invalid; there is no implicit dropping or imputation."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 1.0]), interval_censored=b([False, True]), time2=f([NAN, NAN]))


# ------------------------------------------------------ indicator values


@pytest.mark.step("A2")
def test_event_code_two_is_rejected():
    """spec 3.1, rev 4: other indicator codes are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), np.array([1, 2], dtype=np.int32))


@pytest.mark.step("A2")
def test_r_style_one_two_event_coding_is_not_inferred():
    """spec 3.1, rev 4: R-style 1/2 event coding is not inferred; the 2s are invalid codes."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0, 3.0]), np.array([1, 2, 2], dtype=np.int32))


@pytest.mark.step("A2")
def test_negative_event_code_is_rejected():
    """spec 3.1, rev 4: indicators accept only numbers exactly equal to 0 or 1."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), np.array([1, -1], dtype=np.int32))


@pytest.mark.step("A2")
def test_fractional_event_value_is_rejected():
    """spec 3.1, rev 4: indicators accept only numbers exactly equal to 0 or 1."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), f([1.0, 0.5]))


@pytest.mark.step("A2")
def test_nan_event_is_rejected():
    """spec 3.1, rev 4: missing values and NaN indicators are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), f([1.0, NAN]))


@pytest.mark.step("A2")
def test_infinite_event_is_rejected():
    """spec 3.1, rev 4: infinite indicators are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), f([1.0, INF]))


@pytest.mark.step("A2")
def test_string_event_is_rejected():
    """spec 3.1, rev 4: string indicators are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), np.array(["1", "0"]))


@pytest.mark.step("A2")
def test_object_event_with_none_is_rejected():
    """spec 3.1, rev 4: missing indicator values are rejected."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), np.array([1, None], dtype=object))


@pytest.mark.step("A2")
def test_censoring_flag_code_two_is_rejected():
    """spec 3.1, rev 4: censoring flags accept only booleans or numbers exactly 0 or 1."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0]), right_censored=np.array([0, 2], dtype=np.int32))


# ------------------------------------------------------- shapes (3.2)


@pytest.mark.step("A2")
def test_scalar_event_broadcasts_to_time_shape():
    """spec 3.2, rev 4: other fields broadcast to the shape of `time`."""
    data = cx.SurvivalData(f([1.0, 2.0, 3.0]), b(False))
    assert np.shape(data.exact) == (3,)
    assert_fields(data, exact=[False, False, False], upper=[INF, INF, INF])


@pytest.mark.step("A2")
def test_scalar_truncation_broadcasts_to_time_shape():
    """spec 3.2, rev 4: other fields broadcast to the observation shape."""
    data = cx.SurvivalData(f([1.0, 2.0]), left_trunc=f(0.5), right_trunc=f(5.0))
    assert_fields(data, left_trunc=[0.5, 0.5], right_trunc=[5.0, 5.0])


@pytest.mark.step("A2")
def test_trailing_axis_truncation_broadcasts_over_multidimensional_time():
    """spec 3.2, rev 4: fields broadcast to the observation shape of `time` (2, 3)."""
    data = cx.SurvivalData(f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), left_trunc=f([0.5, 0.0, 0.25]))
    assert np.shape(data.left_trunc) == (2, 3)
    assert_fields(data, left_trunc=[[0.5, 0.0, 0.25], [0.5, 0.0, 0.25]])


@pytest.mark.step("A2")
def test_column_event_cannot_enlarge_vector_time():
    """spec 3.2, rev 4: `time` (N,) with flags (N,1) is invalid for N>1."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0, 3.0]), b([[True], [False], [True]]))


@pytest.mark.step("A2")
def test_column_censoring_flag_cannot_enlarge_vector_time():
    """spec 3.2, rev 4: `time` (N,) with flags (N,1) is invalid for N>1."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0, 3.0]), right_censored=b([[True], [False], [True]]))


@pytest.mark.step("A2")
def test_column_truncation_cannot_enlarge_vector_time():
    """spec 3.2, rev 4: every other field must broadcast to the observation shape without enlarging it."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 2.0, 3.0]), left_trunc=f([[0.0], [0.0], [0.0]]))


@pytest.mark.step("A2")
def test_vector_event_cannot_enlarge_scalar_time():
    """spec 3.2, rev 4: the first time argument determines the observation shape; fields cannot enlarge it."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f(1.0), b([True, False]))


@pytest.mark.step("A2")
def test_column_time2_cannot_enlarge_vector_time():
    """spec 3.2, rev 4: `time2` must broadcast to the shape of `time` without enlarging it."""
    with pytest.raises(REJECTED):
        cx.SurvivalData(f([1.0, 1.0]), interval_censored=b([True, True]), time2=f([[2.0], [3.0]]))


# ---------------------------------------------- outside-interval censoring


@pytest.mark.step("A2")
def test_outside_interval_censored_raises_unsupported():
    """spec 3.1, rev 4: `outside_interval_censored` is recognized only to raise an unsupported-feature error."""
    with pytest.raises(UNSUPPORTED):
        cx.SurvivalData(f([1.0, 2.0]), outside_interval_censored=b([False, True]), time2=f([NAN, 3.0]))


@pytest.mark.step("A2")
def test_all_false_outside_interval_censored_still_raises_unsupported():
    """spec 3.1, rev 4: `outside_interval_censored` is not in the normal signature; any use raises unsupported."""
    with pytest.raises(UNSUPPORTED):
        cx.SurvivalData(f([1.0, 2.0]), outside_interval_censored=b([False, False]))
