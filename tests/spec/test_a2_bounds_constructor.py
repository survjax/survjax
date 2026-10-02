"""A2: the bounds constructor ``SurvivalData.from_bounds`` (spec 3.1, 3.2)."""

import numpy as np
import pytest
from helpers import (
    FLOAT,
    INF,
    MISUSE,
    NAN,
    assert_fields,
    assert_same_fields,
    b,
    f,
)
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import survjax

# ------------------------------------------------------------- semantics


@pytest.mark.step("A2")
def test_finite_bounds_row_is_interval_censored():
    """spec 3.1, rev 4: censored rows mean lower < T <= upper."""
    data = survjax.SurvivalData.from_bounds(f([1.0]), f([2.5]))
    assert_fields(data, exact=[False], lower=[1.0], upper=[2.5])


@pytest.mark.step("A2")
def test_infinite_upper_expresses_right_censoring():
    """spec 3.1, rev 4: `upper=+inf` expresses right censoring."""
    data = survjax.SurvivalData.from_bounds(f([2.0]), f([INF]))
    assert_fields(data, exact=[False], lower=[2.0], upper=[INF])


@pytest.mark.step("A2")
def test_zero_lower_expresses_left_censoring():
    """spec 3.1, rev 4: `lower=0` expresses left censoring."""
    data = survjax.SurvivalData.from_bounds(f([0.0]), f([3.0]))
    assert_fields(data, exact=[False], lower=[0.0], upper=[3.0])


@pytest.mark.step("A2")
def test_exact_mask_row_uses_lower_as_time():
    """spec 3.1, rev 4: exact rows are marked by `exact`; their time is `lower`; `upper` may equal `lower`."""
    data = survjax.SurvivalData.from_bounds(
        f([1.5, 1.0]), f([1.5, 2.0]), exact=b([True, False])
    )
    assert_fields(data, exact=[True, False], lower=[1.5, 1.0], upper=[1.5, 2.0])


@pytest.mark.step("A2")
def test_exact_row_with_nan_upper_normalizes_upper_to_lower():
    """spec 3.1, rev 5: a NaN upper on an exact row normalizes to `lower`."""
    data = survjax.SurvivalData.from_bounds(f([1.5]), f([NAN]), exact=b([True]))
    assert_fields(data, exact=[True], lower=[1.5], upper=[1.5])


@pytest.mark.step("A2")
def test_omitted_exact_mask_means_no_exact_rows():
    """spec 3.1, rev 4: exactness is never inferred; without `exact` every row is censored."""
    data = survjax.SurvivalData.from_bounds(f([1.0, 2.0]), f([3.0, INF]))
    assert_fields(data, exact=[False, False])


@pytest.mark.step("A2")
def test_bounds_truncation_is_stored():
    """spec 3.1/3.2, rev 4: `left_trunc` and `right_trunc` keywords give L and U."""
    data = survjax.SurvivalData.from_bounds(
        f([1.0, 2.0]),
        f([3.0, INF]),
        left_trunc=f([0.5, 0.0]),
        right_trunc=f([4.0, INF]),
    )
    assert_fields(data, left_trunc=[0.5, 0.0], right_trunc=[4.0, INF])


@pytest.mark.step("A2")
def test_bounds_truncation_defaults():
    """spec 3.2, rev 4: absent truncation normalizes to L=0 and U=+inf."""
    data = survjax.SurvivalData.from_bounds(f([1.0]), f([3.0]))
    assert_fields(data, left_trunc=[0.0], right_trunc=[INF])


@pytest.mark.step("A2")
def test_scalar_upper_broadcasts_to_lower_shape():
    """spec 3.2, rev 4: `lower` determines the observation shape; other fields broadcast to it."""
    data = survjax.SurvivalData.from_bounds(f([1.0, 2.0, 3.0]), f(INF))
    assert np.shape(data.upper) == (3,)
    assert_fields(data, upper=[INF, INF, INF])


# ---------------------------------------------------------- invalid rows


@pytest.mark.step("A2")
def test_equal_endpoints_on_non_exact_row_set_issue():
    """spec 3.1, rev 9: equal endpoints on a non-exact row are invalid."""
    data = survjax.SurvivalData.from_bounds(f([2.0]), f([2.0]), exact=b([False]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_equal_endpoints_without_exact_mask_set_issue():
    """spec 3.1, rev 9: exactness is not inferred from equal endpoints."""
    data = survjax.SurvivalData.from_bounds(f([2.0]), f([2.0]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_reversed_bounds_set_issue():
    """spec 7.1, rev 9: reversed endpoints are invalid (censored rows mean lower < T <= upper)."""
    data = survjax.SurvivalData.from_bounds(f([3.0]), f([2.0]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_exact_row_with_different_upper_sets_issue():
    """spec 3.1, rev 9: an exact row's `upper` must equal `lower` or be NaN."""
    data = survjax.SurvivalData.from_bounds(f([1.5]), f([2.0]), exact=b([True]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_exact_row_at_zero_sets_issue():
    """spec 3.3, rev 9: exact times must be strictly positive (bounds constructor)."""
    data = survjax.SurvivalData.from_bounds(f([0.0]), f([0.0]), exact=b([True]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_negative_lower_sets_issue():
    """spec 3.3, rev 9: censoring endpoints are nonnegative."""
    data = survjax.SurvivalData.from_bounds(f([-1.0]), f([2.0]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_infinite_lower_sets_issue():
    """spec 3.3, rev 9: `lower` is finite."""
    data = survjax.SurvivalData.from_bounds(f([INF]), f([INF]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_nan_lower_sets_issue():
    """spec 3.1, rev 9: missing required values are invalid."""
    data = survjax.SurvivalData.from_bounds(f([NAN]), f([2.0]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_nan_upper_on_censored_row_sets_issue():
    """spec 3.1/6.2, rev 9: a non-exact `from_bounds` row with NaN `upper` is invalid (`missing_time2`)."""
    data = survjax.SurvivalData.from_bounds(f([1.0]), f([NAN]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_negative_upper_sets_issue():
    """spec 6.2, rev 9: a supplied `from_bounds` upper that is negative is invalid (`invalid_time2`)."""
    data = survjax.SurvivalData.from_bounds(f([0.0]), f([-1.0]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_minus_infinite_upper_sets_issue():
    """spec 6.2, rev 9: a supplied `from_bounds` upper of `-inf` is invalid (`invalid_time2`)."""
    data = survjax.SurvivalData.from_bounds(f([0.0]), f([-INF]))
    assert bool(np.asarray(data.issues.invalid_rows).any())


@pytest.mark.step("A2")
def test_column_upper_cannot_enlarge_vector_lower():
    """spec 3.2, rev 9: fields must broadcast to the shape of `lower` without enlarging it."""
    with pytest.raises(MISUSE):
        survjax.SurvivalData.from_bounds(f([1.0, 2.0, 3.0]), f([[4.0], [5.0], [6.0]]))


@pytest.mark.step("A2")
def test_column_exact_mask_cannot_enlarge_vector_lower():
    """spec 3.2, rev 9: fields must broadcast to the shape of `lower` without enlarging it."""
    with pytest.raises(MISUSE):
        survjax.SurvivalData.from_bounds(
            f([1.0, 2.0]), f([1.0, 2.0]), exact=b([[True], [True]])
        )


# ------------------------------------ both constructors normalize identically


@pytest.mark.step("A2")
def test_exact_rows_normalize_identically():
    """spec 12.1, rev 4: both constructors normalize identically (exact rows)."""
    flags = survjax.SurvivalData(f([1.0, 2.0]))
    bounds = survjax.SurvivalData.from_bounds(
        f([1.0, 2.0]), f([NAN, 2.0]), exact=b([True, True])
    )
    assert_same_fields(flags, bounds)


@pytest.mark.step("A2")
def test_right_censored_rows_normalize_identically():
    """spec 12.1, rev 4: both constructors normalize identically (right censoring)."""
    flags = survjax.SurvivalData(f([1.0, 2.0]), b([False, False]))
    bounds = survjax.SurvivalData.from_bounds(f([1.0, 2.0]), f([INF, INF]))
    assert_same_fields(flags, bounds)


@pytest.mark.step("A2")
def test_left_censored_rows_normalize_identically():
    """spec 12.1, rev 4: both constructors normalize identically (left censoring)."""
    flags = survjax.SurvivalData(f([1.0, 2.0]), left_censored=b([True, True]))
    bounds = survjax.SurvivalData.from_bounds(f([0.0, 0.0]), f([1.0, 2.0]))
    assert_same_fields(flags, bounds)


@pytest.mark.step("A2")
def test_interval_rows_normalize_identically():
    """spec 12.1, rev 4: both constructors normalize identically (interval censoring)."""
    flags = survjax.SurvivalData(
        f([1.0, 2.0]), interval_censored=b([True, True]), time2=f([1.5, 4.0])
    )
    bounds = survjax.SurvivalData.from_bounds(f([1.0, 2.0]), f([1.5, 4.0]))
    assert_same_fields(flags, bounds)


@pytest.mark.step("A2")
def test_mixed_truncated_rows_normalize_identically():
    """spec 12.1, rev 4: both constructors normalize identically (mixed kinds with truncation)."""
    lt, rt = f([0.5, 0.5, 0.0, 0.25]), f([5.0, INF, 4.0, INF])
    flags = survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 1.0]),
        right_censored=b([False, True, False, False]),
        left_censored=b([False, False, True, False]),
        interval_censored=b([False, False, False, True]),
        time2=f([NAN, NAN, NAN, 2.0]),
        left_trunc=lt,
        right_trunc=rt,
    )
    bounds = survjax.SurvivalData.from_bounds(
        f([1.0, 2.0, 0.0, 1.0]),
        f([1.0, INF, 3.0, 2.0]),
        exact=b([True, False, False, False]),
        left_trunc=lt,
        right_trunc=rt,
    )
    assert_same_fields(flags, bounds)


# Positive times exactly representable in float32 (and therefore in float64).
_times = st.floats(min_value=2.0**-6, max_value=2.0**10, width=32)
_rows = st.lists(
    st.tuples(st.sampled_from(["exact", "right", "left", "interval"]), _times, _times),
    min_size=1,
    max_size=8,
)


@pytest.mark.step("A2")
@settings(derandomize=True, max_examples=100, deadline=None)
@given(_rows)
def test_constructors_normalize_identically_property(rows):
    """spec 12.1, rev 4: for any valid untruncated rows, both constructors give identical fields."""
    time, time2, right, left, interval, exact, lower, upper = ([] for _ in range(8))
    for kind, a, c in rows:
        lo, hi = sorted((a, c))
        if kind == "interval":
            assume(lo < hi)
        time.append(lo if kind == "interval" else a)
        time2.append(hi if kind == "interval" else NAN)
        right.append(kind == "right")
        left.append(kind == "left")
        interval.append(kind == "interval")
        exact.append(kind == "exact")
        lower.append({"exact": a, "right": a, "left": 0.0, "interval": lo}[kind])
        upper.append({"exact": a, "right": INF, "left": a, "interval": hi}[kind])

    flags = survjax.SurvivalData(
        f(time),
        right_censored=b(right),
        left_censored=b(left),
        interval_censored=b(interval),
        time2=f(time2),
    )
    bounds = survjax.SurvivalData.from_bounds(f(lower), f(upper), exact=b(exact))
    assert_same_fields(flags, bounds)
    assert_fields(
        flags,
        exact=exact,
        lower=np.asarray(lower, FLOAT),
        upper=np.asarray(upper, FLOAT),
    )
