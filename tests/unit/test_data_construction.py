"""A2a: eager constructor validation and normalization."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronox_jax import SurvivalData


def test_invalid_event_code_reports_count_and_first_index() -> None:
    """Spec 3.1/6.2: raw event codes are checked before boolean conversion."""
    with pytest.raises(
        ValueError, match=r"invalid_indicator: 2 row\(s\), first index \(1,\)"
    ):
        SurvivalData([1.0, 2.0, 3.0], [1, 2, -1])


def test_event_and_supplied_false_flag_conflict() -> None:
    """Spec 3.1: supplying any flag with event is a conflict."""
    with pytest.raises(ValueError, match="conflicting_indicators"):
        SurvivalData([1.0, 2.0], [1, 0], left_censored=[False, False])


def test_two_true_flags_conflict_on_same_row() -> None:
    """Spec 3.1: no row has two censoring kinds."""
    with pytest.raises(
        ValueError, match=r"conflicting_indicators: 1 row\(s\), first index \(1,\)"
    ):
        SurvivalData(
            [1.0, 2.0], right_censored=[False, True], left_censored=[False, True]
        )


def test_inactive_nan_time2_is_ignored() -> None:
    """Spec 3.1 / D-001: only interval rows use time2."""
    data = SurvivalData(
        [1.0, 2.0], interval_censored=[False, True], time2=[np.nan, 3.0]
    )
    np.testing.assert_array_equal(data.upper, [1.0, 3.0])
    np.testing.assert_array_equal(data._issue_bits, [0, 0])


def test_missing_active_time2_reports_missing_bit() -> None:
    """Spec 6.2: an absent active upper is missing_time2."""
    with pytest.raises(ValueError, match="missing_time2"):
        SurvivalData([1.0], interval_censored=[True])


def test_bounds_exact_nan_upper_normalizes() -> None:
    """Spec 3.1: exact NaN upper is allowed and replaced by lower."""
    data = SurvivalData.from_bounds([2.0], [np.nan], exact=[True])
    np.testing.assert_array_equal(data.upper, [2.0])
    np.testing.assert_array_equal(data._issue_bits, [0])


def test_bounds_mismatched_exact_upper_reports_invalid_time2() -> None:
    """Spec 6.2: an exact upper other than lower or NaN is invalid_time2."""
    with pytest.raises(ValueError, match="invalid_time2"):
        SurvivalData.from_bounds([2.0], [3.0], exact=[True])


def test_outside_interval_keyword_always_reports_unsupported() -> None:
    """Spec 3.1: any use of the unsupported mode raises explicitly."""
    with pytest.raises(NotImplementedError, match="outside_interval_censored"):
        SurvivalData([1.0], outside_interval_censored=[False])


def test_ordinary_constructor_rejects_traced_observations() -> None:
    """Spec 6.2: eager validation points traced callers to with_issues."""
    with pytest.raises(ValueError, match="with_issues"):
        jax.jit(lambda time: SurvivalData(time))(jnp.array([1.0]))
