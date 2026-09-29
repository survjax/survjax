"""A2a: endpoint and truncation boundaries."""

import numpy as np
import pytest

from chronox_jax import SurvivalData


def test_exact_time_at_entry_is_outside_window() -> None:
    """Spec 3.3: exact membership requires L < t."""
    with pytest.raises(ValueError, match="exact_outside_truncation"):
        SurvivalData([2.0], left_trunc=[2.0])


def test_exact_time_at_upper_truncation_is_valid() -> None:
    """Spec 3.3: exact membership allows t = U."""
    data = SurvivalData([2.0], right_trunc=[2.0])
    np.testing.assert_array_equal(data._issue_bits, [0])


def test_right_censoring_at_entry_is_valid_with_finite_window() -> None:
    """Spec 3.3 / D-003: c = L is a valid uninformative row."""
    data = SurvivalData([2.0], [False], left_trunc=[2.0], right_trunc=[3.0])
    np.testing.assert_array_equal(data._issue_bits, [0])


def test_right_censoring_before_entry_is_rejected() -> None:
    """Spec 6.2 / D-003: c < L reports censored_before_entry."""
    with pytest.raises(ValueError, match="censored_before_entry"):
        SurvivalData([1.0], [False], left_trunc=[2.0])


def test_interval_infinite_upper_uses_right_censoring_entry_rule() -> None:
    """Spec 3.1: interval time2=+inf and time=L is valid."""
    data = SurvivalData(
        [2.0], interval_censored=[True], time2=[np.inf], left_trunc=[2.0]
    )
    np.testing.assert_array_equal(data.upper, [np.inf])
    np.testing.assert_array_equal(data._issue_bits, [0])


def test_left_censoring_at_entry_has_empty_intersection() -> None:
    """Spec 2/6.2: (0, L] misses the truncation window."""
    with pytest.raises(ValueError, match="empty_observation"):
        SurvivalData([2.0], left_censored=[True], left_trunc=[2.0])


def test_interval_covering_window_is_valid() -> None:
    """Spec 2: a censored set covering (L, U] is valid."""
    data = SurvivalData(
        [1.0],
        interval_censored=[True],
        time2=[4.0],
        left_trunc=[2.0],
        right_trunc=[3.0],
    )
    np.testing.assert_array_equal(data._issue_bits, [0])


def test_invalid_interval_lower_does_not_report_interval_order() -> None:
    """Spec 6.2 / D-005: endpoint domain precedes interval order."""
    with pytest.raises(ValueError, match="invalid_time: 1 row") as error:
        SurvivalData([-1.0], interval_censored=[True], time2=[-2.0])
    assert "invalid_interval" not in str(error.value)
