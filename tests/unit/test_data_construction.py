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


def test_nan_and_fractional_event_codes_set_invalid_indicator() -> None:
    """Spec 3.1/6.2: neither NaN nor 0.5 is an exact 0/1 indicator."""
    _, issues = SurvivalData.with_issues([1.0, 2.0], [np.nan, 0.5])
    np.testing.assert_array_equal(issues.invalid_indicator, [True, True])
    np.testing.assert_array_equal(issues.invalid_rows, [True, True])


def test_event_and_supplied_false_flag_raise_type_error() -> None:
    """Spec 3.1 / D-006: argument-presence conflict is static misuse."""
    with pytest.raises(TypeError, match="event cannot be combined"):
        SurvivalData([1.0, 2.0], [1, 0], left_censored=[False, False])


def test_pure_event_and_supplied_false_flag_raise_type_error() -> None:
    """Spec 6.2 / D-006: the pure path also rejects static misuse."""
    with pytest.raises(TypeError, match="event cannot be combined"):
        SurvivalData.with_issues([1.0, 2.0], [1, 0], left_censored=[False, False])


def test_two_true_flags_conflict_on_same_row() -> None:
    """Spec 3.1: no row has two censoring kinds."""
    with pytest.raises(
        ValueError,
        match=r"conflicting_indicators: 1 row\(s\), first index \(1,\)",
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


def test_bounds_nan_upper_on_censored_row_reports_missing_time2() -> None:
    """Spec 6.2: a required censored upper of NaN is missing, not unbounded."""
    with pytest.raises(ValueError, match="missing_time2: 1 row"):
        SurvivalData.from_bounds([1.0], [np.nan])


def test_unknown_flag_keyword_is_argument_misuse() -> None:
    """Spec 3.1/6.2: removed and unknown keywords raise TypeError."""
    with pytest.raises(TypeError, match="outside_interval_censored"):
        SurvivalData([1.0], outside_interval_censored=[False])
    with pytest.raises(TypeError, match="outside_interval_censored"):
        SurvivalData.with_issues([1.0], outside_interval_censored=[False])


@pytest.mark.parametrize("constructor", ["flag", "bounds"])
@pytest.mark.parametrize("other_shape", [(2, 1), (3,)])
def test_other_field_cannot_enlarge_or_conflict_with_shape(
    constructor: str, other_shape: tuple[int, ...]
) -> None:
    """Spec 3.2/6.2: malformed observation shapes raise TypeError."""
    other = jnp.ones(other_shape)
    with pytest.raises(TypeError, match="shape"):
        if constructor == "flag":
            SurvivalData(jnp.ones(2), time2=other)
        else:
            SurvivalData.from_bounds(jnp.ones(2), other)


def test_indicator_shape_misuse_raises_under_tracing() -> None:
    """Spec 3.2/6.2: static shape misuse is rejected while tracing."""
    with pytest.raises(TypeError, match="shape"):
        jax.jit(lambda time, flag: SurvivalData.with_issues(time, right_censored=flag))(
            jnp.ones(2), jnp.ones((2, 1))
        )


def test_bounds_shape_misuse_raises_under_tracing() -> None:
    """Spec 3.2/6.2: bounds shapes are checked during tracing."""
    with pytest.raises(TypeError, match="shape"):
        jax.jit(lambda lower, upper: SurvivalData.from_bounds(lower, upper))(
            jnp.ones(2), jnp.ones(3)
        )


def test_nonnumeric_bounds_and_indicators_are_argument_misuse() -> None:
    """Spec 3.3/6.2: malformed field types raise TypeError."""
    with pytest.raises(TypeError):
        SurvivalData.from_bounds([1.0], ["2.0"])
    with pytest.raises(TypeError):
        SurvivalData([1.0], right_censored=["yes"])


def test_unknown_bounds_keyword_is_argument_misuse() -> None:
    """Spec 3.1/6.2: only documented bounds keywords are accepted."""
    with pytest.raises(TypeError, match="unknown"):
        SurvivalData.from_bounds([1.0], [2.0], unknown=True)


def test_ordinary_constructor_rejects_traced_observations() -> None:
    """Spec 6.2: eager validation points traced callers to with_issues."""
    with pytest.raises(ValueError, match="with_issues"):
        jax.jit(lambda time: SurvivalData(time))(jnp.array([1.0]))


def test_traced_bounds_error_explains_flag_only_pure_api() -> None:
    """Spec 6.2: the bounds error must not imply flag with_issues takes bounds."""
    with pytest.raises(ValueError, match="pure bounds constructor") as error:
        jax.jit(lambda lo, hi: SurvivalData.from_bounds(lo, hi).lower)(
            jnp.array([1.0]), jnp.array([2.0])
        )
    assert "with_issues accepts flag inputs only" in str(error.value)
