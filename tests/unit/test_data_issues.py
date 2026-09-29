"""A2b: pure construction and decoded data diagnostics (spec 6.2–6.3)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronox_jax import SurvivalData
from chronox_jax.diagnostics import DataIssues


def test_with_issues_matches_eager_fields_on_valid_rows() -> None:
    """Spec 6.2: pure and eager flag construction share normalization."""
    inputs = {"right_censored": [False, True], "left_trunc": [0.0, 1.0]}
    data, issues = SurvivalData.with_issues([2.0, 3.0], **inputs)
    eager = SurvivalData([2.0, 3.0], **inputs)
    for name in ("exact", "lower", "upper", "left_trunc", "right_trunc"):
        np.testing.assert_array_equal(getattr(data, name), getattr(eager, name))
    np.testing.assert_array_equal(data._issue_bits, [0, 0])
    assert isinstance(issues, DataIssues)
    assert bool(np.asarray(issues.ok))


def test_raw_indicator_and_conflict_masks_overlap() -> None:
    """Spec 3.1/6.2: invalid codes and supplied flag conflicts survive together."""
    data, issues = SurvivalData.with_issues(
        [1.0, 2.0], [1, 2], right_censored=[False, False]
    )
    np.testing.assert_array_equal(issues.invalid_indicator, [False, True])
    np.testing.assert_array_equal(issues.conflicting_indicators, [True, True])
    np.testing.assert_array_equal(issues.invalid_rows, [True, True])
    np.testing.assert_array_equal(data._issue_bits, [2, 3])
    assert not bool(np.asarray(issues.ok))


def test_active_upper_faults_are_distinct_from_interval_order() -> None:
    """Spec 6.2 / D-004: missing, invalid, and reversed uppers differ."""
    _, issues = SurvivalData.with_issues(
        [1.0, 1.0, 2.0, 3.0],
        interval_censored=[False, True, True, True],
        time2=[np.nan, np.nan, -1.0, 2.0],
    )
    np.testing.assert_array_equal(issues.missing_time2, [False, True, False, False])
    np.testing.assert_array_equal(issues.invalid_time2, [False, False, True, False])
    np.testing.assert_array_equal(issues.invalid_interval, [False, False, False, True])


def test_invalid_first_endpoints_do_not_gain_interval_order_fault() -> None:
    """Spec 6.2 / D-005: required first endpoints use invalid_time."""
    _, issues = SurvivalData.with_issues(
        [0.0, np.nan, -1.0],
        right_censored=[False, True, False],
        interval_censored=[False, False, True],
        time2=[np.nan, np.nan, 2.0],
    )
    np.testing.assert_array_equal(issues.invalid_time, [True, True, True])
    np.testing.assert_array_equal(issues.invalid_interval, [False, False, False])


def test_window_faults_mark_their_own_rows() -> None:
    """Spec 2/3.3/6.2: exact, entry, empty, and truncation rules differ."""
    _, issues = SurvivalData.with_issues(
        [1.0, 0.5, 1.0, 2.0],
        right_censored=[False, True, False, False],
        left_censored=[False, False, True, False],
        left_trunc=[1.0, 1.0, 1.0, -1.0],
    )
    np.testing.assert_array_equal(
        issues.exact_outside_truncation, [True, False, False, False]
    )
    np.testing.assert_array_equal(
        issues.censored_before_entry, [False, True, False, False]
    )
    np.testing.assert_array_equal(issues.empty_observation, [False, False, True, False])
    np.testing.assert_array_equal(
        issues.invalid_truncation, [False, False, False, True]
    )
    np.testing.assert_array_equal(issues.invalid_rows, [True, True, True, True])


def test_uninformative_rows_are_valid_and_separate_from_invalid_rows() -> None:
    """Spec 2/6.2: censored sets containing (L,U] are non-fatal."""
    _, issues = SurvivalData.with_issues(
        [1.0, 4.0, 0.5, 2.0, 0.0],
        right_censored=[True, False, False, False, False],
        left_censored=[False, True, False, False, False],
        interval_censored=[False, False, True, False, False],
        time2=[np.nan, np.nan, 4.0, np.nan, np.nan],
        left_trunc=[1.0, 0.0, 1.0, 1.0, 0.0],
        right_trunc=[3.0, 3.0, 3.0, 5.0, np.inf],
    )
    np.testing.assert_array_equal(
        issues.uninformative, [True, True, True, False, False]
    )
    np.testing.assert_array_equal(
        issues.invalid_rows, [False, False, False, False, True]
    )
    assert not bool(np.asarray(issues.ok))


def test_diagnostic_pytree_crosses_jit_and_vmap() -> None:
    """Spec 6.2–6.3: masks and scalar reduction remain arrays under tracing."""

    def build(time: jax.Array) -> DataIssues:
        return SurvivalData.with_issues(time)[1]

    issues = jax.jit(build)(jnp.array([1.0, 0.0]))
    assert len(jax.tree.leaves(issues)) == 13
    np.testing.assert_array_equal(issues.invalid_time, [False, True])
    np.testing.assert_array_equal(issues.invalid_rows, [False, True])
    assert issues.ok.shape == ()

    batch = jax.vmap(build)(jnp.array([[1.0, 0.0], [2.0, 3.0]]))
    np.testing.assert_array_equal(batch.invalid_time, [[False, True], [False, False]])
    np.testing.assert_array_equal(batch.ok, [False, True])


def test_raise_if_invalid_reports_counts_and_first_index() -> None:
    """Spec 6.2: explicit host checkpoint names each invalid rule."""
    _, issues = SurvivalData.with_issues([1.0, 0.0, 0.0])
    with pytest.raises(
        ValueError, match=r"invalid_time: 2 row\(s\), first index \(1,\)"
    ):
        issues.raise_if_invalid()


def test_raise_if_invalid_accepts_valid_uninformative_rows() -> None:
    """Spec 6.2: uninformative is outside the invalid union."""
    _, issues = SurvivalData.with_issues([1.0], right_censored=[True], left_trunc=[1.0])
    issues.raise_if_invalid()


def test_raise_if_invalid_is_host_only() -> None:
    """Spec 6.2: a checkpoint cannot inspect traced validity on the host."""
    with pytest.raises(ValueError, match="host-only"):
        jax.jit(lambda time: SurvivalData.with_issues(time)[1].raise_if_invalid())(
            jnp.array([1.0])
        )


def test_structurally_invalid_time_still_raises() -> None:
    """Spec 6.2: nonnumeric time cannot form an observation object."""
    with pytest.raises(TypeError):
        SurvivalData.with_issues(np.array(["1.0", "2.0"]))


def test_structurally_invalid_shape_still_raises() -> None:
    """Spec 3.2/6.2: a flag may not enlarge the observation shape."""
    with pytest.raises(ValueError, match="enlarge"):
        SurvivalData.with_issues([1.0, 2.0], right_censored=[[True], [False]])
