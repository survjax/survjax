"""A1: normalized representation, shapes and PyTree storage (spec 3.2)."""

import jax
import numpy as np
import pytest
from helpers import (
    FIELDS,
    FLOAT,
    IMMUTABLE,
    INF,
    NAN,
    assert_fields,
    b,
    f,
    leaves,
    treedef,
)

import survjax


def _mixed():
    """Four rows, one of each kind: exact 1, right at 2, left at 3, interval (1, 4]."""
    return survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 1.0]),
        right_censored=b([False, True, False, False]),
        left_censored=b([False, False, True, False]),
        interval_censored=b([False, False, False, True]),
        time2=f([NAN, NAN, NAN, 4.0]),
    )


@pytest.mark.step("A1")
def test_all_five_fields_have_observation_shape():
    """spec 3.2, rev 4: both constructors normalize to five public fields, all with the observation shape."""
    data = survjax.SurvivalData(f([1.0, 2.0, 3.0]), b([True, False, True]))
    for name in FIELDS:
        assert np.shape(getattr(data, name)) == (3,), name


@pytest.mark.step("A1")
def test_exact_field_is_boolean():
    """spec 3.2, rev 4: `exact` is bool."""
    data = survjax.SurvivalData(f([1.0, 2.0]), b([True, False]))
    assert np.asarray(data.exact).dtype == np.bool_


@pytest.mark.step("A1")
def test_exact_row_upper_equals_lower():
    """spec 3.2, rev 4: `upper` equals `lower` for exact rows; `lower` is the exact time."""
    data = survjax.SurvivalData(f([1.5]), b([True]))
    assert_fields(data, exact=[True], lower=[1.5], upper=[1.5])


@pytest.mark.step("A1")
def test_right_censored_row_has_infinite_upper():
    """spec 3.2, rev 4: `upper` is +inf for right censoring; `lower` is the censoring time."""
    data = survjax.SurvivalData(f([2.5]), b([False]))
    assert_fields(data, exact=[False], lower=[2.5], upper=[INF])


@pytest.mark.step("A1")
def test_left_censored_row_has_zero_lower():
    """spec 3.2, rev 4: `lower` is 0 for left censoring; `upper` is the censoring time."""
    data = survjax.SurvivalData(f([3.0]), left_censored=b([True]))
    assert_fields(data, exact=[False], lower=[0.0], upper=[3.0])


@pytest.mark.step("A1")
def test_interval_row_keeps_both_endpoints():
    """spec 3.2, rev 4: interval rows store the set's lower and upper endpoints."""
    data = survjax.SurvivalData(f([1.0]), interval_censored=b([True]), time2=f([4.0]))
    assert_fields(data, exact=[False], lower=[1.0], upper=[4.0])


@pytest.mark.step("A1")
def test_mixed_kinds_normalize_row_by_row():
    """spec 3.2, rev 4: every field is meaningful on every row, with no placeholders."""
    data = _mixed()
    assert_fields(
        data,
        exact=[True, False, False, False],
        lower=[1.0, 2.0, 0.0, 1.0],
        upper=[1.0, INF, 3.0, 4.0],
    )


@pytest.mark.step("A1")
def test_absent_left_truncation_normalizes_to_zero():
    """spec 3.2, rev 4: `left_trunc` is 0 when absent."""
    data = _mixed()
    assert_fields(data, left_trunc=np.zeros(4))


@pytest.mark.step("A1")
def test_absent_right_truncation_normalizes_to_infinity():
    """spec 3.2, rev 4: `right_trunc` is +inf when absent."""
    data = _mixed()
    assert_fields(data, right_trunc=np.full(4, INF))


@pytest.mark.step("A1")
def test_supplied_truncation_is_stored_per_row():
    """spec 3.2, rev 4: `left_trunc` holds L and `right_trunc` holds U per row."""
    data = survjax.SurvivalData(
        f([1.0, 2.0]),
        b([True, True]),
        left_trunc=f([0.5, 0.0]),
        right_trunc=f([INF, 3.0]),
    )
    assert_fields(data, left_trunc=[0.5, 0.0], right_trunc=[INF, 3.0])


@pytest.mark.step("A1")
def test_scalar_time_gives_scalar_fields():
    """spec 3.2, rev 4: scalars stay scalar."""
    data = survjax.SurvivalData(f(2.0))
    for name in FIELDS:
        assert np.shape(getattr(data, name)) == (), name


@pytest.mark.step("A1")
def test_scalar_time_gives_scalar_leaves():
    """spec 3.2, rev 4: every leaf, including the issue bitfield, has the observation shape ()."""
    data = survjax.SurvivalData(f(2.0))
    for leaf in leaves(data):
        assert np.shape(leaf) == ()


@pytest.mark.step("A1")
def test_multidimensional_observations_are_not_flattened():
    """spec 3.2, rev 4: multidimensional observations are not flattened."""
    time = f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    event = b([[True, False, True], [False, True, False]])
    data = survjax.SurvivalData(time, event)
    for name in FIELDS:
        assert np.shape(getattr(data, name)) == (2, 3), name
    assert_fields(data, lower=time, exact=event)


@pytest.mark.step("A1")
def test_pytree_has_exactly_six_leaves_with_observation_shape():
    """spec 3.2, rev 4: array leaves are the five fields plus one issue bitfield, all with the observation shape."""
    data = _mixed()
    data_leaves = leaves(data)
    assert len(data_leaves) == 6
    for leaf in data_leaves:
        assert np.shape(leaf) == (4,)


@pytest.mark.step("A1")
def test_pytree_leaf_dtypes_are_bool_int32_and_four_floats():
    """spec 3.2, rev 4: one bool field, four time fields of one float dtype, one int32 issue bitfield."""
    dtypes = sorted(str(np.asarray(leaf).dtype) for leaf in leaves(_mixed()))
    assert dtypes == sorted(["bool", "int32", *[str(FLOAT)] * 4])


@pytest.mark.step("A1")
def test_issue_bitfield_is_int32_for_scalar_data():
    """spec 3.2, rev 4: the issue bitfield leaf is int32 with the observation shape, in both dtype modes."""
    int_leaves = [
        x
        for x in leaves(survjax.SurvivalData(f(2.0)))
        if np.asarray(x).dtype.kind in "iu"
    ]
    assert len(int_leaves) == 1
    assert np.asarray(int_leaves[0]).dtype == np.int32
    assert np.shape(int_leaves[0]) == ()


@pytest.mark.step("A1")
def test_data_passes_through_jit_unchanged():
    """spec 3.2, rev 4: SurvivalData is a registered PyTree, so it crosses a jit boundary intact."""
    data = _mixed()
    out = jax.jit(lambda d: d)(data)
    assert isinstance(out, survjax.SurvivalData)
    assert treedef(out) == treedef(data)
    for name in FIELDS:
        np.testing.assert_array_equal(
            np.asarray(getattr(out, name)), np.asarray(getattr(data, name))
        )


@pytest.mark.step("A1")
def test_fields_are_usable_under_vmap():
    """spec 3.2, rev 4: reconstruction with traced leaves works; vmap over the observation axis gives per-row values."""
    data = survjax.SurvivalData(
        f([1.0, 1.0]), interval_censored=b([False, True]), time2=f([NAN, 3.0])
    )
    # exact row: upper - lower = 0; interval (1, 3]: 3 - 1 = 2
    width = jax.vmap(lambda d: d.upper - d.lower)(data)
    np.testing.assert_array_equal(np.asarray(width), f([0.0, 2.0]))


@pytest.mark.step("A1")
def test_unflatten_bypasses_public_normalization():
    """spec 3.2, rev 4: reconstruction bypasses public normalization (arbitrary leaves are not validated)."""
    data = _mixed()
    sentinels = [object() for _ in leaves(data)]
    rebuilt = jax.tree_util.tree_unflatten(treedef(data), sentinels)
    assert isinstance(rebuilt, survjax.SurvivalData)


@pytest.mark.step("A1")
def test_data_is_immutable():
    """spec 3.2, rev 4: SurvivalData is immutable."""
    data = _mixed()
    with pytest.raises(IMMUTABLE):
        data.lower = f([9.0, 9.0, 9.0, 9.0])
