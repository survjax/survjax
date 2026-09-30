"""A3: decoding the stored issue bitfield with ``data.issues`` (spec 3.2, 6.2).

The property decodes stored bits after construction, slicing and transformations.
"""

import chronox_jax as cx
import jax
import numpy as np
import pytest
from helpers import (
    INF,
    MASK_FIELDS,
    RULE_FIELDS,
    assert_mask,
    b,
    f,
    flag_data_and_issues,
    mask,
    one_hot,
)


def _mixed():
    """Five rows: valid exact, invalid indicator, conflicting flags, uninformative, valid left.

    row 0: exact at 1                         -> valid
    row 1: right flag = 2                     -> invalid_indicator
    row 2: right and left flags true          -> conflicting_indicators
    row 3: left-censored at c=U=3             -> uninformative (valid)
    row 4: left-censored at 2                 -> valid
    """
    return flag_data_and_issues(
        f([1.0, 2.0, 3.0, 3.0, 2.0]),
        right_censored=np.array([0, 2, 1, 0, 0], dtype=np.int32),
        left_censored=np.array([0, 0, 1, 1, 1], dtype=np.int32),
        right_trunc=f([INF, INF, INF, 3.0, INF]),
    )


def _assert_same_masks(decoded, expected, index=()):
    for name in MASK_FIELDS:
        np.testing.assert_array_equal(mask(decoded, name), mask(expected, name)[index], err_msg=name)


# ------------------------------------------------------- fresh data




@pytest.mark.step("A3")
def test_fresh_flag_data_decodes_invalid_indicator():
    """spec 3.2/6.2, rev 9: fresh flag data decode stored invalid-indicator evidence."""
    data, _ = _mixed()
    assert_mask(data.issues, "invalid_indicator", one_hot(5, 1))



@pytest.mark.step("A3")
def test_fresh_bounds_data_decodes_invalid_interval():
    """spec 3.2/6.2, rev 9: fresh bounds data decode a stored interval issue."""
    data = cx.SurvivalData.from_bounds(f([1.0, 3.0]), f([2.0, 2.0]))
    assert_mask(data.issues, "invalid_interval", one_hot(2, 1))



@pytest.mark.step("A3")
def test_valid_data_carries_the_uninformative_bit():
    """spec 6.2, rev 9: a valid construction can carry the `uninformative` bit."""
    data = cx.SurvivalData(f([1.0, 3.0]), left_censored=b([False, True]), right_trunc=f([INF, 3.0]))
    decoded = data.issues
    for name in RULE_FIELDS:
        assert_mask(decoded, name, np.zeros(2, dtype=bool))
    assert_mask(decoded, "uninformative", one_hot(2, 1))
    assert bool(np.asarray(decoded.ok))


# ------------------------------------------------------- indexing


@pytest.mark.step("A3")
def test_issues_of_a_slice_index_the_parent_masks():
    """spec 6.2, rev 9: for `data[idx]`, each per-row mask equals the parent's mask indexed by `idx` (slice)."""
    data, issues = _mixed()
    _assert_same_masks(data[1:4].issues, issues, np.s_[1:4])


@pytest.mark.step("A3")
def test_issues_of_an_integer_array_index_the_parent_masks():
    """spec 6.2, rev 9: the same for an integer-array index (minibatch)."""
    data, issues = _mixed()
    idx = np.array([4, 1, 3])
    _assert_same_masks(data[idx].issues, issues, idx)


@pytest.mark.step("A3")
def test_issues_of_a_boolean_mask_index_the_parent_masks():
    """spec 6.2, rev 9: the same for a boolean-mask index (split)."""
    data, issues = _mixed()
    keep = b([True, False, True, True, False])
    _assert_same_masks(data[keep].issues, issues, keep)


@pytest.mark.step("A3")
def test_issues_of_an_integer_index_give_scalar_masks():
    """spec 6.2, rev 9: a scalar observation decodes to scalar masks equal to the parent's row."""
    data, issues = _mixed()
    decoded = data[2].issues
    for name in MASK_FIELDS:
        assert np.shape(getattr(decoded, name)) == (), name
    _assert_same_masks(decoded, issues, 2)


@pytest.mark.step("A3")
def test_issues_of_tree_map_slice_index_the_parent_masks():
    """spec 3.2/6.2, rev 9: `jax.tree.map` slicing carries the bitfield, so decoding matches the parent's rows."""
    data, issues = _mixed()
    sliced = jax.tree_util.tree_map(lambda leaf: leaf[0:2], data)
    _assert_same_masks(sliced.issues, issues, np.s_[0:2])


@pytest.mark.step("A3")
def test_ok_of_a_slice_reduces_over_the_slice_only():
    """spec 6.2, rev 9: `ok` reduces over the indexed rows only; a slice of valid rows is ok though its parent is not."""
    data, issues = _mixed()
    assert not bool(np.asarray(issues.ok))
    assert bool(np.asarray(data[3:5].issues.ok))
    assert not bool(np.asarray(data[0:2].issues.ok))


@pytest.mark.step("A3")
def test_issues_has_no_callable_alias():
    """spec 6.2/13, rev 10: the decoder is the `data.issues` property only; there is no callable `data.issues()` alias."""
    data, _ = _mixed()
    assert not callable(data.issues)
    with pytest.raises(TypeError):
        data.issues()


# ------------------------------------------------------- transformations and rebuilding


@pytest.mark.step("A3")
def test_issues_under_jit_matches_eager():
    """spec 6.2, rev 9: `data.issues` is pure and works under jit."""
    data, _ = _mixed()
    jitted = jax.jit(lambda d: d.issues)(data)
    _assert_same_masks(jitted, data.issues)


@pytest.mark.step("A3")
def test_issues_of_data_returned_from_jit_preserve_raw_indicator():
    """spec 3.2/6.2, rev 9: transformed data retain raw invalid-indicator evidence."""
    time = f([1.0, 2.0, 3.0])
    right = np.array([0, 2, 1], dtype=np.int32)
    data = jax.jit(lambda t, r: cx.SurvivalData(t, right_censored=r))(time, right)
    assert_mask(data.issues, "invalid_indicator", one_hot(3, 1))



@pytest.mark.step("A3")
def test_issues_of_data_rebuilt_from_leaves_matches_original():
    """spec 3.2/6.2, rev 9: data rebuilt from its leaves decode to the same issues."""
    data, issues = _mixed()
    rebuilt = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(data), jax.tree_util.tree_leaves(data))
    _assert_same_masks(rebuilt.issues, issues)
