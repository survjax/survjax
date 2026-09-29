"""A3: decoding the stored issue bitfield with ``data.issues()`` (spec 3.2, 6.2).

``data.issues()`` must agree with the ``DataIssues`` a pure constructor
returned, and indexing the data must index every per-row mask the same way.
"""

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
    from_bounds_with_issues,
    mask,
    one_hot,
    with_issues,
)

import chronox_jax as cx


def _mixed():
    """Five rows: valid exact, invalid indicator, conflicting flags, uninformative, valid left.

    row 0: exact at 1                         -> valid
    row 1: right flag = 2                     -> invalid_indicator
    row 2: right and left flags true          -> conflicting_indicators
    row 3: left-censored at c=U=3             -> uninformative (valid)
    row 4: left-censored at 2                 -> valid
    """
    return with_issues(
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
def test_issues_returns_a_data_issues_container():
    """spec 6.2, rev 7: `data.issues() -> DataIssues`, the same container type the pure constructors return."""
    data, issues = _mixed()
    assert type(data.issues()) is type(issues)


@pytest.mark.step("A3")
def test_issues_matches_with_issues_on_fresh_data():
    """spec 6.2, rev 7: for data from a pure constructor, `issues()` returns the same masks as its `DataIssues`."""
    data, issues = _mixed()
    _assert_same_masks(data.issues(), issues)
    assert bool(np.asarray(data.issues().ok)) == bool(np.asarray(issues.ok))


@pytest.mark.step("A3")
def test_issues_matches_from_bounds_with_issues_on_fresh_data():
    """spec 6.2, rev 7: the same holds for the pure bounds constructor."""
    data, issues = from_bounds_with_issues(
        f([1.0, 3.0, 0.5, 0.0]), f([1.0, 2.0, 4.0, INF]), exact=b([True, False, False, False]), left_trunc=f([0.0, 0.0, 0.0, 1.0])
    )
    assert not bool(np.asarray(issues.ok))
    _assert_same_masks(data.issues(), issues)


@pytest.mark.step("A3")
def test_ordinary_constructor_data_carries_only_the_uninformative_bit():
    """spec 6.2, rev 7: data from an ordinary constructor can carry only the `uninformative` bit."""
    data = cx.SurvivalData(f([1.0, 3.0]), left_censored=b([False, True]), right_trunc=f([INF, 3.0]))
    decoded = data.issues()
    for name in RULE_FIELDS:
        assert_mask(decoded, name, np.zeros(2, dtype=bool))
    assert_mask(decoded, "uninformative", one_hot(2, 1))
    assert bool(np.asarray(decoded.ok))


# ------------------------------------------------------- indexing


@pytest.mark.step("A3")
def test_issues_of_a_slice_index_the_parent_masks():
    """spec 6.2, rev 7: for `data[idx]`, each per-row mask equals the parent's mask indexed by `idx` (slice)."""
    data, issues = _mixed()
    _assert_same_masks(data[1:4].issues(), issues, np.s_[1:4])


@pytest.mark.step("A3")
def test_issues_of_an_integer_array_index_the_parent_masks():
    """spec 6.2, rev 7: the same for an integer-array index (minibatch)."""
    data, issues = _mixed()
    idx = np.array([4, 1, 3])
    _assert_same_masks(data[idx].issues(), issues, idx)


@pytest.mark.step("A3")
def test_issues_of_a_boolean_mask_index_the_parent_masks():
    """spec 6.2, rev 7: the same for a boolean-mask index (split)."""
    data, issues = _mixed()
    keep = b([True, False, True, True, False])
    _assert_same_masks(data[keep].issues(), issues, keep)


@pytest.mark.step("A3")
def test_issues_of_an_integer_index_give_scalar_masks():
    """spec 6.2, rev 7: a scalar observation decodes to scalar masks equal to the parent's row."""
    data, issues = _mixed()
    decoded = data[2].issues()
    for name in MASK_FIELDS:
        assert np.shape(getattr(decoded, name)) == (), name
    _assert_same_masks(decoded, issues, 2)


@pytest.mark.step("A3")
def test_issues_of_tree_map_slice_index_the_parent_masks():
    """spec 3.2/6.2, rev 7: `jax.tree.map` slicing carries the bitfield, so decoding matches the parent's rows."""
    data, issues = _mixed()
    sliced = jax.tree_util.tree_map(lambda leaf: leaf[0:2], data)
    _assert_same_masks(sliced.issues(), issues, np.s_[0:2])


@pytest.mark.step("A3")
def test_ok_of_a_slice_reduces_over_the_slice_only():
    """spec 6.2, rev 7: `ok` reduces over the indexed rows only; a slice of valid rows is ok though its parent is not."""
    data, issues = _mixed()
    assert not bool(np.asarray(issues.ok))
    assert bool(np.asarray(data[3:5].issues().ok))
    assert not bool(np.asarray(data[0:2].issues().ok))


# ------------------------------------------------------- transformations and rebuilding


@pytest.mark.step("A3")
def test_issues_under_jit_matches_eager():
    """spec 6.2, rev 7: `issues()` is pure and works under jit."""
    data, _ = _mixed()
    jitted = jax.jit(lambda d: d.issues())(data)
    _assert_same_masks(jitted, data.issues())


@pytest.mark.step("A3")
def test_issues_of_data_returned_from_jit_matches_construction():
    """spec 6.2, rev 7: data returned from a transformed function decode to the issues computed inside it."""
    inputs = (f([1.0, 2.0, 3.0]), np.array([0, 2, 1], dtype=np.int32))
    data, issues = jax.jit(lambda t, r: with_issues(t, right_censored=r))(*inputs)
    _assert_same_masks(data.issues(), issues)


@pytest.mark.step("A3")
def test_issues_of_data_rebuilt_from_leaves_matches_original():
    """spec 3.2/6.2, rev 7: data rebuilt from its leaves decode to the same issues."""
    data, issues = _mixed()
    rebuilt = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(data), jax.tree_util.tree_leaves(data))
    _assert_same_masks(rebuilt.issues(), issues)
