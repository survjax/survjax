"""A3: indexing and slicing of SurvivalData (spec 3.2)."""

import chronox_jax as cx
import jax
import numpy as np
import pytest
from helpers import (
    FIELDS,
    INF,
    NAN,
    assert_same_fields,
    b,
    f,
    flag_data_and_issues,
    leaves,
    treedef,
)
from hypothesis import given, settings
from hypothesis import strategies as st


def _data():
    """Five rows of mixed kinds with truncation (all valid)."""
    return cx.SurvivalData(
        f([1.0, 2.0, 3.0, 1.0, 4.0]),
        right_censored=b([False, True, False, False, False]),
        left_censored=b([False, False, True, False, False]),
        interval_censored=b([False, False, False, True, False]),
        time2=f([NAN, NAN, NAN, 2.5, NAN]),
        left_trunc=f([0.5, 0.0, 0.0, 0.25, 1.0]),
        right_trunc=f([INF, 5.0, INF, INF, 6.0]),
    )


def _assert_indexed(data, idx):
    """data[idx] is a SurvivalData whose fields are the parent's fields indexed by idx."""
    sub = data[idx]
    assert isinstance(sub, cx.SurvivalData)
    for name in FIELDS:
        expected = np.asarray(getattr(data, name))[idx]
        got = np.asarray(getattr(sub, name))
        assert got.shape == expected.shape, name
        assert got.dtype == expected.dtype, name
        np.testing.assert_array_equal(got, expected, err_msg=name)


@pytest.mark.step("A3")
def test_integer_index_gives_scalar_observation():
    """spec 3.2, rev 4: `data[idx]` is supported; an integer index gives scalar fields."""
    data = _data()
    _assert_indexed(data, 1)
    assert np.shape(data[1].lower) == ()


@pytest.mark.step("A3")
def test_negative_integer_index():
    """spec 3.2, rev 4: `data[idx]` follows array indexing, including negative indices."""
    _assert_indexed(_data(), -1)


@pytest.mark.step("A3")
def test_slice_index():
    """spec 3.2, rev 4: slicing over observation axes is supported."""
    _assert_indexed(_data(), slice(1, 4))


@pytest.mark.step("A3")
def test_strided_slice_index():
    """spec 3.2, rev 4: slicing over observation axes is supported (step 2)."""
    _assert_indexed(_data(), slice(None, None, 2))


@pytest.mark.step("A3")
def test_integer_array_index_for_minibatch():
    """spec 3.2, rev 4: indexing makes minibatches (integer-array index, with repetition)."""
    _assert_indexed(_data(), np.array([4, 0, 0, 2]))


@pytest.mark.step("A3")
def test_boolean_mask_index_for_split():
    """spec 3.2, rev 4: indexing makes train/validation/test splits (boolean mask)."""
    _assert_indexed(_data(), np.array([True, False, True, True, False]))


@pytest.mark.step("A3")
def test_indexing_multidimensional_observations():
    """spec 3.2, rev 4: `data[idx]` over multidimensional observation axes."""
    data = cx.SurvivalData(
        f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        b([[True, False, True], [False, True, False]]),
        left_trunc=f([0.5, 0.0, 0.25]),
    )
    _assert_indexed(data, 0)
    _assert_indexed(data, (slice(None), 1))
    _assert_indexed(data, (1, 2))


@pytest.mark.step("A3")
def test_tree_map_slicing_matches_indexing():
    """spec 3.2, rev 4: `jax.tree.map` slicing over observation axes is supported and agrees with `data[idx]`."""
    data = _data()
    idx = np.array([3, 1])
    mapped = jax.tree.map(lambda x: x[idx], data)
    assert isinstance(mapped, cx.SurvivalData)
    assert_same_fields(mapped, data[idx])


@pytest.mark.step("A3")
def test_indexing_carries_every_leaf_including_issue_bitfield():
    """spec 3.2, rev 4: every leaf has the observation shape, so indexing slices each leaf, the bitfield included."""
    data, _ = flag_data_and_issues(f([1.0, -1.0, 2.0, 3.0]), np.array([1, 2, 0, 1], dtype=np.int32))
    idx = np.array([1, 3])
    sub = data[idx]
    assert treedef(sub) == treedef(data)
    for got, parent in zip(leaves(sub), leaves(data), strict=True):
        np.testing.assert_array_equal(np.asarray(got), np.asarray(parent)[idx])


@pytest.mark.step("A3")
def test_slicing_inside_jit():
    """spec 3.2, rev 4: slicing works on data passed through jit (minibatches inside compiled code)."""
    data = _data()
    out = jax.jit(lambda d: d[1:3])(data)
    assert_same_fields(out, data[1:3])


@pytest.mark.step("A3")
@settings(derandomize=True, max_examples=100, deadline=None)
@given(
    st.integers(min_value=1, max_value=8).flatmap(
        lambda n: st.tuples(
            st.lists(st.booleans(), min_size=n, max_size=n),
            st.lists(st.integers(min_value=0, max_value=n - 1), min_size=0, max_size=2 * n),
        )
    )
)
def test_indexing_commutes_with_fields_property(case):
    """spec 3.2, rev 4: for any integer-array index, `data[idx].field == data.field[idx]`."""
    events, idx = case
    n = len(events)
    data = cx.SurvivalData(f(np.arange(1, n + 1)), b(events), left_trunc=f(np.full(n, 0.5)))
    _assert_indexed(data, np.asarray(idx, dtype=np.int32))
