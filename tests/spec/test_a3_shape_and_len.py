"""A3: `data.shape`, `len(data)` and the truth value of SurvivalData (spec 3.2, rev 36)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import b, f, leaves, treedef

import survjax


def _vector(n=5):
    return survjax.SurvivalData(f(np.arange(1, n + 1)), b(np.arange(n) % 2 == 0))


def _matrix():
    return survjax.SurvivalData(
        f([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        b([[True, False, True], [False, True, False]]),
    )


def _assert_python_int_tuple(shape, expected):
    assert type(shape) is tuple
    assert all(type(n) is int for n in shape), shape
    assert shape == expected


# --- data.shape -----------------------------------------------------------


@pytest.mark.step("A3")
def test_shape_of_vector_data_is_a_tuple_of_python_ints():
    """spec 3.2, rev 36: `data.shape` is the observation shape as a tuple of Python ints, `(N,)`."""
    _assert_python_int_tuple(_vector(5).shape, (5,))


@pytest.mark.step("A3")
def test_shape_of_scalar_data_is_empty_tuple():
    """spec 3.2, rev 36: `data.shape` of scalar data is `()`."""
    _assert_python_int_tuple(survjax.SurvivalData(1.0, True).shape, ())


@pytest.mark.step("A3")
def test_shape_of_empty_data():
    """spec 3.2, rev 36: empty data `(0,)` are allowed; `data.shape == (0,)`."""
    data = survjax.SurvivalData(f([]), b([]))
    _assert_python_int_tuple(data.shape, (0,))


@pytest.mark.step("A3")
def test_shape_of_two_dimensional_data():
    """spec 3.2, rev 36: `data.shape` is the full multidimensional observation shape."""
    _assert_python_int_tuple(_matrix().shape, (2, 3))


@pytest.mark.step("A3")
def test_shape_from_bounds_constructor():
    """spec 3.2, rev 36: `data.shape` is the observation shape for the bounds form too."""
    data = survjax.SurvivalData.from_bounds(f([1.0, 2.0, 3.0]), f([1.0, np.inf, 4.0]))
    _assert_python_int_tuple(data.shape, (3,))


@pytest.mark.step("A3")
def test_shape_equals_field_shape():
    """spec 3.2, rev 36: `data.shape` is the observation shape, which every field has (rev 4)."""
    for data in (_vector(4), _matrix(), survjax.SurvivalData(2.0, False)):
        assert data.shape == tuple(np.shape(data.lower))


@pytest.mark.step("A3")
def test_shape_after_slicing():
    """spec 3.2/12.1, rev 36: `data.shape` after indexing with a slice."""
    _assert_python_int_tuple(_vector(5)[1:4].shape, (3,))


@pytest.mark.step("A3")
def test_shape_after_integer_index_is_scalar():
    """spec 3.2/12.1, rev 36: an integer index gives scalar data, `shape == ()`."""
    _assert_python_int_tuple(_vector(5)[2].shape, ())


@pytest.mark.step("A3")
def test_shape_after_integer_array_index():
    """spec 3.2/12.1, rev 36: `data.shape` after an integer-array (minibatch) index."""
    _assert_python_int_tuple(_vector(5)[np.array([0, 0, 3, 4, 1, 2, 2])].shape, (7,))


@pytest.mark.step("A3")
def test_shape_after_indexing_two_dimensional_data():
    """spec 3.2/12.1, rev 36: `data.shape` after indexing two-dimensional data."""
    data = _matrix()
    _assert_python_int_tuple(data[1].shape, (3,))
    _assert_python_int_tuple(data[:, 1].shape, (2,))
    _assert_python_int_tuple(data[1, 2].shape, ())


@pytest.mark.step("A3")
def test_shape_after_empty_slice():
    """spec 3.2/12.1, rev 36: an empty slice gives `(0,)` data."""
    _assert_python_int_tuple(_vector(5)[:0].shape, (0,))


@pytest.mark.step("A3")
def test_shape_is_read_from_stored_leaves():
    """spec 3.2, rev 36: `data.shape` is read from the stored leaves, not kept in the auxiliary data."""
    data = _vector(5)
    rebuilt = jax.tree_util.tree_unflatten(treedef(data), [x[:2] for x in leaves(data)])
    _assert_python_int_tuple(rebuilt.shape, (2,))


@pytest.mark.step("A3")
def test_shape_is_not_in_the_tree_structure():
    """spec 3.2, rev 36: shape is not auxiliary data, so data differing only in `N` share a tree structure."""
    assert treedef(_vector(4)) == treedef(_vector(6))


@pytest.mark.step("A3")
def test_shape_is_static_under_jit():
    """spec 3.2/12.1, rev 36: under `jit`, `data.shape` is a tuple of Python ints usable as a static shape."""
    seen = []

    def fn(d):
        seen.append(d.shape)
        return jnp.zeros(d.shape)

    out = jax.jit(fn)(_matrix())
    assert out.shape == (2, 3)
    _assert_python_int_tuple(seen[0], (2, 3))


@pytest.mark.step("A3")
def test_shape_is_per_example_under_vmap():
    """spec 3.2/12.1, rev 36: under `vmap`, `data.shape` is the per-example shape."""
    seen = []

    def fn(d):
        seen.append(d.shape)
        return d.lower

    jax.vmap(fn)(_vector(5))
    _assert_python_int_tuple(seen[0], ())


@pytest.mark.step("A3")
def test_shape_is_per_example_under_vmap_of_two_dimensional_data():
    """spec 3.2/12.1, rev 36: `vmap` over the leading axis of `(2, 3)` data gives per-example shape `(3,)`."""
    seen = []

    def fn(d):
        seen.append(d.shape)
        return d.lower

    jax.vmap(fn)(_matrix())
    _assert_python_int_tuple(seen[0], (3,))


@pytest.mark.step("A3")
def test_no_size_attribute():
    """spec 3.2, rev 36: "There is no `size` or `ndim`" (`size`)."""
    assert not hasattr(_vector(3), "size")


@pytest.mark.step("A3")
def test_no_ndim_attribute():
    """spec 3.2, rev 36: "There is no `size` or `ndim`" (`ndim`)."""
    assert not hasattr(_vector(3), "ndim")


# --- len(data) ------------------------------------------------------------


@pytest.mark.step("A3")
def test_len_of_vector_data_is_number_of_observations():
    """spec 3.2, rev 36: `len(data)` is `data.shape[0]`, the number of observations for `(N,)` data."""
    assert len(_vector(5)) == 5


@pytest.mark.step("A3")
def test_len_of_scalar_data_raises_type_error():
    """spec 3.2, rev 36: on scalar data `len` raises `TypeError`, as for a 0-d array."""
    with pytest.raises(TypeError):
        len(survjax.SurvivalData(1.0, True))


@pytest.mark.step("A3")
def test_len_of_integer_indexed_data_raises_type_error():
    """spec 3.2/12.1, rev 36: `data[i]` is scalar data, so `len` raises `TypeError`."""
    with pytest.raises(TypeError):
        len(_vector(5)[0])


@pytest.mark.step("A3")
def test_len_of_empty_data_is_zero():
    """spec 3.2, rev 36: empty data `(0,)` are allowed and have length 0."""
    assert len(survjax.SurvivalData(f([]), b([]))) == 0


@pytest.mark.step("A3")
def test_len_of_empty_slice_is_zero():
    """spec 3.2/12.1, rev 36: an empty slice has length 0."""
    assert len(_vector(5)[:0]) == 0


@pytest.mark.step("A3")
def test_len_of_two_dimensional_data_counts_leading_axis():
    """spec 3.2, rev 36: for multidimensional data, `len` counts only the leading axis."""
    data = _matrix()
    assert len(data) == 2
    assert math.prod(data.shape) == 6


@pytest.mark.step("A3")
def test_len_after_slicing():
    """spec 3.2/12.1, rev 36: `len(data)` after indexing."""
    data = _vector(5)
    assert len(data[1:4]) == 3
    assert len(data[np.array([True, False, True, True, False])]) == 3


@pytest.mark.step("A3")
def test_len_under_jit_is_static():
    """spec 3.2/12.1, rev 36: under `jit`, `len(data)` is a Python int (static)."""
    seen = []

    def fn(d):
        seen.append(len(d))
        return jnp.zeros(len(d))

    out = jax.jit(fn)(_vector(4))
    assert out.shape == (4,)
    assert type(seen[0]) is int and seen[0] == 4


@pytest.mark.step("A3")
def test_len_under_vmap_of_vector_data_raises_type_error():
    """spec 3.2/12.1, rev 36: per example under `vmap`, `(N,)` data are scalar, so `len` raises `TypeError`."""
    with pytest.raises(TypeError):
        jax.vmap(lambda d: jnp.zeros(len(d)))(_vector(3))


@pytest.mark.step("A3")
def test_len_under_vmap_of_two_dimensional_data_is_per_example():
    """spec 3.2/12.1, rev 36: `vmap` over `(2, 3)` data gives per-example `len == 3`."""
    seen = []

    def fn(d):
        seen.append(len(d))
        return d.lower

    jax.vmap(fn)(_matrix())
    assert seen[0] == 3


# --- truth value ----------------------------------------------------------


@pytest.mark.step("A3")
def test_bool_of_nonempty_data_raises_type_error():
    """spec 3.2/12.1, rev 36: the truth value is undefined; `bool(data)` raises `TypeError`."""
    with pytest.raises(TypeError):
        bool(_vector(3))


@pytest.mark.step("A3")
def test_bool_of_empty_data_raises_type_error():
    """spec 3.2, rev 36: `bool(data)` raises "instead of testing emptiness", also for `(0,)` data."""
    with pytest.raises(TypeError):
        bool(survjax.SurvivalData(f([]), b([])))


@pytest.mark.step("A3")
def test_bool_of_scalar_data_raises_type_error():
    """spec 3.2, rev 36: `bool(data)` raises `TypeError` for scalar data too."""
    with pytest.raises(TypeError):
        bool(survjax.SurvivalData(1.0, True))


@pytest.mark.step("A3")
def test_if_data_raises_type_error():
    """spec 3.2, rev 36: "(and so `if data:`) raises `TypeError`"."""
    data = _vector(2)
    with pytest.raises(TypeError):
        if data:
            pass
