"""Observation shape, leading-axis length and truth testing (spec 3.2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from survjax import SurvivalData


@pytest.mark.parametrize("bounds", [False, True], ids=["flags", "bounds"])
@pytest.mark.parametrize("shape", [(), (3,), (0,), (2, 3), (2, 0)])
def test_size_follows_input_dimensions(shape: tuple[int, ...], bounds: bool) -> None:
    """Spec 3.2: both constructors expose input dimensions, including empty axes."""
    # Expected dimensions come from these explicit array shapes.
    time = jnp.ones(shape)
    data = (
        SurvivalData.from_bounds(time, time, exact=True)
        if bounds
        else SurvivalData(time)
    )
    assert data.shape == shape
    assert isinstance(data.shape, tuple)
    assert all(type(dimension) is int for dimension in data.shape)
    assert not hasattr(data, "size")
    assert not hasattr(data, "ndim")
    with pytest.raises(AttributeError):
        data.shape = (5,)
    if shape:
        assert type(len(data)) is int
        assert len(data) == shape[0]
    else:
        with pytest.raises(TypeError, match="scalar"):
            len(data)


def test_indexing_and_reconstruction_keep_size_and_tree_structure() -> None:
    """Spec 3.2: size follows selected leaves without adding PyTree metadata."""
    data = SurvivalData(jnp.ones((2, 3)))
    structure = jax.tree.structure(data)
    for selected, shape in (
        (data[0], (3,)),
        (data[:, :2], (2, 2)),
        (data[:0], (0, 3)),
        (jax.tree.map(lambda leaf: leaf[:, 1], data), (2,)),
    ):
        assert selected.shape == shape
        assert len(selected) == shape[0]
        assert jax.tree.structure(selected) == structure
        leaves, selected_structure = jax.tree.flatten(selected)
        assert len(leaves) == 6
        assert selected.tree_flatten()[1] == data.tree_flatten()[1]
        rebuilt = jax.tree.unflatten(selected_structure, leaves)
        assert rebuilt.shape == shape
        assert len(rebuilt) == shape[0]
        for actual, expected in zip(jax.tree.leaves(rebuilt), leaves, strict=True):
            np.testing.assert_array_equal(actual, expected)
    scalar = data[0][1]
    assert scalar.shape == ()
    with pytest.raises(TypeError, match="scalar"):
        len(scalar)


def test_size_is_static_inside_jit() -> None:
    """Spec 3.2: tracing reads Python shape/length from the current leaves."""

    def query(data: SurvivalData) -> tuple[tuple[int, ...], int]:
        assert isinstance(data.shape, tuple)
        assert all(type(dimension) is int for dimension in data.shape)
        assert type(len(data)) is int
        return data.shape, len(data)

    compiled = jax.jit(query)
    for shape in ((3,), (0,), (2, 3), (2, 0)):
        data = SurvivalData(jnp.ones(shape))
        assert query(data) == (shape, shape[0])
        dimensions, length = compiled(data)
        assert tuple(int(dimension) for dimension in dimensions) == shape
        assert int(length) == shape[0]


def test_scalar_shape_and_length_inside_jit() -> None:
    """Spec 3.2: scalar shape is empty and scalar length raises while tracing."""
    scalar = SurvivalData(1.0)
    assert jax.jit(lambda data: data.shape)(scalar) == ()
    with pytest.raises(TypeError, match="scalar"):
        jax.jit(len)(scalar)


def test_vmap_size_describes_each_example() -> None:
    """Spec 3.2: mapped matrix rows have shape (3,) and length 3."""

    def query(row: SurvivalData) -> int:
        assert row.shape == (3,)
        assert type(row.shape[0]) is int
        assert type(len(row)) is int
        return len(row)

    data = SurvivalData(jnp.ones((2, 3)))
    # Two mapped rows, each containing three observations.
    np.testing.assert_array_equal(jax.vmap(query)(data), [3, 3])
    np.testing.assert_array_equal(jax.jit(jax.vmap(query))(data), [3, 3])


def test_vmap_vector_sees_scalar_examples() -> None:
    """Spec 3.2: mapping over a vector leaves scalar examples without length."""

    def query(row: SurvivalData) -> int:
        assert row.shape == ()
        with pytest.raises(TypeError, match="scalar"):
            len(row)
        return 0

    data = SurvivalData(jnp.ones(3))
    np.testing.assert_array_equal(jax.vmap(query)(data), [0, 0, 0])
    np.testing.assert_array_equal(jax.jit(jax.vmap(query))(data), [0, 0, 0])


@pytest.mark.parametrize("shape", [(), (0,), (3,), (2, 3), (2, 0)])
def test_truth_testing_always_raises(shape: tuple[int, ...]) -> None:
    """Spec 3.2: bool and conditionals never fall back to emptiness testing."""
    data = SurvivalData(jnp.ones(shape))
    with pytest.raises(TypeError, match="truth value"):
        bool(data)
    with pytest.raises(TypeError, match="truth value"):
        if data:
            pytest.fail("truth testing must raise before entering the branch")
    with pytest.raises(TypeError, match="truth value"):
        jax.jit(bool)(data)


@pytest.mark.parametrize("shape", [(3,), (2, 3)])
def test_vmap_truth_testing_raises(shape: tuple[int, ...]) -> None:
    """Spec 3.2: scalar and vector mapped examples both reject truth testing."""
    data = SurvivalData(jnp.ones(shape))
    with pytest.raises(TypeError, match="truth value"):
        jax.vmap(bool)(data)
    with pytest.raises(TypeError, match="truth value"):
        jax.jit(jax.vmap(bool))(data)
