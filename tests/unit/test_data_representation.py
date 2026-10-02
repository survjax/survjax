"""Hand-checked normalized observation storage."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from survjax import SurvivalData


def test_mixed_kinds_have_meaningful_bounds() -> None:
    """Spec 3.2: exact, right, left, and interval rows use the stated sets."""
    data = SurvivalData(
        jnp.array([1, 2, 3, 1], dtype=jnp.float32),
        right_censored=[False, True, False, False],
        left_censored=[False, False, True, False],
        interval_censored=[False, False, False, True],
        time2=np.array([np.nan, np.nan, np.nan, 4], dtype=np.float32),
    )
    np.testing.assert_array_equal(data.exact, [True, False, False, False])
    np.testing.assert_array_equal(data.lower, [1, 2, 0, 1])
    np.testing.assert_array_equal(data.upper, [1, np.inf, 3, 4])
    np.testing.assert_array_equal(data.left_trunc, [0, 0, 0, 0])
    np.testing.assert_array_equal(data.right_trunc, [np.inf] * 4)


def test_bounds_constructor_keeps_shape_and_truncation() -> None:
    """Spec 3.2: first lower array fixes observation shape and all bounds align."""
    data = SurvivalData.from_bounds(
        jnp.array([[1, 0], [2, 3]], dtype=jnp.float32),
        jnp.array([[1, 4], [jnp.inf, 5]], dtype=jnp.float32),
        exact=[[True, False], [False, False]],
        left_trunc=0,
        right_trunc=10,
    )
    for leaf in jax.tree.leaves(data):
        assert leaf.shape == (2, 2)
    np.testing.assert_array_equal(data.upper, [[1, 4], [np.inf, 5]])
    np.testing.assert_array_equal(data.left_trunc, np.zeros((2, 2)))
    np.testing.assert_array_equal(data.right_trunc, np.full((2, 2), 10))


def test_flag_cannot_enlarge_observation_shape() -> None:
    """Spec 3.2: a column flag cannot enlarge a vector of observations."""
    with pytest.raises(TypeError, match="enlarge"):
        SurvivalData(
            jnp.array([1, 2], dtype=jnp.float32),
            right_censored=jnp.array([[False], [True]]),
        )


def test_exact_bounds_replace_nan_upper_with_lower() -> None:
    """Spec 3.2: the exact row's stored upper always equals its lower."""
    data = SurvivalData.from_bounds(
        jnp.array([2, 1], dtype=jnp.float32),
        jnp.array([jnp.nan, 3], dtype=jnp.float32),
        exact=[True, False],
    )
    np.testing.assert_array_equal(data.upper, [2, 3])


def test_pytree_has_six_leaves_and_crosses_jit() -> None:
    """Spec 3.2: the registered PyTree contains five fields and one issue leaf."""
    data = SurvivalData(jnp.array([1, 2], dtype=jnp.float32), [True, False])
    leaves = jax.tree.leaves(data)
    assert len(leaves) == 6
    assert [leaf.dtype for leaf in leaves] == [
        jnp.dtype(bool),
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.float32),
        jnp.dtype(jnp.int32),
    ]
    result = jax.jit(lambda d: d)(data)
    np.testing.assert_array_equal(result.upper, [1, np.inf])
    np.testing.assert_array_equal(jax.vmap(lambda d: d.lower)(data), [1, 2])


def test_unflatten_bypasses_normalization() -> None:
    """Spec 3.2: arbitrary tree leaves reconstruct without validation."""
    data = SurvivalData(jnp.array(1, dtype=jnp.float32))
    structure = jax.tree.structure(data)
    rebuilt = jax.tree.unflatten(structure, [object() for _ in range(6)])
    assert isinstance(rebuilt, SurvivalData)
    with pytest.raises(AttributeError):
        data.lower = jnp.array(2)


def test_data_equality_uses_identity() -> None:
    """Spec 3.2: array fields do not produce ambiguous dataclass equality."""
    data = SurvivalData(jnp.array([1, 2], dtype=jnp.float32))
    rebuilt = jax.tree.map(lambda value: value, data)
    assert data == data
    assert data != rebuilt
    assert isinstance(hash(data), int)
