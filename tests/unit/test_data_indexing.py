"""Observation-axis indexing identities (spec 3.2, 6.2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chronox_jax import SurvivalData


@pytest.mark.parametrize(
    "idx",
    [
        1,
        -1,
        slice(None, None, 2),
        slice(0, 0),
        np.array([2, 0, 0]),
        np.array([True, False, True]),
    ],
    ids=["scalar", "negative", "stride", "empty", "minibatch", "boolean"],
)
def test_indexing_preserves_each_leaf(idx: object) -> None:
    """Spec 3.2: indexing is the fieldwise array identity, including raw bits."""
    data = SurvivalData([1.0, 2.0, 3.0], [1, 2, 0])
    before, structure = jax.tree.flatten(data)
    selected = data[idx]
    assert jax.tree.structure(selected) == structure
    for parent, child in zip(before, jax.tree.leaves(selected), strict=True):
        np.testing.assert_array_equal(child, parent[idx])
        assert child.dtype == parent.dtype
    mapped = jax.tree.map(lambda leaf: leaf[idx], data)
    for actual, expected in zip(
        jax.tree.leaves(selected), jax.tree.leaves(mapped), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(data.lower, [1, 2, 3])


def test_multidimensional_indexing_and_transforms() -> None:
    """Spec 3.2/12.1: tuple indexing, jit and vmap preserve observation axes."""
    data = SurvivalData([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(data[:, 1].lower, [2, 4])
    assert data[1, 0].lower.shape == ()
    for selected in (data[1], jax.jit(lambda d: d[1])(data)):
        np.testing.assert_array_equal(selected.lower, [3, 4])
    batched = jax.vmap(lambda i: data[i])(jnp.array([1, 0]))
    np.testing.assert_array_equal(batched.lower, [[3, 4], [1, 2]])


def test_indexed_and_rebuilt_issues_keep_raw_evidence() -> None:
    """Spec 3.2/6.2: raw invalid codes survive; ok reduces only selected rows."""
    data = SurvivalData([1.0, 2.0, 3.0], [1, 2, 0])
    # Hand derivation: event code 2 is invalid even after bool normalization.
    selected = data[jnp.array([1, 0, 1])]
    leaves, structure = jax.tree.flatten(selected)
    rebuilt = jax.tree.unflatten(structure, leaves)
    for result in (selected, rebuilt, jax.jit(lambda d: d)(selected)):
        np.testing.assert_array_equal(
            result.issues.invalid_indicator, [True, False, True]
        )
        assert not bool(result.issues.ok)
    assert bool(jax.jit(lambda d: d[0].issues.ok)(data))
    assert not bool(jax.jit(lambda d: d[1].issues.ok)(data))
    assert bool(data[:0].issues.ok)
