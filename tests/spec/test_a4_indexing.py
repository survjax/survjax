"""A4: batch indexing, tree-map slicing and vmap over distributions (spec 4.1, 4.2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import assert_close, f, treedef


def _grid_power():
    """PowerHazard with batch (2, 3): shape varies by row, scale by column."""
    return fam.power_hazard()(shape=f([[1.5], [2.5]]), scale=f([2.0, 3.0, 4.0]))


@pytest.mark.step("A4")
def test_integer_index_gives_scalar_batch():
    """spec 4.1, rev 11: `dist[idx]` indexes batch dimensions."""
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0]))[1]
    assert tuple(dist.batch_shape) == ()
    np.testing.assert_array_equal(np.asarray(dist.rate), 1.0)


@pytest.mark.step("A4")
def test_slice_index_keeps_selected_entries():
    """spec 4.1, rev 11: `dist[idx]` with a slice selects batch entries."""
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0]))[1:]
    assert tuple(dist.batch_shape) == (2,)
    np.testing.assert_array_equal(np.asarray(dist.rate), [1.0, 2.0])


@pytest.mark.step("A4")
def test_integer_array_index_selects_entries():
    """spec 4.1, rev 11: `dist[idx]` with an integer array (a minibatch) selects batch entries."""
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0, 4.0]))[np.array([3, 0])]
    np.testing.assert_array_equal(np.asarray(dist.rate), [4.0, 0.5])


@pytest.mark.step("A4")
def test_first_axis_index_of_two_dimensional_batch():
    """spec 4.1, rev 11: indexing a `(2, 3)` batch on its first axis gives batch `(3,)` for every leaf."""
    row = _grid_power()[1]
    assert tuple(row.batch_shape) == (3,)
    np.testing.assert_array_equal(np.asarray(row.shape), [2.5, 2.5, 2.5])
    np.testing.assert_array_equal(np.asarray(row.scale), [2.0, 3.0, 4.0])


@pytest.mark.step("A4")
def test_second_axis_index_of_two_dimensional_batch():
    """spec 4.1, rev 11: `dist[:, j]` indexes the second batch axis."""
    column = _grid_power()[:, 2]
    assert tuple(column.batch_shape) == (2,)
    np.testing.assert_array_equal(np.asarray(column.shape), [1.5, 2.5])
    np.testing.assert_array_equal(np.asarray(column.scale), [4.0, 4.0])


@pytest.mark.step("A4")
def test_full_index_of_two_dimensional_batch():
    """spec 4.1, rev 11: indexing every batch axis gives a scalar distribution."""
    entry = _grid_power()[0, 1]
    assert tuple(entry.batch_shape) == ()
    np.testing.assert_array_equal(np.asarray(entry.shape), 1.5)
    np.testing.assert_array_equal(np.asarray(entry.scale), 3.0)


@pytest.mark.step("A4")
def test_indexing_keeps_component_axes():
    """spec 4.1/4.2, rev 11: batch indexing never touches trailing component axes."""
    w = f(np.arange(12.0).reshape(3, 4))
    entry = fam.component_rate()(rate=f([1.0, 2.0, 3.0]), w=w)[1]
    assert tuple(entry.batch_shape) == ()
    np.testing.assert_array_equal(np.asarray(entry.w), [4.0, 5.0, 6.0, 7.0])


@pytest.mark.step("A4")
def test_indexing_preserves_static_fields():
    """spec 4.2, rev 11: indexing changes leaves, not static configuration."""
    entry = fam.component_rate()(rate=f([1.0, 2.0]), w=f([3.0, 4.0]), method="max")[0]
    assert entry.method == "max"
    assert type(entry) is fam.component_rate()


@pytest.mark.step("A4")
def test_indexed_distribution_matches_indexed_values():
    """spec 4.1, rev 11: `dist[idx].log_prob(t)` equals `dist.log_prob(t)[idx]`.

    Both are ``log r - r t`` with ``t=2`` at the selected rates (1.0, 4.0).
    """
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0, 4.0]))
    idx = np.array([1, 3])
    assert_close(dist[idx].log_prob(f(2.0)), np.asarray(dist.log_prob(f(2.0)))[idx])
    assert_close(
        dist[idx].log_prob(f(2.0)), np.log([1.0, 4.0]) - 2.0 * np.array([1.0, 4.0])
    )


@pytest.mark.step("A4")
def test_tree_map_slicing_updates_batch_shape():
    """spec 4.1, rev 16: `batch_shape` follows the current leaves after `jax.tree.map` slicing (it is not stored)."""
    dist = fam.const_hazard()(rate=f(np.ones((4, 3))))
    assert jax.tree.map(lambda x: x[1:3], dist).batch_shape == (2, 3)


@pytest.mark.step("A4")
def test_tree_map_slicing_works():
    """spec 4.1, rev 11: "`jax.tree.map` slicing ... over a distribution work[s]"."""
    dist = fam.component_rate()(
        rate=f([1.0, 2.0, 3.0]), w=f(np.ones((3, 4))), method="max"
    )
    head = jax.tree.map(lambda x: x[:2], dist)
    assert type(head) is type(dist)
    assert head.method == "max"
    assert tuple(head.batch_shape) == (2,)
    assert np.shape(head.w) == (2, 4)


@pytest.mark.step("A4")
def test_tree_map_slice_matches_index_slice():
    """spec 4.1, rev 11: tree-map slicing and `dist[idx]` agree on leaves and tree structure."""
    dist = _grid_power()
    by_map = jax.tree.map(lambda x: x[1:], dist)
    by_index = dist[1:]
    assert treedef(by_map) == treedef(by_index)
    np.testing.assert_array_equal(np.asarray(by_map.shape), np.asarray(by_index.shape))
    np.testing.assert_array_equal(np.asarray(by_map.scale), np.asarray(by_index.scale))


@pytest.mark.step("A4")
def test_vmap_over_distribution_matches_batched_call():
    """spec 4.1, rev 11: `vmap` over a distribution works and agrees with the batched method."""
    dist = fam.power_hazard()(shape=f([0.5, 1.0, 3.0]), scale=f([2.0, 2.0, 2.0]))
    t = f(1.5)
    mapped = jax.vmap(lambda d: d.log_prob(t))(dist)
    assert_close(mapped, dist.log_prob(t))


@pytest.mark.step("A4")
def test_vmap_over_distribution_and_time_pairs_entries():
    """spec 4.1/5, rev 11: `vmap` over a distribution and a time vector pairs entries, like the batched call."""
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0]))
    t = f([1.0, 2.0, 3.0])
    mapped = jax.vmap(lambda d, s: d.log_prob(s))(dist, t)
    assert_close(mapped, dist.log_prob(t))


@pytest.mark.step("A4")
def test_batch_shape_inside_vmap_is_the_per_entry_shape():
    """spec 4.1, rev 16: `batch_shape` is "computed from the current parameter leaves ...; under `vmap` it is the per-entry shape"."""
    dist = fam.component_rate()(rate=f([1.0, 2.0, 3.0]), w=f(np.ones((3, 4))))
    out = jax.vmap(lambda d: jnp.zeros(d.batch_shape))(dist)
    assert out.shape == (3,)


@pytest.mark.step("A4")
def test_vmap_over_component_family_keeps_components_inside():
    """spec 4.1/4.2, rev 11: under `vmap` each entry sees its full component vector.

    ``log S(1) = -rate * sum(w)``: rows ``(1, (1,1,1,1))`` and ``(2, (2,2,2,2))`` give ``-4`` and ``-16``.
    """
    dist = fam.component_rate()(rate=f([1.0, 2.0]), w=f([[1.0] * 4, [2.0] * 4]))
    out = jax.vmap(lambda d: d.log_survival(f(1.0)))(dist)
    assert_close(out, [-4.0, -16.0])


@pytest.mark.step("A4")
def test_jit_of_indexing_with_traced_index():
    """spec 4.1, rev 11: batch indexing composes with jit (dynamic integer index)."""
    dist = fam.const_hazard()(rate=f([0.5, 1.0, 2.0]))
    rate = jax.jit(lambda d, i: d[i].rate)(dist, jnp.asarray(2))
    np.testing.assert_array_equal(np.asarray(rate), 2.0)
