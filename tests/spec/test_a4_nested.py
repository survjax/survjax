"""A4: nested distribution fields share the outer batch shape (spec 4.2, rev 18).

Uses ``ScaledHazard`` (``spec_families.py``): ``log S = exp(a) * log S_base``.
With a ConstHazard base of rate ``r`` the hazard is the constant ``exp(a) r``,
so ``log f(t) = a + log r - exp(a) r t``.
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import MISUSE, X64, assert_close, f, leaves, treedef

import survjax

requires_x64 = pytest.mark.skipif(not X64, reason="needs float64 next to float32")


def _scaled(rate, log_ratio):
    return fam.scaled_hazard()(base=fam.const_hazard()(rate=rate), log_ratio=log_ratio)


# --------------------------------------------------------------- batch shape and broadcasting


@pytest.mark.step("A4")
def test_outer_batch_shape_includes_parameter_batch_with_scalar_base():
    """spec 4.2, rev 18: the outer `batch_shape` is the broadcast of parameter and nested batch shapes."""
    dist = _scaled(f(0.5), f([0.0, 0.1, 0.2]))
    assert dist.batch_shape == (3,)


@pytest.mark.step("A4")
def test_scalar_base_is_broadcast_to_outer_batch():
    """spec 4.2, rev 18: "nested distributions are broadcast to it" at construction."""
    dist = _scaled(f(0.5), f([0.0, 0.1, 0.2]))
    np.testing.assert_array_equal(np.asarray(dist.base.rate), [0.5, 0.5, 0.5])
    assert dist.base.batch_shape == (3,)


@pytest.mark.step("A4")
def test_outer_batch_shape_comes_from_nested_distribution():
    """spec 4.2, rev 18: a nested batch shape counts like a parameter's; outer parameters broadcast to it."""
    dist = _scaled(f([0.5, 1.0, 2.0]), f(0.1))
    assert dist.batch_shape == (3,)
    assert np.shape(dist.log_ratio) == (3,)


@pytest.mark.step("A4")
def test_nested_and_parameter_batches_broadcast_together():
    """spec 4.2, rev 18: base `(2, 1)` and outer parameter `(3,)` give batch `(2, 3)` for every leaf."""
    dist = _scaled(f([[0.5], [1.0]]), f([0.0, 0.1, 0.2]))
    assert dist.batch_shape == (2, 3)
    assert all(np.shape(leaf) == (2, 3) for leaf in leaves(dist))


@pytest.mark.step("A4")
def test_nested_component_axis_stays_trailing():
    """spec 4.2, rev 18: broadcasting a nested distribution keeps its component axes."""
    base = fam.component_rate()(rate=f(1.0), w=f([1.0, 2.0, 3.0, 4.0]))
    dist = fam.scaled_hazard()(base=base, log_ratio=f([0.0, 0.1, 0.2]))
    assert dist.batch_shape == (3,)
    assert np.shape(dist.base.w) == (3, 4)
    assert np.shape(dist.base.rate) == (3,)


@pytest.mark.step("A4")
def test_incompatible_nested_and_parameter_batches_are_rejected():
    """spec 4.1/4.2, rev 18: nested and parameter batches that do not broadcast raise `TypeError`."""
    with pytest.raises(MISUSE):
        _scaled(f([0.5, 1.0]), f([0.0, 0.1, 0.2]))


@pytest.mark.step("A4")
@requires_x64
def test_broadcast_nested_distribution_keeps_its_dtype():
    """spec 3.3/4.2, rev 18: broadcasting keeps the nested distribution's dtype (rev 15: nested dtypes are kept)."""
    dist = _scaled(np.float32(0.5), np.array([0.0, 0.1, 0.2], dtype=np.float64))
    assert np.asarray(dist.base.rate).dtype == np.dtype(np.float32)
    assert np.shape(dist.base.rate) == (3,)


# --------------------------------------------------------------- PyTree structure


@pytest.mark.step("A4")
def test_nested_static_fields_are_part_of_outer_structure():
    """spec 4.2, rev 18: "nested static fields are part of its structure"."""

    def build(method):
        base = fam.component_rate()(rate=f(1.0), w=f([1.0, 2.0]), method=method)
        return fam.scaled_hazard()(base=base, log_ratio=f(0.0))

    assert treedef(build("sum")) == treedef(build("sum"))
    assert treedef(build("sum")) != treedef(build("max"))


@pytest.mark.step("A4")
def test_distribution_helper_takes_no_arguments():
    """spec 4.2, rev 18: "`survjax.distribution()` takes no arguments in v0"."""
    with pytest.raises(TypeError):
        survjax.distribution(fam.const_hazard()(rate=f(1.0)))


def _unregistered_subclass_instance(rate):
    """An instance of an unregistered subclass of the registered ConstHazard.

    Built with ``object.__new__`` so that only the nested-field check is
    observed, not whatever the inherited initializer does with an unregistered
    class.
    """

    class NotRegistered(fam.const_hazard()):
        pass

    obj = object.__new__(NotRegistered)
    object.__setattr__(obj, "rate", rate)
    return obj


@pytest.mark.step("A4")
def test_nested_field_rejects_unregistered_subclass_instance():
    """spec 4.2, rev 33: a `distribution()` field holding an instance of an unregistered subclass of a registered family raises `TypeError`."""
    with pytest.raises(TypeError):
        fam.scaled_hazard()(base=_unregistered_subclass_instance(f(1.0)), log_ratio=f(0.0))


@pytest.mark.step("A4")
def test_nested_field_error_names_the_field():
    """spec 4.2, rev 33: the `TypeError` for an unregistered nested value names the field."""
    with pytest.raises(TypeError) as info:
        fam.scaled_hazard()(base=_unregistered_subclass_instance(f(1.0)), log_ratio=f(0.0))
    assert "base" in str(info.value)


@pytest.mark.step("A4")
def test_nested_field_rejects_non_distribution_before_reading_batch_shape():
    """spec 4.2, rev 33: a non-distribution value raises `TypeError` "before nested batch shapes are read".

    A plain ``object()`` has no batch shape, so reading one first would raise
    ``AttributeError`` instead.
    """
    with pytest.raises(TypeError):
        fam.scaled_hazard()(base=object(), log_ratio=f(0.0))


@pytest.mark.step("A4")
def test_nested_field_rejects_array_value():
    """spec 4.2, rev 33: an array is not an instance of a registered class; a `distribution()` field rejects it with `TypeError`."""
    with pytest.raises(TypeError):
        fam.scaled_hazard()(base=f([1.0, 2.0]), log_ratio=f(0.0))


# --------------------------------------------------------------- indexing, slicing, vmap


@pytest.mark.step("A4")
def test_indexing_applies_to_nested_leaves():
    """spec 4.2, rev 18: `dist[idx]` indexes nested leaves too."""
    entry = _scaled(f([0.5, 1.0, 2.0]), f([0.0, 0.1, 0.2]))[2]
    assert entry.batch_shape == ()
    np.testing.assert_array_equal(np.asarray(entry.base.rate), 2.0)
    assert_close(entry.log_ratio, 0.2)


@pytest.mark.step("A4")
def test_indexing_a_broadcast_scalar_base():
    """spec 4.2, rev 18: a base that was scalar before broadcasting is indexed like any batch leaf."""
    part = _scaled(f(0.5), f([0.0, 0.1, 0.2]))[1:]
    assert part.batch_shape == (2,)
    np.testing.assert_array_equal(np.asarray(part.base.rate), [0.5, 0.5])


@pytest.mark.step("A4")
def test_tree_map_slicing_applies_to_nested_leaves():
    """spec 4.2, rev 18: `jax.tree.map` slicing reaches nested leaves."""
    head = jax.tree.map(lambda x: x[:2], _scaled(f([0.5, 1.0, 2.0]), f(0.1)))
    assert head.batch_shape == (2,)
    np.testing.assert_array_equal(np.asarray(head.base.rate), [0.5, 1.0])


@pytest.mark.step("A4")
def test_vmap_over_outer_distribution_with_broadcast_base():
    """spec 4.2, rev 18: `vmap` over a family built from a scalar base works and matches the batched call.

    ``log f(t) = a + log r - exp(a) r t`` with ``r=0.5``, ``t=2``, ``a = (0, 0.1, 0.2)``.
    """
    a = np.array([0.0, 0.1, 0.2])
    dist = _scaled(f(0.5), f(a))
    mapped = jax.vmap(lambda d: d.log_prob(f(2.0)))(dist)
    assert_close(mapped, a + np.log(0.5) - np.exp(a) * 0.5 * 2.0)
    assert_close(mapped, dist.log_prob(f(2.0)))


@pytest.mark.step("A4")
def test_broadcast_nested_values_in_log_prob():
    """spec 4.2, rev 18: values with a `(2, 3)` broadcast of base and outer parameter.

    ``log f(t) = a + log r - exp(a) r t`` with ``r = (0.5, 1)`` as a column,
    ``a = (0, 0.1, 0.2)`` and ``t = 1``.
    """
    r = np.array([[0.5], [1.0]])
    a = np.array([0.0, 0.1, 0.2])
    out = _scaled(f(r), f(a)).log_prob(f(1.0))
    assert_close(out, a + np.log(r) - np.exp(a) * r)
