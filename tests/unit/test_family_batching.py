"""Nested construction and batch selection (spec 3.3, 4.1–4.2, 7)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads

import survjax


@survjax.register_family
@dataclass(frozen=True, kw_only=True)
class Components(survjax.SurvivalDistribution):
    method: str = survjax.static(default="sum")
    rate: jax.Array = survjax.param()
    weights: jax.Array = survjax.param(component_ndim=2)


@survjax.register_family
@dataclass(frozen=True)
class Wrapper(survjax.SurvivalDistribution):
    base: survjax.SurvivalDistribution = survjax.distribution()
    multiplier: jax.Array = survjax.param()

    def log_survival(self, t):
        return -self.base.rate * self.multiplier * t

    def reference_time(self):
        return 1.0 / self.base.rate


@survjax.register_family
@dataclass(frozen=True)
class NestedOnly(survjax.SurvivalDistribution):
    base: survjax.SurvivalDistribution = survjax.distribution()


def test_nested_field_is_required_and_helper_takes_no_arguments():
    """Spec 4.2: nested fields have no default or helper arguments."""
    with pytest.raises(TypeError):
        Wrapper(multiplier=1)
    with pytest.raises(TypeError):
        survjax.distribution(default=None)


def test_two_levels_broadcast_components_without_mutating_baseline():
    """Spec 4.2/5: (2,1) and (3,) broadcast to (2,3), components untouched."""
    base = Components(rate=[[1], [2]], weights=np.arange(8).reshape(2, 1, 2, 2))
    dist = Wrapper(base=NestedOnly(base=base), multiplier=[3, 4, 5])
    assert dist.batch_shape == dist.base.batch_shape == (2, 3)
    assert dist.base.base.weights.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(dist.base.base.rate, [[1, 1, 1], [2, 2, 2]])
    np.testing.assert_array_equal(dist.base.base.weights[1, 2], [[4, 5], [6, 7]])
    assert base.batch_shape == (2, 1) and base.weights.shape == (2, 1, 2, 2)
    with pytest.raises(TypeError, match="batch shapes"):
        Wrapper(base=base, multiplier=jnp.ones((3, 2)))


def test_nested_round_trip_preserves_static_fields_and_sentinel_identity():
    """Spec 4.2: reconstruction bypasses constructors, even at nested levels."""
    dist = NestedOnly(
        base=Components(rate=[1, 2], weights=jnp.ones((2, 2)), method="max")
    )
    leaves, structure = jax.tree.flatten(dist)
    sentinels = [object() for _ in leaves]
    rebuilt = jax.tree.unflatten(structure, sentinels)
    assert isinstance(rebuilt.base, Components) and rebuilt.base.method == "max"
    assert all(a is b for a, b in zip(jax.tree.leaves(rebuilt), sentinels, strict=True))


@pytest.mark.parametrize(
    "index",
    [
        1,
        slice(1, None),
        np.array([1, 0]),
        (slice(None), 1),
        (1, 2),
        (Ellipsis, 1),
        (None, Ellipsis, 1),
        True,
        (Ellipsis, np.array([False, True, False])),
    ],
)
def test_indexing_preserves_two_component_axes_and_nested_configuration(index):
    """Spec 4.1–4.2: indices consume batch axes, never components."""
    weights = np.arange(24).reshape(2, 3, 2, 2)
    base = Components(rate=np.arange(6).reshape(2, 3), weights=weights, method="max")
    dist = NestedOnly(base=base)[index]
    # NumPy indexes the hand-built batch grid; component selection stays explicit.
    grid = np.arange(6).reshape(2, 3)[index]
    np.testing.assert_array_equal(dist.base.rate, grid)
    np.testing.assert_array_equal(dist.base.weights, weights.reshape(6, 2, 2)[grid])
    assert dist.batch_shape == grid.shape and dist.base.method == "max"


def test_index_cannot_consume_component_axes():
    """Spec 4.1: scalar batches still cannot index their components."""
    dist = Components(rate=1, weights=jnp.ones((2, 2)))
    with pytest.raises(IndexError):
        dist[0]
    np.testing.assert_array_equal(dist[...].weights, dist.weights)


def test_nested_tree_slicing_and_vmap_use_current_batch_shapes():
    """Spec 4.2: mapping reaches nested leaves and retains full components."""
    dist = Wrapper(
        base=Components(rate=[1, 2, 3], weights=jnp.ones((2, 2))), multiplier=[2, 3, 4]
    )
    sliced = jax.tree.map(lambda x: x[1:], dist)
    assert sliced.batch_shape == sliced.base.batch_shape == (2,)
    np.testing.assert_array_equal(sliced.base.rate, dist[1:].base.rate)

    def entry(d):
        assert d.batch_shape == d.base.batch_shape == ()
        assert d.base.weights.shape == (2, 2)
        return d.log_survival(2)

    # Hand derivation: -rate * multiplier * 2.
    np.testing.assert_array_equal(jax.vmap(entry)(dist), [-4, -12, -24])
    np.testing.assert_array_equal(
        jax.jit(lambda d, i: d[i].base.rate)(dist, jnp.array(2)), 3
    )


def test_nested_dtype_survives_broadcast_and_index(x64_enabled):
    """Spec 3.3: outer normalization does not promote nested leaves."""
    dtype = np.float64 if x64_enabled else np.float32
    dist = Wrapper(
        base=Components(rate=np.float32(2), weights=np.ones((2, 2), dtype=np.float32)),
        multiplier=np.ones(3, dtype=dtype),
    )
    assert dist[1].base.rate.dtype == dist.base.weights.dtype == np.float32
    assert dist.multiplier.dtype == dtype


def test_nested_construction_transforms_and_compilation_reuse():
    """Spec 4.1/7.2: equal structure/dtypes/shapes reuse the same compilation."""

    def build(rate, multiplier=2, method="sum"):
        base = Components(rate=rate, weights=jnp.ones((2, 2)), method=method)
        return Wrapper(base=base, multiplier=multiplier)

    rates = jnp.array([1.0, 2.0, 3.0])
    traces = []

    def evaluate(d):
        traces.append(None)
        return d.log_survival(2)

    compiled = jax.jit(evaluate)
    compiled(build(rates)).block_until_ready()
    compiled(build(rates + 1, multiplier=3)).block_until_ready()
    assert len(traces) == 1
    compiled(build(rates, method="max")).block_until_ready()
    assert len(traces) == 2


def test_gradients_through_nested_broadcast_and_selection(x64_enabled):
    """Spec 7.3: parameter derivatives survive broadcasting and indexing."""
    from impl_tolerances import DERIVATIVE_ATOL, DERIVATIVE_RTOL

    def value(rate, multiplier):
        return Wrapper(
            base=Components(rate=rate, weights=jnp.ones((2, 2))),
            multiplier=jnp.stack([multiplier, multiplier]),
        )[1].log_survival(2)

    args = (jnp.asarray(1.5), jnp.asarray(2.0))
    # f=-2*r*m: df/dr=-2*m=-4, df/dm=-2*r=-3.
    np.testing.assert_array_equal(
        jax.jit(jax.grad(value, argnums=(0, 1)))(*args), [-4, -3]
    )
    check_grads(
        value,
        args,
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )


@pytest.mark.parametrize("value", [None, jnp.ones(3)])
def test_nested_field_rejects_non_distributions(value):
    """Spec 4.2/6.2: static nested-field misuse names the field in TypeError."""
    with pytest.raises(TypeError, match="base must be a registered distribution"):
        Wrapper(base=value, multiplier=1)


def test_nested_field_rejects_unregistered_dataclass():
    """Spec 4.2: a baseline needs explicit registration before nesting."""

    @dataclass(frozen=True)
    class Unregistered(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

    with pytest.raises(TypeError, match="base must be a registered distribution"):
        Wrapper(base=Unregistered(rate=jnp.asarray(1.0)), multiplier=1)


def test_nested_field_rejects_inherited_registration():
    """Spec 4.2: a parent's registration does not register its subclasses."""

    @dataclass(frozen=True, kw_only=True)
    class UnregisteredSubclass(Components):
        pass

    base = UnregisteredSubclass(rate=jnp.asarray(1.0), weights=jnp.ones((2, 2)))
    with pytest.raises(TypeError, match="base must be a registered distribution"):
        Wrapper(base=base, multiplier=1)
