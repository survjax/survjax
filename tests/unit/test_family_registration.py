"""Independent flat-family construction cases (spec 3.3, 4.1–4.2, 5)."""

import dataclasses
import inspect
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import survjax


@survjax.register_family
@dataclasses.dataclass(frozen=True, kw_only=True)
class Recipe(survjax.SurvivalDistribution):
    method: str = survjax.static(default="sum")
    rate: jax.Array = survjax.param()
    weights: jax.Array = survjax.param(component_ndim=1)


def test_keyword_only_required_parameters_and_frozen_fields():
    """Spec 4.2: defaults may precede required fields; instances are frozen."""
    for arguments in ({"rate": 1.0}, {"rate": 1.0, "weights": [1.0], "dtype": float}):
        with pytest.raises(TypeError):
            Recipe(**arguments)
    with pytest.raises(TypeError):
        Recipe(1.0, [1.0])
    recipe = Recipe(rate=1.0, weights=[1.0])
    assert recipe.method == "sum"
    assert not hasattr(recipe, "shape")
    with pytest.raises(dataclasses.FrozenInstanceError):
        recipe.rate = jnp.asarray(2.0)
    with pytest.raises(TypeError):
        Recipe(rate=1.0, weights=[1.0], method=[])


def test_broadcasting_preserves_components_and_values():
    """Spec 4.1/5: (2,1) and (3,) batch axes broadcast to (2,3)."""
    recipe = Recipe(
        rate=np.array([[1.0], [2.0]], dtype=np.float32),
        weights=np.ones((3, 4), dtype=np.float32),
    )
    assert recipe.batch_shape == (2, 3)
    assert recipe.weights.shape == (2, 3, 4)
    assert recipe.rate.dtype == recipe.weights.dtype == jnp.float32
    np.testing.assert_array_equal(recipe.rate, [[1, 1, 1], [2, 2, 2]])
    with pytest.raises(TypeError, match="batch shapes"):
        Recipe(rate=[1, 2], weights=np.ones((3, 4), dtype=np.float32))


def test_unflatten_preserves_identity_and_current_batch_shape():
    """Spec 4.2: reconstruction bypasses normalization, including sentinels."""
    recipe = Recipe(rate=[1, 2, 3], weights=[4, 5], method="max")
    children, structure = jax.tree.flatten(recipe)
    sentinels = [object() for _ in children]
    rebuilt = jax.tree.unflatten(structure, sentinels)
    assert rebuilt.rate is sentinels[0] and rebuilt.weights is sentinels[1]
    assert rebuilt.method == "max"
    assert jax.tree.map(lambda x: x[:2], recipe).batch_shape == (2,)
    assert jax.jit(lambda x: x)(recipe).method == "max"
    np.testing.assert_array_equal(
        jax.vmap(lambda x: jnp.zeros(x.batch_shape))(recipe), np.zeros(3)
    )
    assert structure == jax.tree.structure(
        Recipe(rate=[9, 8, 7], weights=[6, 5], method="max")
    )
    assert structure != jax.tree.structure(Recipe(rate=[1, 2, 3], weights=[4, 5]))


def test_traced_construction_stores_invalid_values_without_checks():
    """Spec 4.1: constructors store natural values, regardless of domain."""
    values = jnp.array([-1.0, jnp.nan, jnp.inf])

    def build(rate):
        return Recipe(rate=rate, weights=[1.0, 2.0])

    for result in (build(values), jax.jit(build)(values), jax.vmap(build)(values)):
        np.testing.assert_array_equal(result.rate, values)
        assert result.batch_shape == (3,)


@pytest.mark.parametrize("value", [None, True, 1j, "1", np.array([1], dtype=object)])
def test_constructor_rejects_nonreal_parameters(value):
    """Spec 3.3: misuse rejects kinds, not parameter domains."""
    with pytest.raises(TypeError, match="rate"):
        Recipe(rate=value, weights=[1.0])


def test_explicit_registration_and_shape_parameter():
    """Spec 4.1–4.2: no implicit registration or array-shape alias."""

    @dataclasses.dataclass(frozen=True)
    class ShapeRecipe(survjax.SurvivalDistribution):
        shape: jax.Array = survjax.param()

    unregistered = object.__new__(ShapeRecipe)
    assert jax.tree.leaves(unregistered) == [unregistered]
    survjax.register_family(ShapeRecipe)
    recipe = ShapeRecipe(shape=[2, 3])
    np.testing.assert_array_equal(recipe.shape, [2, 3])
    assert recipe.batch_shape == (2,)


def test_constructor_warning_points_to_user(x64_enabled):
    """Spec 3.3: x32 float64 warnings deduplicate the public call site."""
    if x64_enabled:
        pytest.skip("conversion warning requires 64-bit mode disabled")
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default")
        for _ in range(2):
            call_line = inspect.currentframe().f_lineno + 1
            Recipe(rate=np.float64(1), weights=np.ones(2, dtype=np.float32))
    assert len(recorded) == 1
    assert (recorded[0].filename, recorded[0].lineno) == (__file__, call_line)


def test_plain_default_field_is_rejected_at_registration():
    """Spec 4.2 / D-014: a plain default must not silently replace user values."""

    @dataclasses.dataclass(frozen=True)
    class Plain(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()
        extra: int = 3

    with pytest.raises(
        TypeError, match=r"extra must be declared with param\(\) or static\(\)"
    ):
        survjax.register_family(Plain)


def test_plain_factory_field_is_rejected_at_registration():
    """Spec 4.2 / D-014: a factory field must not vanish on reconstruction."""

    @dataclasses.dataclass(frozen=True)
    class Plain(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()
        coef: jax.Array = dataclasses.field(default_factory=lambda: jnp.ones(2))

    with pytest.raises(TypeError, match="coef must be declared"):
        survjax.register_family(Plain)


@pytest.mark.parametrize("count", [True, False, 1.5, "1", None])
def test_component_count_rejects_noninteger_kinds(count):
    """Spec 4.2: component_ndim counts axes and cannot be a bool or noninteger."""
    with pytest.raises(TypeError, match="component_ndim"):
        survjax.param(component_ndim=count)


def test_component_count_rejects_negative_integer():
    """Spec 4.2: a number of trailing component axes cannot be negative."""
    with pytest.raises(ValueError, match="component_ndim"):
        survjax.param(component_ndim=-1)


def test_inherited_initializer_is_rejected_at_registration():
    """Spec 3.3 / D-014: do not wrap an already-registered initializer again."""

    class Subclass(Recipe):
        pass

    with pytest.raises(TypeError, match="decorate the subclass itself"):
        survjax.register_family(Subclass)


def test_redecorated_subclass_warning_points_to_user(x64_enabled):
    """Spec 3.3: an explicitly decorated subclass keeps caller-site warnings."""
    if x64_enabled:
        pytest.skip("conversion warning requires 64-bit mode disabled")

    @survjax.register_family
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class Subclass(Recipe):
        pass

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default")
        for _ in range(2):
            call_line = inspect.currentframe().f_lineno + 1
            result = Subclass(rate=np.float64(1), weights=[1.0])
    assert result.rate.dtype == jnp.float32
    assert len(recorded) == 1
    assert (recorded[0].filename, recorded[0].lineno) == (__file__, call_line)


def test_unhashable_static_error_names_field():
    """Spec 4.2: malformed static configuration is reported at construction."""
    with pytest.raises(TypeError, match="method: static fields must be hashable"):
        Recipe(rate=1.0, weights=[1.0], method=[])


def test_component_parameter_requires_declared_axes():
    """Spec 4.2/5: a scalar lacks the vector's declared component axis."""
    with pytest.raises(TypeError, match="weights has fewer axes than component_ndim"):
        Recipe(rate=1.0, weights=1.0)


def test_mutable_dataclass_is_rejected_at_registration():
    """Spec 4.1–4.2: distribution instances must be frozen dataclasses."""

    @dataclasses.dataclass
    class Mutable(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

    with pytest.raises(TypeError, match="frozen dataclasses"):
        survjax.register_family(Mutable)
