"""A4: family registration, PyTree structure, static fields and nested distributions (spec 4.1, 4.2)."""

import dataclasses

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import assert_close, f, leaves, treedef

import survjax


def _sorted_leaf_values(tree):
    return sorted(tuple(np.ravel(np.asarray(leaf)).tolist()) for leaf in leaves(tree))


# --------------------------------------------------------------- public names


@pytest.mark.step("A4")
def test_extension_names_are_exported_at_top_level():
    """spec 4.2, rev 11: `survjax.SurvivalDistribution`, `register_family`, `param`, `static` and `distribution` exist."""
    for name in (
        "SurvivalDistribution",
        "register_family",
        "param",
        "static",
        "distribution",
    ):
        assert hasattr(survjax, name), name


@pytest.mark.step("A4")
def test_registered_family_is_a_survival_distribution():
    """spec 4.2, rev 11: the decorator keeps the class a subclass of `SurvivalDistribution`."""
    dist = fam.const_hazard()(rate=f(0.5))
    assert isinstance(dist, survjax.SurvivalDistribution)
    assert isinstance(dist, fam.const_hazard())


# --------------------------------------------------------------- construction


@pytest.mark.step("A4")
def test_positional_parameters_are_rejected():
    """spec 4.2, rev 19: registered families are constructed with keyword arguments only; positional raises `TypeError`."""
    with pytest.raises(TypeError):
        fam.power_hazard()(f(1.5), f(2.0))


@pytest.mark.step("A4")
def test_single_positional_parameter_is_rejected():
    """spec 4.2, rev 19: a single positional parameter also raises `TypeError`."""
    with pytest.raises(TypeError):
        fam.const_hazard()(f(0.5))


@pytest.mark.step("A4")
def test_positional_static_is_rejected():
    """spec 4.2, rev 19: keyword-only applies to static fields too."""
    cls = fam.component_rate()
    with pytest.raises(TypeError):
        cls(f(1.0), f([1.0, 2.0]), "max")


@pytest.mark.step("A4")
def test_missing_parameter_is_rejected():
    """spec 4.2, rev 19: "Parameter fields have no default and are required"."""
    with pytest.raises(TypeError):
        fam.power_hazard()(shape=f(1.5))


@pytest.mark.step("A4")
def test_kw_only_recipe_allows_static_before_parameter():
    """spec 4.2, rev 19: with `kw_only=True`, a defaulted static field may precede a required parameter."""
    dist = fam.kw_only_static_first()(rate=f([0.5, 1.0]))
    assert dist.method == "sum"
    assert dist.batch_shape == (2,)
    assert len(leaves(dist)) == 1


# --------------------------------------------------------------- PyTree


@pytest.mark.step("A4")
def test_parameters_are_the_only_leaves():
    """spec 4.1/4.2, rev 11: parameter arrays are dynamic leaves; a static field is not a leaf."""
    dist = fam.component_rate()(rate=f([1.0, 2.0]), w=f([3.0, 4.0, 5.0]), method="max")
    assert len(leaves(dist)) == 2


@pytest.mark.step("A4")
def test_leaves_hold_the_natural_parameter_values():
    """spec 4.1, rev 11: natural parameters are stored as the leaves (values unchanged, not transformed)."""
    dist = fam.power_hazard()(shape=f([1.5, 2.5]), scale=f([3.0, 7.0]))
    assert _sorted_leaf_values(dist) == sorted([(1.5, 2.5), (3.0, 7.0)])


@pytest.mark.step("A4")
def test_flatten_unflatten_round_trip_preserves_type_and_parameters():
    """spec 4.2, rev 11: register_family generates PyTree flatten/unflatten."""
    dist = fam.power_hazard()(shape=f([1.5, 2.5]), scale=f([3.0, 7.0]))
    rebuilt = jax.tree_util.tree_unflatten(treedef(dist), leaves(dist))
    assert type(rebuilt) is type(dist)
    np.testing.assert_array_equal(np.asarray(rebuilt.shape), [1.5, 2.5])
    np.testing.assert_array_equal(np.asarray(rebuilt.scale), [3.0, 7.0])


@pytest.mark.step("A4")
def test_unflatten_bypasses_init():
    """spec 4.2, rev 11: reconstruction bypasses `__init__` (arbitrary leaves are neither broadcast nor converted)."""
    dist = fam.power_hazard()(shape=f(1.5), scale=f(3.0))
    sentinels = [object() for _ in leaves(dist)]
    rebuilt = jax.tree_util.tree_unflatten(treedef(dist), sentinels)
    assert type(rebuilt) is type(dist)
    assert {id(rebuilt.shape), id(rebuilt.scale)} == {id(s) for s in sentinels}


@pytest.mark.step("A4")
def test_static_field_is_preserved_through_round_trip():
    """spec 4.2, rev 11: `static(...)` holds hashable configuration carried by the PyTree."""
    dist = fam.component_rate()(rate=f(1.0), w=f([1.0, 2.0]), method="max")
    rebuilt = jax.tree_util.tree_unflatten(treedef(dist), leaves(dist))
    assert rebuilt.method == "max"


@pytest.mark.step("A4")
def test_static_default_is_used_when_omitted():
    """spec 4.2, rev 11: `survjax.static(default=...)` supplies the field default."""
    dist = fam.component_rate()(rate=f(1.0), w=f([1.0, 2.0]))
    assert dist.method == "sum"


@pytest.mark.step("A4")
def test_equal_static_values_give_equal_tree_structures():
    """spec 4.2, rev 11: static configuration is not data; equal configuration and shapes share a structure."""
    cls = fam.component_rate()
    one = cls(rate=f(1.0), w=f([1.0, 2.0]), method="max")
    two = cls(rate=f(9.0), w=f([3.0, 4.0]), method="max")
    assert treedef(one) == treedef(two)


@pytest.mark.step("A4")
def test_different_static_values_give_different_tree_structures():
    """spec 4.2, rev 11: static configuration is part of the tree structure."""
    cls = fam.component_rate()
    one = cls(rate=f(1.0), w=f([1.0, 2.0]), method="sum")
    two = cls(rate=f(1.0), w=f([1.0, 2.0]), method="max")
    assert treedef(one) != treedef(two)


@pytest.mark.step("A4")
def test_no_automatic_subclass_registration():
    """spec 4.2, rev 11: "There is no automatic subclass registration": an undecorated subclass is a PyTree leaf."""
    cls = fam.unregistered()
    obj = object.__new__(cls)  # bypass any constructor; only registration is observed
    found = jax.tree_util.tree_leaves(obj)
    assert len(found) == 1
    assert found[0] is obj


@pytest.mark.step("A4")
def test_unmarked_field_is_rejected_at_registration():
    """spec 4.2, rev 33: every dataclass field must use a field helper; registration raises `TypeError`."""

    @dataclasses.dataclass(frozen=True)
    class Unmarked(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()
        unmarked_scale: float = 1.0

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

    with pytest.raises(TypeError):
        survjax.register_family(Unmarked)


@pytest.mark.step("A4")
def test_unmarked_field_error_names_the_field():
    """spec 4.2, rev 33: registration "raises a `TypeError` naming an unmarked field"."""

    @dataclasses.dataclass(frozen=True)
    class Unmarked(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()
        unmarked_scale: float = 1.0

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

    with pytest.raises(TypeError) as info:
        survjax.register_family(Unmarked)
    assert "unmarked_scale" in str(info.value)


@pytest.mark.step("A4")
def test_unmarked_field_without_default_is_rejected_at_registration():
    """spec 4.2, rev 33: a plain annotated field with no helper and no default is also unmarked."""

    @dataclasses.dataclass(frozen=True)
    class Unmarked(survjax.SurvivalDistribution):
        unmarked_rate: jax.Array

        def log_survival(self, t):
            return -self.unmarked_rate * t

        def reference_time(self):
            return 1.0 / self.unmarked_rate

    with pytest.raises(TypeError):
        survjax.register_family(Unmarked)


@pytest.mark.step("A4")
def test_subclass_inheriting_registered_initializer_is_rejected():
    """spec 4.2, rev 33: "a subclass that only inherits a registered initializer raises `TypeError` at registration"."""

    class Inherits(fam.const_hazard()):  # no @dataclass of its own
        pass

    with pytest.raises(TypeError):
        survjax.register_family(Inherits)


# --------------------------------------------------------------- immutability


@pytest.mark.step("A4")
def test_distribution_is_immutable():
    """spec 4.1/4.2, rev 11: distributions are immutable frozen dataclasses."""
    dist = fam.const_hazard()(rate=f(0.5))
    with pytest.raises(dataclasses.FrozenInstanceError):
        dist.rate = f(2.0)


@pytest.mark.step("A4")
def test_methods_do_not_mutate_parameters():
    """spec 4.1, rev 11: methods are pure (no mutation or caches): leaves are unchanged after calls."""
    dist = fam.const_hazard()(rate=f([0.5, 2.0]))
    before = _sorted_leaf_values(dist)
    structure = treedef(dist)
    dist.log_survival(f(1.0))
    dist.log_prob(f(1.0))
    assert _sorted_leaf_values(dist) == before
    assert treedef(dist) == structure


# --------------------------------------------------------------- jit and compile counts


def _counting_jit():
    traces = []

    @jax.jit
    def total(dist):
        traces.append(None)
        return dist.log_survival(f(1.0)).sum()

    return total, traces


@pytest.mark.step("A4")
def test_distribution_passes_through_jit():
    """spec 4.1, rev 11: a distribution is a PyTree that can be passed into and returned from jit."""
    dist = fam.component_rate()(rate=f([1.0, 2.0]), w=f([3.0, 4.0]), method="max")
    out = jax.jit(lambda d: d)(dist)
    assert type(out) is type(dist)
    assert out.method == "max"
    assert treedef(out) == treedef(dist)
    np.testing.assert_array_equal(np.asarray(out.rate), np.asarray(dist.rate))
    np.testing.assert_array_equal(np.asarray(out.w), np.asarray(dist.w))


@pytest.mark.step("A4")
def test_new_parameter_values_reuse_compilation():
    """spec 4.1 and 7.2, rev 11: parameters are dynamic leaves; changed values with equal shapes, dtypes and statics do not retrace."""
    cls = fam.component_rate()
    total, traces = _counting_jit()
    total(cls(rate=f([1.0, 2.0]), w=f([3.0, 4.0]), method="max"))
    total(cls(rate=f([5.0, 6.0]), w=f([7.0, 8.0]), method="max"))
    assert len(traces) == 1


@pytest.mark.step("A4")
def test_different_static_values_compile_separately():
    """spec 4.2 and 7.2, rev 11: static configuration is compile-time; a different value retraces."""
    cls = fam.component_rate()
    total, traces = _counting_jit()
    total(cls(rate=f([1.0, 2.0]), w=f([3.0, 4.0]), method="sum"))
    total(cls(rate=f([1.0, 2.0]), w=f([3.0, 4.0]), method="max"))
    assert len(traces) == 2


@pytest.mark.step("A4")
def test_static_value_selects_behaviour_under_jit():
    """spec 4.2, rev 11: a static field is concrete during tracing, so Python control flow on it works under jit.

    ``log S(1) = -rate * c``: with rate 1 and w=(3, 4), ``c=7`` (sum) or ``c=4`` (max).
    """
    cls = fam.component_rate()
    run = jax.jit(lambda d: d.log_survival(f(1.0)))
    np.testing.assert_allclose(
        np.asarray(run(cls(rate=f(1.0), w=f([3.0, 4.0]), method="sum"))), -7.0
    )
    np.testing.assert_allclose(
        np.asarray(run(cls(rate=f(1.0), w=f([3.0, 4.0]), method="max"))), -4.0
    )


# --------------------------------------------------------------- nested distributions


@pytest.mark.step("A4")
def test_nested_distribution_parameters_are_leaves_of_the_outer_family():
    """spec 4.2, rev 11: `survjax.distribution()` declares a nested distribution; its parameters are dynamic leaves too."""
    base = fam.const_hazard()(rate=f(0.5))
    outer = fam.scaled_hazard()(base=base, log_ratio=f(0.25))
    assert _sorted_leaf_values(outer) == sorted([(0.5,), (0.25,)])


@pytest.mark.step("A4")
def test_nested_distribution_round_trips_with_its_type():
    """spec 4.2, rev 11: flatten/unflatten reconstructs the nested distribution with its own type."""
    base = fam.const_hazard()(rate=f(0.5))
    outer = fam.scaled_hazard()(base=base, log_ratio=f(0.25))
    rebuilt = jax.tree_util.tree_unflatten(treedef(outer), leaves(outer))
    assert type(rebuilt.base) is type(base)
    np.testing.assert_array_equal(np.asarray(rebuilt.base.rate), 0.5)
    np.testing.assert_array_equal(np.asarray(rebuilt.log_ratio), 0.25)


@pytest.mark.step("A4")
def test_nested_distribution_is_differentiable_through_jit():
    """spec 4.1/4.2, rev 11: nested parameters are dynamic leaves, differentiable under jit.

    ``log S(t) = exp(a) * (-r t)``, so ``d/dr = -exp(a) t`` and ``d/da = -exp(a) r t``.
    With ``r=0.5``, ``a=0.25``, ``t=2``: ``d/dr = -2 e^0.25``, ``d/da = -e^0.25``.
    """

    def value(r, a):
        dist = fam.scaled_hazard()(base=fam.const_hazard()(rate=r), log_ratio=a)
        return dist.log_survival(f(2.0))

    dr, da = jax.jit(jax.grad(value, argnums=(0, 1)))(f(0.5), f(0.25))
    e = np.exp(0.25)
    assert_close(dr, -2.0 * e)
    assert_close(da, -e)
