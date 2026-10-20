"""A6.3 declared diagnostic composition (spec 4.7)."""

from dataclasses import dataclass
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import survjax


def family(constraint=None, hook=None, component_ndim=0):
    @survjax.register_family
    @dataclass(frozen=True)
    class Family(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param(component_ndim, constraint)

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return jnp.ones(self.batch_shape, dtype=self.rate.dtype)

        def parameter_conditions(self):
            return {} if hook is None else hook(self)

    return Family


Checked = family(
    survjax.constraints.positive,
    lambda d: MappingProxyType({"z": d.rate > 3, "a": d.rate < -3}),
)


@survjax.register_family
@dataclass(frozen=True)
class Components(survjax.SurvivalDistribution):
    base: survjax.SurvivalDistribution = survjax.distribution()
    z: jax.Array = survjax.param(2, survjax.constraints.ordered)
    a: jax.Array = survjax.param(1, survjax.constraints.positive)
    free: jax.Array = survjax.param()

    def parameter_conditions(self):
        return {"z_limit": self.free > 9, "a_limit": self.free < -9}


def mixed():
    return Components(
        base=Checked(rate=[1, 2, 5]),
        z=[[[0, 1], [0, 1]], [[0, 0], [0, 1]], [[1, 0], [0, 1]]],
        a=[[1, 1], [0, 1], [1, 1]],
        free=[0, 10, 0],
    )


@pytest.mark.parametrize(
    "transform", [lambda f: f, jax.jit, jax.vmap], ids=["eager", "jit", "vmap"]
)
def test_component_reductions_and_ordered_nested_conditions(transform):
    """Spec 4.7: reduce only remaining component axes; preserve own-first order."""
    issues = transform(lambda d: d.parameter_issues())(mixed())
    # z: tie or descending pair; a: one nonpositive component; base.z: rate>3.
    expected = {
        "z": [False, True, True],
        "a": [False, True, False],
        "base.rate": [False] * 3,
    }
    conditions = {
        "z_limit": [False, True, False],
        "a_limit": [False] * 3,
        "base.z": [False, False, True],
        "base.a": [False] * 3,
    }
    assert list(issues.fields) == list(expected)
    assert list(issues.conditions) == list(conditions)
    for name, mask in (*expected.items(), *conditions.items()):
        actual = issues.fields[name] if name in expected else issues.conditions[name]
        np.testing.assert_array_equal(actual, mask)
        assert actual.dtype == jnp.bool_
    np.testing.assert_array_equal(issues.invalid_entries, [False, True, True])
    np.testing.assert_array_equal(
        issues.ok, [True, False, False] if transform is jax.vmap else False
    )


def test_indexing_recomputes_ok_for_selected_valid_row():
    """Spec 4.7: indexed diagnostics match indexed masks and recompute scalar ok."""
    parent, child = mixed().parameter_issues(), mixed()[0].parameter_issues()
    for name in parent.fields:
        np.testing.assert_array_equal(child.fields[name], parent.fields[name][0])
    for name in parent.conditions:
        np.testing.assert_array_equal(
            child.conditions[name], parent.conditions[name][0]
        )
    assert child.invalid_entries.shape == child.ok.shape == ()
    assert not bool(child.invalid_entries) and bool(child.ok)


def test_rebuilt_smaller_nested_batch_broadcasts_diagnostics():
    """Spec 4.7: diagnostic broadcasting works independently of construction."""
    children, tree = jax.tree_util.tree_flatten(mixed())
    # Reconstruction bypasses constructor broadcasting; outer batch remains (3,).
    rebuilt = jax.tree_util.tree_unflatten(tree, [jnp.array(-4.0), *children[1:]])
    issues = rebuilt.parameter_issues()
    np.testing.assert_array_equal(issues.fields["base.rate"], [True] * 3)
    np.testing.assert_array_equal(issues.conditions["base.a"], [True] * 3)


def test_undeclared_and_conditions_only_families():
    """Spec 4.7: no domains are inferred; a condition alone can invalidate rows."""
    empty = family()(rate=[jnp.nan, -1]).parameter_issues()
    assert empty.fields == empty.conditions == {}
    np.testing.assert_array_equal(empty.invalid_entries, [False, False])
    assert bool(empty.ok)
    dist = family(hook=lambda d: {"cap": d.rate > 3})(rate=[2, 4])
    np.testing.assert_array_equal(
        dist.parameter_issues().invalid_entries, [False, True]
    )
    scalar = family(survjax.constraints.ordered, component_ndim=1)(
        rate=[0, 0]
    ).parameter_issues()
    assert scalar.invalid_entries.shape == scalar.ok.shape == ()
    assert bool(scalar.fields["rate"]) and not bool(scalar.ok)


def test_nested_only_family_retains_deeper_prefixes():
    """Spec 4.7: nested-only wrappers retain all levels of field/condition names."""

    @survjax.register_family
    @dataclass(frozen=True)
    class Wrapper(survjax.SurvivalDistribution):
        base: survjax.SurvivalDistribution = survjax.distribution()

    issues = Wrapper(base=Wrapper(base=Checked(rate=-4))).parameter_issues()
    assert list(issues.fields) == ["base.base.rate"]
    assert list(issues.conditions) == ["base.base.z", "base.base.a"]
    assert bool(issues.invalid_entries)
    with pytest.raises(survjax.InvalidParameterError, match="base.base.a"):
        issues.raise_if_invalid()


def test_traced_construction_and_values_reuse_compilation_without_transfers():
    """Spec 4.1/4.7/7.2: same-shaped values reuse compiled, device-local diagnostics."""
    traces = []

    @jax.jit
    def build(rate):
        traces.append(None)
        return Checked(rate=rate).parameter_issues()

    good, bad = jnp.array([1.0, 2.0]), jnp.array([-1.0, 4.0])
    first = build(good)
    with jax.transfer_guard("disallow"):
        second = build(bad)
        second.ok.block_until_ready()
    assert len(traces) == 1
    assert jax.tree.structure(first) == jax.tree.structure(second)
    np.testing.assert_array_equal(second.invalid_entries, [True, True])


@pytest.mark.parametrize(
    "hook",
    [
        lambda d: {"a.b": d.rate > 0},
        lambda d: {"x": d.rate},
        lambda d: {"x": jnp.any(d.rate > 0)},
    ],
    ids=["key", "dtype", "shape"],
)
def test_malformed_conditions_fail_eager_and_traced(hook):
    """Spec 4.7: hook validation uses structure, dtype and shape during tracing."""
    dist = family(hook=hook)(rate=[1, 2])
    for evaluate in (
        lambda d: d.parameter_issues(),
        jax.jit(lambda d: d.parameter_issues()),
    ):
        with pytest.raises(TypeError):
            evaluate(dist)


def test_from_log_reports_natural_parameter_overflow_and_underflow():
    """Spec 4.1/4.7: exp(1000)=inf and exp(-1000)=0 violate positive in both dtypes."""
    issues = survjax.Exponential.from_log(
        log_rate=jnp.array([0.0, 1000.0, -1000.0])
    ).parameter_issues()
    np.testing.assert_array_equal(issues.fields["rate"], [False, True, True])


def test_direct_values_and_construction_do_not_invoke_diagnostics():
    """Spec 4.1/4.5: a finite formula at rate=-1 remains 2, with no hook call."""

    def forbidden(dist):
        raise AssertionError("diagnostics called outside parameter_issues")

    constrained = family(survjax.constraints.positive, forbidden)(rate=-1)
    undeclared = family()(rate=-1)
    np.testing.assert_array_equal(constrained.log_survival(2.0), 2.0)
    np.testing.assert_array_equal(
        constrained.log_survival(2.0), undeclared.log_survival(2.0)
    )
