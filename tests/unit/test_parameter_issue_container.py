"""A6.2 result storage and explicit host checkpoints (spec 4.7/6.3)."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from survjax import InvalidParameterError
from survjax.diagnostics import ParameterIssues


def test_round_trip_preserves_order_masks_and_frozen_fields() -> None:
    """Spec 4.7/6.3: declaration order survives, with only arrays as leaves."""
    yes, no = jnp.array(True), jnp.array(False)
    issues = ParameterIssues(
        {"z": yes, "base.a": no}, {"z_limit": no, "base.a_limit": yes}, yes, no
    )
    leaves, tree = jax.tree.flatten(issues)
    assert len(leaves) == 6
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
    rebuilt = jax.tree.unflatten(tree, leaves)
    assert list(rebuilt.fields) == ["z", "base.a"]
    assert list(rebuilt.conditions) == ["z_limit", "base.a_limit"]
    for actual, expected in zip(
        jax.tree.leaves(rebuilt), [True, False, False, True, True, False], strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == ()
    with pytest.raises(FrozenInstanceError):
        issues.ok = yes


def test_jit_preserves_multidimensional_masks_and_empty_mappings() -> None:
    """Spec 4.7/6.3: reconstruction stores batch masks and scalar ok as supplied."""
    mask = jnp.array([[False, True], [False, False]])
    issues = ParameterIssues({"rate": mask}, {}, mask, jnp.array(False))
    rebuilt = jax.jit(lambda x: x)(issues)
    np.testing.assert_array_equal(
        rebuilt.fields["rate"], [[False, True], [False, False]]
    )
    np.testing.assert_array_equal(
        rebuilt.invalid_entries, [[False, True], [False, False]]
    )
    assert rebuilt.ok.shape == ()
    assert not bool(rebuilt.ok)
    empty = ParameterIssues({}, {}, jnp.array([False, False]), jnp.array(True))
    rebuilt = jax.jit(lambda x: x)(empty)
    assert rebuilt.fields == rebuilt.conditions == {}
    np.testing.assert_array_equal(rebuilt.invalid_entries, [False, False])
    assert bool(rebuilt.ok)
    assert rebuilt.raise_if_invalid() is None


def test_vmap_and_jit_keep_order_and_per_example_ok() -> None:
    """Spec 4.7/6.3: mapped scalar ok stacks, rather than being recomputed."""

    def build(mask):
        return ParameterIssues(
            {"z": mask, "base.a": mask}, {"z": mask, "a": mask}, mask, ~mask
        )

    issues = jax.vmap(build)(jnp.array([False, True]))
    rebuilt = jax.jit(lambda x: x)(issues)
    assert list(rebuilt.fields) == ["z", "base.a"]
    assert list(rebuilt.conditions) == ["z", "a"]
    for mask in (
        *rebuilt.fields.values(),
        *rebuilt.conditions.values(),
        rebuilt.invalid_entries,
    ):
        np.testing.assert_array_equal(mask, [False, True])
    np.testing.assert_array_equal(rebuilt.ok, [True, False])


def test_same_shaped_mask_values_reuse_compilation() -> None:
    """Spec 7.2: array values do not become static container metadata."""
    traces = []

    @jax.jit
    def identity(issues):
        traces.append(None)
        return issues

    for value in (False, True):
        mask = jnp.array(value)
        result = identity(ParameterIssues({"rate": mask}, {}, mask, ~mask))
        assert bool(result.invalid_entries) == value
    assert len(traces) == 1


@pytest.mark.parametrize("category", ["field", "condition"])
def test_checkpoint_reports_scalar_field_or_condition(category) -> None:
    """Spec 4.7: scalar failures report one entry at batch index ()."""
    mask = jnp.array(True)
    fields = {"rate": mask} if category == "field" else {}
    conditions = {"rate": mask} if category == "condition" else {}
    issues = ParameterIssues(fields, conditions, mask, ~mask)
    with pytest.raises(InvalidParameterError) as caught:
        issues.raise_if_invalid()
    assert f"{category} rate: 1 entry(s), first index ()" in str(caught.value)


def test_checkpoint_reports_each_overlapping_violation_with_dotted_names() -> None:
    """Spec 4.7: report each rule's count and first multidimensional index."""
    fields = {"base.rate": jnp.array([[False, True], [True, False]])}
    conditions = {"base.rate": jnp.array([[False, True], [False, False]])}
    issues = ParameterIssues(
        fields, conditions, jnp.array([[False, True], [True, False]]), jnp.array(False)
    )
    with pytest.raises(InvalidParameterError) as caught:
        issues.raise_if_invalid()
    assert "field base.rate: 2 entry(s), first index (0, 1)" in str(caught.value)
    assert "condition base.rate: 1 entry(s), first index (0, 1)" in str(caught.value)


@pytest.mark.parametrize("invalid", [False, True])
def test_host_checkpoint_accepts_jit_results(invalid) -> None:
    """Spec 4.7/6.3: concrete jit results support valid and invalid checkpoints."""
    mask = jnp.array([False, invalid, invalid])
    issues = jax.jit(lambda x: x)(
        ParameterIssues({"rate": mask}, {}, mask, jnp.array(not invalid))
    )
    if invalid:
        with pytest.raises(
            InvalidParameterError, match=r"field rate: 2 entry\(s\), first index \(1,\)"
        ):
            issues.raise_if_invalid()
    else:
        assert issues.raise_if_invalid() is None


@pytest.mark.parametrize("transform", [jax.jit, jax.vmap], ids=["jit", "vmap"])
@pytest.mark.parametrize("invalid", [False, True])
@pytest.mark.parametrize("concrete_ok", [False, True], ids=["traced-ok", "concrete-ok"])
def test_checkpoint_rejects_any_traced_mask(transform, invalid, concrete_ok) -> None:
    """Spec 4.7: reject tracing before host conversion, even with concrete ok."""
    host_ok = jnp.array(not invalid)

    def checkpoint(mask):
        ok = host_ok if concrete_ok else ~mask
        ParameterIssues({"rate": mask}, {}, mask, ok).raise_if_invalid()
        return mask

    with pytest.raises(ValueError, match="host-only") as caught:
        transform(checkpoint)(jnp.array([invalid]))
    assert type(caught.value) is ValueError
