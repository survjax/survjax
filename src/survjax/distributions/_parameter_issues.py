"""Compose declared parameter diagnostics without host inspection (spec 4.7)."""

from collections.abc import Mapping
from dataclasses import fields
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from survjax.diagnostics import ParameterIssues

if TYPE_CHECKING:
    from survjax.distributions.base import SurvivalDistribution


def parameter_issues(self: "SurvivalDistribution") -> ParameterIssues:
    """Return declared parameter violations as an ordered PyTree (spec 4.7).

    Returns:
        ParameterIssues with boolean masks of ``batch_shape``: own constrained
        fields first, then nested fields with dotted prefixes; own conditions
        first, then nested conditions. True marks violations. ``invalid_entries``
        is their union and ``ok`` is scalar negated-any. An empty report does
        not guarantee validity; undeclared domains are not inferred.

    Raises:
        TypeError: A non-mapping condition result, malformed names, nonboolean
            arrays or shapes other than ``batch_shape``, also under tracing.
    """
    batch_shape = self.batch_shape
    declarations = fields(self)  # type: ignore[arg-type]
    field_masks: dict[str, jax.Array] = {}
    for declared in declarations:
        constraint = self.parameter_constraints.get(declared.name)
        if constraint is None:
            continue
        mask = ~constraint(getattr(self, declared.name))
        remaining = declared.metadata["component_ndim"] - getattr(
            constraint, "event_dim", 0
        )
        if remaining:
            axes = tuple(range(mask.ndim - remaining, mask.ndim))
            mask = jnp.any(mask, axis=axes)
        field_masks[declared.name] = jnp.broadcast_to(mask, batch_shape)

    nested_conditions: dict[str, jax.Array] = {}
    for declared in declarations:
        if declared.metadata.get("survjax_kind") != "distribution":
            continue
        nested = getattr(self, declared.name).parameter_issues()
        for name, mask in nested.fields.items():
            field_masks[f"{declared.name}.{name}"] = jnp.broadcast_to(mask, batch_shape)
        for name, mask in nested.conditions.items():
            nested_conditions[f"{declared.name}.{name}"] = jnp.broadcast_to(
                mask, batch_shape
            )

    own_conditions = self.parameter_conditions()
    if not isinstance(own_conditions, Mapping):
        raise TypeError("parameter_conditions must return a mapping")
    for name, mask in own_conditions.items():
        if not isinstance(name, str) or not name or "." in name:
            raise TypeError("condition names must be nonempty strings without dots")
        if (
            getattr(mask, "dtype", None) != jnp.bool_
            or getattr(mask, "shape", None) != batch_shape
        ):
            raise TypeError(
                f"condition {name}: expected boolean array of shape {batch_shape}"
            )
    # spec 4.7: own conditions precede nested ones regardless of field order.
    condition_masks = dict(own_conditions)
    condition_masks.update(nested_conditions)
    invalid_entries = jnp.zeros(batch_shape, dtype=jnp.bool_)
    for mask in (*field_masks.values(), *condition_masks.values()):
        invalid_entries = invalid_entries | mask
    return ParameterIssues(
        field_masks, condition_masks, invalid_entries, ~jnp.any(invalid_entries)
    )
