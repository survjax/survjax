"""Prepare declared-invalid parameter entries for likelihood AD (spec 7.1)."""

from dataclasses import fields

import jax
import jax.numpy as jnp

from survjax.distributions.base import SurvivalDistribution


def stop_invalid_gradients(
    dist: SurvivalDistribution, invalid_entries: jax.Array
) -> SurvivalDistribution:
    """Broadcast parameter leaves and stop only declared-invalid entries.

    The caller supplies the union mask of the original distribution's
    parameter issues. Values and static fields are preserved; this does not
    evaluate family methods or force likelihood NaNs. Inputs are not mutated.
    """
    batch_shape = dist.batch_shape
    invalid_entries = jnp.broadcast_to(invalid_entries, batch_shape)

    def prepare(current: SurvivalDistribution) -> SurvivalDistribution:
        # spec 4.2: reconstruction must not rerun constructors or normalization.
        result = object.__new__(type(current))
        for declared in fields(current):  # type: ignore[arg-type]
            value = getattr(current, declared.name)
            kind = declared.metadata["survjax_kind"]
            if kind == "distribution":
                value = prepare(value)
            elif kind == "param":
                count = declared.metadata["component_ndim"]
                component_shape = value.shape[value.ndim - count :]
                value = jnp.broadcast_to(value, batch_shape + component_shape)
                mask = invalid_entries.reshape(batch_shape + (1,) * count)
                # spec 7.1–7.2: the sole exception for real parameter entries.
                value = jnp.where(mask, jax.lax.stop_gradient(value), value)
            object.__setattr__(result, declared.name, value)
        return result

    return prepare(dist)
