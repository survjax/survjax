"""Exact and right-censored likelihood contributions."""

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from survjax._data_validation import current_field_issues
from survjax._likelihood_parameters import stop_invalid_gradients
from survjax.data import SurvivalData
from survjax.distributions.base import SurvivalDistribution


def _unusable_data_rows(data: SurvivalData, shape: tuple[int, ...]) -> jax.Array:
    """Combine construction evidence with current-field checks."""
    current_bits = current_field_issues(
        data.exact, data.lower, data.upper, data.left_trunc, data.right_trunc
    )
    return jnp.broadcast_to(data.issues.invalid_rows | (current_bits != 0), shape)


def _supported_rows(data: SurvivalData, shape: tuple[int, ...]) -> jax.Array:
    """Read current mode support, independent of diagnostics/metadata."""
    supported = (
        (data.left_trunc == 0)
        & jnp.isposinf(data.right_trunc)
        & (data.exact | jnp.isposinf(data.upper))
    )
    return jnp.broadcast_to(supported, shape)


def _evaluate(
    prepared: SurvivalDistribution,
    data: SurvivalData,
    bad: jax.Array,
    common: DTypeLike,
) -> jax.Array:
    """Evaluate exact/right pieces with safe inactive operands."""
    exact_rows = jnp.broadcast_to(data.exact, bad.shape)
    # spec 3.3: stored times may be narrower than the promoted parameters.
    time = jnp.asarray(data.lower, dtype=common)
    # spec 3.3/7.2: a family reference may use a fixed dtype. Keep substituted
    # times in the common dtype, and stop only this substituted value.
    reference = jax.lax.stop_gradient(
        jnp.asarray(prepared.reference_time(), dtype=common)
    )
    # spec 7.2: keep likelihood-level substitution even though direct methods
    # also guard their inputs under 4.5. Inactive survival uses window (0,inf].
    density_time = jnp.where(~bad & exact_rows, time, reference)
    survival_time = jnp.where(~bad & ~exact_rows, time, jnp.zeros((), dtype=common))
    density = prepared.log_prob(density_time)
    survival = prepared.log_survival(survival_time)
    # spec 3.3: custom overrides can return fixed lower-precision arrays even
    # after their inputs are promoted; contributions still use the common dtype.
    values = jnp.asarray(jnp.where(exact_rows, density, survival), dtype=common)
    return jnp.where(bad, jnp.nan, values)


def _log_likelihood(dist: SurvivalDistribution, data: SurvivalData) -> jax.Array:
    """Validate rows, prepare parameters, then evaluate unsummed contributions.

    Shapes use trailing-axis broadcasting; incompatible shapes raise TypeError.
    Stored/current data issues, parameter issues and unsupported modes give NaN.
    """
    try:
        shape = jnp.broadcast_shapes(data.lower.shape, dist.batch_shape)
    except ValueError as exc:
        raise TypeError("data and distribution batch shapes are incompatible") from exc

    invalid_entries = dist.parameter_issues().invalid_entries
    bad = (
        _unusable_data_rows(data, shape)
        | jnp.broadcast_to(invalid_entries, shape)
        | ~_supported_rows(data, shape)
    )
    # spec 3.3: promote all own/nested leaves without normalizing or mutating inputs.
    common = jnp.result_type(
        data.lower,
        data.upper,
        data.left_trunc,
        data.right_trunc,
        *jax.tree.leaves(dist),
    )
    promoted = jax.tree.map(lambda leaf: leaf.astype(common), dist)
    # spec 7.1: mask invalid entries before numerical methods, including reference_time.
    prepared = stop_invalid_gradients(promoted, invalid_entries)
    return _evaluate(prepared, data, bad, common)
