"""Prepared-data likelihood evaluation for Slice A (spec 7.1–7.3)."""

import jax
import jax.numpy as jnp

from survjax._data_validation import row_issues
from survjax._likelihood_parameters import stop_invalid_gradients
from survjax.data import SurvivalData
from survjax.distributions.base import SurvivalDistribution


def _log_likelihood(dist: SurvivalDistribution, data: SurvivalData) -> jax.Array:
    """Return unsummed exact/right contributions with trailing-axis broadcasting.

    Stored raw-input issues and current normalized fields both invalidate
    rows. Unsupported modes and declared parameter violations produce NaN.
    Shapes incompatible with the distribution batch raise TypeError.
    """
    try:
        shape = jnp.broadcast_shapes(data.lower.shape, dist.batch_shape)
    except ValueError as exc:
        raise TypeError("data and distribution batch shapes are incompatible") from exc

    exact, lower, upper = data.exact, data.lower, data.upper
    zeros = jnp.zeros_like(exact)
    # spec 3.2/6.2: recheck normalized fields, retaining earlier raw faults.
    bits = row_issues(
        exact,
        lower,
        upper,
        data.left_trunc,
        data.right_trunc,
        invalid_indicator=zeros,
        conflicting_indicators=zeros,
        missing_time2=~exact & jnp.isnan(upper),
        invalid_time=~jnp.isfinite(lower) | (lower < 0) | (exact & (lower == 0)),
        invalid_time2=(upper < 0) | jnp.isneginf(upper) | (exact & (upper != lower)),
        check_interval=~exact,
    )
    bad_data = jnp.broadcast_to((bits | data._issue_bits) != 0, shape)
    supported = jnp.broadcast_to(
        (data.left_trunc == 0)
        & jnp.isposinf(data.right_trunc)
        & (exact | jnp.isposinf(upper)),
        shape,
    )
    invalid_entries = dist.parameter_issues().invalid_entries
    bad_parameters = jnp.broadcast_to(invalid_entries, shape)
    common = jnp.result_type(
        lower, upper, data.left_trunc, data.right_trunc, *jax.tree.leaves(dist)
    )
    promoted = jax.tree.map(lambda leaf: leaf.astype(common), dist)
    # spec 7.1: stop invalid parameter entries before even reference_time runs.
    prepared = stop_invalid_gradients(promoted, invalid_entries)
    reference = jax.lax.stop_gradient(
        jnp.asarray(prepared.reference_time(), dtype=common)
    )
    exact = jnp.broadcast_to(exact, shape)
    usable = ~bad_data & ~bad_parameters & supported
    time = jnp.asarray(lower, dtype=common)
    # spec 7.2: inactive density uses a stopped reference; inactive survival
    # uses the untruncated window's zero endpoint through the base policy.
    density_time = jnp.where(usable & exact, time, reference)
    survival_time = jnp.where(usable & ~exact, time, jnp.zeros((), dtype=common))
    density = prepared.log_prob(density_time)
    survival = prepared.log_survival(survival_time)
    values = jnp.asarray(jnp.where(exact, density, survival), dtype=common)
    bad = bad_data | bad_parameters | ~supported
    return jnp.where(bad, jnp.nan, values)
