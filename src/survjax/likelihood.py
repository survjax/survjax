"""Exact and censored likelihood contributions conditional on inclusion."""

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


def _evaluate(
    prepared: SurvivalDistribution,
    data: SurvivalData,
    bad: jax.Array,
    common: DTypeLike,
) -> jax.Array:
    """Evaluate exact/censored pieces with safe inactive operands."""
    exact_rows = jnp.broadcast_to(data.exact, bad.shape)
    # spec 3.3: stored times may be narrower than the promoted parameters.
    time = jnp.asarray(data.lower, dtype=common)
    # spec 7.2: intersect real sets first, then substitute each exact row's
    # own window. Invalid rows use the untruncated window throughout.
    left = jnp.where(~bad, jnp.asarray(data.left_trunc, dtype=common), 0.0)
    right = jnp.where(~bad, jnp.asarray(data.right_trunc, dtype=common), jnp.inf)
    # spec 3.2: _may_contain uses the order defined in _data_metadata.may_contain:
    # 0 exact, 1 right-censored, 2 left-censored, 3 interval-censored,
    # 4 left-truncated, 5 right-truncated. These are dataset-wide possibilities,
    # not per-row masks: False proves absence; True may be conservative.
    # Traced construction sets all flags true; slices inherit their parent's flags.
    # Flags 2–5 describe left/interval censoring and both truncation modes.
    # When all are absent, spec 2 gives Z=1 and right-row mass S(c), including
    # any cure mass at infinity. No family-type or proper-family assumption.
    untruncated_exact_right = not any(data._may_contain[2:6])
    # Zeros are placeholders for omitted pieces, never active row contributions;
    # the final exact_rows mask selects the appropriate piece for each row.
    interval = jnp.zeros(bad.shape, dtype=common)
    # spec 7.2: flags 1–3 cover all censored kinds, each needing conditional
    # set probability. Skip this work only when none of those kinds can occur.
    if any(data._may_contain[1:4]):
        active_set = ~bad & ~exact_rows
        if untruncated_exact_right:
            # The inactive set is (0,+inf], so use its lower endpoint, 0.
            survival_time = jnp.where(active_set, time, 0.0)
            interval = prepared.log_survival(survival_time)
        else:
            lo = jnp.maximum(time, left)
            hi = jnp.minimum(jnp.asarray(data.upper, dtype=common), right)
            lower = jnp.where(active_set, lo, jax.lax.stop_gradient(left))
            upper = jnp.where(active_set, hi, jax.lax.stop_gradient(right))
            interval = prepared.log_conditional_interval_prob(lower, upper, left, right)

    density = jnp.zeros(bad.shape, dtype=common)
    # spec 7.2: flag 0 allows exact observations, which need log density minus
    # log window mass. With no possible exact rows, neither piece is needed here.
    if data._may_contain[0]:
        # spec 3.3/7.2: promote a fixed-dtype family reference, and stop only
        # this substituted value, never the active times or parameters.
        reference = jax.lax.stop_gradient(
            jnp.asarray(prepared.reference_time(), dtype=common)
        )
        density_time = jnp.where(~bad & exact_rows, time, reference)
        density = prepared.log_prob(density_time)
        # Only exact rows subtract log Z: censored rows use the conditional hook
        # to avoid cancellation of large common tail terms (spec 4.2/7.2).
        if not untruncated_exact_right:
            exact_left = jnp.where(~bad & exact_rows, left, 0.0)
            exact_right = jnp.where(~bad & exact_rows, right, jnp.inf)
            log_z = prepared.log_interval_prob(exact_left, exact_right)
            usable_z = jnp.isfinite(log_z)
            density = density - jnp.where(usable_z, log_z, 0.0)
            bad = bad | (exact_rows & ~usable_z)
    # spec 3.3: custom overrides can return fixed lower-precision arrays even
    # after their inputs are promoted; contributions still use the common dtype.
    values = jnp.asarray(jnp.where(exact_rows, density, interval), dtype=common)
    return jnp.where(bad, jnp.nan, values)


def _log_likelihood(dist: SurvivalDistribution, data: SurvivalData) -> jax.Array:
    """Validate rows, prepare parameters, then evaluate unsummed contributions.

    Shapes use trailing-axis broadcasting; incompatible shapes raise TypeError.
    Stored/current data issues, parameter issues and zero window mass give NaN.
    """
    try:
        shape = jnp.broadcast_shapes(data.lower.shape, dist.batch_shape)
    except ValueError as exc:
        raise TypeError("data and distribution batch shapes are incompatible") from exc

    invalid_entries = dist.parameter_issues().invalid_entries
    bad = _unusable_data_rows(data, shape) | jnp.broadcast_to(invalid_entries, shape)
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
