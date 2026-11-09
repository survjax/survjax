"""Weibull interval probabilities from stable cumulative-hazard increments."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax.distributions._survival_methods import log_cdf_from_log_cumulative_hazard

if TYPE_CHECKING:
    from survjax.distributions.weibull import Weibull


def _log_hazard_increment(
    shape: jax.Array,
    scale: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    """Log(H(upper)-H(lower)) for finite 0 < lower < upper and valid parameters."""
    width = upper - lower
    nearby = width <= lower
    # spec 7.2: log1p preserves narrow widths; separated endpoints avoid an
    # overflowing upper/lower ratio. Both inactive branches get safe operands.
    relative_width = jnp.where(nearby, width, 1.0) / jnp.where(nearby, lower, 1.0)
    far_lower = jnp.where(nearby, 1.0, lower)
    far_upper = jnp.where(nearby, 2.0, upper)
    log_ratio = jnp.where(
        nearby, jnp.log1p(relative_width), jnp.log(far_upper) - jnp.log(far_lower)
    )
    log_power_gap = jnp.log(shape) + jnp.log(log_ratio)
    # H(b)-H(a)=H(b)*(1-exp(-k*log(b/a))). Keep even tiny k*log(b/a) in logs.
    log_upper_hazard = shape * (jnp.log(upper) - jnp.log(scale))
    return log_upper_hazard + log_cdf_from_log_cumulative_hazard(log_power_gap)


def log_interval_prob(self: "Weibull", lower: ArrayLike, upper: ArrayLike) -> jax.Array:
    """Return log P(lower < T <= upper) using a stable hazard increment.

    For H(t)=(t/scale)**shape and a < b, the result is
    -H(a) + log(1-exp(-(H(b)-H(a)))). Nearly equal powers are never subtracted.

    Args:
        lower: Nonnegative lower endpoints.
        upper: Upper endpoints, possibly +inf. Endpoints broadcast against
            batch_shape; add trailing singleton axes for time-first grids.
    Returns:
        Broadcast log probabilities. Equal endpoints give -inf; invalid
        endpoints or parameters give NaN. Tiny positive mass stays in log space.
    Raises:
        ValueError: If input shapes cannot broadcast.
    """
    lower, upper = jnp.asarray(lower), jnp.asarray(upper)
    dtype = jnp.result_type(lower, upper, self.shape, self.scale)
    a, b = jnp.broadcast_arrays(lower.astype(dtype), upper.astype(dtype))
    valid = (a >= 0) & (b >= a) & ~self.parameter_issues().invalid_entries
    empty = a == b
    active = valid & ~empty
    interior = active & (a > 0) & jnp.isfinite(b)
    # spec 7.2: substitute parameters too for the inactive finite-interval piece.
    log_increment = _log_hazard_increment(
        jnp.where(interior, self.shape, 1.0),
        jnp.where(interior, self.scale, 1.0),
        jnp.where(interior, a, 1.0),
        jnp.where(interior, b, 2.0),
    )
    log_mass = log_cdf_from_log_cumulative_hazard(log_increment)
    left_mass = self.log_cdf(jnp.where(active & (a == 0), b, 0.0))
    log_survival = self.log_survival(jnp.where(active, a, 0.0))
    value = log_survival + log_mass
    value = jnp.where(a == 0, left_mass, value)
    value = jnp.where(jnp.isposinf(b), log_survival, value)
    return jnp.where(valid, jnp.where(empty, -jnp.inf, value), jnp.nan)
