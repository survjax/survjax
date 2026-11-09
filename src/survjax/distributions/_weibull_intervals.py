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


def _log_window_increment(
    self: "Weibull",
    lower: jax.Array,
    upper: jax.Array,
) -> jax.Array:
    """Log hazard increment for valid 0 <= lower <= upper, lower finite."""
    positive = upper > lower
    interior = positive & (lower > 0) & jnp.isfinite(upper)
    value = _log_hazard_increment(
        jnp.where(interior, self.shape, 1.0),
        jnp.where(interior, self.scale, 1.0),
        jnp.where(interior, lower, 1.0),
        jnp.where(interior, upper, 2.0),
    )
    # spec 7.2: the zero-origin piece is inactive for positive lower endpoints.
    from_zero = self._log_cumulative_hazard(
        jnp.where(positive & (lower == 0), upper, 0.0)
    )
    value = jnp.where(lower == 0, from_zero, value)
    value = jnp.where(jnp.isposinf(upper), jnp.inf, value)
    return jnp.where(positive, value, -jnp.inf)


def log_conditional_interval_prob(
    self: "Weibull",
    lower: ArrayLike,
    upper: ArrayLike,
    left_trunc: ArrayLike,
    right_trunc: ArrayLike,
) -> jax.Array:
    """Return interval log probability conditional on the window (L,U].

    Intersect the requested interval with the window to get (a,b]. With
    D(x,y)=H(y)-H(x), cancel the common H(L) before evaluating
    -D(L,a) + log(1-exp(-D(a,b))) - log(1-exp(-D(L,U))).
    All hazard differences use stable increments rather than power subtraction.

    Args:
        lower: Nonnegative requested lower endpoints.
        upper: Requested upper endpoints, possibly +inf.
        left_trunc: Finite nonnegative window lower endpoints L.
        right_trunc: Window upper endpoints U > L, possibly +inf.
    Returns:
        Log probabilities with trailing-axis batch broadcasting. Full coverage
        is exactly zero; empty intersections give -inf. Invalid endpoints,
        windows or parameters give NaN. Use trailing singleton axes for grids.
    Raises:
        ValueError: If input shapes cannot broadcast.
    """
    arrays = tuple(map(jnp.asarray, (lower, upper, left_trunc, right_trunc)))
    dtype = jnp.result_type(*arrays, self.shape, self.scale)
    a, b, window_lower, window_upper = jnp.broadcast_arrays(
        *(array.astype(dtype) for array in arrays)
    )
    valid = (
        (a >= 0)
        & (b >= a)
        & jnp.isfinite(window_lower)
        & (window_lower >= 0)
        & (window_upper > window_lower)
        & ~self.parameter_issues().invalid_entries
    )
    window_lower = jnp.where(valid, window_lower, 0.0)
    window_upper = jnp.where(valid, window_upper, 1.0)
    lo, hi = jnp.maximum(a, window_lower), jnp.minimum(b, window_upper)
    empty = hi <= lo
    full = (a <= window_lower) & (b >= window_upper)
    active = valid & ~empty & ~full
    # spec 7.2: inactive numerator uses the valid window, without re-intersection.
    lo, hi = jnp.where(active, lo, window_lower), jnp.where(active, hi, window_upper)
    offset = jnp.exp(_log_window_increment(self, window_lower, lo))
    numerator = log_cdf_from_log_cumulative_hazard(_log_window_increment(self, lo, hi))
    denominator = log_cdf_from_log_cumulative_hazard(
        _log_window_increment(self, window_lower, window_upper)
    )
    value = -offset + numerator - denominator
    value = jnp.where(full, 0.0, jnp.where(empty, -jnp.inf, value))
    return jnp.where(valid, value, jnp.nan)
