"""Guarded log arithmetic, without clipping or value validation."""

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from survjax.distributions.base import SurvivalDistribution


def safe_log(x: jax.Array) -> jax.Array:
    """Log a nonnegative value; zero gives -inf without a NaN gradient."""
    zero = x == 0
    # spec 7.2: inactive log(0) must not contaminate active parameter derivatives.
    safe_x = jnp.where(zero, jax.lax.stop_gradient(jnp.ones_like(x)), x)
    return jnp.where(zero, -jnp.inf, jnp.log(safe_x))


def safe_log1mexp(x: jax.Array) -> jax.Array:
    """Return log(1-exp(x)) elementwise, preserving floating dtype and shape.

    For x<=0; zero gives -inf, -inf gives zero, positive/NaN inputs give NaN.
    Guarding zero avoids contaminating active parameter derivatives.

    JAX 0.11.2's nn.log1mexp uses the opposite sign convention and a custom
    JVP based on expm1(-x). Its second derivative can overflow internally
    for large -x, poisoning otherwise finite Hessians (see mixed-objective test).
    """
    zero = x == 0
    near_zero = x > -jnp.log(jnp.asarray(2, dtype=x.dtype))
    # spec 7.2: both branches run under vmap; substitute before every log.
    substitute = jax.lax.stop_gradient(-jnp.ones_like(x))
    expm1_input = jnp.where(near_zero & ~zero, x, substitute)
    exp_input = jnp.where(~near_zero, x, substitute)
    value = jnp.where(
        near_zero, jnp.log(-jnp.expm1(expm1_input)), jnp.log1p(-jnp.exp(exp_input))
    )
    return jnp.where(zero, -jnp.inf, value)


def log_interval_prob(
    self: "SurvivalDistribution", lower: ArrayLike, upper: ArrayLike
) -> jax.Array:
    """Return log P(lower < T <= upper), with trailing-axis broadcasting.

    Args:
        lower: Nonnegative lower endpoints, broadcast with the family batch.
        upper: Upper endpoints, possibly +inf, with the same broadcasting.
    Returns:
        Broadcast log probabilities; equal endpoints give -inf, invalid or
        reversed endpoints give NaN. Unresolved numerical differences give NaN.
    Raises:
        ValueError: If endpoint shapes cannot broadcast.
    """
    a, b = jnp.broadcast_arrays(jnp.asarray(lower), jnp.asarray(upper))
    valid = (a >= 0) & (b >= a) & ~jnp.isnan(a) & ~jnp.isnan(b)
    empty = a == b
    active = valid & ~empty
    # spec 7.2: substitute before evaluating differences, including endpoints.
    a = jnp.where(active, a, 0.0)
    b = jnp.where(active, b, 1.0)
    sa, sb = self.log_survival(a), self.log_survival(b)
    fa, fb = self.log_cdf(a), self.log_cdf(b)
    use_survival = sa < fb
    high = jnp.where(use_survival, sa, fb)
    low = jnp.where(use_survival, sb, fa)
    resolved = jnp.isfinite(high) & (low < high)
    safe_high = jnp.where(resolved, high, 0.0)
    safe_low = jnp.where(resolved, low, -1.0)
    value = safe_high + safe_log1mexp(safe_low - safe_high)
    # spec 7.1: rounded equal finite logs are not evidence of true zero mass.
    value = jnp.where(resolved, value, jnp.where(jnp.isneginf(high), -jnp.inf, jnp.nan))
    value = jnp.where(a == 0, fb, value)
    value = jnp.where(jnp.isposinf(b), sa, value)
    return jnp.where(valid, jnp.where(empty, -jnp.inf, value), jnp.nan)


def log_conditional_interval_prob(
    self: "SurvivalDistribution",
    lower: ArrayLike,
    upper: ArrayLike,
    left_trunc: ArrayLike,
    right_trunc: ArrayLike,
) -> jax.Array:
    """Return log interval mass conditional on (left_trunc, right_trunc].

    Args:
        lower: Nonnegative lower endpoints.
        upper: Upper endpoints, possibly +inf.
        left_trunc: Finite nonnegative window lower endpoints.
        right_trunc: Window upper endpoints strictly above left_trunc.
    Returns:
        Trailing-axis broadcast log probabilities. Empty intersections give
        -inf; full coverage gives exactly zero when window mass is valid.
        Invalid endpoints or zero window mass give NaN.
    Raises:
        ValueError: If input shapes cannot broadcast.
    """
    a, b, window_lower, window_upper = jnp.broadcast_arrays(
        *map(jnp.asarray, (lower, upper, left_trunc, right_trunc))
    )
    valid = (
        (a >= 0)
        & (b >= a)
        & jnp.isfinite(window_lower)
        & (window_lower >= 0)
        & (window_upper > window_lower)
    )
    window_lower, window_upper = (
        jnp.where(valid, window_lower, 0.0),
        jnp.where(valid, window_upper, 1.0),
    )
    lo, hi = jnp.maximum(a, window_lower), jnp.minimum(b, window_upper)
    empty = hi <= lo
    full = (a <= window_lower) & (b >= window_upper)
    # spec 7.2: inactive numerator uses the window, never re-intersected.
    lo = jnp.where(valid & ~empty & ~full, lo, window_lower)
    hi = jnp.where(valid & ~empty & ~full, hi, window_upper)
    denominator = self.log_interval_prob(window_lower, window_upper)
    numerator = self.log_interval_prob(lo, hi)
    usable = valid & jnp.isfinite(denominator)
    value = numerator - jnp.where(usable, denominator, 0.0)
    value = jnp.where(full, 0.0, jnp.where(empty, -jnp.inf, value))
    return jnp.where(usable, value, jnp.nan)
