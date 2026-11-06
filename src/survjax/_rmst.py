"""Private numerical expressions for staged RMST."""

import jax
import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaln
from jax.typing import ArrayLike


def _rmst_cutoff(tau: ArrayLike, batch_shape: tuple[int, ...]) -> jax.Array:
    """Convert and broadcast cutoffs; inspect only static dtype and shape."""
    try:
        cutoff = jnp.asarray(tau)
    except (TypeError, ValueError) as exc:
        raise TypeError("tau must contain real numeric values") from exc
    if not (
        jnp.issubdtype(cutoff.dtype, jnp.integer)
        or jnp.issubdtype(cutoff.dtype, jnp.floating)
    ):
        raise TypeError("tau must contain real numeric values")
    try:
        output_shape = jnp.broadcast_shapes(cutoff.shape, batch_shape)
    except ValueError as exc:
        raise TypeError("tau and distribution batch shapes are incompatible") from exc
    return jnp.broadcast_to(cutoff, output_shape)


def _weibull_rmst(shape: jax.Array, scale: jax.Array, cutoff: jax.Array) -> jax.Array:
    """Evaluate valid inputs using a lower series, gamma middle and mean tail."""
    dtype = jnp.result_type(shape, scale, cutoff)
    shape = shape.astype(dtype)
    scale = scale.astype(dtype)
    cutoff = cutoff.astype(dtype)
    zero = cutoff == 0
    # Math: appendices/weibull-rmst.md#derivatives-and-zero-cutoff.
    # spec 8.7/7.2: zero must not reach log; replace its parameters too.
    safe_cutoff = jnp.where(zero, 1.0, cutoff)
    safe_shape = jnp.where(zero, 1.0, shape)
    safe_scale = jnp.where(zero, 1.0, scale)
    log_ratio = jnp.log(safe_cutoff) - jnp.log(safe_scale)
    # spec 8.7: nearby logs lose cutoff precision, amplified by large shape.
    nearby = jnp.abs(log_ratio) < 0.5
    # spec 7.2: the discarded relative difference must not overflow or reach -1.
    nearby_cutoff = jnp.where(nearby, safe_cutoff, 1.0)
    nearby_scale = jnp.where(nearby, safe_scale, 1.0)
    # spec 7.2/8.7: normalize exactly so AD never squares a tiny denominator.
    scale_mantissa, exponent = jnp.frexp(nearby_scale)
    cutoff_mantissa = jnp.ldexp(nearby_cutoff, -exponent)
    relative_difference = (cutoff_mantissa - scale_mantissa) / scale_mantissa
    log_ratio = jnp.where(nearby, jnp.log1p(relative_difference), log_ratio)
    log_x = safe_shape * log_ratio
    small = log_x < jnp.log(0.1)
    # For a=1/k, x>64+4a is well beyond the gamma tail on k in [0.2,50].
    log_tail = jnp.logaddexp(jnp.log(64.0), jnp.log(4.0) - jnp.log(safe_shape))
    large = log_x > log_tail
    middle = ~(small | large)

    series_x = jnp.exp(jnp.where(small, log_x, 0.0))
    series_shape = jnp.where(small, safe_shape, 1.0)
    term = jnp.ones_like(series_x)
    series = term
    # Math: appendices/weibull-rmst.md#lower-tail-series.
    # Integrate exp(-x*u**k) termwise: sum (-x)^n/(n!*(k*n+1)).
    # At x<0.1, the omitted term is bounded by 0.1**13/13! < 2e-23.
    for n in range(1, 13):
        term = term * (-series_x / n)
        series = series + term / (series_shape * n + 1)

    # Math: appendices/weibull-rmst.md#gamma-middle-region.
    gamma_shape = jnp.where(middle, safe_shape, 1.0)
    gamma_scale = jnp.where(middle, safe_scale, 1.0)
    gamma_x = jnp.exp(jnp.where(middle, log_x, 0.0))
    a = 1 / gamma_shape
    probability = gammainc(a, gamma_x)
    # spec 7.1: a lost gamma probability must not become a false zero RMST.
    safe_probability = jnp.where(probability > 0, probability, 1.0)
    log_value = jnp.log(gamma_scale) + gammaln(1 + a) + jnp.log(safe_probability)
    # spec 7.2/8.7: lost P must not overflow an inactive exponential.
    safe_log_value = jnp.where(probability > 0, log_value, 0.0)
    gamma_value = jnp.where(probability > 0, jnp.exp(safe_log_value), jnp.nan)

    # Math: appendices/weibull-rmst.md#mean-tail-approximation.
    tail_shape = jnp.where(large, safe_shape, 1.0)
    tail_scale = jnp.where(large, safe_scale, 1.0)
    mean = jnp.exp(jnp.log(tail_scale) + gammaln(1 + 1 / tail_shape))
    value = jnp.where(small, safe_cutoff * series, jnp.where(large, mean, gamma_value))
    # spec 8.7: retain the active cutoff derivative of one at the origin.
    return jnp.where(zero, cutoff, value)
