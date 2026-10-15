"""Guarded log arithmetic, without clipping or value validation (spec 7.2)."""

import jax
import jax.numpy as jnp


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


def log_cdf_from_log_cumulative_hazard(log_h: jax.Array) -> jax.Array:
    """Return log(1-exp(-H)) from log H, where H is cumulative hazard.

    For small H, F=H-H**2/2+O(H**3), so log F=log H-H/2+O(H**2).
    Below dtype epsilon, log F=log H to rounding precision. Zero/infinite H
    give -inf/zero; the guarded standard branch retains upper-tail precision.
    Floating dtype and shape are preserved.
    """
    dtype = log_h.dtype
    log_eps = jnp.log(jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype))
    log_max = jnp.log(jnp.asarray(jnp.finfo(dtype).max, dtype=dtype))
    tiny = log_h < log_eps
    upper = log_h >= log_max

    # spec 4.1/7.2: preserve tiny log H; substitute before exp can overflow.
    # zeros_like is constant with zero tangent; stop_gradient is redundant here.
    safe_log_h = jnp.where(tiny | upper, jnp.zeros_like(log_h), log_h)
    log_cdf_from_h = safe_log1mexp(-jnp.exp(safe_log_h))
    log_cdf = jnp.where(tiny, log_h, log_cdf_from_h)
    return jnp.where(upper, jnp.zeros_like(log_cdf), log_cdf)
