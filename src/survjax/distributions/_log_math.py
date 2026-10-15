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
