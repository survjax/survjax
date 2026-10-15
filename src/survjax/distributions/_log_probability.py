"""Stable log-probability arithmetic (spec 7.2; D-022)."""

import jax
import jax.numpy as jnp


def log1mexp(x: jax.Array) -> jax.Array:
    """Return log(1-exp(x)) elementwise, preserving floating dtype and shape.

    Zero gives -inf, -inf gives zero, and positive/NaN inputs give NaN.
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
