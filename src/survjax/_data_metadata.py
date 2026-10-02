"""Private observation-input classification and static flags (spec 3.2, 6.2)."""

import jax
import jax.numpy as jnp


def contains_tracer(value: object) -> bool:
    """Classify supplied inputs, including pytrees."""
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def may_contain(
    exact: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    left_trunc: jax.Array,
    right_trunc: jax.Array,
) -> tuple[bool, ...]:
    """Return six presence flags from concrete fields, including rows with issues."""
    # spec 3.3: (0, +inf] is right censored at entry, taking precedence over left.
    right = ~exact & jnp.isposinf(upper)
    left = ~exact & ~right & (lower == 0)
    interval = ~exact & ~right & ~left
    present = jnp.stack(
        [
            jnp.any(exact),
            jnp.any(right),
            jnp.any(left),
            jnp.any(interval),
            jnp.any(left_trunc > 0),
            jnp.any(right_trunc < jnp.inf),
        ]
    )
    # spec 3.2: concrete callers need static Python flags, not host copies of rows.
    return tuple(bool(flag) for flag in jax.device_get(present))
