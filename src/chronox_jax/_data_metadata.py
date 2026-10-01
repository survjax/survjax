"""Private observation-input classification and static flags (spec 3.2, 6.2)."""

import jax
import numpy as np


def contains_tracer(value: object) -> bool:
    """Classify supplied inputs, including nested Python lists and tuples."""
    if isinstance(value, jax.core.Tracer):
        return True
    if isinstance(value, (list, tuple)):
        return any(contains_tracer(item) for item in value)
    return False


def may_contain(
    exact: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    left_trunc: jax.Array,
    right_trunc: jax.Array,
) -> tuple[bool, bool, bool, bool, bool, bool]:
    """Reduce concrete normalized fields, including rows with issues."""
    # spec 3.2: callers evaluate concrete normalization before host inspection.
    exact_rows = np.asarray(exact)
    lower_values = np.asarray(lower)
    upper_values = np.asarray(upper)
    right = ~exact_rows & np.isposinf(upper_values)
    left = ~exact_rows & ~right & (lower_values == 0)
    interval = ~exact_rows & ~right & ~left
    return (
        bool(np.any(exact_rows)),
        bool(np.any(right)),
        bool(np.any(left)),
        bool(np.any(interval)),
        bool(np.any(np.asarray(left_trunc) > 0)),
        bool(np.any(np.asarray(right_trunc) < np.inf)),
    )
