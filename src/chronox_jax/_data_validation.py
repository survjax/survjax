"""Private input normalization and observation checks (spec 3.1, 3.3, 6.2)."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike


def _input_dtype(
    value: ArrayLike, name: str
) -> tuple[np.dtype | int | float | jax.Array, bool]:
    """Keep Python scalars weak and inspect typed inputs without host copies."""
    try:
        if isinstance(value, (int, float)) and not isinstance(
            value, (bool, np.generic)
        ):
            return value, False
        if hasattr(value, "dtype"):
            raw = jnp.dtype(value.dtype)
            warn_float64 = raw == jnp.dtype(jnp.float64) and not (
                isinstance(value, jax.Array) and value.weak_type
            )
        else:
            # spec 3.3 / D-002: JAX gives Python lists their default array dtype.
            raw = jnp.asarray(value).dtype
            warn_float64 = False
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric times") from exc
    if not (jnp.issubdtype(raw, jnp.integer) or jnp.issubdtype(raw, jnp.floating)):
        raise TypeError(f"{name} must contain real numeric times")
    if isinstance(value, jax.Array) and value.weak_type:
        # spec 3.3: a dtype alone loses the array's weak JAX promotion.
        return value, False
    return jax.dtypes.canonicalize_dtype(raw), warn_float64


def time_dtype(
    values: dict[str, ArrayLike | None], dtype: DTypeLike | None
) -> np.dtype:
    """Choose one floating dtype for all time and bound arrays."""
    inputs = [
        _input_dtype(value, name) for name, value in values.items() if value is not None
    ]
    if dtype is not None:
        requested = jnp.dtype(dtype)
        if not jnp.issubdtype(requested, jnp.floating):
            raise TypeError("dtype must be floating")
        if requested == jnp.dtype(jnp.float64) and not jax.config.jax_enable_x64:
            raise ValueError("float64 requires JAX_ENABLE_X64=1")
        return jnp.result_type(requested, jnp.float32)

    if not jax.config.jax_enable_x64 and any(warn for _, warn in inputs):
        warnings.warn(
            "float64 time input converted to float32 because JAX 64-bit mode is disabled",
            RuntimeWarning,
            stacklevel=3,
        )
    contributions = [value for value, _ in inputs]
    strong_float = any(
        isinstance(value, np.dtype) and jnp.issubdtype(value, jnp.floating)
        for value in contributions
    )
    if strong_float:
        common = jnp.result_type(*contributions, jnp.float32)
    else:
        # spec 3.3 / D-002: integer-only and weak-scalar-only input uses the default float.
        common = jnp.asarray(0.0).dtype
    return common


def to_shape(value: ArrayLike, shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    """Broadcast an input without growing the observation shape."""
    array = jnp.asarray(value, dtype=dtype)
    if jnp.broadcast_shapes(array.shape, shape) != shape:
        raise ValueError(
            f"input shape {array.shape} would enlarge observation shape {shape}"
        )
    return jnp.broadcast_to(array, shape)


ISSUE_NAMES = (
    "invalid_indicator",
    "conflicting_indicators",
    "missing_time2",
    "invalid_time",
    "invalid_time2",
    "invalid_interval",
    "invalid_truncation",
    "exact_outside_truncation",
    "censored_before_entry",
    "empty_observation",
)


def indicator(
    value: ArrayLike | None, shape: tuple[int, ...], name: str
) -> tuple[jax.Array, jax.Array]:
    """Return a boolean mask and invalid-code mask before losing raw codes."""
    if value is None:
        return jnp.zeros(shape, dtype=bool), jnp.zeros(shape, dtype=bool)
    try:
        raw = jnp.asarray(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain booleans or numeric 0/1 values") from exc
    if not (
        jnp.issubdtype(raw.dtype, jnp.bool_)
        or jnp.issubdtype(raw.dtype, jnp.integer)
        or jnp.issubdtype(raw.dtype, jnp.floating)
    ):
        raise TypeError(f"{name} must contain booleans or numeric 0/1 values")
    if jnp.broadcast_shapes(raw.shape, shape) != shape:
        raise ValueError(
            f"{name} shape {raw.shape} would enlarge observation shape {shape}"
        )
    raw = jnp.broadcast_to(raw, shape)
    invalid = (raw != 0) & (raw != 1)
    return raw == 1, invalid


def row_issues(
    exact: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    left_trunc: jax.Array,
    right_trunc: jax.Array,
    *,
    invalid_indicator: jax.Array,
    conflicting_indicators: jax.Array,
    missing_time2: jax.Array,
    invalid_time: jax.Array,
    invalid_time2: jax.Array,
    check_interval: jax.Array,
) -> jax.Array:
    """Assemble the ten section 6.2 invalid-rule bits, one int32 per row."""
    # spec 6.2: only order endpoints after their presence and domains pass.
    valid_endpoints = ~(invalid_time | missing_time2 | invalid_time2)
    invalid_interval = check_interval & valid_endpoints & (lower >= upper)
    valid_observation = valid_endpoints & ~invalid_interval

    invalid_truncation = (
        ~jnp.isfinite(left_trunc)
        | (left_trunc < 0)
        | jnp.isnan(right_trunc)
        | (right_trunc <= left_trunc)
    )
    valid_window = ~invalid_truncation
    exact_outside = (
        exact
        & valid_observation
        & valid_window
        & ((lower <= left_trunc) | (lower > right_trunc))
    )
    censored_before = (
        ~exact
        & valid_observation
        & valid_window
        & jnp.isposinf(upper)
        & (lower < left_trunc)
    )
    empty = (
        ~exact
        & valid_observation
        & valid_window
        & ((upper <= left_trunc) | (lower >= right_trunc))
    )
    masks = (
        invalid_indicator,
        conflicting_indicators,
        missing_time2,
        invalid_time,
        invalid_time2,
        invalid_interval,
        invalid_truncation,
        exact_outside,
        censored_before,
        empty,
    )
    bits = jnp.zeros(exact.shape, dtype=jnp.int32)
    for position, mask in enumerate(masks):
        bits = bits | jnp.where(mask, jnp.int32(1 << position), jnp.int32(0))
    return bits


def raise_if_invalid(bits: jax.Array) -> None:
    """Eager constructor checkpoint with counts and first row indices."""
    if isinstance(bits, jax.core.Tracer):
        raise ValueError(
            "ordinary SurvivalData constructors require concrete observations; "
            "use SurvivalData.with_issues for traced values"
        )
    host_bits = np.asarray(bits)
    problems = []
    for position, name in enumerate(ISSUE_NAMES):
        mask = (host_bits & (1 << position)) != 0
        count = int(np.count_nonzero(mask))
        if count:
            first = tuple(int(index) for index in np.argwhere(mask)[0])
            problems.append(f"{name}: {count} row(s), first index {first}")
    if problems:
        raise ValueError("invalid SurvivalData: " + "; ".join(problems))
