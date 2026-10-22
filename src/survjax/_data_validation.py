"""Private input normalization and observation checks (spec 3.1, 3.3, 6.2)."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike

from ._dtypes import floating_dtype


def time_dtype(
    values: dict[str, ArrayLike | None],
    dtype: DTypeLike | None,
    *,
    warning_stacklevel: int = 3,
) -> np.dtype:
    """Choose one floating dtype for all time and bound arrays."""
    return floating_dtype(
        values,
        dtype,
        input_kind="times",
        warning_noun="time",
        warning_stacklevel=warning_stacklevel + 1,
    )


def to_shape(value: ArrayLike, shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    """Broadcast an input without growing the observation shape."""
    array = jnp.asarray(value, dtype=dtype)
    try:
        broadcast_shape = jnp.broadcast_shapes(array.shape, shape)
    except ValueError as exc:
        raise TypeError(
            f"input shape {array.shape} is incompatible with observation shape {shape}"
        ) from exc
    if broadcast_shape != shape:
        raise TypeError(
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
    try:
        broadcast_shape = jnp.broadcast_shapes(raw.shape, shape)
    except ValueError as exc:
        raise TypeError(
            f"{name} shape {raw.shape} is incompatible with observation shape {shape}"
        ) from exc
    if broadcast_shape != shape:
        raise TypeError(
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
    """Add interval, truncation, and window checks; pack ten bits per row.

    Both constructors call this with stored fields and raw-input fault masks
    (spec 6.2).
    """
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


def stored_field_issues(
    exact: jax.Array,
    lower: jax.Array,
    upper: jax.Array,
    left_trunc: jax.Array,
    right_trunc: jax.Array,
) -> jax.Array:
    """Recheck normalized fields after tree operations (spec 6.2/7.1/12.1).

    Reconstruction bypasses constructors, so current fields can violate rules
    even when the stored construction bits are zero. These bits describe only
    current fields; callers must also retain the original construction issues.
    """
    # spec 3.1/6.2: normalization loses raw status codes and conflicting flags.
    # Their evidence survives only in the original construction issues.
    no_fault = jnp.zeros_like(exact)
    # spec 3.3: lower is finite, positive for events and nonnegative otherwise.
    invalid_time = ~jnp.isfinite(lower) | (lower < 0) | (exact & (lower == 0))
    # spec 3.2: exact stored upper=lower; unlike raw bounds, it is not optional.
    invalid_time2 = (upper < 0) | jnp.isneginf(upper) | (exact & (upper != lower))
    return row_issues(
        exact,
        lower,
        upper,
        left_trunc,
        right_trunc,
        invalid_indicator=no_fault,
        conflicting_indicators=no_fault,
        # spec 6.2: only censored normalized upper endpoints can be missing.
        missing_time2=~exact & jnp.isnan(upper),
        invalid_time=invalid_time,
        invalid_time2=invalid_time2,
        check_interval=~exact,
    )


def flag_fields(
    time: ArrayLike,
    event: ArrayLike | None,
    right_censored: ArrayLike | None,
    left_censored: ArrayLike | None,
    interval_censored: ArrayLike | None,
    time2: ArrayLike | None,
    left_trunc: ArrayLike | None,
    right_trunc: ArrayLike | None,
    dtype: DTypeLike | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Normalize flag inputs and compute issue bits without a host checkpoint.

    Build the five stored fields and input-fault masks, then pass them to
    ``row_issues`` (spec 6.2).
    """
    if event is not None and any(
        flag is not None for flag in (right_censored, left_censored, interval_censored)
    ):
        # spec 3.1 / D-011: argument-presence conflict is static misuse.
        raise TypeError("event cannot be combined with censoring flags")
    common = time_dtype(
        {
            "time": time,
            "time2": time2,
            "left_trunc": left_trunc,
            "right_trunc": right_trunc,
        },
        dtype,
        warning_stacklevel=4,
    )
    first = jnp.asarray(time, dtype=common)
    shape = first.shape
    right, bad_right = indicator(right_censored, shape, "right_censored")
    left, bad_left = indicator(left_censored, shape, "left_censored")
    interval, bad_interval = indicator(interval_censored, shape, "interval_censored")
    invalid_indicator = bad_right | bad_left | bad_interval
    conflicting = (right.astype(jnp.int32) + left + interval) > 1
    if event is None:
        exact = ~(right | left | interval)
    else:
        exact, bad_event = indicator(event, shape, "event")
        invalid_indicator = invalid_indicator | bad_event
        right = ~exact
    second = first if time2 is None else to_shape(time2, shape, common)
    lower = jnp.where(left, jnp.zeros_like(first), first)
    upper = jnp.where(
        right, jnp.full_like(first, jnp.inf), jnp.where(interval, second, first)
    )
    lower_bound = (
        jnp.zeros_like(first)
        if left_trunc is None
        else to_shape(left_trunc, shape, common)
    )
    upper_bound = (
        jnp.full_like(first, jnp.inf)
        if right_trunc is None
        else to_shape(right_trunc, shape, common)
    )
    invalid_time = jnp.where(
        exact,
        (first <= 0) | ~jnp.isfinite(first),
        (first < 0) | ~jnp.isfinite(first),
    )
    missing_time2 = interval & (time2 is None or jnp.isnan(second))
    invalid_time2 = interval & ((second < 0) | jnp.isneginf(second))
    bits = row_issues(
        exact,
        lower,
        upper,
        lower_bound,
        upper_bound,
        invalid_indicator=invalid_indicator,
        conflicting_indicators=conflicting,
        missing_time2=missing_time2,
        invalid_time=invalid_time,
        invalid_time2=invalid_time2,
        check_interval=interval,
    )
    return exact, lower, upper, lower_bound, upper_bound, bits
