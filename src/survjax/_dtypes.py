"""Shared floating-input policy (spec 3.3)."""

import warnings
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike


def _input_dtype(
    value: ArrayLike, name: str, input_kind: str
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
        raise TypeError(f"{name} must contain real numeric {input_kind}") from exc
    if not (jnp.issubdtype(raw, jnp.integer) or jnp.issubdtype(raw, jnp.floating)):
        raise TypeError(f"{name} must contain real numeric {input_kind}")
    if isinstance(value, jax.Array) and value.weak_type:
        # spec 3.3: a dtype alone loses the array's weak JAX promotion.
        return value, False
    return jax.dtypes.canonicalize_dtype(raw), warn_float64


def floating_dtype(
    values: Mapping[str, ArrayLike | None],
    dtype: DTypeLike | None = None,
    *,
    input_kind: str,
    warning_noun: str,
    warning_stacklevel: int = 3,
) -> np.dtype:
    """Choose a common floating dtype without inspecting array values."""
    inputs = [
        _input_dtype(value, name, input_kind)
        for name, value in values.items()
        if value is not None
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
            f"float64 {warning_noun} input converted to float32 "
            "because JAX 64-bit mode is disabled",
            RuntimeWarning,
            stacklevel=warning_stacklevel,
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


def normalize_parameters(
    values: dict[str, ArrayLike], *, warning_stacklevel: int = 2
) -> dict[str, jax.Array]:
    """Cast one family's own leaves, preserving batch and component shapes.

    Nested distributions are excluded by the caller and keep their own dtypes.
    This checks input kinds only, never parameter domains (spec 3.3, 4.1).
    """
    for name, value in values.items():
        # spec 4.2: required parameter leaves cannot mean an omitted input.
        if value is None:
            raise TypeError(f"{name} must contain real numeric parameters")
    common = floating_dtype(
        values,
        input_kind="parameters",
        warning_noun="parameter",
        warning_stacklevel=warning_stacklevel + 1,
    )
    return {name: jnp.asarray(value, dtype=common) for name, value in values.items()}
