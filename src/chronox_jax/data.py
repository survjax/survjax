"""Normalized survival observations (spec 3.1–3.3)."""

import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike


def _input_dtype(value: ArrayLike, name: str) -> np.dtype:
    """Inspect a dtype without moving traced values to the host."""
    try:
        if hasattr(value, "dtype"):
            raw = value.dtype
        elif any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value)):
            raw = jnp.asarray(value).dtype
        else:
            raw = np.asarray(value).dtype
        result = jnp.dtype(raw)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain real numeric times") from exc
    if not (
        jnp.issubdtype(result, jnp.integer) or jnp.issubdtype(result, jnp.floating)
    ):
        raise TypeError(f"{name} must contain real numeric times")
    return result


def _time_dtype(
    values: dict[str, ArrayLike | None], dtype: DTypeLike | None
) -> np.dtype:
    raw_dtypes = [
        _input_dtype(value, name) for name, value in values.items() if value is not None
    ]
    if dtype is not None:
        requested = jnp.dtype(dtype)
        if not jnp.issubdtype(requested, jnp.floating):
            raise TypeError("dtype must be floating")
        if requested == jnp.dtype(jnp.float64) and not jax.config.jax_enable_x64:
            raise ValueError("float64 requires JAX_ENABLE_X64=1")
        return jnp.result_type(requested, jnp.float32)

    if not jax.config.jax_enable_x64 and jnp.dtype(jnp.float64) in raw_dtypes:
        warnings.warn(
            "float64 time input converted to float32 because JAX 64-bit mode is disabled",
            RuntimeWarning,
            stacklevel=3,
        )
    canonical = [jax.dtypes.canonicalize_dtype(raw) for raw in raw_dtypes]
    if any(jnp.issubdtype(raw, jnp.floating) for raw in canonical):
        common = jnp.result_type(*canonical, jnp.float32)
    else:
        common = jnp.asarray(0.0).dtype
    return common


def _to_shape(value: ArrayLike, shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    array = jnp.asarray(value, dtype=dtype)
    if jnp.broadcast_shapes(array.shape, shape) != shape:
        raise ValueError(
            f"input shape {array.shape} would enlarge observation shape {shape}"
        )
    return jnp.broadcast_to(array, shape)


@dataclass(frozen=True, init=False)
class SurvivalData:
    """Observation-shaped arrays for exact and censored times (spec 3.2).

    All six leaves have the shape of ``time`` or ``lower``. Grid conventions do
    not apply to this observation container.
    """

    exact: jax.Array
    lower: jax.Array
    upper: jax.Array
    left_trunc: jax.Array
    right_trunc: jax.Array
    _issue_bits: jax.Array
    _may_contain: tuple[bool, bool, bool, bool, bool, bool]

    def __init__(
        self,
        time: ArrayLike,
        event: ArrayLike | None = None,
        *,
        right_censored: ArrayLike | None = None,
        left_censored: ArrayLike | None = None,
        interval_censored: ArrayLike | None = None,
        time2: ArrayLike | None = None,
        left_trunc: ArrayLike | None = None,
        right_trunc: ArrayLike | None = None,
        dtype: DTypeLike | None = None,
    ) -> None:
        common = _time_dtype(
            {
                "time": time,
                "time2": time2,
                "left_trunc": left_trunc,
                "right_trunc": right_trunc,
            },
            dtype,
        )
        first = jnp.asarray(time, dtype=common)
        shape = first.shape
        right = (
            jnp.zeros(shape, dtype=bool)
            if right_censored is None
            else _to_shape(right_censored, shape, bool)
        )
        left = (
            jnp.zeros(shape, dtype=bool)
            if left_censored is None
            else _to_shape(left_censored, shape, bool)
        )
        interval = (
            jnp.zeros(shape, dtype=bool)
            if interval_censored is None
            else _to_shape(interval_censored, shape, bool)
        )
        if event is None:
            exact = ~(right | left | interval)
        else:
            exact = _to_shape(event, shape, bool)
            right = ~exact
        second = first if time2 is None else _to_shape(time2, shape, common)
        lower = jnp.where(left, jnp.zeros_like(first), first)
        upper = jnp.where(
            right, jnp.full_like(first, jnp.inf), jnp.where(interval, second, first)
        )
        lower_bound = (
            jnp.zeros_like(first)
            if left_trunc is None
            else _to_shape(left_trunc, shape, common)
        )
        upper_bound = (
            jnp.full_like(first, jnp.inf)
            if right_trunc is None
            else _to_shape(right_trunc, shape, common)
        )
        self._assign(
            exact, lower, upper, lower_bound, upper_bound, jnp.zeros(shape, jnp.int32)
        )

    @classmethod
    def from_bounds(
        cls,
        lower: ArrayLike,
        upper: ArrayLike,
        *,
        exact: ArrayLike | None = None,
        left_trunc: ArrayLike | None = None,
        right_trunc: ArrayLike | None = None,
        dtype: DTypeLike | None = None,
    ) -> "SurvivalData":
        """Store valid lower/upper observations without inferring exact rows."""
        common = _time_dtype(
            {
                "lower": lower,
                "upper": upper,
                "left_trunc": left_trunc,
                "right_trunc": right_trunc,
            },
            dtype,
        )
        first = jnp.asarray(lower, dtype=common)
        shape = first.shape
        last = _to_shape(upper, shape, common)
        exact_mask = (
            jnp.zeros(shape, dtype=bool)
            if exact is None
            else _to_shape(exact, shape, bool)
        )
        normalized_upper = jnp.where(exact_mask, first, last)
        lower_bound = (
            jnp.zeros_like(first)
            if left_trunc is None
            else _to_shape(left_trunc, shape, common)
        )
        upper_bound = (
            jnp.full_like(first, jnp.inf)
            if right_trunc is None
            else _to_shape(right_trunc, shape, common)
        )
        result = object.__new__(cls)
        result._assign(
            exact_mask,
            first,
            normalized_upper,
            lower_bound,
            upper_bound,
            jnp.zeros(shape, jnp.int32),
        )
        return result

    def _assign(
        self,
        exact: jax.Array,
        lower: jax.Array,
        upper: jax.Array,
        left_trunc: jax.Array,
        right_trunc: jax.Array,
        issue_bits: jax.Array,
        may_contain: tuple[bool, ...] = (True,) * 6,
    ) -> None:
        for name, value in zip(
            (
                "exact",
                "lower",
                "upper",
                "left_trunc",
                "right_trunc",
                "_issue_bits",
                "_may_contain",
            ),
            (exact, lower, upper, left_trunc, right_trunc, issue_bits, may_contain),
            strict=True,
        ):
            object.__setattr__(self, name, value)

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], tuple[bool, ...]]:
        """Keep the issue bitfield as an array leaf (spec 3.2)."""
        return (
            (
                self.exact,
                self.lower,
                self.upper,
                self.left_trunc,
                self.right_trunc,
                self._issue_bits,
            ),
            self._may_contain,
        )

    @classmethod
    def tree_unflatten(
        cls, may_contain: tuple[bool, ...], children: tuple[jax.Array, ...]
    ) -> "SurvivalData":
        """Rebuild from leaves without applying public normalization (spec 3.2)."""
        result = object.__new__(cls)
        result._assign(
            children[0],
            children[1],
            children[2],
            children[3],
            children[4],
            children[5],
            may_contain=may_contain,
        )
        return result


jax.tree_util.register_pytree_node_class(SurvivalData)
