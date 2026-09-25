"""Normalized survival observations (spec 3.1–3.3)."""

import warnings
from dataclasses import dataclass

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


def _time_dtype(
    values: dict[str, ArrayLike | None], dtype: DTypeLike | None
) -> np.dtype:
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


def _to_shape(value: ArrayLike, shape: tuple[int, ...], dtype: DTypeLike) -> jax.Array:
    array = jnp.asarray(value, dtype=dtype)
    if jnp.broadcast_shapes(array.shape, shape) != shape:
        raise ValueError(
            f"input shape {array.shape} would enlarge observation shape {shape}"
        )
    return jnp.broadcast_to(array, shape)


@dataclass(frozen=True, eq=False, init=False)
class SurvivalData:
    """Exact and censored observations stored as aligned arrays (spec 3.2).

    Unlike R's ``survival::Surv`` matrix with a type attribute, this PyTree
    stores five public fields: ``exact``, ``lower``, ``upper``, ``left_trunc``,
    and ``right_trunc``. Each has the shape of the first time argument; a
    private issue-bitfield leaf has that shape too. There is no time-grid axis.
    """

    exact: jax.Array
    lower: jax.Array
    upper: jax.Array
    left_trunc: jax.Array
    right_trunc: jax.Array
    _issue_bits: jax.Array
    # spec 3.2: exact, right, left, interval, left truncation, right truncation.
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
        """Build observations from event status or censoring flags (spec 3.1).

        ``SurvivalData(time, event)`` follows R's ``Surv(time, event)``:
        the second positional argument is 1/True for an exact event and
        0/False for right censoring. No R-style 1/2 coding is inferred.
        With no status or flags, every row is exact. To mix kinds, use the
        keyword flags ``right_censored``, ``left_censored``, and
        ``interval_censored``; a row with all flags false is exact. Choose
        either ``event`` or flags, and set at most one flag per row.

        ``time`` is the exact time, one-sided censoring time, or interval's
        lower endpoint. ``time2`` supplies the upper endpoint only for
        interval-censored rows. Unlike ``Surv(time, time2, event)`` with its
        counting-process default, ``time2`` here does not mean a stop time.
        Missing truncation bounds mean ``left_trunc=0`` and
        ``right_trunc=+inf``. ``dtype`` selects a common floating dtype.
        All inputs must broadcast to ``time.shape`` without enlarging it.

        Examples:
            ``SurvivalData([2.0, 3.0], [1, 0])`` stores an event at 2 and
            right censoring at 3.
        """
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
        """Build observation sets from endpoints (spec 3.1–3.2).

        This corresponds to R's ``Surv(lower, upper, type="interval2")``
        form, with an explicit ``exact`` mask. A censored row means
        ``lower < T <= upper``; use ``upper=+inf`` for right censoring and
        ``lower=0`` for left censoring. Mark exact rows with ``exact=True``:
        their event time is ``lower``, and their supplied ``upper`` may equal
        ``lower`` or be NaN; the stored upper becomes ``lower``. Equal
        endpoints alone never imply an exact event. Unlike R's ``interval2``
        input, do not use NA for an unbounded censored endpoint.

        ``lower`` fixes the observation shape; other inputs must broadcast
        to it without enlargement. Omitted ``left_trunc`` and
        ``right_trunc`` become 0 and +inf. ``dtype`` selects a common float.

        Example:
            ``SurvivalData.from_bounds([2.0, 0.0], [2.0, 3.0],
            exact=[True, False])`` stores an event at 2 and left censoring
            at 3.
        """
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
        object.__setattr__(self, "exact", exact)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "left_trunc", left_trunc)
        object.__setattr__(self, "right_trunc", right_trunc)
        object.__setattr__(self, "_issue_bits", issue_bits)
        object.__setattr__(self, "_may_contain", may_contain)

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
