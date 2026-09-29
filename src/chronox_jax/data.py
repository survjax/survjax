"""Normalized survival observations (spec 3.1–3.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from chronox_jax._data_validation import (
    indicator,
    raise_if_invalid,
    row_issues,
    time_dtype,
    to_shape,
)


@dataclass(frozen=True, eq=False, init=False)
class SurvivalData:
    """Exact and censored observations stored as aligned arrays (spec 3.2).

    The five public fields are ``exact``, ``lower``, ``upper``, ``left_trunc``,
    and ``right_trunc``. Each has the shape of the first time argument. A
    private issue-bitfield array is a sixth PyTree leaf with the same shape.
    There is no time-grid axis.
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
        **kwargs: ArrayLike,
    ) -> None:
        """Build observations from event status or censoring flags (spec 3.1).

        ``SurvivalData(time, event)`` uses 1/True for an exact event and
        0/False for right censoring. With no status or flags, every row is
        exact. To include left or interval censoring in a mixture, use the
        keyword flags ``right_censored``, ``left_censored``, and
        ``interval_censored``. A row with all flags false is exact. Choose
        either ``event`` or flags, and set at most one flag per row.

        ``time`` is the exact time, one-sided censoring time, or interval's
        lower endpoint. ``time2`` supplies the upper endpoint only for
        interval-censored rows. Missing truncation bounds mean
        ``left_trunc=0`` and ``right_trunc=+inf``. ``dtype`` selects a common
        floating dtype. All inputs must broadcast to ``time.shape`` without
        enlarging it.

        R comparison: the two-argument form uses the event convention of
        ``survival::Surv(time, event)``. Unlike R's default
        ``Surv(time, time2, event)`` counting-process form, ``time2`` here is
        only an interval upper endpoint, never a stop time. R-style 1/2 event
        codes are not inferred.

        Examples:
            ``SurvivalData([2.0, 3.0], [1, 0])`` stores an event at 2 and
            right censoring at 3.
        """
        if "outside_interval_censored" in kwargs:
            raise NotImplementedError("outside_interval_censored is not supported")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")
        common = time_dtype(
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
        right, bad_right = indicator(right_censored, shape, "right_censored")
        left, bad_left = indicator(left_censored, shape, "left_censored")
        interval, bad_interval = indicator(
            interval_censored, shape, "interval_censored"
        )
        invalid_indicator = bad_right | bad_left | bad_interval
        conflicting = (right.astype(jnp.int32) + left + interval) > 1
        if event is None:
            exact = ~(right | left | interval)
        else:
            exact, bad_event = indicator(event, shape, "event")
            invalid_indicator = invalid_indicator | bad_event
            if any(
                flag is not None
                for flag in (right_censored, left_censored, interval_censored)
            ):
                conflicting = jnp.ones(shape, dtype=bool)
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
        raise_if_invalid(bits)
        self._assign(exact, lower, upper, lower_bound, upper_bound, bits)

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

        A censored row represents ``lower < T <= upper``. Use ``upper=+inf``
        for right censoring and ``lower=0`` for left censoring. Mark exact
        rows with ``exact=True``: their event time is ``lower``, and their
        supplied ``upper`` may equal ``lower`` or be NaN. The stored upper
        then becomes ``lower``. Equal endpoints alone do not mark an event.

        ``lower`` fixes the observation shape; other inputs must broadcast
        to it without enlargement. Omitted ``left_trunc`` and
        ``right_trunc`` become 0 and +inf. ``dtype`` selects a common float.

        R comparison: this form resembles
        ``survival::Surv(lower, upper, type="interval2")``, but exactness is
        given by ``exact`` rather than inferred from equal endpoints. Use
        ``+inf`` for an unbounded censored upper endpoint, not R's NA.

        Example:
            ``SurvivalData.from_bounds([2.0, 0.0], [2.0, 3.0],
            exact=[True, False])`` stores an event at 2 and left censoring
            at 3.
        """
        common = time_dtype(
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
        last = to_shape(upper, shape, common)
        exact_mask, invalid_indicator = indicator(exact, shape, "exact")
        normalized_upper = jnp.where(exact_mask, first, last)
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
            exact_mask,
            (first <= 0) | ~jnp.isfinite(first),
            (first < 0) | ~jnp.isfinite(first),
        )
        missing_time2 = ~exact_mask & jnp.isnan(last)
        invalid_time2 = (
            (last < 0)
            | jnp.isneginf(last)
            | (exact_mask & ~jnp.isnan(last) & (last != first))
        )
        bits = row_issues(
            exact_mask,
            first,
            normalized_upper,
            lower_bound,
            upper_bound,
            invalid_indicator=invalid_indicator,
            conflicting_indicators=jnp.zeros(shape, dtype=bool),
            missing_time2=missing_time2,
            invalid_time=invalid_time,
            invalid_time2=invalid_time2,
            check_interval=~exact_mask,
        )
        raise_if_invalid(bits)
        result = object.__new__(cls)
        result._assign(
            exact_mask,
            first,
            normalized_upper,
            lower_bound,
            upper_bound,
            bits,
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
