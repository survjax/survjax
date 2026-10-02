"""Normalized survival observations (spec 3.1–3.3)."""

from contextlib import nullcontext
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike, DTypeLike

from survjax._data_metadata import contains_tracer, may_contain
from survjax._data_validation import (
    flag_fields,
    indicator,
    row_issues,
    time_dtype,
    to_shape,
)
from survjax.diagnostics import DataIssues, _plural

# Longer data print the first and last rows only, like NumPy's summarization.
_REPR_MAX_ROWS = 20
_REPR_HEAD_ROWS = 10
_REPR_TAIL_ROWS = 5


def _observation_text(exact: bool, lower: float, upper: float) -> str:
    """An exact time ``4``, or a censored row's stored set ``(lower, upper]``."""
    if exact:
        return f"{lower:g}"
    return f"({lower:g}, {upper:g}]"


def _window_text(left_trunc: float, right_trunc: float) -> str:
    """The eligibility window, or empty for the default ``(0, inf]``."""
    if left_trunc == 0 and np.isposinf(right_trunc):
        return ""
    return f"window ({left_trunc:g}, {right_trunc:g}]"


@dataclass(frozen=True, eq=False, init=False, repr=False)
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
        enlarging it. Invalid observation values return data with issue bits,
        both eagerly and under JAX transformations. Inspect ``data.issues``
        and explicitly call ``data.issues.raise_if_invalid()`` on the host
        when invalid observations should terminate a task.

        R comparison: the two-argument form uses the event convention of
        ``survival::Surv(time, event)``. Unlike R's default
        ``Surv(time, time2, event)`` counting-process form, ``time2`` here is
        only an interval upper endpoint, never a stop time. R-style 1/2 event
        codes are not inferred.

        Examples:
            ``SurvivalData([2.0, 3.0], [1, 0])`` stores an event at 2 and
            right censoring at 3.
        """
        traced = contains_tracer(
            (
                time,
                event,
                right_censored,
                left_censored,
                interval_censored,
                time2,
                left_trunc,
                right_trunc,
            )
        )
        # spec 6.2: classify inputs before normalization inside a transformation.
        with nullcontext() if traced else jax.ensure_compile_time_eval():
            fields = flag_fields(
                time,
                event,
                right_censored,
                left_censored,
                interval_censored,
                time2,
                left_trunc,
                right_trunc,
                dtype,
            )
            flags = (True,) * 6 if traced else may_contain(*fields[:5])
        self._assign(*fields, may_contain=flags)

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
        then becomes ``lower``. Equal endpoints alone do not mark an event:
        without ``exact=True``, they set ``data.issues.invalid_interval``.
        Construction still returns data; explicit host validation raises.

        ``lower`` fixes the observation shape; other inputs must broadcast
        to it without enlargement. Omitted ``left_trunc`` and
        ``right_trunc`` become 0 and +inf. ``dtype`` selects a common float.
        Invalid observation values return data; inspect ``data.issues`` and
        call its host checkpoint explicitly when needed.

        R comparison: this form resembles
        ``survival::Surv(lower, upper, type="interval2")``, but exactness is
        given by ``exact`` rather than inferred from equal endpoints. Use
        ``+inf`` for an unbounded censored upper endpoint, not R's NA.

        Example:
            ``SurvivalData.from_bounds([2.0, 0.0], [2.0, 3.0],
            exact=[True, False])`` stores an event at 2 and left censoring
            at 3.
        """
        traced = contains_tracer((lower, upper, exact, left_trunc, right_trunc))
        with nullcontext() if traced else jax.ensure_compile_time_eval():
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
            flags = (
                (True,) * 6
                if traced
                else may_contain(
                    exact_mask, first, normalized_upper, lower_bound, upper_bound
                )
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
        result = object.__new__(cls)
        result._assign(
            exact_mask,
            first,
            normalized_upper,
            lower_bound,
            upper_bound,
            bits,
            may_contain=flags,
        )
        return result

    def __getitem__(self, idx: object) -> "SurvivalData":
        """Index observation axes, preserving issue bits and parent flags (spec 3.2).

        All six leaves take the same index and resulting shape. JAX array
        indexing restrictions apply, including fixed output shapes under jit.
        As in JAX, out-of-range integer indices are clamped rather than raising;
        check split indices on the host. There is no time-grid axis.
        """
        return jax.tree.map(lambda leaf: leaf[idx], self)

    @property
    def issues(self) -> DataIssues:
        """Decode per-row issue masks and scalar validity without host checks (spec 6.2)."""
        return DataIssues._from_bits(
            self.exact,
            self.lower,
            self.upper,
            self.left_trunc,
            self.right_trunc,
            self._issue_bits,
        )

    def __repr__(self) -> str:
        """One line per observation; traced data print only shape and dtype.

        Private fields stay out of the repr (spec 3.2); a footer points to
        ``data.issues`` when any row is invalid or uninformative.
        """
        if any(isinstance(leaf, jax.core.Tracer) for leaf in self.tree_flatten()[0]):
            return (
                f"SurvivalData(shape={self.lower.shape}, "
                f"dtype={self.lower.dtype}, traced)"
            )
        shape = self.lower.shape
        indices: list[tuple[int, ...] | None] = list(np.ndindex(shape))
        header = f"SurvivalData: {_plural(len(indices), 'observation')}"
        if len(shape) != 1:
            header += f", shape {shape}"
        header += f", {self.lower.dtype}"
        if len(indices) > _REPR_MAX_ROWS:
            indices = indices[:_REPR_HEAD_ROWS] + [None] + indices[-_REPR_TAIL_ROWS:]

        exact = np.asarray(self.exact)
        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)
        left_trunc = np.asarray(self.left_trunc)
        right_trunc = np.asarray(self.right_trunc)
        rows = []
        for index in indices:
            if index is None:
                rows.append(("...", "", ""))
                continue
            label = str(index[0]) if len(shape) == 1 else str(index)
            observation = _observation_text(
                bool(exact[index]), float(lower[index]), float(upper[index])
            )
            window = _window_text(float(left_trunc[index]), float(right_trunc[index]))
            rows.append((label, observation, window))
        label_width = max((len(row[0]) for row in rows), default=0)
        observation_width = max((len(row[1]) for row in rows), default=0)
        lines = [header]
        for label, observation, window in rows:
            line = f"  {label:>{label_width}}  {observation:<{observation_width}}"
            line = f"{line}   {window}" if window else line
            lines.append(line.rstrip())

        issues = self.issues
        invalid = int(np.count_nonzero(np.asarray(issues.invalid_rows)))
        uninformative = int(np.count_nonzero(np.asarray(issues.uninformative)))
        notes = []
        if invalid:
            notes.append(_plural(invalid, "invalid row"))
        if uninformative:
            notes.append(f"{uninformative} uninformative")
        if notes:
            lines.append(f"issues: {', '.join(notes)} (see .issues)")
        return "\n".join(lines)

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
