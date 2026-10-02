"""Diagnostic containers for survival data (spec 6.2–6.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from survjax._data_validation import ISSUE_NAMES


class InvalidDataError(ValueError):
    """An explicit host checkpoint found invalid observation values (spec 6.2)."""


def _plural(count: int, noun: str) -> str:
    """Return ``"1 row"`` or ``"3 rows"`` for repr text."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _rule_counts(
    names: tuple[str, ...], masks: tuple[jax.Array, ...]
) -> list[tuple[str, int, tuple[int, ...]]]:
    """Name, row count and first index of each rule that fires (host only)."""
    counts = []
    for name, mask in zip(names, masks, strict=True):
        host_mask = np.asarray(mask)
        count = int(np.count_nonzero(host_mask))
        if count:
            first = tuple(int(index) for index in np.argwhere(host_mask)[0])
            counts.append((name, count, first))
    return counts


@dataclass(frozen=True, eq=False, repr=False)
class DataIssues:
    """Per-rule data masks with observation shape and scalar validity (spec 6.2)."""

    invalid_indicator: jax.Array
    conflicting_indicators: jax.Array
    missing_time2: jax.Array
    invalid_time: jax.Array
    invalid_time2: jax.Array
    invalid_interval: jax.Array
    invalid_truncation: jax.Array
    exact_outside_truncation: jax.Array
    censored_before_entry: jax.Array
    empty_observation: jax.Array
    uninformative: jax.Array
    invalid_rows: jax.Array
    ok: jax.Array

    @classmethod
    def _from_bits(
        cls,
        exact: jax.Array,
        lower: jax.Array,
        upper: jax.Array,
        left_trunc: jax.Array,
        right_trunc: jax.Array,
        bits: jax.Array,
    ) -> "DataIssues":
        """Decode the sixth SurvivalData leaf without host reduction."""
        masks = tuple((bits & jnp.int32(1 << i)) != 0 for i in range(len(ISSUE_NAMES)))
        invalid_rows = bits != 0
        # spec 2/6.2: a censored set containing the whole window is non-fatal.
        uninformative = (
            ~exact & ~invalid_rows & (lower <= left_trunc) & (upper >= right_trunc)
        )
        return cls(
            masks[0],
            masks[1],
            masks[2],
            masks[3],
            masks[4],
            masks[5],
            masks[6],
            masks[7],
            masks[8],
            masks[9],
            uninformative,
            invalid_rows,
            ~jnp.any(invalid_rows),
        )

    def raise_if_invalid(self) -> None:
        """Raise descriptive row errors at an explicit host-only checkpoint."""
        if isinstance(self.ok, jax.core.Tracer):
            raise ValueError("DataIssues.raise_if_invalid is host-only")
        rule_masks = self.tree_flatten()[0][:10]
        problems = [
            f"{name}: {count} row(s), first index {first}"
            for name, count, first in _rule_counts(ISSUE_NAMES, rule_masks)
        ]
        if problems:
            raise InvalidDataError("invalid SurvivalData: " + "; ".join(problems))

    def __repr__(self) -> str:
        """Summarize the rules that fire; traced masks print only their shape."""
        if any(isinstance(leaf, jax.core.Tracer) for leaf in self.tree_flatten()[0]):
            return f"DataIssues(shape={self.invalid_rows.shape}, traced)"
        total = int(np.size(self.invalid_rows))
        invalid = int(np.count_nonzero(np.asarray(self.invalid_rows)))
        if invalid:
            lines = [f"DataIssues: {invalid} of {_plural(total, 'row')} invalid"]
        else:
            lines = [f"DataIssues: all {_plural(total, 'row')} valid"]
        # spec 6.2: uninformative rows are listed but are not invalid.
        names = (*ISSUE_NAMES, "uninformative")
        masks = (*self.tree_flatten()[0][:10], self.uninformative)
        for name, count, first in _rule_counts(names, masks):
            note = "  (not an error)" if name == "uninformative" else ""
            lines.append(
                f"  {name:<26}{_plural(count, 'row')}, first index {first}{note}"
            )
        return "\n".join(lines)

    def tree_flatten(self) -> tuple[tuple[jax.Array, ...], None]:
        """Keep every mask and the scalar reduction as PyTree leaves."""
        return (
            (
                self.invalid_indicator,
                self.conflicting_indicators,
                self.missing_time2,
                self.invalid_time,
                self.invalid_time2,
                self.invalid_interval,
                self.invalid_truncation,
                self.exact_outside_truncation,
                self.censored_before_entry,
                self.empty_observation,
                self.uninformative,
                self.invalid_rows,
                self.ok,
            ),
            None,
        )

    @classmethod
    def tree_unflatten(
        cls, _aux: None, children: tuple[jax.Array, ...]
    ) -> "DataIssues":
        """Rebuild from leaves without a host checkpoint (spec 6.3)."""
        return cls(
            children[0],
            children[1],
            children[2],
            children[3],
            children[4],
            children[5],
            children[6],
            children[7],
            children[8],
            children[9],
            children[10],
            children[11],
            children[12],
        )


jax.tree_util.register_pytree_node_class(DataIssues)
