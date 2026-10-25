"""Concordance result types and private finalization (spec 6.3, 9.1–9.2)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from survjax._concordance import _Counts


@dataclass(frozen=True, eq=False)
class ConcordanceCount:
    """An exact nonnegative count stored as two uint32 array words.

    Args:
        hi: Most significant word, contributing hi * 2**32.
        lo: Least significant word. Must have the same shape as hi.

    Both fields are uint32 arrays with the result's leading batch shape;
    they have no observation or time-grid axis. The count is hi * 2**32 + lo.
    The same representation is used with JAX 64-bit mode enabled or disabled.
    """

    hi: jax.Array
    lo: jax.Array


@dataclass(frozen=True, eq=False)
class ConcordanceInfo:
    """Exact pair counts and per-result concordance diagnostics.

    Args:
        concordant: Pairs with a higher score on the earlier event.
        discordant: Pairs with a lower score on the earlier event.
        tied_score: Comparable pairs with equal scores; receive half credit.
        comparable: Total comparable pairs.
        invalid: A score or observation in the cohort is invalid.
        unsupported: An observation in the cohort uses an unsupported mode.
        no_comparable_pairs: An evaluated cohort has no comparable pairs.

    Each mask is boolean, with the same leading batch shape as every count
    word. There is no observation or time-grid axis. Failed cohorts have zero
    counts and no_comparable_pairs=False; invalid and unsupported may overlap.
    """

    concordant: ConcordanceCount
    discordant: ConcordanceCount
    tied_score: ConcordanceCount
    comparable: ConcordanceCount
    invalid: jax.Array
    unsupported: jax.Array
    no_comparable_pairs: jax.Array


jax.tree_util.register_dataclass(ConcordanceCount)
jax.tree_util.register_dataclass(ConcordanceInfo)


def _concordance_result(
    counts: _Counts,
    invalid: jax.Array,
    unsupported: jax.Array,
    dtype: DTypeLike,
) -> tuple[jax.Array, ConcordanceInfo]:
    """Finalize aligned count words/masks in the normalized data float dtype.

    Counts are concordant, discordant, tied-score and comparable uint32 word
    pairs. All words and boolean masks have the result's leading batch shape.
    The caller validates observations and substitutes safe cohorts before
    counting. This helper preserves failure masks and zeros failed counts.
    """
    failed = invalid | unsupported
    zero = jnp.zeros((), dtype=jnp.uint32)
    safe_counts = tuple(
        (jnp.where(failed, zero, hi), jnp.where(failed, zero, lo)) for hi, lo in counts
    )
    concordant, discordant, tied_score, comparable = (
        ConcordanceCount(hi, lo) for hi, lo in safe_counts
    )
    no_pairs = (comparable.hi == 0) & (comparable.lo == 0)
    # spec 3.3/9.1: cast both words before arithmetic, including division.
    word_base = jnp.asarray(2**32, dtype=dtype)
    wins, _, ties, total = (
        hi.astype(dtype) * word_base + lo.astype(dtype) for hi, lo in safe_counts
    )
    # spec 7.2: substitute before division, including for undefined cohorts.
    denominator = jnp.where(no_pairs, jnp.ones((), dtype=dtype), total)
    ratio = (wins + jnp.asarray(0.5, dtype=dtype) * ties) / denominator
    value = jnp.where(failed | no_pairs, jnp.nan, ratio)
    info = ConcordanceInfo(
        concordant,
        discordant,
        tied_score,
        comparable,
        invalid,
        unsupported,
        ~failed & no_pairs,
    )
    return value, info
