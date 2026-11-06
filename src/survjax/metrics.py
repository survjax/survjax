"""Harrell concordance for exact/right observations."""

from dataclasses import dataclass
from math import prod

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike, DTypeLike

from survjax._concordance import _Counts, _pair_counts
from survjax._data_validation import current_field_issues
from survjax._observation_inputs import _resolve_data
from survjax.data import SurvivalData


@dataclass(frozen=True, eq=False)
class ConcordanceCount:
    """An exact nonnegative count stored as two uint32 array words.

    Attributes:
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

    Attributes:
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


def _evaluate_concordance(
    scores: ArrayLike, data: SurvivalData
) -> tuple[jax.Array, ConcordanceInfo]:
    """Validate cohorts and count with O(N) pair workspace per result."""
    try:
        scores = jnp.asarray(scores)
    except (TypeError, ValueError) as exc:
        raise TypeError("scores must contain real numeric values") from exc
    if jnp.issubdtype(scores.dtype, jnp.complexfloating):
        raise TypeError("scores must contain real numeric values")
    if scores.ndim == 0 or data.lower.ndim == 0:
        raise TypeError("scores and data must have a last observation axis")
    if scores.shape[-1] != data.lower.shape[-1]:
        raise TypeError("scores and data must have matching observation lengths")
    try:
        batch_shape = jnp.broadcast_shapes(scores.shape[:-1], data.lower.shape[:-1])
    except ValueError as exc:
        raise TypeError(
            "scores and data have incompatible leading batch shapes"
        ) from exc

    current = current_field_issues(
        data.exact, data.lower, data.upper, data.left_trunc, data.right_trunc
    )
    invalid_data = jnp.any(data.issues.invalid_rows | (current != 0), axis=-1)
    invalid_scores = jnp.any(~jnp.isfinite(scores), axis=-1)
    invalid = jnp.broadcast_to(invalid_data, batch_shape) | jnp.broadcast_to(
        invalid_scores, batch_shape
    )
    unsupported_data = jnp.any(
        (data.left_trunc != 0)
        | ~jnp.isposinf(data.right_trunc)
        | (~data.exact & ~jnp.isposinf(data.upper)),
        axis=-1,
    )
    unsupported = jnp.broadcast_to(unsupported_data, batch_shape)
    failed = (invalid | unsupported)[..., None]
    shape = (*batch_shape, scores.shape[-1])
    # spec 9.1–9.2: substitute whole failed cohorts; never rank a partial cohort.
    safe_scores = jnp.where(failed, 0, jnp.broadcast_to(scores, shape))
    safe_time = jnp.where(failed, 1, jnp.broadcast_to(data.lower, shape))
    safe_exact = jnp.where(failed, False, jnp.broadcast_to(data.exact, shape))
    # Explicit dimensions also handle N=0, where reshape(-1, N) is ambiguous.
    flat_shape = (prod(batch_shape), shape[-1])
    counts = jax.vmap(_pair_counts)(
        safe_scores.reshape(flat_shape),
        safe_time.reshape(flat_shape),
        safe_exact.reshape(flat_shape),
    )
    counts = jax.tree.map(lambda word: word.reshape(batch_shape), counts)
    return _concordance_result(counts, invalid, unsupported, data.lower.dtype)


def concordance_index(
    scores: ArrayLike,
    data: SurvivalData | None = None,
    *,
    time: ArrayLike | None = None,
    event: ArrayLike | None = None,
) -> jax.Array:
    """Return Harrell's C for exact/right observations without truncation.

    Higher scores predict earlier events. Equal-time event/censor pairs put
    the event first; other equal-time pairs are excluded. Score ties receive
    half credit. No useful score gradients are promised. Work is O(N squared),
    pair workspace O(N) per result.

    Args:
        scores: Finite real scores with last axis N, compared at their JAX precision.
        data: Prepared observations, supplied instead of time and event.
        time: Raw exact/right times; requires event.
        event: Raw 1/True event and 0/False censoring indicators; requires time.

    Returns:
        One value per broadcast leading batch, in the data's floating dtype.
        Last observation axes must match and are reduced, never broadcast.
        No time-grid axis. Invalid, unsupported or zero-pair cohorts give NaN.
        Use concordance_index_with_info for counts and diagnostic masks.

    Raises:
        TypeError: Static input-form, type or shape misuse. Invalid runtime
            values return NaN eagerly and under jit/vmap, without raising.
    """
    return concordance_index_with_info(scores, data, time=time, event=event)[0]


def concordance_index_with_info(
    scores: ArrayLike,
    data: SurvivalData | None = None,
    *,
    time: ArrayLike | None = None,
    event: ArrayLike | None = None,
) -> tuple[jax.Array, ConcordanceInfo]:
    """Return Harrell's C and exact pair counts with per-result diagnostics.

    Args:
        scores: Finite real score arrays; higher scores predict earlier events.
        data: Prepared observations, supplied instead of time and event.
        time: Raw exact/right times; requires event.
        event: Raw 1/True event and 0/False censoring indicators; requires time.

    Returns:
        Exactly (value, ConcordanceInfo). Value semantics and complexity are
        those of concordance_index. Every info leaf has the leading broadcast
        result shape, with no observation or time-grid axis. Failed cohorts
        have zero counts; only evaluated zero-pair cohorts set
        no_comparable_pairs. No rows are dropped. No useful score gradients
        are promised.

    Raises:
        TypeError: Static input-form, type or shape misuse, as for
            concordance_index. Runtime issues are returned in info.
    """
    return _evaluate_concordance(scores, _resolve_data(data, time, event))
