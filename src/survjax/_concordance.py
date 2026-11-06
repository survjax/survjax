"""Private exact pair counts for validated cohorts."""

import jax
import jax.numpy as jnp

_Count = tuple[jax.Array, jax.Array]
_Counts = tuple[_Count, _Count, _Count, _Count]


def _add_count(total: _Count, increment: jax.Array) -> _Count:
    """Add a nonnegative int32 row count to unsigned (hi, lo) words."""
    high, low = total
    new_low = low + increment.astype(jnp.uint32)
    # spec 9.1: retain the carry instead of overflowing an int32 total.
    carry = (new_low < low).astype(jnp.uint32)
    return high + carry, new_low


def _pair_counts(scores: jax.Array, time: jax.Array, exact: jax.Array) -> _Counts:
    """Count concordant, discordant, score-tied and comparable pairs.

    Scores and times are finite (N,) arrays; exact is an aligned boolean
    (N,) mask for one already validated exact/right-censored cohort.
    Each returned count is a pair of uint32 scalar arrays (hi, lo),
    representing hi * 2**32 + lo. Empty cohorts give zero counts. Work is
    O(N squared), with O(N) pair workspace; no pair
    matrix is allocated. Leading-axis batching belongs to the caller.
    """
    zero = (jnp.zeros((), dtype=jnp.uint32), jnp.zeros((), dtype=jnp.uint32))
    initial = (zero, zero, zero, zero)
    size = time.shape[0]
    # Even a zero-trip JAX loop traces its body, whose indexing needs N > 0.
    if size == 0:
        return initial
    indices = jnp.arange(size)

    def count_row(index: jax.Array, totals: _Counts) -> _Counts:
        same_time = time[index] == time
        # spec 9.2: equal-time event/censor pairs place the event first.
        row_earlier = (time[index] < time) | (same_time & exact[index] & ~exact)
        other_earlier = (time < time[index]) | (same_time & exact & ~exact[index])
        comparable = (indices > index) & (
            (row_earlier & exact[index]) | (other_earlier & exact)
        )
        concordant = comparable & (
            (row_earlier & (scores[index] > scores))
            | (other_earlier & (scores > scores[index]))
        )
        discordant = comparable & (
            (row_earlier & (scores[index] < scores))
            | (other_earlier & (scores < scores[index]))
        )
        tied_score = comparable & (scores[index] == scores)
        return (
            _add_count(totals[0], jnp.sum(concordant, dtype=jnp.int32)),
            _add_count(totals[1], jnp.sum(discordant, dtype=jnp.int32)),
            _add_count(totals[2], jnp.sum(tied_score, dtype=jnp.int32)),
            _add_count(totals[3], jnp.sum(comparable, dtype=jnp.int32)),
        )

    # spec 9.2: index ordering counts each unordered pair exactly once.
    return jax.lax.fori_loop(0, size, count_row, initial)
