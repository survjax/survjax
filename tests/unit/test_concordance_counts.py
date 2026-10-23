"""Hand-counted private concordance and exact accumulation (spec 9.1–9.2)."""

from itertools import permutations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from survjax._concordance import _add_count, _pair_counts


def _assert_counts(counts, expected):
    for (high, low), value in zip(counts, expected, strict=True):
        assert high.shape == low.shape == ()
        assert high.dtype == low.dtype == jnp.dtype(jnp.uint32)
        assert int(high) * 2**32 + int(low) == value


def test_mixed_cohort_has_hand_enumerated_counts():
    """Spec 9.2: the earlier event's score classifies each comparable pair."""
    time = jnp.array([1.0, 2.0, 3.0, 4.0])
    exact = jnp.array([True, False, True, True])
    scores = jnp.array([3.0, 1.0, 3.0, 4.0])
    # Comparable (0,1),(0,2),(0,3),(2,3): respectively C,T,D,D.
    counts = _pair_counts(scores, time, exact)
    _assert_counts(counts, (1, 2, 1, 4))
    assert sum(int(hi) * 2**32 + int(lo) for hi, lo in counts[:3]) == 4


@pytest.mark.parametrize(
    ("exact", "expected"),
    [([True, True], (0, 0, 0, 0)), ([False, False], (0, 0, 0, 0))],
    ids=["events", "censorings"],
)
def test_equal_time_same_status_is_not_comparable(exact, expected):
    """Spec 9.2: equal-time event/event and censor/censor pairs are excluded."""
    _assert_counts(
        _pair_counts(jnp.array([2.0, 1.0]), jnp.ones(2), jnp.array(exact)), expected
    )


@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([2.0, 1.0], (1, 0, 0, 1)),
        ([1.0, 2.0], (0, 1, 0, 1)),
        ([2.0, 2.0], (0, 0, 1, 1)),
    ],
    ids=["concordant", "discordant", "score-tied"],
)
def test_equal_time_event_precedes_censoring(reverse, scores, expected):
    """Spec 9.1–9.2: equal-time event/censor pairs use event-first ordering."""
    order = jnp.array([1, 0] if reverse else [0, 1])
    _assert_counts(
        _pair_counts(
            jnp.array(scores)[order], jnp.ones(2), jnp.array([True, False])[order]
        ),
        expected,
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_earlier_censoring_is_not_comparable(reverse):
    """Spec 9.2: an event after censoring does not identify the earlier event."""
    order = jnp.array([1, 0] if reverse else [0, 1])
    _assert_counts(
        _pair_counts(
            jnp.array([2.0, 1.0])[order],
            jnp.array([1.0, 2.0])[order],
            jnp.array([False, True])[order],
        ),
        (0, 0, 0, 0),
    )


@pytest.mark.parametrize(
    ("scores", "expected"),
    [
        ([3.0, 2.0, 1.0], (3, 0, 0, 3)),
        ([1.0, 2.0, 3.0], (0, 3, 0, 3)),
        ([1.0, 1.0, 1.0], (0, 0, 3, 3)),
    ],
    ids=["perfect", "reversed", "tied"],
)
def test_complete_event_cohort_score_order(scores, expected):
    """Spec 9.2: all 3 choose 2 pairs contribute, including score ties."""
    _assert_counts(
        _pair_counts(
            jnp.array(scores), jnp.array([1.0, 2.0, 3.0]), jnp.ones(3, dtype=bool)
        ),
        expected,
    )


def test_adjacent_floating_values_are_not_ties():
    """Spec 9.2: distinct representable times and scores use no tie tolerance."""
    one = jnp.asarray(1.0)
    above = jnp.nextafter(one, jnp.asarray(2.0))
    # Two exact events at distinct times form one pair; the earlier score wins.
    _assert_counts(
        _pair_counts(
            jnp.stack([above, one]), jnp.stack([one, above]), jnp.ones(2, dtype=bool)
        ),
        (1, 0, 0, 1),
    )


@pytest.mark.parametrize("size", [0, 1, 3], ids=["empty", "singleton", "censored"])
def test_zero_pair_cohorts_have_zero_counts(size):
    """Spec 9.2: no pair can be identified without an observed earlier event."""
    time = jnp.arange(1, size + 1, dtype=float)
    exact = jnp.array([True]) if size == 1 else jnp.zeros(size, dtype=bool)
    for count in (_pair_counts, jax.jit(_pair_counts)):
        _assert_counts(count(time, time, exact), (0, 0, 0, 0))


def test_all_permutations_preserve_mixed_time_and_score_ties():
    """Spec 9.2: time/status, rather than array position, determine ordering."""
    time = jnp.array([1.0, 2.0, 2.0, 3.0])
    exact = jnp.array([True, True, False, True])
    scores = jnp.array([3.0, 3.0, 1.0, 4.0])
    # Six pairs: (0,1) T; (0,2),(1,2) C; (0,3),(1,3) D; (2,3) excluded.
    count = jax.jit(_pair_counts)
    for order in permutations(range(4)):
        index = jnp.array(order)
        _assert_counts(count(scores[index], time[index], exact[index]), (2, 2, 1, 5))


def test_eager_jit_and_vmap_agree_on_separate_cohorts():
    """Spec 9.1/12.1: mapped cohorts retain independent exact pair totals."""
    scores = jnp.array([[3.0, 1.0, 3.0, 4.0], [3.0, 3.0, 1.0, 4.0]])
    time = jnp.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 2.0, 3.0]])
    exact = jnp.array([[True, False, True, True], [True, True, False, True]])
    # Pair lists are enumerated in the mixed-cohort and permutation tests.
    expected = [(1, 2, 1, 4), (2, 2, 1, 5)]
    for count in (jax.vmap(_pair_counts), jax.jit(jax.vmap(_pair_counts))):
        batched = count(scores, time, exact)
        for row in range(2):
            _assert_counts(
                jax.tree.map(lambda leaf, row=row: leaf[row], batched), expected[row]
            )
    for row in range(2):
        _assert_counts(_pair_counts(scores[row], time[row], exact[row]), expected[row])


@pytest.mark.parametrize(
    ("high", "low", "increments"),
    [
        (0, 2**31 - 1, [1]),
        (0, 2**32 - 1, [1]),
        (3, 2**32 - 2, [2**31 - 1] * 5),
        (2**30, 17, [0, 2**31 - 1]),
    ],
    ids=["signed-overflow", "low-carry", "repeated-carry", "high-word"],
)
def test_accumulator_matches_python_integer_arithmetic(high, low, increments):
    """Spec 9.1: totals retain high words and carries in either JAX dtype mode."""
    # Python integers supply an exact oracle above int32 capacity, without N² work.
    expected = high * 2**32 + low
    for add in (_add_count, jax.jit(_add_count)):
        total = (jnp.uint32(high), jnp.uint32(low))
        reference = expected
        for increment in increments:
            total = add(total, jnp.int32(increment))
            reference += increment
            _assert_counts((total,), (reference,))
        np.testing.assert_equal(int(total[0]), reference // 2**32)
