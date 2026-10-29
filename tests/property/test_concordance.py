"""Reproducible public concordance identities (spec 9.1–9.2, 12.1)."""

import jax
import jax.numpy as jnp
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from impl_tolerances import ATOL, RTOL

import survjax

PROFILE = settings(derandomize=True, max_examples=40, deadline=None)


@st.composite
def cohorts(draw):
    size = draw(st.integers(1, 8))
    time = draw(st.lists(st.integers(1, 4), min_size=size, max_size=size))
    event = draw(st.lists(st.booleans(), min_size=size, max_size=size))
    scores = draw(st.lists(st.integers(-2, 2), min_size=size, max_size=size))
    order = draw(st.permutations(tuple(range(size))))
    return jnp.array(time, dtype=float), jnp.array(event), jnp.array(scores), order


# Keep compilation outside Hypothesis so each generated cohort size is cached.
_evaluate = jax.jit(
    lambda s, t, e: survjax.concordance_index_with_info(s, time=t, event=e)
)


def _total(count):
    # Python integer arithmetic reconstructs the public two-word representation.
    return int(count.hi) * 2**32 + int(count.lo)


@PROFILE
@given(cohorts())
def test_joint_row_permutation_preserves_every_result_leaf(cohort):
    """Spec 9.2/12.1: unordered pairs make joint row order irrelevant."""
    time, event, scores, order = cohort
    index = jnp.array(order)
    original = _evaluate(scores, time, event)
    permuted = _evaluate(scores[index], time[index], event[index])
    for before, after in zip(
        jax.tree.leaves(original), jax.tree.leaves(permuted), strict=True
    ):
        np.testing.assert_array_equal(before, after)


@PROFILE
@given(cohorts())
def test_pair_partition_and_undefined_mask_match_exact_counts(cohort):
    """Spec 9.2: C,D,T partition comparable pairs; zero pairs alone is undefined."""
    time, event, scores, _ = cohort
    value, info = _evaluate(scores, time, event)
    wins, losses, ties, total = (
        _total(getattr(info, name))
        for name in ("concordant", "discordant", "tied_score", "comparable")
    )
    assert total == wins + losses + ties
    assert not bool(info.invalid | info.unsupported)
    assert bool(info.no_comparable_pairs) == (total == 0)
    assert bool(jnp.isnan(value)) == (total == 0)
    if total:
        assert 0 <= float(value) <= 1


@PROFILE
@given(cohorts())
def test_score_negation_swaps_wins_and_losses_and_complements_value(cohort):
    """Spec 9.2: negation swaps strict ranks and preserves equal-score pairs."""
    time, event, scores, _ = cohort
    value, info = _evaluate(scores, time, event)
    negated, reverse = _evaluate(-scores, time, event)
    for name, original in (
        ("concordant", info.discordant),
        ("discordant", info.concordant),
        ("tied_score", info.tied_score),
        ("comparable", info.comparable),
        ("invalid", info.invalid),
        ("unsupported", info.unsupported),
        ("no_comparable_pairs", info.no_comparable_pairs),
    ):
        for actual, expected in zip(
            jax.tree.leaves(getattr(reverse, name)),
            jax.tree.leaves(original),
            strict=True,
        ):
            np.testing.assert_array_equal(actual, expected)
    if _total(info.comparable):
        x64 = time.dtype == jnp.dtype(jnp.float64)
        # (D+T/2)/M = 1-(C+T/2)/M by the exact pair partition.
        np.testing.assert_allclose(
            negated, 1 - float(value), atol=ATOL[x64], rtol=RTOL[x64]
        )
    else:
        assert np.isnan(value) and np.isnan(negated)
