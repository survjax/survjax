"""A10: property-based concordance checks (spec 9.1, 9.2, 12.1; rev 34).

Hypothesis with ``derandomize=True``. Times are small integers and scores
come from a few values, so time ties of every kind and score ties are
frequent. ``helpers.harrell_reference`` enumerates the 9.2 pair rules directly
in Python (a transcription of the spec, never survjax). Identities:

- counts and value equal the enumeration;
- ``comparable = concordant + discordant + tied_score``;
- row permutation leaves counts unchanged;
- negating scores swaps concordant and discordant: ``C(-s) = 1 - C(s)``;
- no comparable pairs gives NaN and the no-comparable-pair mask.
"""

import numpy as np
import pytest
from helpers import assert_close, b, count, f, harrell_reference, result_mask
from hypothesis import given, settings
from hypothesis import strategies as st

import survjax

PROFILE = settings(derandomize=True, max_examples=60, deadline=None)


@st.composite
def cohorts(draw, min_size=1, max_size=12):
    n = draw(st.integers(min_size, max_size))
    time = draw(st.lists(st.integers(1, 6), min_size=n, max_size=n))
    event = draw(st.lists(st.booleans(), min_size=n, max_size=n))
    scores = draw(st.lists(st.sampled_from([-1.0, 0.0, 0.5, 1.0, 2.0]), min_size=n, max_size=n))
    return np.asarray(time, dtype=float), np.asarray(event), np.asarray(scores)


def _run(time, event, scores):
    value, info = survjax.concordance_index_with_info(
        f(scores), survjax.SurvivalData(f(time), b(event))
    )
    return np.asarray(value), info


@pytest.mark.step("A10")
@PROFILE
@given(cohorts())
def test_counts_and_value_equal_direct_enumeration(cohort):
    """spec 9.2, rev 34: pair counts and `C = (concordant + 0.5 tied_score)/comparable` equal the rules enumerated pair by pair."""
    time, event, scores = cohort
    value, info = _run(time, event, scores)
    conc, disc, tied, comp = harrell_reference(time, event, scores)
    assert count(info, "concordant") == conc
    assert count(info, "discordant") == disc
    assert count(info, "tied_score") == tied
    assert count(info, "comparable") == comp
    if comp == 0:
        assert np.isnan(value)
        assert result_mask(info, "no_comparable_pairs", ()).all()
    else:
        assert_close(value, (conc + 0.5 * tied) / comp)
        assert not result_mask(info, "no_comparable_pairs", ()).any()


@pytest.mark.step("A10")
@PROFILE
@given(cohorts(min_size=2), st.randoms(use_true_random=False))
def test_row_permutation_invariance(cohort, rng):
    """spec 9.2, rev 34: pairs are unordered; permuting rows with their scores leaves value and counts unchanged."""
    time, event, scores = cohort
    perm = list(range(time.shape[0]))
    rng.shuffle(perm)
    value, info = _run(time, event, scores)
    value_p, info_p = _run(time[perm], event[perm], scores[perm])
    for name in ("concordant", "discordant", "tied_score", "comparable"):
        assert count(info_p, name) == count(info, name), name
    np.testing.assert_array_equal(np.isnan(value_p), np.isnan(value))
    if not np.isnan(value):
        assert_close(value_p, value)


@pytest.mark.step("A10")
@PROFILE
@given(cohorts(min_size=2))
def test_negated_scores_give_one_minus_c(cohort):
    """spec 9.2, rev 34: negation swaps concordant and discordant and keeps ties, so `C(-s) = 1 - C(s)` when pairs exist."""
    time, event, scores = cohort
    value, info = _run(time, event, scores)
    value_n, info_n = _run(time, event, -scores)
    assert count(info_n, "concordant") == count(info, "discordant")
    assert count(info_n, "discordant") == count(info, "concordant")
    assert count(info_n, "tied_score") == count(info, "tied_score")
    if not np.isnan(value):
        assert_close(value_n, 1.0 - value)


@pytest.mark.step("A10")
@PROFILE
@given(st.lists(cohorts(min_size=6, max_size=6), min_size=2, max_size=4))
def test_batched_datasets_equal_separate_calls(batch):
    """spec 9.1, rev 34: leading axes "are never pooled": a `(B,N)` call equals `B` separate `(N,)` calls."""
    time = np.stack([c[0] for c in batch])
    event = np.stack([c[1] for c in batch])
    scores = np.stack([c[2] for c in batch])
    value = np.asarray(
        survjax.concordance_index(f(scores), survjax.SurvivalData(f(time), b(event)))
    )
    assert value.shape == (len(batch),)
    for i, (t, e, s) in enumerate(batch):
        expected, _ = _run(t, e, s)
        np.testing.assert_array_equal(np.isnan(value[i]), np.isnan(expected))
        if not np.isnan(expected):
            assert_close(value[i], expected)
