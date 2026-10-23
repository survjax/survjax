"""A10: hand-enumerated Harrell's concordance values and pair counts (spec 9.1, 9.2; rev 34).

Spec 9.2: "Each unordered pair is counted once. Unequal times are comparable
when the earlier one is exact. Equal-time event/event and censor/censor pairs
are excluded; an equal-time event/censor pair is comparable, with the event
earlier. A larger score on the earlier event is concordant, a smaller one
discordant, tied scores get half credit: C = (concordant + 0.5 * tied_score) /
comparable. No comparable pairs gives NaN." 9.1: "at equal times, events
precede censorings". Higher scores mean earlier predicted events.

Every expected count below is enumerated by hand in the docstring;
``helpers.harrell_reference`` is used only in the property tests.
"""

import jax
import numpy as np
import pytest
from helpers import assert_close, b, count, f, result_mask

import survjax

EXACT, CENS = True, False


def _with_info(time, event, scores):
    value, info = survjax.concordance_index_with_info(
        f(scores), survjax.SurvivalData(f(time), b(event))
    )
    return np.asarray(value), info


def _assert_counts(info, concordant, discordant, tied_score, comparable):
    assert count(info, "concordant") == concordant
    assert count(info, "discordant") == discordant
    assert count(info, "tied_score") == tied_score
    assert count(info, "comparable") == comparable


def _assert_no_comparable_pairs(time, event, scores):
    value, info = _with_info(time, event, scores)
    assert np.isnan(value)
    assert count(info, "comparable") == 0
    assert result_mask(info, "no_comparable_pairs", ()).all()
    assert not result_mask(info, "invalid", ()).any()
    assert not result_mask(info, "unsupported", ()).any()


# ------------------------------------------------------------ basic orderings


@pytest.mark.step("A10")
def test_perfectly_concordant_scores_give_one():
    """spec 9.2, rev 34: "higher means earlier predicted event"; scores decreasing in event time give `C=1`.

    Four events at ``1,2,3,4``, scores ``4,3,2,1``: all 6 pairs comparable and
    concordant.
    """
    value, info = _with_info([1.0, 2.0, 3.0, 4.0], [EXACT] * 4, [4.0, 3.0, 2.0, 1.0])
    assert_close(value, 1.0)
    _assert_counts(info, 6, 0, 0, 6)


@pytest.mark.step("A10")
def test_reversed_scores_give_zero():
    """spec 9.2, rev 34: a smaller score on the earlier event is discordant; all 6 pairs discordant give `C=0`."""
    value, info = _with_info([1.0, 2.0, 3.0, 4.0], [EXACT] * 4, [1.0, 2.0, 3.0, 4.0])
    assert_close(value, 0.0)
    _assert_counts(info, 0, 6, 0, 6)


@pytest.mark.step("A10")
def test_all_tied_scores_give_one_half():
    """spec 9.2, rev 34: "tied scores get half credit"; 6 tied comparable pairs give `C=0.5`."""
    value, info = _with_info([1.0, 2.0, 3.0, 4.0], [EXACT] * 4, [7.0] * 4)
    assert_close(value, 0.5)
    _assert_counts(info, 0, 0, 6, 6)


@pytest.mark.step("A10")
def test_mixed_case_with_censoring_and_ties():
    """spec 9.2, rev 34: the five-row case of `test_a10_concordance_api.py`: 4 concordant, 1 discordant, 1 tied, `C=0.75`."""
    value, info = _with_info(
        [1.0, 2.0, 3.0, 3.0, 4.0],
        [EXACT, CENS, EXACT, CENS, EXACT],
        [0.9, 0.5, 0.7, 0.7, 0.8],
    )
    assert_close(value, 0.75)
    _assert_counts(info, 4, 1, 1, 6)


@pytest.mark.step("A10")
def test_counts_partition_comparable_pairs():
    """spec 9.2, rev 34: each comparable pair is concordant, discordant or tied: `comparable = concordant + discordant + tied_score`.

    Seven rows with ties in time and score: ``time 1,1,2,2,2,3,5``,
    ``event E,C,E,E,C,C,E``, ``score 3,3,2,1,2,0,1``. Pairs (``first`` is
    the earlier row): from row 0 (E at 1): ``(0,1)`` tie event/censor, scores
    ``3=3`` tied; ``(0,2)``, ``(0,3)``, ``(0,4)``, ``(0,5)``, ``(0,6)``
    concordant (3 > 2, 1, 2, 0, 1). Row 1 (C at 1) is earlier than all later
    rows: none. Row 2 (E at 2): ``(2,3)`` E/E equal time, excluded; ``(2,4)``
    E/C equal time, ``2=2`` tied; ``(2,5)`` ``2>0`` concordant; ``(2,6)``
    ``2>1`` concordant. Row 3 (E at 2): ``(3,4)`` E/C equal time, ``1<2``
    discordant; ``(3,5)`` ``1>0`` concordant; ``(3,6)`` ``1=1`` tied.
    Rows 4, 5 (censored) are earlier than row 6: none. Concordant 8,
    discordant 1, tied 3, comparable 12; ``C = (8 + 1.5)/12 = 0.791666...``.
    """
    value, info = _with_info(
        [1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 5.0],
        [EXACT, CENS, EXACT, EXACT, CENS, CENS, EXACT],
        [3.0, 3.0, 2.0, 1.0, 2.0, 0.0, 1.0],
    )
    _assert_counts(info, 8, 1, 3, 12)
    assert_close(value, 9.5 / 12.0)


# ------------------------------------------------------------- pair rules


@pytest.mark.step("A10")
def test_earlier_exact_later_censored_is_comparable():
    """spec 9.2, rev 34: "Unequal times are comparable when the earlier one is exact"."""
    value, info = _with_info([1.0, 2.0], [EXACT, CENS], [2.0, 1.0])
    assert_close(value, 1.0)
    _assert_counts(info, 1, 0, 0, 1)


@pytest.mark.step("A10")
def test_earlier_censored_is_not_comparable():
    """spec 9.2, rev 34: a pair whose earlier time is censored is not comparable; alone it gives NaN."""
    _assert_no_comparable_pairs([1.0, 2.0], [CENS, EXACT], [2.0, 1.0])


@pytest.mark.step("A10")
def test_equal_time_event_pair_is_excluded():
    """spec 9.2, rev 34: "Equal-time event/event ... pairs are excluded"."""
    _assert_no_comparable_pairs([2.0, 2.0], [EXACT, EXACT], [3.0, 1.0])


@pytest.mark.step("A10")
def test_equal_time_censor_pair_is_excluded():
    """spec 9.2, rev 34: "Equal-time ... censor/censor pairs are excluded"."""
    _assert_no_comparable_pairs([2.0, 2.0], [CENS, CENS], [3.0, 1.0])


@pytest.mark.step("A10")
def test_equal_time_event_censor_pair_with_larger_event_score_is_concordant():
    """spec 9.1 and 9.2, rev 34: "an equal-time event/censor pair is comparable, with the event earlier"."""
    value, info = _with_info([2.0, 2.0], [EXACT, CENS], [3.0, 1.0])
    assert_close(value, 1.0)
    _assert_counts(info, 1, 0, 0, 1)


@pytest.mark.step("A10")
def test_equal_time_event_censor_pair_with_smaller_event_score_is_discordant():
    """spec 9.1 and 9.2, rev 34: the event is earlier, so a larger score on the censored row is discordant."""
    value, info = _with_info([2.0, 2.0], [EXACT, CENS], [1.0, 3.0])
    assert_close(value, 0.0)
    _assert_counts(info, 0, 1, 0, 1)


@pytest.mark.step("A10")
def test_equal_time_event_censor_pair_does_not_depend_on_row_order():
    """spec 9.2, rev 34: the censored row listed first changes nothing; the event is still earlier."""
    value, info = _with_info([2.0, 2.0], [CENS, EXACT], [1.0, 3.0])
    assert_close(value, 1.0)
    _assert_counts(info, 1, 0, 0, 1)


@pytest.mark.step("A10")
def test_equal_time_event_censor_pair_with_tied_scores_gets_half_credit():
    """spec 9.2, rev 34: tied scores on a comparable equal-time pair give half credit."""
    value, info = _with_info([2.0, 2.0], [EXACT, CENS], [1.0, 1.0])
    assert_close(value, 0.5)
    _assert_counts(info, 0, 0, 1, 1)


@pytest.mark.step("A10")
def test_each_unordered_pair_is_counted_once():
    """spec 9.2, rev 34: "Each unordered pair is counted once": three events give 3 comparable pairs, not 6."""
    _, info = _with_info([1.0, 2.0, 3.0], [EXACT] * 3, [3.0, 2.0, 1.0])
    assert count(info, "comparable") == 3


# ------------------------------------------------------- no comparable pairs


@pytest.mark.step("A10")
def test_all_censored_gives_nan():
    """spec 9.2, rev 34: "No comparable pairs gives NaN"; all-censored data have none."""
    _assert_no_comparable_pairs([1.0, 2.0, 3.0], [CENS] * 3, [1.0, 2.0, 3.0])


@pytest.mark.step("A10")
def test_single_observation_gives_nan():
    """spec 9.2, rev 34: a singleton cohort has no pairs, so no comparable pairs: NaN."""
    _assert_no_comparable_pairs([1.0], [EXACT], [1.0])


@pytest.mark.step("A10")
def test_events_all_at_one_time_give_nan():
    """spec 9.2, rev 34: only equal-time event/event pairs: none comparable, NaN."""
    _assert_no_comparable_pairs([2.0, 2.0, 2.0], [EXACT] * 3, [1.0, 2.0, 3.0])


@pytest.mark.step("A10")
def test_no_comparable_pair_mask_is_false_when_pairs_exist():
    """spec 9.2, rev 34: the no-comparable-pair mask marks only results without comparable pairs."""
    _, info = _with_info([1.0, 2.0], [EXACT, EXACT], [2.0, 1.0])
    assert not result_mask(info, "no_comparable_pairs", ()).any()


# ------------------------------------------------------------ invariances


@pytest.mark.step("A10")
def test_row_permutation_leaves_value_and_counts_unchanged():
    """spec 9.2, rev 34: pairs are unordered, so permuting rows (with their scores) changes nothing."""
    time = np.array([1.0, 2.0, 3.0, 3.0, 4.0])
    event = np.array([EXACT, CENS, EXACT, CENS, EXACT])
    scores = np.array([0.9, 0.5, 0.7, 0.7, 0.8])
    perm = np.array([3, 0, 4, 1, 2])
    value, info = _with_info(time[perm], event[perm], scores[perm])
    assert_close(value, 0.75)
    _assert_counts(info, 4, 1, 1, 6)


@pytest.mark.step("A10")
def test_negated_scores_swap_concordant_and_discordant():
    """spec 9.2, rev 34: negating scores swaps concordant and discordant and keeps ties, so `C(-s) = 1 - C(s)`."""
    value, info = _with_info(
        [1.0, 2.0, 3.0, 3.0, 4.0],
        [EXACT, CENS, EXACT, CENS, EXACT],
        [-0.9, -0.5, -0.7, -0.7, -0.8],
    )
    assert_close(value, 0.25)
    _assert_counts(info, 1, 4, 1, 6)


@pytest.mark.step("A10")
def test_increasing_transformation_of_scores_leaves_value_unchanged():
    """spec 9.2, rev 34: only score order matters: `exp(3 s)` gives the same counts as `s`."""
    scores = np.array([0.9, 0.5, 0.7, 0.7, 0.8])
    value, info = _with_info(
        [1.0, 2.0, 3.0, 3.0, 4.0], [EXACT, CENS, EXACT, CENS, EXACT], np.exp(3.0 * scores)
    )
    assert_close(value, 0.75)
    _assert_counts(info, 4, 1, 1, 6)


@pytest.mark.step("A10")
def test_uninformative_row_adds_no_pair_and_no_issue():
    """spec 2, 3.3 and 9.2, rev 34: a row right-censored at `0` is valid and is never earlier-and-exact, so it adds no pair."""
    value, info = _with_info(
        [0.0, 1.0, 2.0, 3.0, 3.0, 4.0],
        [CENS, EXACT, CENS, EXACT, CENS, EXACT],
        [5.0, 0.9, 0.5, 0.7, 0.7, 0.8],
    )
    assert_close(value, 0.75)
    _assert_counts(info, 4, 1, 1, 6)
    assert not result_mask(info, "invalid", ()).any()


@pytest.mark.step("A10")
def test_bounds_form_data_give_the_flag_form_value():
    """spec 3.1 and 9.2, rev 34: exact and right-censored rows built with `from_bounds` give the same counts."""
    data = survjax.SurvivalData.from_bounds(
        f([1.0, 2.0, 3.0, 3.0, 4.0]),
        f([1.0, np.inf, 3.0, np.inf, 4.0]),
        exact=b([EXACT, CENS, EXACT, CENS, EXACT]),
    )
    value, info = survjax.concordance_index_with_info(f([0.9, 0.5, 0.7, 0.7, 0.8]), data)
    assert_close(np.asarray(value), 0.75)
    _assert_counts(info, 4, 1, 1, 6)


@pytest.mark.step("A10")
def test_value_under_jit_matches_eager():
    """spec 6.1 and 12.1, rev 34: eager and `jit` agree on the value and the counts."""
    data = survjax.SurvivalData(f([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 5.0]), b([EXACT, CENS, EXACT, EXACT, CENS, CENS, EXACT]))
    scores = f([3.0, 3.0, 2.0, 1.0, 2.0, 0.0, 1.0])
    value, info = jax.jit(survjax.concordance_index_with_info)(scores, data)
    assert_close(np.asarray(value), 9.5 / 12.0)
    _assert_counts(info, 8, 1, 3, 12)


@pytest.mark.step("A10")
def test_traced_construction_inside_jit():
    """spec 6.2 and 9.1, rev 34: data constructed inside `jit` from traced inputs give the eager value."""

    def run(scores, time, event):
        return survjax.concordance_index(scores, survjax.SurvivalData(time, event))

    value = jax.jit(run)(f([0.9, 0.5, 0.7, 0.7, 0.8]), f([1.0, 2.0, 3.0, 3.0, 4.0]), b([EXACT, CENS, EXACT, CENS, EXACT]))
    assert_close(np.asarray(value), 0.75)
