"""A10: invalid, unsupported and undefined concordance results (spec 6.1, 6.2, 9.1, 9.2; rev 34).

9.2: "Scores are finite, caller-supplied". 6.1: "invalid runtime values give
NaN plus issues; static misuse ... raises", identically eagerly and under
`jit`. 9.1: "Supported observations are exact and right-censored rows ...
Other rows give NaN plus an `unsupported` diagnostic; rows are never dropped
or replaced with representative times." Since rows are never dropped, the
whole concordance value that contains such a row is NaN.

Base case: the five-row reference case of ``test_a10_concordance_api.py``
(``C = 0.75``); row 2 is replaced in each case below. Mask shapes: Q-045.
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, b, f, result_mask

import survjax

TIME = [1.0, 2.0, 3.0, 3.0, 4.0]
EVENT = [True, False, True, False, True]
SCORES = [0.9, 0.5, 0.7, 0.7, 0.8]


def _data():
    return survjax.SurvivalData(f(TIME), b(EVENT))


def _scores_with(row, value):
    scores = list(SCORES)
    scores[row] = value
    return f(scores)


def _check(value, info, *, invalid, unsupported, shape=()):
    value = np.asarray(value)
    assert value.shape == shape
    bad = np.asarray(invalid, dtype=bool) | np.asarray(unsupported, dtype=bool)
    np.testing.assert_array_equal(np.isnan(value), np.broadcast_to(bad, shape))
    np.testing.assert_array_equal(result_mask(info, "invalid", shape), np.broadcast_to(invalid, shape))
    np.testing.assert_array_equal(
        result_mask(info, "unsupported", shape), np.broadcast_to(unsupported, shape)
    )


# ------------------------------------------------------------ invalid scores


@pytest.mark.step("A10")
@pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "inf", "-inf"])
def test_non_finite_score_gives_nan_and_invalid(bad):
    """spec 9.2 and 6.1, rev 34: "Scores are finite"; a non-finite score is an invalid runtime value: NaN plus the invalid mask."""
    value, info = survjax.concordance_index_with_info(_scores_with(2, bad), _data())
    _check(value, info, invalid=True, unsupported=False)


@pytest.mark.step("A10")
def test_non_finite_score_does_not_raise_under_jit():
    """spec 6.1, rev 34: invalid runtime values give NaN plus issues identically under `jit`."""
    value, info = jax.jit(survjax.concordance_index_with_info)(_scores_with(2, NAN), _data())
    _check(value, info, invalid=True, unsupported=False)


@pytest.mark.step("A10")
def test_non_finite_score_affects_only_its_draw():
    """spec 9.1, rev 34: leading axes "are never pooled": a NaN in draw 1 of `(2,N)` scores leaves draw 0 at `0.75`."""
    scores = np.stack([f(SCORES), _scores_with(2, NAN)])
    value, info = survjax.concordance_index_with_info(scores, _data())
    _check(value, info, invalid=[False, True], unsupported=False, shape=(2,))
    assert_close(np.asarray(value)[0], 0.75)


# ------------------------------------------------------- stored data issues


@pytest.mark.step("A10")
def test_invalid_event_code_gives_nan_and_invalid():
    """spec 6.1, 6.2 and 9.1, rev 34: data with an invalid indicator (code 2) give NaN plus the invalid mask, without raising."""
    event = np.asarray([1, 0, 2, 0, 1], dtype=np.int32)
    data = survjax.SurvivalData(f(TIME), event)
    value, info = survjax.concordance_index_with_info(f(SCORES), data)
    _check(value, info, invalid=True, unsupported=False)


@pytest.mark.step("A10")
@pytest.mark.parametrize("bad", [-1.0, 0.0, INF, NAN], ids=["negative", "zero", "inf", "nan"])
def test_invalid_exact_time_gives_nan_and_invalid(bad):
    """spec 3.3 and 6.2, rev 34: an exact time that is not finite and positive is an invalid row (`invalid_time`)."""
    time = list(TIME)
    time[2] = bad
    value, info = survjax.concordance_index_with_info(
        f(SCORES), survjax.SurvivalData(f(time), b(EVENT))
    )
    _check(value, info, invalid=True, unsupported=False)


@pytest.mark.step("A10")
def test_invalid_data_under_jit_with_traced_construction():
    """spec 6.1 and 6.2, rev 34: traced construction inside `jit` records the issue; the result is NaN with the invalid mask."""

    def run(scores, time, event):
        return survjax.concordance_index_with_info(scores, survjax.SurvivalData(time, event))

    event = np.asarray([1, 0, 2, 0, 1], dtype=np.int32)
    value, info = jax.jit(run)(f(SCORES), f(TIME), event)
    _check(value, info, invalid=True, unsupported=False)


@pytest.mark.step("A10")
def test_invalid_indexed_data_keeps_its_issue():
    """spec 6.2 and 9.1, rev 34: stored issues survive indexing; the indexed subset with the invalid row is NaN, without it valid."""
    event = np.asarray([1, 0, 2, 0, 1, 1], dtype=np.int32)
    data = survjax.SurvivalData(f([*TIME, 5.0]), event)
    with_bad = data[np.array([0, 1, 2, 3, 4])]
    value, info = survjax.concordance_index_with_info(f(SCORES), with_bad)
    _check(value, info, invalid=True, unsupported=False)
    without_bad = data[np.array([0, 1, 3, 4, 5])]
    # times 1,2,3,4,5; events E,C,C,E,E; scores 0.9,0.5,0.7,0.8,0.1:
    # (0,1..4) concordant 4; (3,4) 0.8>0.1 concordant; C = 1.
    value = survjax.concordance_index(f([0.9, 0.5, 0.7, 0.8, 0.1]), without_bad)
    assert_close(np.asarray(value), 1.0)


@pytest.mark.step("A10")
def test_invalid_data_in_one_dataset_leaves_the_other_dataset():
    """spec 9.1, rev 34: `(B,N)` data are never pooled: an invalid row in dataset 1 leaves dataset 0 at `0.75`."""
    event = np.asarray([[1, 0, 1, 0, 1], [1, 0, 2, 0, 1]], dtype=np.int32)
    data = survjax.SurvivalData(f([TIME, TIME]), event)
    value, info = survjax.concordance_index_with_info(f([SCORES, SCORES]), data)
    _check(value, info, invalid=[False, True], unsupported=False, shape=(2,))
    assert_close(np.asarray(value)[0], 0.75)


# ------------------------------------------------------------ unsupported rows


def _left_censored():
    return survjax.SurvivalData(f(TIME), left_censored=b([False, False, True, False, False]))


def _interval_censored():
    return survjax.SurvivalData(
        f(TIME),
        interval_censored=b([False, False, True, False, False]),
        time2=f([1.0, 2.0, 3.5, 3.0, 4.0]),
    )


def _left_truncated():
    return survjax.SurvivalData(f(TIME), b(EVENT), left_trunc=f([0.0, 0.0, 0.5, 0.0, 0.0]))


def _right_truncated():
    return survjax.SurvivalData(f(TIME), b(EVENT), right_trunc=f([INF, INF, 10.0, INF, INF]))


def _bounds_interval():
    return survjax.SurvivalData.from_bounds(
        f([1.0, 2.0, 2.5, 3.0, 4.0]),
        f([1.0, INF, 3.0, INF, 4.0]),
        exact=b([True, False, False, False, True]),
    )


UNSUPPORTED = {
    "left-censored": _left_censored,
    "interval-censored": _interval_censored,
    "left-truncated": _left_truncated,
    "right-truncated": _right_truncated,
    "bounds-interval": _bounds_interval,
}


@pytest.mark.step("A10")
@pytest.mark.parametrize("case", sorted(UNSUPPORTED))
def test_unsupported_row_is_valid_data(case):
    """spec 3.2 and 9.1, rev 34: the rows are valid observations; only their mode is unsupported."""
    assert bool(UNSUPPORTED[case]().issues.ok)


@pytest.mark.step("A10")
@pytest.mark.parametrize("case", sorted(UNSUPPORTED))
def test_unsupported_row_gives_nan_and_unsupported(case):
    """spec 9.1, rev 34: "Other rows give NaN plus an `unsupported` diagnostic"; left/interval censoring and every truncation."""
    value, info = survjax.concordance_index_with_info(f(SCORES), UNSUPPORTED[case]())
    _check(value, info, invalid=False, unsupported=True)


@pytest.mark.step("A10")
@pytest.mark.parametrize("case", sorted(UNSUPPORTED))
def test_unsupported_row_gives_nan_under_jit(case):
    """spec 6.1 and 9.1, rev 34: the same under `jit`."""
    value, info = jax.jit(survjax.concordance_index_with_info)(f(SCORES), UNSUPPORTED[case]())
    _check(value, info, invalid=False, unsupported=True)


@pytest.mark.step("A10")
def test_unsupported_row_from_traced_construction_gives_nan():
    """spec 3.2 and 9.1, rev 62: traced construction (`left_censored` supplied: all four kind flags true, before and after Slice B) still finds the unsupported row by value."""

    def run(scores, time, flags):
        return survjax.concordance_index_with_info(
            scores, survjax.SurvivalData(time, left_censored=flags)
        )

    value, info = jax.jit(run)(f(SCORES), f(TIME), b([False, False, True, False, False]))
    _check(value, info, invalid=False, unsupported=True)


@pytest.mark.step("A10")
def test_traced_construction_of_supported_rows_is_not_unsupported():
    """spec 3.2 and 9.1, rev 62: traced metadata (all true before Slice B; exact/right untruncated from the `event` form after) do not make supported rows unsupported."""

    def run(scores, time, event):
        return survjax.concordance_index_with_info(scores, survjax.SurvivalData(time, event))

    value, info = jax.jit(run)(f(SCORES), f(TIME), b(EVENT))
    _check(value, info, invalid=False, unsupported=False)
    assert_close(np.asarray(value), 0.75)


@pytest.mark.step("A10")
def test_unsupported_row_in_one_dataset_leaves_the_other_dataset():
    """spec 9.1, rev 34: `(B,N)` data are never pooled: a left-truncated row in dataset 1 leaves dataset 0 at `0.75`."""
    data = survjax.SurvivalData(
        f([TIME, TIME]), b([EVENT, EVENT]), left_trunc=f([[0.0] * 5, [0.0, 0.0, 0.5, 0.0, 0.0]])
    )
    value, info = survjax.concordance_index_with_info(f([SCORES, SCORES]), data)
    _check(value, info, invalid=False, unsupported=[False, True], shape=(2,))
    assert_close(np.asarray(value)[0], 0.75)


@pytest.mark.step("A10")
def test_explicit_zero_left_and_infinite_right_truncation_are_supported():
    """spec 3.2 and 9.1, rev 34: `left_trunc=0` and `right_trunc=+inf` are no truncation, so the rows stay supported."""
    data = survjax.SurvivalData(
        f(TIME), b(EVENT), left_trunc=f([0.0] * 5), right_trunc=f([INF] * 5)
    )
    value, info = survjax.concordance_index_with_info(f(SCORES), data)
    _check(value, info, invalid=False, unsupported=False)
    assert_close(np.asarray(value), 0.75)
