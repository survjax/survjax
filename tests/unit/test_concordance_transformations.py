"""Public concordance composition with independently counted cohorts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


def _assert_result(result, values, counts, invalid, unsupported, no_pairs, x64):
    value, info = result
    expected = np.asarray(values)
    assert value.shape == expected.shape
    np.testing.assert_allclose(value, expected, atol=ATOL[x64], rtol=RTOL[x64])
    for name, total in zip(
        ("concordant", "discordant", "tied_score", "comparable"), counts, strict=True
    ):
        words = getattr(info, name)
        for word in (words.hi, words.lo):
            assert word.shape == expected.shape
            assert word.dtype == jnp.dtype(jnp.uint32)
        np.testing.assert_array_equal(words.hi, np.zeros(expected.shape, np.uint32))
        np.testing.assert_array_equal(words.lo, total)
    for name, mask in zip(
        ("invalid", "unsupported", "no_comparable_pairs"),
        (invalid, unsupported, no_pairs),
        strict=True,
    ):
        actual = getattr(info, name)
        assert actual.shape == expected.shape
        assert actual.dtype == jnp.dtype(bool)
        np.testing.assert_array_equal(actual, mask)


@pytest.mark.parametrize("prepared", [False, True], ids=["raw", "prepared"])
def test_crossed_cohorts_and_draws_under_nested_transformations(prepared, x64_enabled):
    """Spec 6.1/9.1–9.2/12.1: B=2 and D=3 remain separate through composition."""
    time = jnp.array([[1.0, 2.0, 2.0, 3.0], [3.0, 1.0, 2.0, 2.0]])
    event = jnp.array([[True, True, False, True], [True, True, True, False]])
    scores = jnp.array([[3.0, 3.0, 1.0, 4.0], [4.0, 3.0, 1.0, 2.0], [2.0] * 4])
    # A pairs (0,1),(0,2),(0,3),(1,2),(1,3): T,C,D,C,D for draw 0;
    # all C for draw 1. B pairs (1,0),(1,2),(1,3),(2,0),(2,3): T,C,D,D,D
    # for draw 0, D,C,C,D,D for draw 1. Draw 2 ties all five pairs in both.
    counts = (
        [[2, 5, 0], [1, 2, 0]],
        [[2, 0, 0], [3, 3, 0]],
        [[1, 0, 5], [1, 0, 5]],
        [[5, 5, 5], [5, 5, 5]],
    )
    if prepared:
        run = survjax.concordance_index_with_info
        args = (scores, survjax.SurvivalData(time[:, None], event[:, None]))
        mapped_args = (scores, survjax.SurvivalData(time, event))
        mapped = jax.vmap(jax.vmap(run, in_axes=(0, None)), in_axes=(None, 0))
    else:

        def run(s, t, e):
            return survjax.concordance_index_with_info(s, time=t, event=e)

        args = (scores, time[:, None], event[:, None])
        mapped_args = (scores, time, event)
        mapped = jax.vmap(jax.vmap(run, in_axes=(0, None, None)), in_axes=(None, 0, 0))
    for evaluate, inputs in (
        (run, args),
        (jax.jit(run), args),
        (mapped, mapped_args),
        (jax.jit(mapped), mapped_args),
    ):
        _assert_result(
            evaluate(*inputs),
            [[0.5, 1.0, 0.5], [0.3, 0.4, 0.5]],
            counts,
            np.zeros((2, 3), bool),
            np.zeros((2, 3), bool),
            np.zeros((2, 3), bool),
            x64_enabled,
        )


def test_transformed_failures_preserve_whole_cohort_masks(x64_enabled):
    """Spec 6.2–6.3/9.1–9.2/12.1: failed cohorts and evaluated zero pairs differ."""
    time = jnp.broadcast_to(jnp.array([1.0, 2.0, 3.0]), (5, 3))
    # Rows: valid; invalid raw event; unsupported; both failures; no pairs.
    event = jnp.array([[1, 1, 1], [2, 1, 1], [1, 1, 1], [2, 1, 1], [0, 0, 0]])
    left = jnp.array([0.0, 0.0, 0.5, 0.5, 0.0])[:, None]
    scores = jnp.array([[3.0, 2.0, 1.0], [1.0, 2.0, 3.0]])

    def run(s, t, e, left_trunc):
        data = survjax.SurvivalData(t, e, left_trunc=left_trunc)
        return survjax.concordance_index_with_info(s, data)

    mapped = jax.vmap(
        jax.vmap(run, in_axes=(0, None, None, None)), in_axes=(None, 0, 0, 0)
    )
    zeros = [[0, 0]] * 4
    for evaluate, inputs in (
        (run, (scores, time[:, None], event[:, None], left[:, None])),
        (jax.jit(run), (scores, time[:, None], event[:, None], left[:, None])),
        (mapped, (scores, time, event, left)),
        (jax.jit(mapped), (scores, time, event, left)),
    ):
        # Three unequal-time exact pairs, all C for draw 0 and D for draw 1.
        _assert_result(
            evaluate(*inputs),
            [[1.0, 0.0]] + [[np.nan, np.nan]] * 4,
            ([[3, 0]] + zeros, [[0, 3]] + zeros, [[0, 0]] * 5, [[3, 3]] + zeros),
            [[False] * 2, [True] * 2, [False] * 2, [True] * 2, [False] * 2],
            [[False] * 2, [False] * 2, [True] * 2, [True] * 2, [False] * 2],
            [[False] * 2] * 4 + [[True] * 2],
            x64_enabled,
        )


def test_dynamic_scores_times_and_events_reuse_raw_compilation(x64_enabled):
    """Spec 6.1/7.2: raw construction composes with a fixed-shape compiled call."""
    traces = []

    @jax.jit
    def run(scores, time, event):
        traces.append(None)
        return survjax.concordance_index_with_info(scores, time=time, event=event)

    # First: three exact pairs all C. Second: only the event at time 1 is
    # earlier, giving (1,0) C and (1,2) D, hence 1/2 with two comparable pairs.
    cases = (
        ([3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [True] * 3, 1.0, (3, 0, 0, 3)),
        ([1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [False, True, False], 0.5, (1, 1, 0, 2)),
    )
    for scores, time, event, value, counts in cases:
        result = run(jnp.array(scores), jnp.array(time), jnp.array(event))
        _assert_result(result, value, counts, False, False, False, x64_enabled)
    assert len(traces) == 1


def test_prepared_metadata_reuses_compilation_when_values_change(x64_enabled):
    """Spec 3.2/7.2/12.1: equal metadata flags reuse a prepared-data trace."""
    traces = []

    @jax.jit
    def run(scores, data):
        traces.append(None)
        return survjax.concordance_index_with_info(scores, data)

    # Both cohorts contain exact and right-censored rows, so their static
    # flags agree despite different times/events/scores. First pairs (0,1),
    # (0,2),(1,2) are all C; second pairs (1,0),(1,2) are C,D, hence 1/2.
    cases = (
        ([3.0, 2.0, 1.0], [1.0, 2.0, 3.0], [True, True, False], 1.0, (3, 0, 0, 3)),
        ([1.0, 2.0, 3.0], [3.0, 1.0, 2.0], [False, True, False], 0.5, (1, 1, 0, 2)),
    )
    for scores, time, event, value, counts in cases:
        data = survjax.SurvivalData(jnp.array(time), jnp.array(event))
        result = run(jnp.array(scores), data)
        _assert_result(result, value, counts, False, False, False, x64_enabled)
    assert len(traces) == 1
    # Spec 3.2: removing right censoring changes the flags/tree structure,
    # so this call must compile separately rather than reuse the mixed trace.
    exact = survjax.SurvivalData(jnp.array([1.0, 2.0, 3.0]))
    result = run(jnp.array([3.0, 2.0, 1.0]), exact)
    _assert_result(result, 1.0, (3, 0, 0, 3), False, False, False, x64_enabled)
    assert len(traces) == 2
