"""Public C-index inputs, cohort axes and failures (spec 6, 9.1–9.2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import survjax

TIME = jnp.array([1.0, 2.0, 3.0, 4.0])
EVENT = jnp.array([True, False, True, True])
SCORE = jnp.array([3.0, 1.0, 3.0, 4.0])


def test_public_forms_return_hand_counted_value_and_counts():
    """Spec 6.1/9.2: C,T,D,D pairs give (1+1/2)/4=3/8 in both forms."""
    data = survjax.SurvivalData(TIME, EVENT)
    value, info = survjax.concordance_index_with_info(SCORE, time=TIME, event=EVENT)
    np.testing.assert_array_equal(value, 0.375)
    np.testing.assert_array_equal(survjax.concordance_index(SCORE, data), 0.375)
    for name, expected in zip(
        ("concordant", "discordant", "tied_score", "comparable"),
        (1, 2, 1, 4),
        strict=True,
    ):
        count = getattr(info, name)
        assert int(count.hi) == 0
        assert int(count.lo) == expected


@pytest.mark.parametrize(
    "data_batch,score_batch,expected",
    [
        ((), (), 0.375),
        ((), (2,), [0.375, 0.625]),
        ((2,), (2,), [0.375, 0.625]),
        ((2, 1), (2,), [[0.375, 0.625], [0.375, 0.625]]),
        ((), (2, 2), [[0.375, 0.625], [0.375, 0.625]]),
    ],
    ids=["paired", "draws", "datasets", "datasets-by-draws", "multiple-score-axes"],
)
def test_shape_table_and_every_diagnostic_leaf(data_batch, score_batch, expected):
    """Spec 9.2/N-7: all five shape rows have per-result counts and masks."""
    data = survjax.SurvivalData(
        jnp.broadcast_to(TIME, (*data_batch, 4)),
        jnp.broadcast_to(EVENT, (*data_batch, 4)),
    )
    scores = (
        jnp.broadcast_to(jnp.stack([SCORE, -SCORE]), (*score_batch, 4))
        if score_batch
        else SCORE
    )
    # Reversing ranks complements C, including the half-credit pair: 1-3/8=5/8.
    for evaluate in (
        survjax.concordance_index_with_info,
        jax.jit(survjax.concordance_index_with_info),
    ):
        value, info = evaluate(scores, data)
        np.testing.assert_array_equal(value, expected)
        for leaf in jax.tree.leaves(info):
            assert leaf.shape == value.shape


@pytest.mark.parametrize("failure", ["raw-code", "score", "left-trunc", "both"])
def test_bad_cohort_isolated_from_valid_neighbor(failure):
    """Spec 6.2/9.1: invalid/support evidence is reduced per cohort, not pooled."""
    scores = jnp.stack([SCORE, SCORE])
    events = jnp.stack([EVENT, EVENT]).astype(jnp.int32)
    if failure == "raw-code":
        events = events.at[1, 1].set(2)
    if failure in ("score", "both"):
        scores = scores.at[1, 0].set(jnp.nan)
    left = jnp.array([0.0, 0.5 if failure in ("left-trunc", "both") else 0.0])[:, None]
    data = survjax.SurvivalData(jnp.stack([TIME, TIME]), events, left_trunc=left)
    value, info = jax.jit(survjax.concordance_index_with_info)(scores, data)
    np.testing.assert_array_equal(np.isnan(value), [False, True])
    np.testing.assert_array_equal(info.invalid, [False, failure != "left-trunc"])
    np.testing.assert_array_equal(
        info.unsupported, [False, failure in ("left-trunc", "both")]
    )
    np.testing.assert_array_equal(info.no_comparable_pairs, [False, False])
    np.testing.assert_array_equal(info.comparable.lo, [4, 0])
    np.testing.assert_array_equal(value[0], 0.375)


@pytest.mark.parametrize("mode", ["right-trunc", "interval", "left-censored"])
def test_unsupported_bounds_cohort_isolated_from_valid_neighbor(mode):
    """Spec 9.1: valid unsupported observations fail their entire cohort."""
    lower = jnp.stack([TIME, TIME])
    upper = jnp.broadcast_to(jnp.where(EVENT, TIME, jnp.inf), lower.shape)
    right = jnp.array([jnp.inf, 10.0 if mode == "right-trunc" else jnp.inf])[:, None]
    if mode in ("interval", "left-censored"):
        upper = upper.at[1, 1].set(2.5 if mode == "interval" else 2.0)
    if mode == "left-censored":
        lower = lower.at[1, 1].set(0.0)
    data = survjax.SurvivalData.from_bounds(
        lower, upper, exact=EVENT, right_trunc=right
    )
    assert bool(data.issues.ok)
    for evaluate in (
        survjax.concordance_index_with_info,
        jax.jit(survjax.concordance_index_with_info),
    ):
        value, info = evaluate(SCORE, data)
        np.testing.assert_array_equal(np.isnan(value), [False, True])
        np.testing.assert_array_equal(info.invalid, [False, False])
        np.testing.assert_array_equal(info.unsupported, [False, True])
        np.testing.assert_array_equal(info.no_comparable_pairs, [False, False])
        # The valid neighbor has C,T,D,D pairs: (1+1/2)/4 = 3/8.
        np.testing.assert_array_equal(value[0], 0.375)
        for name, expected in zip(
            ("concordant", "discordant", "tied_score", "comparable"),
            (1, 2, 1, 4),
            strict=True,
        ):
            count = getattr(info, name)
            np.testing.assert_array_equal(count.hi, [0, 0])
            np.testing.assert_array_equal(count.lo, [expected, 0])


def test_edited_fields_are_rechecked_alongside_stored_issues():
    """Spec 6.2: pytree reconstruction can bypass constructor field validation."""
    data = survjax.SurvivalData(TIME, EVENT)
    leaves, structure = jax.tree.flatten(data)
    leaves[1] = leaves[1].at[0].set(-1.0)
    edited = jax.tree.unflatten(structure, leaves)
    assert bool(edited.issues.ok)
    value, info = survjax.concordance_index_with_info(SCORE, edited)
    assert np.isnan(value)
    assert bool(info.invalid)


def test_empty_cohort_has_no_comparable_pairs():
    """Accepted D-027: empty input is an evaluated undefined cohort."""
    value, info = survjax.concordance_index_with_info(
        SCORE[:0], time=TIME[:0], event=EVENT[:0]
    )
    assert np.isnan(value)
    assert bool(info.no_comparable_pairs)
    assert not bool(info.invalid | info.unsupported)


@pytest.mark.parametrize(
    "data_shape,score_shape",
    [
        ((4,), ()),
        ((), (4,)),
        ((4,), (1,)),
        ((4,), (3,)),
        ((4,), (4, 1)),
        ((2, 4), (3, 4)),
    ],
)
def test_axis_misuse_raises_type_error(data_shape, score_shape):
    """Spec 9.1: observation axes never broadcast; leading axes must broadcast."""
    data = survjax.SurvivalData(jnp.ones(data_shape))
    with pytest.raises(TypeError):
        survjax.concordance_index(jnp.ones(score_shape), data)


def test_score_precision_is_preserved_with_float32_data(x64_enabled):
    """Spec 9.2: distinct ranks must not become score ties via a data-dtype cast."""
    data = survjax.SurvivalData(jnp.array([1.0, 2.0], dtype=jnp.float32))
    dtype = jnp.float64 if x64_enabled else jnp.float32
    one = jnp.asarray(1.0, dtype=dtype)
    scores = jnp.stack([jnp.nextafter(one, jnp.asarray(2.0, dtype=dtype)), one])
    value = survjax.concordance_index(scores, data)
    assert value.dtype == jnp.dtype(jnp.float32)
    np.testing.assert_array_equal(value, 1.0)
    np.testing.assert_array_equal(
        survjax.concordance_index(jnp.array([2**25 + 1, 2**25], dtype=jnp.int32), data),
        1.0,
    )


def test_jit_of_vmap_returns_values_and_result_shaped_info():
    """Spec 6.3/12.1: outer score batching composes with the cohort evaluator."""
    data = survjax.SurvivalData(TIME, EVENT)
    value, info = jax.jit(
        jax.vmap(lambda s: survjax.concordance_index_with_info(s, data))
    )(jnp.stack([SCORE, -SCORE]))
    np.testing.assert_array_equal(value, [0.375, 0.625])
    for leaf in jax.tree.leaves(info):
        assert leaf.shape == (2,)


@pytest.mark.parametrize("scores", [None, ["x"] * 4, [1j] * 4])
def test_scores_must_be_real_numeric(scores):
    """Spec 6.2: nonnumeric and nonreal score kinds are static misuse."""
    with pytest.raises(TypeError):
        survjax.concordance_index(scores, survjax.SurvivalData(TIME, EVENT))
