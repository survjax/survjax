"""A10: Harrell's concordance, entry points, forms and result container (spec 6.1–6.3, 9.1, 9.2, 10.2; rev 34).

Step A10 is the maintainer's replan of 2026-10-23 (Harrell's C as the final
Slice A sub-slice; Q-047). The requirements are those of spec revision 34,
sections 9.1–9.2, unchanged by the replan.

Reference case (hand-enumerated, spec 9.2; ``i: time, event, score``):
``0: 1, exact, 0.9``; ``1: 2, censored, 0.5``; ``2: 3, exact, 0.7``;
``3: 3, censored, 0.7``; ``4: 4, exact, 0.8``.
Comparable pairs: ``(0,1)``, ``(0,2)``, ``(0,3)``, ``(0,4)`` (earlier exact,
larger score: concordant); ``(2,3)`` (equal time event/censor, event earlier;
equal scores: tied); ``(2,4)`` (``0.7 < 0.8``: discordant). Rows 1 and 3 are
censored before every later row, so ``(1,·)`` and ``(3,4)`` are not
comparable. Concordant 4, discordant 1, tied 1, comparable 6,
``C = (4 + 0.5)/6 = 0.75``.
"""

import jax
import numpy as np
import pytest
from helpers import (
    CONCORDANCE_COUNTS,
    CONCORDANCE_MASKS,
    FLOAT,
    assert_close,
    b,
    concordance_info_class,
    count,
    f,
    result_mask,
)

import survjax

TIME = [1.0, 2.0, 3.0, 3.0, 4.0]
EVENT = [True, False, True, False, True]
SCORES = [0.9, 0.5, 0.7, 0.7, 0.8]
C = 0.75
COUNTS = {"concordant": 4, "discordant": 1, "tied_score": 1, "comparable": 6}


def _data():
    return survjax.SurvivalData(f(TIME), b(EVENT))


def _ci(scores, data=None, **raw):
    return np.asarray(survjax.concordance_index(scores, data, **raw))


# --------------------------------------------------------------- public names


@pytest.mark.step("A10")
def test_concordance_entry_points_are_exported_at_top_level():
    """spec 10.2, rev 34: "Re-export ... KM and metric entry points" at top level; 9.2 names both."""
    assert callable(survjax.concordance_index)
    assert callable(survjax.concordance_index_with_info)


@pytest.mark.step("A10")
def test_metrics_module_holds_the_entry_points():
    """spec 10.2, rev 34: the layout places concordance in `survjax/metrics.py` (Q-042)."""
    import survjax.metrics as module

    assert module.concordance_index is survjax.concordance_index
    assert module.concordance_index_with_info is survjax.concordance_index_with_info


@pytest.mark.step("A10")
def test_concordance_info_is_discoverable_in_its_module():
    """spec 10.2, rev 34: "result and configuration types stay discoverable in their modules" (Q-042)."""
    assert isinstance(concordance_info_class(), type)


# ------------------------------------------------------------ result container


@pytest.mark.step("A10")
def test_with_info_returns_exactly_value_and_info():
    """spec 6.3, rev 34: "`_with_info` functions return exactly `(value, info)`"."""
    out = survjax.concordance_index_with_info(f(SCORES), _data())
    assert isinstance(out, tuple)
    assert len(out) == 2
    assert isinstance(out[1], concordance_info_class())


@pytest.mark.step("A10")
def test_with_info_value_equals_plain_value():
    """spec 9.2, rev 34: `concordance_index_with_info(...)` returns the `concordance_index` value with its info."""
    value, _ = survjax.concordance_index_with_info(f(SCORES), _data())
    np.testing.assert_array_equal(np.asarray(value), _ci(f(SCORES), _data()))
    assert_close(np.asarray(value), C)


@pytest.mark.step("A10")
@pytest.mark.parametrize("name", CONCORDANCE_COUNTS)
def test_info_holds_hand_enumerated_counts(name):
    """spec 9.2, rev 34: "`ConcordanceInfo` holds concordant, discordant, tied-score and comparable counts" (Q-043, Q-044)."""
    _, info = survjax.concordance_index_with_info(f(SCORES), _data())
    assert count(info, name) == COUNTS[name]


@pytest.mark.step("A10")
@pytest.mark.parametrize("name", CONCORDANCE_MASKS)
def test_info_masks_are_false_for_valid_comparable_data(name):
    """spec 9.2, rev 34: the info holds "invalid, unsupported and no-comparable-pair masks"; none is set here (Q-043, Q-045)."""
    _, info = survjax.concordance_index_with_info(f(SCORES), _data())
    assert not result_mask(info, name, ()).any()


@pytest.mark.step("A10")
def test_info_is_a_pytree_returned_from_jit():
    """spec 6.3, rev 34: every diagnostic container "is a registered PyTree, so it can be returned from `jit` and `vmap`"."""
    value, info = jax.jit(survjax.concordance_index_with_info)(f(SCORES), _data())
    assert isinstance(info, concordance_info_class())
    assert_close(np.asarray(value), C)
    for name in CONCORDANCE_COUNTS:
        assert count(info, name) == COUNTS[name], name


@pytest.mark.step("A10")
def test_info_is_returned_from_vmap():
    """spec 6.3, rev 34: the info is returned from `vmap` over score draws, one entry per draw."""
    scores = f([SCORES, [-s for s in SCORES]])
    value, info = jax.vmap(lambda s: survjax.concordance_index_with_info(s, _data()))(scores)
    assert isinstance(info, concordance_info_class())
    assert_close(np.asarray(value), [C, 1.0 - C])
    assert list(count(info, "concordant")) == [4, 1]
    assert list(count(info, "discordant")) == [1, 4]


@pytest.mark.step("A10")
def test_info_leaves_rebuild_an_equal_info():
    """spec 6.3, rev 34: flattening and unflattening the PyTree gives an info with the same fields."""
    _, info = survjax.concordance_index_with_info(f(SCORES), _data())
    leaves, treedef = jax.tree_util.tree_flatten(info)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    for name in CONCORDANCE_COUNTS:
        assert count(rebuilt, name) == COUNTS[name], name


# ------------------------------------------------------------ the two forms


@pytest.mark.step("A10")
def test_data_form_positional():
    """spec 6.1 and 9.2, rev 34: `concordance_index(scores, data)`, predictions first and data second."""
    assert_close(_ci(f(SCORES), _data()), C)


@pytest.mark.step("A10")
def test_data_form_by_keyword():
    """spec 9.2, rev 34: the signature names the second parameter `data`."""
    assert_close(np.asarray(survjax.concordance_index(f(SCORES), data=_data())), C)


@pytest.mark.step("A10")
def test_raw_form_equals_data_form():
    """spec 6.1, rev 34: the raw `time=`/`event=` form equals the data form."""
    raw = _ci(f(SCORES), time=f(TIME), event=b(EVENT))
    np.testing.assert_array_equal(raw, _ci(f(SCORES), _data()))


@pytest.mark.step("A10")
def test_raw_form_accepts_integer_event_codes():
    """spec 3.1 and 6.1, rev 34: raw `event` accepts numbers exactly 0 or 1."""
    event = np.asarray(EVENT, dtype=np.int32)
    assert_close(_ci(f(SCORES), time=f(TIME), event=event), C)


@pytest.mark.step("A10")
def test_raw_form_with_info():
    """spec 6.1, rev 34: the `_with_info` variant accepts the raw form too."""
    value, info = survjax.concordance_index_with_info(f(SCORES), time=f(TIME), event=b(EVENT))
    assert_close(np.asarray(value), C)
    assert count(info, "comparable") == 6


@pytest.mark.step("A10")
def test_raw_array_passed_as_data_raises_type_error():
    """spec 6.1, rev 34: "Passing a raw array positionally as `data` raises a `TypeError`"."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f(SCORES), f(TIME))


@pytest.mark.step("A10")
def test_mixing_forms_raises_type_error():
    """spec 6.1, rev 34: "Mixing the forms is rejected"."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f(SCORES), _data(), time=f(TIME), event=b(EVENT))


@pytest.mark.step("A10")
def test_raw_form_without_event_raises_type_error():
    """spec 6.1, rev 34: the raw form takes `time` and `event`, "both required"."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f(SCORES), time=f(TIME))


@pytest.mark.step("A10")
def test_raw_form_without_time_raises_type_error():
    """spec 6.1, rev 34: the raw form takes `time` and `event`, "both required"."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f(SCORES), event=b(EVENT))


@pytest.mark.step("A10")
def test_no_data_at_all_raises_type_error():
    """spec 6.1, rev 34: neither form supplied is static misuse."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f(SCORES))


@pytest.mark.step("A10")
def test_raw_form_rejects_constructor_keywords():
    """spec 6.1, rev 34: "The raw form takes only `time` and `event`"."""
    with pytest.raises(TypeError):
        survjax.concordance_index(
            f(SCORES), time=f(TIME), event=b(EVENT), left_trunc=f([0.0] * 5)
        )


@pytest.mark.step("A10")
def test_time_grid_keyword_is_not_accepted():
    """spec 6.1 and 9.2, rev 34: Harrell's C has no `eval_times`; an unknown keyword is a `TypeError`."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f(SCORES), _data(), eval_times=f([1.0, 2.0]))


@pytest.mark.step("A10")
def test_raw_form_invalid_event_code_returns_nan_eagerly():
    """spec 6.1, rev 34: "An eager raw-form call with, say, event code 2 returns NaN rather than raising"."""
    event = np.asarray([1, 0, 2, 0, 1], dtype=np.int32)
    value, info = survjax.concordance_index_with_info(f(SCORES), time=f(TIME), event=event)
    assert np.isnan(np.asarray(value))
    assert result_mask(info, "invalid", ()).all()


@pytest.mark.step("A10")
def test_raw_form_under_jit_matches_eager():
    """spec 6.1, rev 34: every entry point "behaves identically eagerly and under `jit`"."""

    def run(scores, time, event):
        return survjax.concordance_index(scores, time=time, event=event)

    assert_close(np.asarray(jax.jit(run)(f(SCORES), f(TIME), b(EVENT))), C)


@pytest.mark.step("A10")
def test_result_dtype_is_data_float_dtype():
    """spec 9.1, rev 34: "ratios are computed in the data's floating dtype"."""
    assert np.asarray(survjax.concordance_index(f(SCORES), _data())).dtype == FLOAT


@pytest.mark.step("A10")
@pytest.mark.skipif(FLOAT != np.float64, reason="needs float64 scores beside float32 data")
def test_result_dtype_follows_float32_data_with_float64_scores():
    """spec 9.1, rev 34: the ratio is in "the data's floating dtype", here float32 data under 64-bit mode."""
    data = survjax.SurvivalData(np.asarray(TIME, dtype=np.float32), b(EVENT))
    value = np.asarray(survjax.concordance_index(np.asarray(SCORES, dtype=np.float64), data))
    assert value.dtype == np.float32
    assert_close(value, C)
