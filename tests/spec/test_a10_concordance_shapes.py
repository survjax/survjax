"""A10: concordance observation axis, broadcasting and transformations (spec 9.1, 9.2, 12.1; rev 34).

9.1: "The observation axis is the **last** axis, and it must have the same
length in data and predictions. It is never broadcast ... Leading axes
broadcast normally and are never pooled." 9.2 table:

| Data | Scores | Result |
| `(N,)` | `(N,)` | `()` |
| `(N,)` | `(D,N)` | `(D,)` |
| `(B,N)` | `(B,N)` | `(B,)` |
| `(B,1,N)` | `(D,N)` | `(B,D)` |
| `(N,)` | `(K,D,N)` | `(K,D)` |

Expected values come from per-entry calls on ``(N,)`` data and ``(N,)``
scores, whose values the hand-enumerated tests fix; the four score vectors
below are checked by hand against two datasets:

Dataset P (the reference case): ``time 1,2,3,3,4``, ``event E,C,E,C,E``.
Dataset Q: ``time 4,3,2,1,5``, ``event E,E,C,E,C``. Pairs of Q (earlier
row first): ``(3,0)``, ``(3,1)``, ``(3,2)``, ``(3,4)`` (row 3 exact at 1);
``(1,0)``, ``(1,4)`` (row 1 exact at 3); ``(0,4)`` (row 0 exact at 4);
row 2 (censored at 2) is earlier than rows 1, 0, 4: none. 7 comparable.

Score vectors and results (concordant, discordant, tied):

- ``s1 = 0.9, 0.5, 0.7, 0.7, 0.8``: P ``(4,1,1)``, ``C=4.5/6``;
  Q: ``(3,0)`` 0.7<0.9 d, ``(3,1)`` 0.7>0.5 c, ``(3,2)`` 0.7=0.7 t,
  ``(3,4)`` 0.7<0.8 d, ``(1,0)`` 0.5<0.9 d, ``(1,4)`` 0.5<0.8 d,
  ``(0,4)`` 0.9>0.8 c: ``(2,4,1)``, ``C=2.5/7``.
- ``s2 = 5, 4, 3, 2, 1``: P: ``(0,·)`` 4 c; ``(2,3)`` 3>2 c; ``(2,4)`` 3>1
  c: ``(6,0,0)``, ``C=1``. Q: ``(3,0)`` 2<5 d, ``(3,1)`` 2<4 d, ``(3,2)``
  2<3 d, ``(3,4)`` 2>1 c, ``(1,0)`` 4<5 d, ``(1,4)`` 4>1 c, ``(0,4)`` 5>1
  c: ``(3,4,0)``, ``C=3/7``.
- ``s3 = -s2``: P ``C=0``; Q ``C=4/7``.
- ``s4 = 1, 1, 1, 1, 1``: P ``C=0.5``; Q ``C=0.5``.
"""

import jax
import numpy as np
import pytest
from helpers import assert_close, b, count, f, result_mask

import survjax

E, C_ = True, False
P_TIME, P_EVENT = [1.0, 2.0, 3.0, 3.0, 4.0], [E, C_, E, C_, E]
Q_TIME, Q_EVENT = [4.0, 3.0, 2.0, 1.0, 5.0], [E, E, C_, E, C_]
S = {
    "s1": [0.9, 0.5, 0.7, 0.7, 0.8],
    "s2": [5.0, 4.0, 3.0, 2.0, 1.0],
    "s3": [-5.0, -4.0, -3.0, -2.0, -1.0],
    "s4": [1.0, 1.0, 1.0, 1.0, 1.0],
}
EXPECTED = {
    "P": {"s1": 4.5 / 6.0, "s2": 1.0, "s3": 0.0, "s4": 0.5},
    "Q": {"s1": 2.5 / 7.0, "s2": 3.0 / 7.0, "s3": 4.0 / 7.0, "s4": 0.5},
}


def _p():
    return survjax.SurvivalData(f(P_TIME), b(P_EVENT))


def _ci(scores, data):
    return np.asarray(survjax.concordance_index(scores, data))


@pytest.mark.step("A10")
@pytest.mark.parametrize("name", sorted(S))
def test_dataset_q_hand_values(name):
    """spec 9.2, rev 34: the per-vector values on dataset Q (module docstring) that the batched tests rely on."""
    data = survjax.SurvivalData(f(Q_TIME), b(Q_EVENT))
    assert_close(_ci(f(S[name]), data), EXPECTED["Q"][name])


@pytest.mark.step("A10")
def test_vector_data_and_vector_scores_give_a_scalar():
    """spec 9.2, rev 34: data `(N,)`, scores `(N,)` give `()`."""
    value = _ci(f(S["s1"]), _p())
    assert value.shape == ()
    assert_close(value, EXPECTED["P"]["s1"])


@pytest.mark.step("A10")
def test_score_draws_give_one_value_per_draw():
    """spec 9.2, rev 34: data `(N,)`, scores `(D,N)` give `(D,)`."""
    names = ["s1", "s2", "s3", "s4"]
    value = _ci(f([S[n] for n in names]), _p())
    assert value.shape == (4,)
    assert_close(value, [EXPECTED["P"][n] for n in names])


@pytest.mark.step("A10")
def test_batched_datasets_with_matching_scores():
    """spec 9.2, rev 34: data `(B,N)`, scores `(B,N)` give `(B,)`; dataset `b` with score row `b`, never pooled."""
    data = survjax.SurvivalData(f([P_TIME, Q_TIME]), b([P_EVENT, Q_EVENT]))
    value = _ci(f([S["s1"], S["s2"]]), data)
    assert value.shape == (2,)
    assert_close(value, [EXPECTED["P"]["s1"], EXPECTED["Q"]["s2"]])


@pytest.mark.step("A10")
def test_dataset_column_against_score_draws():
    """spec 9.2, rev 34: data `(B,1,N)`, scores `(D,N)` give `(B,D)`."""
    data = survjax.SurvivalData(f([[P_TIME], [Q_TIME]]), b([[P_EVENT], [Q_EVENT]]))
    names = ["s1", "s2", "s3"]
    value = _ci(f([S[n] for n in names]), data)
    assert value.shape == (2, 3)
    assert_close(value, [[EXPECTED[d][n] for n in names] for d in ("P", "Q")])


@pytest.mark.step("A10")
def test_two_leading_score_axes():
    """spec 9.2, rev 34: data `(N,)`, scores `(K,D,N)` give `(K,D)`."""
    grid = [["s1", "s2"], ["s3", "s4"], ["s2", "s1"]]
    value = _ci(f([[S[n] for n in row] for row in grid]), _p())
    assert value.shape == (3, 2)
    assert_close(value, [[EXPECTED["P"][n] for n in row] for row in grid])


@pytest.mark.step("A10")
def test_raw_form_follows_the_shape_table():
    """spec 6.1 and 9.2, rev 34: the raw form gives the data form's `(B,D)` result for `(B,1,N)` arrays."""
    names = ["s1", "s2", "s3"]
    value = np.asarray(
        survjax.concordance_index(
            f([S[n] for n in names]),
            time=f([[P_TIME], [Q_TIME]]),
            event=b([[P_EVENT], [Q_EVENT]]),
        )
    )
    assert value.shape == (2, 3)
    assert_close(value, [[EXPECTED[d][n] for n in names] for d in ("P", "Q")])


@pytest.mark.step("A10")
def test_with_info_counts_follow_the_result_shape():
    """spec 9.2, rev 34: with `(D,N)` scores the info counts have one entry per draw (Q-044)."""
    _, info = survjax.concordance_index_with_info(f([S["s1"], S["s2"]]), _p())
    assert list(count(info, "concordant")) == [4, 6]
    assert list(count(info, "comparable")) == [6, 6]


# ---------------------------------------------------------------- misuse


@pytest.mark.step("A10")
def test_mismatched_observation_length_raises_type_error():
    """spec 9.1 and 6.2, rev 34: the last axis "must have the same length in data and predictions" (static shape misuse)."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f([0.1, 0.2, 0.3, 0.4]), _p())


@pytest.mark.step("A10")
def test_single_score_is_not_broadcast_across_the_cohort():
    """spec 9.1, rev 34: the observation axis "is never broadcast": `(1,)` scores against `(N,)` data raise."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f([0.5]), _p())


@pytest.mark.step("A10")
def test_column_of_scores_is_not_broadcast_across_the_cohort():
    """spec 9.1, rev 34: `(N,1)` scores have last-axis length 1, not `N`: `TypeError`."""
    with pytest.raises(TypeError):
        survjax.concordance_index(f([[s] for s in S["s1"]]), _p())


@pytest.mark.step("A10")
def test_single_observation_data_is_not_broadcast_across_scores():
    """spec 9.1, rev 34: `(1,)` data against `(N,)` scores raise; one individual is never broadcast."""
    data = survjax.SurvivalData(f([1.0]), b([E]))
    with pytest.raises(TypeError):
        survjax.concordance_index(f(S["s1"]), data)


@pytest.mark.step("A10")
def test_incompatible_leading_axes_raise_type_error():
    """spec 9.1 and 6.2, rev 34: leading axes "broadcast normally"; `(2,N)` data against `(3,N)` scores cannot broadcast."""
    data = survjax.SurvivalData(f([P_TIME, Q_TIME]), b([P_EVENT, Q_EVENT]))
    with pytest.raises(TypeError):
        survjax.concordance_index(f([S["s1"], S["s2"], S["s3"]]), data)


@pytest.mark.step("A10")
def test_mismatched_observation_length_raises_under_jit():
    """spec 6.1, rev 34: static misuse raises during tracing as well."""
    with pytest.raises(TypeError):
        jax.jit(survjax.concordance_index)(f([0.1, 0.2, 0.3, 0.4]), _p())


# ----------------------------------------------------------- transformations


@pytest.mark.step("A10")
def test_vmap_over_score_draws_matches_batched_call():
    """spec 12.1, rev 34: `vmap` over draws equals the `(D,N)` call."""
    scores = f([S["s1"], S["s2"], S["s3"]])
    mapped = jax.vmap(lambda s: survjax.concordance_index(s, _p()))(scores)
    assert_close(np.asarray(mapped), _ci(scores, _p()))


@pytest.mark.step("A10")
def test_vmap_over_datasets_matches_batched_call():
    """spec 12.1, rev 34: `vmap` over `(B,N)` data and scores equals the batched call."""
    data = survjax.SurvivalData(f([P_TIME, Q_TIME]), b([P_EVENT, Q_EVENT]))
    scores = f([S["s1"], S["s2"]])
    mapped = jax.vmap(survjax.concordance_index)(scores, data)
    assert_close(np.asarray(mapped), [EXPECTED["P"]["s1"], EXPECTED["Q"]["s2"]])


@pytest.mark.step("A10")
def test_jit_of_vmap_matches_eager():
    """spec 12.1, rev 34: eager, `jit` and `vmap` agree, including the info masks."""
    data = survjax.SurvivalData(f([P_TIME, Q_TIME]), b([P_EVENT, Q_EVENT]))
    scores = f([S["s1"], S["s2"]])
    value, info = jax.jit(jax.vmap(survjax.concordance_index_with_info))(scores, data)
    assert_close(np.asarray(value), [EXPECTED["P"]["s1"], EXPECTED["Q"]["s2"]])
    assert not result_mask(info, "invalid", (2,)).any()
    assert not result_mask(info, "no_comparable_pairs", (2,)).any()


@pytest.mark.step("A10")
def test_new_scores_reuse_compilation():
    """spec 7.2, rev 34: changing dynamic values with equal shapes and dtypes reuses compilation (stated for "the same compiled function")."""
    traces = []

    @jax.jit
    def run(scores, data):
        traces.append(None)
        return survjax.concordance_index(scores, data)

    run(f(S["s1"]), _p())
    run(f(S["s2"]), survjax.SurvivalData(f(Q_TIME), b(Q_EVENT)))
    assert len(traces) == 1
