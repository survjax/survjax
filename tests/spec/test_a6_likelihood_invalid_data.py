"""A6: stored input issues reach the likelihood (spec 3.3, 6.1, 6.2, 7.1, 12.1; rev 29).

Every row in ``data.issues.invalid_rows`` yields NaN, "even if normalization
made its fields appear valid", "in every constructor form and to indexed, rebuilt and
transformed data". Each case pairs one invalid row (index 1) with a valid
exact row (index 0, exponential ``r=0.5, t=2``: ``log 0.5 - 1``), whose
value must be unchanged. Since revision 29 there is no ``LikelihoodIssues``;
the cause of a NaN row is read from ``data.issues`` (7.1 recipe), so each case
checks that the NaN rows are exactly ``data.issues.invalid_rows``.
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, b, f, ll, ll_method, nan_causes

import survjax

EXACT = True
RIGHT = False
VALID_ROW = -np.log(2.0) - 1.0
DIST_RATE = 0.5


def _dist():
    return survjax.Exponential(rate=f(DIST_RATE))


def _assert_row1_invalid(value, invalid_rows):
    """Row 1 is NaN, row 0 keeps its value, and the NaN rows are `invalid_rows`."""
    value = np.asarray(value)
    assert np.isnan(value[1])
    assert_close(value[0], VALID_ROW)
    np.testing.assert_array_equal(np.asarray(invalid_rows).astype(bool), [False, True])
    np.testing.assert_array_equal(np.isnan(value), np.asarray(invalid_rows).astype(bool))


# ---------------------------------------------------- flag constructor rows


@pytest.mark.step("A6")
@pytest.mark.parametrize("t", [0.0, -1.0, INF, NAN], ids=["zero", "negative", "inf", "nan"])
def test_invalid_exact_time_is_nan(t):
    """spec 3.3 and 7.1, rev 29: exact times must be finite and `>0`; others are in `invalid_rows` and give NaN."""
    data = survjax.SurvivalData(f([2.0, t]), b([EXACT, EXACT]))
    _assert_row1_invalid(ll(_dist(), data), data.issues.invalid_rows)


@pytest.mark.step("A6")
@pytest.mark.parametrize("c", [-1.0, INF, NAN], ids=["negative", "inf", "nan"])
def test_invalid_right_censoring_time_is_nan(c):
    """spec 3.3 and 7.1, rev 29: right censoring at a negative, `+inf` or NaN time is invalid: NaN on that row."""
    data = survjax.SurvivalData(f([2.0, c]), b([EXACT, RIGHT]))
    _assert_row1_invalid(ll(_dist(), data), data.issues.invalid_rows)


@pytest.mark.step("A6")
def test_invalid_event_code_is_nan_although_normalized_fields_look_valid():
    """spec 7.1, rev 29: an invalid indicator yields NaN "even if normalization made its fields appear valid"."""
    data = survjax.SurvivalData(f([2.0, 2.0]), np.array([1, 2], dtype=np.int32))
    _assert_row1_invalid(ll(_dist(), data), data.issues.invalid_rows)


@pytest.mark.step("A6")
def test_nan_event_indicator_is_nan():
    """spec 3.1 and 7.1, rev 29: a NaN indicator sets `invalid_indicator`; the likelihood row is NaN."""
    data = survjax.SurvivalData(f([2.0, 2.0]), f([1.0, NAN]))
    _assert_row1_invalid(ll(_dist(), data), data.issues.invalid_rows)


@pytest.mark.step("A6")
def test_bounds_exact_row_with_mismatched_upper_is_nan():
    """spec 3.1 and 7.1, rev 29: an exact `from_bounds` row with `upper` neither `lower` nor NaN is invalid."""
    data = survjax.SurvivalData.from_bounds(f([2.0, 2.0]), f([2.0, 3.0]), exact=b([True, True]))
    _assert_row1_invalid(ll(_dist(), data), data.issues.invalid_rows)


@pytest.mark.step("A6")
def test_bounds_valid_exact_and_right_rows_match_flag_form():
    """spec 3.1 and 7.1, rev 28: equivalent bounds and flag data give identical contributions."""
    bounds = survjax.SurvivalData.from_bounds(
        f([2.0, 3.0]), f([NAN, INF]), exact=b([True, False])
    )
    flags = survjax.SurvivalData(f([2.0, 3.0]), b([EXACT, RIGHT]))
    np.testing.assert_array_equal(ll(_dist(), bounds), ll(_dist(), flags))
    assert_close(ll(_dist(), bounds), [VALID_ROW, -1.5])


# ---------------------------------------- indexed, rebuilt and transformed data


def _with_invalid_indicator():
    """Rows: valid exact, invalid indicator, valid exact."""
    return survjax.SurvivalData(f([2.0, 2.0, 2.0]), np.array([1, 2, 1], dtype=np.int32))


@pytest.mark.step("A6")
def test_indexed_data_keeps_invalid_rows():
    """spec 7.1, rev 29: stored issues propagate "after indexing"."""
    data = _with_invalid_indicator()[:2]
    _assert_row1_invalid(ll(_dist(), data), data.issues.invalid_rows)


@pytest.mark.step("A6")
def test_rebuilt_data_keeps_invalid_rows():
    """spec 7.1, rev 29: stored issues propagate after flatten/unflatten reconstruction."""
    leaves, tree = jax.tree_util.tree_flatten(_with_invalid_indicator()[:2])
    data = jax.tree_util.tree_unflatten(tree, leaves)
    _assert_row1_invalid(ll(_dist(), data), data.issues.invalid_rows)


@pytest.mark.step("A6")
def test_data_returned_from_jit_keeps_invalid_rows():
    """spec 7.1, rev 29: data constructed inside `jit` and returned keeps its issue evidence."""
    data = jax.jit(survjax.SurvivalData)(f([2.0, 2.0]), np.array([1, 2], dtype=np.int32))
    _assert_row1_invalid(ll(_dist(), data), data.issues.invalid_rows)


@pytest.mark.step("A6")
def test_traced_construction_and_likelihood_under_jit():
    """spec 6.2 and 7.1, rev 29: construction and likelihood inside `jit` give NaN on `invalid_rows` without exceptions."""

    def run(time, event):
        data = survjax.SurvivalData(time, event)
        return survjax.log_likelihood(_dist(), data), data.issues.invalid_rows

    value, invalid_rows = jax.jit(run)(f([2.0, 2.0]), np.array([1, 2], dtype=np.int32))
    _assert_row1_invalid(value, invalid_rows)


@pytest.mark.step("A6")
def test_vmap_over_data_keeps_invalid_rows():
    """spec 7.1, rev 29: stored issues propagate under `vmap` over observations."""
    data = _with_invalid_indicator()
    value = np.asarray(jax.vmap(lambda d: survjax.log_likelihood(_dist(), d))(data))
    np.testing.assert_array_equal(np.isnan(value), [False, True, False])
    np.testing.assert_array_equal(np.isnan(value), np.asarray(data.issues.invalid_rows))
    assert_close(value[[0, 2]], [VALID_ROW, VALID_ROW])


@pytest.mark.step("A6")
def test_method_matches_function_on_invalid_data():
    """spec 7.1, rev 29: `dist.log_likelihood` "returns exactly what the function returns", NaN rows included, without raising."""
    data = _with_invalid_indicator()
    plain = ll(_dist(), data)
    method = ll_method(_dist(), data)
    np.testing.assert_array_equal(method, plain)
    np.testing.assert_array_equal(np.isnan(plain), [False, True, False])


# ---------------------------------------------------------------- raw form


@pytest.mark.step("A6")
def test_raw_form_invalid_event_code_returns_nan_eagerly():
    """spec 6.1, rev 29: "An eager raw-form call with, say, event code 2 returns NaN rather than raising".

    The cause is read from ``SurvivalData(time, event).issues`` (7.1 recipe).
    """
    time, event = f([2.0, 2.0]), np.array([1, 2], dtype=np.int32)
    value = ll(_dist(), time=time, event=event)
    _assert_row1_invalid(value, survjax.SurvivalData(time, event).issues.invalid_rows)


@pytest.mark.step("A6")
def test_raw_form_invalid_time_returns_nan_eagerly():
    """spec 6.1 and 7.1, rev 29: raw-form invalid times give NaN, on the rows `SurvivalData(time, event)` marks."""
    time, event = f([2.0, 0.0]), b([EXACT, EXACT])
    value = ll(_dist(), time=time, event=event)
    _assert_row1_invalid(value, survjax.SurvivalData(time, event).issues.invalid_rows)


@pytest.mark.step("A6")
def test_raw_form_method_invalid_event_code_returns_nan_eagerly():
    """spec 6.1 and 7.1, rev 29: the method's raw form also returns NaN rather than raising."""
    time, event = f([2.0, 2.0]), np.array([1, 2], dtype=np.int32)
    value = _dist().log_likelihood(time=time, event=event)
    _assert_row1_invalid(value, survjax.SurvivalData(time, event).issues.invalid_rows)


@pytest.mark.step("A6")
def test_raw_form_invalid_event_code_under_jit():
    """spec 6.1, rev 29: the raw form "behaves identically eagerly and under `jit`": NaN on the invalid row."""

    def run(time, event):
        return survjax.log_likelihood(_dist(), time=time, event=event)

    time, event = f([2.0, 2.0]), np.array([1, 2], dtype=np.int32)
    value = jax.jit(run)(time, event)
    _assert_row1_invalid(value, survjax.SurvivalData(time, event).issues.invalid_rows)


# ----------------------------------------------------- non-fatal and separation


@pytest.mark.step("A6")
def test_uninformative_bits_alone_do_not_invalidate():
    """spec 7.1, rev 29: "Non-fatal `uninformative` bits alone do not invalidate a contribution"."""
    data = survjax.SurvivalData(f([2.0, 0.0]), b([EXACT, RIGHT]))
    value = ll(_dist(), data)
    assert_close(value, [VALID_ROW, 0.0])
    assert not np.asarray(data.issues.invalid_rows).any()


@pytest.mark.step("A6")
def test_invalid_data_rows_triage_to_data_only():
    """spec 7.1, rev 29: in the documented triage recipe, invalid-data NaN rows are `bad_data`, not `bad_params` or `other`."""
    data = survjax.SurvivalData(f([2.0, 0.0]), b([EXACT, EXACT]))
    value = ll(_dist(), data)
    bad_data, bad_params, other = nan_causes(_dist(), data, value)
    np.testing.assert_array_equal(bad_data, [False, True])
    assert not bad_params.any()
    assert not other.any()


@pytest.mark.step("A6")
def test_invalid_data_broadcasts_over_draws():
    """spec 5 and 7.1, rev 29: an invalid row is NaN for every draw of a `(D,1)` distribution; the recipe broadcasts `invalid_rows` to `(D,N)`."""
    data = survjax.SurvivalData(f([2.0, 0.0]), b([EXACT, EXACT]))
    dist = survjax.Exponential(rate=f([[0.5], [1.0]]))
    value = ll(dist, data)
    assert value.shape == (2, 2)
    np.testing.assert_array_equal(np.isnan(value), [[False, True], [False, True]])
    bad_data, _, other = nan_causes(dist, data, value)
    np.testing.assert_array_equal(bad_data, np.isnan(value))
    assert not other.any()
