"""A6: likelihood shapes and broadcasting (spec 5, 7.1; rev 28, 29).

Exponential contributions, hand-derived: exact ``log r - r t``, right-censored
``-r t``.
"""

import numpy as np
import pytest
from helpers import assert_close, b, f, ll, ll_method

import survjax

EXACT = True
RIGHT = False


def _expected(rate, time, event):
    rate, time, event = np.broadcast_arrays(
        np.asarray(rate, dtype=np.float64),
        np.asarray(time, dtype=np.float64),
        np.asarray(event, dtype=bool),
    )
    return np.where(event, np.log(rate) - rate * time, -rate * time)


TIME = [1.0, 2.0, 4.0]
EVENT = [EXACT, RIGHT, EXACT]


def _data():
    return survjax.SurvivalData(f(TIME), b(EVENT))


@pytest.mark.step("A6")
def test_scalar_data_scalar_distribution_gives_scalar():
    """spec 5, rev 28: scalar data, scalar distribution: `()`."""
    value = ll(survjax.Exponential(rate=f(0.5)), survjax.SurvivalData(f(2.0), b(EXACT)))
    assert value.shape == ()
    assert_close(value, _expected(0.5, 2.0, EXACT))


@pytest.mark.step("A6")
def test_vector_data_scalar_distribution():
    """spec 5, rev 28: data `(N,)`, scalar distribution: `(N,)`."""
    value = ll(survjax.Exponential(rate=f(0.5)), _data())
    assert value.shape == (3,)
    assert_close(value, _expected(0.5, TIME, EVENT))


@pytest.mark.step("A6")
def test_vector_data_paired_distribution():
    """spec 5, rev 28: data `(N,)`, distribution `(N,)`: `(N,)`, paired element by element."""
    rate = [0.5, 1.0, 2.0]
    value = ll(survjax.Exponential(rate=f(rate)), _data())
    assert value.shape == (3,)
    assert_close(value, _expected(rate, TIME, EVENT))


@pytest.mark.step("A6")
def test_draw_column_distribution_gives_draws_by_individuals():
    """spec 5, rev 28: data `(N,)`, distribution `(D,1)`: `(D,N)`, every draw against every individual."""
    rate = [[0.5], [1.0]]
    value = ll(survjax.Exponential(rate=f(rate)), _data())
    assert value.shape == (2, 3)
    assert_close(value, _expected(rate, TIME, EVENT))


@pytest.mark.step("A6")
def test_full_draw_matrix_distribution():
    """spec 5, rev 28: data `(N,)`, distribution `(D,N)`: `(D,N)`."""
    rate = [[0.5, 1.0, 2.0], [0.25, 0.5, 0.75]]
    value = ll(survjax.Exponential(rate=f(rate)), _data())
    assert value.shape == (2, 3)
    assert_close(value, _expected(rate, TIME, EVENT))


@pytest.mark.step("A6")
def test_weibull_draw_column_regression_pattern():
    """spec 5, rev 28: the documented `Weibull(shape=k[:, None], scale=...)` pattern gives `(D,N)`."""
    k = f([1.0, 2.0])
    scale = f([[2.0, 2.0, 2.0], [1.0, 2.0, 4.0]])
    value = ll(survjax.Weibull(shape=k[:, None], scale=scale), _data())
    assert value.shape == (2, 3)
    z = np.asarray(TIME) / np.asarray(scale, dtype=np.float64)
    kk = np.asarray(k, dtype=np.float64)[:, None]
    expected = np.where(
        np.asarray(EVENT),
        np.log(kk) - np.log(np.asarray(scale, dtype=np.float64)) + (kk - 1.0) * np.log(z) - z**kk,
        -(z**kk),
    )
    assert_close(value, expected)


@pytest.mark.step("A6")
def test_multidimensional_data_keeps_its_axes():
    """spec 3.2 and 5, rev 28: multidimensional observations are not flattened; likelihoods keep every axis."""
    time = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    event = [[EXACT, RIGHT, EXACT], [RIGHT, EXACT, RIGHT]]
    data = survjax.SurvivalData(f(time), b(event))
    value = ll(survjax.Exponential(rate=f(0.5)), data)
    assert value.shape == (2, 3)
    assert_close(value, _expected(0.5, time, event))


@pytest.mark.step("A6")
def test_batched_datasets_against_trailing_distribution():
    """spec 5, rev 28: trailing-axis broadcasting of data `(B,N)` against distribution `(N,)` gives `(B,N)`."""
    time = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    event = [[EXACT, RIGHT, EXACT], [RIGHT, EXACT, RIGHT]]
    rate = [0.5, 1.0, 2.0]
    value = ll(survjax.Exponential(rate=f(rate)), survjax.SurvivalData(f(time), b(event)))
    assert value.shape == (2, 3)
    assert_close(value, _expected(rate, time, event))


@pytest.mark.step("A6")
def test_mismatched_bare_batch_raises_type_error():
    """spec 5, rev 28: a bare batch `(D,)` against data `(N,)` with `D != N`, neither 1, raises `TypeError`."""
    with pytest.raises(TypeError):
        survjax.log_likelihood(survjax.Exponential(rate=f([0.5, 1.0])), _data())


@pytest.mark.step("A6")
def test_mismatched_bare_batch_raises_type_error_in_method():
    """spec 5 and 7.1, rev 29: the method, identical to the function, raises the same `TypeError` for a static shape mismatch."""
    with pytest.raises(TypeError):
        survjax.Exponential(rate=f([0.5, 1.0])).log_likelihood(_data())


@pytest.mark.step("A6")
def test_method_follows_broadcast_contribution_shape():
    """spec 5 and 7.1, rev 29: `dist.log_likelihood` gives the function's `(D,N)` contribution shape and values."""
    dist = survjax.Exponential(rate=f([[0.5], [1.0]]))
    value = ll_method(dist, _data())
    assert value.shape == (2, 3)
    np.testing.assert_array_equal(value, ll(dist, _data()))


@pytest.mark.step("A6")
def test_method_scalar_contribution_for_scalar_inputs():
    """spec 5 and 7.1, rev 29: scalar data and distribution give a scalar contribution through the method."""
    value = ll_method(survjax.Exponential(rate=f(0.5)), survjax.SurvivalData(f(2.0), b(EXACT)))
    assert value.shape == ()
    assert_close(value, np.log(0.5) - 1.0)
