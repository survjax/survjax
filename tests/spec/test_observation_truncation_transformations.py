"""observation.truncation: eager, `jit` and `vmap` agreement and compilation with truncation (spec 3.2, 6.2, 7.2, 12.1; rev 51).

3.2: the static metadata carry "whether any left truncation is present (any
`left_trunc > 0`), and, as a separate flag, whether any right truncation is
present"; traced construction makes every flag true. 7.2: pruning by these
flags is an optimization "with identical results"; new values with unchanged
shapes, dtypes, tree structure and static configuration reuse compilation.
Hand values as in ``test_observation_truncation_values.py``: Exponential
``r``, window ``(L, U]``, ``log Z = -r L + log(-expm1(-r (U - L)))``; exact
``log r - r t - log Z``, right ``-r c + log(-expm1(-r (U - c))) - log Z``.
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, b, f, flag_data, ll

import survjax

TRUNC = pytest.mark.feature("observation.truncation")
BOTH = pytest.mark.feature("observation.censoring", "observation.truncation")

L0, U0 = 0.5, 3.0


def _log_z(r, lt, rt):
    return -r * lt + (np.log(-np.expm1(-r * (rt - lt))) if np.isfinite(rt) else 0.0)


def _expected(r, lt=L0, rt=U0):
    r = float(np.asarray(f(r)))
    lz = _log_z(r, lt, rt)
    right = -r * 1.0 + (np.log(-np.expm1(-r * (rt - 1.0))) if np.isfinite(rt) else 0.0)
    return np.array([np.log(r) - r * 1.0 - lz, right - lz, np.log(r) - r * 2.5 - lz])


def _data(lt=L0, rt=U0):
    return survjax.SurvivalData(f([1.0, 1.0, 2.5]), b([True, False, True]), left_trunc=f([lt] * 3), right_trunc=f([rt] * 3))


@TRUNC
def test_prepared_truncated_data_under_jit_matches_hand_values():
    """spec 12.1, rev 51: data and distribution passed into `jit` give the eager hand values."""
    dist = survjax.Exponential(rate=f(0.7))
    assert_close(np.asarray(jax.jit(lambda d, x: d.log_likelihood(x))(dist, _data())), _expected(0.7))
    assert_close(ll(dist, _data()), _expected(0.7))


@TRUNC
def test_traced_construction_with_truncation_matches_eager():
    """spec 3.2, 6.2 and 7.2, rev 51: construction from traced inputs, truncation included (all metadata true), gives the same values."""

    def run(rate, time, event, lt, rt):
        data = survjax.SurvivalData(time, event, left_trunc=lt, right_trunc=rt)
        return survjax.Exponential(rate=rate).log_likelihood(data)

    out = jax.jit(run)(f(0.7), f([1.0, 1.0, 2.5]), b([True, False, True]), f([L0] * 3), f([U0] * 3))
    assert_close(np.asarray(out), _expected(0.7))


@BOTH
def test_traced_bounds_construction_of_the_full_matrix_matches_eager():
    """spec 3.2, 6.2 and 7.2, rev 51: traced bounds construction with left/interval rows and truncation equals concrete data."""
    lower, upper = f([1.0, 1.0, 0.0, 0.2]), f([1.0, INF, 2.0, 4.0])
    exact = b([True, False, False, False])
    lt, rt = f([L0] * 4), f([U0] * 4)
    dist = survjax.Weibull(shape=f(1.5), scale=f(2.0))
    eager = ll(dist, survjax.SurvivalData.from_bounds(lower, upper, exact=exact, left_trunc=lt, right_trunc=rt))

    def run(lo, hi, ex, a, c):
        data = survjax.SurvivalData.from_bounds(lo, hi, exact=ex, left_trunc=a, right_trunc=c)
        return survjax.Weibull(shape=f(1.5), scale=f(2.0)).log_likelihood(data)

    assert_close(np.asarray(jax.jit(run)(lower, upper, exact, lt, rt)), eager)
    assert np.isfinite(eager).all()


@TRUNC
def test_untruncated_rows_agree_with_and_without_truncation_metadata():
    """spec 3.2 and 7.2, rev 51: an untruncated row has the same value in a dataset whose metadata show truncation and in one without.

    Pruning on the "no truncation" flags must give "identical results"
    (compared at the 12.1 tolerance).
    """
    dist = survjax.Exponential(rate=f(0.7))
    plain = ll(dist, survjax.SurvivalData(f([1.0, 2.0]), b([True, False])))
    mixed = ll(
        dist,
        survjax.SurvivalData(f([1.0, 2.0, 1.0]), b([True, False, True]), left_trunc=f([0.0, 0.0, L0]), right_trunc=f([INF, INF, U0])),
    )
    assert_close(mixed[:2], plain)


@TRUNC
def test_vmap_over_draws_and_datasets():
    """spec 5 and 12.1, rev 51: `vmap` over rate draws and over datasets with different windows."""
    data = _data()
    rates = f([0.5, 0.7, 2.0])
    out = np.asarray(jax.vmap(lambda r: survjax.Exponential(rate=r).log_likelihood(data))(rates))
    assert_close(out, np.stack([_expected(r) for r in rates]))

    stacked = survjax.SurvivalData(
        f([[1.0, 1.0, 2.5]] * 2),
        b([[True, False, True]] * 2),
        left_trunc=f([[L0] * 3, [0.0] * 3]),
        right_trunc=f([[U0] * 3, [INF] * 3]),
    )
    per = np.asarray(jax.vmap(lambda d: survjax.Exponential(rate=f(0.7)).log_likelihood(d))(stacked))
    assert_close(per, [_expected(0.7), _expected(0.7, 0.0, INF)])


@TRUNC
def test_indexing_truncated_data_keeps_windows():
    """spec 3.2, rev 51: `data[idx]` keeps each row's truncation; contributions equal the parent's selected rows."""
    data = survjax.SurvivalData(f([1.0, 1.0, 2.5]), b([True, False, True]), left_trunc=f([0.0, L0, L0]), right_trunc=f([U0, INF, U0]))
    dist = survjax.Exponential(rate=f(0.7))
    full = ll(dist, data)
    assert_close(ll(dist, data[1:]), full[1:])
    assert_close(ll(dist, data[np.array([2, 0])]), full[[2, 0]])


@TRUNC
def test_new_truncation_values_reuse_compilation():
    """spec 7.2, rev 51: new parameter, time and window values with equal metadata reuse compilation."""
    traces = []

    @jax.jit
    def run(dist, data):
        traces.append(None)
        return dist.log_likelihood(data)

    run(survjax.Exponential(rate=f(0.7)), _data())
    run(survjax.Exponential(rate=f(1.5)), survjax.SurvivalData(f([2.0, 1.5, 2.0]), b([True, False, True]), left_trunc=f([1.0] * 3), right_trunc=f([4.0] * 3)))
    assert len(traces) == 1


@TRUNC
def test_traced_construction_with_truncation_reuses_compilation():
    """spec 7.2, rev 51: compiled construction from traced inputs with new window values does not retrace."""
    traces = []

    @jax.jit
    def run(rate, time, event, lt, rt):
        traces.append(None)
        data = survjax.SurvivalData(time, event, left_trunc=lt, right_trunc=rt)
        return survjax.Exponential(rate=rate).log_likelihood(data)

    run(f(0.7), f([1.0, 2.0]), b([True, False]), f([L0, L0]), f([U0, U0]))
    run(f(1.1), f([2.0, 1.0]), b([False, True]), f([0.0, 0.2]), f([INF, 5.0]))
    assert len(traces) == 1


@BOTH
def test_full_matrix_under_jit_of_vmap():
    """spec 12.1, rev 51: `jit(vmap)` over draws on a truncated mixed batch equals the eager `(D, 1)` batch."""
    data = flag_data(
        ["exact", "right", "left", "interval"], [1.0, 1.0, 2.0, 0.2], [NAN, NAN, NAN, 4.0], left_trunc=[L0] * 4, right_trunc=[U0] * 4
    )
    rates = f([0.5, 2.0])
    mapped = jax.jit(jax.vmap(lambda r: survjax.Exponential(rate=r).log_likelihood(data)))(rates)
    assert_close(np.asarray(mapped), ll(survjax.Exponential(rate=f([[0.5], [2.0]])), data))
