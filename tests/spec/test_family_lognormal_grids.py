"""Feature `family.lognormal`: time-first `_grid` twins on `LogNormal` (spec 5, 8.1, 12.1; rev 55).

5: "For `t` of shape `(K,)`, `m_grid(t)` equals the direct method `m`
evaluated at `t` reshaped to `(K, 1, ..., 1)` ... and returns shape `(K,
*batch_shape)`"; "The base provides the grid methods for every registered
family"; the time vector "follows the conversion, dtype and value rules of
section 4.5. A `t` that is not one-dimensional ... raises `TypeError`"; "`t`
need not be sorted or distinct". 12.1 (Shapes): "time-first `_grid` twins of
every pointwise direct method, including `K=N`".
"""

import family_reference as fr
import jax
import numpy as np
import pytest
from helpers import INF, MISUSE, assert_close, f

import survjax

LOGNORMAL = pytest.mark.feature("family.lognormal")
METHODS = fr.DIRECT_METHODS


def _draws_by_individuals():
    """Batch ``(D, N) = (2, 3)``."""
    return survjax.LogNormal(meanlog=f([0.0, 1.0, -0.5]), sdlog=f([[0.5], [1.5]]))


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_grid_equals_direct_method_with_explicit_axes(method):
    """spec 5, rev 55: `m_grid(t)` has shape `(K, D, N)` and equals `m(t[:, None, None])`."""
    d = _draws_by_individuals()
    t = f([0.0, 0.3, 1.0, 4.0, INF])
    grid = getattr(d, f"{method}_grid")(t)
    assert np.shape(grid) == (5, 2, 3)
    assert_close(grid, getattr(d, method)(t[:, None, None]))


@LOGNORMAL
@pytest.mark.parametrize("method", METHODS)
def test_grid_with_as_many_times_as_individuals(method):
    """spec 5/12.1, rev 55: with `K = N = 3` the grid is `(K, N)`, row `i` the method at `t[i]` for every individual, unlike the paired direct call `(N,)`."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0, -0.5]), sdlog=f([0.5, 1.0, 1.5]))
    t = f([0.4, 1.0, 2.5])
    grid = np.asarray(getattr(d, f"{method}_grid")(t))
    assert grid.shape == (3, 3)
    for i in range(3):
        assert_close(grid[i], getattr(d, method)(t[i]))
    assert np.shape(getattr(d, method)(t)) == (3,)


@LOGNORMAL
def test_survival_grid_matches_mpmath_and_starts_at_one():
    """spec 4.5/5, rev 55: `survival_grid` from `t = 0` (base-supplied `S(0) = 1`) through interior times, against mpmath."""
    d = survjax.LogNormal(meanlog=f([0.0, 1.0]), sdlog=f([0.5, 1.5]))
    t = [0.0, 0.5, 2.0, 10.0]
    grid = np.asarray(d.survival_grid(f(t)))
    np.testing.assert_array_equal(grid[0], [1.0, 1.0])
    for i, ti in enumerate(t[1:], start=1):
        for j, (mu, sigma) in enumerate([(0.0, 0.5), (1.0, 1.5)]):
            want, rtol = fr.direct("lognormal", (mu, sigma), ti, "survival")
            np.testing.assert_allclose(grid[i, j], want, rtol=rtol, atol=0.0)


@LOGNORMAL
def test_grid_times_need_not_be_sorted_or_distinct():
    """spec 5, rev 55: "`t` need not be sorted or distinct; output row `i` belongs to `t[i]`"."""
    d = _draws_by_individuals()
    t = f([3.0, 0.5, 3.0])
    grid = np.asarray(d.log_survival_grid(t))
    assert_close(grid[0], grid[2])
    assert_close(grid[1], d.log_survival(f(0.5)))


@LOGNORMAL
@pytest.mark.parametrize("bad", [np.float64(1.0), np.ones((2, 2))], ids=["scalar", "two-dimensional"])
def test_grid_time_that_is_not_one_dimensional_is_rejected(bad):
    """spec 5, rev 55: a `t` that is not one-dimensional is static misuse and raises `TypeError`."""
    with pytest.raises(MISUSE):
        _draws_by_individuals().survival_grid(f(bad))


@LOGNORMAL
def test_boolean_grid_time_is_rejected():
    """spec 4.5/5, rev 55: the grid `t` follows the 4.5 dtype rule: a boolean vector raises `TypeError`."""
    with pytest.raises(TypeError):
        _draws_by_individuals().log_prob_grid(np.array([True, False]))


@LOGNORMAL
def test_grid_under_jit_matches_eager():
    """spec 5/12.1, rev 55: eager and `jit` grids agree."""
    d = _draws_by_individuals()
    t = f([0.2, 1.0, 5.0])
    assert_close(jax.jit(lambda dd, tt: dd.log_hazard_grid(tt))(d, t), d.log_hazard_grid(t))
