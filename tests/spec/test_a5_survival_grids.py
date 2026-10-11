"""A5: survival grids and the direct-method shape rules (spec 5, 8.1).

Spec 5: "`survival_grid`, `cumulative_hazard_grid` and `hazard_grid` take a
one-dimensional shared time vector and return time-first outputs, equivalent
to passing `grid[:, None, None]` for batch `(D,N)`. Component axes never appear
in outputs." Rev 22: the base provides the grid methods for every registered
family; the time vector is the first positional argument, named `t`, follows
the rules of section 4.5, and a `t` that is not one-dimensional raises
`TypeError`.

Hand values: Exponential ``S = exp(-r t)``, ``H = r t``, ``h = r``; Weibull
``S = exp(-(t/b)**k)``, ``H = (t/b)**k``, ``h = (k/b)(t/b)**(k-1)``.
"""

import warnings

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, MISUSE, NAN, X64, assert_close, f

import survjax

GRID_METHODS = {
    "survival_grid": "survival",
    "cumulative_hazard_grid": "cumulative_hazard",
    "hazard_grid": "hazard",
}

# --------------------------------------------------------------- section 5 shape table (direct methods)


@pytest.mark.step("A5")
def test_scalar_time_against_batch_gives_batch_shape():
    """spec 5, rev 21: "Survival | time scalar, distribution `(N,)` | `(N,)`"."""
    out = survjax.Exponential(rate=f([0.5, 1.0, 2.0])).survival(f(2.0))
    assert np.shape(out) == (3,)
    assert_close(out, np.exp(-np.array([0.5, 1.0, 2.0]) * 2.0))


@pytest.mark.step("A5")
def test_time_vector_against_equal_batch_pairs_entries():
    """spec 5, rev 21: "Survival | time `(N,)`, distribution `(N,)` | `(N,)`, paired"."""
    r, t = np.array([0.5, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])
    out = survjax.Exponential(rate=f(r)).survival(f(t))
    assert np.shape(out) == (3,)
    assert_close(out, np.exp(-r * t))


@pytest.mark.step("A5")
def test_time_that_does_not_broadcast_against_batch_is_rejected():
    """spec 4.1/5, rev 21: a time argument that does not broadcast against `batch_shape` raises `TypeError`."""
    with pytest.raises(MISUSE):
        survjax.Weibull(shape=f([1.0, 2.0]), scale=f(1.0)).survival(f([1.0, 2.0, 3.0]))


@pytest.mark.step("A5")
def test_explicit_time_first_axis_gives_shared_grid_shape():
    """spec 5, rev 21: trailing-axis broadcasting; times `(K,1,1)` against batch `(D,N)` give `(K,D,N)`."""
    dist = survjax.Weibull(shape=f([[1.0], [2.0]]), scale=f([1.0, 2.0, 3.0]))
    out = dist.survival(f([0.5, 1.0, 2.0, 4.0])[:, None, None])
    assert np.shape(out) == (4, 2, 3)


# --------------------------------------------------------------- grid shapes


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_on_scalar_distribution_has_grid_shape(method):
    """spec 5, rev 21: time-first output; a scalar batch gives `(K,)`."""
    out = getattr(survjax.Exponential(rate=f(0.5)), method)(f([0.5, 1.0, 2.0, 4.0]))
    assert np.shape(out) == (4,)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_on_batch_is_time_first(method):
    """spec 5, rev 21: time-first output; batch `(N,)` gives `(K,N)`."""
    out = getattr(survjax.Exponential(rate=f([0.5, 1.0, 2.0])), method)(
        f([0.5, 1.0, 2.0, 4.0])
    )
    assert np.shape(out) == (4, 3)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_on_two_dimensional_batch_gives_k_d_n(method):
    """spec 5, rev 21: "Shared grid | time `(K,)`, distribution `(D,N)` | `(K,D,N)`"."""
    dist = survjax.Weibull(shape=f([[1.0], [2.0]]), scale=f([1.0, 2.0, 3.0]))
    out = getattr(dist, method)(f([0.5, 1.0, 2.0, 4.0]))
    assert np.shape(out) == (4, 2, 3)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_equals_direct_method_on_grid_with_new_axes(method):
    """spec 5, rev 21: grids are "equivalent to passing `grid[:, None, None]` for batch `(D,N)`"."""
    dist = survjax.Weibull(shape=f([[0.7], [1.0], [2.5]]), scale=f([0.5, 2.0]))
    grid = f([0.1, 0.5, 1.0, 3.0, 10.0])
    direct = getattr(dist, GRID_METHODS[method])(grid[:, None, None])
    assert_close(getattr(dist, method)(grid), direct)


@pytest.mark.step("A5")
def test_grid_with_length_equal_to_batch_is_not_paired():
    """spec 5, rev 21: a grid is shared, not paired: `K=N=3` gives `(3,3)` with `out[i,j] = S_j(t_i)`."""
    r, t = np.array([0.5, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])
    out = survjax.Exponential(rate=f(r)).survival_grid(f(t))
    assert np.shape(out) == (3, 3)
    assert_close(out, np.exp(-t[:, None] * r[None, :]))


@pytest.mark.step("A5")
def test_grid_has_no_component_axes():
    """spec 5, rev 22: "Component axes never appear in outputs"; "The base provides the grid methods for every registered family".

    ``ComponentRate``: ``S = exp(-rate * sum(w) * t)``; rates ``(1, 2)``,
    component sums ``(6, 1.5)``, so effective rates ``(6, 3)``.
    """
    dist = fam.component_rate()(
        rate=f([1.0, 2.0]), w=f([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    )
    t = np.array([0.0, 0.1, 0.5])
    out = dist.survival_grid(f(t))
    assert np.shape(out) == (3, 2)
    assert_close(out, np.exp(-t[:, None] * np.array([6.0, 3.0])[None, :]))


# --------------------------------------------------------------- grid values


@pytest.mark.step("A5")
def test_weibull_survival_grid_values():
    """spec 5/8.1, rev 21: `survival_grid` values `exp(-(t/b)**k)`, time-first."""
    k, b = np.array([0.5, 1.0, 3.0]), np.array([2.0, 1.0, 4.0])
    t = np.array([0.0, 0.5, 2.0, 8.0])
    out = survjax.Weibull(shape=f(k), scale=f(b)).survival_grid(f(t))
    assert_close(out, np.exp(-((t[:, None] / b) ** k)))


@pytest.mark.step("A5")
def test_weibull_cumulative_hazard_grid_values():
    """spec 5/8.1, rev 21: `cumulative_hazard_grid` values `(t/b)**k`, time-first."""
    k, b = np.array([0.5, 1.0, 3.0]), np.array([2.0, 1.0, 4.0])
    t = np.array([0.0, 0.5, 2.0, 8.0])
    out = survjax.Weibull(shape=f(k), scale=f(b)).cumulative_hazard_grid(f(t))
    assert_close(out, (t[:, None] / b) ** k)


@pytest.mark.step("A5")
def test_weibull_hazard_grid_values():
    """spec 5/8.1, rev 21: `hazard_grid` values `(k/b)(t/b)**(k-1)` on positive times, time-first."""
    k, b = np.array([0.5, 1.0, 3.0]), np.array([2.0, 1.0, 4.0])
    t = np.array([0.5, 2.0, 8.0])
    out = survjax.Weibull(shape=f(k), scale=f(b)).hazard_grid(f(t))
    assert_close(out, (k / b) * (t[:, None] / b) ** (k - 1.0))


@pytest.mark.step("A5")
def test_exponential_grids_values():
    """spec 5/8.1, rev 21: Exponential grids `exp(-r t)`, `r t` and `r`."""
    r, t = np.array([0.5, 2.0]), np.array([0.0, 1.0, 3.0])
    dist = survjax.Exponential(rate=f(r))
    assert_close(dist.survival_grid(f(t)), np.exp(-t[:, None] * r))
    assert_close(dist.cumulative_hazard_grid(f(t)), t[:, None] * r)
    assert_close(dist.hazard_grid(f(t)), np.broadcast_to(r, (3, 2)))


# --------------------------------------------------------------- grids from t = 0


@pytest.mark.step("A5")
def test_survival_grid_from_zero_with_weibull_shape_one():
    """spec 4.5/12.1, rev 22: test case "a survival grid starting at `t=0` with Weibull `k=1`".

    The section 11.1 fixed vector `(1, 10, -0.5, -1)`: ``k=1``, Good-group
    scale ``10``, ``beta_Medium=-0.5``, ``beta_Poor=-1``, so the three group
    scales are ``10``, ``10 e**-0.5`` and ``10 e**-1``. ``S = exp(-t/b)``, and
    ``S(0) = 1`` exactly.
    """
    b = 10.0 * np.exp(np.array([0.0, -0.5, -1.0]))
    t = np.array([0.0, 0.5, 1.0, 5.0, 20.0])
    out = np.asarray(survjax.Weibull(shape=f(1.0), scale=f(b)).survival_grid(f(t)))
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[0], [1.0, 1.0, 1.0])
    assert_close(out, np.exp(-t[:, None] / b))


@pytest.mark.step("A5")
def test_hazard_grid_from_zero_with_weibull_shape_one_is_finite():
    """spec 4.5, rev 21: with `k=1` the grid from `t=0` gives `h = 1/b` throughout, `t=0` included."""
    b = np.array([10.0, 2.0])
    t = np.array([0.0, 1.0, 5.0])
    out = survjax.Weibull(shape=f(1.0), scale=f(b)).hazard_grid(f(t))
    assert_close(out, np.broadcast_to(1.0 / b, (3, 2)))


@pytest.mark.step("A5")
def test_log_prob_on_grid_from_zero_with_weibull_shape_one_is_finite():
    """spec 4.5, rev 21: `k=1`, times `(K,1)` from `t=0`: `log f = -log b - t/b`, finite throughout."""
    b = np.array([10.0, 2.0])
    t = np.array([0.0, 1.0, 5.0])
    out = survjax.Weibull(shape=f(1.0), scale=f(b)).log_prob(f(t)[:, None])
    assert np.all(np.isfinite(np.asarray(out)))
    assert_close(out, -np.log(b) - t[:, None] / b)


# --------------------------------------------------------------- policy on grids


@pytest.mark.step("A5")
def test_survival_grid_applies_value_policy():
    """spec 4.5/5, rev 21: grids follow the direct-method value policy: `t<0` `1`, `+inf` `0`, NaN NaN."""
    out = survjax.Weibull(shape=f([0.5, 2.0]), scale=f(2.0)).survival_grid(
        f([-1.0, 0.0, INF, NAN])
    )
    assert_close(out, [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [NAN, NAN]])


@pytest.mark.step("A5")
def test_hazard_grid_at_infinity_uses_analytic_limit():
    """spec 4.5, rev 21: where `S(t)=0`, an overriding family gives "the limit" (Weibull `k=0.5`: `0`; `k=2`: `+inf`; `t<0`: `0`)."""
    out = survjax.Weibull(shape=f([0.5, 2.0]), scale=f(2.0)).hazard_grid(f([-1.0, INF]))
    assert_close(out, [[0.0, 0.0], [0.0, INF]])


@pytest.mark.step("A5")
@pytest.mark.skipif(
    X64, reason="float64 conversion of `t` applies with 64-bit mode disabled"
)
def test_float64_numpy_grid_converts_without_warning_under_32bit_mode():
    """spec 4.5/5, rev 21: a float64 NumPy grid converts to float32 without a warning ("NumPy grids are float64 by default")."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    grid = np.linspace(0.0, 2.0, 5)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = dist.survival_grid(grid)
    record = [
        w
        for w in record
        if not issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))
    ]
    assert record == []
    assert np.dtype(out.dtype) == np.float32
    assert_close(out, np.exp(-grid[:, None] * np.array([0.5, 2.0])))


# --------------------------------------------------------------- transformations


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_agrees_under_jit(method):
    """spec 12.1, rev 21: eager and `jit` agree for grids."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))
    grid = f([0.0, 0.5, 1.0, 4.0])
    jitted = jax.jit(lambda d, g: getattr(d, method)(g))(dist, grid)
    assert_close(jitted, getattr(dist, method)(grid))


@pytest.mark.step("A5")
def test_vmap_over_distribution_of_survival_grid():
    """spec 4.1/12.1, rev 21: `vmap` over the batch gives each entry's `(K,)` grid; stacked, the transpose of the batched grid."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))
    grid = f([0.0, 0.5, 1.0, 4.0])
    mapped = jax.vmap(lambda d: d.survival_grid(grid))(dist)
    assert np.shape(mapped) == (3, 4)
    assert_close(mapped, np.asarray(dist.survival_grid(grid)).T)


@pytest.mark.step("A5")
def test_gradient_through_survival_grid():
    """spec 7.3, rev 21: parameter gradients pass through grids; `d/dr sum_t exp(-r t) = -sum_t t exp(-r t)`."""
    t = np.array([0.0, 0.5, 2.0])
    g = jax.grad(lambda r: survjax.Exponential(rate=r).survival_grid(f(t)).sum())(
        f(0.8)
    )
    assert_close(g, -np.sum(t * np.exp(-0.8 * t)))


# --------------------------------------------------------------- grid argument (rev 22)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_time_can_be_passed_as_keyword_t(method):
    """spec 5, rev 22: the time vector is "the first positional argument, named `t`"."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    grid = f([0.0, 1.0, 3.0])
    assert_close(getattr(dist, method)(t=grid), getattr(dist, method)(grid))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_scalar_grid_is_rejected(method):
    """spec 5, rev 22: "A `t` that is not one-dimensional (a scalar or a 2-D array) is static misuse and raises `TypeError`"."""
    with pytest.raises(MISUSE):
        getattr(survjax.Exponential(rate=f([0.5, 2.0])), method)(f(1.0))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_two_dimensional_grid_is_rejected(method):
    """spec 5, rev 22: a 2-D `t` raises `TypeError`."""
    with pytest.raises(MISUSE):
        getattr(survjax.Exponential(rate=f([0.5, 2.0])), method)(
            f([[0.0, 1.0], [2.0, 3.0]])
        )


@pytest.mark.step("A5")
def test_column_grid_is_rejected():
    """spec 5, rev 22: a `(K, 1)` grid is 2-D, so it raises `TypeError` even though it would broadcast."""
    with pytest.raises(MISUSE):
        survjax.Exponential(rate=f([0.5, 2.0])).survival_grid(f([[0.0], [1.0]]))


@pytest.mark.step("A5")
def test_two_dimensional_grid_is_rejected_under_jit():
    """spec 5/6.2, rev 22: the shape check is static, so it raises `TypeError` during tracing too."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    with pytest.raises(MISUSE):
        jax.jit(lambda d, g: d.survival_grid(g))(dist, f([[0.0, 1.0], [2.0, 3.0]]))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_boolean_grid_is_rejected(method):
    """spec 5/4.5, rev 22: grid `t` "follows the conversion and dtype rules of section 4.5": boolean `t` raises `TypeError`."""
    with pytest.raises(MISUSE):
        getattr(survjax.Exponential(rate=f(0.5)), method)(np.asarray([True, False]))


@pytest.mark.step("A5")
def test_integer_grid_matches_float_grid():
    """spec 5/4.5, rev 22: an integer grid follows JAX conversion; `S = exp(-r t)` at `t = 0, 1, 2`."""
    dist = survjax.Exponential(rate=f(0.5))
    out = dist.survival_grid(np.asarray([0, 1, 2], dtype=np.int32))
    assert_close(out, np.exp(-0.5 * np.array([0.0, 1.0, 2.0])))
