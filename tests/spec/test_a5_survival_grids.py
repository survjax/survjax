"""A5: `_grid` twins and the direct-method shape rules (spec 5, 8.1).

Spec 5, rev 24: "Every pointwise direct method of `t` (`survival`,
`log_survival`, `log_prob`, `log_cdf`, `cumulative_hazard`, `log_hazard`,
`hazard`) has a grid twin named with a `_grid` suffix ... For `t` of shape
`(K,)`, `m_grid(t)` equals the direct method `m` evaluated at `t` reshaped to
`(K, 1, ..., 1)`, with one trailing 1 per batch axis, and returns shape
`(K, *batch_shape)` ... Component axes never appear in outputs. The base
provides the grid methods for every registered family; families do not
implement them." The time vector is the first positional argument, named `t`,
follows the conversion, dtype and value rules of section 4.5, and a `t` that is
not one-dimensional raises `TypeError`. "`t` need not be sorted or distinct;
output row `i` belongs to `t[i]`."

Hand values: Exponential ``S = exp(-r t)``, ``H = r t``, ``h = r``; Weibull
``S = exp(-(t/b)**k)``, ``H = (t/b)**k``, ``h = (k/b)(t/b)**(k-1)``, so
``log S = -(t/b)**k``, ``log F = log(-expm1(-(t/b)**k))``,
``log h = log(k/b) + (k-1) log(t/b)`` and ``log f = log h + log S``.
"""

import warnings

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, MISUSE, NAN, X64, assert_close, f

import survjax

# Spec 5, rev 24: every pointwise direct method of `t` has a `_grid` twin.
DIRECT_METHODS = (
    "survival",
    "log_survival",
    "log_prob",
    "log_cdf",
    "cumulative_hazard",
    "log_hazard",
    "hazard",
)
GRID_METHODS = {m + "_grid": m for m in DIRECT_METHODS}

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
    """spec 5, rev 24: time-first output; a scalar batch gives `(K,)`."""
    out = getattr(survjax.Exponential(rate=f(0.5)), method)(f([0.5, 1.0, 2.0, 4.0]))
    assert np.shape(out) == (4,)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_on_batch_is_time_first(method):
    """spec 5, rev 24: time-first output; batch `(N,)` gives `(K,N)`."""
    out = getattr(survjax.Exponential(rate=f([0.5, 1.0, 2.0])), method)(
        f([0.5, 1.0, 2.0, 4.0])
    )
    assert np.shape(out) == (4, 3)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_on_two_dimensional_batch_gives_k_d_n(method):
    """spec 5, rev 24: "Shared grid | time `(K,)`, distribution `(D,N)` | `(K,D,N)`"."""
    dist = survjax.Weibull(shape=f([[1.0], [2.0]]), scale=f([1.0, 2.0, 3.0]))
    out = getattr(dist, method)(f([0.5, 1.0, 2.0, 4.0]))
    assert np.shape(out) == (4, 2, 3)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_equals_direct_method_on_grid_with_new_axes(method):
    """spec 5, rev 24: "for batch `(D,N)` this is `m(t[:, None, None])`, shape `(K,D,N)`"."""
    dist = survjax.Weibull(shape=f([[0.7], [1.0], [2.5]]), scale=f([0.5, 2.0]))
    grid = f([0.1, 0.5, 1.0, 3.0, 10.0])
    direct = getattr(dist, GRID_METHODS[method])(grid[:, None, None])
    out = getattr(dist, method)(grid)
    assert np.shape(out) == (5, 3, 2)
    assert_close(out, direct)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_with_length_equal_to_batch_equals_direct_method_on_column(method):
    """spec 5, rev 24: with `K=N`, `m_grid(t)` is still `m(t[:, None])`, shape `(K,N)`, not the paired `m(t)`."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 4.0]))
    grid = f([0.5, 1.0, 3.0])
    out = getattr(dist, method)(grid)
    assert np.shape(out) == (3, 3)
    assert_close(out, getattr(dist, GRID_METHODS[method])(grid[:, None]))


@pytest.mark.step("A5")
def test_grid_with_length_equal_to_batch_is_not_paired():
    """spec 5, rev 21: a grid is shared, not paired: `K=N=3` gives `(3,3)` with `out[i,j] = S_j(t_i)`."""
    r, t = np.array([0.5, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])
    out = survjax.Exponential(rate=f(r)).survival_grid(f(t))
    assert np.shape(out) == (3, 3)
    assert_close(out, np.exp(-t[:, None] * r[None, :]))


@pytest.mark.step("A5")
def test_log_prob_grid_with_length_equal_to_batch_is_not_paired():
    """spec 5, rev 24: `log_prob_grid` with `K=N=3` gives `(3,3)` with `out[i,j] = log f_j(t_i) = log r_j - r_j t_i`."""
    r, t = np.array([0.5, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])
    out = survjax.Exponential(rate=f(r)).log_prob_grid(f(t))
    assert np.shape(out) == (3, 3)
    assert_close(out, np.log(r)[None, :] - t[:, None] * r[None, :])


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_accepts_unsorted_repeated_times(method):
    """spec 5, rev 24: "`t` need not be sorted or distinct; output row `i` belongs to `t[i]`"."""
    dist = survjax.Weibull(shape=f([0.5, 2.0]), scale=f([1.0, 3.0]))
    grid = np.array([3.0, 0.5, 3.0, 1.0])
    out = np.asarray(getattr(dist, method)(f(grid)))
    order = np.argsort(grid, kind="stable")
    assert np.shape(out) == (4, 2)
    assert_close(out[order], getattr(dist, method)(f(grid[order])))
    assert_close(out[0], out[2])


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


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_custom_family_with_only_log_survival_has_every_grid(method):
    """spec 4.2/5, rev 24: "The base provides the grid methods for every registered family; families do not implement them".

    ``ConstHazard`` defines only ``log_survival = -rate * t``; each grid equals
    its direct method on ``t[:, None]`` (the direct methods use the generic
    defaults of section 4.2).
    """
    dist = fam.const_hazard()(rate=f([0.5, 2.0]))
    grid = f([0.25, 1.0, 3.0])
    out = getattr(dist, method)(grid)
    assert np.shape(out) == (3, 2)
    assert_close(out, getattr(dist, GRID_METHODS[method])(grid[:, None]))


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
def test_weibull_log_survival_grid_values():
    """spec 5/8.1, rev 24: `log_survival_grid` values `-(t/b)**k`, time-first."""
    k, b = np.array([0.5, 1.0, 3.0]), np.array([2.0, 1.0, 4.0])
    t = np.array([0.0, 0.5, 2.0, 8.0])
    out = survjax.Weibull(shape=f(k), scale=f(b)).log_survival_grid(f(t))
    assert_close(out, -((t[:, None] / b) ** k))


@pytest.mark.step("A5")
def test_weibull_log_prob_grid_values():
    """spec 5/8.1, rev 24: `log_prob_grid` values `log(k/b) + (k-1) log(t/b) - (t/b)**k` on positive times, time-first."""
    k, b = np.array([0.5, 1.0, 3.0]), np.array([2.0, 1.0, 4.0])
    t = np.array([0.5, 2.0, 8.0])
    z = t[:, None] / b
    out = survjax.Weibull(shape=f(k), scale=f(b)).log_prob_grid(f(t))
    assert_close(out, np.log(k / b) + (k - 1.0) * np.log(z) - z**k)


@pytest.mark.step("A5")
def test_weibull_log_cdf_grid_values():
    """spec 5/8.1, rev 24: `log_cdf_grid` values `log(-expm1(-(t/b)**k))` on positive times, time-first."""
    k, b = np.array([0.5, 1.0, 3.0]), np.array([2.0, 1.0, 4.0])
    t = np.array([0.5, 2.0, 8.0])
    out = survjax.Weibull(shape=f(k), scale=f(b)).log_cdf_grid(f(t))
    assert_close(out, np.log(-np.expm1(-((t[:, None] / b) ** k))))


@pytest.mark.step("A5")
def test_weibull_log_hazard_grid_values():
    """spec 5/8.1, rev 24: `log_hazard_grid` values `log(k/b) + (k-1) log(t/b)` on positive times, time-first."""
    k, b = np.array([0.5, 1.0, 3.0]), np.array([2.0, 1.0, 4.0])
    t = np.array([0.5, 2.0, 8.0])
    out = survjax.Weibull(shape=f(k), scale=f(b)).log_hazard_grid(f(t))
    assert_close(out, np.log(k / b) + (k - 1.0) * np.log(t[:, None] / b))


@pytest.mark.step("A5")
def test_exponential_grids_values():
    """spec 5/8.1, rev 21: Exponential grids `exp(-r t)`, `r t` and `r`."""
    r, t = np.array([0.5, 2.0]), np.array([0.0, 1.0, 3.0])
    dist = survjax.Exponential(rate=f(r))
    assert_close(dist.survival_grid(f(t)), np.exp(-t[:, None] * r))
    assert_close(dist.cumulative_hazard_grid(f(t)), t[:, None] * r)
    assert_close(dist.hazard_grid(f(t)), np.broadcast_to(r, (3, 2)))


@pytest.mark.step("A5")
def test_exponential_log_grids_values():
    """spec 5/8.1, rev 24: Exponential log grids `-r t`, `log r - r t`, `log(-expm1(-r t))` and `log r`."""
    r, t = np.array([0.5, 2.0]), np.array([0.5, 1.0, 3.0])
    dist = survjax.Exponential(rate=f(r))
    assert_close(dist.log_survival_grid(f(t)), -t[:, None] * r)
    assert_close(dist.log_prob_grid(f(t)), np.log(r) - t[:, None] * r)
    assert_close(dist.log_cdf_grid(f(t)), np.log(-np.expm1(-t[:, None] * r)))
    assert_close(dist.log_hazard_grid(f(t)), np.broadcast_to(np.log(r), (3, 2)))


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


@pytest.mark.step("A5")
def test_log_prob_grid_from_zero_with_weibull_shape_one_is_finite():
    """spec 4.5/5, rev 24: `k=1`, `log_prob_grid` from `t=0`: `log f = -log b - t/b`, finite throughout."""
    b = np.array([10.0, 2.0])
    t = np.array([0.0, 1.0, 5.0])
    out = survjax.Weibull(shape=f(1.0), scale=f(b)).log_prob_grid(f(t))
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
def test_log_survival_grid_applies_value_policy():
    """spec 4.5/5, rev 24: `log_survival_grid`: `t<0` `0`, `+inf` `-inf` (proper family), NaN NaN."""
    out = survjax.Weibull(shape=f([0.5, 2.0]), scale=f(2.0)).log_survival_grid(
        f([-1.0, INF, NAN])
    )
    assert_close(out, [[0.0, 0.0], [-INF, -INF], [NAN, NAN]])


@pytest.mark.step("A5")
def test_log_prob_grid_applies_value_policy():
    """spec 4.5/5, rev 24: `log_prob_grid`: `t<0` `-inf`, `+inf` `-inf`, NaN NaN."""
    out = survjax.Weibull(shape=f([0.5, 2.0]), scale=f(2.0)).log_prob_grid(
        f([-1.0, INF, NAN])
    )
    assert_close(out, [[-INF, -INF], [-INF, -INF], [NAN, NAN]])


@pytest.mark.step("A5")
def test_log_cdf_grid_applies_value_policy():
    """spec 4.5/5, rev 24: `log_cdf_grid`: `t<0` `-inf`; `+inf` `log F = log(1 - 0) = 0`; NaN NaN."""
    out = survjax.Weibull(shape=f([0.5, 2.0]), scale=f(2.0)).log_cdf_grid(
        f([-1.0, INF, NAN])
    )
    assert_close(out, [[-INF, -INF], [0.0, 0.0], [NAN, NAN]])


@pytest.mark.step("A5")
def test_log_hazard_grid_applies_value_policy():
    """spec 4.5/5, rev 24: `log_hazard_grid`: `t<0` `-inf`; at `+inf` the analytic limit (Weibull `k=0.5`: `-inf`; `k=2`: `+inf`); NaN NaN."""
    out = survjax.Weibull(shape=f([0.5, 2.0]), scale=f(2.0)).log_hazard_grid(
        f([-1.0, INF, NAN])
    )
    assert_close(out, [[-INF, -INF], [-INF, INF], [NAN, NAN]])


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
    """spec 12.1, rev 24: eager and `jit` agree for grids."""
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


# --------------------------------------------------------------- grid argument (rev 22, rev 24)


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_grid_time_can_be_passed_as_keyword_t(method):
    """spec 5, rev 24: the time vector is "the first positional argument, named `t`"."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    grid = f([0.0, 1.0, 3.0])
    assert_close(getattr(dist, method)(t=grid), getattr(dist, method)(grid))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_scalar_grid_is_rejected(method):
    """spec 5, rev 24: "A `t` that is not one-dimensional (a scalar or a 2-D array) is static misuse and raises `TypeError`"."""
    with pytest.raises(MISUSE):
        getattr(survjax.Exponential(rate=f([0.5, 2.0])), method)(f(1.0))


@pytest.mark.step("A5")
@pytest.mark.parametrize("method", list(GRID_METHODS))
def test_two_dimensional_grid_is_rejected(method):
    """spec 5, rev 24: a 2-D `t` raises `TypeError`."""
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
    """spec 5/4.5, rev 24: grid `t` "follows the conversion and dtype rules of section 4.5": boolean `t` raises `TypeError`."""
    with pytest.raises(MISUSE):
        getattr(survjax.Exponential(rate=f(0.5)), method)(np.asarray([True, False]))


@pytest.mark.step("A5")
def test_integer_grid_matches_float_grid():
    """spec 5/4.5, rev 22: an integer grid follows JAX conversion; `S = exp(-r t)` at `t = 0, 1, 2`."""
    dist = survjax.Exponential(rate=f(0.5))
    out = dist.survival_grid(np.asarray([0, 1, 2], dtype=np.int32))
    assert_close(out, np.exp(-0.5 * np.array([0.0, 1.0, 2.0])))


# --------------------------------------------------------------- base-supplied boundaries (rev 26)


@pytest.mark.step("A5")
def test_survival_grid_of_naive_log_logistic_uses_base_boundary_values():
    """spec 4.5/5, rev 26: a grid through `t=0` and `t=+inf` for the plain softplus log-logistic: `(1, 1/2, 0)`.

    ``alpha=2``, ``beta=3``: ``S(0)=1`` and ``S(+inf)=0`` are base-supplied,
    ``S(2) = 1/(1+1) = 1/2``. The gradient of the sum in ``beta`` is that of
    the interior entry alone, ``dS(2)/dbeta = -u log(t/alpha)/(1+u)**2 = 0``
    (``u = 1``, ``log 1 = 0``), and is finite (not NaN from the boundaries).
    """

    def total(bb):
        dist = fam.softplus_log_logistic()(alpha=f(2.0), beta=bb)
        return dist.survival_grid(f([0.0, 2.0, INF])).sum()

    dist = fam.softplus_log_logistic()(alpha=f(2.0), beta=f(3.0))
    assert_close(dist.survival_grid(f([0.0, 2.0, INF])), [1.0, 0.5, 0.0])
    np.testing.assert_array_equal(np.asarray(jax.grad(total)(f(3.0))), 0.0)
