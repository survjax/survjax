"""A6: exact and right-censored likelihood values without truncation (spec 2, 4.1, 7.1; rev 28, 29).

Without truncation ``Z = 1`` (spec 2), so an exact row contributes
``log f(t)`` and a right-censored row ``log S(c)``. Hand derivations:

- Exponential, rate ``r``: ``log f = log r - r t``; ``log S = -r t``.
- Weibull, shape ``k``, scale ``b``, ``z = t/b``:
  ``log f = log k - log b + (k-1) log z - z**k``; ``log S = -z**k``.

Tails, boundaries, mpmath and ``jax_debug_nans`` checks are A7; R fixtures A8;
the derivative suite A9.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    X64,
    assert_close,
    b,
    f,
    ll,
    nan_causes,
)

import survjax

EXACT = True
RIGHT = False
LOG2 = np.log(2.0)


def _no_issues(dist, data, value):
    """No NaN row and no cause in the 7.1 triage recipe (rev 29)."""
    assert not np.isnan(value).any()
    for mask in nan_causes(dist, data, value):
        assert not mask.any()


# ------------------------------------------------------------- exponential


@pytest.mark.step("A6")
def test_exponential_exact_contribution():
    """spec 2 and 7.1, rev 28: an exact row contributes `log f(t)`; exponential `r=0.5, t=2`: `log 0.5 - 1`."""
    value = ll(survjax.Exponential(rate=f(0.5)), survjax.SurvivalData(f(2.0), b(EXACT)))
    assert_close(value, -LOG2 - 1.0)


@pytest.mark.step("A6")
def test_exponential_right_censored_contribution():
    """spec 2 and 7.1, rev 28: a right-censored row contributes `log S(c)`; exponential `r=0.5, c=2`: `-1`."""
    value = ll(survjax.Exponential(rate=f(0.5)), survjax.SurvivalData(f(2.0), b(RIGHT)))
    assert_close(value, -1.0)


@pytest.mark.step("A6")
def test_exponential_mixed_rows_with_per_row_rates():
    """spec 7.1, rev 28: per-observation contributions for mixed exact and right-censored rows, no implicit sum."""
    rate = f([0.5, 2.0, 1.0, 0.25])
    data = survjax.SurvivalData(f([2.0, 0.5, 3.0, 8.0]), b([EXACT, RIGHT, RIGHT, EXACT]))
    dist = survjax.Exponential(rate=rate)
    value = ll(dist, data)
    # exact: log r - r t; right: -r t
    expected = [-LOG2 - 1.0, -1.0, -3.0, np.log(0.25) - 2.0]
    assert value.shape == (4,)
    assert_close(value, expected)
    _no_issues(dist, data, value)


@pytest.mark.step("A6")
def test_exponential_shared_rate_over_rows():
    """spec 5 and 7.1, rev 28: a scalar distribution against data `(N,)` gives one contribution per row."""
    data = survjax.SurvivalData(f([1.0, 2.0, 4.0]), b([EXACT, RIGHT, EXACT]))
    value = ll(survjax.Exponential(rate=f(0.5)), data)
    assert_close(value, [-LOG2 - 0.5, -1.0, -LOG2 - 2.0])


@pytest.mark.step("A6")
def test_exponential_from_log_matches_ordinary_constructor():
    """spec 4.1, rev 28: `from_log` values equal the ordinary constructor's, here through the likelihood."""
    data = survjax.SurvivalData(f([1.0, 2.0]), b([EXACT, RIGHT]))
    by_log = ll(survjax.Exponential.from_log(log_rate=f([-LOG2, 0.0])), data)
    direct = ll(survjax.Exponential(rate=f([0.5, 1.0])), data)
    assert_close(by_log, direct)
    assert_close(by_log, [-LOG2 - 0.5, -2.0])


# ----------------------------------------------------------------- Weibull


@pytest.mark.step("A6")
def test_weibull_exact_contribution_shape_above_one():
    """spec 4.1 and 7.1, rev 28: Weibull `k=2, b=3, t=1.5` (`z=0.5`): `log(2/3) + log 0.5 - 0.25`."""
    dist = survjax.Weibull(shape=f(2.0), scale=f(3.0))
    value = ll(dist, survjax.SurvivalData(f(1.5), b(EXACT)))
    assert_close(value, np.log(2.0 / 3.0) + np.log(0.5) - 0.25)


@pytest.mark.step("A6")
def test_weibull_right_censored_contribution_shape_above_one():
    """spec 4.1 and 7.1, rev 28: Weibull `k=2, b=3, c=1.5`: `log S = -0.25`."""
    dist = survjax.Weibull(shape=f(2.0), scale=f(3.0))
    value = ll(dist, survjax.SurvivalData(f(1.5), b(RIGHT)))
    assert_close(value, -0.25)


@pytest.mark.step("A6")
def test_weibull_exact_contribution_shape_below_one():
    """spec 4.1 and 7.1, rev 28: Weibull `k=0.5, b=2, t=8` (`z=4`): `log 0.25 - 0.5 log 4 - 2 = -3 log 2 - 2`."""
    dist = survjax.Weibull(shape=f(0.5), scale=f(2.0))
    value = ll(dist, survjax.SurvivalData(f(8.0), b(EXACT)))
    assert_close(value, -3.0 * LOG2 - 2.0)


@pytest.mark.step("A6")
def test_weibull_right_censored_contribution_shape_below_one():
    """spec 4.1 and 7.1, rev 28: Weibull `k=0.5, b=2, c=8`: `log S = -sqrt(4) = -2`."""
    dist = survjax.Weibull(shape=f(0.5), scale=f(2.0))
    value = ll(dist, survjax.SurvivalData(f(8.0), b(RIGHT)))
    assert_close(value, -2.0)


@pytest.mark.step("A6")
def test_weibull_shape_one_equals_exponential():
    """spec 4.1 and 7.1, rev 28: Weibull `k=1, b=1/r` is exponential `r`, row by row."""
    data = survjax.SurvivalData(f([0.5, 1.0, 3.0, 7.0]), b([EXACT, RIGHT, EXACT, RIGHT]))
    weibull = ll(survjax.Weibull(shape=f(1.0), scale=f(2.0)), data)
    exponential = ll(survjax.Exponential(rate=f(0.5)), data)
    assert_close(weibull, exponential)


@pytest.mark.step("A6")
def test_weibull_mixed_rows_with_per_row_parameters():
    """spec 7.1, rev 28: Weibull mixed rows with per-row shape and scale, hand-derived."""
    dist = survjax.Weibull(shape=f([2.0, 2.0, 0.5, 0.5]), scale=f([3.0, 3.0, 2.0, 2.0]))
    data = survjax.SurvivalData(f([1.5, 1.5, 8.0, 8.0]), b([EXACT, RIGHT, EXACT, RIGHT]))
    value = ll(dist, data)
    expected = [np.log(2.0 / 3.0) + np.log(0.5) - 0.25, -0.25, -3.0 * LOG2 - 2.0, -2.0]
    assert_close(value, expected)
    _no_issues(dist, data, value)


# ------------------------------------------------------------ custom family


@pytest.mark.step("A6")
def test_custom_family_uses_generic_defaults():
    """spec 4.2 and 7.1, rev 28: "The likelihood never dispatches on family type": a custom family with only `log_survival` works."""
    dist = fam.const_hazard()(rate=f([0.5, 0.5]))
    data = survjax.SurvivalData(f([2.0, 2.0]), b([EXACT, RIGHT]))
    value = ll(dist, data)
    assert_close(value, [-LOG2 - 1.0, -1.0])
    _no_issues(dist, data, value)


@pytest.mark.step("A6")
def test_custom_family_matches_built_in():
    """spec 4.2 and 7.1, rev 28: a custom Weibull-like family gives the built-in Weibull likelihood."""
    data = survjax.SurvivalData(f([0.3, 1.5, 4.0, 9.0]), b([EXACT, RIGHT, EXACT, RIGHT]))
    custom = ll(fam.power_hazard()(shape=f(1.7), scale=f(2.5)), data)
    built_in = ll(survjax.Weibull(shape=f(1.7), scale=f(2.5)), data)
    assert_close(custom, built_in)


@pytest.mark.step("A6")
def test_log_prob_override_is_used_for_exact_rows():
    """spec 4.2 and 7.1, rev 28: the exact-row density comes from the family's `log_prob`, override included."""
    dist = fam.log_prob_override()(rate=f(0.5))
    value = ll(dist, survjax.SurvivalData(f([2.0, 2.0]), b([EXACT, RIGHT])))
    assert_close(value, [fam.OVERRIDE_SENTINEL, -1.0])


# ------------------------------------------------- zero and uninformative rows


@pytest.mark.step("A6")
def test_right_censored_at_zero_contributes_exactly_zero():
    """spec 2, 3.3 and 7.1, rev 28: right-censored at `c=L=0` covers the window: valid, `uninformative`, exactly zero."""
    data = survjax.SurvivalData(f([0.0, 2.0]), b([RIGHT, EXACT]))
    assert bool(np.asarray(data.issues.uninformative)[0])
    dist = survjax.Weibull(shape=f(1.5), scale=f(2.0))
    value = ll(dist, data)
    assert value[0] == 0.0
    assert np.isfinite(value[1])
    _no_issues(dist, data, value)


@pytest.mark.step("A6")
def test_bounds_form_zero_to_infinity_contributes_exactly_zero():
    """spec 3.3 and 7.1, rev 28: `from_bounds(0, inf)` without truncation is uninformative and contributes exactly zero."""
    data = survjax.SurvivalData.from_bounds(f([0.0]), f([np.inf]))
    dist = survjax.Exponential(rate=f(3.0))
    value = ll(dist, data)
    np.testing.assert_array_equal(value, [0.0])
    _no_issues(dist, data, value)


@pytest.mark.step("A6")
def test_uninformative_row_has_zero_gradient():
    """spec 4.5 and 7.1, rev 28: the full-coverage zero is structural (`log S(0)=0` from the base), so its parameter gradient is zero."""
    data = survjax.SurvivalData(f(0.0), b(RIGHT))

    def value(k, s):
        return survjax.Weibull(shape=k, scale=s).log_likelihood(data)

    grads = jax.grad(value, argnums=(0, 1))(jnp.asarray(f(1.5)), jnp.asarray(f(2.0)))
    for g in grads:
        np.testing.assert_array_equal(np.asarray(g), 0.0)


@pytest.mark.step("A6")
def test_zero_density_is_minus_infinity_not_nan():
    """spec 7.1, rev 28: "true zero model mass or density, positive `Z`" gives `-inf`, "not automatically an error".

    ``bounded_support`` with ``end=2``: the density is zero for ``t > 2``.
    """
    dist = fam.bounded_support()(end=f(2.0))
    data = survjax.SurvivalData(f([1.0, 3.0]), b([EXACT, EXACT]))
    value = ll(dist, data)
    assert_close(value[0], np.log(0.5))  # f(t) = 1/end inside the support
    assert value[1] == -np.inf
    _no_issues(dist, data, value)


@pytest.mark.step("A6")
def test_zero_survival_is_minus_infinity_not_nan():
    """spec 7.1, rev 28: right-censored at or beyond a finite support end has zero mass: `-inf`, no issue."""
    dist = fam.bounded_support()(end=f(2.0))
    data = survjax.SurvivalData(f([1.0, 2.0, 5.0]), b([RIGHT, RIGHT, RIGHT]))
    value = ll(dist, data)
    assert_close(value[0], np.log(0.5))  # S(1) = 1 - 1/2
    np.testing.assert_array_equal(value[1:], [-np.inf, -np.inf])
    _no_issues(dist, data, value)


@pytest.mark.step("A6")
def test_all_censored_exponential_data_is_valid():
    """spec 12.1, rev 28: "all-censored exponential data remain structurally valid despite no finite log-rate MLE"."""
    data = survjax.SurvivalData(f([1.0, 2.0, 3.0]), b([RIGHT, RIGHT, RIGHT]))
    assert bool(np.asarray(data.issues.ok))
    dist = survjax.Exponential(rate=f(0.5))
    value = ll(dist, data)
    assert_close(value, [-0.5, -1.0, -1.5])
    _no_issues(dist, data, value)


# --------------------------------------------------------------------- dtype


@pytest.mark.step("A6")
def test_likelihood_dtype_is_default_float_in_32bit_mode():
    """spec 3.3, rev 28: calculations "use the common promoted data/parameter dtype"; float32 in 32-bit mode."""
    if X64:
        pytest.skip("32-bit mode only")
    value = survjax.Exponential(rate=f([0.5])).log_likelihood(survjax.SurvivalData(f([2.0]), b([EXACT]))
    )
    assert value.dtype == jnp.float32


@pytest.mark.step("A6")
def test_likelihood_promotes_float32_parameters_with_float64_data():
    """spec 3.3, rev 28: float32 parameters with float64 data give a float64 likelihood (64-bit mode)."""
    if not X64:
        pytest.skip("64-bit mode only")
    dist = survjax.Exponential(rate=np.asarray([0.5], dtype=np.float32))
    data = survjax.SurvivalData(np.asarray([2.0], dtype=np.float64), b([EXACT]))
    value = dist.log_likelihood(data)
    assert value.dtype == jnp.float64
    assert_close(np.asarray(value), [-LOG2 - 1.0])


@pytest.mark.step("A6")
def test_likelihood_promotes_float32_data_with_float64_parameters():
    """spec 3.3, rev 28: float32 data with float64 parameters give a float64 likelihood (64-bit mode)."""
    if not X64:
        pytest.skip("64-bit mode only")
    dist = survjax.Exponential(rate=np.asarray([0.5], dtype=np.float64))
    data = survjax.SurvivalData(np.asarray([2.0], dtype=np.float32), b([RIGHT]))
    value = dist.log_likelihood(data)
    assert value.dtype == jnp.float64
    assert_close(np.asarray(value), [-1.0])
