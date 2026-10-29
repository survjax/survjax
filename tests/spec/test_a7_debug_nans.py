"""A7: the likelihood run eagerly under `jax_debug_nans` on valid mixed batches (spec 7.2, 12.1, 12.2; rev 33).

Spec 12.1 ("Numerical behaviour"): "the likelihood suite run eagerly under
`jax_debug_nans` on valid mixed batches"; 12.2 lists the "eager
`jax_debug_nans` run" in the Slice A gate and (rev 33) limits Slice A to exact
and right-censored rows without truncation. Spec 7.2 requires safe inputs
"selected before each operation, including inside helpers and unselected
branches", so no primitive may produce a NaN, in values or in parameter
gradients, on valid data with valid parameters. ``jax_debug_nans`` raises on
the first NaN any primitive produces, so these tests pass only if no branch,
active or not, produces one.

Every batch mixes these valid rows (``Z = 1``):

====  ========  =================================================
row   kind      purpose
====  ========  =================================================
0     exact     interior
1     exact     tiny time ``1e-30`` (lower tail)
2     exact     upper tail (``f`` underflows)
3     right     at ``0`` (uninformative, exactly zero)
4     right     tiny ``H``
5     right     interior
6     right     upper tail (``S`` underflows)
====  ========  =================================================

Values are also checked: finite everywhere, exactly zero on row 3, and, for
the exponential, equal to the hand values ``log r - r t`` / ``-r c``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    assert_tail_close,
    b,
    f,
    mp_exponential_contribution,
)

import survjax

KINDS = b([True, True, True, False, False, False, False])


def _times(upper_tail):
    """The seven row times; ``upper_tail`` is a time where ``H`` is about 1e3 to 1e4 for the family."""
    return f([1.5, 1e-30, upper_tail, 0.0, 1e-15, 2.0, upper_tail])


def _check(values, grads):
    values = np.asarray(values)
    assert np.isfinite(values).all(), values
    np.testing.assert_array_equal(values[3], 0.0)
    for g in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(g)).all(), g


def _run(make, params, times):
    """Values and the gradient of the summed contributions, eagerly, under `jax_debug_nans`."""
    data = survjax.SurvivalData(times, KINDS)

    def total(p):
        dist = make(**p)
        out = dist.log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, has_aux=True)(params)
    _check(values, grads)
    return np.asarray(values)


@pytest.mark.step("A7")
def test_exponential_mixed_batch_under_debug_nans():
    """spec 7.2/12.1, rev 33: Exponential `r=2`: no NaN in values or the rate gradient; values match `log r - r t`, `-r c`."""
    times = _times(1e3)
    values = _run(survjax.Exponential, {"rate": jnp.asarray(f(2.0))}, times)
    for i, (t, e) in enumerate(zip(np.asarray(times), np.asarray(KINDS), strict=True)):
        if i == 3:
            continue
        expected, rtol = mp_exponential_contribution(2.0, t, exact=bool(e))
        assert_tail_close(values[i], expected, rtol, err_msg=f"row {i}")


@pytest.mark.step("A7")
def test_exponential_per_row_rates_under_debug_nans():
    """spec 5/7.2/12.1, rev 33: per-row rates spanning `1e-3` to `1e3` raise no NaN in values or gradients."""
    rates = jnp.asarray(f([1e-3, 0.5, 1.0, 2.0, 1e3, 1.0, 10.0]))
    _run(survjax.Exponential, {"rate": rates}, _times(1e3))


@pytest.mark.step("A7")
@pytest.mark.parametrize("k", [0.3, 0.5, 1.0, 2.5, 8.0])
def test_weibull_mixed_batch_under_debug_nans(k):
    """spec 7.2/12.1, rev 33: Weibull `b=2` and shape `k`: no NaN in values or the `k` and `b` gradients.

    The upper-tail time is ``b * 1e3**(1/k)`` (``H = 1e3``).
    """
    params = {"shape": jnp.asarray(f(k)), "scale": jnp.asarray(f(2.0))}
    _run(survjax.Weibull, params, _times(2.0 * 1e3 ** (1.0 / k)))


@pytest.mark.step("A7")
def test_weibull_per_row_parameters_under_debug_nans():
    """spec 5/7.2/12.1, rev 33: per-row shapes and scales raise no NaN in values or gradients."""
    params = {
        "shape": jnp.asarray(f([0.5, 0.5, 2.0, 0.5, 3.0, 1.0, 2.0])),
        "scale": jnp.asarray(f([1.0, 2.0, 1.0, 1e3, 1.0, 1e-2, 1.0])),
    }
    _run(survjax.Weibull, params, _times(100.0))


@pytest.mark.step("A7")
def test_weibull_from_log_under_debug_nans():
    """spec 4.1/7.2/12.1, rev 33: `Weibull.from_log` with gradients in the log parameters raises no NaN."""
    params = {"log_shape": jnp.asarray(f(np.log(1.5))), "log_scale": jnp.asarray(f(np.log(2.0)))}
    _run(survjax.Weibull.from_log, params, _times(2.0 * 1e3 ** (1.0 / 1.5)))


@pytest.mark.step("A7")
def test_custom_family_with_generic_defaults_under_debug_nans():
    """spec 4.2/7.2/12.1, rev 33: a custom family with only `log_survival` and `reference_time` raises no NaN (`ConstHazard`, `rate=2`)."""
    _run(fam.const_hazard(), {"rate": jnp.asarray(f(2.0))}, _times(1e3))


@pytest.mark.step("A7")
def test_improper_family_under_debug_nans():
    """spec 2/4.2/7.2/12.1, rev 33: the improper `mixture_cure` family raises no NaN.

    Its exact upper-tail row uses ``t=20`` (``rate t = 10``): the generic
    ``log_prob`` default carries no tail guarantee for this family, so the
    exact rows stay where ``f`` is representable; right-censored rows go to
    ``t=20`` as well.
    """
    params = {"cure": jnp.asarray(f(0.3)), "rate": jnp.asarray(f(0.5))}
    _run(fam.mixture_cure(), params, _times(20.0))


@pytest.mark.step("A7")
def test_bounds_form_batch_under_debug_nans():
    """spec 3.1/3.2/7.2/12.1, rev 33: bounds-form data with exact rows and `(c, +inf]` rows, including `(0, +inf]`, raise no NaN.

    Rev 33 classifies ``(0, +inf]`` as right-censored, so it is a supported,
    uninformative row contributing exactly zero.
    """
    data = survjax.SurvivalData.from_bounds(
        f([1.5, 1e-30, 0.0, 2.0, 1e3]),
        f([1.5, 1e-30, np.inf, np.inf, np.inf]),
        exact=b([True, True, False, False, False]),
    )

    def total(k, s):
        out = survjax.Weibull(shape=k, scale=s).log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, argnums=(0, 1), has_aux=True)(
            jnp.asarray(f(0.5)), jnp.asarray(f(2.0))
        )
    values = np.asarray(values)
    assert np.isfinite(values).all()
    np.testing.assert_array_equal(values[2], 0.0)
    for g in grads:
        assert np.isfinite(np.asarray(g))
