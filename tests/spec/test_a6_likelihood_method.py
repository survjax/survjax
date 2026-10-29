"""A6: the `dist.log_likelihood` base method (spec 4.2, 6.1, 7.1, 12.1, 12.2; rev 29, 35).

``dist.log_likelihood(data=None, *, time=None, event=None)`` is a base method
on every registered family and, since rev 35, "the only public likelihood
entry point" (7.1). Its receiver takes the place of the predictions in the
section 6.1 input convention. The input forms, static misuse and
transformations are in ``test_a6_likelihood_entry_points.py``; this file
covers availability, keyword-only arguments and the 12.1 agreement of the two
input forms in values and first and second parameter derivatives.

Hand derivations as in the other A6 files: exponential exact ``log r - r t``,
right-censored ``-r t``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import assert_close, b, f, ll

import survjax

EXACT = True
RIGHT = False
TIME = [1.0, 2.0, 4.0]
EVENT = [EXACT, RIGHT, EXACT]
LOG2 = np.log(2.0)


def _dist():
    return survjax.Exponential(rate=f(0.5))


# ------------------------------------------------------------ availability


@pytest.mark.step("A6")
def test_built_in_families_have_the_method():
    """spec 4.2 and 7.1, rev 29: the method "is a base method on every registered family", built-ins included."""
    assert callable(survjax.Exponential(rate=f(0.5)).log_likelihood)
    assert callable(survjax.Weibull(shape=f(1.5), scale=f(2.0)).log_likelihood)


@pytest.mark.step("A6")
def test_custom_family_has_the_method_without_defining_it():
    """spec 4.2, rev 29: a custom family that implements only `log_survival` gets `log_likelihood` from the base."""
    dist = fam.const_hazard()(rate=f([0.5, 0.5]))
    data = survjax.SurvivalData(f([2.0, 2.0]), b([EXACT, RIGHT]))
    assert_close(ll(dist, data), [-LOG2 - 1.0, -1.0])


@pytest.mark.step("A6")
def test_nested_family_has_the_method():
    """spec 4.2 and 7.1, rev 35: a family with a nested distribution field has the method.

    ``scaled_capped`` with ``log_ratio=0`` reproduces its ``capped_rate``
    baseline: right-censored ``log S(2) = -2 r``, exact ``log r - r``.
    """
    base = fam.capped_rate()(rate=f([1.0, 0.5]), cap=f([2.0, 2.0]))
    dist = fam.scaled_capped()(base=base, log_ratio=f([0.0, 0.0]))
    data = survjax.SurvivalData(f([2.0, 1.0]), b([RIGHT, EXACT]))
    assert_close(ll(dist, data), [-2.0, -LOG2 - 0.5])


# ------------------------------------------------------------------- values


@pytest.mark.step("A6")
def test_weibull_per_row_parameters_are_hand_derived():
    """spec 7.1, rev 29: Weibull per-row parameters, hand-derived, through the method.

    ``k=2, b=3, t=1.5`` (``z=0.5``): exact ``log(2/3) + log 0.5 - 0.25``,
    right ``-0.25``.
    """
    dist = survjax.Weibull(shape=f([2.0, 2.0]), scale=f([3.0, 3.0]))
    data = survjax.SurvivalData(f([1.5, 1.5]), b([EXACT, RIGHT]))
    assert_close(ll(dist, data), [np.log(2.0 / 3.0) + np.log(0.5) - 0.25, -0.25])


# ------------------------------------------------------------ signature


@pytest.mark.step("A6")
def test_time_and_event_are_keyword_only():
    """spec 7.1, rev 29: the signature `log_likelihood(data=None, *, time=None, event=None)` makes `time`/`event` keyword-only."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(None, f(TIME), b(EVENT))


# --------------------------------------- derivatives: raw form equals data form
#
# 12.1 (rev 35) requires `dist.log_likelihood(time=..., event=...)` equal to
# `dist.log_likelihood(SurvivalData(time, event))` "in values and first and
# second parameter derivatives". Values are compared exactly; derivatives with
# the 12.1 tolerances (assert_close).

K, B = 1.5, 2.0
W_TIME = [0.5, 1.5, 3.0, 4.0]
W_EVENT = [EXACT, RIGHT, EXACT, RIGHT]


def _weibull_sum(form):
    def total(params):
        dist = survjax.Weibull(shape=params[0], scale=params[1])
        if form == "data":
            return jnp.sum(dist.log_likelihood(survjax.SurvivalData(f(W_TIME), b(W_EVENT))))
        return jnp.sum(dist.log_likelihood(time=f(W_TIME), event=b(W_EVENT)))

    return total


def _params():
    return jnp.asarray(f([K, B]))


@pytest.mark.step("A6")
def test_raw_form_value_equals_data_form_value():
    """spec 6.1 and 12.1, rev 35: Weibull summed value, identical in the raw and the `SurvivalData` form."""
    raw = np.asarray(_weibull_sum("raw")(_params()))
    data = np.asarray(_weibull_sum("data")(_params()))
    assert np.isfinite(raw)
    np.testing.assert_array_equal(raw, data)


@pytest.mark.step("A6")
def test_raw_form_per_row_values_equal_data_form():
    """spec 6.1 and 12.1, rev 35: per-row Weibull contributions, identical in the two input forms."""
    dist = survjax.Weibull(shape=f(K), scale=f(B))
    raw = ll(dist, time=f(W_TIME), event=b(W_EVENT))
    np.testing.assert_array_equal(raw, ll(dist, survjax.SurvivalData(f(W_TIME), b(W_EVENT))))


@pytest.mark.step("A6")
def test_raw_form_gradient_equals_data_form_gradient():
    """spec 6.1 and 12.1, rev 35: first parameter derivatives agree between the raw and the `SurvivalData` form."""
    raw = np.asarray(jax.grad(_weibull_sum("raw"))(_params()))
    data = np.asarray(jax.grad(_weibull_sum("data"))(_params()))
    assert np.isfinite(raw).all()
    assert_close(raw, data)


@pytest.mark.step("A6")
def test_raw_form_hessian_equals_data_form_hessian():
    """spec 6.1 and 12.1, rev 35: second parameter derivatives agree between the raw and the `SurvivalData` form."""
    raw = np.asarray(jax.hessian(_weibull_sum("raw"))(_params()))
    data = np.asarray(jax.hessian(_weibull_sum("data"))(_params()))
    assert raw.shape == (2, 2)
    assert np.isfinite(raw).all()
    assert_close(raw, data)
