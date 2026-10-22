"""A6: the `dist.log_likelihood` base method (spec 4.2, 6.1, 7.1, 12.1, 12.2; rev 29).

``dist.log_likelihood(data=None, *, time=None, event=None)`` is a base method
on every registered family and "returns exactly what the function returns for
the same arguments, values and derivatives alike" (7.1). Its receiver takes the
place of the predictions in the section 6.1 input convention.

Hand derivations as in the other A6 files: exponential exact ``log r - r t``,
right-censored ``-r t``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import assert_close, b, f, ll, ll_method

import survjax

EXACT = True
RIGHT = False
TIME = [1.0, 2.0, 4.0]
EVENT = [EXACT, RIGHT, EXACT]
LOG2 = np.log(2.0)
EXPECTED = [-LOG2 - 0.5, -1.0, -LOG2 - 2.0]  # r = 0.5


def _dist():
    return survjax.Exponential(rate=f(0.5))


def _data():
    return survjax.SurvivalData(f(TIME), b(EVENT))


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
    value = ll_method(dist, data)
    np.testing.assert_array_equal(value, ll(dist, data))
    assert_close(value, [-LOG2 - 1.0, -1.0])


@pytest.mark.step("A6")
def test_nested_family_method_equals_function():
    """spec 4.2 and 7.1, rev 29: a family with a nested distribution field has the method, equal to the function.

    ``scaled_capped`` with ``log_ratio=0`` reproduces its ``capped_rate``
    baseline: right-censored ``log S(2) = -2 r``, exact ``log r - r``.
    """
    base = fam.capped_rate()(rate=f([1.0, 0.5]), cap=f([2.0, 2.0]))
    dist = fam.scaled_capped()(base=base, log_ratio=f([0.0, 0.0]))
    data = survjax.SurvivalData(f([2.0, 1.0]), b([RIGHT, EXACT]))
    value = ll_method(dist, data)
    np.testing.assert_array_equal(value, ll(dist, data))
    assert_close(value, [-2.0, -LOG2 - 0.5])


# ------------------------------------------------------- the two input forms


@pytest.mark.step("A6")
def test_data_form_positional():
    """spec 6.1 and 7.1, rev 29: `dist.log_likelihood(data)` equals `survjax.log_likelihood(dist, data)`."""
    value = ll_method(_dist(), _data())
    np.testing.assert_array_equal(value, ll(_dist(), _data()))
    assert_close(value, EXPECTED)


@pytest.mark.step("A6")
def test_data_form_by_keyword():
    """spec 7.1, rev 29: the method's signature names its first parameter `data`."""
    value = np.asarray(_dist().log_likelihood(data=_data()))
    np.testing.assert_array_equal(value, ll(_dist(), _data()))


@pytest.mark.step("A6")
def test_raw_form_equals_function_raw_form():
    """spec 6.1 and 7.1, rev 29: the method accepts the raw `time=`/`event=` form, equal to the function's."""
    value = ll_method(_dist(), time=f(TIME), event=b(EVENT))
    np.testing.assert_array_equal(value, ll(_dist(), time=f(TIME), event=b(EVENT)))
    assert_close(value, EXPECTED)


@pytest.mark.step("A6")
def test_raw_form_equals_data_form():
    """spec 6.1, rev 29: through the method, the raw form equals the data form."""
    raw = ll_method(_dist(), time=f(TIME), event=b(EVENT))
    np.testing.assert_array_equal(raw, ll_method(_dist(), _data()))


@pytest.mark.step("A6")
def test_raw_form_accepts_integer_event_codes():
    """spec 3.1 and 6.1, rev 29: the method's raw `event` accepts numbers exactly 0 or 1."""
    value = ll_method(_dist(), time=f(TIME), event=np.array([1, 0, 1], dtype=np.int32))
    assert_close(value, EXPECTED)


@pytest.mark.step("A6")
def test_weibull_method_equals_function():
    """spec 7.1, rev 29: Weibull per-row parameters, hand-derived, identical through the method.

    ``k=2, b=3, t=1.5`` (``z=0.5``): exact ``log(2/3) + log 0.5 - 0.25``,
    right ``-0.25``.
    """
    dist = survjax.Weibull(shape=f([2.0, 2.0]), scale=f([3.0, 3.0]))
    data = survjax.SurvivalData(f([1.5, 1.5]), b([EXACT, RIGHT]))
    value = ll_method(dist, data)
    np.testing.assert_array_equal(value, ll(dist, data))
    assert_close(value, [np.log(2.0 / 3.0) + np.log(0.5) - 0.25, -0.25])


# ------------------------------------------------------------ static misuse


@pytest.mark.step("A6")
def test_raw_array_passed_as_data_raises_type_error():
    """spec 6.1, rev 29: "Passing a raw array positionally as `data` raises a `TypeError`", through the method too."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(f(TIME))


@pytest.mark.step("A6")
def test_mixing_forms_raises_type_error():
    """spec 6.1 and 6.2, rev 29: "Mixing the forms is rejected" by the method (static misuse, `TypeError`)."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(_data(), time=f(TIME), event=b(EVENT))


@pytest.mark.step("A6")
def test_raw_form_without_event_raises_type_error():
    """spec 6.1, rev 29: the method's raw form takes `time` and `event`, "both required"."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(time=f(TIME))


@pytest.mark.step("A6")
def test_raw_form_without_time_raises_type_error():
    """spec 6.1, rev 29: the method's raw form takes `time` and `event`, "both required"."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(event=b(EVENT))


@pytest.mark.step("A6")
def test_no_data_at_all_raises_type_error():
    """spec 6.1 and 6.2, rev 29: calling the method with neither form is static misuse."""
    with pytest.raises(TypeError):
        _dist().log_likelihood()


@pytest.mark.step("A6")
def test_time_and_event_are_keyword_only():
    """spec 7.1, rev 29: the signature `log_likelihood(data=None, *, time=None, event=None)` makes `time`/`event` keyword-only."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(None, f(TIME), b(EVENT))


@pytest.mark.step("A6")
@pytest.mark.parametrize("name", ["right_censored", "left_trunc", "time2"])
def test_raw_form_rejects_constructor_keywords(name):
    """spec 6.1, rev 29: "The raw form takes only `time` and `event`", through the method too."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(time=f(TIME), event=b(EVENT), **{name: f([0.0, 0.0, 0.0])})


# ------------------------------------------- derivatives: method equals function
#
# 12.1 requires equality "in values and first and second parameter
# derivatives, in both input forms". Derivatives are compared with the 12.1
# tolerances (assert_close); values exactly.

K, B = 1.5, 2.0
W_TIME = [0.5, 1.5, 3.0, 4.0]
W_EVENT = [EXACT, RIGHT, EXACT, RIGHT]


def _weibull_sum(via, form):
    data = survjax.SurvivalData(f(W_TIME), b(W_EVENT))

    def total(params):
        dist = survjax.Weibull(shape=params[0], scale=params[1])
        if form == "data":
            args, kwargs = (data,), {}
        else:
            args, kwargs = (), {"time": f(W_TIME), "event": b(W_EVENT)}
        if via == "method":
            return jnp.sum(dist.log_likelihood(*args, **kwargs))
        return jnp.sum(survjax.log_likelihood(dist, *args, **kwargs))

    return total


def _params():
    return jnp.asarray(f([K, B]))


@pytest.mark.step("A6")
@pytest.mark.parametrize("form", ["data", "raw"])
def test_method_value_equals_function_value(form):
    """spec 7.1 and 12.1, rev 29: Weibull summed value, identical through method and function, in both input forms."""
    method = np.asarray(_weibull_sum("method", form)(_params()))
    function = np.asarray(_weibull_sum("function", form)(_params()))
    np.testing.assert_array_equal(method, function)


@pytest.mark.step("A6")
@pytest.mark.parametrize("form", ["data", "raw"])
def test_method_gradient_equals_function_gradient(form):
    """spec 7.1 and 12.1, rev 29: first parameter derivatives agree between method and function, in both input forms."""
    method = np.asarray(jax.grad(_weibull_sum("method", form))(_params()))
    function = np.asarray(jax.grad(_weibull_sum("function", form))(_params()))
    assert np.isfinite(method).all()
    assert_close(method, function)


@pytest.mark.step("A6")
@pytest.mark.parametrize("form", ["data", "raw"])
def test_method_hessian_equals_function_hessian(form):
    """spec 7.1 and 12.1, rev 29: second parameter derivatives agree between method and function, in both input forms."""
    method = np.asarray(jax.hessian(_weibull_sum("method", form))(_params()))
    function = np.asarray(jax.hessian(_weibull_sum("function", form))(_params()))
    assert method.shape == (2, 2)
    assert np.isfinite(method).all()
    assert_close(method, function)


@pytest.mark.step("A6")
def test_method_gradient_is_hand_derived():
    """spec 7.1 and 7.3, rev 29: `d/dr sum = sum(event/r - t)` through the method; `r=0.5`: `(2-1) - 2 + (2-4) = -3`."""

    def loss(r):
        return jnp.sum(survjax.Exponential(rate=r).log_likelihood(_data()))

    assert_close(np.asarray(jax.grad(loss)(jnp.asarray(f(0.5)))), -3.0)


# ----------------------------------------------------------- transformations


@pytest.mark.step("A6")
def test_method_under_jit_matches_eager():
    """spec 7.1 and 12.1, rev 29: the method under `jit` (distribution and data traced) agrees with the eager function."""
    jitted = np.asarray(jax.jit(lambda d, x: d.log_likelihood(x))(_dist(), _data()))
    assert_close(jitted, ll(_dist(), _data()))


@pytest.mark.step("A6")
def test_method_raw_form_under_jit_matches_eager():
    """spec 6.1 and 7.1, rev 29: the method's raw form "behaves identically eagerly and under `jit`"."""

    def run(time, event):
        return _dist().log_likelihood(time=time, event=event)

    assert_close(np.asarray(jax.jit(run)(f(TIME), b(EVENT))), EXPECTED)


@pytest.mark.step("A6")
def test_method_under_vmap_over_draws_matches_batched_call():
    """spec 5 and 7.1, rev 29: `vmap` of the method over draws equals the `(D,1)` broadcast function call."""
    rates = f([0.5, 1.0])
    mapped = jax.vmap(lambda r: survjax.Exponential(rate=r).log_likelihood(_data()))(rates)
    batched = ll(survjax.Exponential(rate=rates[:, None]), _data())
    assert_close(np.asarray(mapped), batched)
