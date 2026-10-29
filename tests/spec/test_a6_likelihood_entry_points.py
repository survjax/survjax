"""A6: the likelihood entry point, its input forms, transformations and compilation (spec 6.1, 6.3, 7.1, 7.2; rev 28, 29, 35).

Since rev 35, ``dist.log_likelihood`` is the only public likelihood entry
point. Exponential contributions, hand-derived: exact ``log r - r t``,
right-censored ``-r t``. Method-specific checks (custom families,
keyword-only arguments, agreement of the two input forms in derivatives) are
in ``test_a6_likelihood_method.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_close, b, f, ll

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


# --------------------------------------------------------------- public names


@pytest.mark.step("A6")
def test_no_top_level_log_likelihood():
    """spec 7.1, 10.2 and 12.1, rev 35: "There is no top-level `survjax.log_likelihood` function"; the method is the only entry point."""
    assert not hasattr(survjax, "log_likelihood")


@pytest.mark.step("A6")
def test_likelihood_has_no_with_info_variant():
    """spec 6.2, rev 29: the likelihood "has no `_with_info` variant"; decision 0043 removed it and `LikelihoodIssues`."""
    assert not hasattr(survjax, "log_likelihood_with_info")
    assert not hasattr(_dist(), "log_likelihood_with_info")


@pytest.mark.step("A6")
def test_likelihood_issues_container_is_removed():
    """spec 6.3, rev 29: `LikelihoodIssues` is no longer a diagnostic container (removed by decision 0043)."""
    assert not hasattr(survjax, "LikelihoodIssues")


# ------------------------------------------------------------- the two forms


@pytest.mark.step("A6")
def test_data_form_positional():
    """spec 6.1, rev 35: `dist.log_likelihood(data)`; the receiver takes the place of the predictions."""
    assert_close(ll(_dist(), _data()), EXPECTED)


@pytest.mark.step("A6")
def test_data_form_by_keyword():
    """spec 7.1, rev 35: the method signature `log_likelihood(data=None, *, time=None, event=None)` names its first parameter `data`."""
    value = np.asarray(_dist().log_likelihood(data=_data()))
    assert_close(value, EXPECTED)


@pytest.mark.step("A6")
def test_raw_form_equals_data_form():
    """spec 6.1, rev 35: `dist.log_likelihood(time=..., event=...)` is the raw exact/right-censored form, equal to the data form."""
    raw = ll(_dist(), time=f(TIME), event=b(EVENT))
    np.testing.assert_array_equal(raw, ll(_dist(), _data()))
    assert_close(raw, EXPECTED)


@pytest.mark.step("A6")
def test_raw_form_accepts_integer_event_codes():
    """spec 3.1 and 6.1, rev 35: raw `event` accepts numbers exactly 0 or 1."""
    raw = ll(_dist(), time=f(TIME), event=np.array([1, 0, 1], dtype=np.int32))
    assert_close(raw, EXPECTED)


@pytest.mark.step("A6")
def test_raw_array_passed_as_data_raises_type_error():
    """spec 6.1, rev 35: "Passing a raw array positionally as `data` raises a `TypeError` pointing to the keyword form"."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(f(TIME))


@pytest.mark.step("A6")
def test_mixing_forms_raises_type_error():
    """spec 6.1 and 6.2, rev 35: "Mixing the forms is rejected" (static misuse, `TypeError`)."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(_data(), time=f(TIME), event=b(EVENT))


@pytest.mark.step("A6")
def test_data_with_only_event_keyword_raises_type_error():
    """spec 6.1, rev 35: mixing includes `data` together with only one raw keyword."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(_data(), event=b(EVENT))


@pytest.mark.step("A6")
def test_raw_form_without_event_raises_type_error():
    """spec 6.1, rev 35: the raw form takes `time` and `event`, "both required"."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(time=f(TIME))


@pytest.mark.step("A6")
def test_raw_form_without_time_raises_type_error():
    """spec 6.1, rev 35: the raw form takes `time` and `event`, "both required"."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(event=b(EVENT))


@pytest.mark.step("A6")
def test_no_data_at_all_raises_type_error():
    """spec 6.1 and 6.2, rev 35: neither form supplied is static misuse."""
    with pytest.raises(TypeError):
        _dist().log_likelihood()


@pytest.mark.step("A6")
@pytest.mark.parametrize("name", ["right_censored", "left_trunc", "time2"])
def test_raw_form_rejects_constructor_keywords(name):
    """spec 6.1, rev 35: "The raw form takes only `time` and `event`"; "The constructor signature is not copied"."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(time=f(TIME), event=b(EVENT), **{name: f([0.0, 0.0, 0.0])})


@pytest.mark.step("A6")
def test_raw_form_shape_mismatch_raises_type_error():
    """spec 6.2, rev 35: incompatible raw shapes are static misuse (`TypeError`)."""
    with pytest.raises(TypeError):
        _dist().log_likelihood(time=f(TIME), event=b([EXACT, RIGHT]))


# ----------------------------------------------------------- transformations


@pytest.mark.step("A6")
def test_likelihood_under_jit_matches_eager():
    """spec 7.1 and 12.1, rev 35: eager and `jit` agree on values."""
    jitted = np.asarray(jax.jit(lambda d, x: d.log_likelihood(x))(_dist(), _data()))
    assert_close(jitted, ll(_dist(), _data()))


@pytest.mark.step("A6")
def test_traced_construction_inside_jit_matches_eager():
    """spec 6.2 and 12.1, rev 35: ordinary construction and likelihood compose inside `jit` (traced metadata all true)."""

    def run(rate, time, event):
        data = survjax.SurvivalData(time, event)
        return survjax.Exponential(rate=rate).log_likelihood(data)

    value = np.asarray(jax.jit(run)(f(0.5), f(TIME), b(EVENT)))
    assert_close(value, EXPECTED)


@pytest.mark.step("A6")
def test_raw_form_inside_jit_matches_eager():
    """spec 6.1, rev 35: the raw form "behaves identically eagerly and under `jit`"."""

    def run(time, event):
        return _dist().log_likelihood(time=time, event=event)

    value = np.asarray(jax.jit(run)(f(TIME), b(EVENT)))
    assert_close(value, EXPECTED)


@pytest.mark.step("A6")
def test_vmap_over_data_matches_batched_call():
    """spec 7.1 and 12.1, rev 35: `vmap` over observations equals the vectorized call."""
    mapped = jax.vmap(lambda d: _dist().log_likelihood(d))(_data())
    assert_close(np.asarray(mapped), ll(_dist(), _data()))


@pytest.mark.step("A6")
def test_vmap_over_parameter_draws_matches_batched_call():
    """spec 5 and 12.1, rev 35: `vmap` over draws equals the `(D,1)` broadcast call."""
    rates = f([0.5, 1.0])
    mapped = jax.vmap(
        lambda r: survjax.Exponential(rate=r).log_likelihood(_data())
    )(rates)
    batched = ll(survjax.Exponential(rate=rates[:, None]), _data())
    assert_close(np.asarray(mapped), batched)


@pytest.mark.step("A6")
def test_gradient_of_summed_likelihood_is_hand_derived():
    """spec 7.1 and 7.3, rev 35: callers sum contributions; `d/dr sum = sum(event/r - t)`.

    ``r = 0.5``: ``(1/r - 1) + (-2) + (1/r - 4) = 1 - 2 - 2 = -3``. The full
    derivative suite (``check_grads``, Hessians) is A9.
    """

    def loss(r):
        return jnp.sum(survjax.Exponential(rate=r).log_likelihood(_data()))

    assert_close(np.asarray(jax.grad(loss)(jnp.asarray(f(0.5)))), -3.0)
    assert_close(np.asarray(jax.jit(jax.grad(loss))(jnp.asarray(f(0.5)))), -3.0)


# ------------------------------------------------------- compilation and device


@pytest.mark.step("A6")
def test_new_values_reuse_compilation():
    """spec 7.2, rev 35: changing parameter and observation values with equal shapes, dtypes and metadata reuses compilation."""
    traces = []

    @jax.jit
    def run(dist, data):
        traces.append(None)
        return dist.log_likelihood(data)

    run(survjax.Exponential(rate=f([0.5, 1.0])), survjax.SurvivalData(f([1.0, 2.0]), b([EXACT, RIGHT])))
    run(survjax.Exponential(rate=f([2.0, 3.0])), survjax.SurvivalData(f([3.0, 4.0]), b([EXACT, RIGHT])))
    assert len(traces) == 1


@pytest.mark.step("A6")
def test_traced_construction_reuses_compilation():
    """spec 7.2, rev 35: repeated compiled construction from traced inputs with new values does not retrace."""
    traces = []

    @jax.jit
    def run(rate, time, event):
        traces.append(None)
        data = survjax.SurvivalData(time, event)
        return survjax.Exponential(rate=rate).log_likelihood(data), data.issues.ok

    run(f([0.5, 1.0]), f([1.0, 2.0]), b([EXACT, RIGHT]))
    run(f([2.0, 3.0]), f([3.0, 4.0]), b([RIGHT, EXACT]))
    assert len(traces) == 1


@pytest.mark.step("A6")
def test_compiled_path_makes_no_host_transfers():
    """spec 7.2, rev 35: compiled construction, diagnostic access, likelihood and gradient run "without library-induced host transfers".

    Compiled once outside the guard; inputs are placed on the device before
    it, so the guarded call itself must not move data to or from the host.
    """

    def run(rate, time, event):
        data = survjax.SurvivalData(time, event)

        def loss(r):
            return jnp.sum(survjax.Exponential(rate=r).log_likelihood(data))

        dist = survjax.Exponential(rate=rate)
        value = dist.log_likelihood(data)
        bad_params = dist.parameter_issues().invalid_entries
        return value, data.issues.ok, bad_params, jax.grad(loss)(rate)

    compiled = jax.jit(run)
    jax.block_until_ready(compiled(*(jax.device_put(x) for x in (f([0.5, 1.0]), f([1.0, 2.0]), b([EXACT, RIGHT])))))
    args = tuple(jax.device_put(x) for x in (f([2.0, 3.0]), f([3.0, 4.0]), b([RIGHT, EXACT])))
    jax.block_until_ready(args)
    with jax.transfer_guard("disallow"):
        out = compiled(*args)
        jax.block_until_ready(out)
    value = np.asarray(out[0])
    assert_close(value, [-6.0, np.log(3.0) - 12.0])
