"""A9: compiled composition of construction, diagnostics, likelihood and derivatives (spec 6.2, 7.2, 7.3, 12.1, 12.3; rev 34).

12.1 (Compiled composition): "ordinary construction, diagnostic access and
likelihood evaluation compose inside `jit`; first and second parameter
derivatives agree with direct evaluation on valid smooth interior cases within
section 7.3's guarantees; verify these library capabilities directly, without
requiring an external solver or simulation workflow."

The Newton iterations below are written in the test with ``lax`` primitives;
they are not a survjax API and use no external solver. The exponential MLE is
closed form: ``r_hat = d/T`` (``d`` events, ``T`` total follow-up), so with
``u = log r`` the summed log likelihood ``d u - exp(u) T`` has
``u_hat = log(d/T)`` (spec 2, 4.1).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_close, b, f

import survjax

TIME = [0.5, 1.2, 2.0, 3.5, 0.8, 2.6]
EVENT = [True, False, True, False, True, True]
# d = 4, T = 10.6
MLE_LOG_RATE = np.log(4.0 / 10.6)


def _loss_on(data):
    def loss(params):
        dist = survjax.Weibull(shape=params[0], scale=params[1])
        return jnp.sum(survjax.log_likelihood(dist, data))

    return loss


@pytest.mark.step("A9")
def test_construction_diagnostics_and_derivatives_inside_jit_match_direct_evaluation():
    """spec 12.1, rev 33: value, gradient and Hessian with construction and `data.issues.ok` inside `jit` equal direct evaluation on prepared data."""

    @jax.jit
    def compiled(params, time, event):
        data = survjax.SurvivalData(time, event)
        loss = _loss_on(data)
        return data.issues.ok, loss(params), jax.grad(loss)(params), jax.hessian(loss)(params)

    params = jnp.asarray(f([1.4, 2.2]))
    ok, value, grad, hess = compiled(params, f(TIME), b(EVENT))
    direct = _loss_on(survjax.SurvivalData(f(TIME), b(EVENT)))
    assert bool(ok)
    assert_close(np.asarray(value), np.asarray(direct(params)))
    assert_close(np.asarray(grad), np.asarray(jax.grad(direct)(params)))
    assert_close(np.asarray(hess), np.asarray(jax.hessian(direct)(params)))


@pytest.mark.step("A9")
def test_diagnostics_select_the_loss_in_compiled_control_flow():
    """spec 6.2 and 12.1, rev 33: `data.issues.ok` is "usable in compiled control flow"; `lax.cond` on it selects the loss and its gradient."""

    @jax.jit
    def guarded(rate, time, event):
        data = survjax.SurvivalData(time, event)

        def loss(r):
            return jnp.sum(survjax.log_likelihood(survjax.Exponential(rate=r), data))

        return jax.lax.cond(
            data.issues.ok,
            lambda r: (loss(r), jax.grad(loss)(r)),
            lambda r: (jnp.zeros_like(r), jnp.zeros_like(r)),
            rate,
        )

    rate = jnp.asarray(f(0.5))
    value, grad = guarded(rate, f(TIME), b(EVENT))
    # 4 log 0.5 - 0.5 * 10.6; d/dr = 4/0.5 - 10.6
    assert_close(np.asarray(value), 4.0 * np.log(0.5) - 5.3)
    assert_close(np.asarray(grad), 8.0 - 10.6)
    bad_event = np.asarray([1, 0, 2, 0, 1, 1], dtype=np.int32)
    value, grad = guarded(rate, f(TIME), bad_event)
    assert float(value) == 0.0
    assert float(grad) == 0.0


@pytest.mark.step("A9")
def test_newton_iterations_inside_jit_reach_the_closed_form_exponential_mle():
    """spec 7.3 and 12.1, rev 33: first and second derivatives compose in a compiled loop; Newton on `log r` reaches `log(d/T)`."""

    @jax.jit
    def fit(time, event):
        data = survjax.SurvivalData(time, event)

        def loss(u):
            dist = survjax.Exponential.from_log(log_rate=u)
            return jnp.sum(survjax.log_likelihood(dist, data))

        def step(_, u):
            return u - jax.grad(loss)(u) / jax.hessian(loss)(u)

        return jax.lax.fori_loop(0, 30, step, jnp.zeros((), dtype=time.dtype))

    u_hat = fit(jnp.asarray(f(TIME)), jnp.asarray(b(EVENT)))
    assert_close(np.asarray(u_hat), MLE_LOG_RATE)


@pytest.mark.step("A9")
def test_vmapped_compiled_fits_over_replicate_datasets():
    """spec 6.2, 7.3 and 12.1, rev 33: construction, diagnostics and second-order fits under `vmap` over replicates.

    Replicate 1 holds event code 2, an invalid observation: its ``ok`` is
    false and it does not affect the others (spec 6.2, "One problematic
    replicate does not inherently abort the study").
    """
    times = f([TIME, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]])
    events = np.asarray([[1, 0, 1, 0, 1, 1], [1, 2, 1, 1, 0, 0], [1, 1, 1, 1, 1, 0]], dtype=np.int32)

    def fit_one(time, event):
        data = survjax.SurvivalData(time, event)

        def loss(u):
            dist = survjax.Exponential.from_log(log_rate=u)
            return jnp.sum(survjax.log_likelihood(dist, data))

        def step(_, u):
            return u - jax.grad(loss)(u) / jax.hessian(loss)(u)

        u_hat = jax.lax.fori_loop(0, 30, step, jnp.zeros((), dtype=time.dtype))
        return data.issues.ok, u_hat

    ok, u_hat = jax.jit(jax.vmap(fit_one))(jnp.asarray(times), jnp.asarray(events))
    np.testing.assert_array_equal(np.asarray(ok), [True, False, True])
    # replicate 0: d=4, T=10.6; replicate 2: d=5, T=4.2
    assert_close(np.asarray(u_hat)[[0, 2]], [MLE_LOG_RATE, np.log(5.0 / 4.2)])


@pytest.mark.step("A9")
def test_compiled_hessian_reuses_compilation_for_new_values():
    """spec 7.2 and 12.3, rev 28: changed parameter and observation values with equal shapes reuse the compiled Hessian."""
    traces = []

    @jax.jit
    def hess(params, time, event):
        traces.append(None)
        return jax.hessian(_loss_on(survjax.SurvivalData(time, event)))(params)

    hess(f([1.4, 2.2]), f(TIME), b(EVENT))
    hess(f([0.9, 3.0]), f([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), b([False, True, True, False, True, False]))
    assert len(traces) == 1


@pytest.mark.step("A9")
def test_minibatch_gradients_reuse_compilation():
    """spec 7.2 and 12.3, rev 28: "same-shaped minibatches from one parent dataset" reuse the compiled gradient."""
    parent = survjax.SurvivalData(f(TIME), b(EVENT))
    traces = []

    @jax.jit
    def grad(params, data):
        traces.append(None)
        return jax.grad(_loss_on(data))(params)

    params = f([1.4, 2.2])
    first = grad(params, parent[np.array([0, 1, 2])])
    grad(params, parent[np.array([3, 4, 5])])
    grad(params, parent[np.array([5, 0, 3])])
    assert len(traces) == 1
    direct = jax.grad(_loss_on(parent[np.array([0, 1, 2])]))(jnp.asarray(params))
    assert_close(np.asarray(first), np.asarray(direct))
