"""Analytical anchors for compiled likelihood composition (spec 7.3/12.1)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


def _distribution(family, p):
    if family == "exponential":
        return survjax.Exponential(rate=p[0])
    return survjax.Weibull(shape=p[0], scale=p[1])


def _analytical(family, p, time, event):
    p, t, e = (
        np.asarray(p, dtype=float),
        np.asarray(time, dtype=float),
        np.asarray(event, dtype=float),
    )
    if family == "exponential":
        r = p[0]
        # Sum e*log(r)-r*t: gradient d/r-T and Hessian -d/r**2.
        return (
            e.sum() * np.log(r) - r * t.sum(),
            [e.sum() / r - t.sum()],
            [[-e.sum() / r**2]],
        )
    k, b = p
    u = np.log(t / b)
    h = (t / b) ** k
    # Sum e*(log(k)-log(b)+(k-1)*u)-h, differentiated in (k,b).
    value = np.sum(e * (np.log(k) - np.log(b) + (k - 1) * u) - h)
    gradient = [np.sum(e / k + (e - h) * u), np.sum(k * (h - e) / b)]
    hkk = np.sum(-e / k**2 - h * u**2)
    hkb = np.sum((h - e + k * h * u) / b)
    hbb = np.sum(k * (e - (k + 1) * h) / b**2)
    return value, gradient, [[hkk, hkb], [hkb, hbb]]


def _evaluate(family, p, data):
    def loss(q):
        return _distribution(family, q).log_likelihood(data).sum()

    return (
        data.issues.ok,
        _distribution(family, p).parameter_issues().ok,
        loss(p),
        jax.jacfwd(loss)(p),
        jax.jacrev(loss)(p),
        jax.jacfwd(jax.jacfwd(loss))(p),
        jax.hessian(loss)(p),
        jax.jacrev(jax.jacrev(loss))(p),
    )


def _assert_result(result, expected, x64_enabled):
    data_ok, parameters_ok, value, forward, reverse, *hessians = result
    assert data_ok and parameters_ok
    for actual, wanted in zip(
        [value, forward, reverse, *hessians],
        [expected[0], expected[1], expected[1], *([expected[2]] * 3)],
        strict=True,
    ):
        np.testing.assert_allclose(
            actual, wanted, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


@pytest.mark.parametrize("family", ["exponential", "weibull"])
@pytest.mark.parametrize("bounds", [False, True], ids=["flags", "bounds"])
def test_traced_construction_diagnostics_and_derivatives(family, bounds, x64_enabled):
    """Spec 6.2/7.3/12.1: both constructors compose through second order."""
    time, event = jnp.array([0.5, 1.0, 1.5]), jnp.array([True, False, True])
    p = jnp.array([1.5] if family == "exponential" else [1.5, 2.0])

    def evaluate(q, t, e):
        data = (
            survjax.SurvivalData.from_bounds(t, jnp.where(e, t, jnp.inf), exact=e)
            if bounds
            else survjax.SurvivalData(t, e)
        )
        return _evaluate(family, q, data)

    expected = _analytical(family, p, time, event)
    prepared = survjax.SurvivalData(time, event)
    for result in (
        _evaluate(family, p, prepared),
        evaluate(p, time, event),
        jax.jit(evaluate)(p, time, event),
    ):
        _assert_result(result, expected, x64_enabled)


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_mapped_likelihood_derivatives_are_analytical(family, x64_enabled):
    """Spec 7.3/12.1: vmap and jit-of-vmap preserve independent replicates."""
    times = jnp.array([[0.5, 1.0, 1.5], [1.5, 0.5, 2.0]])
    events = jnp.array([[True, False, True], [False, True, False]])
    parameters = jnp.array(
        [[1.5], [0.75]] if family == "exponential" else [[1.5, 2.0], [0.75, 1.25]]
    )

    def evaluate(p, t, e):
        return _evaluate(family, p, survjax.SurvivalData(t, e))

    mapped = jax.vmap(evaluate)
    for results in (
        mapped(parameters, times, events),
        jax.jit(mapped)(parameters, times, events),
    ):
        for i in range(2):
            expected = _analytical(family, parameters[i], times[i], events[i])
            _assert_result(
                jax.tree.map(lambda a, i=i: a[i], results), expected, x64_enabled
            )


def test_diagnostics_guard_invalid_replicate_under_jit_vmap(x64_enabled):
    """Spec 6.2/12.1: diagnostic control flow leaves valid replicates intact."""

    def evaluate(rate, time, event):
        data = survjax.SurvivalData(time, event)
        ok = data.issues.ok & survjax.Exponential(rate=rate).parameter_issues().ok

        def guarded(r):
            return jax.lax.cond(
                ok,
                lambda q: survjax.Exponential(rate=q).log_likelihood(data).sum(),
                lambda q: jnp.zeros_like(q),
                r,
            )

        return ok, guarded(rate), jax.grad(guarded)(rate), jax.hessian(guarded)(rate)

    time = jnp.array([[0.5, 1.0], [1.0, 2.0], [1.5, 0.5]])
    event = jnp.array([[1, 0], [2, 0], [0, 1]])
    ok, *values = jax.jit(jax.vmap(evaluate))(jnp.array([1.0, 1.0, 1.0]), time, event)
    np.testing.assert_array_equal(ok, [True, False, True])
    # One event per valid replicate: at r=1, value=-T, grad=1-T, Hessian=-1.
    for actual, expected in zip(
        values, [[-1.5, 0, -2], [-0.5, 0, -1], [-1, 0, -1]], strict=True
    ):
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


@pytest.mark.parametrize(
    "prepared", [False, True], ids=["dynamic-arrays", "parent-minibatches"]
)
def test_derivative_compilation_reused_for_changed_values(prepared, x64_enabled):
    """Spec 7.2/12.3: dynamic values and inherited minibatch flags reuse traces."""
    parent = survjax.SurvivalData([0.5, 1.0, 1.5, 2.0], [True, False, False, True])
    traces = []

    def evaluate(p, *inputs):
        traces.append(None)
        data = inputs[0] if prepared else survjax.SurvivalData(*inputs)
        return _evaluate("weibull", p, data)

    compiled = jax.jit(evaluate)
    for index, parameters in (
        ([0, 1], [1.5, 2.0]),
        ([2, 3], [0.75, 1.25]),
        ([3, 0], [2.0, 1.5]),
    ):
        data = parent[jnp.array(index)]
        assert jax.tree.structure(data) == jax.tree.structure(parent)
        time, event = data.lower, data.exact
        inputs = (data,) if prepared else (time, event)
        p = jnp.array(parameters)
        result = jax.block_until_ready(compiled(p, *inputs))
        _assert_result(result, _analytical("weibull", p, time, event), x64_enabled)
    assert len(traces) == 1
