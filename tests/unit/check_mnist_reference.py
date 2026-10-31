"""Explicit examples-group checks for use-case 03 intercept-only reference."""

import sys
from pathlib import Path
from types import SimpleNamespace

import impl_tolerances as tol
import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import optimistix as optx
import pytest
from jax.test_util import check_grads

import survjax

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist import model  # noqa: E402

requires_x64 = pytest.mark.skipif(
    not jax.config.x64_enabled,
    reason="reference fit requires a separate float64 process",
)


@requires_x64
@pytest.mark.parametrize("censor_time", [0, 2, 16])
def test_intercept_fit_matches_independent_profile_score_root(censor_time):
    """03 Reference: profile b^k=sum(t^k)/d; solve d/k+sum_event(log t)-d*weighted_log_t=0."""
    times = [1, censor_time, 4, 8]
    fitted = model.fit_intercept(jnp.array(times, dtype=jnp.float32), [1, 0, 1, 1])
    with mp.workdps(50):
        # At t=0, both t^k and its k derivative vanish for k>0.
        logs = [mp.log(t) for t in times if t > 0]

        def score(k):
            powers = [mp.exp(k * x) for x in logs]
            return (
                3 / k
                + mp.log(32)
                - 3
                * sum(p * x for p, x in zip(powers, logs, strict=True))
                / sum(powers)
            )

        k = mp.findroot(score, (0.5, 2))
        log_b = mp.log(sum(mp.exp(k * x) for x in logs) / 3) / k
        expected = [float(mp.log(k)), float(log_b)]
    assert fitted.dtype == np.float64 and fitted.shape == (2,)
    np.testing.assert_allclose(
        fitted, expected, atol=tol.ATOL[True], rtol=tol.RTOL[True]
    )
    data = survjax.SurvivalData(jnp.array(times, dtype=jnp.float64), [1, 0, 1, 1])
    gradient = jax.grad(model._intercept_nll)(jnp.asarray(fitted), data)
    np.testing.assert_allclose(
        gradient, 0, atol=tol.DERIVATIVE_ATOL[True], rtol=tol.DERIVATIVE_RTOL[True]
    )


@requires_x64
def test_reference_objective_values_and_parameter_transformations():
    """Spec 7.2/7.3: exponential mean NLL, including uninformative censoring at0."""
    data = survjax.SurvivalData(jnp.array([1.0, 0.0, 3.0]), [1, 0, 1])
    theta = jnp.array([0.0, jnp.log(2.0)])
    expected = (2 + 2 * np.log(2)) / 3  # sum(t/b + event*log b)/N, k=1,b=2.
    eager = model._intercept_nll(theta, data)
    compiled = jax.jit(model._intercept_nll)(theta, data)
    batched = jax.vmap(model._intercept_nll, in_axes=(0, None))(
        jnp.stack((theta, theta + jnp.array([0.0, jnp.log(2.0)]))), data
    )
    np.testing.assert_allclose(
        [eager, compiled, batched[0]],
        expected,
        atol=tol.ATOL[True],
        rtol=tol.RTOL[True],
    )
    np.testing.assert_allclose(
        batched[1], (1 + 2 * np.log(4)) / 3, atol=tol.ATOL[True], rtol=tol.RTOL[True]
    )
    check_grads(
        lambda t: model._intercept_nll(t, data),
        (theta,),
        order=2,
        modes=("fwd", "rev"),
        eps=tol.LIKELIHOOD_GRAD_EPS[True],
        atol=tol.DERIVATIVE_ATOL[True],
        rtol=tol.DERIVATIVE_RTOL[True],
    )


def test_fit_refuses_float32_process_without_changing_precision(x64_enabled):
    """03 Precision: caller launches float64 process; library never toggles mode."""
    if x64_enabled:
        pytest.skip("float32 refusal is meaningful only with x64 disabled")
    with pytest.raises(RuntimeError, match="JAX_ENABLE_X64=1"):
        model.fit_intercept([1, 2], [1, 0])
    assert not jax.config.x64_enabled


@requires_x64
@pytest.mark.parametrize("time,event", [([], []), (1, 1), ([1, 2], [1])])
def test_fit_requires_matching_nonempty_vectors(time, event):
    """03: reject missing or misaligned training cohorts before optimization."""
    with pytest.raises(ValueError, match="vectors"):
        model.fit_intercept(time, event)


@requires_x64
@pytest.mark.parametrize("time,event", [([-1, 2], [1, 0]), ([1, 2], [1, 2])])
def test_fit_raises_on_invalid_observations(time, event):
    """03/spec 6.2: public observation diagnostics must abort fitting."""
    with pytest.raises(ValueError):
        model.fit_intercept(time, event)


@requires_x64
def test_fit_rejects_boolean_times():
    """Spec 3.3: explicit float64 storage must not turn boolean times into real times."""
    with pytest.raises(TypeError):
        model.fit_intercept([True, True], [1, 0])


@requires_x64
def test_fit_rejects_no_event_cohort():
    """03: no finite intercept-only maximum exists for an entirely censored cohort."""
    with pytest.raises(ValueError, match="at least one event"):
        model.fit_intercept([1, 2], [0, 0])


@requires_x64
def test_unsuccessful_solver_never_returns_reference(monkeypatch):
    """03: expose BFGS failure, never silently retain a starting guess."""
    failed = SimpleNamespace(
        value=jnp.zeros(2), result=optx.RESULTS.nonlinear_max_steps_reached
    )
    monkeypatch.setattr(model.optx, "minimise", lambda *args, **kw: failed)
    with pytest.raises(RuntimeError, match="BFGS failed"):
        model.fit_intercept([1, 2], [1, 0])


@requires_x64
@pytest.mark.parametrize(
    "parameters,value,gradient",
    [
        ([float("nan"), 0], 0, [0, 0]),
        ([0, 0], float("nan"), [0, 0]),
        ([0, 0], 0, [float("inf"), 0]),
    ],
)
def test_nonfinite_success_result_never_returns_reference(
    monkeypatch, parameters, value, gradient
):
    """03: reject nonfinite parameters, loss or gradients even with success status."""
    result = SimpleNamespace(
        value=jnp.array(parameters), result=optx.RESULTS.successful
    )
    monkeypatch.setattr(model.optx, "minimise", lambda *args, **kw: result)
    monkeypatch.setattr(
        model.jax,
        "value_and_grad",
        lambda fn: lambda *args: (jnp.asarray(value), jnp.asarray(gradient)),
    )
    with pytest.raises(RuntimeError, match="nonfinite"):
        model.fit_intercept([1, 2], [1, 0])
