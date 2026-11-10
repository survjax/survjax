"""Untruncated censoring integration (spec 2, 4.2, 7.1–7.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import (
    ATOL,
    CENSORING_GRAD_EPS,
    DERIVATIVE_ATOL,
    DERIVATIVE_RTOL,
    RTOL,
)
from jax.test_util import check_grads

import survjax


def mixed_data():
    return survjax.SurvivalData.from_bounds(
        [1.0, 2.0, 0.0, 1.0, 0.0],
        [1.0, np.inf, 2.0, 3.0, np.inf],
        exact=[True, False, False, False, False],
    )


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_mixed_censoring_values_and_transforms(family, x64_enabled):
    """Spec 2/7.2: all kinds, including uninformative (0,inf], are safe."""
    dist = (
        survjax.Exponential(rate=0.5)
        if family == "exponential"
        else survjax.Weibull(shape=2.0, scale=2.0)
    )
    # Exp H=t/2; Weibull H=t^2/4. Interval: -H(a)+log(1-exp(-dH)).
    expected = (
        [np.log(0.5) - 0.5, -1, np.log(-np.expm1(-1)), -0.5 + np.log(-np.expm1(-1)), 0]
        if family == "exponential"
        else [
            np.log(0.5) - 0.25,
            -1,
            np.log(-np.expm1(-1)),
            -0.25 + np.log(-np.expm1(-2)),
            0,
        ]
    )
    data = mixed_data()
    with jax.debug_nans(True):
        actual = dist.log_likelihood(data)
    np.testing.assert_allclose(
        actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    np.testing.assert_allclose(
        jax.jit(lambda d: d.log_likelihood(data))(dist),
        actual,
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    batched = jax.tree.map(lambda x: jnp.stack([x, x]), dist)
    mapped = jax.vmap(lambda d: d.log_likelihood(data))(batched)
    np.testing.assert_allclose(
        mapped,
        jnp.stack([actual, actual]),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


def test_mixed_rate_derivatives_are_hand_derived(x64_enabled):
    """Spec 7.3: shared-rate first/second derivatives in all AD compositions."""
    data = mixed_data()

    def value(rate):
        return survjax.Exponential(rate=rate).log_likelihood(data)

    rate = jnp.asarray(0.5)
    # For width w=2: d log(1-exp(-rw))/dr=w/expm1(rw),
    # d2=-w^2*exp(rw)/expm1(rw)^2. Exact d=1/r-t, d2=-1/r^2.
    interval_grad = 2 / np.expm1(1)
    interval_hess = -4 * np.exp(1) / np.expm1(1) ** 2
    gradient = [1, -2, interval_grad, -1 + interval_grad, 0]
    hessian = [-4, 0, interval_hess, interval_hess, 0]
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(first(value))(rate),
            gradient,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        for second in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                second(first(value))(rate),
                hessian,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
    check_grads(
        value,
        (rate,),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=CENSORING_GRAD_EPS[x64_enabled],
    )


@survjax.register_family
@dataclass(frozen=True)
class IntervalProbe(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def reference_time(self):
        return jnp.ones_like(self.rate)

    def log_prob(self, t):
        return jnp.log(self.rate) - self.rate * t

    def log_survival(self, t):
        raise AssertionError("The evaluator must dispatch through the interval hook")

    def log_interval_prob(self, lower, upper):
        # Probe requires the spec 7.2 full-window substitute on inactive rows.
        valid = (lower >= 0) & (upper > lower)
        return jnp.where(valid, -self.rate * lower, jnp.nan)


def test_interval_override_receives_safe_inputs_before_evaluation(monkeypatch):
    """Spec 4.2/7.2: exact and invalid rows never send empty sets to the hook."""
    data = survjax.SurvivalData.from_bounds(
        [1.0, 0.0, 1.0, 2.0, -1.0],
        [1.0, 2.0, 3.0, np.inf, 3.0],
        exact=[True, False, False, False, False],
    )
    original = IntervalProbe.log_conditional_interval_prob

    def check_endpoints(self, lower, upper, left_trunc, right_trunc):
        # Spec 7.2: inspect operands before any final selection can hide errors.
        # Exact and invalid rows must receive (0,inf]; other sets stay active.
        np.testing.assert_array_equal(lower, [0, 0, 1, 2, 0])
        np.testing.assert_array_equal(upper, [np.inf, 2, 3, np.inf, np.inf])
        return original(self, lower, upper, left_trunc, right_trunc)

    # Host assertions are only for this eager operand check, not traced calls.
    with monkeypatch.context() as patch:
        patch.setattr(IntervalProbe, "log_conditional_interval_prob", check_endpoints)
        IntervalProbe(rate=1.0).log_likelihood(data)

    value = jax.jit(lambda r: IntervalProbe(rate=r).log_likelihood(data))(1.0)
    np.testing.assert_array_equal(value[:4], [-1, 0, -1, -2])
    assert np.isnan(value[4])

    # The hook is linear in rate; invalid rows do not contaminate valid rows.
    def fn(r):
        return IntervalProbe(rate=r).log_likelihood(data)[:4].sum()

    assert jax.grad(fn)(1.0) == -3
    assert jax.hessian(fn)(1.0) == -1
