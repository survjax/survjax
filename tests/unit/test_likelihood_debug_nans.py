"""Eager valid mixed likelihoods and inactive density safety (spec 7.2/12.1)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL, TAIL_ATOL, TAIL_RTOL

import survjax


@survjax.register_family
@dataclass(frozen=True)
class RootHazard(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def log_survival(self, t: jax.Array) -> jax.Array:
        # Spec 4.2/7.2: the generic density at zero is deliberately unsafe.
        return -self.rate * jnp.sqrt(t)

    def reference_time(self) -> jax.Array:
        return jnp.ones_like(self.rate)


@survjax.register_family
@dataclass(frozen=True)
class Cure(survjax.SurvivalDistribution):
    fraction: jax.Array = survjax.param(constraint=survjax.constraints.unit_interval)
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def log_survival(self, t: jax.Array) -> jax.Array:
        return jnp.logaddexp(
            jnp.log(self.fraction), jnp.log1p(-self.fraction) - self.rate * t
        )

    def log_survival_at_infinity(self) -> jax.Array:
        return jnp.log(self.fraction)

    def reference_time(self) -> jax.Array:
        return jnp.ones_like(self.rate)


def _check_eager(
    rows,
    parameters,
    expected_values,
    expected_gradient,
    x64_enabled,
    *,
    tiny_column=None,
):
    with jax.debug_nans(True):
        values = rows(parameters)
        gradient = jax.grad(lambda p: rows(p).sum())(parameters)
    # Spec 12.1: debugging alone is not evidence of mathematical correctness.
    np.testing.assert_allclose(
        values,
        expected_values,
        atol=TAIL_ATOL[x64_enabled],
        rtol=TAIL_RTOL[x64_enabled],
    )
    np.testing.assert_allclose(
        gradient, expected_gradient, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    gradient, reference = np.asarray(gradient), np.asarray(expected_gradient)
    np.testing.assert_array_equal(gradient[reference == 0], 0)
    if tiny_column is not None:
        # Spec 12.1: an absolute tolerance must not accept lost tiny derivatives.
        np.testing.assert_allclose(
            gradient[..., tiny_column],
            reference[..., tiny_column],
            atol=TAIL_ATOL[x64_enabled],
            rtol=TAIL_RTOL[x64_enabled],
        )
    assert values[0] == 0


@pytest.mark.parametrize("per_row", [False, True], ids=["shared", "per-row"])
def test_exponential_mixed_values_and_gradients_under_debug_nans(per_row, x64_enabled):
    """Spec 7.2/12.1: interior, lower/upper tails and c=0 are safe eagerly."""
    time, event = (
        jnp.array([0.0, 1e-20, 1.0, 4e6, 1e-10, 2.0]),
        jnp.array([False, True, True, True, False, False]),
    )
    rates = jnp.array([1.0, 0.5, 2.0, 1.5, 3.0, 1.0] if per_row else 2.0)

    def rows(r):
        return survjax.Exponential(rate=r).log_likelihood(time=time, event=event)

    # log L=event*log r-r*t; d/dr=event/r-t, exactly zero at c=0.
    t = np.asarray(time, dtype=float)
    e = np.asarray(event)
    r = np.asarray(rates, dtype=float)
    expected = e * np.log(r) - r * t
    derivative = e / r - t
    _check_eager(
        rows,
        rates,
        expected,
        derivative if per_row else derivative.sum(),
        x64_enabled,
        tiny_column=4 if per_row else None,
    )


@pytest.mark.parametrize("mode", ["shared", "per-row", "from-log"])
def test_weibull_mixed_values_and_gradients_under_debug_nans(mode, x64_enabled):
    """Spec 4.1/7.2/12.1: both natural and log parameters have safe mixed batches."""
    time, event = (
        jnp.array([0.0, 1e-20, 1.0, 4e6, 1e-10, 2.0]),
        jnp.array([False, True, True, True, False, False]),
    )
    parameters = jnp.array(
        [[0.5, 0.5, 2.0, 1.5, 1.0, 2.0], [2.0, 2.0, 1.0, 1.0, 2.0, 2.0]]
        if mode == "per-row"
        else [0.5, 2.0]
    )
    if mode == "from-log":
        parameters = jnp.log(jnp.array([1.5, 2.0]))
    data = survjax.SurvivalData.from_bounds(
        time, jnp.where(event, time, jnp.inf), exact=event
    )

    def rows(p):
        dist = (
            survjax.Weibull.from_log(log_shape=p[0], log_scale=p[1])
            if mode == "from-log"
            else survjax.Weibull(shape=p[0], scale=p[1])
        )
        return dist.log_likelihood(data)

    p = np.asarray(parameters, dtype=float)
    natural = np.exp(p) if mode == "from-log" else p
    k, b = natural
    t, e = np.asarray(time, dtype=float), np.asarray(event)
    # For c=0 use its known constant value/derivative, never log(0).
    log_ratio = np.log(np.where(t == 0, b, t)) - np.log(b)
    cumulative_hazard = np.where(t == 0, 0.0, np.exp(k * log_ratio))
    expected = e * (np.log(k) - np.log(b) + (k - 1) * log_ratio) - cumulative_hazard
    derivative = np.stack(
        [e / k + (e - cumulative_hazard) * log_ratio, k * (cumulative_hazard - e) / b]
    )
    if mode != "per-row":
        derivative = derivative.sum(axis=1)
    if mode == "from-log":
        derivative = derivative * natural
    _check_eager(
        rows,
        parameters,
        expected,
        derivative,
        x64_enabled,
        tiny_column=4 if mode == "per-row" else None,
    )


@pytest.mark.parametrize("debug", [False, True], ids=["ordinary", "debug"])
def test_custom_density_at_zero_is_inactive(debug, x64_enabled):
    """Spec 4.2/7.2: reviewer S1, singular generic density must use the reference."""
    time = jnp.array([0.0, 1.0])
    event = jnp.array([False, True])
    rate = jnp.asarray(2.0)

    def rows(r, t, e):
        dist = RootHazard(rate=r)
        return dist.log_likelihood(time=t, event=e)

    # f(1)=r/2*exp(-r): at r=2, rows=[0,-2], d/dr sum=1/r-1=-.5.
    with jax.debug_nans(debug):
        for run in (rows, jax.jit(rows), jax.vmap(rows, in_axes=(None, 0, 0))):
            values = run(rate, time, event)
            gradient = jax.grad(lambda r, run=run: run(r, time, event).sum())(rate)
            assert values[0] == 0
            np.testing.assert_allclose(
                values, [0, -2], atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
            )
            np.testing.assert_allclose(
                gradient, -0.5, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
            )
        for differentiate in (jax.jacfwd, jax.jacrev):
            assert differentiate(lambda r: rows(r, time, event)[0])(rate) == 0


def test_cure_mixed_values_and_gradients_under_debug_nans(x64_enabled):
    """Spec 2/4.2/7.2: right censoring includes cure mass; generic exact density works."""
    time, event = (
        jnp.array([0.0, 1.0, 2.0, 200.0]),
        jnp.array([False, True, False, False]),
    )
    parameters = jnp.array([0.25, 0.5])

    def rows(p):
        return Cure(fraction=p[0], rate=p[1]).log_likelihood(time=time, event=event)

    # S=p+(1-p)*exp(-r*t); f=(1-p)*r*exp(-r*t), including S(0)=1.
    t, e = np.asarray(time, dtype=float), np.asarray(event)
    p, r = np.asarray(parameters, dtype=float)
    q = np.exp(-r * t)
    s = p + (1 - p) * q
    expected = np.where(e, np.log((1 - p) * r) - r * t, np.log(s))
    # At r*t=100, the residual survival term is below rounding in both dtypes.
    expected[-1] = np.log(p)
    derivative = [
        np.where(e, -1 / (1 - p), (1 - q) / s).sum(),
        np.where(e, 1 / r - t, -(1 - p) * t * q / s).sum(),
    ]
    _check_eager(rows, parameters, expected, derivative, x64_enabled)
