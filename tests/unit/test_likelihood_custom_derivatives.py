"""Generic density and reference substitution derivatives (spec 4.2/7.2–7.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import (
    ATOL,
    DERIVATIVE_ATOL,
    DERIVATIVE_RTOL,
    LIKELIHOOD_GRAD_EPS,
    RTOL,
)
from jax.test_util import check_grads

import survjax


@survjax.register_family
@dataclass(frozen=True)
class PowerSurvival(survjax.SurvivalDistribution):
    shape: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def log_survival(self, t: jax.Array) -> jax.Array:
        return -(t**self.shape)

    def reference_time(self) -> jax.Array:
        return jnp.ones_like(self.shape)


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_generic_density_parameter_derivatives(compiled, x64_enabled):
    """Spec 4.2/7.3: density autodiff adds a derivative order to the recipe."""
    time, event = jnp.array([0.5, 1.0, 1.5]), jnp.array([True, False, True])
    data = survjax.SurvivalData(time, event)

    def loss(k):
        return PowerSurvival(shape=k).log_likelihood(data).mean()

    evaluate = jax.jit(loss) if compiled else loss
    k = jnp.asarray(1.5)
    t, e = np.asarray(time, dtype=float), np.asarray(event, dtype=float)
    u, h = np.log(t), t ** float(k)
    # Mean e*(log(k)+(k-1)*log(t))-t**k; differentiate twice in k.
    expected = [
        np.mean(e * (np.log(float(k)) + (float(k) - 1) * u) - h),
        np.mean(e / float(k) + (e - h) * u),
        np.mean(-e / float(k) ** 2 - h * u**2),
    ]
    for actual, wanted in zip(
        [
            evaluate(k),
            jax.jacfwd(evaluate)(k),
            jax.jacrev(evaluate)(k),
            jax.jacfwd(jax.jacfwd(evaluate))(k),
            jax.hessian(evaluate)(k),
        ],
        [expected[0], expected[1], expected[1], expected[2], expected[2]],
        strict=True,
    ):
        np.testing.assert_allclose(
            actual, wanted, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
    mapped = jax.vmap(jax.hessian(evaluate))(jnp.stack([k, k]))
    np.testing.assert_allclose(
        mapped, [expected[2]] * 2, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    check_grads(
        evaluate,
        (k,),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
        eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
    )


@survjax.register_family
@dataclass(frozen=True)
class SingularReference(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def log_survival(self, t: jax.Array) -> jax.Array:
        return -self.rate * t

    def reference_time(self) -> jax.Array:
        # Spec 7.2: finite value 1 at r=1, but dr reference=+inf must not leak.
        return 1 + jnp.sqrt(jnp.abs(self.rate - 1))


@pytest.mark.parametrize(
    "event", [[False, False, False], [True, False, True]], ids=["right-only", "mixed"]
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_substituted_reference_derivatives_are_isolated(event, compiled, x64_enabled):
    """Spec 7.2/12.1: inactive density references cannot change real derivatives."""

    def loss(rate):
        data = survjax.SurvivalData(jnp.array([0.5, 1.0, 1.5]), jnp.array(event))
        return SingularReference(rate=rate).log_likelihood(data).sum()

    evaluate = jax.jit(loss) if compiled else loss
    # Sum e*log(r)-r*T, r=1, T=3: value=-3, gradient=d-3, Hessian=-d.
    rate, d = jnp.asarray(1.0), sum(event)
    np.testing.assert_allclose(
        evaluate(rate), -3, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            first(evaluate)(rate), d - 3, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
        for second in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                second(first(evaluate))(rate),
                -d,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
