"""AFT numerical and integration identities, spec 4.4–4.5, 5 and 7."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL, TAIL_RTOL

from survjax import (
    AcceleratedFailureTime,
    Exponential,
    SurvivalData,
    SurvivalDistribution,
    Weibull,
    constraints,
    param,
    register_family,
)


@pytest.mark.parametrize("sign", [-1, 1])
def test_extreme_predictor_preserves_representable_time(sign, x64_enabled):
    """Spec 4.4/7.1: t*exp(-eta) can be normal even when exp(-eta) is not."""
    eta = sign * (720.0 if x64_enabled else 90.0)
    time = (1e300 if x64_enabled else 1e30) ** sign
    # Unit-rate baseline: log S=-exp(log(t)-eta), log f=log S-eta,
    # log h=-eta. Host float64 arithmetic supplies the analytical reference.
    expected_log_s = -np.exp(np.log(time) - eta)
    dist = AcceleratedFailureTime(baseline=Exponential(rate=1), log_time_ratio=eta)
    for fn in (dist.log_survival, jax.jit(dist.log_survival)):
        np.testing.assert_allclose(
            fn(time), expected_log_s, atol=0, rtol=TAIL_RTOL[x64_enabled]
        )
    np.testing.assert_allclose(
        dist.log_prob(time), expected_log_s - eta, atol=0, rtol=TAIL_RTOL[x64_enabled]
    )
    np.testing.assert_allclose(
        dist.log_hazard(time), -eta, atol=0, rtol=TAIL_RTOL[x64_enabled]
    )

    def log_s(e):
        return AcceleratedFailureTime(
            baseline=Exponential(rate=1), log_time_ratio=e
        ).log_survival(time)

    # d/deta log S=-log S; d²/deta² log S=log S.
    for derivative, expected in (
        (jax.jacfwd, -expected_log_s),
        (jax.jacrev, -expected_log_s),
        (jax.hessian, expected_log_s),
    ):
        np.testing.assert_allclose(
            jax.jit(derivative(log_s))(eta),
            expected,
            atol=0,
            rtol=TAIL_RTOL[x64_enabled],
        )

    def joint(p):
        return AcceleratedFailureTime(
            baseline=Exponential(rate=p[0]), log_time_ratio=p[1]
        ).log_survival(time)

    # At rate=1, d/dr log S=-u, d/deta log S=u;
    # H_rr=0, H_r,eta=u, H_eta,eta=-u. Log S=-u above.
    p = jnp.array([1.0, eta])
    for transform, expected in (
        (jax.jacfwd, [expected_log_s, -expected_log_s]),
        (jax.jacrev, [expected_log_s, -expected_log_s]),
        (jax.hessian, [[0, -expected_log_s], [-expected_log_s, expected_log_s]]),
    ):
        np.testing.assert_allclose(
            jax.jit(transform(joint))(p),
            expected,
            atol=0,
            rtol=TAIL_RTOL[x64_enabled],
        )


@register_family
@dataclass(frozen=True)
class CureBaseline(SurvivalDistribution):
    """S(t)=exp(-c*(1-exp(-t))); atom exp(-c) at infinity."""

    amount: jax.Array = param(constraint=constraints.positive)

    def log_survival(self, t):
        return self.amount * jnp.expm1(-jnp.asarray(t))

    def reference_time(self):
        return jnp.ones_like(self.amount)

    def log_survival_at_infinity(self):
        return -self.amount


@pytest.mark.parametrize("name", ["log_survival", "log_prob", "log_hazard"])
def test_custom_baseline_generic_direct_methods(name, x64_enabled):
    """Spec 4.2/4.4: generic baseline density/hazard compose with AFT."""
    time = np.array([0.2, 0.8, 2.0])
    eta, amount = 0.4, 1.3
    u = time * np.exp(-eta)
    # dH0/du=c*exp(-u); log h=log(c)-u-eta; log f=log h+log S.
    log_s = amount * np.expm1(-u)
    log_h = np.log(amount) - u - eta
    expected = {"log_survival": log_s, "log_hazard": log_h, "log_prob": log_h + log_s}[
        name
    ]
    dist = AcceleratedFailureTime(
        baseline=CureBaseline(amount=amount), log_time_ratio=eta
    )
    for evaluate in (
        getattr(dist, name),
        jax.jit(getattr(dist, name)),
        jax.vmap(getattr(dist, name)),
    ):
        np.testing.assert_allclose(
            evaluate(jnp.asarray(time)),
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )


def test_improper_infinity_limit_and_parameter_derivatives(x64_enabled):
    """Spec 4.4–4.5/7.3: AFT leaves cure mass exp(-c) independent of eta."""

    def fn(p):
        return AcceleratedFailureTime(
            baseline=CureBaseline(amount=p[0]), log_time_ratio=p[1]
        ).survival(jnp.inf)

    p = jnp.array([1.3, 0.4])
    mass = np.exp(-1.3)
    for transform, expected in (
        (lambda f: f, mass),
        (jax.jacfwd, [-mass, 0]),
        (jax.jacrev, [-mass, 0]),
        (jax.hessian, [[mass, 0], [0, 0]]),
    ):
        np.testing.assert_allclose(
            jax.jit(transform(fn))(p),
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )


def test_nested_wrappers_time_first_grid(x64_enabled):
    """Spec 4.4/5: nested eta values add, broadcasting (2,1) with (2,2)."""
    rate = np.array([[1.0], [2.0]])
    inner_eta = np.array([[0.2], [-0.3]])
    outer_eta = np.array([[0.1, 0.4], [0.5, -0.2]])
    dist = AcceleratedFailureTime(
        baseline=AcceleratedFailureTime(
            baseline=Exponential(rate=rate.tolist()), log_time_ratio=inner_eta.tolist()
        ),
        log_time_ratio=outer_eta.tolist(),
    )
    times = jnp.array([0.0, 0.5, 2.0, jnp.inf])
    # Nested rescalings multiply: S(t)=exp(-r*t*exp(-inner_eta-outer_eta)).
    equivalent_rate = rate * np.exp(-inner_eta - outer_eta)
    expected = np.exp(-np.asarray(times)[:, None, None] * equivalent_rate)
    assert dist.batch_shape == dist.baseline.baseline.batch_shape == (2, 2)
    for actual in (
        dist.survival_grid(times),
        jax.jit(dist.survival_grid)(times),
        jax.vmap(dist.survival)(times),
    ):
        assert actual.shape == (4, 2, 2)
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


@pytest.mark.parametrize("family", ["weibull", "exponential"])
def test_likelihood_values_and_mixed_parameter_hessians(family, x64_enabled):
    """Spec 4.4/6.3/7.3: mixed exact/right likelihood and full parameter Hessians."""
    data = SurvivalData([0.7, 1.8], [1, 0])

    def contributions(p):
        baseline = (
            Weibull(shape=p[0], scale=p[1])
            if family == "weibull"
            else Exponential(rate=p[0])
        )
        return AcceleratedFailureTime(
            baseline=baseline, log_time_ratio=p[-1]
        ).log_likelihood(data)

    def analytical(p):
        t = jnp.array([0.7, 1.8])
        if family == "weibull":
            k, b, eta = p
            log_u = jnp.log(t) - jnp.log(b) - eta
            log_s = -jnp.exp(k * log_u)
            log_h = jnp.log(k) - jnp.log(b) - eta + (k - 1) * log_u
        else:
            rate, eta = p
            log_s = -rate * t * jnp.exp(-eta)
            log_h = jnp.log(rate) - eta
        return log_s + jnp.array([1, 0]) * log_h

    p = jnp.array([1.4, 2.0, 0.3] if family == "weibull" else [1.2, 0.3])
    equivalent = (
        Weibull(shape=p[0], scale=p[1] * jnp.exp(p[-1]))
        if family == "weibull"
        else Exponential(rate=p[0] * jnp.exp(-p[-1]))
    )
    with jax.debug_nans(True):
        actual = contributions(p)
    np.testing.assert_allclose(
        actual,
        equivalent.log_likelihood(data),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    for transform in (
        lambda f: f,
        jax.jacfwd,
        jax.jacrev,
        lambda f: jax.jacfwd(jax.jacfwd(f)),
        lambda f: jax.jacrev(jax.jacrev(f)),
        lambda f: jax.jacfwd(jax.jacrev(f)),
    ):
        expected = transform(analytical)(p)
        evaluate = transform(contributions)
        for actual in (
            evaluate(p),
            jax.jit(evaluate)(p),
        ):
            np.testing.assert_allclose(
                actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
            )

        batch = jnp.stack([p, p + 0.1])
        expected_batch = jnp.stack([expected, transform(analytical)(p + 0.1)])
        np.testing.assert_allclose(
            jax.jit(jax.vmap(evaluate))(batch),
            expected_batch,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )


def test_mixed_invalid_baseline_and_predictor_isolate_valid_rows(x64_enabled):
    """Spec 7.1–7.3: nested invalid entries do not poison valid parameter Hessians."""
    data = SurvivalData([1.0, 1.0, 1.0], [1, 1, 1])

    def rows(p):
        return AcceleratedFailureTime(
            baseline=Exponential(rate=p[:, 0]), log_time_ratio=p[:, 1]
        ).log_likelihood(data)

    p = jnp.array([[2.0, 0.0], [-1.0, 0.0], [2.0, jnp.nan]])
    values = jax.jit(rows)(p)
    assert jnp.isnan(values[1:]).all()
    np.testing.assert_allclose(
        values[0], np.log(2) - 2, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    # Valid row: log r-eta-r*exp(-eta). At r=2, eta=0,
    # gradient=(-.5,1), Hessian=((-1/4,1),(1,-2)); all other leaves inactive.
    expected_gradient = np.zeros((3, 2))
    expected_gradient[0] = [-0.5, 1]
    expected_hessian = np.zeros((3, 2, 3, 2))
    expected_hessian[0, :, 0, :] = [[-0.25, 1], [1, -2]]

    def fn(p):
        return rows(p)[0]

    for transform, expected in (
        (jax.jacfwd, expected_gradient),
        (jax.jacrev, expected_gradient),
        (jax.hessian, expected_hessian),
    ):
        np.testing.assert_allclose(
            jax.jit(transform(fn))(p),
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
