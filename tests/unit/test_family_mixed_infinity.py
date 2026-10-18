"""Finite-entry parameter derivatives beside infinity (spec 4.5/7.2–7.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL
from jax.typing import ArrayLike

import survjax


@survjax.register_family
@dataclass(frozen=True)
class Constant(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param()

    def log_survival(self, t: ArrayLike) -> jax.Array:
        return -self.rate * jnp.asarray(t)

    def reference_time(self) -> jax.Array:
        return 1.0 / self.rate


@survjax.register_family
@dataclass(frozen=True)
class Quadratic(Constant):
    def log_survival(self, t: ArrayLike) -> jax.Array:
        return -self.rate * jnp.asarray(t) ** 2


@survjax.register_family
@dataclass(frozen=True)
class Cure(Constant):
    fraction: jax.Array = survjax.param()

    def log_survival_at_infinity(self) -> jax.Array:
        return jnp.log(self.fraction)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        infinite = jnp.isposinf(time)
        safe_time = jnp.where(
            infinite, jax.lax.stop_gradient(jnp.ones_like(time)), time
        )
        decay = jnp.exp(-self.rate * safe_time)
        decay = jnp.where(infinite & ~jnp.isnan(decay), 0.0, decay)
        return jnp.log(self.fraction + (1.0 - self.fraction) * decay)


@survjax.register_family
@dataclass(frozen=True)
class Analytic(Constant):
    def log_hazard(self, t: ArrayLike) -> jax.Array:
        return jnp.log(self.rate) + jnp.zeros_like(jnp.asarray(t))

    def cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        return self.rate + jnp.zeros_like(jnp.asarray(t))


@survjax.register_family
@dataclass(frozen=True)
class InheritedAnalytic(Analytic):
    pass


@survjax.register_family
@dataclass(frozen=True)
class HazardOverride(Constant):
    def hazard(self, t: ArrayLike) -> jax.Array:
        return self.rate + jnp.zeros_like(jnp.asarray(t))


@survjax.register_family
@dataclass(frozen=True)
class DensityOverride(Constant):
    def log_prob(self, t: ArrayLike) -> jax.Array:
        # Deliberately use an infinity-sensitive formula to test override policy.
        return -self.rate * jnp.asarray(t)


@survjax.register_family
@dataclass(frozen=True)
class CureDensityOverride(Cure):
    def log_prob(self, t: ArrayLike) -> jax.Array:
        # Finite cure limit distinguishes all-infinity from zero-limit guarding.
        return -self.rate * jnp.asarray(t)


@survjax.register_family
@dataclass(frozen=True)
class NaNLimit(Constant):
    def log_survival_at_infinity(self) -> jax.Array:
        return jnp.full_like(self.rate, jnp.nan)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        infinite = jnp.isposinf(time)
        safe_time = jnp.where(infinite, jnp.ones_like(time), time)
        return jnp.where(infinite, jnp.nan, -self.rate * safe_time)


@pytest.mark.parametrize("family", [Constant, Quadratic])
@pytest.mark.parametrize(
    "name", ["log_survival", "cumulative_hazard", "log_prob", "log_hazard", "hazard"]
)
def test_finite_component_derivatives_beside_infinity(family, name, x64_enabled):
    """Spec 7.2/7.3: differentiate only the finite output, not a nonfinite sum."""
    # At t=1: log S=-r; H=r; log f=log(r)-r (+log(2) for quadratic);
    # log h=log(r) (+log(2) for quadratic); h=r (or 2r).
    multiplier = 2 if family is Quadratic else 1
    expected = {
        "log_survival": (-2.0, -1.0, 0.0),
        "cumulative_hazard": (2.0, 1.0, 0.0),
        "log_prob": (np.log(2 * multiplier) - 2, -0.5, -0.25),
        "log_hazard": (np.log(2 * multiplier), 0.5, -0.25),
        "hazard": (2 * multiplier, multiplier, 0.0),
    }[name]
    boundary = {
        "log_survival": -np.inf,
        "cumulative_hazard": np.inf,
        "log_prob": -np.inf,
        "log_hazard": np.nan,
        "hazard": np.nan,
    }[name]
    np.testing.assert_array_equal(getattr(family(rate=2.0), name)(jnp.inf), boundary)

    def value(rate):
        return getattr(family(rate=rate), name)(jnp.array([1.0, jnp.inf]))[0]

    for function, target in (
        (value, expected[0]),
        (jax.jacfwd(value), expected[1]),
        (jax.jacrev(value), expected[1]),
        (jax.hessian(value), expected[2]),
    ):
        for result in (
            function(2.0),
            jax.jit(function)(2.0),
            jax.jit(jax.vmap(function))(jnp.array([2.0, 3.0]))[0],
        ):
            np.testing.assert_allclose(
                result,
                target,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
    np.testing.assert_allclose(
        jax.jit(jax.jacrev(value))(jnp.array([2.0, 2.0])),
        [expected[1], 0.0],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


@pytest.mark.parametrize("name,sign", [("log_survival", 1), ("cumulative_hazard", -1)])
def test_cure_infinite_log_limit_remains_parameter_dependent(name, sign, x64_enabled):
    """Spec 4.2/7.3: log S(inf)=log(p), with derivatives 1/p and -1/p**2."""

    def value(fraction):
        return getattr(Cure(rate=2.0, fraction=fraction), name)(jnp.inf)

    for function, expected in (
        (value, sign * np.log(0.25)),
        (jax.jacfwd(value), sign * 4.0),
        (jax.jacrev(value), sign * 4.0),
        (jax.hessian(value), -sign * 16.0),
    ):
        np.testing.assert_allclose(
            jax.jit(function)(0.25),
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )


@pytest.mark.parametrize("family", [Analytic, InheritedAnalytic, HazardOverride])
def test_analytic_infinite_hazard_limits_keep_gradients(family, x64_enabled):
    """Spec 4.2/4.5: h=r and log h=log(r) overrides retain their derivatives."""
    names = (
        ["hazard"]
        if family is HazardOverride
        else ["hazard", "log_hazard", "cumulative_hazard"]
    )
    for name in names:

        def value(rate, name=name):
            return getattr(family(rate=rate), name)(jnp.inf)

        # H(inf)=+inf comes from the proper default, overriding the raw H=r.
        expected = (
            0.0 if name == "cumulative_hazard" else 0.5 if name == "log_hazard" else 1.0
        )
        for transform in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                jax.jit(transform(value))(2.0),
                expected,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )


def test_density_nan_status_comes_from_infinity_hook():
    """Spec 4.5: finite log S at reference cannot hide a NaN infinity hook."""
    dist = NaNLimit(rate=2.0)
    assert jnp.isfinite(dist.log_prob(1.0))
    assert jnp.isnan(jax.jit(lambda d: d.log_prob(jnp.inf))(dist))


def test_density_override_infinity_does_not_poison_finite_derivative():
    """Spec 4.5/7.2: policy overrides log f(inf); finite formula is -rt."""

    def value(rate):
        return DensityOverride(rate=rate).log_prob(jnp.array([1.0, jnp.inf]))[0]

    assert jnp.isneginf(DensityOverride(rate=2.0).log_prob(jnp.inf))
    assert jax.jacfwd(value)(2.0) == -1.0
    assert jax.jit(jax.jacrev(value))(2.0) == -1.0


def test_cure_density_override_infinity_preserves_finite_derivative():
    """Spec 4.5/7.2: finite log S(inf)=log(p) still needs density substitution."""

    def values(rate):
        return CureDensityOverride(rate=rate, fraction=0.25).log_prob(
            jnp.array([1.0, jnp.inf])
        )

    # Raw finite log f=-rt=-2 at t=1; policy log f(inf)=-inf, even for cures.
    np.testing.assert_array_equal(values(2.0), [-2.0, -np.inf])
    np.testing.assert_array_equal(jax.jit(values)(2.0), [-2.0, -np.inf])

    def finite_value(rate):
        return values(rate)[0]

    # d(-rt)/dr=-t=-1; differentiate only the finite component.
    for transform in (jax.jacfwd, jax.jacrev):
        derivative = transform(finite_value)
        assert derivative(2.0) == -1.0
        assert jax.jit(derivative)(2.0) == -1.0
        np.testing.assert_array_equal(
            jax.jit(jax.vmap(derivative))(jnp.array([2.0, 3.0])), [-1.0, -1.0]
        )


@pytest.mark.parametrize(
    "name", ["log_survival", "cumulative_hazard", "log_prob", "log_hazard", "hazard"]
)
def test_mixed_call_reuses_one_reference(name, monkeypatch):
    """Spec 4.5/7.2: reuse the reference for all substitutions in one call."""
    calls = []
    original = Constant.reference_time

    def reference_time(self):
        calls.append(None)
        return original(self)

    monkeypatch.setattr(Constant, "reference_time", reference_time)
    getattr(Constant(rate=2.0), name)(jnp.array([1.0, jnp.inf]))
    assert len(calls) == 1
