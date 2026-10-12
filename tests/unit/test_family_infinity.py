"""Infinity values and finite boundary gradients (spec 4.5/7.2 revision 22)."""

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
class FiniteDensity(Constant):
    def log_prob(self, t: ArrayLike) -> jax.Array:
        # Deliberately independent of the primitive to distinguish policy dispatch.
        return jnp.full_like(jnp.asarray(t), 17.0)


@survjax.register_family
@dataclass(frozen=True)
class NaNDensity(Constant):
    def log_prob(self, t: ArrayLike) -> jax.Array:
        return jnp.full_like(jnp.asarray(t), jnp.nan)


@pytest.mark.parametrize("family", [Constant, Quadratic])
def test_nan_parameter_density_at_infinity_is_nan(family):
    """Spec 4.5: NaN log S(+inf) gives NaN density, eagerly/jit/vmap."""
    dist = family(rate=jnp.nan)
    assert np.isnan(dist.log_prob(jnp.inf))
    assert np.isnan(jax.jit(lambda d: d.log_prob(jnp.inf))(dist))
    assert np.isnan(jax.vmap(dist.log_prob)(jnp.array([jnp.inf]))).all()


@pytest.mark.parametrize("family", [Constant, Quadratic])
def test_infinite_density_keeps_nan_and_valid_batch_entries_separate(family):
    """Spec 4.5: log S=-r*t or -r*t**2; valid limits give -inf density.

    For r=2, the quadratic density formula at infinity has inf-inf. That
    is not an invalid parameter: the policy still gives -inf beside NaN r.
    """
    dist = family(rate=[jnp.nan, 2.0])
    for result in (
        dist.log_prob(jnp.inf),
        jax.jit(lambda d: d.log_prob(jnp.inf))(dist),
        jax.vmap(lambda d: d.log_prob(jnp.inf))(dist),
    ):
        np.testing.assert_array_equal(result, [np.nan, -np.inf])


def test_infinite_density_uses_survival_nan_despite_finite_override():
    """Spec 4.5: a finite density override cannot hide NaN log S(+inf)."""
    dist = FiniteDensity(rate=[jnp.nan, 2.0])
    np.testing.assert_array_equal(dist.log_prob(jnp.inf), [np.nan, -np.inf])
    np.testing.assert_array_equal(
        jax.jit(lambda d: d.log_prob(jnp.inf))(dist), [np.nan, -np.inf]
    )


def test_infinite_density_ignores_nan_override_for_defined_survival_limit():
    """Spec 4.5: even a NaN override gives -inf if log S(+inf) is defined."""
    dist = NaNDensity(rate=[2.0, jnp.nan])
    np.testing.assert_array_equal(dist.log_prob(jnp.inf), [-np.inf, np.nan])
    np.testing.assert_array_equal(
        jax.vmap(lambda d: d.log_prob(jnp.inf))(dist), [-np.inf, np.nan]
    )


def test_density_override_keeps_other_time_policy_rows():
    """Spec 4.2/4.5: the infinity exception changes no other override rows.

    The raw override is 17 even for a NaN rate. Negative times therefore
    retain the -inf policy; finite nonnegative times retain 17. Only at
    +inf does the primitive's NaN require a NaN density result.
    """
    times = jnp.array([-1.0, 0.0, 1.0, jnp.inf, jnp.nan])
    np.testing.assert_array_equal(
        FiniteDensity(rate=jnp.nan).log_prob(times),
        [-np.inf, 17.0, 17.0, np.nan, np.nan],
    )
    np.testing.assert_array_equal(
        FiniteDensity(rate=2.0).log_prob(times),
        [-np.inf, 17.0, 17.0, -np.inf, np.nan],
    )


def test_nan_density_override_is_unchanged_at_finite_time():
    """Spec 4.5: a NaN family density at finite t remains NaN."""
    assert np.isnan(NaNDensity(rate=2.0).log_prob(1.0))


@survjax.register_family
@dataclass(frozen=True)
class Cure(Constant):
    fraction: jax.Array = survjax.param()

    def log_survival(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        infinite = jnp.isposinf(time)
        # spec 7.2: guard multiplication before selecting the cure limit.
        safe_time = jnp.where(
            infinite, jax.lax.stop_gradient(jnp.ones_like(time)), time
        )
        remaining = jnp.exp(-self.rate * safe_time)
        remaining = jnp.where(infinite & ~jnp.isnan(remaining), 0.0, remaining)
        return jnp.log(self.fraction + (1.0 - self.fraction) * remaining)


@survjax.register_family
@dataclass(frozen=True)
class Override(Constant):
    def survival(self, t: ArrayLike) -> jax.Array:
        return jnp.full_like(jnp.asarray(t), 0.25)

    def log_cdf(self, t: ArrayLike) -> jax.Array:
        return jnp.full_like(jnp.asarray(t), -3.0)


@survjax.register_family
@dataclass(frozen=True)
class InheritedOverride(Override):
    pass


@pytest.mark.parametrize("family", [Constant, Quadratic])
@pytest.mark.parametrize("name", ["survival", "log_cdf"])
def test_proper_infinity_has_zero_parameter_gradient(family, name):
    """Spec 4.5/7.2: S(inf)=0, log F(inf)=0, independent of positive r."""

    def value(rate):
        return getattr(family(rate=rate), name)(jnp.inf)

    rates = jnp.array([1.0, 2.0])
    for function in (value, jax.jacfwd(value), jax.jacrev(value)):
        assert function(2.0) == 0.0
        assert jax.jit(function)(2.0) == 0.0
        np.testing.assert_array_equal(jax.jit(jax.vmap(function))(rates), [0.0, 0.0])


@pytest.mark.parametrize("name", ["survival", "log_cdf"])
def test_mixed_finite_infinite_gradient_and_hessian(name, x64_enabled):
    """Spec 7.2/7.3: the infinite entry adds zero to all finite derivatives."""
    rate = 2.0
    # S(1)=exp(-r); log F(1)=log(1-exp(-r)); differentiate analytically.
    if name == "survival":
        expected = [np.exp(-rate), -np.exp(-rate), np.exp(-rate)]
    else:
        expected = [
            np.log(-np.expm1(-rate)),
            1 / np.expm1(rate),
            -np.exp(rate) / np.expm1(rate) ** 2,
        ]

    def value(r):
        return getattr(Constant(rate=r), name)(jnp.array([1.0, jnp.inf])).sum()

    for function, result in zip(
        (value, jax.grad(value), jax.hessian(value)), expected, strict=True
    ):
        np.testing.assert_allclose(
            jax.jit(function)(rate),
            result,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
    np.testing.assert_allclose(
        jax.jit(jax.grad(value))(jnp.array([rate, rate])),
        [expected[1], 0.0],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


@pytest.mark.parametrize("name", ["survival", "log_cdf"])
def test_cure_limit_preserves_fraction_derivative(name, x64_enabled):
    """Spec 4.2/7.3: S=p+(1-p)exp(-rt), so S(inf)=p, log F(inf)=log(1-p)."""

    def value(parameters, time):
        return getattr(Cure(rate=parameters[0], fraction=parameters[1]), name)(time)

    parameters = jnp.array([2.0, 0.25])
    # Finite case derived from the same stated survival identity.
    for time in (1.0, jnp.inf):
        decay = np.exp(-2.0 * time)
        survival = 0.25 + 0.75 * decay
        ds = np.array([-0.75 * (time if np.isfinite(time) else 0.0) * decay, 1 - decay])
        expected = survival if name == "survival" else np.log1p(-survival)
        derivative = ds if name == "survival" else -ds / (1 - survival)
        for function in (value, jax.jit(value)):
            np.testing.assert_allclose(
                function(parameters, time),
                expected,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
        for transform in (jax.jacfwd, jax.jacrev):
            function = transform(lambda p, time=time: value(p, time))
            np.testing.assert_allclose(
                jax.jit(jax.vmap(function))(parameters[None, :]),
                derivative[None, :],
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )


@pytest.mark.parametrize("name", ["survival", "log_cdf"])
def test_infinite_survival_methods_keep_nan_entries(name):
    """Spec 4.5: a zero limit beside a NaN limit does not hide the NaN."""
    dist = Constant(rate=[2.0, jnp.nan])
    np.testing.assert_array_equal(
        jax.jit(lambda d: getattr(d, name)(jnp.inf))(dist), [0.0, np.nan]
    )


@pytest.mark.parametrize("family", [Override, InheritedOverride])
def test_infinite_survival_overrides_are_honored(family):
    """Spec 4.2: explicit and inherited overrides retain their own limits."""
    dist = family(rate=2.0)
    assert dist.survival(jnp.inf) == 0.25
    assert dist.log_cdf(jnp.inf) == -3.0


@pytest.mark.parametrize("name", ["survival", "log_cdf"])
def test_infinite_default_uses_one_reference_call(name, monkeypatch):
    """Spec 4.5: the infinity guard reuses the policy's reference input."""
    calls = []
    original = Constant.reference_time

    def reference_time(self):
        calls.append(None)
        return original(self)

    monkeypatch.setattr(Constant, "reference_time", reference_time)
    getattr(Constant(rate=2.0), name)(jnp.inf)
    assert len(calls) == 1


def test_finite_tail_and_lower_tail_remain_active(x64_enabled):
    """Spec 4.2/7.2: S=exp(-rt); log F has derivative t/expm1(rt)."""
    assert Constant(rate=1.0).survival(1000.0) == 0.0
    assert Constant(rate=1.0).log_cdf(1000.0) == 0.0
    time = 1e-5

    def value(r):
        return Constant(rate=r).log_cdf(time)

    np.testing.assert_allclose(
        [value(2.0), jax.grad(value)(2.0)],
        [np.log(-np.expm1(-2.0 * time)), time / np.expm1(2.0 * time)],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
