"""Density policy at infinity, from specification 4.5 revision 22."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
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
