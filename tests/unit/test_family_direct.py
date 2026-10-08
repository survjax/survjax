"""Direct-method defaults/policy (spec 4.2, 4.5, 5, 7.2–7.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import ATOL, DERIVATIVE_ATOL, DERIVATIVE_RTOL, RTOL
from jax.test_util import check_grads

import survjax


@survjax.register_family
@dataclass(frozen=True)
class Constant(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param()

    def log_survival(self, t):
        return -self.rate * t

    def reference_time(self):
        return 1.0 / self.rate


@survjax.register_family
@dataclass(frozen=True)
class Root(Constant):
    def log_survival(self, t):
        return -self.rate * jnp.sqrt(t)

    def reference_time(self):
        return 4.0 / self.rate**2


@survjax.register_family
@dataclass(frozen=True)
class Overrides(Constant):
    def survival(self, t):
        return jnp.exp(-self.rate * t)

    def log_cdf(self, t):
        return jnp.log(-jnp.expm1(-self.rate * t))

    def cumulative_hazard(self, t):
        return self.rate * t


@pytest.mark.parametrize(
    "name, expected",
    [
        ("log_survival", [0, 0, 0, -2, -np.inf, np.nan]),
        ("survival", [1, 1, 1, np.exp(-2), 0, np.nan]),
        ("log_cdf", [-np.inf, -np.inf, -np.inf, np.log(-np.expm1(-2)), 0, np.nan]),
        ("cumulative_hazard", [0, 0, 0, 2, np.inf, np.nan]),
        ("log_prob", [-np.inf, -np.inf, np.log(2), np.log(2) - 2, -np.inf, np.nan]),
        ("log_hazard", [-np.inf, -np.inf, np.log(2), np.log(2), np.nan, np.nan]),
        ("hazard", [0, 0, 2, 2, np.nan, np.nan]),
    ],
)
def test_mixed_time_policy_eager_jit_vmap(name, expected, x64_enabled):
    """Spec 4.5: exponential identities and all time-policy rows agree."""
    dist = Constant(rate=2)
    times = jnp.array([-jnp.inf, -1, 0, 1, jnp.inf, jnp.nan])
    method = getattr(dist, name)
    for actual in (method(times), jax.jit(method)(times), jax.vmap(method)(times)):
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_paired_and_draw_broadcasting_and_shape_misuse():
    """Spec 5: (N,) pairs; (D,1) broadcasts to (D,N), with no implicit grid."""
    dist = Constant(rate=[2, 3])
    np.testing.assert_array_equal(dist.cumulative_hazard([4, 5]), [8, 15])
    np.testing.assert_array_equal(
        dist[:, None].cumulative_hazard([4, 5, 6]), [[8, 10, 12], [12, 15, 18]]
    )
    with pytest.raises(TypeError, match="batch shapes"):
        jax.jit(dist.survival)(jnp.ones(3))


@pytest.mark.parametrize(
    "name",
    [
        "log_survival",
        "survival",
        "log_cdf",
        "cumulative_hazard",
        "log_prob",
        "log_hazard",
        "hazard",
    ],
)
def test_safe_reference_substitution_and_inactive_gradients(name):
    """Spec 7.2: sqrt receives positive references, including NaN time slots."""

    def value(rate, time):
        return getattr(Root(rate=rate), name)(time)

    for time in (-1.0, -jnp.inf, jnp.nan):
        result, derivative = jax.jit(jax.value_and_grad(value))(jnp.asarray(2.0), time)
        assert np.isnan(result) if np.isnan(time) else not np.isnan(result)
        np.testing.assert_array_equal(derivative, 0)


@pytest.mark.parametrize("name", ["survival", "log_cdf", "cumulative_hazard"])
def test_overrides_and_decorated_subclasses_dispatch_without_primitive(
    name, monkeypatch, x64_enabled
):
    """Spec 4.2/4.5: inherited overrides retain their dispatch and time policy."""

    @survjax.register_family
    @dataclass(frozen=True)
    class Child(Overrides):
        pass

    def unused(self, t):
        raise AssertionError("derived override must not call the primitive")

    monkeypatch.setattr(Child, "log_survival", unused)
    dist = Child(rate=2)
    expected = {
        "survival": np.exp(-2),
        "log_cdf": np.log(-np.expm1(-2)),
        "cumulative_hazard": 2,
    }
    np.testing.assert_allclose(
        getattr(dist, name)(1),
        expected[name],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    # No nested policy wrappers are retained after subclass registration.
    assert not hasattr(getattr(Child, name)._survjax_raw, "_survjax_raw")
    negative = {"survival": 1, "log_cdf": -np.inf, "cumulative_hazard": 0}
    np.testing.assert_array_equal(getattr(dist, name)(-1), negative[name])
    assert np.isnan(getattr(dist, name)(jnp.nan))


@pytest.mark.parametrize(
    "name", ["log_survival", "survival", "log_cdf", "cumulative_hazard"]
)
def test_nan_parameters_propagate_without_poisoning_valid_entries(name):
    """Spec 7.1/4.5: preserve NaNs even on negative-time policy slots."""
    dist = Constant(rate=[jnp.nan, 2])
    actual = getattr(dist, name)(-1)
    assert np.isnan(actual[0]) and not np.isnan(actual[1])
    assert np.isnan(getattr(dist, name)(1)[0])


def test_log_cdf_tiny_mass_against_mpmath(x64_enabled):
    """Spec 4.2: stable log-CDF does not subtract probabilities near one."""
    with mp.workdps(100):
        expected = float(mp.log(-mp.expm1(-mp.mpf("1e-20"))))
    np.testing.assert_allclose(
        Constant(rate=1).log_cdf(1e-20),
        expected,
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    np.testing.assert_array_equal(Constant(rate=1).log_cdf(0), -np.inf)
    np.testing.assert_array_equal(
        jax.grad(lambda r: Constant(rate=r).log_cdf(0))(1.0), 0
    )


@pytest.mark.parametrize(
    "name", ["log_survival", "survival", "log_cdf", "cumulative_hazard"]
)
def test_second_order_parameter_derivatives_and_purity(name, x64_enabled):
    """Spec 7.3: valid interior derivatives agree forward/reverse to order two."""

    def value(rate):
        return getattr(Root(rate=rate), name)(jnp.asarray(4.0))

    check_grads(
        value,
        (jnp.asarray(1.5),),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )
    dist = Root(rate=1.5)
    before = jax.tree.leaves(dist)
    getattr(dist, name)(2)
    assert all(a is b for a, b in zip(before, jax.tree.leaves(dist), strict=True))


@pytest.mark.parametrize(
    "name",
    [
        "log_survival",
        "survival",
        "log_cdf",
        "cumulative_hazard",
        "log_prob",
        "log_hazard",
        "hazard",
    ],
)
@pytest.mark.parametrize(
    "time",
    [
        True,
        np.array([True, False]),
        1j,
        np.array([1 + 0j]),
        "1",
        np.array([1], dtype=object),
    ],
)
def test_direct_times_reject_dtype_misuse(name, time):
    """Spec 4.5 rev 21: nonreal time dtypes are static argument misuse."""
    with pytest.raises(TypeError):
        getattr(Constant(rate=2), name)(time)


@pytest.mark.parametrize(
    "name",
    [
        "log_survival",
        "survival",
        "log_cdf",
        "cumulative_hazard",
        "log_prob",
        "log_hazard",
        "hazard",
    ],
)
@pytest.mark.parametrize("time", [np.array([True, False]), np.array([1j, 2j])])
def test_direct_dtype_rejection_works_on_tracers(name, time):
    """Spec 4.5 rev 21: dtype-only rejection composes with jit and vmap."""
    method = getattr(Constant(rate=2), name)
    with pytest.raises(TypeError):
        jax.jit(method)(time)
    with pytest.raises(TypeError):
        jax.vmap(method)(time)


@pytest.mark.parametrize("name", ["log_survival", "log_prob", "log_hazard", "hazard"])
def test_float64_times_follow_promotion_without_warning(name, x64_enabled):
    """Spec 4.5 rev 21: transient float64 times convert silently under x32."""
    import warnings

    dist = Constant(rate=np.float32(2))
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        actual = getattr(dist, name)(np.array([1.0, 2.0], dtype=np.float64))
    assert recorded == []
    assert actual.dtype == (np.float64 if x64_enabled else np.float32)
    # Hand derivation: log S=-2*t, log f=log(2)-2*t, log h=log(2), h=2.
    expected = {
        "log_survival": [-2, -4],
        "log_prob": np.log(2) - np.array([2, 4]),
        "log_hazard": np.log(2),
        "hazard": 2,
    }
    np.testing.assert_allclose(
        actual, expected[name], atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


def test_zero_time_reaches_family_time_derivative():
    """Spec 4.5: t=0 is active, needed by A4.5's density autodiff default."""

    # Hand derivation: d(-rate*t)/dt = -rate = -2, also at zero.
    def value(t):
        return Constant(rate=2).log_survival(t)

    np.testing.assert_array_equal(jax.grad(value)(0.0), -2.0)
    np.testing.assert_array_equal(jax.jit(jax.grad(value))(0.0), -2.0)
