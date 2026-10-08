"""Hand-derived density/hazard defaults and derivatives (spec 4.2/4.5/7.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
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
class Power(Constant):
    def log_survival(self, t):
        return -(t**self.rate)


@survjax.register_family
@dataclass(frozen=True)
class Quadratic(Constant):
    def log_survival(self, t):
        return -self.rate * t**2


@survjax.register_family
@dataclass(frozen=True)
class Bounded(Constant):
    def log_survival(self, t):
        inside = t < self.rate
        safe_t = jnp.where(inside, t, 0.0)
        return jnp.where(inside, jnp.log1p(-safe_t / self.rate), -jnp.inf)


@pytest.mark.parametrize("name", ["log_prob", "log_hazard", "hazard"])
def test_constant_values_broadcast_transform_and_reuse(name, x64_enabled):
    """Spec 4.2/5/7.2: h=r, log h=log r, log f=log r-r*t; no grid."""
    traces = []

    def evaluate(dist, time):
        traces.append(None)
        return getattr(dist, name)(time)

    compiled = jax.jit(evaluate)
    for rates in ([1.0, 2.0], [3.0, 4.0]):
        dist = Constant(rate=rates)
        for time in (jnp.array([2, 3]), jnp.array([[2], [3], [4]])):
            expected = np.broadcast_to(rates, np.broadcast_shapes(time.shape, (2,)))
            if name != "hazard":
                expected = np.log(expected)
            if name == "log_prob":
                expected = expected - np.asarray(rates) * np.asarray(time)
            for actual in (getattr(dist, name)(time), compiled(dist, time)):
                np.testing.assert_allclose(
                    actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
                )
        np.testing.assert_allclose(
            jax.vmap(lambda d: getattr(d, name)(2))(dist), getattr(dist, name)(2)
        )
    assert len(traces) == 2
    with pytest.raises(TypeError, match="batch shapes"):
        compiled(dist, jnp.ones(3))


def test_power_boundaries_and_large_cumulative_hazard(x64_enabled):
    """Spec 4.2/4.5: h=k*t**(k-1); H cannot cancel from log h."""
    dist = Power(rate=[0.5, 1.0, 2.0])
    for name in ("log_prob", "log_hazard"):
        np.testing.assert_array_equal(getattr(dist, name)(0.0), [np.inf, 0, -np.inf])
    time = 1e6
    expected_h = np.log([0.5, 1.0, 2.0]) + np.array([-0.5, 0, 1.0]) * np.log(time)
    np.testing.assert_allclose(
        dist.log_hazard(time),
        expected_h,
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    np.testing.assert_allclose(
        dist.log_prob(time),
        expected_h - time ** np.array([0.5, 1, 2]),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    np.testing.assert_array_equal(dist.log_prob(jnp.inf), [-np.inf] * 3)
    assert np.isnan(dist.hazard(jnp.inf)).all()


def test_bounded_support_and_underflow(x64_enabled):
    """Spec 4.5: uniform h=1/(end-t); finite log S still permits hazard."""
    dist = Bounded(rate=4)
    np.testing.assert_allclose(
        dist.log_hazard([1, 3]),
        -np.log([3, 1]),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    assert np.isnan(jax.jit(dist.hazard)(jnp.array([4, 6, jnp.inf]))).all()
    assert Constant(rate=2).survival(1000) == 0
    np.testing.assert_allclose(
        Constant(rate=2).log_hazard(1000),
        np.log(2),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


@pytest.mark.parametrize("name", ["log_prob", "log_hazard", "hazard"])
def test_zero_hazard_does_not_poison_active_derivatives(name):
    """Spec 7.2: h=0 gives -inf logs/zero h; dh/dr=1 in the active slot."""

    def active(rate):
        return getattr(Constant(rate=jnp.stack([jnp.zeros_like(rate), rate])), name)(
            1.0
        )[1]

    result = getattr(Constant(rate=[0, 2]), name)(1.0)
    assert result[0] == (0 if name == "hazard" else -jnp.inf)
    assert jax.grad(active)(2.0) == (
        -0.5 if name == "log_prob" else 0.5 if name == "log_hazard" else 1.0
    )


def test_zero_time_hazard_has_zero_parameter_gradient():
    """Spec 4.5/7.2: h(0)=2*r*0=0, so d h(0)/dr=0."""
    derivative = jax.grad(lambda r: Quadratic(rate=r).hazard(0.0))(1.5)
    assert np.isfinite(derivative)
    assert derivative == 0


def test_zero_time_hazard_gradient_is_finite_in_mixed_batch():
    """Spec 7.2: d/dr [h(0)+h(1)] = d/dr [0+2*r] = 2."""
    derivative = jax.grad(
        lambda r: Quadratic(rate=r).hazard(jnp.array([0.0, 1.0])).sum()
    )(1.5)
    assert np.isfinite(derivative)
    assert derivative == 2


@pytest.mark.parametrize(
    "name",
    [
        "log_survival",
        "survival",
        "log_prob",
        "log_cdf",
        "cumulative_hazard",
        "log_hazard",
        "hazard",
    ],
)
def test_direct_method_calls_reference_time_once(name, monkeypatch):
    """Spec 4.5: generic direct methods apply the safe-time policy once."""
    calls = []
    original = Constant.reference_time

    def reference_time(self):
        calls.append(None)
        return original(self)

    monkeypatch.setattr(Constant, "reference_time", reference_time)
    getattr(Constant(rate=2), name)(1.0)
    assert len(calls) == 1


def test_inherited_overrides_dispatch_and_policy(x64_enabled):
    """Spec 4.2/4.5: density is independent of log-hazard overrides."""

    @survjax.register_family
    @dataclass(frozen=True)
    class Analytic(Constant):
        def log_hazard(self, t):
            # Deliberately different from -d log S/dt to distinguish dispatch.
            return jnp.log(2 * self.rate) + jnp.zeros_like(t)

    @survjax.register_family
    @dataclass(frozen=True)
    class Child(Analytic):
        def log_prob(self, t):
            return self.rate + jnp.zeros_like(t)

    dist = Child(rate=2)
    before = jax.tree.leaves(dist)
    np.testing.assert_allclose(
        dist.hazard([1, jnp.inf]),
        [4, 4],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    np.testing.assert_array_equal(
        dist.log_prob([-1, 0, jnp.inf]), [-np.inf, 2, -np.inf]
    )
    assert np.isnan(dist.hazard(jnp.nan))
    for name in ("log_prob", "log_hazard", "hazard"):
        assert not hasattr(getattr(Child, name)._survjax_raw, "_survjax_raw")
    assert all(a is b for a, b in zip(before, jax.tree.leaves(dist), strict=True))
    np.testing.assert_allclose(
        Analytic(rate=2).log_prob(1),
        np.log(2) - 2,
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


@pytest.mark.parametrize("name", ["log_prob", "log_hazard", "hazard"])
def test_unguarded_power_safe_reference_and_nan_isolation(name):
    """Spec 4.5/7.2: parameter-dependent references are stopped only as inputs."""
    for time in (-1.0, -jnp.inf, jnp.nan):

        def value(rate, time=time):
            return getattr(Power(rate=rate), name)(time)

        assert jax.jit(jax.grad(value))(0.5) == 0
    result = getattr(Constant(rate=[2, jnp.nan, -2]), name)(-1)
    assert result[0] == (0 if name == "hazard" else -jnp.inf)
    assert np.isnan(result[1:]).all()


@pytest.mark.parametrize("name", ["log_prob", "log_hazard", "hazard"])
def test_interior_parameter_derivatives(name, x64_enabled):
    """Spec 7.3: analytic first/second derivatives composed with jit/vmap."""

    def value(rate):
        return getattr(Constant(rate=rate), name)(0.5)

    rates = jnp.array([1.5, 2.0])
    expected_first = 1 / np.asarray(rates) - (0.5 if name == "log_prob" else 0)
    expected_second = -1 / np.asarray(rates) ** 2
    if name == "hazard":
        expected_first, expected_second = np.ones(2), np.zeros(2)
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(jax.vmap(first(value)))(rates),
            expected_first,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        np.testing.assert_allclose(
            jax.jit(jax.vmap(first(first(value))))(rates),
            expected_second,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
    check_grads(
        value,
        (jnp.asarray(1.5),),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )
