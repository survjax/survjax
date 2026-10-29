"""PH adapter warning locations and generic-tail guards (spec 3.3/7.2)."""

import inspect
import warnings
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import TAIL_RTOL

from survjax import Exponential, SurvivalDistribution, param, register_family
from survjax.distributions.wrappers import ProportionalHazards


def test_float64_warnings_deduplicate_two_user_call_sites(x64_enabled):
    """Spec 3.3: positional/keyword baseline forms warn once per user's line."""
    if x64_enabled:
        pytest.skip("float64 conversion warnings require x64 disabled")
    baseline = Exponential(rate=1)
    eta = np.array([0.0], dtype=np.float64)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("default")
        for _ in range(2):
            positional_line = inspect.currentframe().f_lineno + 1
            ProportionalHazards(baseline, log_hazard_ratio=eta)
            keyword_line = inspect.currentframe().f_lineno + 1
            ProportionalHazards(baseline=baseline, log_hazard_ratio=eta)
    assert [(w.filename, w.lineno) for w in recorded] == [
        (__file__, positional_line),
        (__file__, keyword_line),
    ]
    assert all(w.category is RuntimeWarning for w in recorded)


@register_family
@dataclass(frozen=True)
class GenericExponential(SurvivalDistribution):
    rate: jax.Array = param()

    def log_survival(self, t):
        return -self.rate * jnp.asarray(t)

    def reference_time(self):
        return 1 / self.rate


def test_generic_log_cdf_underflow_keeps_baseline_tangent(x64_enabled):
    """Spec 4.4/7.3: d log S / dr = -exp(eta)*t even without log-H recovery."""
    rate, time, eta = (1e-200, 1e-200, 300.0) if x64_enabled else (1e-30, 1e-20, 40.0)
    expected = -np.exp(eta) * time

    def fn(r):
        return ProportionalHazards(
            GenericExponential(rate=r), log_hazard_ratio=eta
        ).log_survival(time)

    assert jnp.isneginf(GenericExponential(rate=rate).log_cdf(time))
    for derivative in (jax.jacfwd(fn), jax.jacrev(fn)):
        np.testing.assert_allclose(
            jax.jit(derivative)(rate), expected, atol=0, rtol=TAIL_RTOL[x64_enabled]
        )


@register_family
@dataclass(frozen=True)
class LogCureBaseline(SurvivalDistribution):
    log_cure: jax.Array = param()

    def log_survival(self, t):
        return self.log_cure * (-jnp.expm1(-jnp.asarray(t)))

    def log_survival_at_infinity(self):
        return self.log_cure

    def reference_time(self):
        return jnp.ones_like(self.log_cure)


def test_zero_limit_linear_factor_uses_promoted_dtype(x64_enabled):
    """Spec 3.3/4.4: dS(inf)/dc = exp(eta) at log cure c=0, in float64."""
    if not x64_enabled:
        pytest.skip("mixed float64 baseline/float32 predictor requires x64")
    eta = jnp.array(90.0, dtype=jnp.float32)

    def fn(c):
        return ProportionalHazards(
            LogCureBaseline(log_cure=c), log_hazard_ratio=eta
        ).survival(jnp.inf)

    c = jnp.array(0.0, dtype=jnp.float64)
    assert fn(c).dtype == jnp.float64 and fn(c) == 1
    for derivative in (jax.jacfwd(fn), jax.jacrev(fn)):
        np.testing.assert_allclose(
            jax.jit(derivative)(c), np.exp(90.0), atol=0, rtol=TAIL_RTOL[x64_enabled]
        )
