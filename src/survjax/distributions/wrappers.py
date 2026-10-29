"""Proportional-hazards composition (spec 4.4)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax import constraints
from survjax.distributions.base import (
    SurvivalDistribution,
    distribution,
    param,
    register_family,
)


@register_family
@dataclass(frozen=True)
class ProportionalHazards(SurvivalDistribution):
    """Scale a baseline hazard by exp(eta), with eta = log_hazard_ratio.

    H(t) = exp(eta)*H0(t), S(t) = S0(t)**exp(eta), and
    log f(t) = log h0(t) + eta + log S(t). Positive eta means higher
    hazard under PH; the later AFT convention instead means longer times.
    Covariate mappings and fitting belong to the caller. Eta is dimensionless;
    time units match the baseline. For an Exponential(rate=r) baseline the
    equivalent rate is r*exp(eta); eta=log(2) doubles the hazard.

    Args:
        baseline: Registered distribution, keyword only; its batch shape
            broadcasts with eta, preserving nested parameter dtypes.
        log_hazard_ratio: Finite real predictor, keyword only, with any
            broadcast-compatible shape. Values are stored without checking.

    Returns:
        A frozen distribution with broadcast parameter leaves. Pointwise
        methods pair times with batch entries; inherited grids return
        (K, *batch_shape). Diagnostics prefix nested keys with ``baseline.``.

    Raises:
        TypeError: For malformed types, positional arguments or incompatible
            shapes. Runtime parameter violations appear in parameter_issues().
    """

    baseline: SurvivalDistribution = distribution()
    log_hazard_ratio: jax.Array = param(constraint=constraints.real)

    def reference_time(self) -> jax.Array:
        """Return baseline reference times in the predictor dtype.

        Returns:
            Positive finite times of shape batch_shape for valid parameters.
            If dtype conversion loses the baseline reference, use one time
            unit for inactive evaluations only (spec 4.2/7.2).
        """
        reference = jnp.asarray(
            self.baseline.reference_time(), dtype=self.log_hazard_ratio.dtype
        )
        usable = jnp.isfinite(reference) & (reference > 0)
        return jnp.where(usable, reference, jnp.ones_like(reference))

    def _scale_log_survival(self, log_s: jax.Array) -> jax.Array:
        eta = self.log_hazard_ratio
        boundary = ((log_s == 0) | jnp.isinf(log_s)) & jnp.isfinite(eta)
        # spec 7.2: avoid exp(eta)*0 and exp(eta)*inf, including in the hook.
        safe_log_s = jnp.where(boundary, -jnp.ones_like(log_s), log_s)
        safe_eta = jnp.where(boundary, jnp.zeros_like(eta), eta)
        log_magnitude = safe_eta + jnp.log(jnp.abs(safe_log_s))
        overflow = log_magnitude >= jnp.log(jnp.finfo(log_magnitude.dtype).max)
        # spec 7.1/7.2: restore IEEE overflow without an inactive exp tangent.
        magnitude = jnp.exp(jnp.where(overflow, 0.0, log_magnitude))
        value = jnp.sign(safe_log_s) * magnitude
        overflow_value = jnp.where(safe_log_s < 0, -jnp.inf, jnp.inf)
        value = jnp.where(overflow, overflow_value, value)
        return jnp.where(boundary, log_s, value)

    def log_survival_at_infinity(self) -> jax.Array:
        """Return exp(eta)*log S0(+inf), retaining improper-family limits.

        Returns:
            Log survival limits of shape batch_shape in the promoted dtype.
            Proper baselines retain -inf for every finite predictor.
        """
        return self._scale_log_survival(self.baseline.log_survival_at_infinity())

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return exp(eta)*log S0(t) under the base time-value policy.

        Args:
            t: Real times broadcast against batch_shape.
        Returns:
            Log survival with the broadcast shape; zero at t=0, and the
            scaled baseline infinity limit at +inf. NaN times give NaN.
        """
        return self._scale_log_survival(self.baseline.log_survival(t))

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return log h0(t) + eta, including baseline hazard limits.

        Args:
            t: Real times broadcast against batch_shape.
        Returns:
            Log hazard with the broadcast shape, -inf before zero and NaN
            at NaN times. Baseline overrides determine zero/infinity limits.
        """
        return self.baseline.log_hazard(t) + self.log_hazard_ratio

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return log h0(t) + eta + exp(eta)*log S0(t).

        Args:
            t: Real times broadcast against batch_shape.
        Returns:
            Log density with the broadcast shape; -inf before zero and at
            +inf, NaN at NaN times, and the baseline-scaled limit at zero.
        """
        return self.log_hazard(t) + self.log_survival(t)
