"""Proportional-hazards composition (spec 4.4)."""

from collections.abc import Callable
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
        baseline: Registered distribution, positional or keyword; its batch shape
            broadcasts with eta, preserving nested parameter dtypes.
        log_hazard_ratio: Finite real predictor, keyword only, with any
            broadcast-compatible shape. Values are stored without checking.

    Returns:
        A frozen distribution with broadcast parameter leaves. Pointwise
        methods pair times with batch entries; inherited grids return
        (K, *batch_shape). Diagnostics prefix nested keys with ``baseline.``.

    Raises:
        TypeError: For malformed types, extra positional arguments or incompatible
            shapes. Runtime parameter violations appear in parameter_issues().
    """

    baseline: SurvivalDistribution = distribution()
    log_hazard_ratio: jax.Array = param(constraint=constraints.real)

    def reference_time(self) -> jax.Array:
        """Return baseline reference times in the promoted parameter dtype.

        Returns:
            Positive finite times of shape batch_shape for valid parameters.
            Baseline precision is retained when the predictor dtype is narrower.
        """
        reference = self.baseline.reference_time()
        return jnp.asarray(
            reference, dtype=jnp.result_type(reference, self.log_hazard_ratio)
        )

    def _scale_log_survival(
        self, log_s: jax.Array, log_lower_tail: jax.Array | None = None
    ) -> jax.Array:
        eta = self.log_hazard_ratio
        zero = log_s == 0
        recover = jnp.zeros_like(zero)
        if log_lower_tail is not None:
            recover = zero & jnp.isfinite(log_lower_tail)
        boundary = (zero | jnp.isinf(log_s)) & jnp.isfinite(eta) & ~recover
        # spec 7.2: avoid exp(eta)*0 and exp(eta)*inf, including in the hook.
        safe_log_s = jnp.where(boundary | recover, -jnp.ones_like(log_s), log_s)
        safe_eta = jnp.where(boundary, jnp.zeros_like(eta), eta)
        log_h = jnp.log(jnp.abs(safe_log_s))
        if log_lower_tail is not None:
            # spec 4.4/7.2: when H0 underflows, log H0 = log F0 to rounding precision.
            log_h = jnp.where(recover, log_lower_tail, log_h)
        log_magnitude = safe_eta + log_h
        overflow = log_magnitude >= jnp.log(jnp.finfo(log_magnitude.dtype).max)
        # spec 7.1/7.2: restore IEEE overflow without an inactive exp tangent.
        magnitude = jnp.exp(jnp.where(overflow, 0.0, log_magnitude))
        value = jnp.sign(safe_log_s) * magnitude
        overflow_value = jnp.where(safe_log_s < 0, -jnp.inf, jnp.inf)
        value = jnp.where(overflow, overflow_value, value)
        # A true zero log-limit may still have a nonzero baseline tangent.
        linear = zero & boundary & (eta < jnp.log(jnp.finfo(eta.dtype).max))
        zero_value = jnp.where(linear, log_s, 0.0) * jnp.exp(
            jnp.where(linear, eta, 0.0)
        )
        boundary_value = jnp.where(zero, zero_value, log_s)
        return jnp.where(boundary, boundary_value, value)

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
        log_s = self.baseline.log_survival(t)
        reference = jax.lax.stop_gradient(self.baseline.reference_time())
        lower_tail_time = jnp.where(log_s == 0, jnp.asarray(t), reference)
        log_lower_tail = self.baseline.log_cdf(lower_tail_time)
        return self._scale_log_survival(log_s, log_lower_tail)

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


_registered_ph_init: Callable[..., None] = ProportionalHazards.__init__


def _initialize_ph(
    self: ProportionalHazards,
    baseline: SurvivalDistribution,
    *,
    log_hazard_ratio: ArrayLike,
) -> None:
    # D-028: the maintainer permits a positional baseline only for this wrapper.
    _registered_ph_init(self, baseline=baseline, log_hazard_ratio=log_hazard_ratio)


ProportionalHazards.__init__ = _initialize_ph  # type: ignore[method-assign,assignment]
