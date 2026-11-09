"""Proportional-hazards and accelerated-failure-time composition."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax import constraints
from survjax._rmst import _rmst_cutoff
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
        ``(K, *batch_shape)``. Diagnostics prefix nested keys with ``baseline.``.

    Raises:
        TypeError: For malformed types, extra positional arguments or incompatible
            shapes. Runtime parameter violations appear in parameter_issues().
    """

    baseline: SurvivalDistribution = distribution()
    log_hazard_ratio: jax.Array = param(constraint=constraints.real)

    def rmst(self, tau: ArrayLike) -> jax.Array:
        """Return restricted mean time through the baseline's private PH closure.

        Args:
            tau: Finite nonnegative cutoffs broadcast against batch_shape.
                Use tau_grid[:, None, None] for a batch of shape (S, 2).
        Returns:
            Broadcast RMST in time units; NaN for invalid/failed entries.
            At zero, value and parameter derivatives are 0, d/dtau is 1.
            For Weibull, only first derivatives are guaranteed at zero;
            higher interior derivatives involving shape are unsupported.
        Raises:
            TypeError: Nonreal cutoffs or incompatible shapes.
            NotImplementedError: Baseline has no staged PH closure, or an
                unsupported higher Weibull shape derivative is requested.
        """
        if type(self.baseline)._ph_equivalent is SurvivalDistribution._ph_equivalent:
            raise NotImplementedError("rmst requires a supported baseline PH closure")
        cutoff = _rmst_cutoff(tau, self.batch_shape)
        invalid = (
            ~jnp.isfinite(cutoff)
            | (cutoff < 0)
            | self.parameter_issues().invalid_entries
        )
        dtype = jnp.result_type(cutoff, *jax.tree.leaves(self))
        baseline = jax.tree.map(lambda leaf: leaf.astype(dtype), self.baseline)
        eta = self.log_hazard_ratio.astype(dtype)
        equivalent = baseline._ph_equivalent(eta, invalid)
        if equivalent is None:
            raise NotImplementedError("rmst requires a supported baseline PH closure")
        # spec 8.7: zero is exact even if the natural closure is unrepresentable.
        failed_zero = (cutoff == 0) & equivalent.parameter_issues().invalid_entries
        equivalent = jax.tree.map(
            lambda leaf: jnp.where(failed_zero, 1.0, leaf), equivalent
        )
        value = equivalent.rmst(jnp.where(invalid, 0.0, cutoff))
        return jnp.where(invalid, jnp.nan, value)

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
        self, log_s: jax.Array, baseline_log_cdf: jax.Array | None = None
    ) -> jax.Array:
        """Apply the PH multiplier to baseline log survival, guarding tails.

        Args:
            log_s: Baseline log survival with evaluated shape E, containing no
                component axes. Pointwise callers use the broadcast shape of
                time and ``self.batch_shape``. Scalar time and the infinity
                hook give E = batch_shape; a shared grid gives
                E = (K, *batch_shape), including all leading time axes.
            baseline_log_cdf: Optional baseline log CDF, with the same evaluated
                shape E as log_s (not merely batch_shape on a grid). Finite
                entries recover a tiny cumulative hazard where log_s rounded
                to zero. Only log_s == 0 entries are read; others may contain
                reference-time values. The infinity hook passes None.

        Returns:
            Scaled log survival with shape E for these callers, using JAX
            floating promotion. The predictor has shape self.batch_shape and
            broadcasts across any leading evaluated-time axes.

        Mathematical appendix: ``appendices/ph-log-survival.md``.
        """
        eta = jnp.asarray(
            self.log_hazard_ratio, dtype=jnp.result_type(log_s, self.log_hazard_ratio)
        )
        # Finite nonzero H0 uses log scaling; an accurate log F0 recovers tiny H0.
        # Without that tail, true/rounded zero uses the safe linear tangent.
        # Infinite limits and overflow are restored after substituting operands.
        zero = log_s == 0
        recover = jnp.zeros_like(zero)
        if baseline_log_cdf is not None:
            recover = zero & jnp.isfinite(baseline_log_cdf)
        boundary = (zero | jnp.isinf(log_s)) & jnp.isfinite(eta) & ~recover
        # spec 7.2: avoid exp(eta)*0 and exp(eta)*inf, including in the hook.
        safe_log_s = jnp.where(boundary | recover, -jnp.ones_like(log_s), log_s)
        safe_eta = jnp.where(boundary, jnp.zeros_like(eta), eta)
        log_baseline_cumulative_hazard = jnp.log(jnp.abs(safe_log_s))
        if baseline_log_cdf is not None:
            # Math: appendices/ph-log-survival.md#lower-tail-recovery (spec 4.4/7.2).
            log_baseline_cumulative_hazard = jnp.where(
                recover, baseline_log_cdf, log_baseline_cumulative_hazard
            )
        log_scaled_cumulative_hazard = safe_eta + log_baseline_cumulative_hazard
        overflow = log_scaled_cumulative_hazard >= jnp.log(
            jnp.finfo(log_scaled_cumulative_hazard.dtype).max
        )
        # Math: appendices/ph-log-survival.md#safe-overflow (spec 7.1/7.2).
        cumulative_hazard_magnitude = jnp.exp(
            jnp.where(overflow, 0.0, log_scaled_cumulative_hazard)
        )
        value = jnp.sign(safe_log_s) * cumulative_hazard_magnitude
        overflow_value = jnp.where(safe_log_s < 0, -jnp.inf, jnp.inf)
        value = jnp.where(overflow, overflow_value, value)
        # Math: appendices/ph-log-survival.md#zero-limit-tangents.
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
        baseline_log_cdf = self.baseline.log_cdf(lower_tail_time)
        return self._scale_log_survival(log_s, baseline_log_cdf)

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


@register_family
@dataclass(frozen=True)
class AcceleratedFailureTime(SurvivalDistribution):
    """Rescale baseline event times by exp(eta), with eta = log_time_ratio.

    Positive eta means longer times under AFT; under PH it means higher
    hazard. Covariate mappings and fitting belong to the caller. Time units
    match the baseline, and eta is dimensionless.

    Args:
        baseline: Registered distribution, keyword only. Nested parameter
            leaves broadcast with eta and retain their dtypes.
        log_time_ratio: Real predictor, keyword only, broadcast against the
            baseline batch shape. Values are stored without domain checks.

    Returns:
        Frozen distribution with broadcast leaves and nested diagnostics
        prefixed by ``baseline.``. Pointwise methods broadcast times against
        batch_shape; inherited grids have shape ``(K, *batch_shape)``.

    Raises:
        TypeError: For malformed types, positional arguments or incompatible
            shapes. Parameter violations appear in parameter_issues().
    """

    baseline: SurvivalDistribution = distribution()
    log_time_ratio: jax.Array = param(constraint=constraints.real)

    def reference_time(self) -> jax.Array:
        """Return positive finite baseline reference times, shape batch_shape.

        Returns:
            Baseline reference times in the promoted parameter dtype. Keeping
            this scale avoids overflowing exp(eta) for finite predictors.
        """
        reference = self.baseline.reference_time()
        return jnp.asarray(
            reference, dtype=jnp.result_type(reference, self.log_time_ratio)
        )

    def log_survival_at_infinity(self) -> jax.Array:
        """Return the baseline log survival limit, shape batch_shape.

        Returns:
            Baseline proper or improper limit in the promoted parameter dtype;
            a NaN predictor propagates as it does in the direct formulas.
        """
        limit = self.baseline.log_survival_at_infinity()
        # spec 4.5: invalid predictors must retain arithmetic NaN precedence.
        return limit + jnp.zeros_like(self.log_time_ratio) * self.log_time_ratio

    def _rescale_time(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        boundary = (time == 0) | jnp.isposinf(time)
        # spec 4.4/7.2: log-space scaling avoids exp(-eta) overflow and 0*inf.
        log_time = jnp.log(time) - self.log_time_ratio
        overflow = log_time >= jnp.log(jnp.finfo(log_time.dtype).max)
        scaled = jnp.exp(jnp.where(overflow, 0.0, log_time))
        # spec 7.1: a positive interior time lost during rescaling is a
        # numerical failure, not a true zero/infinity boundary of the model.
        failed = ~boundary & ((scaled == 0) | overflow)
        scaled = jnp.where(failed, jnp.nan, scaled)
        return jnp.where(boundary, time, scaled)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return log S0(t*exp(-eta)) under the direct-method value policy.

        Args:
            t: Real times broadcast against batch_shape.
        Returns:
            Broadcast log survival; zero at zero and the baseline limit at
            infinity. Negative times give zero (spec 4.5); NaN times and
            unrepresentable interior rescaled times give NaN.
        """
        return self.baseline.log_survival(self._rescale_time(t))

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return log f0(t*exp(-eta))-eta, retaining baseline zero limits.

        Args:
            t: Real times broadcast against batch_shape.
        Returns:
            Broadcast log density, -inf before zero and at infinity, NaN at
            NaN times or unrepresentable interior rescaled times. The baseline
            supplies the density limit at zero.
        """
        return self.baseline.log_prob(self._rescale_time(t)) - self.log_time_ratio

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return log h0(t*exp(-eta))-eta, retaining baseline boundary limits.

        Args:
            t: Real times broadcast against batch_shape.
        Returns:
            Broadcast log hazard; -inf before zero and NaN at NaN times.
            Baseline overrides supply the zero and infinity limits; an
            unrepresentable interior rescaled time gives NaN.
        """
        return self.baseline.log_hazard(self._rescale_time(t)) - self.log_time_ratio
