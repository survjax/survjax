"""Exponential family in natural rate parameters."""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax import constraints
from survjax._dtypes import normalize_parameters
from survjax._rmst import _rmst_cutoff
from survjax.distributions._survival_methods import log_cdf_from_log_cumulative_hazard
from survjax.distributions.base import SurvivalDistribution, param, register_family


@register_family
@dataclass(frozen=True)
class Exponential(SurvivalDistribution):
    """Exponential distribution: a constant hazard.

    The hazard is the same at every time, h(t) = rate, so::

        H(t) = rate * t,    S(t) = exp(-rate * t),    f(t) = rate * exp(-rate * t).

    The mean time to event is 1 / rate and the median log(2) / rate. This is
    R's ``pexp(t, rate, lower.tail = FALSE)``. Methods are evaluated in
    closed form and accept any time: negative times count as before the time
    origin, and zero and +inf give the exact limits.

    Args:
        rate: Hazard per unit time, positive and finite, keyword only. Its
            array shape is the ``batch_shape``; each entry is one
            distribution. Values are stored as given and not checked.

    Raises:
        TypeError: For positional arguments or nonreal ``rate``. Methods also
            raise it for nonreal times or times that do not broadcast against
            ``batch_shape``.

    Converting from other parameterisations: a log rate ``a`` gives
    ``rate=exp(a)``, a mean ``m`` gives ``rate=1/m`` and a log mean gives
    ``rate=exp(-log_mean)``; ``from_log`` takes the log rate directly. R's
    ``survreg(..., dist = "exponential")`` intercept is the log mean.

    >>> import survjax
    >>> progression = survjax.Exponential(rate=1 / 20)  # mean 20 months
    >>> round(float(progression.survival(12.0)), 4)  # exp(-0.6)
    0.5488
    """

    rate: jax.Array = param(constraint=constraints.positive)

    def _ph_equivalent(self, eta: jax.Array, invalid: jax.Array) -> "Exponential":
        # spec 4.4/7.2: mask original issues before constructing the closure.
        rate = jnp.where(invalid, 1.0, self.rate)
        eta = jnp.where(invalid, 0.0, eta)
        log_rate = jnp.log(rate) + eta
        overflow = log_rate >= jnp.log(jnp.finfo(log_rate.dtype).max)
        rate = jnp.exp(jnp.where(overflow, 0.0, log_rate))
        return Exponential(rate=jnp.where(overflow, jnp.inf, rate))

    def rmst(self, tau: ArrayLike) -> jax.Array:
        """Return restricted mean event time, ``-expm1(-rate*tau)/rate``.

        Args:
            tau: Cutoffs, broadcast against ``batch_shape``. For a time-first
                cutoff grid, add one trailing singleton per batch axis.

        Returns:
            RMST with the broadcast shape. Negative, NaN or infinite cutoffs
            and entries with parameter issues give NaN. At zero the value and
            parameter derivatives are zero and the cutoff derivative is one.
            JAX supports derivatives of all orders on valid smooth inputs.

        Raises:
            TypeError: For nonreal cutoffs or incompatible batch shapes.
        """
        cutoff = _rmst_cutoff(tau, self.batch_shape)
        invalid = (
            ~jnp.isfinite(cutoff)
            | (cutoff < 0)
            | self.parameter_issues().invalid_entries
        )
        # spec 8.7/7.2: mask both operands before arithmetic, including shared
        # parameters beside invalid cutoffs. Zero is active to retain d/dtau=1.
        safe_cutoff = jnp.where(invalid, jnp.zeros_like(cutoff), cutoff)
        safe_rate = jnp.where(invalid, jnp.ones_like(self.rate), self.rate)
        value = -jnp.expm1(-safe_rate * safe_cutoff) / safe_rate
        return jnp.where(invalid, jnp.nan, value)

    @classmethod
    def from_log(cls, *, log_rate: ArrayLike) -> Self:
        """Build an exponential from its log rate, ``rate = exp(log_rate)``.

        Useful when the rate comes from an unconstrained optimiser or a
        neural network output. The result is an ordinary ``Exponential``
        storing the rate itself, with the same values and gradients as
        ``Exponential(rate=jnp.exp(log_rate))``. For a log mean, pass
        ``log_rate=-log_mean``.

        Args:
            log_rate: Log of the rate, any real value, keyword only. Its array
                shape becomes the ``batch_shape``.

        Returns:
            An ``Exponential`` with ``rate = exp(log_rate)``. If the
            exponential overflows (above about 88 in float32), the rate is
            infinite and results follow the formulas for an invalid rate.

        Raises:
            TypeError: For nonreal ``log_rate``.
        """
        values = normalize_parameters({"log_rate": log_rate}, warning_stacklevel=3)
        return cls(rate=jnp.exp(values["log_rate"]))

    def reference_time(self) -> jax.Array:
        """Return a typical time for each distribution: the mean, 1 / rate.

        Used internally as a safe evaluation time (see
        ``SurvivalDistribution.reference_time``). When 1 / rate cannot be
        represented, one time unit is used instead; this never changes a
        returned value.

        Returns:
            Positive finite times of shape ``batch_shape`` in the rate's
            dtype, for valid rates.
        """
        # spec 4.2: unlike a stored time scale, 1/rate may not be representable.
        # This guard changes only the inactive reference, not model parameters.
        limits = jnp.finfo(self.rate.dtype)
        usable = (self.rate >= limits.tiny) & (self.rate <= 1 / limits.tiny)
        safe_rate = jnp.where(usable, self.rate, jnp.ones_like(self.rate))
        return jnp.reciprocal(safe_rate)

    def _cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        infinite = jnp.isposinf(time)

        # spec 7.2: don't multiply an inactive infinity by a live parameter.
        safe_time = jnp.where(infinite, jnp.zeros_like(time), time)

        # spec 4.1: interior H(t)=rate*t, so log S(t)=-H(t).
        value = self.rate * safe_time

        # spec 4.5: restore the infinity limit separately from the formula.
        limit = jnp.where(self.rate > 0, jnp.inf, jnp.nan)
        limit = jnp.where(self.rate < 0, -jnp.inf, limit)
        return jnp.where(infinite, limit, value)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return the log survival probability, ``log S(t) = -rate * t``.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Log survival; 0 for t <= 0 and -inf at t = +inf. NaN times or
            rates give NaN.
        """
        return -self._cumulative_hazard(t)

    def cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        """Return the cumulative hazard, ``H(t) = rate * t``.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Cumulative hazard; 0 for t <= 0 and +inf at t = +inf. NaN times
            or rates give NaN.
        """
        return self._cumulative_hazard(t)

    def _log_cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        zero = time == 0
        infinite = jnp.isposinf(time)
        boundary = zero | infinite

        # spec 7.2: substitute before log(t), including at both boundaries.
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)

        # spec 4.1: log H = log(rate) + log(t), without forming rate*t.
        value = jnp.log(self.rate) + jnp.log(safe_time)
        zero_limit = jnp.where(jnp.isfinite(self.rate), -jnp.inf, jnp.nan)
        infinity_limit = jnp.where(self.rate > 0, jnp.inf, jnp.nan)
        value = jnp.where(zero, zero_limit, jnp.where(infinite, infinity_limit, value))
        return value

    def log_cdf(self, t: ArrayLike) -> jax.Array:
        """Return the log probability of an event by time t, log F(t).

        ``F(t) = 1 - exp(-rate * t)``. It is computed from
        ``log H = log(rate) + log(t)``, so it keeps relative precision both for
        very early times, where F is tiny (log F is close to log H), as long as
        log H is representable, and for very late times, where log F is close
        to -S(t), as long as S(t) is a normal floating-point number. Beyond
        that S underflows and log F rounds to 0.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Log CDF; -inf for t <= 0 and 0 at t = +inf. NaN times or rates
            give NaN.
        """
        return log_cdf_from_log_cumulative_hazard(self._log_cumulative_hazard(t))

    def log_interval_prob(self, lower: ArrayLike, upper: ArrayLike) -> jax.Array:
        """Return log P(lower < T <= upper) from the interval width.

        For endpoints a < b, this is -rate*a + log(1-exp(-rate*(b-a))).
        The width term is evaluated through the stable log CDF, avoiding
        subtraction of nearly equal survival probabilities.

        Args:
            lower: Nonnegative lower endpoints.
            upper: Upper endpoints, possibly +inf. Both endpoints broadcast
                against batch_shape; add trailing singleton axes for grids.
        Returns:
            Broadcast log probabilities. Equal endpoints give -inf; invalid
            endpoints or rates give NaN. Positive mass is retained in log space
            even when rate times the interval width underflows.
        Raises:
            ValueError: If endpoint shapes cannot broadcast.
        """
        a, b = jnp.broadcast_arrays(jnp.asarray(lower), jnp.asarray(upper))
        valid = (a >= 0) & (b >= a) & ~self.parameter_issues().invalid_entries
        empty = a == b
        active = valid & ~empty
        # spec 7.2: in particular, never subtract inf-inf for an empty interval.
        a = jnp.where(active, a, 0.0)
        b = jnp.where(active, b, 1.0)
        value = -self.rate * a + self.log_cdf(b - a)
        return jnp.where(valid, jnp.where(empty, -jnp.inf, value), jnp.nan)

    def log_conditional_interval_prob(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        left_trunc: ArrayLike,
        right_trunc: ArrayLike,
    ) -> jax.Array:
        """Return the interval's log probability conditional on (L,U].

        Intersect (lower,upper] with the truncation window to get (a,b].
        Cancel the common survival factor S(L) before evaluating the ratio:
        -rate*(a-L) + log F(b-a) - log F(U-L), where F is this family's CDF.

        Args:
            lower: Nonnegative requested lower endpoints.
            upper: Requested upper endpoints, possibly +inf.
            left_trunc: Finite nonnegative window lower endpoints L.
            right_trunc: Window upper endpoints U > L, possibly +inf.
        Returns:
            Log probabilities with trailing-axis batch broadcasting. Full
            coverage gives exactly zero; empty intersections give -inf.
            Invalid endpoints or rates give NaN.
        Raises:
            ValueError: If input shapes cannot broadcast.
        """
        a, b, window_lower, window_upper = jnp.broadcast_arrays(
            *map(jnp.asarray, (lower, upper, left_trunc, right_trunc))
        )
        valid = (
            (a >= 0)
            & (b >= a)
            & jnp.isfinite(window_lower)
            & (window_lower >= 0)
            & (window_upper > window_lower)
            & ~self.parameter_issues().invalid_entries
        )
        window_lower = jnp.where(valid, window_lower, 0.0)
        window_upper = jnp.where(valid, window_upper, 1.0)
        lo, hi = jnp.maximum(a, window_lower), jnp.minimum(b, window_upper)
        empty = hi <= lo
        full = (a <= window_lower) & (b >= window_upper)
        active = valid & ~empty & ~full
        # spec 7.2: inactive numerator uses the window, without re-intersection.
        lo = jnp.where(active, lo, window_lower)
        hi = jnp.where(active, hi, window_upper)
        # spec 4.2: cancel the common cumulative hazard before multiplication.
        value = (
            -self.rate * (lo - window_lower)
            + self.log_cdf(hi - lo)
            - self.log_cdf(window_upper - window_lower)
        )
        value = jnp.where(full, 0.0, jnp.where(empty, -jnp.inf, value))
        return jnp.where(valid, value, jnp.nan)

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return the log density, ``log f(t) = log(rate) - rate * t``.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Log density; log(rate) at t = 0, and -inf for t < 0 and at
            t = +inf. NaN times or rates give NaN.
        """
        # spec 4.1: log f(t)=log(rate)-H(t), since h(t)=rate.
        return jnp.log(self.rate) - self._cumulative_hazard(t)

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return the log hazard, log h(t) = log(rate), at every t >= 0.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            log(rate) for t >= 0, including t = +inf, where survival is zero
            but the constant hazard is still defined; -inf for t < 0. NaN
            times or rates give NaN.
        """
        # spec 4.1/5: log h(t)=log(rate) is constant in time but must broadcast.
        output_shape = jnp.broadcast_shapes(jnp.shape(t), self.batch_shape)
        return jnp.broadcast_to(jnp.log(self.rate), output_shape)

    def hazard(self, t: ArrayLike) -> jax.Array:
        """Return the hazard, h(t) = rate, at every t >= 0.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            The rate for t >= 0, including t = +inf (with derivative 1 in the
            rate there too); 0 for t < 0. NaN times or rates give NaN.
        """
        # spec 4.1/5: h(t)=rate is constant in time but must broadcast.
        output_shape = jnp.broadcast_shapes(jnp.shape(t), self.batch_shape)
        return jnp.broadcast_to(self.rate, output_shape)
