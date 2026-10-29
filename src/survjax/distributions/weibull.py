"""Weibull family in natural shape and scale parameters (spec 4.1)."""

from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax import constraints
from survjax._dtypes import normalize_parameters
from survjax._numerics import _rmst_cutoff, _weibull_rmst
from survjax.distributions._survival_methods import log_cdf_from_log_cumulative_hazard
from survjax.distributions.base import SurvivalDistribution, param, register_family


@register_family
@dataclass(frozen=True)
class Weibull(SurvivalDistribution):
    """Weibull distribution: a hazard that rises, stays flat or falls.

    With shape k and scale b::

        H(t) = (t / b)**k,    S(t) = exp(-(t / b)**k),
        h(t) = (k / b) * (t / b)**(k - 1).

    The hazard falls over time for k < 1, is constant (the exponential with
    rate 1 / b) for k = 1, and rises for k > 1. At t = b, S(b) = exp(-1)
    for every shape. Methods are evaluated in closed form and accept any
    time: negative times count as before the time origin, and zero and +inf
    give the exact limits (for example the hazard at zero is +inf, 1 / b or
    0 for k < 1, k = 1 or k > 1).

    Args:
        shape: Shape k, dimensionless, positive and finite, keyword only.
        scale: Scale b, in time units, positive and finite, keyword only.
            The two broadcast to a common ``batch_shape``; each entry is one
            distribution. Values are stored as given and not checked.

    Raises:
        TypeError: For positional arguments, nonreal parameters or batch
            shapes that do not broadcast. Methods also raise it for nonreal
            times or times that do not broadcast against ``batch_shape``.

    Other conventions, and how to convert them:

    - base R ``dweibull``/``pweibull`` and flexsurv ``"weibull"``:
      ``shape=k``, ``scale=b``, as here. flexsurv's natural estimates
      (``fit$res``) are k and b, but ``coef(fit)`` reports log k and log b.
    - NumPyro ``Weibull``: ``concentration=k``, ``scale=b``.
    - ``survival::survreg``: log-time location ``mu = log(b)`` (the
      intercept) and reported ``Scale`` ``sigma = 1/k``. For example
      ``mu = log(10)``, ``sigma = 0.5`` is ``Weibull(shape=2, scale=10)``.
    - flexsurv ``"weibullPH"``: ``shape=k``, ``scale=m`` with
      ``S(t) = exp(-m * t**k)``, so ``m = b**(-k)`` and ``b = m**(-1/k)``.
    - inverse time scale r, ``S(t) = exp(-(r * t)**k)``, gives ``b = 1/r``.
    - log parameters: ``k = exp(log_shape)``, ``b = exp(log_scale)``; see
      ``from_log``.

    With a common shape, an accelerated failure time model log(b) = X @ beta
    is also a proportional hazards model, with log hazard ratios
    ``gamma = -k * beta``. If the shape varies, there is no such conversion.

    At k = 1 the log density at t = 0 jumps from +inf to -inf as k crosses 1,
    so its derivative in k there is not defined; its derivative in b is.

    >>> import survjax
    >>> progression = survjax.Weibull(shape=2.0, scale=10.0)
    >>> round(float(progression.survival(10.0)), 4)  # exp(-1)
    0.3679
    """

    shape: jax.Array = param(constraint=constraints.positive)
    scale: jax.Array = param(constraint=constraints.positive)

    def rmst(self, tau: ArrayLike) -> jax.Array:
        """Return the restricted mean event time, integral of S(t) to tau.

        Args:
            tau: Broadcast cutoffs; use trailing ones for a time-first grid.
        Returns:
            Broadcast RMST; NaN for invalid cutoffs or parameter issues.
            At zero, value/parameter derivatives are 0 and d/dtau is 1.
            First derivatives use JAX; higher shape derivatives are unsupported.
        Raises:
            TypeError: Nonreal cutoffs or incompatible shapes.
            NotImplementedError: Higher derivatives involving shape.
        """
        cutoff = _rmst_cutoff(tau, self.batch_shape)
        invalid = (
            ~jnp.isfinite(cutoff)
            | (cutoff < 0)
            | self.parameter_issues().invalid_entries
        )
        shape = jnp.where(invalid, 1.0, self.shape)
        scale = jnp.where(invalid, 1.0, self.scale)
        cutoff = jnp.where(invalid, jnp.zeros_like(cutoff), cutoff)
        return jnp.where(invalid, jnp.nan, _weibull_rmst(shape, scale, cutoff))

    @classmethod
    def from_log(cls, *, log_shape: ArrayLike, log_scale: ArrayLike) -> Self:
        """Build a Weibull from log parameters, k = exp(log_shape), b = exp(log_scale).

        Useful when parameters come from an unconstrained optimiser, a
        neural network output or flexsurv's ``coef(fit)``. The result is an
        ordinary ``Weibull`` storing k and b themselves, with the same values
        and gradients as ``Weibull(shape=jnp.exp(log_shape),
        scale=jnp.exp(log_scale))``. From ``survreg``, pass
        ``log_shape=-log(sigma)`` and ``log_scale=mu``.

        Args:
            log_shape: Log of the shape, any real value, keyword only.
            log_scale: Log of the scale, any real value, keyword only. The
                two broadcast to a common ``batch_shape``.

        Returns:
            A ``Weibull`` with ``shape = exp(log_shape)`` and
            ``scale = exp(log_scale)``. If an exponential overflows (above
            about 88 in float32), that parameter is infinite and results
            follow the formulas for an invalid parameter.

        Raises:
            TypeError: For nonreal inputs or batch shapes that do not
                broadcast.
        """
        values = normalize_parameters(
            {"log_shape": log_shape, "log_scale": log_scale}, warning_stacklevel=3
        )
        return cls(
            shape=jnp.exp(values["log_shape"]), scale=jnp.exp(values["log_scale"])
        )

    def reference_time(self) -> jax.Array:
        """Return a typical time for each distribution: the scale b.

        Used internally as a safe evaluation time (see
        ``SurvivalDistribution.reference_time``); it never changes a returned
        value.

        Returns:
            The scale, of shape ``batch_shape`` in the parameter dtype.
        """
        # spec 4.2: valid scale is already a positive finite time; Exponential's
        # 1/rate needs a reciprocal representability guard.
        return self.scale

    def _cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        zero = time == 0
        infinite = jnp.isposinf(time)
        boundary = zero | infinite

        # spec 7.2: substitute before logs/products, including the scale operand.
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)
        safe_scale = jnp.where(boundary, jnp.ones_like(self.scale), self.scale)

        # spec 4.1: interior H(t)=(t/scale)**shape. Logs avoid forming a ratio
        # that could underflow/overflow before the power becomes representable.
        log_ratio = jnp.log(safe_time) - jnp.log(safe_scale)
        value = jnp.exp(self.shape * log_ratio)

        # spec 4.5: restore boundary limits without hiding undefined parameters.
        zero_limit = jnp.where(self.shape > 0, 0.0, jnp.inf)
        infinity_limit = jnp.where(self.shape > 0, jnp.inf, 0.0)
        value = jnp.where(zero, zero_limit, jnp.where(infinite, infinity_limit, value))
        value = jnp.where(boundary & (self.shape == 0), 1.0, value)
        undefined = jnp.isnan(self.shape) | jnp.isnan(jnp.log(self.scale))
        return jnp.where(undefined, jnp.nan, value)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return the log survival probability, ``log S(t) = -(t / b)**k``.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Log survival; 0 for t <= 0 and -inf at t = +inf. NaN times or
            parameters give NaN.
        """
        return -self._cumulative_hazard(t)

    def cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        """Return the cumulative hazard, ``H(t) = (t / b)**k``.

        Computed as ``exp(k * (log t - log b))``, so t / b is never formed and
        cannot underflow or overflow before the power is taken.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Cumulative hazard; 0 for t <= 0 and +inf at t = +inf. NaN times
            or parameters give NaN.
        """
        return self._cumulative_hazard(t)

    def _log_cumulative_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        zero = time == 0
        infinite = jnp.isposinf(time)
        boundary = zero | infinite

        # spec 7.2: substitute before logs, including the scale operand.
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)
        safe_scale = jnp.where(boundary, jnp.ones_like(self.scale), self.scale)

        # spec 4.1: log H = k * log(t / b), without forming H in the lower tail.
        log_ratio = jnp.log(safe_time) - jnp.log(safe_scale)
        value = self.shape * log_ratio

        # spec 4.5: restore the limits implied by H=(t/b)**k.
        zero_limit = jnp.where(
            self.shape > 0,
            -jnp.inf,
            jnp.where(self.shape == 0, 0.0, jnp.inf),
        )
        infinity_limit = jnp.where(
            self.shape > 0,
            jnp.inf,
            jnp.where(self.shape == 0, 0.0, -jnp.inf),
        )
        value = jnp.where(zero, zero_limit, jnp.where(infinite, infinity_limit, value))
        undefined = jnp.isnan(self.shape) | jnp.isnan(jnp.log(self.scale))
        return jnp.where(undefined, jnp.nan, value)

    def log_cdf(self, t: ArrayLike) -> jax.Array:
        """Return the log probability of an event by time t, log F(t).

        ``F(t) = 1 - exp(-(t / b)**k)``. It is computed from
        ``log H = k * (log t - log b)``, so it keeps relative precision both
        for very early times, where F is tiny (log F is close to log H, even
        when H itself underflows), as long as log H is representable, and for
        very late times, where log F is close to -S(t), as long as S(t) is a
        normal floating-point number. Beyond that S underflows and log F
        rounds to 0.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Log CDF; -inf for t <= 0 and 0 at t = +inf. NaN times or
            parameters give NaN.
        """
        return log_cdf_from_log_cumulative_hazard(self._log_cumulative_hazard(t))

    def log_hazard(self, t: ArrayLike) -> jax.Array:
        """Return the log hazard, ``log h(t) = log k - log b + (k - 1) * log(t / b)``.

        Computed directly rather than from the density and survival, so it
        stays accurate far in the tail where survival underflows.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Log hazard, with the exact limits at the boundaries. At t = 0 it
            is +inf, -log b or -inf for k < 1, k = 1 or k > 1, and at t = +inf
            -inf, -log b or +inf. Negative times give -inf; NaN times or
            parameters give NaN.
        """
        return self._log_hazard(t)

    def _log_hazard(self, t: ArrayLike) -> jax.Array:
        time = jnp.asarray(t)
        zero = time == 0
        infinite = jnp.isposinf(time)
        boundary = zero | infinite

        # spec 4.1/7.2: xlogy-style zero coefficients need safe AD operands too.
        safe_time = jnp.where(boundary, jnp.ones_like(time), time)
        safe_scale = jnp.where(boundary, jnp.ones_like(self.scale), self.scale)
        safe_shape = jnp.where(boundary, jnp.ones_like(self.shape), self.shape)

        # spec 4.1: interior h(t)=(shape/scale)*(t/scale)**(shape-1), in log
        # space; computing it directly avoids cancellation when H is large.
        log_ratio = jnp.log(safe_time) - jnp.log(safe_scale)
        value = jnp.log(safe_shape) - jnp.log(safe_scale) + (safe_shape - 1) * log_ratio

        # spec 4.5: shape=1 retains its finite -log(scale) boundary value.
        log_scale = jnp.log(self.scale)
        power_limit = jnp.where(self.shape < 1, jnp.inf, -jnp.inf)
        zero_limit = jnp.where(self.shape == 1, -log_scale, power_limit)
        infinity_limit = jnp.where(self.shape == 1, -log_scale, -power_limit)
        value = jnp.where(zero, zero_limit, jnp.where(infinite, infinity_limit, value))
        undefined = jnp.isnan(jnp.log(self.shape)) | jnp.isnan(log_scale)
        return jnp.where(undefined, jnp.nan, value)

    def log_prob(self, t: ArrayLike) -> jax.Array:
        """Return the log density, ``log f(t) = log h(t) - (t / b)**k``.

        Args:
            t: Times, broadcast against ``batch_shape``.

        Returns:
            Log density. At t = 0 it equals the log hazard there, so +inf,
            -log b or -inf for k < 1, k = 1 or k > 1. Negative times and
            t = +inf give -inf; NaN times or parameters give NaN.
        """
        # spec 4.1: log f(t)=log h(t)-H(t); the base guards density at infinity.
        return self._log_hazard(t) - self._cumulative_hazard(t)
