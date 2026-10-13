"""Pointwise survival methods: generic defaults (4.2) and value policy (4.5)."""

from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

if TYPE_CHECKING:
    from survjax.distributions.base import SurvivalDistribution


def _raw_log_survival(
    self: "SurvivalDistribution",
) -> Callable[["SurvivalDistribution", ArrayLike], jax.Array]:
    # spec 4.5: defaults call the raw primitive on already-substituted safe times.
    return getattr(type(self).log_survival, "_survjax_raw", type(self).log_survival)


def survival(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return P(T > t), with the broadcast shape of t and batch_shape.

    The default is exp(log_survival(t)). Equal-length vectors pair entries;
    use singleton axes for all pairs. Shared time-first grids are separate.
    Negative times return one; NaN times return NaN.
    At +inf, a raw log-survival limit of -inf gives zero with zero parameter
    gradient. Finite, parameter-dependent limits remain active; the primitive
    author must make their operands safe for differentiation.
    """
    raw = _raw_log_survival(self)
    return jnp.exp(raw(self, t))


def _log_survival_and_derivative(
    self: "SurvivalDistribution", t: ArrayLike
) -> tuple[jax.Array, jax.Array]:
    time = jnp.asarray(t)
    raw = _raw_log_survival(self)
    # spec 4.2/5: pointwise times need only a directional derivative, not a Jacobian.
    return jax.jvp(lambda time: raw(self, time), (time,), (jnp.ones_like(time),))


def _safe_log(x: jax.Array) -> jax.Array:
    """Log a nonnegative value; zero gives -inf without a NaN gradient."""
    zero = x == 0
    # spec 7.2: inactive log(0) must not contaminate active parameter derivatives.
    safe_x = jnp.where(zero, jax.lax.stop_gradient(jnp.ones_like(x)), x)
    return jnp.where(zero, -jnp.inf, jnp.log(safe_x))


def log_prob(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return log density, broadcasting t against batch_shape (not a grid).

    Forward-mode autodiff gives log(-d log S/dt) + log S independently of
    log_hazard overrides. The hazard must be representable in linear space.
    Negative times return -inf; +inf returns -inf unless the primitive's
    log survival there is NaN. Zero uses the primitive's boundary derivative,
    and NaN times return NaN.
    Infinity inputs are substituted before density evaluation; the original
    primitive limit determines their policy value. This isolates finite-entry
    parameter derivatives in mixed batches, including density overrides.
    """
    log_s, derivative = _log_survival_and_derivative(self, t)
    return _safe_log(-derivative) + log_s


def log_hazard(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return log hazard, broadcasting t against batch_shape (not a grid).

    Uses log(-d log S/dt) directly to avoid large-H cancellation. The hazard
    must be representable in linear space. Exactly -inf log survival gives
    NaN, even at finite support ends; finite log survival remains eligible
    when exp(log survival) underflows. Negative times return -inf.
    Zero uses the primitive's derivative; NaN times return NaN. Override
    this method to supply analytic limits where survival is zero.
    At +inf with raw log survival -inf, safe-input substitution isolates
    finite-entry parameter derivatives. The NaN boundary has no derivative
    guarantee; arbitrary unsafe custom overrides remain the author's concern.
    """
    log_s, derivative = _log_survival_and_derivative(self, t)
    undefined = jnp.isneginf(log_s) | jnp.isnan(log_s)
    safe_hazard = jnp.where(undefined, 1.0, -derivative)
    return jnp.where(undefined, jnp.nan, _safe_log(safe_hazard))


def hazard(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return exp(log_hazard(t)), honoring the family's log-hazard override.

    Output broadcasts t against batch_shape; shared time-first grids are
    separate. Negative times return zero; NaN times return NaN. At zero or
    +inf the log-hazard method supplies its boundary value.
    The generic zero-survival infinity guard applies only when log_hazard is
    also generic. Analytic overrides retain their parameter-dependent limits.
    """
    # spec 4.5: hazard's wrapper has already applied the input policy.
    method = type(self).log_hazard
    raw = getattr(method, "_survjax_raw", method)
    return jnp.exp(raw(self, t))


def cumulative_hazard(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return -log P(T > t), broadcasting t against batch_shape.

    Negative times return zero; NaN times return NaN. This is pointwise,
    rather than a shared time-first grid operation.
    A raw -inf limit at +inf is substituted before evaluation, then restored
    as +inf, isolating finite-entry parameter derivatives. Finite cure limits
    stay active; derivatives of nonfinite outputs are not guaranteed.
    """
    raw = _raw_log_survival(self)
    return -raw(self, t)


def log_cdf(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
    """Return log P(T <= t), broadcasting t against batch_shape.

    Uses log(-expm1(log_survival(t))) without subtracting nearly equal
    probabilities. Precision is retained until the family's log survival
    underflows to zero (roughly CDF 1e-38 in float32 or 1e-308 in float64).
    Override for CDF-natural families or smaller lower tails. Negative times
    and true zero CDF return -inf; NaN times return NaN. Evaluation is pointwise,
    not a shared time-first grid operation.
    At +inf, a raw log-survival limit of -inf gives zero with zero parameter
    gradient. Finite cure limits retain their parameter dependence, provided
    the primitive's operands are safe for differentiation.
    """
    # spec 4.5: the outer wrapper already substituted safe times.
    raw = _raw_log_survival(self)
    log_s = raw(self, t)
    zero_mass = log_s == 0
    # spec 7.2: substitute before log; an unselected log(0) poisons gradients.
    safe_log_s = jnp.where(
        zero_mass, jax.lax.stop_gradient(-jnp.ones_like(log_s)), log_s
    )
    value = jnp.log(-jnp.expm1(safe_log_s))
    return jnp.where(zero_mass, -jnp.inf, value)


def apply_policy(
    method: Callable[["SurvivalDistribution", ArrayLike], jax.Array],
    negative_value: float,
    *,
    density: bool = False,
    primitive: bool = False,
) -> Callable[["SurvivalDistribution", ArrayLike], jax.Array]:
    """Wrap one direct method, preserving its raw implementation on inheritance."""
    raw = getattr(method, "_survjax_raw", method)
    guard_zero_limit = primitive or raw in (
        survival,
        log_cdf,
        cumulative_hazard,
        log_hazard,
    )
    if primitive:
        zero_limit_value = -jnp.inf
    elif raw is cumulative_hazard:
        zero_limit_value = jnp.inf
    elif raw is log_hazard or raw is hazard:
        zero_limit_value = jnp.nan
    else:
        zero_limit_value = 0.0

    @wraps(raw)
    def evaluate(self: "SurvivalDistribution", t: ArrayLike) -> jax.Array:
        # spec 4.5: reject dtype misuse during tracing, without inspecting values.
        try:
            time = jnp.asarray(t)
        except (TypeError, ValueError) as exc:
            raise TypeError("time must contain real numeric values") from exc
        if not (
            jnp.issubdtype(time.dtype, jnp.integer)
            or jnp.issubdtype(time.dtype, jnp.floating)
        ):
            raise TypeError("time must contain real numeric values")
        try:
            output_shape = jnp.broadcast_shapes(time.shape, self.batch_shape)
        except ValueError as exc:
            raise TypeError(
                "time and distribution batch shapes are incompatible"
            ) from exc
        time = jnp.broadcast_to(time, output_shape)
        negative = time < 0
        nan_time = jnp.isnan(time)
        inactive = negative | nan_time
        # spec 4.5/7.2: stop only substituted inputs, never real parameters.
        reference = jax.lax.stop_gradient(self.reference_time())
        safe_time = jnp.where(inactive, reference, time)
        guard_limit = guard_zero_limit
        if raw is hazard:
            method = type(self).log_hazard
            guard_limit = getattr(method, "_survjax_raw", method) is log_hazard
        infinite = jnp.isposinf(time)
        if guard_limit or density:
            # spec 4.5/7.2: probe the true limit only for a Boolean condition.
            # Finite cure limits must retain their parameter dependence.
            log_s = _raw_log_survival(self)(self, safe_time)
            zero_limit = infinite & jnp.isneginf(log_s)
            substitute = infinite if density else zero_limit
            safe_time = jnp.where(substitute, reference, safe_time)
        value = raw(self, safe_time)
        if guard_limit:
            # Substitute before operating: a final mask alone gives 0 * inf AD.
            value = jnp.where(zero_limit, zero_limit_value, value)
        if density:
            # spec 4.5: the original limit, not the reference, decides NaN status.
            infinite_value = jnp.where(jnp.isnan(log_s), jnp.nan, -jnp.inf)
            value = jnp.where(infinite, infinite_value, value)
        policy_value = jnp.where(negative, negative_value, value)
        # spec 4.5: a NaN family value takes precedence over policy constants.
        policy_value = jnp.where(jnp.isnan(value), value, policy_value)
        policy_value = jnp.where(nan_time, jnp.nan, policy_value)
        return policy_value

    # A decorated subclass reuses the raw override, not its parent's wrapper.
    evaluate._survjax_raw = raw  # type: ignore[attr-defined]
    return evaluate
