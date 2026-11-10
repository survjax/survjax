"""Custom families defined by the tests through the public extension interface (spec 4.2).

A4 tests the distribution base with families written here, never with the
built-ins (those arrive at A5). Every family follows the documented pattern
exactly: ``@survjax.register_family`` over a frozen dataclass subclass of
``survjax.SurvivalDistribution`` using ``survjax.param``, ``survjax.static``
and ``survjax.distribution`` fields, supplying only ``log_survival`` and
``reference_time`` unless the test is about an override.

The guarded families' ``log_survival`` itself meets the direct-method value
policy of spec 4.5 (``log S = 0`` for ``t < 0``, NaN for NaN ``t``, correct
limits at ``0`` and ``+inf``), using safe inputs so that unselected branches
carry no NaN tangents. Since spec rev 12 the base applies that policy itself
(Q-019, resolved), and since rev 26 it also supplies ``log S``, ``S``,
``log F`` and ``H`` at ``t=0`` and ``t=+inf`` and calls ``log_survival`` only
for finite ``t>0`` (except inside the generic ``log_prob`` and
``log_hazard`` defaults at ``t=0``). The guarded families keep the earlier
tests independent of that; ``naive_power``, ``naive_log_prob_override``,
``boundary_nan_hazard``, ``softplus_log_logistic`` and ``mixture_cure`` test it
directly.

Classes are built lazily and cached: a missing survjax name then fails only
the A4 tests that use it, not collection of earlier steps, and every call
returns the same class, so tree structures compare equal across calls.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp

import survjax


def _safe_time(t):
    """Nonnegative time for active branches; NaN and +inf pass through unchanged."""
    t = jnp.asarray(t)
    return t, jnp.where(t < 0, 0.0, t)


@functools.cache
def const_hazard():
    """Exponential-like: ``S(t) = exp(-rate * t)``; ``log f = log(rate) - rate * t``."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ConstHazard(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -self.rate * ts)

        def reference_time(self):
            return 1.0 / self.rate

    return ConstHazard


@functools.cache
def power_hazard():
    """Weibull-like, with the Weibull keyword ``shape`` (spec 4.1): ``S(t) = exp(-(t/scale)**shape)``.

    With ``k = shape``, ``b = scale``, ``z = t/b``:
    ``log f = log k - log b + (k-1) log z - z**k``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class PowerHazard(survjax.SurvivalDistribution):
        shape: jax.Array = survjax.param()
        scale: jax.Array = survjax.param()

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -((ts / self.scale) ** self.shape))

        def reference_time(self):
            return self.scale

    return PowerHazard


@functools.cache
def component_rate():
    """A batch parameter, a trailing component axis and a static field (the spec 4.2 example).

    ``S(t) = exp(-rate * c * t)`` with ``c = sum(w)`` (``method="sum"``) or
    ``c = max(w)`` (``method="max"``) over the component axis.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ComponentRate(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()
        w: jax.Array = survjax.param(component_ndim=1)
        method: str = survjax.static(default="sum")

        def _c(self):
            if self.method == "sum":
                return jnp.sum(self.w, axis=-1)
            return jnp.max(self.w, axis=-1)

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -self.rate * self._c() * ts)

        def reference_time(self):
            return 1.0 / (self.rate * self._c())

    return ComponentRate


@functools.cache
def scaled_hazard():
    """A nested distribution field (``survjax.distribution()``): ``log S = exp(log_ratio) * log S_base``."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ScaledHazard(survjax.SurvivalDistribution):
        base: survjax.SurvivalDistribution = survjax.distribution()
        log_ratio: jax.Array = survjax.param()

        def log_survival(self, t):
            return jnp.exp(self.log_ratio) * self.base.log_survival(t)

        def reference_time(self):
            return self.base.reference_time()

    return ScaledHazard


OVERRIDE_SENTINEL = 7.0


@functools.cache
def log_prob_override():
    """ConstHazard with a ``log_prob`` override returning a sentinel, to observe dispatch only."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class LogProbOverride(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -self.rate * ts)

        def reference_time(self):
            return 1.0 / self.rate

        def log_prob(self, t):
            shape = jnp.broadcast_shapes(jnp.shape(t), jnp.shape(self.rate))
            return jnp.full(shape, OVERRIDE_SENTINEL)

    return LogProbOverride


def unregistered():
    """A frozen dataclass subclass deliberately not passed to ``register_family``."""

    @dataclasses.dataclass(frozen=True)
    class Unregistered(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            t, ts = _safe_time(t)
            return jnp.where(t < 0, 0.0, -self.rate * ts)

        def reference_time(self):
            return 1.0 / self.rate

    return Unregistered


@functools.cache
def naive_power():
    """Weibull-like with no input guards: ``log S = -(t/scale)**shape`` as written for ``t >= 0``.

    Spec 4.2/4.5 (rev 12): family code need only be correct on ``[0, +inf]``;
    the base applies the value policy to other inputs. For ``t < 0`` this
    formula gives NaN (non-integer ``shape``) or a positive ``log S``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class NaivePower(survjax.SurvivalDistribution):
        shape: jax.Array = survjax.param()
        scale: jax.Array = survjax.param()

        def log_survival(self, t):
            return -((t / self.scale) ** self.shape)

        def reference_time(self):
            return self.scale

    return NaivePower


@functools.cache
def naive_log_prob_override():
    """NaivePower plus an unguarded ``log_prob`` override (the Weibull log density for ``t >= 0``)."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class NaiveLogProbOverride(survjax.SurvivalDistribution):
        shape: jax.Array = survjax.param()
        scale: jax.Array = survjax.param()

        def log_survival(self, t):
            return -((t / self.scale) ** self.shape)

        def reference_time(self):
            return self.scale

        def log_prob(self, t):
            k, b = self.shape, self.scale
            return jnp.log(k / b) + (k - 1.0) * jnp.log(t / b) - (t / b) ** k

    return NaiveLogProbOverride


@functools.cache
def log_hazard_override():
    """ConstHazard with an analytic ``log_hazard`` override: ``log h = log(rate)`` for every ``t >= 0``."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class LogHazardOverride(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

        def log_hazard(self, t):
            return jnp.broadcast_to(
                jnp.log(self.rate),
                jnp.broadcast_shapes(jnp.shape(t), jnp.shape(self.rate)),
            )

    return LogHazardOverride


@functools.cache
def bounded_support():
    """Uniform on ``(0, end]``: ``S(t) = 1 - t/end`` for ``t < end`` and ``0`` from ``end`` on.

    ``log S = log1p(-t/end)`` for ``t < end`` (``-inf`` otherwise), so
    ``h(t) = 1/(end - t)`` for ``t < end``. Only ``log_survival`` and
    ``reference_time`` are supplied (``support()`` is not needed by A4 tests).
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class BoundedSupport(survjax.SurvivalDistribution):
        end: jax.Array = survjax.param()

        def log_survival(self, t):
            inside = t < self.end
            ts = jnp.where(inside, t, 0.0)
            return jnp.where(inside, jnp.log1p(-ts / self.end), -jnp.inf)

        def reference_time(self):
            return 0.5 * self.end

    return BoundedSupport


@functools.cache
def kw_only_static_first():
    """Spec 4.2 (rev 19) recipe variant: ``kw_only=True`` lets a defaulted static field precede a parameter."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True, kw_only=True)
    class KwOnlyStaticFirst(survjax.SurvivalDistribution):
        method: str = survjax.static(default="sum")
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

    return KwOnlyStaticFirst


@functools.cache
def power_log_prob_override():
    """PowerHazard plus a ``log_prob`` override with correct limits at ``t=0`` (spec 4.5, rev 26).

    ``log f = log(k/b) + xlogy(k-1, t/b) - (t/b)**k``; ``xlogy(0, 0) = 0``, so
    at ``t=0``: ``+inf`` for ``k<1``, ``log(1/b)`` for ``k=1``, ``-inf`` for
    ``k>1`` (the density limits the override supplies at ``t=0``).
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class PowerLogProbOverride(survjax.SurvivalDistribution):
        shape: jax.Array = survjax.param()
        scale: jax.Array = survjax.param()

        def log_survival(self, t):
            return -((t / self.scale) ** self.shape)

        def reference_time(self):
            return self.scale

        def log_prob(self, t):
            k, z = self.shape, t / self.scale
            return jnp.log(k / self.scale) + jax.scipy.special.xlogy(k - 1.0, z) - z**k

    return PowerLogProbOverride


@functools.cache
def boundary_nan_hazard():
    """Exponential-like, but ``log_survival`` is NaN at ``t=0`` and ``t=+inf`` (spec 4.2/4.5, rev 26).

    ``log S = -rate * t * (t/t)``: exact ``-rate * t`` for finite ``t > 0``
    (``t/t`` is exactly 1), NaN at ``0`` (``0/0``) and ``+inf`` (``inf/inf``).
    Rev 26: ``log_survival`` "Called for finite `t>0`; the base supplies the
    values at `t=0` and `t=+inf`", so the formula "need not handle the
    boundaries".
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class BoundaryNanHazard(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            return -self.rate * t * (t / t)

        def reference_time(self):
            return 1.0 / self.rate

    return BoundaryNanHazard


@functools.cache
def softplus_log_logistic():
    """Spec 4.5 (rev 26) test case: log-logistic with the plain ``log S = -softplus(beta*(log t - log alpha))``.

    At ``t=0``: ``log t = -inf``, ``softplus(-inf) = 0``, so the value is
    ``0`` but ``d/dbeta = -sigmoid(-inf) * (-inf) = 0 * inf = NaN``. At
    ``t=+inf`` the value is ``-inf`` and ``S = exp(-inf) = 0`` has a NaN
    gradient by the same product. Interior: ``S(t) = 1/(1 + (t/alpha)**beta)``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class SoftplusLogLogistic(survjax.SurvivalDistribution):
        alpha: jax.Array = survjax.param()
        beta: jax.Array = survjax.param()

        def log_survival(self, t):
            return -jax.nn.softplus(self.beta * (jnp.log(t) - jnp.log(self.alpha)))

        def reference_time(self):
            return jnp.broadcast_to(
                self.alpha,
                jnp.broadcast_shapes(jnp.shape(self.alpha), jnp.shape(self.beta)),
            )

    return SoftplusLogLogistic


@functools.cache
def mixture_cure():
    """An improper (cure) family: ``S(t) = c + (1-c) exp(-rate * t)``, ``S(+inf) = c`` (spec 2, 4.2, rev 26).

    ``log S = logaddexp(log c, log1p(-c) - rate * t)``; it overrides
    ``log_survival_at_infinity()`` to return ``log c`` per batch entry. With
    ``c < 0`` both are NaN (``log`` of a negative).
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class MixtureCure(survjax.SurvivalDistribution):
        cure: jax.Array = survjax.param()
        rate: jax.Array = survjax.param()

        def _batch(self):
            return jnp.broadcast_shapes(jnp.shape(self.cure), jnp.shape(self.rate))

        def log_survival(self, t):
            return jnp.logaddexp(
                jnp.log(self.cure), jnp.log1p(-self.cure) - self.rate * t
            )

        def log_survival_at_infinity(self):
            return jnp.broadcast_to(jnp.log(self.cure), self._batch())

        def reference_time(self):
            return jnp.broadcast_to(1.0 / self.rate, self._batch())

    return MixtureCure


# ---------------------------------------------- declared constraints (spec 4.7, rev 28)


def _rate_family(constrained):
    """``S(t) = exp(-rate * t)`` as written (no guards); ``rate`` declared ``positive`` or undeclared."""
    kw = {"constraint": survjax.constraints.positive} if constrained else {}

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class RateFamily(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param(**kw)

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

    return RateFamily


@functools.cache
def constrained_rate():
    """Exponential-like with ``rate`` declared ``positive`` (spec 4.2 recipe, rev 28)."""
    return _rate_family(True)


@functools.cache
def unconstrained_rate():
    """The same formula as ``constrained_rate`` with no declaration, for value comparisons."""
    return _rate_family(False)


@functools.cache
def component_constrained():
    """Constraints on batch and component fields, with an undeclared field between them (spec 4.7, rev 28).

    Declaration order ``rate`` (positive), ``free`` (none), ``w`` (positive,
    ``component_ndim=1``: one remaining component axis, reduced by ``any``),
    ``knots`` (ordered, ``component_ndim=1``: ``event_dim=1`` consumes it).
    ``S(t) = exp(-rate * sum(w) * t)``; ``free`` and ``knots`` do not enter it.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ComponentConstrained(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)
        free: jax.Array = survjax.param()
        w: jax.Array = survjax.param(
            component_ndim=1, constraint=survjax.constraints.positive
        )
        knots: jax.Array = survjax.param(
            component_ndim=1, constraint=survjax.constraints.ordered
        )

        def log_survival(self, t):
            return -self.rate * jnp.sum(self.w, axis=-1) * t

        def reference_time(self):
            return 1.0 / (self.rate * jnp.sum(self.w, axis=-1))

    return ComponentConstrained


@functools.cache
def capped_rate():
    """``rate`` and ``cap`` declared ``positive`` plus one named condition ``rate_above_cap`` (spec 4.7, rev 28).

    ``S(t) = exp(-rate * t)``; the condition is violated where ``rate > cap``.
    The formula is finite there, so only the declaration can flag it.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class CappedRate(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)
        cap: jax.Array = survjax.param(constraint=survjax.constraints.positive)

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

        def parameter_conditions(self):
            return {"rate_above_cap": self.rate > self.cap}

    return CappedRate


@functools.cache
def scaled_capped():
    """A nested ``capped_rate`` baseline with an own constraint and condition (spec 4.7, rev 28).

    ``log S = exp(log_ratio) * log S_base``; ``log_ratio`` declares ``real``;
    own condition ``ratio_too_large`` where ``log_ratio > 2``. Expected keys:
    fields ``log_ratio, base.rate, base.cap``; conditions
    ``ratio_too_large, base.rate_above_cap``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ScaledCapped(survjax.SurvivalDistribution):
        base: survjax.SurvivalDistribution = survjax.distribution()
        log_ratio: jax.Array = survjax.param(constraint=survjax.constraints.real)

        def log_survival(self, t):
            return jnp.exp(self.log_ratio) * self.base.log_survival(t)

        def reference_time(self):
            return self.base.reference_time()

        def parameter_conditions(self):
            return {"ratio_too_large": self.log_ratio > 2.0}

    return ScaledCapped


# ------------------------------------- post-release generic summaries (spec 8.8, rev 50)


@functools.cache
def uniform_support():
    """Uniform on ``(0, end]`` with a ``support()`` override (spec 4.2, 8.1-8.3, rev 50).

    ``S(t) = 1 - t/end`` for ``t < end`` and ``0`` from ``end`` on, so
    ``log S = log1p(-t/end)`` inside and ``-inf`` beyond. ``support()`` returns
    ``(0, end)`` per batch entry (Q-059). Hand values: ``rmst(tau) = tau -
    tau**2/(2 end)`` for ``tau <= end`` and ``end/2`` beyond; ``mean = end/2``;
    ``quantile(p) = p * end``, with ``quantile(1) = end`` from ``support()``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class UniformSupport(survjax.SurvivalDistribution):
        end: jax.Array = survjax.param()

        def log_survival(self, t):
            inside = t < self.end
            ts = jnp.where(inside, t, 0.0)
            return jnp.where(inside, jnp.log1p(-ts / self.end), -jnp.inf)

        def reference_time(self):
            return 0.5 * self.end

        def support(self):
            return jnp.zeros_like(self.end), self.end

    return UniformSupport


@functools.cache
def nan_beyond():
    """Valid parameters whose formula is NaN beyond ``cut`` (spec 8.3 "A non-finite integrand at any node is a failure status").

    ``log S = -rate * t`` for ``t <= cut`` and NaN for ``t > cut``; nothing is
    declared, so no parameter issue is reported. ``reference_time()`` is
    ``0.25 / rate``, inside ``(0, cut]`` for the cases used (``rate = 1``,
    ``cut = 1``), so an RMST to ``tau <= cut`` never evaluates the NaN region
    and one to ``tau > cut`` must.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class NanBeyond(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()
        cut: jax.Array = survjax.param()

        def log_survival(self, t):
            return jnp.where(t <= self.cut, -self.rate * t, jnp.nan)

        def reference_time(self):
            return jnp.broadcast_to(
                0.25 / self.rate,
                jnp.broadcast_shapes(jnp.shape(self.rate), jnp.shape(self.cut)),
            )

    return NanBeyond


@functools.cache
def far_reference_log_logistic():
    """Log-logistic ``S(t) = 1/(1 + (t/alpha)**beta)`` whose ``reference_time()`` is always ``1`` (spec 4.2, 8.2).

    ``reference_time()`` need only be positive and finite (4.2). With
    ``alpha = 1e20`` every quantile lies about 46 log-units from the bracket
    start ``log reference_time() = 0``, so a solver limited to one bracket
    expansion cannot bracket it (8.2: "expands geometrically ... up to a static
    step limit").
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class FarReferenceLogLogistic(survjax.SurvivalDistribution):
        alpha: jax.Array = survjax.param()
        beta: jax.Array = survjax.param()

        def log_survival(self, t):
            return -jax.nn.softplus(self.beta * (jnp.log(t) - jnp.log(self.alpha)))

        def reference_time(self):
            return jnp.ones(
                jnp.broadcast_shapes(jnp.shape(self.alpha), jnp.shape(self.beta)),
                dtype=jnp.result_type(self.alpha),
            )

    return FarReferenceLogLogistic


# ------------------------------------------- mean_with_info overrides (spec 8.1, rev 54, 56)


def _override_info(status_code, shape, dtype):
    """A ``SummaryInfo`` for a family's ``mean_with_info`` override (spec 8.1 rev 56, 8.5 schema; Q-056).

    Rev 56 (8.1): "An analytical answer reports status `converged` (or
    `known_infinite`), a NaN `error_estimate` and zero work counts." Status
    codes are ``int32`` (Q-056: integer codes comparing equal to the enum).
    """
    zeros = jnp.zeros(shape, dtype=jnp.int32)
    return survjax.SummaryInfo(
        status=jnp.broadcast_to(jnp.asarray(status_code, dtype=jnp.int32), shape),
        error_estimate=jnp.full(shape, jnp.nan, dtype=dtype),
        num_levels=zeros,
        num_solver_steps=zeros,
        num_bracket_steps=zeros,
    )


@jax.custom_jvp
def _plus_inf_with_nan_tangent(x):
    """``+inf`` shaped like ``x`` whose tangent is NaN (spec 8.4: known-infinite means get NaN tangents).

    The tangent rule ``dx * nan`` is linear, so reverse mode works. Callers
    pass ``x = where(finite, 0, params)``, so forward tangents of the finite
    entries are untouched, and with per-entry parameters their reverse
    gradients are too (the guarding ``where`` transposes to zero there). A
    linear NaN coefficient still meets a zero cotangent (``0 * nan``), so in
    reverse mode the infinite entries' own parameters get NaN gradients even
    from objectives that exclude them; tests use forward-mode tangents only.
    """
    return jnp.full_like(x, jnp.inf)


@_plus_inf_with_nan_tangent.defjvp
def _plus_inf_with_nan_tangent_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    return _plus_inf_with_nan_tangent(x), dx * jnp.nan


@functools.cache
def known_mean_log_logistic():
    """Log-logistic ``S(t) = 1/(1 + (t/alpha)**beta)`` reporting its mean through a ``mean_with_info`` override (spec 8.1, rev 54, 56).

    8.1: "A family reports it [a known infinite mean] by overriding
    `mean_with_info`; `mean` returns the value part of `mean_with_info`, so
    one override serves both." The override follows the 8.1 table:
    ``alpha*(pi/beta)/sin(pi/beta)`` with status ``converged`` for
    ``beta > 1``; ``+inf`` with status ``known_infinite`` for ``beta <= 1``.
    ``log S = -softplus(beta*(log t - log alpha))`` (spec 4.1 recipe);
    ``reference_time() = alpha``.

    Rev 56 (8.1) fixes the override contract this family follows: the
    signature ``mean_with_info(self, *, options=None) -> tuple[Array,
    SummaryInfo]`` ("may ignore `options`"); an analytical answer reports
    ``converged`` or ``known_infinite``, a NaN ``error_estimate`` and zero
    work counts; and "The override is responsible for the section 8.4 NaN
    tangents of a known infinite mean", so the ``+inf`` entries carry a NaN
    tangent (``_plus_inf_with_nan_tangent``). This is a custom override of
    the extension interface; the built-in ``LogLogistic`` (feature
    ``family.loglogistic``) is tested separately.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class KnownMeanLogLogistic(survjax.SurvivalDistribution):
        alpha: jax.Array = survjax.param()
        beta: jax.Array = survjax.param()

        def log_survival(self, t):
            return -jax.nn.softplus(self.beta * (jnp.log(t) - jnp.log(self.alpha)))

        def reference_time(self):
            return jnp.broadcast_to(
                self.alpha,
                jnp.broadcast_shapes(jnp.shape(self.alpha), jnp.shape(self.beta)),
            )

        def mean_with_info(self, *, options=None):
            alpha, beta = jnp.broadcast_arrays(self.alpha, self.beta)
            finite = beta > 1.0
            x = jnp.pi / jnp.where(finite, beta, 2.0)
            infinite = _plus_inf_with_nan_tangent(jnp.where(finite, 0.0, alpha + beta))
            value = jnp.where(finite, alpha * x / jnp.sin(x), infinite)
            status = survjax.SummaryInfo.Status
            code = jnp.where(finite, int(status.converged), int(status.known_infinite))
            return value, _override_info(code, value.shape, value.dtype)

    return KnownMeanLogLogistic


MEAN_SENTINEL = 7.0


@functools.cache
def mean_with_info_sentinel():
    """``S(t) = exp(-rate*t)`` whose ``mean_with_info`` override returns ``MEAN_SENTINEL`` (not ``1/rate``), to observe dispatch only (spec 8.1, rev 54).

    "`mean` returns the value part of `mean_with_info`, so one override
    serves both": ``mean()`` must return the sentinel, not the generic
    quadrature value ``1/rate``. Status ``converged``.
    """

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class MeanWithInfoSentinel(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

        def mean_with_info(self, *, options=None):
            value = jnp.full(jnp.shape(self.rate), MEAN_SENTINEL, dtype=jnp.result_type(self.rate))
            code = int(survjax.SummaryInfo.Status.converged)
            return value, _override_info(code, value.shape, value.dtype)

    return MeanWithInfoSentinel
