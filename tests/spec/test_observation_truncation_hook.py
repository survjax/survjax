"""observation.truncation: the `log_conditional_interval_prob` hook (spec 4.2, 12.2; rev 51).

4.2: "`log_conditional_interval_prob(lower, upper, left_trunc, right_trunc)` |
Generic `log P(max(lower,L)<T<=min(upper,U)) - log Z`. Families override it
to simplify terms that would otherwise cancel (for example, Weibull
cumulative-hazard increments). The likelihood never dispatches on family
type." Overrides "receive endpoints `0` and `+inf` (the untruncated window)
and must handle them". 12.2 lists the hook in Slice B. Calling convention
guessed in Q-066: positional ``(lower, upper, left_trunc, right_trunc)``,
broadcasting against ``batch_shape`` like a direct-method argument, on every
registered family. References: ``observation_reference`` (mpmath).
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as ref
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, f, flag_data, ll

import survjax

FEATURE = pytest.mark.feature("observation.censoring", "observation.truncation")

# (lower, upper, L, U): inside, partial at L, partial at U, right-censored, left-censored, untruncated.
CASES = [
    (1.0, 2.0, 0.5, 3.0),
    (0.2, 1.5, 0.5, 3.0),
    (1.0, 4.0, 0.5, 3.0),
    (1.0, INF, 0.5, INF),
    (0.0, 2.0, 0.5, 3.0),
    (0.7, 2.6, 0.0, INF),
]


def _families():
    return {
        "exponential": (survjax.Exponential(rate=f(0.5)), ("exponential", (0.5,))),
        "weibull": (survjax.Weibull(shape=f(1.5), scale=f(2.0)), ("weibull", (1.5, 2.0))),
        "custom": (fam.power_hazard()(shape=f(1.5), scale=f(2.0)), ("weibull", (1.5, 2.0))),
    }


@FEATURE
@pytest.mark.parametrize("name", ["exponential", "weibull", "custom"])
def test_hook_values_match_the_spec_definition(name):
    """spec 4.2, rev 51: `log P(max(lower,L)<T<=min(upper,U)) - log Z` for inside, partial-overlap, one-sided and untruncated sets."""
    dist, (family, params) = _families()[name]
    lo, hi, lt, rt = (f([c[i] for c in CASES]) for i in range(4))
    out = np.asarray(dist.log_conditional_interval_prob(lo, hi, lt, rt))
    want = [ref.contribution(family, params, a, c, left=x, right=y) for a, c, x, y in CASES]
    assert_close(out, want)


@FEATURE
@pytest.mark.parametrize("name", ["exponential", "weibull", "custom"])
def test_untruncated_hook_is_the_interval_probability(name):
    """spec 4.2, rev 51: with the untruncated window `(0, +inf]` the hook equals `log_interval_prob` (`Z = 1`)."""
    dist, _ = _families()[name]
    a = np.asarray(dist.log_conditional_interval_prob(f(0.7), f(2.6), f(0.0), f(INF)))
    assert_close(a, np.asarray(dist.log_interval_prob(f(0.7), f(2.6))))


@FEATURE
@pytest.mark.parametrize("name", ["exponential", "weibull", "custom"])
def test_hook_on_the_whole_window_is_zero(name):
    """spec 2 and 4.2, rev 51: the set `(L, U]` itself has conditional probability one."""
    dist, _ = _families()[name]
    assert_close(np.asarray(dist.log_conditional_interval_prob(f(0.5), f(3.0), f(0.5), f(3.0))), 0.0)


@FEATURE
def test_hook_agrees_with_censored_row_contributions():
    """spec 2 and 4.2, rev 51: for valid censored rows, the hook's value is the row's likelihood contribution."""
    dist = survjax.Weibull(shape=f(1.5), scale=f(2.0))
    data = flag_data(
        ["interval", "interval", "right", "left"], [1.0, 0.2, 1.0, 2.0], [2.0, 1.5, NAN, NAN],
        left_trunc=[0.5] * 4, right_trunc=[3.0, 3.0, INF, 3.0],
    )
    hook = dist.log_conditional_interval_prob(
        f([1.0, 0.2, 1.0, 0.0]), f([2.0, 1.5, INF, 2.0]), f([0.5] * 4), f([3.0, 3.0, INF, 3.0])
    )
    assert_close(ll(dist, data), np.asarray(hook))


@FEATURE
def test_weibull_hook_keeps_precision_for_a_narrow_deep_window():
    """spec 4.2 and 7.2, rev 51: Weibull override with cumulative-hazard increments: `(30.1, 30.2]` in `(30, 31]` (`k=2`, `b=1`, `Z ~ e**-900`)."""
    dist = survjax.Weibull(shape=f(2.0), scale=f(1.0))
    out = np.asarray(dist.log_conditional_interval_prob(f(30.1), f(30.2), f(30.0), f(31.0)))
    want = ref.contribution("weibull", (2.0, 1.0), 30.1, 30.2, left=30.0, right=31.0)
    assert np.isfinite(out)
    assert_close(out, want)


@FEATURE
def test_hook_broadcasts_against_the_batch():
    """spec 5, rev 51: endpoint arrays of shape `(m, 1)` against a batch `(2,)` give `(m, 2)` (Q-066)."""
    dist = survjax.Exponential(rate=f([0.5, 2.0]))
    out = np.asarray(
        dist.log_conditional_interval_prob(f([[1.0], [0.2]]), f([[2.0], [1.5]]), f([[0.5], [0.5]]), f([[3.0], [3.0]]))
    )
    assert out.shape == (2, 2)
    want = [
        [ref.contribution("exponential", (r,), lo, hi, left=0.5, right=3.0) for r in (0.5, 2.0)]
        for lo, hi in ((1.0, 2.0), (0.2, 1.5))
    ]
    assert_close(out, want)


@FEATURE
def test_hook_gradient_under_jit():
    """spec 7.3, rev 51: the hook composes with `jit` and `grad`: Exponential `(1, 2]` in `(0.5, 3]`.

    ``g(lo, hi) = -r lo + log(-expm1(-r (hi - lo)))`` and the hook is
    ``g(1, 2) - g(0.5, 3)``, so ``d/dr = (-1 + 1/expm1(r)) - (-0.5 + 2.5/expm1(2.5 r))``.
    """

    def fn(r):
        return survjax.Exponential(rate=r).log_conditional_interval_prob(f(1.0), f(2.0), f(0.5), f(3.0))

    r = jnp.asarray(f(0.5))
    rr = float(np.asarray(f(0.5)))
    want = (-1.0 + 1.0 / np.expm1(rr)) - (-0.5 + 2.5 / np.expm1(2.5 * rr))
    assert_close(jax.jit(jax.grad(fn))(r), want)


@functools.cache
def _overriding_family():
    """Exponential-like family overriding the hook with the closed form ``g(lo, hi) - g(L, U)`` (spec 4.2).

    The override handles the endpoints ``0`` and ``+inf`` (``g(lo, +inf) = -r lo``).
    """

    def g(r, lo, hi):
        finite = jnp.isfinite(hi)
        w = jnp.where(finite, hi - lo, 1.0)
        return -r * lo + jnp.where(finite, jnp.log(-jnp.expm1(-r * w)), 0.0)

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class ClosedFormHook(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

        def log_conditional_interval_prob(self, lower, upper, left_trunc, right_trunc):
            lo = jnp.maximum(lower, left_trunc)
            hi = jnp.minimum(upper, right_trunc)
            return g(self.rate, lo, hi) - g(self.rate, left_trunc, right_trunc)

    return ClosedFormHook


@FEATURE
def test_a_correct_override_gives_the_exponential_likelihood():
    """spec 4.2, rev 51: a custom family overriding the hook correctly has the built-in Exponential likelihood on truncated censored rows.

    Which rows reach the hook is not specified (Q-066); a correct override
    must in any case leave every contribution equal to the closed form.
    """
    data = flag_data(
        ["exact", "right", "left", "interval"], [1.0, 1.0, 2.0, 0.2], [NAN, NAN, NAN, 1.5],
        left_trunc=[0.5] * 4, right_trunc=[3.0] * 4,
    )
    custom = _overriding_family()(rate=f(0.5))
    assert_close(ll(custom, data), ll(survjax.Exponential(rate=f(0.5)), data))
