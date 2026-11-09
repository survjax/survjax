"""observation.censoring: tails, tiny masses, narrow intervals and `jax_debug_nans` for left/interval rows (spec 4.1, 7.1, 7.2, 12.1, 12.2; rev 51).

12.2 makes "narrow intervals, interval cancellation ... Slice B evidence";
12.1 (Numerical behaviour): "tiny nonzero masses, both tails, narrow
intervals, cancellation, boundaries, inactive branches; no floors ... the
likelihood suite run eagerly under `jax_debug_nans` on valid mixed batches".
7.2: "for a finite exponential interval `(a,b]`, `-r*a + log(-expm1(-r*(b-a)))`;
Weibull uses cumulative-hazard increments computed without subtracting nearly
equal powers. Never clip probabilities upward, floor `Z`, add an epsilon to
data, or turn a narrow interval into a density."

References are 50-digit mpmath values of the spec 2 table at the
dtype-rounded inputs (``observation_reference``). Comparisons use zero
absolute tolerance (12.1: "Comparisons of tiny log probabilities ... use zero
absolute tolerance, so that an erroneous zero cannot pass"). The relative
scale is the 12.1 ``rtol`` unless stated: every value here is a sum of a few
terms each accurate to a few ulps times at most ``1 + k |log(t/b)|`` (the
``z**k`` amplification of ``helpers._reference``), and that bound is below
the 12.1 ``rtol`` for all cases. Upper-tail ``log F`` uses
``helpers.tail_rtol`` (as for A5's ``log_cdf`` tails).
"""

import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as ref
import pytest
import spec_families as fam
from helpers import (
    INF,
    NAN,
    X64,
    assert_tail_close,
    f,
    flag_data,
    ll,
    mp_log_cdf,
    mp_weibull_cumulative_hazard,
    tail_rtol,
)
from tolerances import TOLERANCES

import survjax

FEATURE = pytest.mark.feature("observation.censoring")
RTOL = TOLERANCES[np.dtype(np.float64 if X64 else np.float32)]["rtol"]


def _exp(r=1.0):
    return survjax.Exponential(rate=f(r))


def _weib(k, s):
    return survjax.Weibull(shape=f(k), scale=f(s))


# ------------------------------------------------------------------- lower tail


@FEATURE
def test_exponential_left_row_at_tiny_time_keeps_relative_precision():
    """spec 4.1 and 12.1, rev 51: left at `c = 1e-30`, `r = 1`: `log F = log(-expm1(-1e-30)) ~ -69.08`, never `-inf`."""
    out = ll(_exp(), flag_data(["left"], [1e-30]))
    assert_tail_close(out, [ref.contribution("exponential", (1.0,), 0.0, 1e-30)], RTOL)


@FEATURE
def test_weibull_left_row_where_the_cumulative_hazard_underflows():
    """spec 4.1 and 12.1, rev 51: Weibull `k=2.5`, `b=2`, left at `1e-30`: `H = (5e-31)**2.5` underflows float32, but `log F ~ log H ~ -174.6` is representable and required.

    4.1: the lower tail keeps relative precision "wherever `log H` is
    representable, even where `H` or `F` underflows".
    """
    out = ll(_weib(2.5, 2.0), flag_data(["left"], [1e-30]))
    assert np.isfinite(out).all()
    assert_tail_close(out, [ref.contribution("weibull", (2.5, 2.0), 0.0, 1e-30)], RTOL)


@FEATURE
def test_exponential_interval_at_tiny_times():
    """spec 7.2 and 12.1, rev 51: `(1e-30, 2e-30]`, `r=1`: `-r a + log(-expm1(-r w)) ~ -69.08`, a tiny nonzero mass with no floor."""
    out = ll(_exp(), flag_data(["interval"], [1e-30], [2e-30]))
    assert_tail_close(out, [ref.contribution("exponential", (1.0,), 1e-30, 2e-30)], RTOL)


@FEATURE
def test_weibull_interval_at_tiny_times():
    """spec 7.2 and 12.1, rev 51: Weibull `k=1.5`, `b=2`, `(1e-10, 2e-10]`: mass about `2.5e-16` (log about `-35.9`)."""
    out = ll(_weib(1.5, 2.0), flag_data(["interval"], [1e-10], [2e-10]))
    assert_tail_close(out, [ref.contribution("weibull", (1.5, 2.0), 1e-10, 2e-10)], RTOL)


# ------------------------------------------------------------------- upper tail

UPPER_H = [20.0, 33.0, 55.0, 600.0 if X64 else 70.0]


@FEATURE
@pytest.mark.parametrize("h", UPPER_H)
def test_exponential_left_row_in_the_upper_tail(h):
    """spec 4.1 and 12.1, rev 51: left at `c = H` (`r = 1`): `log F ~ -exp(-H)`, negative and nonzero, with relative precision."""
    (c,) = ref.rounded(h)
    out = ll(_exp(), flag_data(["left"], [c]))
    assert out[0] < 0
    assert_tail_close(out, [mp_log_cdf(c)], tail_rtol(c))


@FEATURE
@pytest.mark.parametrize("h", UPPER_H)
def test_weibull_left_row_in_the_upper_tail(h):
    """spec 4.1 and 12.1, rev 51: Weibull `k=2`, `b=1`, left at `sqrt(H)`: `log F ~ -exp(-H)` against mpmath at the rounded time."""
    (c,) = ref.rounded(np.sqrt(h))
    big_h = mp_weibull_cumulative_hazard(2.0, 1.0, c)
    out = ll(_weib(2.0, 1.0), flag_data(["left"], [c]))
    assert out[0] < 0
    assert_tail_close(out, [mp_log_cdf(big_h)], tail_rtol(float(big_h)))


@FEATURE
def test_exponential_interval_deep_in_the_upper_tail():
    """spec 7.2 and 12.1, rev 51: `(1000, 1001]`, `r=1`: `S(a)` and `S(b)` underflow, but `-1000 + log(1 - e**-1)` is finite and required."""
    out = ll(_exp(), flag_data(["interval"], [1000.0], [1001.0]))
    assert_tail_close(out, [ref.contribution("exponential", (1.0,), 1000.0, 1001.0)], RTOL)


@FEATURE
def test_weibull_interval_deep_in_the_upper_tail():
    """spec 7.2 and 12.1, rev 51: Weibull `k=2`, `b=1`, `(30, 30.5]`: `H(a) = 900`; value about `-900`, from increments in log space."""
    out = ll(_weib(2.0, 1.0), flag_data(["interval"], [30.0], [30.5]))
    assert_tail_close(out, [ref.contribution("weibull", (2.0, 1.0), 30.0, 30.5)], RTOL)


# ------------------------------------------------------------- narrow intervals

NARROW = [2.0**-20] + ([2.0**-40] if X64 else [])


@FEATURE
@pytest.mark.parametrize("width", NARROW, ids=lambda w: f"2^{int(np.log2(w))}")
def test_exponential_narrow_interval(width):
    """spec 7.2 and 12.2, rev 51: `(1, 1 + w]` with `w = 2**-20` (both dtypes) or `2**-40` (float64): `-1 + log(-expm1(-w))`.

    The endpoints are exact in the dtype, so ``b - a = w`` exactly; a narrow
    interval must not become a density or zero.
    """
    out = ll(_exp(), flag_data(["interval"], [1.0], [1.0 + width]))
    assert_tail_close(out, [ref.contribution("exponential", (1.0,), 1.0, 1.0 + width)], RTOL)


@FEATURE
@pytest.mark.parametrize("width", NARROW, ids=lambda w: f"2^{int(np.log2(w))}")
@pytest.mark.parametrize("k", [0.5, 1.5, 4.0])
def test_weibull_narrow_interval_uses_cumulative_hazard_increments(k, width):
    """spec 7.2, rev 51: Weibull `b=2`, `(1, 1 + w]`: "cumulative-hazard increments computed without subtracting nearly equal powers".

    ``H(b) - H(a) = H(a) expm1(k log1p(w/a))`` is about ``k w H(a)/a``; a
    naive ``H(b) - H(a)`` in the dtype has relative error ``eps H(a)/(H(b)-H(a))
    ~ eps/(k w)`` (about ``0.1`` in float32 for ``w = 2**-20``), far outside
    the 12.1 ``rtol`` on ``log(H(b) - H(a)) ~ log(k w/2**k)``.
    """
    out = ll(_weib(k, 2.0), flag_data(["interval"], [1.0], [1.0 + width]))
    assert_tail_close(out, [ref.contribution("weibull", (k, 2.0), 1.0, 1.0 + width)], RTOL)


# --------------------------------------------------------- jax_debug_nans (12.1)


def _mixed_rows(scale):
    """Valid mixed rows at time scale ``scale`` (the upper-tail time, ``H`` about 1e3).

    exact interior, right interior, right at 0 (uninformative), left at 1e-30,
    left interior, left upper tail, interval narrow, interval interior,
    interval upper tail.
    """
    kinds = ["exact", "right", "right", "left", "left", "left", "interval", "interval", "interval"]
    time = [1.5, 2.0, 0.0, 1e-30, 1.2, scale, 1.0, 0.5, scale]
    time2 = [NAN, NAN, NAN, NAN, NAN, NAN, 1.0 + 2.0**-20, 2.5, 1.5 * scale]
    return flag_data(kinds, time, time2)


def _run(make, params, data):
    def total(p):
        out = make(**p).log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, has_aux=True)(params)
    values = np.asarray(values)
    assert np.isfinite(values).all(), values
    np.testing.assert_array_equal(values[2], 0.0)
    for g in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(g)).all(), g
    return values


@FEATURE
def test_exponential_mixed_censored_batch_under_debug_nans():
    """spec 7.2 and 12.1, rev 51: Exponential `r=2` on the mixed rows: no NaN in any primitive, value or rate gradient."""
    _run(survjax.Exponential, {"rate": jnp.asarray(f(2.0))}, _mixed_rows(500.0))


@FEATURE
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_mixed_censored_batch_under_debug_nans(k):
    """spec 7.2 and 12.1, rev 51: Weibull `b=2` and shape `k` on the mixed rows: no NaN in values or `(k, b)` gradients."""
    params = {"shape": jnp.asarray(f(k)), "scale": jnp.asarray(f(2.0))}
    _run(survjax.Weibull, params, _mixed_rows(2.0 * 1e3 ** (1.0 / k)))


@FEATURE
def test_weibull_per_row_parameters_under_debug_nans():
    """spec 5, 7.2 and 12.1, rev 51: per-row shapes and scales on the mixed rows raise no NaN."""
    params = {
        "shape": jnp.asarray(f([0.5, 2.0, 0.5, 3.0, 1.0, 2.0, 0.7, 1.5, 2.0])),
        "scale": jnp.asarray(f([1.0, 2.0, 1e3, 1.0, 1e-2, 1.0, 2.0, 1.0, 1.0])),
    }
    _run(survjax.Weibull, params, _mixed_rows(40.0))


@FEATURE
def test_custom_family_mixed_censored_batch_under_debug_nans():
    """spec 4.2, 7.2 and 12.1, rev 51: `const_hazard` on generic defaults raises no NaN on the mixed rows."""
    _run(fam.const_hazard(), {"rate": jnp.asarray(f(2.0))}, _mixed_rows(500.0))


@FEATURE
def test_bounds_form_censored_batch_under_debug_nans():
    """spec 3.1, 7.2 and 12.1, rev 51: bounds-form left `(0, c]` and interval rows with exact rows raise no NaN."""
    data = survjax.SurvivalData.from_bounds(
        f([1.5, 0.0, 0.0, 0.5, 2.0]),
        f([1.5, 1e-30, 1.2, 2.5, INF]),
        exact=jnp.asarray([True, False, False, False, False]),
    )

    def total(k, s):
        out = survjax.Weibull(shape=k, scale=s).log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, argnums=(0, 1), has_aux=True)(
            jnp.asarray(f(0.5)), jnp.asarray(f(2.0))
        )
    assert np.isfinite(np.asarray(values)).all()
    for g in grads:
        assert np.isfinite(np.asarray(g))
