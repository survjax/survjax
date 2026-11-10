"""observation.truncation: tiny `Z`, narrow windows, tails and `jax_debug_nans` (spec 7.1, 7.2, 12.1, 12.2; rev 51).

7.1: "Positive but tiny `Z` | evaluated accurately in log space; no floor";
7.2: "Never clip probabilities upward, floor `Z`, add an epsilon to data, or
turn a narrow interval into a density"; 12.2: "`Z=0` and tiny-`Z` cases ...
are Slice B evidence"; 12.1: `jax_debug_nans` on valid mixed batches.
References: ``observation_reference`` (50-digit mpmath of the spec 2 table
at dtype-rounded inputs), compared with zero absolute tolerance at the 12.1
``rtol``. Error bound, as in ``test_observation_censoring_numerics.py``: each
value is a difference of a few terms (``H(t) - H(L)`` included), each
accurate to a few ulps times ``1 + k |log(t/b)|`` (at most about 60 here);
relative to the values here (``|value|`` at least about 0.2) that is below
the 12.1 ``rtol`` in both dtypes.
"""

import jax
import jax.numpy as jnp
import numpy as np
import observation_reference as ref
import pytest
import spec_families as fam
from helpers import INF, NAN, X64, assert_tail_close, f, flag_data, ll
from tolerances import TOLERANCES

import survjax

TRUNC = pytest.mark.feature("observation.truncation")
BOTH = pytest.mark.feature("observation.censoring", "observation.truncation")
RTOL = TOLERANCES[np.dtype(np.float64 if X64 else np.float32)]["rtol"]

# Entry deep in the upper tail: Z = e**-L underflows even float64 for L = 750
# (float32: L = 110, Z ~ 1.7e-48).
DEEP_L = 750.0 if X64 else 110.0


def _check(dist, data, family, params, table):
    out = ll(dist, data)
    want = [
        ref.contribution(family, params, lo, hi, exact=ex, left=lt, right=rt) for lo, hi, ex, lt, rt in table
    ]
    assert np.isfinite(out).all(), out
    assert_tail_close(out, want, RTOL)
    return out


# ------------------------------------------------------------------- tiny Z


@TRUNC
def test_exponential_entry_deep_in_the_tail_is_evaluated_in_log_space():
    """spec 7.1 and 7.2, rev 51: Exponential `r=1`, `L = 750` (float64) / `110` (float32), where `Z = e**-L` underflows.

    Memorylessness gives the exact values: exact ``t = L+1``:
    ``log r - r (t - L) = -1``; right at ``L + 0.5``: ``-0.5``; exact
    ``L + 0.5`` in ``(L, L+1]``: ``-0.5 - log(1 - e**-1)``.
    """
    lt = DEEP_L
    data = flag_data(["exact", "right", "exact"], [lt + 1, lt + 0.5, lt + 0.5], left_trunc=[lt] * 3, right_trunc=[INF, INF, lt + 1])
    out = _check(
        survjax.Exponential(rate=f(1.0)),
        data,
        "exponential",
        (1.0,),
        [(lt + 1, lt + 1, True, lt, INF), (lt + 0.5, INF, False, lt, INF), (lt + 0.5, lt + 0.5, True, lt, lt + 1)],
    )
    assert_tail_close(out, [-1.0, -0.5, -0.5 - np.log(-np.expm1(-1.0))], RTOL)


@TRUNC
def test_weibull_entry_deep_in_the_tail():
    """spec 7.1 and 7.2, rev 51: Weibull `k=2`, `b=1`, `L=30` (`Z = e**-900`): exact at `30.5` and right at `30.25`."""
    data = flag_data(["exact", "right"], [30.5, 30.25], left_trunc=[30.0, 30.0])
    _check(
        survjax.Weibull(shape=f(2.0), scale=f(1.0)),
        data,
        "weibull",
        (2.0, 1.0),
        [(30.5, 30.5, True, 30.0, INF), (30.25, INF, False, 30.0, INF)],
    )


@TRUNC
def test_tiny_upper_truncation_gives_a_large_positive_contribution():
    """spec 7.1, rev 51: window `(0, 1e-30]`, exact at `5e-31` (Exponential `r=1`): `log f - log F(U) ~ +69.08`, a valid density ratio.

    ``Z = F(1e-30) ~ 1e-30`` is tiny but positive; no floor may cap it.
    """
    data = flag_data(["exact"], [5e-31], right_trunc=[1e-30])
    out = _check(survjax.Exponential(rate=f(1.0)), data, "exponential", (1.0,), [(5e-31, 5e-31, True, 0.0, 1e-30)])
    assert out[0] > 60


@TRUNC
def test_weibull_tiny_upper_truncation():
    """spec 7.1, rev 51: Weibull `k=2.5`, `b=2`, window `(0, 1e-10]`, exact at `5e-11` and right at `5e-11`."""
    data = flag_data(["exact", "right"], [5e-11, 5e-11], right_trunc=[1e-10, 1e-10])
    _check(
        survjax.Weibull(shape=f(2.5), scale=f(2.0)),
        data,
        "weibull",
        (2.5, 2.0),
        [(5e-11, 5e-11, True, 0.0, 1e-10), (5e-11, INF, False, 0.0, 1e-10)],
    )


@TRUNC
@pytest.mark.parametrize("k", [0.5, 1.5, 4.0])
def test_narrow_two_sided_window(k):
    """spec 7.2, rev 51: window `(1, 1 + 2**-20]` (exact in both dtypes), exact at `1 + 2**-21`: `log f - log Z ~ log(1/w)`.

    ``Z`` is a narrow-interval mass, which must come from cumulative-hazard
    increments (7.2), not from subtracting nearly equal survivals.
    """
    lt, rt = 1.0, 1.0 + 2.0**-20
    t = 1.0 + 2.0**-21
    data = flag_data(["exact", "right"], [t, t], left_trunc=[lt, lt], right_trunc=[rt, rt])
    _check(
        survjax.Weibull(shape=f(k), scale=f(2.0)),
        data,
        "weibull",
        (k, 2.0),
        [(t, t, True, lt, rt), (t, INF, False, lt, rt)],
    )


@BOTH
def test_censored_rows_in_a_deep_window():
    """spec 7.1 and 7.2, rev 51: left at `L + 0.5` and interval `(L + 0.2, L + 0.7]` in `(L, L + 1]` with `L` deep in the tail (Exponential `r=1`).

    Memorylessness: left ``log((1 - e**-0.5)/(1 - e**-1))``; interval
    ``-0.2 + log(1 - e**-0.5) - log(1 - e**-1)``.
    """
    lt = DEEP_L
    data = flag_data(["left", "interval"], [lt + 0.5, lt + 0.2], [NAN, lt + 0.7], left_trunc=[lt, lt], right_trunc=[lt + 1, lt + 1])
    out = ll(survjax.Exponential(rate=f(1.0)), data)
    want = ref.rows("exponential", (1.0,), [(0.0, lt + 0.5, False), (lt + 0.2, lt + 0.7, False)], left=lt, right=lt + 1)
    assert_tail_close(out, want, RTOL)


# --------------------------------------------------------- jax_debug_nans (12.1)


def _run(make, params, data):
    def total(p):
        out = make(**p).log_likelihood(data)
        return out.sum(), out

    with jax.debug_nans(True):
        (_, values), grads = jax.value_and_grad(total, has_aux=True)(params)
    values = np.asarray(values)
    assert np.isfinite(values).all(), values
    for g in jax.tree_util.tree_leaves(grads):
        assert np.isfinite(np.asarray(g)).all(), g
    return values


def _exact_right_rows(scale):
    """Valid truncated exact/right rows; ``scale`` is a time with ``H`` about 1e3. Row 3 is uninformative (right at entry)."""
    kinds = ["exact", "right", "exact", "right", "exact", "right"]
    time = [1.5, 2.0, 1.0 + 2.0**-21, 0.5, scale * 1.01, 5e-31]
    lt = [0.5, 0.5, 1.0, 0.5, scale, 0.0]
    rt = [INF, 3.0, 1.0 + 2.0**-20, 3.0, INF, 1e-30]
    return flag_data(kinds, time, left_trunc=lt, right_trunc=rt)


@TRUNC
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_truncated_batch_under_debug_nans(k):
    """spec 7.2 and 12.1, rev 51: Weibull `b=2`: no NaN in values or `(k, b)` gradients on truncated exact/right rows; the uninformative row is exactly zero."""
    params = {"shape": jnp.asarray(f(k)), "scale": jnp.asarray(f(2.0))}
    values = _run(survjax.Weibull, params, _exact_right_rows(2.0 * 1e3 ** (1.0 / k)))
    np.testing.assert_array_equal(values[3], 0.0)


@TRUNC
def test_exponential_and_custom_truncated_batch_under_debug_nans():
    """spec 4.2, 7.2 and 12.1, rev 51: Exponential and `const_hazard` (generic defaults) raise no NaN on the truncated rows."""
    for make in (survjax.Exponential, fam.const_hazard()):
        _run(make, {"rate": jnp.asarray(f(2.0))}, _exact_right_rows(500.0))


@BOTH
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_full_matrix_under_debug_nans(k):
    """spec 7.2 and 12.1, rev 51: all kinds with partial overlaps, covering rows (exactly zero) and a deep window raise no NaN."""
    scale = 2.0 * 1e3 ** (1.0 / k)
    kinds = ["exact", "right", "left", "interval", "interval", "left", "interval", "left"]
    time = [1.5, 2.0, 2.0, 0.2, 1.0, 3.0, 0.1, scale * 1.2]
    time2 = [NAN, NAN, NAN, 1.5, 4.0, NAN, 5.0, NAN]
    lt = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, scale]
    rt = [3.0, INF, INF, 3.0, 3.0, 3.0, 3.0, scale * 1.5]
    data = flag_data(kinds, time, time2, left_trunc=lt, right_trunc=rt)
    values = _run(survjax.Weibull, {"shape": jnp.asarray(f(k)), "scale": jnp.asarray(f(2.0))}, data)
    # Rows 5 (left at U) and 6 ((0.1, 5] containing (0.5, 3]) cover the window.
    np.testing.assert_array_equal(values[[5, 6]], [0.0, 0.0])
