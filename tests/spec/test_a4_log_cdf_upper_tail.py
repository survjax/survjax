"""A4: the generic `log_cdf` default in the upper tail (spec 4.2, 7.2, 12.1; rev 25).

Spec 4.2, rev 25: the default is "`log1mexp(log S(t))` (section 7.2). In the
upper tail it keeps relative precision (`log F ~ -S`, negative and nonzero)
wherever `S` is a normal floating-point number; it never rounds a
representable `log F` to zero." Spec 12.1 lists "a custom family that relies on
the generic default" among the upper-tail checks, with derivatives on both
sides of the `log1mexp` split and mixed vectors under `jax_debug_nans`.
`log_cdf_grid` on this family is A5 (`test_a5_log_cdf_tails.py`).

ConstHazard: ``log S = -rate t``, so ``H = rate t``; ``rate = 0.5`` at
``t = 2H`` (exact in both dtypes). Derivatives as for the exponential:
``d/dr log F = t/expm1(H)``, ``d2/dr2 = -t**2 exp(H)/expm1(H)**2``.
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    INF,
    X64,
    assert_close,
    assert_rel_close,
    assert_tail_close,
    f,
    log1mexp,
    mp_log_cdf,
    tail_rtol,
)

UPPER_TAIL_H = (20.0, 33.0, 55.0, 600.0 if X64 else 70.0)
SPLIT_H = (0.5, 0.69, 0.70, 1.2)
DERIVATIVE_TAIL_H = (20.0, 33.0)


def _dist(rate=0.5):
    return fam.const_hazard()(rate=f(rate))


def _assert_upper_tail(out, hs):
    out = np.asarray(out)
    assert np.all(out < 0.0), out
    for i, h in enumerate(hs):
        assert_tail_close(out[i], mp_log_cdf(h), tail_rtol(h), err_msg=f"H={h}")


@pytest.mark.step("A4")
@pytest.mark.parametrize("h", UPPER_TAIL_H)
def test_default_log_cdf_in_upper_tail_is_negative_and_accurate(h):
    """spec 4.2/12.1, rev 25: the default "never rounds a representable `log F` to zero"; against mpmath `log1p(-exp(-H))`."""
    out = np.asarray(_dist().log_cdf(f(2.0 * h)))
    assert out < 0.0
    assert_tail_close(out, mp_log_cdf(h), tail_rtol(h))


@pytest.mark.step("A4")
def test_default_log_cdf_upper_tail_under_jit():
    """spec 12.1, rev 25: the default's upper-tail values hold under `jit`."""
    t = f([2.0 * h for h in UPPER_TAIL_H])
    out = jax.jit(lambda d, s: d.log_cdf(s))(_dist(), t)
    _assert_upper_tail(out, UPPER_TAIL_H)


@pytest.mark.step("A4")
def test_default_log_cdf_upper_tail_under_vmap_over_time():
    """spec 12.1, rev 25: the default's upper-tail values hold under `vmap` over scalar times."""
    t = f([2.0 * h for h in UPPER_TAIL_H])
    out = jax.vmap(_dist().log_cdf)(t)
    _assert_upper_tail(out, UPPER_TAIL_H)


@pytest.mark.step("A4")
def test_default_log_cdf_on_both_sides_of_log1mexp_split():
    """spec 7.2/12.1, rev 25: the default on both sides of `H = log 2` (`H` = 0.5, 0.69, 0.70, 1.2); `H = 0.5 t` at the rounded `t`."""
    t = f([2.0 * h for h in SPLIT_H])
    out = _dist().log_cdf(t)
    assert_rel_close(out, [mp_log_cdf(0.5 * float(x)) for x in t])


@pytest.mark.step("A4")
@pytest.mark.parametrize("h", SPLIT_H + DERIVATIVE_TAIL_H)
def test_default_log_cdf_rate_derivatives(h):
    """spec 7.3/12.1, rev 25: `d/dr log F = t/expm1(H)`, `d2/dr2 = -t**2 exp(H)/expm1(H)**2`; zero atol at upper-tail points."""
    t = f(2.0 * h)
    tt = float(t)
    hh = 0.5 * tt
    em1 = np.expm1(hh)
    d1, d2 = tt / em1, -(tt**2) * np.exp(hh) / em1**2

    def fn(r):
        return fam.const_hazard()(rate=r).log_cdf(t)

    def check(a, e):
        if h > 1:
            assert_tail_close(a, e, tail_rtol(h))
        else:
            assert_close(a, e)

    check(jax.grad(fn)(f(0.5)), d1)
    check(jax.jacfwd(fn)(f(0.5)), d1)
    check(jax.grad(jax.grad(fn))(f(0.5)), d2)
    check(jax.jacfwd(jax.jacrev(fn))(f(0.5)), d2)


@pytest.mark.step("A4")
def test_default_log_cdf_mixed_times_under_debug_nans():
    """spec 4.2/12.1, rev 25: `t = (0, 2, 66, +inf)` (`H = 0, 1, 33, inf`) raises no NaN under `jax_debug_nans`.

    Expected ``(log1mexp(0), log1mexp(-1), log1mexp(-33), log1mexp(-inf)) =
    (-inf, -0.4587, -4.66e-15, 0)``.
    """
    with jax.debug_nans(True):
        out = np.asarray(_dist().log_cdf(f([0.0, 2.0, 66.0, INF])))
    np.testing.assert_array_equal(out[[0, 3]], [-INF, 0.0])
    assert_close(out[1], log1mexp(-1.0))
    assert_tail_close(out[2], mp_log_cdf(33.0), tail_rtol(33.0))
