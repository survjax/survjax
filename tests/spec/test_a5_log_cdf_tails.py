"""A5: `log_cdf` in both tails for the built-in families (spec 4.1, 4.2, 7.2, 12.1; rev 25).

Spec 4.1, rev 25: Exponential and Weibull compute `log_cdf` from `log H`
(`log r + log t`; `k*(log t - log b)`), "`log H` itself where `H` is tiny,
otherwise `log1mexp(-H)`". "Both tails keep relative precision: the lower tail
wherever `log H` is representable, even where `H` or `F` underflows; the upper
tail (`log F ~ -S`) wherever `S=exp(-H)` is a normal floating-point number, so
`log_cdf` is negative there and never rounds to zero."

Spec 12.1, rev 25 ("`log_cdf` tails"): upper tail against mpmath
`log1p(-exp(-H))` at `H` of about 20, 33 and 55 and deeper in each dtype's
normal range (`H=70` in float32, `H=600` in float64), eager, under `jit` and
`vmap`, and through `log_cdf_grid`; built-in lower tails where `H` or `F`
underflows; values and first and second parameter derivatives on both sides of
the `log1mexp` split (`H = log 2`) and at upper-tail points; mixed vectors with
`0`, interior, upper-tail and `+inf` times under `jax_debug_nans`.
"Comparisons of tiny log probabilities or derivatives use zero absolute
tolerance."

Cases (``H`` is recomputed from the dtype-rounded inputs):

- Exponential ``r = 0.5`` at ``t = 2H`` (exact in both dtypes), ``H = r t``.
- Weibull ``k = 2.5``, ``b = 3`` (non-unit shape and scale) at
  ``t = 3 H**(1/k)``, ``H = (t/b)**k`` in mpmath.
- ConstHazard (``log S = -rate t``, generic `log_cdf` default) at ``t = 2H``,
  for `log_cdf_grid` only; its direct-method tail tests are A4
  (`test_a4_log_cdf_upper_tail.py`).

Derivatives of ``log F = log1mexp(-H)``: with ``g1 = 1/expm1(H)`` and
``g2 = -exp(H)/expm1(H)**2`` (``d/dH`` and ``d2/dH2``):
Exponential ``d/dr = t g1`` (spec: ``t/expm1(H)``), ``d2/dr2 = t**2 g2``.
Weibull, ``L = log(t/b)``: ``dH/dk = H L``, ``d2H/dk2 = H L**2``,
``dH/db = -k H/b``, ``d2H/db2 = k (k+1) H/b**2``; then
``d/dp log F = g1 H'`` and ``d2/dp2 log F = g2 H'**2 + g1 H''``.
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
    mp_weibull_cumulative_hazard,
    tail_rtol,
)

import survjax

# Spec 12.1, rev 25: "about 20, 33 and 55 and deeper in each dtype's normal range".
UPPER_TAIL_H = (20.0, 33.0, 55.0, 600.0 if X64 else 70.0)
# Spec 12.1, rev 25: "both sides of the `log1mexp` split (`H = log 2`)".
SPLIT_H = (0.5, 0.69, 0.70, 1.2)
DERIVATIVE_TAIL_H = (20.0, 33.0)

WEIBULL_K, WEIBULL_B = 2.5, 3.0


def _exponential_case(h):
    t = f(2.0 * h)
    return survjax.Exponential(rate=f(0.5)), t, 0.5 * float(t)


def _weibull_case(h):
    t = f(WEIBULL_B * h ** (1.0 / WEIBULL_K))
    hk = mp_weibull_cumulative_hazard(WEIBULL_K, WEIBULL_B, t)
    return survjax.Weibull(shape=f(WEIBULL_K), scale=f(WEIBULL_B)), t, hk


def _generic_case(h):
    t = f(2.0 * h)
    return fam.const_hazard()(rate=f(0.5)), t, 0.5 * float(t)


CASES = {
    "exponential": _exponential_case,
    "weibull": _weibull_case,
}


def _vector_case(name, hs):
    """One scalar distribution, a time vector, and the exact ``H`` per time."""
    make = {**CASES, "generic": _generic_case}[name]
    parts = [make(h) for h in hs]
    t = np.array([p[1] for p in parts], dtype=parts[0][1].dtype)
    return parts[0][0], t, [p[2] for p in parts]


def _assert_upper_tail(out, hs_exact, hs_nominal):
    out = np.asarray(out)
    assert np.all(out < 0.0), out
    for i, (hx, hn) in enumerate(zip(hs_exact, hs_nominal, strict=True)):
        assert_tail_close(out[i], mp_log_cdf(hx), tail_rtol(hn), err_msg=f"H={hn}")


# --------------------------------------------------------------- upper tail, eager


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", UPPER_TAIL_H)
def test_exponential_log_cdf_in_upper_tail_is_negative_and_accurate(h):
    """spec 4.1/12.1, rev 25: Exponential upper-tail `log F` against mpmath `log1p(-exp(-H))`, negative and nonzero."""
    dist, t, hx = _exponential_case(h)
    out = np.asarray(dist.log_cdf(t))
    assert out < 0.0
    assert_tail_close(out, mp_log_cdf(hx), tail_rtol(h))


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", UPPER_TAIL_H)
def test_weibull_log_cdf_in_upper_tail_is_negative_and_accurate(h):
    """spec 4.1/12.1, rev 25: Weibull (`k=2.5`, `b=3`) upper-tail `log F` against mpmath, negative and nonzero."""
    dist, t, hx = _weibull_case(h)
    out = np.asarray(dist.log_cdf(t))
    assert out < 0.0
    assert_tail_close(out, mp_log_cdf(hx), tail_rtol(h))


# --------------------------------------------------------------- upper tail, transformed


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", list(CASES))
def test_upper_tail_log_cdf_under_jit(name):
    """spec 12.1, rev 25: the upper-tail values hold under `jit`."""
    dist, t, hx = _vector_case(name, UPPER_TAIL_H)
    out = jax.jit(lambda d, s: d.log_cdf(s))(dist, t)
    _assert_upper_tail(out, hx, UPPER_TAIL_H)


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", list(CASES))
def test_upper_tail_log_cdf_under_vmap_over_time(name):
    """spec 12.1, rev 25: the upper-tail values hold under `vmap` over scalar times."""
    dist, t, hx = _vector_case(name, UPPER_TAIL_H)
    out = jax.vmap(dist.log_cdf)(t)
    _assert_upper_tail(out, hx, UPPER_TAIL_H)


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", [*CASES, "generic"])
def test_upper_tail_log_cdf_grid(name):
    """spec 5/12.1, rev 25: the upper-tail values hold "through `log_cdf_grid`" (built-ins and the generic default)."""
    dist, t, hx = _vector_case(name, UPPER_TAIL_H)
    out = dist.log_cdf_grid(t)
    assert np.shape(out) == (len(UPPER_TAIL_H),)
    _assert_upper_tail(out, hx, UPPER_TAIL_H)


# --------------------------------------------------------------- lower tail where H or F underflows


@pytest.mark.step("A5")
def test_exponential_log_cdf_where_cumulative_hazard_underflows():
    """spec 4.1, rev 25: lower tail "wherever `log H` is representable, even where `H` or `F` underflows".

    ``r = t = 1e-30`` (float32) or ``1e-200`` (float64): ``H = r t`` underflows
    (``1e-60``, ``1e-400``), but ``log F = log H - H/2 + ... = log r + log t``
    (``-138.2``, ``-921.0``) at the dtype-rounded inputs.
    """
    x = f(1e-200 if X64 else 1e-30)
    out = np.asarray(survjax.Exponential(rate=x).log_cdf(x))
    assert np.isfinite(out)
    assert_close(out, 2.0 * np.log(float(x)))


@pytest.mark.step("A5")
def test_weibull_log_cdf_where_cumulative_hazard_underflows():
    """spec 4.1, rev 25: Weibull `log F = k (log t - log b)` where `H = (t/b)**k` underflows.

    ``k = 2.5``, ``b = 3``, ``t = 3e-150`` (float64; ``H = 1e-375``) or
    ``3e-20`` (float32; ``H = 1e-50``).
    """
    t = f(3e-150 if X64 else 3e-20)
    out = np.asarray(survjax.Weibull(shape=f(WEIBULL_K), scale=f(WEIBULL_B)).log_cdf(t))
    assert np.isfinite(out)
    assert_close(out, WEIBULL_K * (np.log(float(t)) - np.log(WEIBULL_B)))


# --------------------------------------------------------------- values at the log1mexp split


@pytest.mark.step("A5")
@pytest.mark.parametrize("name", list(CASES))
def test_log_cdf_on_both_sides_of_log1mexp_split(name):
    """spec 7.2/12.1, rev 25: values on both sides of `H = log 2` against mpmath (`H` = 0.5, 0.69, 0.70, 1.2)."""
    dist, t, hx = _vector_case(name, SPLIT_H)
    out = dist.log_cdf(t)
    assert_rel_close(out, [mp_log_cdf(h) for h in hx])


# --------------------------------------------------------------- parameter derivatives


def _g(h):
    """``d/dH`` and ``d2/dH2`` of ``log1mexp(-H)``."""
    em1 = np.expm1(h)
    return 1.0 / em1, -np.exp(h) / em1**2


def _check_derivatives(fn, p, d1, d2, h):
    """First and second derivatives, reverse and forward mode; zero atol in the tail."""
    check = (
        (lambda a, e: assert_tail_close(a, e, tail_rtol(h))) if h > 1 else assert_close
    )
    check(jax.grad(fn)(p), d1)
    check(jax.jacfwd(fn)(p), d1)
    check(jax.grad(jax.grad(fn))(p), d2)
    check(jax.jacfwd(jax.jacrev(fn))(p), d2)


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", SPLIT_H + DERIVATIVE_TAIL_H)
def test_exponential_log_cdf_rate_derivatives(h):
    """spec 7.3/12.1, rev 25: `d/dr log F = t/expm1(H)`, `d2/dr2 = -t**2 exp(H)/expm1(H)**2`, `H = r t`."""
    t = f(2.0 * h)
    tt = float(t)
    g1, g2 = _g(0.5 * tt)
    _check_derivatives(
        lambda r: survjax.Exponential(rate=r).log_cdf(t), f(0.5), tt * g1, tt**2 * g2, h
    )


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", SPLIT_H + DERIVATIVE_TAIL_H)
def test_weibull_log_cdf_shape_derivatives(h):
    """spec 7.3/12.1, rev 25: `d/dk log F = g1 H L`, `d2/dk2 = g2 (H L)**2 + g1 H L**2`, `L = log(t/b)`."""
    k, b = WEIBULL_K, WEIBULL_B
    t = f(b * h ** (1.0 / k))
    lz = np.log(float(t) / b)
    hh = float(mp_weibull_cumulative_hazard(k, b, t))
    g1, g2 = _g(hh)
    d1 = g1 * hh * lz
    d2 = g2 * (hh * lz) ** 2 + g1 * hh * lz**2
    _check_derivatives(
        lambda kk: survjax.Weibull(shape=kk, scale=f(b)).log_cdf(t), f(k), d1, d2, h
    )


@pytest.mark.step("A5")
@pytest.mark.parametrize("h", SPLIT_H + DERIVATIVE_TAIL_H)
def test_weibull_log_cdf_scale_derivatives(h):
    """spec 7.3/12.1, rev 25: `d/db log F = -g1 k H/b`, `d2/db2 = g2 (k H/b)**2 + g1 k (k+1) H/b**2`."""
    k, b = WEIBULL_K, WEIBULL_B
    t = f(b * h ** (1.0 / k))
    hh = float(mp_weibull_cumulative_hazard(k, b, t))
    g1, g2 = _g(hh)
    d1 = -g1 * k * hh / b
    d2 = g2 * (k * hh / b) ** 2 + g1 * k * (k + 1.0) * hh / b**2
    _check_derivatives(
        lambda bb: survjax.Weibull(shape=f(k), scale=bb).log_cdf(t), f(b), d1, d2, h
    )


# --------------------------------------------------------------- jax_debug_nans on mixed vectors


@pytest.mark.step("A5")
def test_exponential_log_cdf_mixed_times_under_debug_nans():
    """spec 4.1/12.1, rev 25: `t = (0, 2, 66, +inf)` with `r = 0.5` (`H = 0, 1, 33, inf`) raises no NaN under `jax_debug_nans`.

    Expected ``(-inf, log1mexp(-1), log1mexp(-33), 0)``.
    """
    with jax.debug_nans(True):
        out = np.asarray(
            survjax.Exponential(rate=f(0.5)).log_cdf(f([0.0, 2.0, 66.0, INF]))
        )
    np.testing.assert_array_equal(out[[0, 3]], [-INF, 0.0])
    assert_close(out[1], log1mexp(-1.0))
    assert_tail_close(out[2], mp_log_cdf(33.0), tail_rtol(33.0))


@pytest.mark.step("A5")
@pytest.mark.parametrize("k", [0.5, 1.0, 2.5])
def test_weibull_log_cdf_mixed_times_under_debug_nans(k):
    """spec 4.1/12.1, rev 25: Weibull `b = 3`, `t = (0, 3, 3*33**(1/k), +inf)` (`H ~ 0, 1, 33, inf`) under `jax_debug_nans`."""
    b = 3.0
    t = f([0.0, b, b * 33.0 ** (1.0 / k), INF])
    with jax.debug_nans(True):
        out = np.asarray(survjax.Weibull(shape=f(k), scale=f(b)).log_cdf(t))
    np.testing.assert_array_equal(out[[0, 3]], [-INF, 0.0])
    assert_close(out[1], log1mexp(-1.0))
    hx = mp_weibull_cumulative_hazard(k, b, t[2])
    assert_tail_close(out[2], mp_log_cdf(hx), tail_rtol(33.0))
