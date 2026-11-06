"""A11: `ProportionalHazards` direct methods, grid twins and value policy (spec 4.4, 4.5, 5, 7.3, 8.1; rev 41).

Hand derivation. Under PH, ``H(t) = exp(eta) H0(t)`` (4.4), so with
``u = exp(eta) H0(t)``: ``log S = -u``, ``S = exp(-u)``, ``H = u``,
``log F = log1mexp(-u)``, ``log h = log h0(t) + eta``, ``h = exp(log h)`` and
``log f = log h0 + eta + exp(eta) log S0 = log h - u`` (the 4.4 table; rev 43
adds the ``log_hazard`` row ``log h0(t) + eta``, evaluated through the
baseline's ``log_hazard`` so its boundary limits carry through).

- Weibull baseline ``(k, b0)``, ``z = t/b0``: ``H0 = z**k``,
  ``log h0 = log k - log b0 + (k-1) log z``.
- Exponential baseline ``r``: ``H0 = r t``, ``log h0 = log r``.

The custom baseline ``spec_families.power_hazard`` has the Weibull law and
only ``log_survival``/``reference_time``, so its hazard comes from the generic
autodiff defaults (4.2) and the wrapper must give the same PH values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    FLOAT,
    INF,
    MISUSE,
    NAN,
    X64,
    assert_close,
    assert_rel_close,
    assert_tail_close,
    f,
    log1mexp,
    ph,
)

import survjax

requires_x64 = pytest.mark.skipif(not X64, reason="the spec 7.3 examples need float64 range")

METHODS = (
    "log_survival",
    "survival",
    "log_cdf",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)
GRID_METHODS = {m + "_grid": m for m in METHODS}

# Paired interior cases: k<1, k=1, k>1; eta<0, 0, >0.
K = np.array([0.7, 1.0, 2.5])
B0 = np.array([0.5, 2.0, 3.0])
ETA = np.array([-1.5, 0.0, 0.8])
T = np.array([0.3, 1.0, 4.0])
RATE = np.array([0.5, 1.0, 2.0])


def _reference(method, log_h, u):
    """PH values from ``log h`` and ``u = H`` (float64 NumPy)."""
    return {
        "log_survival": -u,
        "survival": np.exp(-u),
        "cumulative_hazard": u,
        "log_cdf": log1mexp(-u),
        "log_hazard": log_h,
        "hazard": np.exp(log_h),
        "log_prob": log_h - u,
    }[method]


def ph_weibull_reference(method, k, b0, eta, t):
    k, b0, eta, t = (np.asarray(x, dtype=np.float64) for x in (k, b0, eta, t))
    z = t / b0
    u = np.exp(eta) * z**k
    log_h = np.log(k) - np.log(b0) + (k - 1.0) * np.log(z) + eta
    return _reference(method, log_h, u)


def ph_exponential_reference(method, r, eta, t):
    r, eta, t = (np.asarray(x, dtype=np.float64) for x in (r, eta, t))
    return _reference(method, np.log(r) + eta + 0.0 * t, r * np.exp(eta) * t)


def _compare(method, actual, expected):
    if method == "log_cdf":  # 12.1: log probabilities with zero absolute tolerance
        assert_rel_close(actual, expected, err_msg=method)
    else:
        assert_close(actual, expected, err_msg=method)


def _ph_weibull(k=K, b0=B0, eta=ETA):
    return ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta))


# --------------------------------------------------------------- interior values


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", METHODS)
def test_ph_weibull_method_is_hand_derived(method):
    """spec 4.4/8.1, rev 41: PH-Weibull direct methods follow `H = exp(eta) (t/b0)**k`."""
    out = getattr(_ph_weibull(), method)(f(T))
    _compare(method, out, ph_weibull_reference(method, K, B0, ETA, T))


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", METHODS)
def test_ph_exponential_method_is_hand_derived(method):
    """spec 4.4/8.1, rev 41: PH-Exponential direct methods follow `H = r exp(eta) t`, `log h = log r + eta`."""
    out = getattr(ph(survjax.Exponential(rate=f(RATE)), f(ETA)), method)(f(T))
    _compare(method, out, ph_exponential_reference(method, RATE, ETA, T))


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", METHODS)
def test_ph_custom_baseline_method_is_hand_derived(method):
    """spec 4.2/4.4, rev 41: "Wrappers take any baseline distribution"; a custom Weibull-law baseline with generic defaults."""
    base = fam.power_hazard()(shape=f(K), scale=f(B0))
    out = getattr(ph(base, f(ETA)), method)(f(T))
    _compare(method, out, ph_weibull_reference(method, K, B0, ETA, T))


@pytest.mark.step("A11")
def test_zero_predictor_leaves_the_baseline_law():
    """spec 4.4, rev 41: `eta = 0` gives `H = H0`; every method equals the baseline's."""
    base = survjax.Weibull(shape=f(K), scale=f(B0))
    d = ph(base, f(0.0))
    for method in METHODS:
        _compare(method, getattr(d, method)(f(T)), getattr(base, method)(f(T)))


@pytest.mark.step("A11")
def test_positive_predictor_means_higher_hazard():
    """spec 4.4, rev 41: "`eta>0` means higher hazard under PH"."""
    base = survjax.Weibull(shape=f(1.5), scale=f(2.0))
    t = f([0.5, 1.0, 3.0])
    low = np.asarray(ph(base, f(-0.5)).hazard(t))
    high = np.asarray(ph(base, f(0.5)).hazard(t))
    assert (high > low).all()
    assert (np.asarray(ph(base, f(0.5)).survival(t)) < np.asarray(ph(base, f(-0.5)).survival(t))).all()


# --------------------------------------------------------------- value policy (4.5)


@pytest.mark.step("A11")
def test_negative_time_policy_values():
    """spec 4.5, rev 41: the base applies `t<0` rows "for every registered family"."""
    d = _ph_weibull()
    t = f([-1.0, -1e-3, -INF])[:, None]
    expected = {
        "log_survival": 0.0,
        "survival": 1.0,
        "log_prob": -INF,
        "log_cdf": -INF,
        "cumulative_hazard": 0.0,
        "log_hazard": -INF,
        "hazard": 0.0,
    }
    for method, value in expected.items():
        np.testing.assert_array_equal(
            np.asarray(getattr(d, method)(t)), np.full((3, 3), value), err_msg=method
        )


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", METHODS)
def test_nan_time_gives_nan(method):
    """spec 4.5, rev 41: "NaN `t` | NaN"."""
    assert np.isnan(np.asarray(getattr(_ph_weibull(), method)(f(NAN)))).all()


@pytest.mark.step("A11")
def test_base_supplied_values_at_zero():
    """spec 4.5, rev 41: at `t=0` the base supplies `log S=0`, `S=1`, `log_cdf=-inf`, `H=0`."""
    for d in (_ph_weibull(), ph(survjax.Exponential(rate=f(RATE)), f(ETA))):
        np.testing.assert_array_equal(np.asarray(d.log_survival(f(0.0))), [0.0] * 3)
        np.testing.assert_array_equal(np.asarray(d.survival(f(0.0))), [1.0] * 3)
        np.testing.assert_array_equal(np.asarray(d.log_cdf(f(0.0))), [-INF] * 3)
        np.testing.assert_array_equal(np.asarray(d.cumulative_hazard(f(0.0))), [0.0] * 3)


@pytest.mark.step("A11")
def test_base_supplied_values_at_infinity():
    """spec 4.5, rev 41: at `t=+inf` (proper baselines) `log S=-inf`, `S=0`, `log_cdf=0`, `H=+inf`, `log_prob=-inf`."""
    for d in (_ph_weibull(), ph(survjax.Exponential(rate=f(RATE)), f(ETA))):
        np.testing.assert_array_equal(np.asarray(d.log_survival(f(INF))), [-INF] * 3)
        np.testing.assert_array_equal(np.asarray(d.survival(f(INF))), [0.0] * 3)
        np.testing.assert_array_equal(np.asarray(d.log_cdf(f(INF))), [0.0] * 3)
        np.testing.assert_array_equal(np.asarray(d.cumulative_hazard(f(INF))), [INF] * 3)
        np.testing.assert_array_equal(np.asarray(d.log_prob(f(INF))), [-INF] * 3)


@pytest.mark.step("A11")
def test_density_limit_at_zero_for_weibull_baseline():
    """spec 4.4/4.5, rev 41: `log f(0) = log h0(0) + eta`: `k=1` gives `-log b0 + eta`; `k>1` `-inf`; `k<1` `+inf`.

    The PH ``log_prob`` formula of 4.4 with the Weibull limits of 4.5, which
    also equals the equivalent ``Weibull(k, b0 exp(-eta/k))`` (4.4 identity).
    """
    d = ph(survjax.Weibull(shape=f([1.0, 2.0, 0.5]), scale=f([2.0, 2.0, 2.0])), f([0.3, 0.3, 0.3]))
    out = np.asarray(d.log_prob(f(0.0)))
    assert_close(out[0], -np.log(2.0) + 0.3)
    assert out[1] == -INF
    assert out[2] == INF


# Weibull hazard limits (4.5): at t=0, k<1 +inf, k=1 1/b0, k>1 0; at t=+inf the
# reverse (k<1 0, k=1 1/b0, k>1 +inf). PH shifts log h by eta (4.4, rev 43).
HAZARD_LIMIT_K = [0.5, 1.0, 2.0]
HAZARD_LIMIT_B0 = 2.0
HAZARD_LIMIT_ETA = 0.3
LOG_HAZARD_AT_ZERO = [INF, -np.log(2.0) + 0.3, -INF]
LOG_HAZARD_AT_INF = [-INF, -np.log(2.0) + 0.3, INF]


def _hazard_limit_wrapper():
    k = f(HAZARD_LIMIT_K)
    return ph(survjax.Weibull(shape=k, scale=f(HAZARD_LIMIT_B0)), f(HAZARD_LIMIT_ETA))


@pytest.mark.step("A11")
@pytest.mark.parametrize(
    ("t", "expected"), [(0.0, LOG_HAZARD_AT_ZERO), (INF, LOG_HAZARD_AT_INF)], ids=["zero", "inf"]
)
def test_ph_weibull_log_hazard_limits_carry_through(t, expected):
    """spec 4.4/4.5, rev 43: "the baseline's boundary limits (section 4.5) carry through"; PH over Weibull `k=1` gives `log(1/b0) + eta` at `t=0`."""
    out = np.asarray(_hazard_limit_wrapper().log_hazard(f(t)))
    assert out[0] == expected[0]
    assert_close(out[1], expected[1])
    assert out[2] == expected[2]


@pytest.mark.step("A11")
@pytest.mark.parametrize(
    ("t", "expected"), [(0.0, LOG_HAZARD_AT_ZERO), (INF, LOG_HAZARD_AT_INF)], ids=["zero", "inf"]
)
def test_ph_weibull_hazard_limits_carry_through(t, expected):
    """spec 4.4/4.5, rev 43: `hazard` limits are `exp` of the `log_hazard` limits: `+inf`, `exp(eta)/b0`, `0`."""
    out = np.asarray(_hazard_limit_wrapper().hazard(f(t)))
    assert_close(out, np.exp(np.asarray(expected)))


@pytest.mark.step("A11")
@pytest.mark.parametrize("t", [0.0, INF], ids=["zero", "inf"])
@pytest.mark.parametrize("method", ["log_hazard", "hazard"])
def test_ph_weibull_hazard_limits_equal_equivalent_weibull(method, t):
    """spec 4.4, rev 43: at the boundaries too, PH-Weibull hazards equal those of `Weibull(shape=k, scale=b0*exp(-eta/k))`."""
    k = np.asarray(HAZARD_LIMIT_K)
    eq = survjax.Weibull(shape=f(k), scale=f(HAZARD_LIMIT_B0 * np.exp(-HAZARD_LIMIT_ETA / k)))
    _compare(method, getattr(_hazard_limit_wrapper(), method)(f(t)), getattr(eq, method)(f(t)))


@pytest.mark.step("A11")
@pytest.mark.parametrize("t", [0.0, INF], ids=["zero", "inf"])
def test_ph_exponential_hazard_at_the_boundaries(t):
    """spec 4.4/4.5, rev 43: PH-Exponential `log h = log r + eta` at `t=0` and `t=+inf`, not NaN."""
    d = ph(survjax.Exponential(rate=f(RATE)), f(ETA))
    assert_close(d.log_hazard(f(t)), np.log(RATE) + ETA)
    assert_close(d.hazard(f(t)), RATE * np.exp(ETA))


# Finite-or-limit base-supplied values whose derivative is 0 (4.5): `S(0)`,
# `log F(0)`, `H(0)`, `log S(0)`, `S(+inf)`, `log F(+inf)` and every `t<0` row.
ZERO_GRADIENT_CASES = [
    ("log_survival", 0.0),
    ("survival", 0.0),
    ("log_cdf", 0.0),
    ("cumulative_hazard", 0.0),
    ("survival", INF),
    ("log_cdf", INF),
    ("log_survival", -1.0),
    ("survival", -1.0),
    ("log_cdf", -1.0),
    ("cumulative_hazard", -1.0),
]


@pytest.mark.step("A11")
@pytest.mark.parametrize(("method", "t"), ZERO_GRADIENT_CASES)
def test_base_supplied_values_have_zero_parameter_gradients(method, t):
    """spec 4.5, rev 41: base-supplied values have parameter gradients "equal to the derivative of the value": `0`, in the baseline parameters and `eta`."""

    def fn(k, b0, eta):
        return getattr(ph(survjax.Weibull(shape=k, scale=b0), eta), method)(f(t))

    grads = jax.grad(fn, argnums=(0, 1, 2))(f(1.5), f(2.0), f(0.4))
    for g in grads:
        np.testing.assert_array_equal(np.asarray(g), 0.0)


# --------------------------------------------------------------- 7.3 extreme-scaling values


@pytest.mark.step("A11")
@requires_x64
@pytest.mark.parametrize(
    ("eta", "t"), [(-720.0, 1e150), (720.0, 1e-150)], ids=["eta-720", "eta+720"]
)
def test_extreme_scaling_weibull_values_stay_accurate(eta, t):
    """spec 7.3/12.1, rev 41: the 7.3 examples (Weibull `k=2`, `b0=1`) are outside the derivative guarantee, but "Value accuracy ... unchanged".

    ``log H = eta + k log t`` (mpmath, 50 digits); ``log S = -exp(log H)``,
    ``log h = log k + (k-1) log t + eta``, ``log f = log h - H``. ``log H``
    carries an absolute error of about ``eps (|eta| + k |log t|)``, which is
    the relative error of ``H``; the tolerance is 16 times that or the 12.1
    ``rtol``, with zero absolute tolerance. (First example: ``log S ≈ -2.032e-13``,
    the eta gradient quoted in 7.3.)
    """
    import mpmath

    k = 2.0
    d = ph(survjax.Weibull(shape=f(k), scale=f(1.0)), f(eta))
    with mpmath.workdps(50):
        log_t = mpmath.log(mpmath.mpf(t))
        log_hh = mpmath.mpf(eta) + k * log_t
        hh = mpmath.exp(log_hh)
        log_haz = mpmath.log(k) + (k - 1) * log_t + mpmath.mpf(eta)
        ref = {
            "log_survival": float(-hh),
            "cumulative_hazard": float(hh),
            "log_hazard": float(log_haz),
            "log_prob": float(log_haz - hh),
        }
    eps = float(np.finfo(np.float64).eps)
    rtol = max(1e-8, 16.0 * eps * (abs(eta) + k * abs(float(log_t)) + 1.0))
    for method, value in ref.items():
        assert_tail_close(getattr(d, method)(f(t)), value, rtol=rtol, err_msg=method)
    if eta < 0:
        assert_tail_close(d.log_survival(f(t)), -2.032e-13, rtol=1e-3)


@pytest.mark.step("A11")
@requires_x64
def test_extreme_scaling_exponential_values_stay_accurate():
    """spec 7.3/12.1, rev 41: Exponential `rate=1e-200`, `eta=500`, `t=1e-200` (a 7.3 example): values stay accurate.

    ``log H = log r + eta + log t ≈ -421``; ``log S = -H``, ``log h = log r + eta``.
    """
    import mpmath

    r, eta, t = 1e-200, 500.0, 1e-200
    d = ph(survjax.Exponential(rate=f(r)), f(eta))
    with mpmath.workdps(50):
        log_r = mpmath.log(mpmath.mpf(r))
        hh = mpmath.exp(log_r + eta + mpmath.log(mpmath.mpf(t)))
        ref = {
            "log_survival": float(-hh),
            "cumulative_hazard": float(hh),
            "log_hazard": float(log_r + eta),
            "log_prob": float(log_r + eta - hh),
        }
    eps = float(np.finfo(np.float64).eps)
    rtol = max(1e-8, 16.0 * eps * (2 * 460.6 + eta))
    for method, value in ref.items():
        assert_tail_close(getattr(d, method)(f(t)), value, rtol=rtol, err_msg=method)


# --------------------------------------------------------------- shapes and grid twins


def _draws_by_arm(s=3):
    k = np.array([0.8, 1.0, 2.0])[:s, None]
    b0 = np.array([1.0, 2.0, 0.5])[:s, None]
    eta = np.array([[0.0, 0.7], [-0.4, 0.3], [1.0, -1.0]])[:s]
    return ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta)), k, b0, eta


@pytest.mark.step("A11")
def test_scalar_time_against_draw_by_arm_batch():
    """spec 4.4/5, rev 41: baseline `(S, 1)`, `eta` `(S, 2)`; a scalar time gives `(S, 2)` hand values."""
    d, k, b0, eta = _draws_by_arm()
    out = d.survival(f(1.5))
    assert np.shape(out) == (3, 2)
    assert_close(out, ph_weibull_reference("survival", k, b0, eta, 1.5))


@pytest.mark.step("A11")
def test_paired_time_vector_against_arms():
    """spec 5, rev 41: a time `(2,)` pairs with the trailing arm axis of batch `(S, 2)`."""
    d, k, b0, eta = _draws_by_arm()
    t = np.array([0.5, 2.0])
    out = d.log_hazard(f(t))
    assert np.shape(out) == (3, 2)
    assert_close(out, ph_weibull_reference("log_hazard", k, b0, eta, t))


@pytest.mark.step("A11")
def test_explicit_time_first_axis():
    """spec 5, rev 41: `t[:, None, None]` against batch `(S, 2)` gives `(K, S, 2)`."""
    d, k, b0, eta = _draws_by_arm()
    t = np.array([0.25, 1.0, 3.0, 8.0])
    out = d.cumulative_hazard(f(t)[:, None, None])
    assert np.shape(out) == (4, 3, 2)
    assert_close(out, ph_weibull_reference("cumulative_hazard", k, b0, eta, t[:, None, None]))


@pytest.mark.step("A11")
def test_time_that_does_not_broadcast_is_rejected():
    """spec 4.1/5, rev 41: a time that does not broadcast against `batch_shape` raises `TypeError`."""
    d, *_ = _draws_by_arm()
    with pytest.raises(MISUSE):
        d.survival(f([1.0, 2.0, 3.0]))


@pytest.mark.step("A11")
@pytest.mark.parametrize("grid", list(GRID_METHODS))
def test_grid_twin_equals_time_first_direct_method(grid):
    """spec 5, rev 41: `m_grid(t)` equals `m(t[:, None, None])` with shape `(K, *batch_shape)`, here `(K, S, 2)`."""
    d, k, b0, eta = _draws_by_arm()
    t = np.array([3.0, 0.5, 0.5, 1.0, 6.0])  # unsorted, with a repeat
    out = getattr(d, grid)(f(t))
    method = GRID_METHODS[grid]
    assert np.shape(out) == (5, 3, 2)
    _compare(method, out, getattr(d, method)(f(t)[:, None, None]))
    expected = ph_weibull_reference(method, k, b0, eta, t[:, None, None])
    if method != "log_cdf":
        _compare(method, out, expected)
        return
    # 4.2: upper-tail `log F` keeps relative precision "wherever `S` is a normal
    # floating-point number". The reference S reaches ~3.2e-43 and ~1.0e-170,
    # below float32's smallest normal, so the relative check is limited to
    # entries whose reference S is normal in the active dtype (all of them in
    # float64). Outside that range only a finite, nonpositive value is required;
    # a backend may keep subnormals or flush to zero.
    s_ref = ph_weibull_reference("survival", k, b0, eta, t[:, None, None])
    normal = s_ref >= np.finfo(FLOAT).tiny
    if X64:
        assert normal.all()
    actual = np.asarray(out).astype(np.float64)
    assert_rel_close(actual[normal], expected[normal], err_msg=method)
    assert np.isfinite(actual[~normal]).all()
    assert (actual[~normal] <= 0.0).all()


@pytest.mark.step("A11")
def test_grid_with_as_many_times_as_entries_is_not_paired():
    """spec 5, rev 41: with `K=N` the grid evaluates every time against every entry; output row `i` belongs to `t[i]`."""
    d = ph(survjax.Exponential(rate=f([0.5, 1.0, 2.0])), f([0.0, 0.5, -0.5]))
    t = np.array([1.0, 2.0, 3.0])
    out = d.log_survival_grid(f(t))
    assert np.shape(out) == (3, 3)
    expected = -np.array([0.5, 1.0, 2.0])[None, :] * np.exp([0.0, 0.5, -0.5])[None, :] * t[:, None]
    assert_close(out, expected)


@pytest.mark.step("A11")
@pytest.mark.parametrize("bad", [np.float32(1.0), np.ones((2, 2))], ids=["scalar", "2d"])
def test_grid_rejects_time_that_is_not_one_dimensional(bad):
    """spec 5, rev 41: "A `t` that is not one-dimensional ... raises `TypeError`"."""
    d, *_ = _draws_by_arm()
    with pytest.raises(MISUSE):
        d.survival_grid(f(bad))


@pytest.mark.step("A11")
def test_grid_from_zero_includes_base_supplied_values():
    """spec 4.5/5, rev 41: a survival grid starting at `t=0` (baseline `k=1`) gives `S(0)=1` and hand values after."""
    d = ph(survjax.Weibull(shape=f(1.0), scale=f(10.0)), f([0.0, -0.5, -1.0]))
    t = np.array([0.0, 1.0, 5.0])
    out = d.survival_grid(f(t))
    assert_close(out, ph_weibull_reference("survival", 1.0, 10.0, np.array([0.0, -0.5, -1.0]), t[:, None]))
    np.testing.assert_array_equal(np.asarray(out)[0], [1.0, 1.0, 1.0])


# --------------------------------------------------------------- transformations


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", METHODS)
def test_methods_agree_under_jit(method):
    """spec 12.1, rev 41: eager and `jit` agree for every direct method."""
    d, *_ = _draws_by_arm()
    t = f([[0.4], [1.0], [2.5]])
    _compare(method, jax.jit(lambda dd, s: getattr(dd, method)(s))(d, t), getattr(d, method)(t))


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", METHODS)
def test_methods_agree_under_vmap(method):
    """spec 12.1, rev 41: `vmap` over paired wrapper entries and times agrees with the paired batched call."""
    d = _ph_weibull()
    t = f(T)
    mapped = jax.vmap(lambda dd, s: getattr(dd, method)(s))(d, t)
    _compare(method, mapped, getattr(d, method)(t))


@pytest.mark.step("A11")
def test_result_dtype_is_the_parameter_dtype():
    """spec 3.3/4.5, rev 41: calculations use the common parameter dtype; `t` follows JAX conversion."""
    out = _ph_weibull().log_prob(np.asarray(T, dtype=np.float32))
    assert np.dtype(jnp.asarray(out).dtype) == FLOAT

