"""A11: `ProportionalHazards` likelihood (spec 2, 4.4, 5, 6.1, 7.1, 12.1, 12.2; rev 41).

Exact and right-censored rows without truncation, through
``dist.log_likelihood`` (data and raw forms). 4.4: "The direct methods and the
likelihood keep the PH formulas": an exact row contributes
``log f = log h0(t) + eta + exp(eta) log S0(t)``, a right-censored row
``log S = exp(eta) log S0(t)``. Weibull ``(k, b0)``, ``z = t/b0``:
``log h0 = log k - log b0 + (k-1) log z``, ``log S0 = -z**k``; Exponential
``log h0 = log r``, ``log S0 = -r t``. Unsupported modes return NaN on their
rows (7.1, 12.2).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    INF,
    MISUSE,
    NAN,
    X64,
    assert_close,
    b,
    f,
    ll,
    nan_causes,
    ph,
    unsupported_nan,
)

import survjax

EXACT, RIGHT = True, False


def weibull_contribution(k, b0, eta, t, exact):
    z = t / b0
    log_s = -np.exp(eta) * z**k
    if not exact:
        return log_s
    return np.log(k) - np.log(b0) + (k - 1.0) * np.log(z) + eta + log_s


def exponential_contribution(r, eta, t, exact):
    log_s = -r * np.exp(eta) * t
    return np.log(r) + eta + log_s if exact else log_s


def _ph_weibull(k=1.5, b0=2.0, eta=0.3):
    return ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta))


# --------------------------------------------------------------- values


@pytest.mark.step("A11")
def test_ph_weibull_mixed_rows_are_hand_derived():
    """spec 4.4/7.1, rev 41: exact rows give `log h0 + eta + exp(eta) log S0`, right-censored rows `exp(eta) log S0`."""
    t = [0.5, 1.0, 2.0, 4.0]
    ev = [EXACT, RIGHT, EXACT, RIGHT]
    k, b0, eta = 1.5, 2.0, 0.3
    data = survjax.SurvivalData(f(t), b(ev))
    expected = [weibull_contribution(k, b0, eta, ti, ei) for ti, ei in zip(t, ev, strict=True)]
    assert_close(ll(_ph_weibull(k, b0, eta), data), expected)


@pytest.mark.step("A11")
def test_ph_exponential_mixed_rows_are_hand_derived():
    """spec 4.4/7.1, rev 41: PH-Exponential contributions `log r + eta - r exp(eta) t` and `-r exp(eta) t`."""
    t = [0.5, 1.0, 3.0]
    ev = [EXACT, RIGHT, EXACT]
    r, eta = 0.7, -0.6
    data = survjax.SurvivalData(f(t), b(ev))
    d = ph(survjax.Exponential(rate=f(r)), f(eta))
    expected = [exponential_contribution(r, eta, ti, ei) for ti, ei in zip(t, ev, strict=True)]
    assert_close(ll(d, data), expected)


@pytest.mark.step("A11")
def test_per_row_predictor_pairs_with_rows():
    """spec 5, rev 41: "data `(N,)`, distribution `(N,)` | `(N,)`, paired"; `eta` per individual, shared baseline."""
    t = np.array([0.5, 1.0, 2.0])
    ev = [EXACT, RIGHT, EXACT]
    eta = np.array([-1.0, 0.0, 1.0])
    data = survjax.SurvivalData(f(t), b(ev))
    expected = [weibull_contribution(1.5, 2.0, e, ti, ei) for e, ti, ei in zip(eta, t, ev, strict=True)]
    assert_close(ll(_ph_weibull(eta=eta), data), expected)


@pytest.mark.step("A11")
def test_custom_baseline_likelihood_is_hand_derived():
    """spec 4.2/4.4, rev 41: a custom baseline (Weibull law, generic defaults) gives the PH-Weibull contributions."""
    t = [0.5, 2.0]
    ev = [EXACT, RIGHT]
    d = ph(fam.power_hazard()(shape=f(1.5), scale=f(2.0)), f(0.3))
    expected = [weibull_contribution(1.5, 2.0, 0.3, ti, ei) for ti, ei in zip(t, ev, strict=True)]
    assert_close(ll(d, survjax.SurvivalData(f(t), b(ev))), expected)


@pytest.mark.step("A11")
def test_right_censored_at_zero_contributes_exactly_zero():
    """spec 2/7.1, rev 41: right-censored at `c=L=0` covers the whole window: "exactly zero"."""
    data = survjax.SurvivalData(f([0.0, 1.0]), b([RIGHT, EXACT]))
    out = ll(_ph_weibull(), data)
    assert out[0] == 0.0
    assert_close(out[1], weibull_contribution(1.5, 2.0, 0.3, 1.0, True))


@pytest.mark.step("A11")
def test_overflowing_contribution_is_minus_infinity():
    """spec 7.1, rev 41: a valid row whose log contribution is below the finite range is `-inf`, not NaN.

    Right-censored ``log S = -r exp(eta) t``: ``exp(700) * 1e10`` exceeds the
    float64 range and ``exp(80) * 1e10`` (about 5.5e44) the float32 range.
    """
    eta = 700.0 if X64 else 80.0
    data = survjax.SurvivalData(f([1e10, 1.0]), b([RIGHT, RIGHT]))
    out = ll(ph(survjax.Exponential(rate=f(1.0)), f([eta, 0.0])), data)
    assert out[0] == -INF
    assert_close(out[1], -1.0)


# --------------------------------------------------------------- shapes


@pytest.mark.step("A11")
def test_scalar_wrapper_against_observations():
    """spec 5, rev 41: "data `(N,)`, scalar distribution | `(N,)`"."""
    data = survjax.SurvivalData(f([0.5, 1.0, 2.0]), b([EXACT, RIGHT, EXACT]))
    assert np.shape(ll(_ph_weibull(), data)) == (3,)


@pytest.mark.step("A11")
def test_draw_column_against_observations():
    """spec 5, rev 41: "data `(N,)`, distribution `(D,1)` | `(D,N)`"; `eta` of shape `(D, 1)`."""
    t = np.array([0.5, 1.0, 2.0])
    ev = [EXACT, RIGHT, EXACT]
    eta = np.array([[-0.5], [0.5]])
    out = ll(_ph_weibull(eta=eta), survjax.SurvivalData(f(t), b(ev)))
    assert out.shape == (2, 3)
    expected = [[weibull_contribution(1.5, 2.0, e[0], ti, ei) for ti, ei in zip(t, ev, strict=True)] for e in eta]
    assert_close(out, expected)


@pytest.mark.step("A11")
def test_draw_by_arm_batch_against_paired_arms():
    """spec 4.4/5, rev 41: baseline `(S, 1)`, `eta` `(S, 2)`, data `(2,)` gives `(S, 2)`."""
    k = np.array([[1.5], [0.8]])
    b0 = np.array([[2.0], [1.0]])
    eta = np.array([[0.0, 0.5], [-0.5, 1.0]])
    t, ev = np.array([1.2, 2.5]), [EXACT, RIGHT]
    d = ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta))
    out = ll(d, survjax.SurvivalData(f(t), b(ev)))
    assert out.shape == (2, 2)
    expected = [
        [weibull_contribution(k[s, 0], b0[s, 0], eta[s, j], t[j], ev[j]) for j in range(2)] for s in range(2)
    ]
    assert_close(out, expected)


@pytest.mark.step("A11")
def test_mismatched_batch_and_data_lengths_are_rejected():
    """spec 5, rev 41: a bare batch `(D,)` against data `(N,)` with `D != N` raises `TypeError`."""
    data = survjax.SurvivalData(f([0.5, 1.0, 2.0]), b([EXACT, RIGHT, EXACT]))
    with pytest.raises(MISUSE):
        _ph_weibull(eta=[0.0, 1.0]).log_likelihood(data)


# --------------------------------------------------------------- entry-point forms and transformations


@pytest.mark.step("A11")
def test_raw_form_equals_data_form_in_values_and_derivatives():
    """spec 6.1/12.1, rev 41: `log_likelihood(time=..., event=...)` equals the `SurvivalData` form in values, first and second derivatives."""
    t, ev = f([0.5, 1.0, 2.0]), b([EXACT, RIGHT, EXACT])
    data = survjax.SurvivalData(t, ev)

    def by_data(x):
        return jnp.sum(ph(survjax.Weibull(shape=x[0], scale=x[1]), x[2]).log_likelihood(data))

    def by_raw(x):
        return jnp.sum(ph(survjax.Weibull(shape=x[0], scale=x[1]), x[2]).log_likelihood(time=t, event=ev))

    x = f([1.5, 2.0, 0.3])
    assert_close(by_raw(x), by_data(x))
    assert_close(jax.grad(by_raw)(x), jax.grad(by_data)(x))
    assert_close(jax.hessian(by_raw)(x), jax.hessian(by_data)(x))


@pytest.mark.step("A11")
def test_likelihood_agrees_under_jit():
    """spec 12.1, rev 41: eager and `jit` agree; wrapper and data as traced arguments."""
    data = survjax.SurvivalData(f([0.5, 1.0, 2.0]), b([EXACT, RIGHT, EXACT]))
    d = _ph_weibull(eta=[0.1, -0.2, 0.4])
    assert_close(jax.jit(lambda dd, da: dd.log_likelihood(da))(d, data), ll(d, data))


@pytest.mark.step("A11")
def test_likelihood_agrees_under_vmap_over_predictor_draws():
    """spec 12.1, rev 41: `vmap` over `eta` draws equals the explicit `(D, 1)` batch."""
    data = survjax.SurvivalData(f([0.5, 1.0, 2.0]), b([EXACT, RIGHT, EXACT]))
    eta = f([-0.5, 0.0, 0.7])

    def one(e):
        return ph(survjax.Weibull(shape=f(1.5), scale=f(2.0)), e).log_likelihood(data)

    assert_close(jax.vmap(one)(eta), ll(_ph_weibull(eta=np.asarray(eta)[:, None]), data))


@pytest.mark.step("A11")
def test_traced_construction_and_likelihood_inside_jit():
    """spec 4.1/12.1, rev 41: wrapper construction from traced parameters and the likelihood compose inside `jit`."""
    t, ev = [0.5, 2.0], [EXACT, RIGHT]

    @jax.jit
    def fn(k, b0, eta):
        data = survjax.SurvivalData(jnp.asarray(f(t)), jnp.asarray(b(ev)))
        return ph(survjax.Weibull(shape=k, scale=b0), eta).log_likelihood(data)

    expected = [weibull_contribution(1.5, 2.0, 0.3, ti, ei) for ti, ei in zip(t, ev, strict=True)]
    assert_close(fn(f(1.5), f(2.0), f(0.3)), expected)


# --------------------------------------------------------------- NaN rows


@pytest.mark.step("A11")
def test_invalid_data_rows_are_nan_and_others_kept():
    """spec 7.1, rev 41: every row in `data.issues.invalid_rows` yields NaN; zero and negative exact times are invalid (3.3)."""
    data = survjax.SurvivalData(f([1.0, 0.0, -1.0, 2.0]), b([EXACT, EXACT, EXACT, RIGHT]))
    out = ll(_ph_weibull(), data)
    assert np.isnan(out[1:3]).all()
    assert_close(out[[0, 3]], [weibull_contribution(1.5, 2.0, 0.3, 1.0, True), weibull_contribution(1.5, 2.0, 0.3, 2.0, False)])
    bad_data, _, other = nan_causes(_ph_weibull(), data, out)
    np.testing.assert_array_equal(bad_data, [False, True, True, False])
    assert not other.any()


@pytest.mark.step("A11")
@pytest.mark.parametrize("bad", [NAN, INF, -INF], ids=["nan", "posinf", "neginf"])
def test_non_finite_predictor_row_is_nan(bad):
    """spec 4.4/7.1, rev 41: rows marked by `invalid_entries` are NaN; `eta` declares `real` (finite)."""
    data = survjax.SurvivalData(f([1.0, 1.0, 2.0]), b([EXACT, EXACT, RIGHT]))
    d = _ph_weibull(eta=[0.3, bad, 0.3])
    out = ll(d, data)
    assert np.isnan(out[1])
    assert_close(out[[0, 2]], [weibull_contribution(1.5, 2.0, 0.3, 1.0, True), weibull_contribution(1.5, 2.0, 0.3, 2.0, False)])
    _, bad_params, other = nan_causes(d, data, out)
    np.testing.assert_array_equal(bad_params, [False, True, False])
    assert not other.any()


@pytest.mark.step("A11")
@pytest.mark.parametrize("bad", [0.0, -1.0, NAN, INF], ids=["zero", "negative", "nan", "posinf"])
def test_invalid_baseline_row_is_nan(bad):
    """spec 4.7/7.1, rev 41: nested baseline issues (`baseline.rate`) select NaN rows of the wrapper's likelihood."""
    data = survjax.SurvivalData(f([1.0, 1.0, 2.0]), b([EXACT, RIGHT, EXACT]))
    d = ph(survjax.Exponential(rate=f([0.5, bad, 2.0])), f(0.2))
    out = ll(d, data)
    assert np.isnan(out[1])
    assert_close(out[[0, 2]], [exponential_contribution(0.5, 0.2, 1.0, True), exponential_contribution(2.0, 0.2, 2.0, True)])


@pytest.mark.step("A11")
def test_custom_baseline_condition_row_is_nan():
    """spec 4.7/7.1, rev 41: a nested named-condition violation (`baseline.rate_above_cap`) makes its row NaN.

    ``capped_rate`` has ``log S0 = -r t``: rows 0 and 2 are ``log r + eta - r e**eta t``.
    """
    data = survjax.SurvivalData(f([1.0, 1.0, 2.0]), b([EXACT, EXACT, EXACT]))
    d = ph(fam.capped_rate()(rate=f([0.5, 3.0, 1.0]), cap=f(2.0)), f(0.2))
    out = ll(d, data)
    assert np.isnan(out[1])
    assert_close(out[[0, 2]], [exponential_contribution(0.5, 0.2, 1.0, True), exponential_contribution(1.0, 0.2, 2.0, True)])


# --------------------------------------------------------------- unsupported modes


def _unsupported_cases():
    return {
        "left-censored": survjax.SurvivalData(f([1.0, 3.0]), left_censored=b([False, True])),
        "interval-censored": survjax.SurvivalData(
            f([1.0, 1.0]), interval_censored=b([False, True]), time2=f([1.0, 3.0])
        ),
        "left-truncated": survjax.SurvivalData(f([1.0, 2.0]), b([EXACT, EXACT]), left_trunc=f([0.0, 1.0])),
        "right-truncated": survjax.SurvivalData(f([1.0, 2.0]), b([EXACT, RIGHT]), right_trunc=f([INF, 5.0])),
    }


@pytest.mark.step("A11")
@pytest.mark.parametrize("case", list(_unsupported_cases()))
def test_unsupported_mode_row_is_nan_and_supported_row_kept(case):
    """spec 7.1/12.2, rev 41: an unimplemented mode gives NaN on its row, eagerly and under `jit`; the supported row keeps its value."""
    data = _unsupported_cases()[case]
    d = _ph_weibull()
    want0 = weibull_contribution(1.5, 2.0, 0.3, 1.0, True)
    for out in (ll(d, data), np.asarray(jax.jit(lambda dd, da: dd.log_likelihood(da))(d, data))):
        assert np.isnan(out[1])
        assert_close(out[0], want0)
    out = ll(d, data)
    np.testing.assert_array_equal(unsupported_nan(d, data, out), [False, True])
    assert not np.asarray(data.issues.invalid_rows).any()
    assert not np.asarray(d.parameter_issues().invalid_entries).any()


@pytest.mark.step("A11")
def test_unsupported_row_leaves_supported_row_derivatives():
    """spec 7.1, rev 41: supported rows retain "their parameter derivatives"; `d/deta` of the exact row is `1 - w`."""
    data = _unsupported_cases()["left-censored"]
    k, b0, t = 1.5, 2.0, 1.0

    def row0(eta):
        return ph(survjax.Weibull(shape=f(k), scale=f(b0)), eta).log_likelihood(data)[0]

    eta = 0.3
    assert_close(jax.grad(row0)(f(eta)), 1.0 - np.exp(eta) * (t / b0) ** k)
