"""observation.truncation: hand-derived truncated contributions for every censoring kind (spec 2, 3.1, 3.2, 4.4, 5, 7.1; rev 51).

Spec 2: the likelihood conditions on inclusion ``L < T <= U``, with
``Z = P(L<T<=U)``:

=========  ======================================
exact t    ``f(t)/Z``, requiring ``L<t<=U``
right c    ``P(max(c,L)<T<=U)/Z``
left c     ``P(L<T<=min(c,U))/Z``
interval   ``P(max(a,L)<T<=min(b,U))/Z``
=========  ======================================

"Partial overlap with the truncation window is allowed." With
``P(lo<T<=hi) = S(lo) - S(hi)``, ``S(0) = 1``, ``S(+inf) = 0``, every row is

``log P(lo<T<=hi) = -H(lo) + log(-expm1(-(H(hi) - H(lo))))`` (``H(+inf) = +inf``
gives ``-H(lo)``), and ``log Z`` is the same expression at ``(L, U]``.

So a censored row is ``g(H(lo), H(hi)) - g(H(L), H(U))`` with
``g(x, y) = -x + log(-expm1(-(y - x)))``, ``lo = max(lower, L)``,
``hi = min(upper, U)``, and an exact row is ``log f(t) - g(H(L), H(U))``.
Exponential ``H = r t``, ``log f = log r - r t``; Weibull ``H = (t/b)**k``,
``log f = log k - log b + (k-1) log(t/b) - H``. Computed in float64 NumPy at
the dtype-rounded inputs; ``test_hand_values_agree_with_mpmath`` checks the
helper against ``observation_reference``.

Tests whose rows are only exact or right-censored need ``observation.truncation``;
tests with left or interval rows also need ``observation.censoring``.
"""

import numpy as np
import observation_reference as ref
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, b, f, flag_data, ll, ph

import survjax

TRUNC = pytest.mark.feature("observation.truncation")
BOTH = pytest.mark.feature("observation.censoring", "observation.truncation")

WINDOWS = {
    "left": (0.5, INF),
    "right": (0.0, 3.0),
    "two-sided": (0.5, 3.0),
}


def g(x, y):
    """``log(exp(-x) - exp(-y))`` for ``x <= y`` (``y`` may be ``inf``)."""
    with np.errstate(over="ignore"):
        return -x + np.log(-np.expm1(-(y - x)))


def hand(family, params, kind, t, t2, lt, rt):
    """Hand-derived contribution (module docstring) at dtype-rounded inputs."""
    params = ref.rounded(*params)
    t, t2, lt, rt = ref.rounded(t, NAN if t2 is None else t2, lt, rt)
    if family == "exponential":
        (r,) = params

        def big_h(x):
            return r * x

        def log_f(x):
            return np.log(r) - r * x
    else:
        k, s = params

        def big_h(x):
            return (x / s) ** k

        def log_f(x):
            return np.log(k) - np.log(s) + (k - 1.0) * np.log(x / s) - (x / s) ** k

    log_z = g(big_h(lt), big_h(rt))
    if kind == "exact":
        return log_f(t) - log_z
    lower, upper = {"right": (t, INF), "left": (0.0, t), "interval": (t, t2)}[kind]
    lo, hi = max(lower, lt), min(upper, rt)
    return g(big_h(lo), big_h(hi)) - log_z


# Rows valid in every window of WINDOWS: (kind, time, time2).
EXACT_RIGHT = [("exact", 1.0, None), ("right", 1.0, None), ("exact", 2.5, None), ("right", 2.0, None)]
CENSORED = [
    ("left", 2.0, None),
    ("interval", 1.0, 2.0),  # inside the window
    ("interval", 0.2, 1.5),  # partial overlap at L
    ("interval", 1.0, 4.0),  # partial overlap at U
    ("left", 1.0, None),
]


def _data(rows, lt, rt):
    kinds = [k for k, _, _ in rows]
    time = [t for _, t, _ in rows]
    time2 = [NAN if t2 is None else t2 for _, _, t2 in rows]
    n = len(rows)
    return flag_data(kinds, time, time2, left_trunc=[lt] * n, right_trunc=[rt] * n)


def _expected(family, params, rows, lt, rt):
    return [hand(family, params, k, t, t2, lt, rt) for k, t, t2 in rows]


FAMILIES = {
    "exponential": (lambda: survjax.Exponential(rate=f(0.5)), ("exponential", (0.5,))),
    "weibull": (lambda: survjax.Weibull(shape=f(1.5), scale=f(2.0)), ("weibull", (1.5, 2.0))),
    "weibull-k<1": (lambda: survjax.Weibull(shape=f(0.6), scale=f(1.5)), ("weibull", (0.6, 1.5))),
}


# ----------------------------------------------------------------- reference check


@TRUNC
def test_hand_values_agree_with_mpmath():
    """spec 2, rev 51: the hand helper equals the 50-digit mpmath evaluation of the spec 2 table (no survjax call)."""
    for family, params in (("exponential", (0.5,)), ("weibull", (1.5, 2.0))):
        for lt, rt in WINDOWS.values():
            for kind, t, t2 in EXACT_RIGHT + CENSORED:
                lower, upper = {"exact": (t, t), "right": (t, INF), "left": (0.0, t), "interval": (t, t2)}[kind]
                want = ref.contribution(family, params, lower, upper, exact=kind == "exact", left=lt, right=rt)
                np.testing.assert_allclose(hand(family, params, kind, t, t2, lt, rt), want, rtol=1e-12)


# ----------------------------------------------------------------- exact and right rows


@TRUNC
@pytest.mark.parametrize("window", list(WINDOWS))
@pytest.mark.parametrize("family", list(FAMILIES))
def test_truncated_exact_and_right_rows_are_hand_derived(family, window):
    """spec 2 and 7.1, rev 51: exact `f(t)/Z` and right `P(max(c,L)<T<=U)/Z` under left, right and two-sided truncation."""
    make, (fname, params) = FAMILIES[family]
    lt, rt = WINDOWS[window]
    out = ll(make(), _data(EXACT_RIGHT, lt, rt))
    assert_close(out, _expected(fname, params, EXACT_RIGHT, lt, rt))


@TRUNC
def test_exponential_left_truncation_is_memoryless_by_hand():
    """spec 2, rev 51: Exponential `r=0.5`, `L=1`: exact `t=3` gives `log r - r (t - L)`; right `c=2` gives `-r (c - L)`.

    ``log f(t) - log S(L) = log r - r t + r L``; ``log S(c) - log S(L) = -r (c - L)``.
    """
    data = survjax.SurvivalData(f([3.0, 2.0]), b([True, False]), left_trunc=f([1.0, 1.0]))
    assert_close(ll(survjax.Exponential(rate=f(0.5)), data), [np.log(0.5) - 1.0, -0.5])


@TRUNC
def test_right_censored_beyond_entry_with_right_truncation_by_hand():
    """spec 2, rev 51: Exponential `r=1`, window `(0, 2]`, right at `c=1`: `log(e**-1 - e**-2) - log(1 - e**-2)`."""
    data = survjax.SurvivalData(f([1.0]), b([False]), right_trunc=f([2.0]))
    expected = np.log(np.exp(-1.0) - np.exp(-2.0)) - np.log1p(-np.exp(-2.0))
    assert_close(ll(survjax.Exponential(rate=f(1.0)), data), [expected])


@TRUNC
def test_per_row_truncation_windows():
    """spec 1 and 2, rev 51: "absent, left, right or two-sided truncation may vary by individual"."""
    rows = [("exact", 1.0, None, 0.0, INF), ("exact", 1.0, None, 0.5, INF), ("right", 1.0, None, 0.0, 3.0), ("right", 1.0, None, 0.5, 3.0)]
    data = survjax.SurvivalData(
        f([r[1] for r in rows]),
        b([r[0] == "exact" for r in rows]),
        left_trunc=f([r[3] for r in rows]),
        right_trunc=f([r[4] for r in rows]),
    )
    expected = [hand("weibull", (1.5, 2.0), k, t, t2, lt, rt) for k, t, t2, lt, rt in rows]
    assert_close(ll(survjax.Weibull(shape=f(1.5), scale=f(2.0)), data), expected)


@TRUNC
def test_bounds_form_with_truncation_equals_flag_form():
    """spec 3.1 and 12.1, rev 51: equivalent flag and bounds data with truncation give identical contributions."""
    lt, rt = WINDOWS["two-sided"]
    flags = _data(EXACT_RIGHT, lt, rt)
    bounds = survjax.SurvivalData.from_bounds(
        f([1.0, 1.0, 2.5, 2.0]),
        f([1.0, INF, 2.5, INF]),
        exact=b([True, False, True, False]),
        left_trunc=f([lt] * 4),
        right_trunc=f([rt] * 4),
    )
    dist = survjax.Weibull(shape=f(1.5), scale=f(2.0))
    np.testing.assert_array_equal(ll(dist, bounds), ll(dist, flags))


@TRUNC
def test_scalar_truncation_broadcasts_to_rows():
    """spec 3.2, rev 51: a scalar `left_trunc` broadcasts to the observation shape and gives the per-row result."""
    rows = EXACT_RIGHT
    scalar = survjax.SurvivalData(f([t for _, t, _ in rows]), b([k == "exact" for k, _, _ in rows]), left_trunc=f(0.5))
    per_row = _data(rows, 0.5, INF)
    dist = survjax.Exponential(rate=f(0.5))
    assert_close(ll(dist, scalar), ll(dist, per_row))


@TRUNC
def test_draw_column_against_truncated_observations():
    """spec 5, rev 51: data `(N,)` against a `(D, 1)` distribution: `(D, N)` hand values."""
    lt, rt = WINDOWS["two-sided"]
    out = ll(survjax.Exponential(rate=f([[0.5], [2.0]])), _data(EXACT_RIGHT, lt, rt))
    assert out.shape == (2, 4)
    assert_close(out, [_expected("exponential", (r,), EXACT_RIGHT, lt, rt) for r in (0.5, 2.0)])


# ------------------------------------------------------- every censoring kind


@BOTH
@pytest.mark.parametrize("window", list(WINDOWS))
@pytest.mark.parametrize("family", list(FAMILIES))
def test_truncated_left_and_interval_rows_are_hand_derived(family, window):
    """spec 2 and 7.1, rev 51: left and interval rows, including partial overlaps at `L` and at `U`, under every truncation kind."""
    make, (fname, params) = FAMILIES[family]
    lt, rt = WINDOWS[window]
    out = ll(make(), _data(CENSORED, lt, rt))
    assert_close(out, _expected(fname, params, CENSORED, lt, rt))


@BOTH
@pytest.mark.parametrize("window", list(WINDOWS))
def test_full_mixed_matrix_in_one_batch(window):
    """spec 1, 2 and 12.1, rev 51: "the full likelihood matrix": all four kinds mixed in one truncated batch (Weibull)."""
    lt, rt = WINDOWS[window]
    rows = EXACT_RIGHT + CENSORED
    out = ll(survjax.Weibull(shape=f(1.5), scale=f(2.0)), _data(rows, lt, rt))
    assert_close(out, _expected("weibull", (1.5, 2.0), rows, lt, rt))


@BOTH
def test_mixed_kinds_and_mixed_windows_in_one_batch():
    """spec 1, rev 51: "All four observation types may be mixed ... and absent, left, right or two-sided truncation may vary by individual"."""
    rows = [
        ("exact", 1.0, None, 0.0, INF),
        ("right", 1.0, None, 0.5, INF),
        ("left", 2.0, None, 0.0, 3.0),
        ("interval", 0.2, 4.0, 0.5, 3.0),
        ("interval", 1.0, 2.0, 0.0, INF),
        ("left", 2.0, None, 0.5, INF),
    ]
    data = flag_data(
        [r[0] for r in rows],
        [r[1] for r in rows],
        [NAN if r[2] is None else r[2] for r in rows],
        left_trunc=[r[3] for r in rows],
        right_trunc=[r[4] for r in rows],
    )
    expected = [hand("weibull", (1.5, 2.0), *r) for r in rows]
    # Row 3, (0.2, 4] containing (0.5, 3], covers the window: exactly 0 (spec 2).
    out = ll(survjax.Weibull(shape=f(1.5), scale=f(2.0)), data)
    assert out[3] == 0.0
    assert_close(out, expected)


@BOTH
def test_custom_family_on_generic_defaults_matches_weibull():
    """spec 4.2, rev 51: `power_hazard` (only `log_survival`) uses the generic `log_conditional_interval_prob` and matches Weibull."""
    lt, rt = WINDOWS["two-sided"]
    rows = EXACT_RIGHT + CENSORED
    dist = fam.power_hazard()(shape=f(1.5), scale=f(2.0))
    assert_close(ll(dist, _data(rows, lt, rt)), _expected("weibull", (1.5, 2.0), rows, lt, rt))


@BOTH
def test_improper_family_with_right_truncation():
    """spec 2 and 4.2, rev 51: `mixture_cure` (`c=0.3`, `r=0.5`) under `(0.5, 3]`: left at `2` is `log((S(0.5)-S(2))/(S(0.5)-S(3)))`.

    ``S(t) = c + (1-c) e**(-r t)``, so ``S(a) - S(b) = (1-c)(e**(-r a) - e**(-r b))``
    and the cure fraction cancels: ``g(r 0.5, r 2) - g(r 0.5, r 3)``. Right at
    ``1`` with ``U = +inf``: ``log(S(1)/S(0.5))``, which keeps the cure mass
    (sets closed at ``+inf`` include ``T = +inf``).
    """
    cure, r = ref.rounded(0.3, 0.5)
    dist = fam.mixture_cure()(cure=f(0.3), rate=f(0.5))
    data = flag_data(["left", "right"], [2.0, 1.0], left_trunc=[0.5, 0.5], right_trunc=[3.0, INF])

    def s(t):
        return cure + (1 - cure) * np.exp(-r * t)

    expected = [g(r * 0.5, r * 2.0) - g(r * 0.5, r * 3.0), np.log(s(1.0)) - np.log(s(0.5))]
    assert_close(ll(dist, data), expected)


# ----------------------------------------------------------------- PH wrapper (4.4)


def _ph_equivalent():
    k, b0, eta = 1.5, 2.0, 0.4
    wrapped = ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta))
    equivalent = survjax.Weibull(shape=f(k), scale=f(b0 * np.exp(-eta / k)))
    return wrapped, equivalent


@TRUNC
def test_ph_weibull_truncated_likelihood_equals_the_equivalent_weibull():
    """spec 4.4, rev 51: PH-Weibull's likelihood equals `Weibull(k, b0*exp(-eta/k))` on truncated exact/right rows."""
    wrapped, equivalent = _ph_equivalent()
    data = _data(EXACT_RIGHT, *WINDOWS["two-sided"])
    assert_close(ll(wrapped, data), ll(equivalent, data))


@BOTH
def test_ph_weibull_full_matrix_equals_the_equivalent_weibull():
    """spec 4.4, rev 51: the same identity on left/interval rows under truncation, and untruncated."""
    wrapped, equivalent = _ph_equivalent()
    for window in ((0.0, INF), WINDOWS["two-sided"]):
        data = _data(EXACT_RIGHT + CENSORED, *window)
        assert_close(ll(wrapped, data), ll(equivalent, data), err_msg=str(window))
