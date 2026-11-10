"""Feature `summary.rmst`: the generic RMST tail cut (spec 8.3, 8.4, 8.5; rev 58).

8.3 (rev 58): "Cut the upper range where `log S` falls below the floor
`log(finfo(dtype).tiny)` (the smallest positive normal number; backends may
flush subnormals). The cut is needed only where `log S(tau)` is below the
floor; otherwise there is no search. The cut point `c` is found by the section
8.2 iteration on `log S(c) = floor` (`H(c) = -floor`) in the known bracket
`[min(tau, s), tau]`, so no bracket expansion occurs. Any iterate with
`floor - 1 <= log S(c) <= floor` is accepted; where `log S` reaches `-inf`
before the floor (a finite support end), the cut is that end. Its steps count
against `max_solver_steps` and are reported in `num_solver_steps` (zero where
no search runs); an entry whose search exhausts them has status
`work_exhausted` and a NaN value. `c` enters the integral through
`stop_gradient`: the omitted term `S(c)·dc` is below `tiny`, so the section
8.4 tangents are unchanged." 8.3 also records the motivating case: "one
tanh-sinh rule on `[0, tau]` gave a 38% gradient error for Weibull `k=20`,
`tau=1000*b`; splitting at `b` and cutting the tail gave ~1e-15."

Only custom families (``spec_families``) are used, so every value comes from
the generic path, never a built-in closed form. Where `c` lies is not
asserted: the spec fixes only the acceptance window, and the omitted mass
beyond any accepted `c` is below `tiny` times the remaining interval, far
below every tolerance here. Values and derivatives therefore equal the full
`∫_0^tau S` and its derivatives (mpmath, ``rmst_reference``,
``summary_reference``), compared at the 12.1 generic scale
``generic_rtol()`` (``10*rtol``, rev 54) in the 12.1 log units.

Light-tail cases (``log S(tau)`` far below the floor in both dtypes, so a
search runs): Weibull-like ``k=20``, ``tau=1000*b`` (``H(tau)=1e60``; in
float32 ``log S`` overflows to ``-inf`` near ``t = 84 b``, beyond the floor
point ``t ~ 1.25 b``); exponential-like ``r=0.5``, ``tau=1e4`` (``H(tau) =
5000``); Gompertz-like ``a=0.5``, ``g=1``, ``tau=50`` (``H(tau) ~ 2.6e21``).

Not forced: ``work_exhausted`` from the cut search. The spec fixes neither the
search's starting point nor how an accepted start is counted, so no static
``max_solver_steps`` provably exhausts it (Q-072). The tight-limit tests
assert what holds either way and check exhaustion when it occurs.
"""

import math

import jax
import mpmath
import numpy as np
import pytest
import rmst_reference as rref
import spec_families as fam
import summary_reference as sref
from helpers import (
    FLOAT,
    assert_generic_close,
    assert_generic_log_derivative,
    f,
    generic_rtol,
    rounded,
    status,
)
from summary_cases import HESSIANS, MODES, RMST, RMST_INFO, assert_hessian, entry_sum_grad, grad

import survjax

# Spec 8.3 (rev 58): the floor of the active dtype (float32 about -87.34, float64 about -708.40).
FLOOR = float(np.log(np.finfo(FLOAT).tiny))
CONVERGED = 0
WORK_EXHAUSTED = 5


def _power(k, b):
    return fam.power_hazard()(shape=k, scale=b)


def _gompertz(a, g):
    return fam.gompertz_like()(a=a, g=g)


# (rmst as a function of (*params, tau), params, tau)
LIGHT = {
    "power-k20": (lambda k, b, t: _power(k, b).rmst(t), (20.0, 2.0), 2000.0),
    "exponential": (lambda r, t: fam.const_hazard()(rate=r).rmst(t), (0.5,), 1e4),
    "gompertz": (lambda a, g, t: _gompertz(a, g).rmst(t), (0.5, 1.0), 50.0),
}


def _reference(case, params, tau):
    """``(value, [d/dtheta ...], d/dtau)`` at the dtype-rounded inputs, by mpmath."""
    if case == "power-k20":
        value, dk, db, dtau = rref.weibull(*params, tau)
        return value, [dk, db], dtau
    if case == "exponential":
        value, dr, dtau = rref.exponential(*params, tau)
        return value, [dr], dtau
    a, g = params
    value = sref.gompertz_rmst(a, g, tau)
    da = sref.derivative(sref.gompertz_rmst_mp, (a, g, tau), (1, 0, 0))
    dg = sref.derivative(sref.gompertz_rmst_mp, (a, g, tau), (0, 1, 0))
    with mpmath.workdps(40):
        # d/dtau = S(tau) = exp(-a expm1(g tau)) (8.4), about exp(-1.3e21).
        dtau = float(mpmath.exp(-mpmath.mpf(a) * mpmath.expm1(mpmath.mpf(g) * mpmath.mpf(tau))))
    return value, [da, dg], dtau


def _light(case):
    fn, params, tau = LIGHT[case]
    *params, tau = rounded(*params, tau)
    return fn, params, tau


def _assert_far_below_floor(case):
    """Premise: ``log S(tau) = -H(tau)`` is far below the floor of the active dtype (hand values above)."""
    _, params, tau = _light(case)
    if case == "power-k20":
        h = (tau / params[1]) ** params[0]
    elif case == "exponential":
        h = params[0] * tau
    else:
        h = params[0] * math.expm1(params[1] * tau)
    assert -h < FLOOR - 100, (case, h, FLOOR)


# =============================================================== light tails: values and derivatives


@pytest.mark.parametrize("case", list(LIGHT))
@RMST
def test_light_tail_rmst_matches_mpmath(case):
    """spec 8.3, rev 58: with `log S(tau)` far below `log(finfo(dtype).tiny)` the cut tail RMST equals `∫_0^tau S dt` (mpmath) at the generic scale."""
    _assert_far_below_floor(case)
    fn, params, tau = _light(case)
    got = fn(*(f(p) for p in params), f(tau))
    assert_generic_close(got, _reference(case, params, tau)[0], err_msg=case)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(LIGHT))
@RMST
def test_light_tail_rmst_first_derivatives_match_mpmath(case, mode):
    """spec 8.3/8.4, rev 58: "`c` enters the integral through `stop_gradient` ... the section 8.4 tangents are unchanged": parameter and `tau` derivatives against mpmath, forward and reverse."""
    fn, params, tau = _light(case)
    value, dparams, dtau = _reference(case, params, tau)
    n = len(params)
    grads = grad(fn, tuple(range(n + 1)), mode)(*(f(p) for p in params), f(tau))
    for i in range(n):
        assert_generic_log_derivative(grads[i], dparams[i], params[i], value, f"{case} d/dtheta[{i}]")
    # 8.4: d/dtau = S(tau), here 0 to within exp(-5000); 12.1: the tau derivative uses absolute scale 1.
    np.testing.assert_allclose(float(grads[n]), dtau, rtol=0, atol=generic_rtol(), err_msg=f"{case} d/dtau")


@pytest.mark.parametrize("mode", list(HESSIANS))
@RMST
def test_light_tail_rmst_parameter_hessian(mode):
    """spec 8.3/8.4, rev 58: Weibull-like `k=20`, `tau=1000*b`: the `(k, b)` Hessian with the cut ("Forward, reverse and higher-order derivatives all work").

    Reference: ``rmst - b Gamma(1+1/k) = -∫_tau^inf S`` with
    ``S(tau) = exp(-1e60)``; its parameter derivatives are integrals of ``S``
    times polynomials in ``H`` and ``log t``, equally negligible, so the
    Hessian is that of the mean ``b Gamma(1+1/k)`` (mpmath ``diff``).
    ``tau`` is held fixed: in float32 the family's own ``log S`` overflows to
    ``-inf`` at ``tau``, where its parameter tangent is ``0 * inf``; the
    ``tau`` derivative is tested to first order above.
    """
    _, (kk, bb), tt = _light("power-k20")
    value = rref.weibull(kk, bb, tt)[0]
    got = HESSIANS[mode](lambda x: _power(x[0], x[1]).rmst(f(tt)))(f([kk, bb]))
    assert_hessian(
        got, (kk, bb), lambda o: sref.weibull_mean_derivative(kk, bb, o), value, generic_rtol(), "power-k20"
    )


# =============================================================== finite support end before the floor


@pytest.mark.parametrize("tau", [10.0, 1e3, 1e6])
@pytest.mark.parametrize("mode", MODES)
@RMST
def test_support_end_before_the_floor(tau, mode):
    """spec 8.3, rev 58: "where `log S` reaches `-inf` before the floor (a finite support end), the cut is that end": uniform `(0, end]` beyond its end has RMST `end/2`, `d/dend = 1/2`, `d/dtau = S(tau) = 0` (hand-derived).

    ``∫_0^end (1 - t/end) dt = end/2``. The 8.4 tangent with ``c = end`` held
    by ``stop_gradient``: ``∫_0^end dS/dend dt = ∫_0^end t/end**2 dt = 1/2``,
    and the omitted ``S(c) dc`` is ``S(end) dc = 0``. At the last representable
    ``t < end``, ``1 - t/end >= eps/2``, so ``log1p(-t/end)`` stays above about
    ``-37`` (float64) or ``-17`` (float32) and then jumps to ``-inf``: the floor
    is never reached first.
    """
    end, tt = rounded(2.0, tau)
    value = fam.uniform_support()(end=f(end)).rmst(f(tt))
    g_end, g_tau = grad(lambda e, t: fam.uniform_support()(end=e).rmst(t), (0, 1), mode)(f(end), f(tt))
    assert_generic_close(value, end / 2)
    assert_generic_log_derivative(g_end, 0.5, end, end / 2, "d/dend")
    np.testing.assert_allclose(float(g_tau), 0.0, rtol=0, atol=generic_rtol(), err_msg="d/dtau")


@RMST_INFO
def test_support_end_cut_reports_converged():
    """spec 8.3/8.5, rev 58: RMST past a finite support end converges per entry, with `end/2` beyond the end and `tau - tau**2/(2 end)` inside."""
    end = 2.0
    values, info = fam.uniform_support()(end=f(end)).rmst_with_info(f([1e3, 1.0, 1e6]))
    np.testing.assert_array_equal(status(info), CONVERGED)
    assert_generic_close(values, [end / 2, 1.0 - 1.0 / (2 * end), end / 2])


# =============================================================== search accounting


@RMST_INFO
def test_no_cut_search_where_log_survival_at_tau_is_above_the_floor():
    """spec 8.3/8.5, rev 58: "The cut is needed only where `log S(tau)` is below the floor; otherwise there is no search": `num_solver_steps` is zero ("zero where no search runs"), including `log S(tau) = floor + 2`."""
    (tt,) = rounded(-FLOOR - 2.0)
    taus = [0.5, 1.0, 20.0, tt]
    values, info = fam.const_hazard()(rate=f(1.0)).rmst_with_info(f(taus))
    np.testing.assert_array_equal(status(info), CONVERGED)
    np.testing.assert_array_equal(np.asarray(info.num_solver_steps), 0)
    np.testing.assert_array_equal(np.asarray(info.num_bracket_steps), 0)
    assert_generic_close(values, [-math.expm1(-t) for t in taus])


@pytest.mark.parametrize("case", list(LIGHT))
@RMST_INFO
def test_cut_search_steps_are_counted(case):
    """spec 8.3/8.5, rev 58: below the floor a search runs; "Its steps count against `max_solver_steps` and are reported in `num_solver_steps`" (between 1 and the default 64; Q-071); "no bracket expansion occurs" (`num_bracket_steps == 0`); status `converged`."""
    _assert_far_below_floor(case)
    _, params, tau = _light(case)
    dist = {
        "power-k20": lambda: _power(f(params[0]), f(params[1])),
        "exponential": lambda: fam.const_hazard()(rate=f(params[0])),
        "gompertz": lambda: _gompertz(f(params[0]), f(params[1])),
    }[case]()
    values, info = dist.rmst_with_info(f(tau))
    assert int(status(info)) == CONVERGED
    steps = int(np.asarray(info.num_solver_steps))
    assert 1 <= steps <= 64, steps
    assert int(np.asarray(info.num_bracket_steps)) == 0
    assert_generic_close(values, _reference(case, params, tau)[0])


@RMST_INFO
def test_cut_search_counts_are_per_entry():
    """spec 8.3/8.5, rev 58: per broadcast entry, `num_solver_steps` is zero where `log S(tau)` is above the floor and positive where it is below (Q-071); all converge."""
    taus = [1.0, 1e4, 5.0]
    values, info = fam.const_hazard()(rate=f(1.0)).rmst_with_info(f(taus))
    np.testing.assert_array_equal(status(info), CONVERGED)
    steps = np.asarray(info.num_solver_steps)
    assert steps[0] == 0 and steps[2] == 0, steps
    assert 1 <= steps[1] <= 64, steps
    assert_generic_close(values, [-math.expm1(-t) for t in taus])


# =============================================================== tight solver limit (Q-072)

# Gompertz-like a = 0.5, g = 1 (s = 1): entries 0 and 2 have log S(tau) = -0.5 expm1(tau), -0.86 and
# -3.2, above the floor (no search); entry 1 (tau = 50) searches on the bracket [1, 50] for
# H(c) = -floor. Since d log S/dt = -g (H + a), the window floor - 1 <= log S <= floor has width
# about 1/88 in t near 5.17 (float32) and 1/709 near 7.26 (float64).
TIGHT_TAUS = [1.0, 50.0, 2.0]


def _tight():
    return survjax.SummaryOptions(max_solver_steps=1)


def _tight_rmst(a, g, taus=TIGHT_TAUS):
    return _gompertz(a, g).rmst(f(taus), options=_tight())


@RMST_INFO
def test_cut_search_under_a_one_step_limit_is_consistent():
    """spec 8.3/8.5, rev 58: with `max_solver_steps=1`, the searching entry uses at most one step and either converges to the right value or is `work_exhausted` with "a NaN value"; the entries without a search keep zero steps, `converged` and their values."""
    a, g = rounded(0.5, 1.0)
    values, info = _gompertz(f(a), f(g)).rmst_with_info(f(TIGHT_TAUS), options=_tight())
    values, codes, steps = np.asarray(values), status(info), np.asarray(info.num_solver_steps)
    want = [sref.gompertz_rmst(a, g, t) for t in rounded(*TIGHT_TAUS)]
    np.testing.assert_array_equal(codes[[0, 2]], CONVERGED)
    np.testing.assert_array_equal(steps[[0, 2]], 0)
    assert_generic_close(values[[0, 2]], [want[0], want[2]])
    assert steps[1] <= 1, steps
    assert codes[1] in (CONVERGED, WORK_EXHAUSTED), codes
    if codes[1] == WORK_EXHAUSTED:
        assert np.isnan(values[1])
    else:
        assert_generic_close(values[1], want[1])


@pytest.mark.parametrize("mode", MODES)
@RMST_INFO
def test_cut_search_limit_isolates_other_entries(mode):
    """spec 8.4, rev 57/58: under `max_solver_steps=1` the other entries' derivatives in the shared `(a, g)` are finite and equal those of the batch without the searching entry, whatever its status ("Isolation takes precedence")."""
    a, g = f(rounded(0.5, 1.0))
    full = entry_sum_grad(_tight_rmst, (0, 2), (a, g), mode, argnums=(0, 1))
    alone = grad(lambda a_, g_: _tight_rmst(a_, g_, [1.0, 2.0]).sum(), (0, 1), mode)(a, g)
    for got, want, name in zip(full, alone, ("a", "g"), strict=True):
        assert np.isfinite(np.asarray(got)).all(), name
        np.testing.assert_allclose(np.asarray(got, np.float64), np.asarray(want, np.float64), rtol=generic_rtol())


@RMST_INFO
def test_exhausted_cut_search_entry_has_a_nan_tangent():
    """spec 8.3/8.4, rev 58: if the one-step search is `work_exhausted`, that entry's forward tangent is NaN ("Entries that failed ... get NaN tangents"), the others' finite; skipped when the search converged (Q-072)."""
    a, g = f(rounded(0.5, 1.0))
    _, info = _gompertz(a, g).rmst_with_info(f(TIGHT_TAUS), options=_tight())
    if status(info)[1] != WORK_EXHAUSTED:
        pytest.skip("the one-step cut search converged; exhaustion cannot be forced by the spec (Q-072)")
    _, tangent = jax.jvp(lambda a_: _tight_rmst(a_, g), (a,), (f(1.0),))
    tangent = np.asarray(tangent)
    assert np.isnan(tangent[1])
    assert np.isfinite(tangent[[0, 2]]).all()
