"""Feature `summary.rmst`: generic RMST inputs, shapes, statuses and transformations (spec 4.5, 8.3, 8.5, 8.7, 8.8, 12.1; rev 51).

8.8: "Section 8.7's tail accuracy, value policy and shape rules remain in
force"; each step is gated by the 12.1 Summaries row "including the section
4.5 input policy". 8.7 value policy: boolean, complex, string and object `tau`
raise `TypeError`; "NaN for negative, NaN or infinite `tau`; `0` at `tau=0`";
"NaN wherever `parameter_issues()` reports a violation". 8.7 shapes: `tau`
broadcasts against `batch_shape`; a non-broadcasting `tau` raises `TypeError`.
8.3: "A non-finite integrand at any node is a failure status, never dropped."
8.5: work limits are static; "convergence and status are per broadcast
entry"; counts "record quadrature levels, solver steps and bracket steps used
per entry; unused counts are zero"; "Error estimates are in time units".

Status for invalid `tau` and for parameter violations is only required not
to be `converged` (Q-057).
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    INF,
    NAN,
    STATUS_CODES,
    X64,
    assert_close,
    assert_status,
    f,
    status,
)
from summary_cases import FAMILIES, GENERIC, RMST, RMST_INFO, VIOLATIONS, draws_by_arms, families, violations

import survjax

# --------------------------------------------------------------- generic tau policy


@pytest.mark.parametrize("family", GENERIC)
@pytest.mark.parametrize("tau", [-1.0, NAN, INF, -INF], ids=["negative", "nan", "posinf", "neginf"])
@RMST
def test_generic_invalid_cutoff_gives_nan(family, tau):
    """spec 4.5/8.7, rev 51: the 8.7 value policy on the generic path: NaN for negative, NaN or infinite `tau`."""
    assert np.isnan(np.asarray(families()[family]().rmst(f(tau))))


@pytest.mark.parametrize("family", GENERIC)
@pytest.mark.parametrize("tau", [-1.0, NAN, INF], ids=["negative", "nan", "posinf"])
@RMST_INFO
def test_generic_invalid_cutoff_is_not_reported_converged(family, tau):
    """spec 8.5, rev 51: an invalid-`tau` NaN entry is not reported `converged` (Q-057)."""
    values, info = families()[family]().rmst_with_info(f(tau))
    assert np.isnan(np.asarray(values))
    assert int(status(info)) != STATUS_CODES["converged"]


@pytest.mark.parametrize("family", GENERIC)
@RMST
def test_generic_rmst_at_zero_is_zero(family):
    """spec 4.5/8.7, rev 51: `rmst(0) = 0` on the generic path."""
    assert np.asarray(families()[family]().rmst(f(0.0))) == 0.0


@pytest.mark.parametrize("family", GENERIC)
@pytest.mark.parametrize(
    "bad",
    [np.array(True), np.array(1.0 + 0j), np.array("1.0"), np.array(1.0, dtype=object)],
    ids=["bool", "complex", "string", "object"],
)
@RMST
def test_generic_cutoff_with_a_non_numeric_dtype_is_rejected(family, bad):
    """spec 4.5/8.7, rev 51: boolean, complex, string and object `tau` raise `TypeError` on the generic path."""
    with pytest.raises(TypeError):
        families()[family]().rmst(bad)


@pytest.mark.parametrize("family", GENERIC)
@RMST
def test_generic_boolean_cutoff_is_rejected_during_tracing(family):
    """spec 4.5/8.7, rev 51: the dtype check also applies to a traced `tau`."""
    d = families()[family]()
    with pytest.raises(TypeError):
        jax.jit(lambda t: d.rmst(t))(np.asarray(True))


@pytest.mark.skipif(X64, reason="the float64-to-float32 conversion exists only with 64-bit mode disabled")
@pytest.mark.parametrize("family", GENERIC)
@RMST
def test_generic_float64_cutoff_converts_without_warning_in_32_bit_mode(family):
    """spec 4.5/8.7, rev 51: a float64 `tau` with 64-bit mode disabled converts to float32 without a warning."""
    d = families()[family]()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = d.rmst(np.asarray([0.5, 2.0], dtype=np.float64))
    assert jnp.result_type(got) == jnp.float32
    assert_close(got, d.rmst(f([0.5, 2.0])))


# --------------------------------------------------------------- shapes


@pytest.mark.parametrize("family", GENERIC)
@RMST
def test_generic_cutoff_grid_with_explicit_axes(family):
    """spec 8.7, rev 51: `tau_grid[:, None, None]` against a `(S, 2)` batch gives `(m, S, 2)`, entry `[i]` equal to the scalar-`tau` result."""
    d = draws_by_arms(family)
    grid = f([0.5, 1.0, 5.0])
    got = d.rmst(grid[:, None, None])
    assert np.shape(got) == (3, 3, 2)
    for i, t in enumerate(grid):
        assert_close(np.asarray(got)[i], d.rmst(f(t)))


@pytest.mark.parametrize("family", FAMILIES)
@RMST_INFO
def test_cutoff_grid_with_info_against_draws_by_arms(family):
    """spec 8.5/8.7, rev 51: `tau_grid[:, None, None]` gives `(m, S, 2)` values and info on every family."""
    values, info = draws_by_arms(family).rmst_with_info(f([0.5, 1.0, 5.0])[:, None, None])
    assert np.shape(values) == (3, 3, 2)
    for name in ("status", "error_estimate", "num_levels"):
        assert np.shape(getattr(info, name)) == (3, 3, 2), name


@pytest.mark.parametrize("family", GENERIC)
@RMST
def test_generic_cutoff_that_does_not_broadcast_is_rejected(family):
    """spec 4.1/8.7, rev 51: a non-broadcasting `tau` raises `TypeError` on generic paths."""
    with pytest.raises(TypeError):
        draws_by_arms(family).rmst(f([0.5, 1.0, 5.0]))


# --------------------------------------------------------------- parameter violations


@pytest.mark.parametrize("case", VIOLATIONS)
@RMST
def test_parameter_violations_give_nan_rmst(case):
    """spec 8.7, rev 51: "NaN wherever `parameter_issues()` reports a violation"; valid entries stay finite (all families, generic paths included)."""
    d = violations()[case]
    np.testing.assert_array_equal(np.asarray(d.parameter_issues().invalid_entries), [False, True, False])
    values = np.asarray(d.rmst(f(1.5)))
    assert np.isnan(values[1])
    assert np.isfinite(values[[0, 2]]).all()


@pytest.mark.parametrize("case", VIOLATIONS)
@RMST_INFO
def test_parameter_violations_are_not_reported_converged(case):
    """spec 8.5/8.7, rev 51: the violating entry is NaN and not `converged`; the valid entries converge (Q-057: `invalid_input` expected, not asserted)."""
    values, info = violations()[case].rmst_with_info(f(1.5))
    codes = status(info)
    assert np.isnan(np.asarray(values)[1])
    assert codes[1] != STATUS_CODES["converged"]
    assert (codes[[0, 2]] == STATUS_CODES["converged"]).all()


@RMST_INFO
def test_parameter_violation_status_under_jit():
    """spec 8.5/8.7, rev 51: the same values and status under `jit`."""
    d = violations()["custom-condition"]
    v0, i0 = d.rmst_with_info(f(1.5))
    v1, i1 = jax.jit(lambda dd: dd.rmst_with_info(f(1.5)))(d)
    assert_close(v1, v0)
    np.testing.assert_array_equal(status(i1), status(i0))


# --------------------------------------------------------------- failure statuses


@RMST
def test_non_finite_integrand_gives_nan_rmst():
    """spec 8.3, rev 51: "A non-finite integrand at any node is a failure status, never dropped": the entry integrating past `cut` is NaN; the others are `-expm1(-t)`."""
    values = np.asarray(fam.nan_beyond()(rate=f(1.0), cut=f(1.0)).rmst(f([0.5, 2.0, 0.9])))
    assert np.isnan(values[1])
    assert_close(values[[0, 2]], [-np.expm1(-0.5), -np.expm1(-0.9)])


@RMST_INFO
def test_non_finite_integrand_is_a_failure_status():
    """spec 8.3/8.5, rev 51: the failing entry reports `non_finite_arithmetic`; the others converge with nonzero level counts (status per entry)."""
    _, info = fam.nan_beyond()(rate=f(1.0), cut=f(1.0)).rmst_with_info(f([0.5, 2.0, 0.9]))
    assert_status(info, ["converged", "non_finite_arithmetic", "converged"])
    assert (np.asarray(info.num_levels)[[0, 2]] > 0).all()


@RMST_INFO
def test_quadrature_work_exhaustion_is_reported():
    """spec 8.3/8.5, rev 51: with one level and tolerances no two levels can meet, the RMST is NaN with `work_exhausted`."""
    opts = survjax.SummaryOptions(rtol=f(1e-30), atol=f(1e-30), max_levels=1)
    values, info = fam.const_hazard()(rate=f(0.5)).rmst_with_info(f(3.0), options=opts)
    assert np.isnan(np.asarray(values))
    assert_status(info, "work_exhausted")


# --------------------------------------------------------------- counts and error estimates


@RMST_INFO
def test_generic_rmst_counts():
    """spec 8.5, rev 51: generic RMST uses quadrature levels and no solver or bracket steps ("unused counts are zero"); converged error estimates are finite and nonnegative."""
    _, info = fam.const_hazard()(rate=f([0.5, 2.0])).rmst_with_info(f(1.5))
    assert_status(info, "converged")
    assert (np.asarray(info.num_levels) > 0).all()
    assert (np.asarray(info.num_solver_steps) == 0).all()
    assert (np.asarray(info.num_bracket_steps) == 0).all()
    err = np.asarray(info.error_estimate)
    assert np.isfinite(err).all() and (err >= 0).all()


@RMST_INFO
def test_error_estimate_is_in_time_units():
    """spec 8.5, rev 51: "Error estimates are in time units": under time scales `c` the converged estimate stays within ten times the stopping tolerance `atol + rtol*|rmst|` (Q-061)."""
    for c in (1e-6, 1.0, 1e6):
        d = fam.power_hazard()(shape=f(2.5), scale=f(2.0 * c))
        values, info = d.rmst_with_info(f(3.0 * c))
        err = float(np.asarray(info.error_estimate))
        value = float(np.asarray(values))
        rtol, atol_factor = (1e-8, 1e-10) if X64 else (1e-4, 1e-6)
        # 8.3/8.5: convergence requires successive levels within atol + rtol*|estimate|, atol = factor * s, s = scale.
        assert 0 <= err <= 10 * (atol_factor * 2.0 * c + rtol * abs(value)), c


# --------------------------------------------------------------- transformations


@pytest.mark.parametrize("family", FAMILIES)
@RMST
def test_rmst_agrees_under_jit_and_vmap(family):
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of RMST values on every family."""
    d = draws_by_arms(family)
    want = d.rmst(f(1.5))
    assert_close(jax.jit(lambda dd: dd.rmst(f(1.5)))(d), want)
    assert_close(jax.vmap(lambda dd: dd.rmst(f(1.5)))(d), want)


@pytest.mark.parametrize("family", FAMILIES)
@RMST_INFO
def test_rmst_with_info_agrees_under_jit_and_vmap(family):
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of values and status."""
    d = draws_by_arms(family)
    call = lambda dd: dd.rmst_with_info(f(1.5))  # noqa: E731
    v0, i0 = call(d)
    for v1, i1 in (jax.jit(call)(d), jax.vmap(call)(d)):
        assert_close(v1, v0)
        np.testing.assert_array_equal(status(i1), status(i0))


@RMST_INFO
def test_options_cross_jit_into_a_generic_rmst():
    """spec 8.5, rev 51: options passed into `jit`: tolerance values reuse one trace, a work limit retraces."""
    d = fam.const_hazard()(rate=f(0.5))
    traces = []

    @jax.jit
    def run(dd, opts):
        traces.append(None)
        return dd.rmst(f(2.0), options=opts)

    a = run(d, survjax.SummaryOptions(rtol=f(1e-4), atol=f(1e-6)))
    b = run(d, survjax.SummaryOptions(rtol=f(1e-5), atol=f(1e-7)))
    assert len(traces) == 1
    run(d, survjax.SummaryOptions(rtol=f(1e-5), atol=f(1e-7), max_levels=5))
    assert len(traces) == 2
    want = -np.expm1(-1.0) / 0.5
    np.testing.assert_allclose(np.asarray([a, b], dtype=np.float64), want, rtol=1e-3)
