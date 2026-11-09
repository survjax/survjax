"""Feature `summary.mean`: mean shapes, statuses and transformations (spec 8.1, 8.3, 8.5, 8.7, 8.8, 12.1; rev 51).

8.8: "Section 8.7's ... value policy and shape rules remain in force". 8.7:
"NaN wherever `parameter_issues()` reports a violation". The mean takes no
argument, so its shape is the distribution's `batch_shape`. 8.3: "A
non-finite integrand at any node is a failure status, never dropped." 8.5:
status per broadcast entry; counts "record quadrature levels, solver steps
and bracket steps used per entry; unused counts are zero".

Status for parameter violations is only required not to be `converged`
(Q-057).
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import STATUS_CODES, assert_close, assert_status, f, status
from summary_cases import FAMILIES, MEAN, MEAN_INFO, VIOLATIONS, draws_by_arms, violations

import survjax

# --------------------------------------------------------------- shapes


@pytest.mark.parametrize("family", FAMILIES)
@MEAN
def test_mean_has_the_batch_shape(family):
    """spec 8.1/8.7, rev 51: the mean of a `(S, 2)` batch has shape `(S, 2)`."""
    d = draws_by_arms(family)
    assert d.batch_shape == (3, 2)
    assert np.shape(d.mean()) == (3, 2)


@pytest.mark.parametrize("family", FAMILIES)
@MEAN_INFO
def test_mean_info_has_the_batch_shape(family):
    """spec 8.5, rev 51: `mean_with_info` info fields are shaped like the `(S, 2)` values."""
    values, info = draws_by_arms(family).mean_with_info()
    assert np.shape(values) == (3, 2)
    assert np.shape(info.status) == (3, 2)


# --------------------------------------------------------------- parameter violations


@pytest.mark.parametrize("case", VIOLATIONS)
@MEAN
def test_parameter_violations_give_nan_mean(case):
    """spec 8.7/8.8, rev 51: "NaN wherever `parameter_issues()` reports a violation"; valid entries stay finite."""
    d = violations()[case]
    np.testing.assert_array_equal(np.asarray(d.parameter_issues().invalid_entries), [False, True, False])
    values = np.asarray(d.mean())
    assert np.isnan(values[1])
    assert np.isfinite(values[[0, 2]]).all()


@pytest.mark.parametrize("case", VIOLATIONS)
@MEAN_INFO
def test_parameter_violations_are_not_reported_converged(case):
    """spec 8.5/8.7, rev 51: the violating entry is not `converged`; the valid entries converge (Q-057)."""
    values, info = violations()[case].mean_with_info()
    codes = status(info)
    assert np.isnan(np.asarray(values)[1])
    assert codes[1] != STATUS_CODES["converged"]
    assert (codes[[0, 2]] == STATUS_CODES["converged"]).all()


# --------------------------------------------------------------- failure statuses


@MEAN
def test_non_finite_integrand_gives_nan_mean():
    """spec 8.3, rev 51: the mean integrates past `cut`, where `log S` is NaN, so the result is NaN (never dropped)."""
    assert np.isnan(np.asarray(fam.nan_beyond()(rate=f(1.0), cut=f(1.0)).mean()))


@MEAN_INFO
def test_non_finite_integrand_in_the_mean_is_a_failure_status():
    """spec 8.3/8.5, rev 51: the same mean reports `non_finite_arithmetic`."""
    values, info = fam.nan_beyond()(rate=f(1.0), cut=f(1.0)).mean_with_info()
    assert np.isnan(np.asarray(values))
    assert_status(info, "non_finite_arithmetic")


@MEAN_INFO
def test_quadrature_work_exhaustion_of_the_mean_is_reported():
    """spec 8.3/8.5, rev 51: with one level and tolerances no two levels can meet, the generic mean is NaN with `work_exhausted`."""
    opts = survjax.SummaryOptions(rtol=f(1e-30), atol=f(1e-30), max_levels=1)
    values, info = fam.const_hazard()(rate=f(0.5)).mean_with_info(options=opts)
    assert np.isnan(np.asarray(values))
    assert_status(info, "work_exhausted")


@MEAN_INFO
def test_generic_mean_counts():
    """spec 8.5, rev 51: generic mean uses quadrature levels and no solver or bracket steps; converged error estimates are finite and nonnegative."""
    _, info = fam.const_hazard()(rate=f([0.5, 2.0])).mean_with_info()
    assert_status(info, "converged")
    assert (np.asarray(info.num_levels) > 0).all()
    assert (np.asarray(info.num_solver_steps) == 0).all()
    assert (np.asarray(info.num_bracket_steps) == 0).all()
    err = np.asarray(info.error_estimate)
    assert np.isfinite(err).all() and (err >= 0).all()


# --------------------------------------------------------------- transformations


@pytest.mark.parametrize("family", FAMILIES)
@MEAN
def test_mean_agrees_under_jit_and_vmap(family):
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of mean values on every family."""
    d = draws_by_arms(family)
    want = d.mean()
    assert_close(jax.jit(lambda dd: dd.mean())(d), want)
    assert_close(jax.vmap(lambda dd: dd.mean())(d), want)


@pytest.mark.parametrize("family", FAMILIES)
@MEAN_INFO
def test_mean_with_info_agrees_under_jit_and_vmap(family):
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of values and status."""
    d = draws_by_arms(family)
    call = lambda dd: dd.mean_with_info()  # noqa: E731
    v0, i0 = call(d)
    for v1, i1 in (jax.jit(call)(d), jax.vmap(call)(d)):
        assert_close(v1, v0)
        np.testing.assert_array_equal(status(i1), status(i0))
