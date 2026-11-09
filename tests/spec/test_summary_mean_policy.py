"""Feature `summary.mean`: mean shapes, statuses and transformations (spec 8.1, 8.3, 8.5, 8.7, 8.8, 12.1; rev 51, 52, 53).

8.8: "Section 8.7's ... value policy and shape rules remain in force". 8.7:
"NaN wherever `parameter_issues()` reports a violation". The mean takes no
argument, so its shape is the distribution's `batch_shape`. 8.3: "A
non-finite integrand at any node is a failure status, never dropped." 8.5:
status per broadcast entry; counts "record quadrature levels, solver steps
and bracket steps used per entry; unused counts are zero". Rev 52 (8.3,
8.5): one adjacent pair of summed combined estimates converges; one evaluated
level alone cannot; `num_levels` counts rounds per entry, not split
intervals; `error_estimate` is the last difference, NaN with fewer than two
estimates or a non-finite integrand. Rev 53 (8.3): a converged entry returns
`Q_l` of the converging pair; a failure status (`non_finite_arithmetic` or
`work_exhausted`) returns NaN.

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
    """spec 8.3, rev 51, 53: the mean integrates past `cut`, where `log S` is NaN (never dropped); a failure status "returns NaN"."""
    assert np.isnan(np.asarray(fam.nan_beyond()(rate=f(1.0), cut=f(1.0)).mean()))


@MEAN_INFO
def test_non_finite_integrand_in_the_mean_is_a_failure_status():
    """spec 8.3/8.5, rev 51, 53: the same mean reports `non_finite_arithmetic`, and its value is NaN (a failure status "returns NaN")."""
    values, info = fam.nan_beyond()(rate=f(1.0), cut=f(1.0)).mean_with_info()
    assert np.isnan(np.asarray(values))
    assert_status(info, "non_finite_arithmetic")


@MEAN_INFO
def test_non_finite_integrand_in_the_mean_has_a_nan_error_estimate():
    """spec 8.5, rev 52: `error_estimate` "is NaN if ... a non-finite integrand ... makes the comparison meaningless"."""
    _, info = fam.nan_beyond()(rate=f(1.0), cut=f(1.0)).mean_with_info()
    assert np.isnan(np.asarray(info.error_estimate))


@MEAN_INFO
def test_quadrature_work_exhaustion_of_the_mean_is_reported():
    """spec 8.3/8.5, rev 52, 53: with `max_levels=1` only level 0 is evaluated and "one evaluated level alone cannot establish convergence": `work_exhausted`, and a failure status "returns NaN"."""
    opts = survjax.SummaryOptions(rtol=f(1e-30), atol=f(1e-30), max_levels=1)
    values, info = fam.const_hazard()(rate=f(0.5)).mean_with_info(options=opts)
    assert np.isnan(np.asarray(values))
    assert_status(info, "work_exhausted")


@MEAN_INFO
def test_single_level_mean_reports_one_round_and_a_nan_error_estimate():
    """spec 8.5, rev 52: with `max_levels=1` the tanh-sinh and exp-sinh pieces share one round (`num_levels == 1`), and the error estimate is NaN because "fewer than two estimates exist"."""
    opts = survjax.SummaryOptions(rtol=f(1e-30), atol=f(1e-30), max_levels=1)
    _, info = fam.const_hazard()(rate=f(0.5)).mean_with_info(options=opts)
    np.testing.assert_array_equal(np.asarray(info.num_levels), 1)
    assert np.isnan(np.asarray(info.error_estimate))


@MEAN_INFO
def test_mean_work_exhaustion_counts_rounds_not_split_intervals():
    """spec 8.3/8.5, rev 52, 53: tolerances no difference can meet exhaust all `max_levels=3` rounds, so the value is NaN (a failure status "returns NaN"); `[0, s]` and `[s, +inf)` share each round, so `num_levels == 3` (not 6); the error estimate is the last difference, finite and positive ("including on work exhaustion")."""
    opts = survjax.SummaryOptions(rtol=f(1e-30), atol=f(1e-30), max_levels=3)
    values, info = fam.const_hazard()(rate=f(0.5)).mean_with_info(options=opts)
    assert np.isnan(np.asarray(values))
    assert_status(info, "work_exhausted")
    np.testing.assert_array_equal(np.asarray(info.num_levels), 3)
    err = float(np.asarray(info.error_estimate))
    # Not converged, so the last difference exceeds atol + rtol*abs(Q_l) > 0.
    assert np.isfinite(err) and err > 0


@MEAN_INFO
def test_one_adjacent_pair_suffices_for_mean_convergence():
    """spec 8.3/8.5, rev 52: "No second successful pair is required": with `atol` far above any possible level difference, levels 0 and 1 converge, so `num_levels == 2` and the error estimate is within `atol`."""
    # The true mean is 1/rate = 2; with S in (0, 1] and positive weights the summed level-0 and
    # level-1 estimates are O(1), so abs(Q_1 - Q_0) is far below atol = 1e3.
    opts = survjax.SummaryOptions(rtol=f(1e-30), atol=f(1e3), max_levels=8)
    _, info = fam.const_hazard()(rate=f(0.5)).mean_with_info(options=opts)
    assert_status(info, "converged")
    np.testing.assert_array_equal(np.asarray(info.num_levels), 2)
    err = float(np.asarray(info.error_estimate))
    assert 0 <= err <= 1e3


@MEAN_INFO
def test_generic_mean_counts():
    """spec 8.3/8.5, rev 52: generic mean uses between 2 and `max_levels` (default 8) rounds and no solver or bracket steps; converged error estimates are finite and nonnegative."""
    _, info = fam.const_hazard()(rate=f([0.5, 2.0])).mean_with_info()
    assert_status(info, "converged")
    levels = np.asarray(info.num_levels)
    assert ((levels >= 2) & (levels <= 8)).all()
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
