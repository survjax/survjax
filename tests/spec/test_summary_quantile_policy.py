"""Feature `summary.quantile`: quantile inputs, shapes, statuses and transformations (spec 4.5, 8.2, 8.5, 8.7, 8.8, 12.1; rev 51).

8.8: "Section 8.7's tail accuracy, value policy and shape rules remain in
force and apply in the same way to `p` for quantiles"; gated by the 12.1
Summaries row "including the section 4.5 input policy and the section 8.1
endpoints". 4.5: "summaries: ... `p` outside `[0,1]`" give NaN; boolean,
complex, string and object arguments raise `TypeError`, also during tracing;
float64 converts to float32 without a warning in 32-bit mode. 8.7 shapes,
applied to `p`: `p` broadcasts against `batch_shape`; a non-broadcasting `p`
raises `TypeError`; `p_grid[:, None, None]` adds axes. 8.2: "Non-convergence
gives NaN and a failure status"; "The bracket starts at `log
reference_time()` and expands geometrically ... up to a static step limit".

Status for invalid `p` and parameter violations is only required not to be
`converged` (Q-057). The input policy of `inverse_cumulative_hazard` for
negative, NaN or non-numeric `e` is not specified (Q-063) and is not tested.
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
from summary_cases import FAMILIES, QUANTILE, QUANTILE_INFO, VIOLATIONS, draws_by_arms, families, violations

import survjax

# --------------------------------------------------------------- p values


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("p", [-0.1, 1.5, NAN, INF, -INF], ids=["negative", "above-one", "nan", "posinf", "neginf"])
@QUANTILE
def test_invalid_probability_gives_nan(family, p):
    """spec 4.5/8.8, rev 51: "`p` outside `[0,1]`" and NaN `p` give NaN, eagerly and under `jit`."""
    d = families()[family]()
    assert np.isnan(np.asarray(d.quantile(f(p))))
    assert np.isnan(np.asarray(jax.jit(lambda dd, pp: dd.quantile(pp))(d, f(p))))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_invalid_probability_is_not_reported_converged(family):
    """spec 8.5, rev 51: a NaN entry from an invalid `p` is not reported `converged`; valid entries are (Q-057)."""
    values, info = families()[family]().quantile_with_info(f([0.3, 1.5, NAN, 0.6]))
    values, codes = np.asarray(values), status(info)
    assert np.isnan(values[[1, 2]]).all()
    assert np.isfinite(values[[0, 3]]).all()
    assert (codes[[1, 2]] != STATUS_CODES["converged"]).all()
    assert (codes[[0, 3]] == STATUS_CODES["converged"]).all()


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_invalid_probability_entries_leave_valid_entries(family):
    """spec 8.5/8.8, rev 51: per-entry policy: valid `p` entries equal their values computed alone."""
    d = families()[family]()
    got = np.asarray(d.quantile(f([0.3, -1.0, 0.6])))
    assert_close(got[[0, 2]], d.quantile(f([0.3, 0.6])))


NON_NUMERIC = pytest.mark.parametrize(
    "bad",
    [np.array(True), np.array(0.5 + 0j), np.array("0.5"), np.array(0.5, dtype=object)],
    ids=["bool", "complex", "string", "object"],
)


@pytest.mark.parametrize("family", FAMILIES)
@NON_NUMERIC
@QUANTILE
def test_probability_with_a_non_numeric_dtype_is_rejected(family, bad):
    """spec 4.5/8.8, rev 51: boolean, complex, string and object `p` raise `TypeError`."""
    with pytest.raises(TypeError):
        families()[family]().quantile(bad)


@pytest.mark.parametrize("family", FAMILIES)
@NON_NUMERIC
@QUANTILE_INFO
def test_probability_with_a_non_numeric_dtype_is_rejected_with_info(family, bad):
    """spec 4.5/8.8, rev 51: the same `TypeError` from `quantile_with_info`."""
    with pytest.raises(TypeError):
        families()[family]().quantile_with_info(bad)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_boolean_probability_is_rejected_during_tracing(family):
    """spec 4.5/8.8, rev 51: the dtype check also applies to a traced `p` (it uses the dtype, not values)."""
    d = families()[family]()
    with pytest.raises(TypeError):
        jax.jit(lambda pp: d.quantile(pp))(jnp.asarray(True))


@pytest.mark.skipif(X64, reason="the float64-to-float32 conversion exists only with 64-bit mode disabled")
@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_float64_probability_converts_without_warning_in_32_bit_mode(family):
    """spec 4.5/8.8, rev 51: a float64 `p` converts to float32 without a warning."""
    d = families()[family]()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = d.quantile(np.asarray([0.3, 0.6], dtype=np.float64))
    assert jnp.result_type(got) == jnp.float32
    assert_close(got, d.quantile(f([0.3, 0.6])))


# --------------------------------------------------------------- shapes


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_scalar_probability_against_draws_by_arms(family):
    """spec 8.7/8.8, rev 51: shape rules applied to `p`: scalar `p` against a `(S, 2)` batch gives `(S, 2)`; so does `median()`."""
    d = draws_by_arms(family)
    assert np.shape(d.quantile(f(0.5))) == (3, 2)
    assert np.shape(d.median()) == (3, 2)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_probability_grid_with_explicit_axes(family):
    """spec 8.7/8.8, rev 51: `p_grid[:, None, None]` gives `(m, S, 2)`, entry `[i]` equal to the scalar-`p` result."""
    d = draws_by_arms(family)
    grid = f([0.1, 0.5, 0.9, 0.99])
    values = d.quantile(grid[:, None, None])
    assert np.shape(values) == (4, 3, 2)
    for i, p in enumerate(grid):
        assert_close(np.asarray(values)[i], d.quantile(f(p)))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_probability_grid_info_has_the_broadcast_shape(family):
    """spec 8.5, rev 51: info fields of `quantile_with_info(p_grid[:, None, None])` have shape `(m, S, 2)`."""
    values, info = draws_by_arms(family).quantile_with_info(f([0.1, 0.5, 0.9, 0.99])[:, None, None])
    assert np.shape(values) == (4, 3, 2)
    assert np.shape(info.status) == (4, 3, 2)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_probability_that_does_not_broadcast_is_rejected(family):
    """spec 4.1/8.7/8.8, rev 51: a `p` of shape `(3,)` against batch `(S, 2)` raises `TypeError`."""
    with pytest.raises(TypeError):
        draws_by_arms(family).quantile(f([0.1, 0.5, 0.9]))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_probability_that_does_not_broadcast_is_rejected_with_info(family):
    """spec 4.1/8.7/8.8, rev 51: the same `TypeError` from `quantile_with_info`."""
    with pytest.raises(TypeError):
        draws_by_arms(family).quantile_with_info(f([0.1, 0.5, 0.9]))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_inverse_cumulative_hazard_grid_with_explicit_axes(family):
    """spec 8.1/8.7, rev 51: `e_grid[:, None, None]` against a `(S, 2)` batch gives `(m, S, 2)`, entry `[i]` equal to the scalar-`e` result (Q-063: broadcasting as for direct methods)."""
    d = draws_by_arms(family)
    grid = f([0.1, 1.0, 5.0])
    values = d.inverse_cumulative_hazard(grid[:, None, None])
    assert np.shape(values) == (3, 3, 2)
    for i, e in enumerate(grid):
        assert_close(np.asarray(values)[i], d.inverse_cumulative_hazard(f(e)))


# --------------------------------------------------------------- parameter violations


@pytest.mark.parametrize("case", VIOLATIONS)
@QUANTILE
def test_parameter_violations_give_nan_quantile(case):
    """spec 8.7/8.8, rev 51: "NaN wherever `parameter_issues()` reports a violation", applied to quantiles and the median; valid entries stay finite."""
    d = violations()[case]
    np.testing.assert_array_equal(np.asarray(d.parameter_issues().invalid_entries), [False, True, False])
    for values in (np.asarray(d.quantile(f(0.4))), np.asarray(d.median())):
        assert np.isnan(values[1])
        assert np.isfinite(values[[0, 2]]).all()


@pytest.mark.parametrize("case", VIOLATIONS)
@QUANTILE_INFO
def test_parameter_violations_are_not_reported_converged(case):
    """spec 8.5/8.7, rev 51: the violating entry is not `converged`; the valid entries converge (Q-057)."""
    values, info = violations()[case].quantile_with_info(f(0.4))
    codes = status(info)
    assert np.isnan(np.asarray(values)[1])
    assert codes[1] != STATUS_CODES["converged"]
    assert (codes[[0, 2]] == STATUS_CODES["converged"]).all()


# --------------------------------------------------------------- solver limits


@QUANTILE_INFO
def test_bracket_limit_failure_is_reported():
    """spec 8.2/8.5, rev 51: with one bracket step from `log reference_time() = 0` the root at `log 1e20` is unreachable: NaN and `failed_bracket` or `work_exhausted` (Q-061)."""
    d = fam.far_reference_log_logistic()(alpha=f(1e20), beta=f(2.0))
    opts = survjax.SummaryOptions(max_bracket_steps=1)
    values, info = d.quantile_with_info(f(0.5), options=opts)
    assert np.isnan(np.asarray(values))
    assert int(status(info)) in (STATUS_CODES["failed_bracket"], STATUS_CODES["work_exhausted"])


@QUANTILE
def test_far_reference_quantile_converges_with_default_limits():
    """spec 8.2/8.5, rev 51: the same quantile, `alpha*(p/(1-p))**(1/beta) = 1e20` at `p = 0.5`, with the default 64 bracket steps."""
    d = fam.far_reference_log_logistic()(alpha=f(1e20), beta=f(2.0))
    np.testing.assert_allclose(np.asarray(d.quantile(f(0.5)), dtype=np.float64), 1e20, rtol=10 * (1e-8 if X64 else 1e-4))


@QUANTILE_INFO
def test_far_reference_quantile_reports_converged_with_default_limits():
    """spec 8.5, rev 51: that quantile reports `converged`."""
    _, info = fam.far_reference_log_logistic()(alpha=f(1e20), beta=f(2.0)).quantile_with_info(f(0.5))
    assert_status(info, "converged")


@QUANTILE_INFO
def test_generic_quantile_counts():
    """spec 8.5, rev 51: the generic quantile uses no quadrature levels and at most the static solver and bracket limits."""
    d = fam.power_hazard()(shape=f([0.7, 2.5]), scale=f(2.0))
    opts = survjax.SummaryOptions(max_solver_steps=40, max_bracket_steps=30)
    _, info = d.quantile_with_info(f(0.9), options=opts)
    assert_status(info, "converged")
    assert (np.asarray(info.num_levels) == 0).all()
    assert (np.asarray(info.num_solver_steps) <= 40).all()
    assert (np.asarray(info.num_bracket_steps) <= 30).all()
    assert (np.asarray(info.num_solver_steps) + np.asarray(info.num_bracket_steps) > 0).all()


# --------------------------------------------------------------- transformations


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_quantile_agrees_under_jit_and_vmap(family):
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of quantile, median and inverse cumulative hazard values."""
    d = draws_by_arms(family)
    for call in (
        lambda dd: dd.quantile(f(0.4)),
        lambda dd: dd.median(),
        lambda dd: dd.inverse_cumulative_hazard(f(0.7)),
    ):
        want = call(d)
        assert_close(jax.jit(call)(d), want)
        assert_close(jax.vmap(call)(d), want)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_quantile_with_info_agrees_under_jit_and_vmap(family):
    """spec 8.8/12.1, rev 51: eager/`jit`/`vmap` agreement of values and status."""
    d = draws_by_arms(family)
    call = lambda dd: dd.quantile_with_info(f(0.4))  # noqa: E731
    v0, i0 = call(d)
    for v1, i1 in (jax.jit(call)(d), jax.vmap(call)(d)):
        assert_close(v1, v0)
        np.testing.assert_array_equal(status(i1), status(i0))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_quantile_agrees_under_vmap_over_probabilities(family):
    """spec 8.8/12.1, rev 51: `vmap` over `p` equals the explicit-axis grid."""
    d = draws_by_arms(family)
    grid = f([0.1, 0.5, 0.9])
    assert_close(jax.vmap(d.quantile)(grid), d.quantile(grid[:, None, None]))
