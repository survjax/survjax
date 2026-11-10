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

Rev 54 (Q-057, Q-061, Q-063 resolved). 8.5: "Invalid arguments (... `p`
outside `[0,1]`, negative or NaN `e`) and entries with parameter issues have
status `invalid_input` and NaN values. Exact endpoint values (... endpoint
quantiles from `support()`) have status `converged`.
`inverse_cumulative_hazard` takes `options` for its solver limits." 8.2:
"`failed_bracket` when the bracket step limit (`max_bracket_steps`) is reached
without bracketing the root, `work_exhausted` when `max_solver_steps` is
reached inside a bracket." 8.1: `inverse_cumulative_hazard(e, *,
options=None)`: "negative or NaN `e` gives NaN; no `_with_info` twin (failures
are NaN ...)". Design d104627 (8.1, within rev 54) adds "the section 4.5 dtype
rule applies to `e` as to `t`" (Q-063 settled), so `e` gets the `p` dtype
tests.

Rev 56 (8.5): "On an improper family, a quantile that is `+inf` because `p >=
1 - S(+inf)` has status `known_infinite`; `quantile(1)=+inf` on a proper family
is an endpoint and `converged`" (Q-057 settled). The improper `p = 1` case is
read as `known_infinite` too (Q-068).
"""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import (
    INF,
    NAN,
    X64,
    assert_close,
    assert_generic_close,
    assert_status,
    f,
    rounded,
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
def test_invalid_probability_reports_invalid_input(family):
    """spec 8.5, rev 54: "`p` outside `[0,1]`" (and NaN `p`) has status `invalid_input` and a NaN value; valid entries converge."""
    values, info = families()[family]().quantile_with_info(f([0.3, 1.5, NAN, 0.6, -0.1, INF]))
    values = np.asarray(values)
    assert np.isnan(values[[1, 2, 4, 5]]).all()
    assert np.isfinite(values[[0, 3]]).all()
    assert_status(info, ["converged", "invalid_input", "invalid_input", "converged", "invalid_input", "invalid_input"])


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_endpoint_quantiles_report_converged(family):
    """spec 8.5, rev 54: "endpoint quantiles from `support()` have status `converged`": `p = 0` and `p = 1` give `0` and `+inf` (default support), `converged`."""
    values, info = families()[family]().quantile_with_info(f([0.0, 1.0]))
    np.testing.assert_array_equal(np.asarray(values), [0.0, INF])
    assert_status(info, "converged")


@QUANTILE_INFO
def test_bounded_support_endpoint_quantiles_report_converged():
    """spec 8.1/8.5, rev 54: uniform `(0, end]`: `quantile(0) = 0` and `quantile(1) = end` from `support()`, both `converged` (Q-059)."""
    values, info = fam.uniform_support()(end=f(2.0)).quantile_with_info(f([0.0, 1.0]))
    values = np.asarray(values)
    assert values[0] == 0.0
    assert_close(values[1], 2.0)
    assert_status(info, "converged")


@QUANTILE_INFO
def test_improper_family_upper_endpoint_quantile_reports_known_infinite():
    """spec 8.5, rev 56: on an improper family "a quantile that is `+inf` because `p >= 1 - S(+inf)` has status `known_infinite`"; only "on a proper family" is `quantile(1)=+inf` an endpoint and `converged` (Q-068: `p = 1` read as covered by the improper rule)."""
    values, info = fam.mixture_cure()(cure=f(0.3), rate=f(1.0)).quantile_with_info(f(1.0))
    assert np.asarray(values) == INF
    assert_status(info, "known_infinite")


@QUANTILE_INFO
def test_improper_family_quantile_beyond_the_cure_level_reports_known_infinite():
    """spec 8.2/8.5, rev 56: `mixture_cure` with `c = 0.3`: interior `p >= 1 - c = 0.7` is `+inf` with status `known_infinite`; `p = 0.5` below the cure level is finite and `converged`, per entry."""
    values, info = fam.mixture_cure()(cure=f(0.3), rate=f(1.0)).quantile_with_info(f([0.5, 0.75, 0.9]))
    values = np.asarray(values)
    assert np.isfinite(values[0])
    assert (values[1:] == INF).all()
    assert_status(info, ["converged", "known_infinite", "known_infinite"])


@QUANTILE_INFO
def test_known_infinite_quantile_is_decided_per_entry_of_the_batch():
    """spec 8.5, rev 56: cure `[0.3, 0]` at `p = 0.9`: the improper entry (`0.9 >= 1 - 0.3`) is `+inf`, `known_infinite`; with `c = 0` the law is the proper exponential (`q = -log(0.1)/r`), `converged`."""
    (r,) = rounded(0.5)
    values, info = fam.mixture_cure()(cure=f([0.3, 0.0]), rate=f([1.0, r])).quantile_with_info(f(0.9))
    values = np.asarray(values)
    assert values[0] == INF
    assert_generic_close(values[1], -math.log(1.0 - float(f(0.9))) / r)
    assert_status(info, ["known_infinite", "converged"])


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
def test_parameter_violations_report_invalid_input(case):
    """spec 8.5, rev 54: "entries with parameter issues have status `invalid_input` and NaN values"; the valid entries converge."""
    values, info = violations()[case].quantile_with_info(f(0.4))
    assert np.isnan(np.asarray(values)[1])
    assert_status(info, ["converged", "invalid_input", "converged"])


# --------------------------------------------------------------- solver limits


@QUANTILE_INFO
def test_bracket_limit_reports_failed_bracket():
    """spec 8.2/8.5, rev 54: with one bracket step from `log reference_time() = 0` the root at `log 1e20` is unreachable: NaN and "`failed_bracket` when the bracket step limit (`max_bracket_steps`) is reached without bracketing the root"; the bracket count is that limit."""
    d = fam.far_reference_log_logistic()(alpha=f(1e20), beta=f(2.0))
    opts = survjax.SummaryOptions(max_bracket_steps=1)
    values, info = d.quantile_with_info(f(0.5), options=opts)
    assert np.isnan(np.asarray(values))
    assert_status(info, "failed_bracket")
    np.testing.assert_array_equal(np.asarray(info.num_bracket_steps), 1)


@QUANTILE_INFO
def test_solver_step_limit_inside_a_bracket_reports_work_exhausted():
    """spec 8.2/8.5, rev 54: "`work_exhausted` when `max_solver_steps` is reached inside a bracket": one solver step with unattainable tolerances; NaN, and the solver count is that limit.

    Custom log-logistic ``alpha = 2``, ``beta = 2`` at ``p = 0.99``: the root
    ``alpha*(p/(1-p))**(1/beta) = 2*sqrt(99) ~ 19.9`` is about 2.3 log-units
    from ``log reference_time() = log 2``, well inside the default 64 bracket
    steps. ``g(u) = log H(e^u) - log e`` is not linear in ``u`` here (unlike
    the Weibull case, which one Newton step solves exactly, 8.2), and
    ``rtol = atol = 1e-30`` cannot be met after one step. (Even if the
    solver's stopping tolerance were not these options: one Newton step from
    ``u = log 2`` moves about 1.3 of the 2.3 log-units, far outside any
    default tolerance.)
    """
    d = fam.softplus_log_logistic()(alpha=f(2.0), beta=f(2.0))
    opts = survjax.SummaryOptions(rtol=f(1e-30), atol=f(1e-30), max_solver_steps=1)
    values, info = d.quantile_with_info(f(0.99), options=opts)
    assert np.isnan(np.asarray(values))
    assert_status(info, "work_exhausted")
    np.testing.assert_array_equal(np.asarray(info.num_solver_steps), 1)


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


# --------------------------------------------------------------- inverse cumulative hazard inputs and options


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("e", [-0.5, NAN, -INF], ids=["negative", "nan", "neginf"])
@QUANTILE
def test_invalid_hazard_level_gives_nan(family, e):
    """spec 8.1, rev 54: `inverse_cumulative_hazard`: "negative or NaN `e` gives NaN", eagerly and under `jit`."""
    d = families()[family]()
    assert np.isnan(np.asarray(d.inverse_cumulative_hazard(f(e))))
    assert np.isnan(np.asarray(jax.jit(lambda dd, ee: dd.inverse_cumulative_hazard(ee))(d, f(e))))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_invalid_hazard_levels_leave_valid_entries(family):
    """spec 8.1/8.5, rev 54: per-entry policy: valid `e` entries beside a negative and a NaN `e` equal their values computed alone."""
    d = families()[family]()
    got = np.asarray(d.inverse_cumulative_hazard(f([0.3, -1.0, NAN, 2.0])))
    assert np.isnan(got[[1, 2]]).all()
    assert_close(got[[0, 3]], d.inverse_cumulative_hazard(f([0.3, 2.0])))


@pytest.mark.parametrize("family", FAMILIES)
@NON_NUMERIC
@QUANTILE
def test_hazard_level_with_a_non_numeric_dtype_is_rejected(family, bad):
    """spec 4.5/8.1, rev 54 (design d104627): "the section 4.5 dtype rule applies to `e` as to `t`": boolean, complex, string and object `e` raise `TypeError`."""
    with pytest.raises(TypeError):
        families()[family]().inverse_cumulative_hazard(bad)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_boolean_hazard_level_is_rejected_during_tracing(family):
    """spec 4.5/8.1, rev 54 (design d104627): as for `t`, the dtype check also applies to a traced `e` ("the check uses the dtype, not values")."""
    d = families()[family]()
    with pytest.raises(TypeError):
        jax.jit(lambda ee: d.inverse_cumulative_hazard(ee))(jnp.asarray(True))


@pytest.mark.skipif(X64, reason="the float64-to-float32 conversion exists only with 64-bit mode disabled")
@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_float64_hazard_level_converts_without_warning_in_32_bit_mode(family):
    """spec 4.5/8.1, rev 54 (design d104627): as for `t`, a float64 `e` "with 64-bit mode disabled is converted to float32 without a warning"."""
    d = families()[family]()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = d.inverse_cumulative_hazard(np.asarray([0.3, 2.0], dtype=np.float64))
    assert jnp.result_type(got) == jnp.float32
    assert_close(got, d.inverse_cumulative_hazard(f([0.3, 2.0])))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_inverse_cumulative_hazard_has_no_with_info_twin(family):
    """spec 8.1, rev 54: "no `_with_info` twin": accessing `inverse_cumulative_hazard_with_info` raises `AttributeError`."""
    with pytest.raises(AttributeError):
        families()[family]().inverse_cumulative_hazard_with_info  # noqa: B018


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_inverse_cumulative_hazard_options_none_selects_the_defaults(family):
    """spec 8.1/8.5, rev 54: `inverse_cumulative_hazard(e, *, options=None)`: `options=None`, omitted and `SummaryOptions()` agree."""
    d = families()[family]()
    omitted = d.inverse_cumulative_hazard(f(0.7))
    assert_close(d.inverse_cumulative_hazard(f(0.7), options=None), omitted)
    assert_close(d.inverse_cumulative_hazard(f(0.7), options=survjax.SummaryOptions()), omitted)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_inverse_cumulative_hazard_options_are_keyword_only(family):
    """spec 8.1, rev 54: the `*` in `inverse_cumulative_hazard(e, *, options=None)`: positional options raise `TypeError`."""
    with pytest.raises(TypeError):
        families()[family]().inverse_cumulative_hazard(f(0.7), survjax.SummaryOptions())


@QUANTILE_INFO
def test_inverse_cumulative_hazard_options_set_the_solver_limits():
    """spec 8.1/8.5, rev 54: "`inverse_cumulative_hazard` takes `options` for its solver limits"; "failures are NaN".

    ``far_reference_log_logistic`` (``alpha = 1e20``, ``beta = 2``,
    ``reference_time() = 1``): ``H = log1p((t/alpha)**beta)``, so
    ``H⁻¹(log 2) = alpha``. One bracket step cannot reach it (NaN); the
    default 64 steps can (generic tolerance, 12.1 rev 54).
    """
    d = fam.far_reference_log_logistic()(alpha=f(1e20), beta=f(2.0))
    e = f(np.log(2.0))
    assert np.isnan(np.asarray(d.inverse_cumulative_hazard(e, options=survjax.SummaryOptions(max_bracket_steps=1))))
    np.testing.assert_allclose(
        np.asarray(d.inverse_cumulative_hazard(e), dtype=np.float64), 1e20, rtol=10 * (1e-8 if X64 else 1e-4)
    )


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
