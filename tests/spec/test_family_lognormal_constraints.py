"""Feature `family.lognormal`: declared constraints and `ParameterIssues` of `LogNormal` (spec 4.1, 4.5, 4.7, 12.1; rev 55).

4.1: "`rate`, `shape`, `scale`, `sdlog` are `positive` (finite and `>0`), and
`meanlog` is `real` (finite). Constraints are checks, not transforms";
"`from_log` constructors store natural parameters, which are checked like any
other." 4.7: `parameter_constraints` maps "each constrained field name of the
class itself to its constraint"; `ParameterIssues` has `fields`,
`conditions`, `invalid_entries` and `ok`; the container works under `jit` and
`vmap`, and `dist[idx].parameter_issues()` equals the parent's masks indexed.
12.1 (Parameter diagnostics): "built-in declarations; direct-method values
unchanged by declared violations". 4.5: "Declared constraints ... do not
change direct-method values."

The declaration order of the two fields is not fixed by the spec, so only the
key set is asserted.
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, f, lognormal_from_log

import survjax

LOGNORMAL = pytest.mark.feature("family.lognormal")


def _fields(issues):
    return {k: np.asarray(v) for k, v in issues.fields.items()}


@LOGNORMAL
def test_lognormal_declares_meanlog_and_sdlog():
    """spec 4.1/4.7, rev 55: both parameters are declared: `parameter_constraints` has the keys `meanlog` and `sdlog`."""
    assert set(survjax.LogNormal.parameter_constraints) == {"meanlog", "sdlog"}


@LOGNORMAL
def test_meanlog_declaration_is_real():
    """spec 4.1/4.7, rev 55: `meanlog` is `real`: finite values of either sign are inside, NaN and `±inf` outside."""
    con = survjax.LogNormal.parameter_constraints["meanlog"]
    np.testing.assert_array_equal(
        np.asarray(con(f([-1e30, -2.0, 0.0, 3.0, NAN, INF, -INF]))),
        [True, True, True, True, False, False, False],
    )


@LOGNORMAL
def test_sdlog_declaration_is_positive():
    """spec 4.1/4.7, rev 55: `sdlog` is `positive` (finite and `>0`)."""
    con = survjax.LogNormal.parameter_constraints["sdlog"]
    np.testing.assert_array_equal(
        np.asarray(con(f([1e-30, 1.0, 0.0, -1.0, NAN, INF]))),
        [True, True, False, False, False, False],
    )


@LOGNORMAL
def test_parameter_issues_mark_each_violated_field():
    """spec 4.7, rev 55: one mask per constrained parameter; `invalid_entries` is their union; `ok` is false."""
    d = survjax.LogNormal(meanlog=f([0.0, NAN, 1.0, INF, 0.5]), sdlog=f([1.0, 1.0, 0.0, 1.0, -2.0]))
    issues = d.parameter_issues()
    fields = _fields(issues)
    np.testing.assert_array_equal(fields["meanlog"], [False, True, False, True, False])
    np.testing.assert_array_equal(fields["sdlog"], [False, False, True, False, True])
    np.testing.assert_array_equal(np.asarray(issues.invalid_entries), [False, True, True, True, True])
    assert not bool(issues.ok)


@LOGNORMAL
def test_valid_lognormal_reports_no_issue():
    """spec 4.7, rev 55: valid parameters give all-false masks, no conditions, and `ok`."""
    issues = survjax.LogNormal(meanlog=f([-3.0, 0.0, 4.0]), sdlog=f([0.1, 1.0, 5.0])).parameter_issues()
    assert not np.asarray(issues.invalid_entries).any()
    assert dict(issues.conditions) == {}
    assert bool(issues.ok)


@LOGNORMAL
def test_parameter_issues_have_the_broadcast_batch_shape():
    """spec 4.7, rev 55: masks have shape `batch_shape` after construction broadcasting."""
    issues = survjax.LogNormal(meanlog=f([0.0, 1.0, 2.0]), sdlog=f([[1.0], [-1.0]])).parameter_issues()
    fields = _fields(issues)
    assert fields["meanlog"].shape == (2, 3)
    np.testing.assert_array_equal(fields["sdlog"], [[False] * 3, [True] * 3])


@LOGNORMAL
def test_parameter_issues_agree_under_jit_and_vmap():
    """spec 4.7/12.1, rev 55: the container works under `jit` and `vmap` with equal masks."""
    d = survjax.LogNormal(meanlog=f([0.0, NAN, 1.0]), sdlog=f([1.0, 1.0, 0.0]))
    want = np.asarray(d.parameter_issues().invalid_entries)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda dd: dd.parameter_issues().invalid_entries)(d)), want)
    np.testing.assert_array_equal(np.asarray(jax.vmap(lambda dd: dd.parameter_issues().invalid_entries)(d)), want)


@LOGNORMAL
def test_indexed_parameter_issues_equal_the_parents_indexed():
    """spec 4.7, rev 55: "`dist[idx].parameter_issues()` equals the parent's masks indexed by `idx`"."""
    d = survjax.LogNormal(meanlog=f([0.0, NAN, 1.0]), sdlog=f([1.0, 1.0, 0.0]))
    parent = _fields(d.parameter_issues())
    child = _fields(d[1:].parameter_issues())
    for name in parent:
        np.testing.assert_array_equal(child[name], parent[name][1:], err_msg=name)


@LOGNORMAL
def test_host_checkpoint_raises_on_a_violation():
    """spec 4.7, rev 55: `raise_if_invalid()` raises `survjax.InvalidParameterError` (a `ValueError`) on a violation and returns on valid parameters."""
    survjax.LogNormal(meanlog=f(0.0), sdlog=f(1.0)).parameter_issues().raise_if_invalid()
    with pytest.raises(survjax.InvalidParameterError):
        survjax.LogNormal(meanlog=f(0.0), sdlog=f(-1.0)).parameter_issues().raise_if_invalid()


@LOGNORMAL
def test_declared_violation_leaves_other_entries_direct_values_unchanged():
    """spec 4.5/12.1, rev 55: declared constraints "do not change direct-method values": a violating entry beside valid ones leaves the valid entries' values as computed alone (the violating entry's own value is "whatever the family's formula gives" and is not asserted)."""
    t = f([0.5, 2.0, 1.5])
    batch = survjax.LogNormal(meanlog=f([0.0, 0.0, 1.0]), sdlog=f([1.0, -1.0, 0.5])).log_survival(t)
    alone = survjax.LogNormal(meanlog=f([0.0, 1.0]), sdlog=f([1.0, 0.5])).log_survival(t[np.asarray([0, 2])])
    assert_close(np.asarray(batch)[[0, 2]], alone)


@LOGNORMAL
def test_from_log_overflow_is_reported_as_parameter_issue():
    """spec 4.1, rev 55: `from_log` natural parameters are "checked like any other": `exp(1000) = inf` violates `positive` (Q-069)."""
    d = lognormal_from_log(f([0.0, 0.0]), f([0.0, 1000.0]))
    fields = _fields(d.parameter_issues())
    np.testing.assert_array_equal(fields["meanlog"], [False, False])
    np.testing.assert_array_equal(fields["sdlog"], [False, True])
