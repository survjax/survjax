"""Feature `family.loglogistic`: declared constraints and `ParameterIssues` of `LogLogistic` (spec 4.1, 4.5, 4.7, 12.1; rev 55).

4.1: "`rate`, `shape`, `scale`, `sdlog` are `positive` (finite and `>0`)";
"`from_log` constructors store natural parameters, which are checked like any
other." 4.7: `parameter_constraints`, `ParameterIssues` (`fields`,
`conditions`, `invalid_entries`, `ok`), `jit`/`vmap`, indexing, host
checkpoint. 12.1 (Parameter diagnostics): "built-in declarations". The
declaration order is not fixed by the spec; only the key set is asserted.
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, f, loglogistic_from_log

import survjax

LOGLOGISTIC = pytest.mark.feature("family.loglogistic")


def _fields(issues):
    return {k: np.asarray(v) for k, v in issues.fields.items()}


@LOGLOGISTIC
def test_loglogistic_declares_shape_and_scale():
    """spec 4.1/4.7, rev 55: both parameters are declared: keys `shape` and `scale`."""
    assert set(survjax.LogLogistic.parameter_constraints) == {"shape", "scale"}


@LOGLOGISTIC
def test_both_declarations_are_positive():
    """spec 4.1/4.7, rev 55: `shape` and `scale` are `positive` (finite and `>0`)."""
    x = f([1e-30, 1.0, 0.0, -1.0, NAN, INF])
    for name, con in survjax.LogLogistic.parameter_constraints.items():
        np.testing.assert_array_equal(np.asarray(con(x)), [True, True, False, False, False, False], err_msg=name)


@LOGLOGISTIC
def test_parameter_issues_mark_each_violated_field():
    """spec 4.7, rev 55: one mask per constrained parameter; `invalid_entries` is their union; `ok` is false."""
    d = survjax.LogLogistic(shape=f([2.0, 0.0, 1.0, NAN, 1.5]), scale=f([1.0, 1.0, -1.0, 1.0, INF]))
    issues = d.parameter_issues()
    fields = _fields(issues)
    np.testing.assert_array_equal(fields["shape"], [False, True, False, True, False])
    np.testing.assert_array_equal(fields["scale"], [False, False, True, False, True])
    np.testing.assert_array_equal(np.asarray(issues.invalid_entries), [False, True, True, True, True])
    assert not bool(issues.ok)


@LOGLOGISTIC
def test_heavy_tailed_shapes_are_valid_parameters():
    """spec 4.1/4.7, rev 55: `shape <= 1` (infinite mean, 8.1) is inside `positive`: no parameter issue."""
    issues = survjax.LogLogistic(shape=f([0.2, 0.5, 1.0]), scale=f(1.0)).parameter_issues()
    assert not np.asarray(issues.invalid_entries).any()
    assert dict(issues.conditions) == {}
    assert bool(issues.ok)


@LOGLOGISTIC
def test_parameter_issues_have_the_broadcast_batch_shape():
    """spec 4.7, rev 55: masks have shape `batch_shape` after construction broadcasting."""
    issues = survjax.LogLogistic(shape=f([1.0, 2.0, 3.0]), scale=f([[1.0], [0.0]])).parameter_issues()
    fields = _fields(issues)
    assert fields["shape"].shape == (2, 3)
    np.testing.assert_array_equal(fields["scale"], [[False] * 3, [True] * 3])


@LOGLOGISTIC
def test_parameter_issues_agree_under_jit_and_vmap():
    """spec 4.7/12.1, rev 55: the container works under `jit` and `vmap` with equal masks."""
    d = survjax.LogLogistic(shape=f([2.0, -1.0, 1.0]), scale=f([1.0, 1.0, NAN]))
    want = np.asarray(d.parameter_issues().invalid_entries)
    np.testing.assert_array_equal(np.asarray(jax.jit(lambda dd: dd.parameter_issues().invalid_entries)(d)), want)
    np.testing.assert_array_equal(np.asarray(jax.vmap(lambda dd: dd.parameter_issues().invalid_entries)(d)), want)


@LOGLOGISTIC
def test_indexed_parameter_issues_equal_the_parents_indexed():
    """spec 4.7, rev 55: "`dist[idx].parameter_issues()` equals the parent's masks indexed by `idx`"."""
    d = survjax.LogLogistic(shape=f([2.0, -1.0, 1.0]), scale=f([1.0, 1.0, NAN]))
    parent = _fields(d.parameter_issues())
    child = _fields(d[1:].parameter_issues())
    for name in parent:
        np.testing.assert_array_equal(child[name], parent[name][1:], err_msg=name)


@LOGLOGISTIC
def test_host_checkpoint_raises_on_a_violation():
    """spec 4.7, rev 55: `raise_if_invalid()` raises `survjax.InvalidParameterError` on a violation and returns on valid parameters."""
    survjax.LogLogistic(shape=f(0.5), scale=f(1.0)).parameter_issues().raise_if_invalid()
    with pytest.raises(survjax.InvalidParameterError):
        survjax.LogLogistic(shape=f(0.0), scale=f(1.0)).parameter_issues().raise_if_invalid()


@LOGLOGISTIC
def test_declared_violation_leaves_other_entries_direct_values_unchanged():
    """spec 4.5/12.1, rev 55: declared constraints "do not change direct-method values": valid entries beside a violating one equal their values computed alone."""
    t = f([0.5, 2.0, 1.5])
    batch = survjax.LogLogistic(shape=f([2.0, -1.0, 0.7]), scale=f([1.0, 1.0, 3.0])).log_survival(t)
    alone = survjax.LogLogistic(shape=f([2.0, 0.7]), scale=f([1.0, 3.0])).log_survival(t[np.asarray([0, 2])])
    assert_close(np.asarray(batch)[[0, 2]], alone)


@LOGLOGISTIC
def test_from_log_overflow_is_reported_as_parameter_issue():
    """spec 4.1, rev 55: `from_log` natural parameters are "checked like any other": `exp(1000) = inf` violates `positive` (Q-069)."""
    fields = _fields(loglogistic_from_log(f([0.0, 1000.0]), f([0.0, 0.0])).parameter_issues())
    np.testing.assert_array_equal(fields["shape"], [False, True])
    np.testing.assert_array_equal(fields["scale"], [False, False])
