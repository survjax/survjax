"""A11: `ProportionalHazards` constraints and parameter issues (spec 4.4, 4.7, 6.3; rev 41).

4.4: "The linear predictor declares `real`; the baseline's parameter issues
appear in the wrapper's `parameter_issues()` under the `baseline.` prefix".
4.7: own constrained fields first, then nested ones "with keys prefixed by
the field name and a dot (`baseline.shape`), broadcast to the outer
`batch_shape`", then conditions; `real` is "finite". The custom baseline
``capped_rate`` (``spec_families``) declares ``rate`` and ``cap`` positive and
the condition ``rate_above_cap``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, f, ph

import survjax


def _fields(issues):
    return {k: np.asarray(v) for k, v in issues.fields.items()}


def _conditions(issues):
    return {k: np.asarray(v) for k, v in issues.conditions.items()}


def _assert_masks(actual, expected):
    assert list(actual) == list(expected)
    for key, value in expected.items():
        got = np.asarray(actual[key])
        assert got.dtype == np.bool_, key
        np.testing.assert_array_equal(got, np.asarray(value, dtype=bool), err_msg=key)


# --------------------------------------------------------------- declarations


@pytest.mark.step("A11")
def test_wrapper_declares_only_the_linear_predictor():
    """spec 4.7, rev 41: `parameter_constraints` maps "each constrained field name of the class itself"; nested ones are not included."""
    assert list(survjax.ProportionalHazards.parameter_constraints) == ["log_hazard_ratio"]


@pytest.mark.step("A11")
def test_linear_predictor_constraint_is_real():
    """spec 4.4/4.7, rev 41: the linear predictor declares `real`: finite; NaN and `±inf` are outside."""
    con = survjax.ProportionalHazards.parameter_constraints["log_hazard_ratio"]
    x = f([-50.0, 0.0, 3.0, NAN, INF, -INF])
    np.testing.assert_array_equal(np.asarray(con(x)), [True, True, True, False, False, False])


# --------------------------------------------------------------- keys and masks


@pytest.mark.step("A11")
def test_exponential_baseline_keys_are_prefixed():
    """spec 4.4/4.7, rev 41: own `log_hazard_ratio` first, then `baseline.rate`; no conditions."""
    d = ph(survjax.Exponential(rate=f([1.0, 0.0, 2.0, NAN])), f([0.0, 0.0, INF, 0.5]))
    issues = d.parameter_issues()
    _assert_masks(
        _fields(issues),
        {
            "log_hazard_ratio": [False, False, True, False],
            "baseline.rate": [False, True, False, True],
        },
    )
    _assert_masks(_conditions(issues), {})
    np.testing.assert_array_equal(np.asarray(issues.invalid_entries), [False, True, True, True])
    assert not bool(np.asarray(issues.ok))


@pytest.mark.step("A11")
def test_weibull_baseline_keys_are_prefixed():
    """spec 4.4/4.7, rev 41: `baseline.shape` and `baseline.scale` follow `log_hazard_ratio`."""
    base = survjax.Weibull(shape=f([1.0, -1.0, 2.0]), scale=f([1.0, 1.0, INF]))
    issues = ph(base, f([-INF, 0.0, 0.0])).parameter_issues()
    fields = _fields(issues)
    assert list(fields)[0] == "log_hazard_ratio"
    assert set(fields) == {"log_hazard_ratio", "baseline.shape", "baseline.scale"}
    np.testing.assert_array_equal(fields["log_hazard_ratio"], [True, False, False])
    np.testing.assert_array_equal(fields["baseline.shape"], [False, True, False])
    np.testing.assert_array_equal(fields["baseline.scale"], [False, False, True])
    np.testing.assert_array_equal(np.asarray(issues.invalid_entries), [True, True, True])


@pytest.mark.step("A11")
def test_valid_wrapper_reports_no_issue():
    """spec 4.7, rev 41: True marks a violation; valid finite parameters, including large negative `eta`, report none."""
    d = ph(survjax.Weibull(shape=f(2.0), scale=f(1.0)), f([-30.0, 0.0, 30.0]))
    issues = d.parameter_issues()
    assert not np.asarray(issues.invalid_entries).any()
    assert bool(np.asarray(issues.ok))


@pytest.mark.step("A11")
def test_shared_baseline_mask_is_broadcast_to_the_outer_batch():
    """spec 4.7, rev 41: nested masks are "broadcast to the outer `batch_shape`"."""
    d = ph(survjax.Exponential(rate=f(-1.0)), f([0.0, 1.0, 2.0]))
    issues = d.parameter_issues()
    fields = _fields(issues)
    assert fields["baseline.rate"].shape == (3,)
    np.testing.assert_array_equal(fields["baseline.rate"], [True, True, True])
    np.testing.assert_array_equal(fields["log_hazard_ratio"], [False, False, False])


@pytest.mark.step("A11")
def test_draw_baseline_masks_broadcast_against_draw_by_arm_predictor():
    """spec 4.4/4.7, rev 41: baseline `(S, 1)` with `eta` `(S, 2)`; every mask has shape `(S, 2)`."""
    base = survjax.Weibull(shape=f([[1.0], [0.0], [2.0]]), scale=f([[1.0], [1.0], [1.0]]))
    issues = ph(base, f([[0.0, NAN], [0.0, 0.0], [0.0, 0.0]])).parameter_issues()
    fields = _fields(issues)
    for key, value in fields.items():
        assert value.shape == (3, 2), key
    np.testing.assert_array_equal(fields["baseline.shape"], [[False, False], [True, True], [False, False]])
    np.testing.assert_array_equal(
        np.asarray(issues.invalid_entries), [[False, True], [True, True], [False, False]]
    )


@pytest.mark.step("A11")
def test_custom_baseline_condition_appears_with_the_prefix():
    """spec 4.7, rev 41: nested conditions follow own ones "with dotted keys"; a custom baseline's `rate_above_cap`."""
    base = fam.capped_rate()(rate=f([1.0, 3.0, 1.0]), cap=f([2.0, 2.0, -1.0]))
    issues = ph(base, f(0.2)).parameter_issues()
    _assert_masks(
        _fields(issues),
        {
            "log_hazard_ratio": [False, False, False],
            "baseline.rate": [False, False, False],
            "baseline.cap": [False, False, True],
        },
    )
    _assert_masks(_conditions(issues), {"baseline.rate_above_cap": [False, True, False]})
    np.testing.assert_array_equal(np.asarray(issues.invalid_entries), [False, True, True])


# --------------------------------------------------------------- transformations and indexing


def _mixed():
    base = survjax.Weibull(shape=f([1.0, 0.0, 2.0, 1.5]), scale=f([1.0, 1.0, 1.0, NAN]))
    return ph(base, f([INF, 0.0, 0.3, 0.0]))


def _flat(issues):
    return (
        {k: np.asarray(v) for k, v in issues.fields.items()},
        {k: np.asarray(v) for k, v in issues.conditions.items()},
        np.asarray(issues.invalid_entries),
        bool(np.asarray(issues.ok)),
    )


def _assert_same(left, right):
    lf, lc, li, lo = _flat(left)
    rf, rc, ri, ro = _flat(right)
    _assert_masks(lf, rf)
    _assert_masks(lc, rc)
    np.testing.assert_array_equal(li, ri)
    assert lo == ro


@pytest.mark.step("A11")
def test_parameter_issues_agree_under_jit():
    """spec 4.7/6.3, rev 41: the container is a PyTree with static keys; it works under `jit`."""
    d = _mixed()
    _assert_same(jax.jit(lambda dd: dd.parameter_issues())(d), d.parameter_issues())


@pytest.mark.step("A11")
def test_parameter_issues_agree_under_vmap():
    """spec 4.7, rev 41: per-entry diagnostics under `vmap` equal the batched masks."""
    d = _mixed()
    mapped = jax.vmap(lambda dd: dd.parameter_issues().invalid_entries)(d)
    np.testing.assert_array_equal(np.asarray(mapped), [True, True, False, True])


@pytest.mark.step("A11")
def test_indexed_wrapper_issues_equal_indexed_masks():
    """spec 4.7, rev 41: "`dist[idx].parameter_issues()` equals the parent's masks indexed by `idx`"."""
    d = _mixed()
    idx = jnp.asarray([3, 0])
    parent = _flat(d.parameter_issues())
    child = _flat(d[idx].parameter_issues())
    for key in parent[0]:
        np.testing.assert_array_equal(child[0][key], parent[0][key][np.asarray(idx)], err_msg=key)
    np.testing.assert_array_equal(child[2], parent[2][np.asarray(idx)])


@pytest.mark.step("A11")
def test_raise_if_invalid_names_the_prefixed_baseline_field():
    """spec 4.7, rev 41: `raise_if_invalid()` raises `InvalidParameterError` "naming the violated fields"."""
    d = ph(survjax.Weibull(shape=f([1.0, 1.0]), scale=f([1.0, -2.0])), f(0.0))
    with pytest.raises(survjax.InvalidParameterError, match=r"baseline\.scale"):
        d.parameter_issues().raise_if_invalid()


@pytest.mark.step("A11")
def test_raise_if_invalid_names_the_linear_predictor():
    """spec 4.7, rev 41: a non-finite `eta` is named as `log_hazard_ratio`."""
    d = ph(survjax.Exponential(rate=f(1.0)), f([0.0, NAN]))
    with pytest.raises(survjax.InvalidParameterError, match="log_hazard_ratio"):
        d.parameter_issues().raise_if_invalid()


@pytest.mark.step("A11")
def test_raise_if_invalid_returns_for_valid_wrapper():
    """spec 4.7, rev 41: "it returns normally when no violation is reported"."""
    ph(survjax.Exponential(rate=f([1.0, 2.0])), f([0.0, -3.0])).parameter_issues().raise_if_invalid()


# --------------------------------------------------------------- direct methods unchanged by declarations


@pytest.mark.step("A11")
def test_declared_violation_does_not_change_finite_direct_values():
    """spec 4.5/4.7, rev 41: declared constraints "do not change direct-method values".

    A custom baseline with ``rate > cap`` (only the named condition flags it):
    the PH law ``log S = -exp(eta) r t`` is unchanged.
    """
    r, eta, t = 3.0, 0.2, 1.5
    d = ph(fam.capped_rate()(rate=f(r), cap=f(2.0)), f(eta))
    assert bool(np.asarray(d.parameter_issues().invalid_entries))
    assert_close(d.log_survival(f(t)), -np.exp(eta) * r * t)
