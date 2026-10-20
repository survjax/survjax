"""A6: declared constraints, `parameter_conditions()` and `ParameterIssues` (spec 4.1, 4.2, 4.5, 4.7, 6.3; rev 28).

The families are in ``spec_families.py``: ``component_constrained`` (batch and
component constraints), ``capped_rate`` (one named condition),
``scaled_capped`` (a nested ``capped_rate`` with its own constraint and
condition), ``const_hazard`` (nothing declared), and the
``constrained_rate``/``unconstrained_rate`` twins.
"""

import dataclasses
import re
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, f, leaves

import survjax


def _np(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def _fields(issues):
    return {k: np.asarray(v) for k, v in issues.fields.items()}


def _conditions(issues):
    return {k: np.asarray(v) for k, v in issues.conditions.items()}


def _assert_masks(actual, expected):
    """Same keys in the same order, boolean masks with equal values."""
    assert list(actual) == list(expected)
    for key, value in expected.items():
        got = np.asarray(actual[key])
        assert got.dtype == np.bool_, key
        np.testing.assert_array_equal(got, np.asarray(value, dtype=bool), err_msg=key)


def _assert_same_issues(left, right):
    _assert_masks(_fields(left), _fields(right))
    _assert_masks(_conditions(left), _conditions(right))
    np.testing.assert_array_equal(
        np.asarray(left.invalid_entries), np.asarray(right.invalid_entries)
    )
    assert bool(np.asarray(left.ok)) == bool(np.asarray(right.ok))


def _component(rate, w, knots, free=None):
    rate = f(rate)
    return fam.component_constrained()(
        rate=rate,
        free=f(np.zeros(np.shape(rate)) if free is None else free),
        w=f(w),
        knots=f(knots),
    )


def _capped(rate, cap):
    return fam.capped_rate()(rate=f(rate), cap=f(cap))


def _scaled(rate, cap, log_ratio):
    return fam.scaled_capped()(base=_capped(rate, cap), log_ratio=f(log_ratio))


GOOD_KNOTS = [0.0, 1.0, 2.0]


# ---------------------------------------------------------- field declarations


@pytest.mark.step("A6")
def test_param_accepts_constraint_keyword():
    """spec 4.2, rev 28: `survjax.param(component_ndim=0, constraint=None)` takes a constraint."""
    dist = fam.constrained_rate()(rate=f([1.0, 2.0]))
    assert dist.batch_shape == (2,)


@pytest.mark.step("A6")
def test_constraints_are_not_pytree_leaves():
    """spec 4.7, rev 28: "Constraints live in field metadata, not in the PyTree"."""
    rate = f([1.0, 2.0])
    dist = fam.constrained_rate()(rate=rate)
    (leaf,) = leaves(dist)
    np.testing.assert_array_equal(np.asarray(leaf), rate)


@pytest.mark.step("A6")
def test_constrained_and_unconstrained_twins_have_equal_leaves():
    """spec 4.7, rev 28: declaring a constraint adds no leaf and stores the parameter unchanged."""
    rate = f([-1.0, 2.0])
    a = leaves(fam.constrained_rate()(rate=rate))
    b = leaves(fam.unconstrained_rate()(rate=rate))
    assert len(a) == len(b) == 1
    np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))


@pytest.mark.step("A6")
def test_parameter_constraints_lists_own_constrained_fields_in_declaration_order():
    """spec 4.7, rev 28: `parameter_constraints` maps each constrained field "in declaration order"; undeclared fields are absent."""
    cls = fam.component_constrained()
    pc = cls.parameter_constraints
    assert list(pc) == ["rate", "w", "knots"]
    assert pc["rate"] is survjax.constraints.positive
    assert pc["w"] is survjax.constraints.positive
    assert pc["knots"] is survjax.constraints.ordered


@pytest.mark.step("A6")
def test_parameter_constraints_excludes_nested_distributions():
    """spec 4.7, rev 28: "nested distributions are not included" in `parameter_constraints`."""
    pc = fam.scaled_capped().parameter_constraints
    assert list(pc) == ["log_ratio"]
    assert pc["log_ratio"] is survjax.constraints.real


@pytest.mark.step("A6")
def test_parameter_constraints_is_empty_without_declarations():
    """spec 4.7, rev 28: a family with no declared constraint has an empty `parameter_constraints`."""
    assert len(fam.const_hazard().parameter_constraints) == 0


@pytest.mark.step("A6")
def test_parameter_constraints_is_read_only():
    """spec 4.7, rev 28: `parameter_constraints` is "a read-only mapping" (Q-034)."""
    pc = fam.constrained_rate().parameter_constraints
    with pytest.raises(TypeError):
        pc["rate"] = survjax.constraints.real
    assert pc["rate"] is survjax.constraints.positive


def _undecorated(**param_kwargs):
    @dataclasses.dataclass(frozen=True)
    class Candidate(survjax.SurvivalDistribution):
        x: jax.Array = survjax.param(**param_kwargs)

        def log_survival(self, t):
            return -self.x * t

        def reference_time(self):
            return 1.0 / self.x

    return Candidate


@pytest.mark.step("A6")
@pytest.mark.parametrize("bad", [1.0, "positive"], ids=["float", "str"])
def test_register_family_rejects_non_callable_constraint(bad):
    """spec 4.7, rev 28: "At decoration, `register_family` raises `TypeError` if a declared constraint is not callable"."""
    cls = _undecorated(constraint=bad)
    with pytest.raises(TypeError):
        survjax.register_family(cls)


@pytest.mark.step("A6")
def test_register_family_rejects_vector_constraint_on_scalar_field():
    """spec 4.7, rev 28: `ordered` (`event_dim=1`) on a `component_ndim=0` field raises `TypeError` at decoration."""
    cls = _undecorated(constraint=survjax.constraints.ordered)
    with pytest.raises(TypeError):
        survjax.register_family(cls)


class _EventDimTwo:
    """A callable constraint reducing two trailing axes (NumPyro protocol, spec 4.7)."""

    event_dim = 2

    def __call__(self, x):
        return jnp.all(jnp.isfinite(x), axis=(-2, -1))


@pytest.mark.step("A6")
def test_register_family_rejects_event_dim_above_component_ndim():
    """spec 4.7, rev 28: `event_dim` greater than the field's `component_ndim` raises `TypeError`."""
    cls = _undecorated(component_ndim=1, constraint=_EventDimTwo())
    with pytest.raises(TypeError):
        survjax.register_family(cls)


@pytest.mark.step("A6")
def test_register_family_accepts_event_dim_equal_to_component_ndim():
    """spec 4.7, rev 28: `event_dim == component_ndim` is valid (no remaining component axis)."""
    cls = survjax.register_family(_undecorated(component_ndim=2, constraint=_EventDimTwo()))
    issues = cls(x=f(np.ones((3, 2, 2)))).parameter_issues()
    _assert_masks(_fields(issues), {"x": [False, False, False]})


@pytest.mark.step("A6")
def test_plain_function_is_a_constraint_with_event_dim_zero():
    """spec 4.7, rev 28: a constraint is any callable `c(x) -> bool array`; `event_dim` defaults to `0`."""

    def below_ten(x):
        return x < 10.0

    cls = survjax.register_family(_undecorated(constraint=below_ten))
    assert cls.parameter_constraints["x"] is below_ten
    issues = cls(x=f([1.0, 10.0, 11.0])).parameter_issues()
    _assert_masks(_fields(issues), {"x": [False, True, True]})


@pytest.mark.step("A6")
def test_numpyro_constraint_object_is_accepted():
    """spec 4.7 and 12.1, rev 28: NumPyro constraint objects are accepted as they are (Bayesian-demo environment only)."""
    numpyro_constraints = pytest.importorskip("numpyro.distributions.constraints")
    cls = survjax.register_family(_undecorated(constraint=numpyro_constraints.positive))
    issues = cls(x=f([1.0, 0.0, -1.0])).parameter_issues()
    _assert_masks(_fields(issues), {"x": [False, True, True]})


# ------------------------------------------------------- parameter_conditions


@pytest.mark.step("A6")
def test_default_parameter_conditions_is_empty():
    """spec 4.2, rev 28: the `parameter_conditions()` default is "An empty mapping"."""
    conditions = fam.const_hazard()(rate=f([1.0, 2.0])).parameter_conditions()
    assert len(conditions) == 0


# ------------------------------------------------- ParameterIssues: no declarations


@pytest.mark.step("A6")
def test_family_with_nothing_declared_reports_no_issues():
    """spec 4.7, rev 28: nothing declared gives empty `fields`/`conditions`, `invalid_entries` all False and `ok` True."""
    dist = fam.const_hazard()(rate=f([1.0, -1.0, NAN]))
    issues = dist.parameter_issues()
    assert dict(issues.fields) == {}
    assert dict(issues.conditions) == {}
    entries = np.asarray(issues.invalid_entries)
    assert entries.shape == (3,)
    assert entries.dtype == np.bool_
    assert not entries.any()
    assert bool(np.asarray(issues.ok))


@pytest.mark.step("A6")
def test_issue_fields_are_dicts():
    """spec 4.7, rev 28: `fields` and `conditions` are `dict`s."""
    issues = _capped([1.0], [2.0]).parameter_issues()
    assert isinstance(issues.fields, dict)
    assert isinstance(issues.conditions, dict)


# ---------------------------------------------- ParameterIssues: own declarations


@pytest.mark.step("A6")
def test_field_mask_marks_violations_true():
    """spec 4.7, rev 28: one mask per constrained field, `~c(leaf)`; "True marks a violation"."""
    dist = _component(
        rate=[1.0, 0.0, -1.0, NAN, INF],
        w=np.ones((5, 2)),
        knots=np.tile(GOOD_KNOTS, (5, 1)),
    )
    fields = _fields(dist.parameter_issues())
    np.testing.assert_array_equal(fields["rate"], [False, True, True, True, True])


@pytest.mark.step("A6")
def test_undeclared_fields_are_absent_from_fields():
    """spec 4.7, rev 28: "Fields without a declared constraint are absent from `fields`"."""
    dist = _component(
        rate=[1.0], w=[[1.0, 1.0]], knots=[GOOD_KNOTS], free=[NAN]
    )
    issues = dist.parameter_issues()
    assert list(issues.fields) == ["rate", "w", "knots"]
    assert not np.asarray(issues.invalid_entries).any()


@pytest.mark.step("A6")
def test_remaining_component_axis_is_reduced_by_any():
    """spec 4.7, rev 28: a scalar constraint on a `component_ndim=1` field is reduced by `any` over the component axis."""
    w = np.ones((4, 3))
    w[1, 2] = 0.0
    w[3, :] = -1.0
    dist = _component(rate=np.ones(4), w=w, knots=np.tile(GOOD_KNOTS, (4, 1)))
    mask = _fields(dist.parameter_issues())["w"]
    assert mask.shape == (4,)
    np.testing.assert_array_equal(mask, [False, True, False, True])


@pytest.mark.step("A6")
def test_vector_constraint_consumes_the_component_axis():
    """spec 4.7, rev 28: `ordered` on a `component_ndim=1` field gives one entry per batch element."""
    knots = np.tile(GOOD_KNOTS, (3, 1))
    knots[2] = [0.0, 1.0, 1.0]
    dist = _component(rate=np.ones(3), w=np.ones((3, 2)), knots=knots)
    mask = _fields(dist.parameter_issues())["knots"]
    assert mask.shape == (3,)
    np.testing.assert_array_equal(mask, [False, False, True])


@pytest.mark.step("A6")
def test_masks_have_batch_shape_for_two_dimensional_batch():
    """spec 4.7, rev 28: every mask in `fields` and `invalid_entries` has shape `batch_shape`."""
    rate = np.ones((2, 3))
    rate[1, 0] = -1.0
    dist = _component(rate=rate, w=np.ones((2, 3, 2)), knots=np.tile(GOOD_KNOTS, (2, 3, 1)))
    issues = dist.parameter_issues()
    for key, mask in _fields(issues).items():
        assert mask.shape == (2, 3), key
    entries = np.asarray(issues.invalid_entries)
    assert entries.shape == (2, 3)
    np.testing.assert_array_equal(entries, [[False] * 3, [True, False, False]])


@pytest.mark.step("A6")
def test_condition_mask_marks_violations_true():
    """spec 4.7, rev 28: one mask per named condition from `parameter_conditions()`, True where violated."""
    dist = _capped(rate=[1.0, 3.0, 2.0], cap=[2.0, 2.0, 2.0])
    _assert_masks(
        _conditions(dist.parameter_issues()),
        {"rate_above_cap": [False, True, False]},
    )


@pytest.mark.step("A6")
def test_invalid_entries_is_union_of_fields_and_conditions():
    """spec 4.7, rev 28: `invalid_entries` is the "union of all masks in `fields` and `conditions`"."""
    dist = _capped(rate=[1.0, 3.0, -1.0, 1.0], cap=[2.0, 2.0, 2.0, NAN])
    issues = dist.parameter_issues()
    _assert_masks(
        _fields(issues),
        {"rate": [False, False, True, False], "cap": [False, False, False, True]},
    )
    np.testing.assert_array_equal(
        np.asarray(issues.invalid_entries), [False, True, True, True]
    )


@pytest.mark.step("A6")
def test_ok_is_scalar_negated_any():
    """spec 4.7, rev 28: `ok` is the scalar `~jnp.any(invalid_entries)`."""
    bad = _capped(rate=[1.0, 3.0], cap=[2.0, 2.0]).parameter_issues()
    good = _capped(rate=[1.0, 1.5], cap=[2.0, 2.0]).parameter_issues()
    assert np.shape(bad.ok) == ()
    assert np.asarray(bad.ok).dtype == np.bool_
    assert not bool(np.asarray(bad.ok))
    assert bool(np.asarray(good.ok))


@pytest.mark.step("A6")
def test_scalar_distribution_has_scalar_masks():
    """spec 4.7, rev 28: for `batch_shape == ()` every mask is a scalar."""
    issues = _capped(rate=3.0, cap=2.0).parameter_issues()
    for mask in (*_fields(issues).values(), *_conditions(issues).values()):
        assert mask.shape == ()
    assert np.asarray(issues.invalid_entries).shape == ()
    assert bool(np.asarray(issues.invalid_entries))


# -------------------------------------------------- ParameterIssues: nesting


@pytest.mark.step("A6")
def test_nested_field_keys_are_dotted_and_follow_own_fields():
    """spec 4.7, rev 28: own fields first, then nested ones "with keys prefixed by the field name and a dot"."""
    issues = _scaled(rate=[1.0, -1.0], cap=[2.0, 2.0], log_ratio=[0.0, 0.0]).parameter_issues()
    _assert_masks(
        _fields(issues),
        {
            "log_ratio": [False, False],
            "base.rate": [False, True],
            "base.cap": [False, False],
        },
    )


@pytest.mark.step("A6")
def test_nested_condition_keys_are_dotted_and_follow_own_conditions():
    """spec 4.7, rev 28: conditions are "own first, then nested ones with dotted keys"."""
    issues = _scaled(
        rate=[1.0, 3.0, 1.0], cap=[2.0, 2.0, 2.0], log_ratio=[0.0, 0.0, 5.0]
    ).parameter_issues()
    _assert_masks(
        _conditions(issues),
        {
            "ratio_too_large": [False, False, True],
            "base.rate_above_cap": [False, True, False],
        },
    )
    np.testing.assert_array_equal(
        np.asarray(issues.invalid_entries), [False, True, True]
    )


@pytest.mark.step("A6")
def test_nested_masks_are_broadcast_to_outer_batch_shape():
    """spec 4.7, rev 28: nested issues are "broadcast to the outer `batch_shape`"."""
    base = _capped(rate=-1.0, cap=2.0)  # batch ()
    dist = fam.scaled_capped()(base=base, log_ratio=f([0.0, 0.0, 0.0]))
    assert dist.batch_shape == (3,)
    issues = dist.parameter_issues()
    for key, mask in (*_fields(issues).items(), *_conditions(issues).items()):
        assert mask.shape == (3,), key
    np.testing.assert_array_equal(_fields(issues)["base.rate"], [True, True, True])


@pytest.mark.step("A6")
def test_nested_violation_makes_outer_issues_not_ok():
    """spec 4.7, rev 28: nested masks join `invalid_entries`, so a nested violation fails `ok`."""
    issues = _scaled(rate=[1.0], cap=[NAN], log_ratio=[0.0]).parameter_issues()
    np.testing.assert_array_equal(np.asarray(issues.invalid_entries), [True])
    assert not bool(np.asarray(issues.ok))


@pytest.mark.step("A6")
def test_issue_keys_are_fixed_by_the_class():
    """spec 4.7, rev 28: "Keys are fixed by the class", whether or not any entry is violated."""
    good = _scaled(rate=[1.0], cap=[2.0], log_ratio=[0.0]).parameter_issues()
    bad = _scaled(rate=[-1.0], cap=[NAN], log_ratio=[9.0]).parameter_issues()
    assert list(good.fields) == list(bad.fields)
    assert list(good.conditions) == list(bad.conditions)
    assert jax.tree_util.tree_structure(good) == jax.tree_util.tree_structure(bad)


# ---------------------------------------------------- indexing and transforms


def _mixed_scaled():
    return _scaled(
        rate=[1.0, -1.0, 3.0, 1.0, 1.0],
        cap=[2.0, 2.0, 2.0, NAN, 2.0],
        log_ratio=[0.0, 0.0, 0.0, 0.0, 9.0],
    )


@pytest.mark.step("A6")
@pytest.mark.parametrize(
    "idx",
    [slice(1, 4), 2, np.array([4, 0, 1])],
    ids=["slice", "int", "int-array"],
)
def test_indexed_issues_equal_indexed_parent_masks(idx):
    """spec 4.7, rev 28: "`dist[idx].parameter_issues()` equals the parent's masks indexed by `idx`"."""
    parent = _mixed_scaled().parameter_issues()
    child = _mixed_scaled()[idx].parameter_issues()
    _assert_masks(_fields(child), {k: v[idx] for k, v in _fields(parent).items()})
    _assert_masks(
        _conditions(child), {k: v[idx] for k, v in _conditions(parent).items()}
    )
    expected_entries = np.asarray(parent.invalid_entries)[idx]
    np.testing.assert_array_equal(np.asarray(child.invalid_entries), expected_entries)
    assert bool(np.asarray(child.ok)) == (not np.any(expected_entries))


@pytest.mark.step("A6")
def test_parameter_issues_under_jit_matches_eager():
    """spec 4.7 and 6.3, rev 28: `ParameterIssues` is a registered PyTree returned from `jit` with eager values."""
    dist = _mixed_scaled()
    _assert_same_issues(
        _np(jax.jit(lambda d: d.parameter_issues())(dist)), dist.parameter_issues()
    )


@pytest.mark.step("A6")
def test_parameter_issues_with_traced_construction_matches_eager():
    """spec 4.1 and 4.7, rev 28: constructors never check values; issues from traced parameters equal eager ones."""
    rate, cap = f([1.0, -1.0, 3.0]), f([2.0, 2.0, 2.0])

    def run(r, c):
        return fam.capped_rate()(rate=r, cap=c).parameter_issues()

    _assert_same_issues(_np(jax.jit(run)(rate, cap)), run(rate, cap))


@pytest.mark.step("A6")
def test_parameter_issues_under_vmap_matches_eager():
    """spec 4.7, rev 28: `parameter_issues()` works under `vmap`; mapped masks equal the eager masks per entry."""
    dist = _mixed_scaled()
    mapped = _np(jax.vmap(lambda d: d.parameter_issues())(dist))
    eager = dist.parameter_issues()
    _assert_masks(_fields(mapped), _fields(eager))
    _assert_masks(_conditions(mapped), _conditions(eager))
    np.testing.assert_array_equal(
        np.asarray(mapped.invalid_entries), np.asarray(eager.invalid_entries)
    )
    # per-entry ok under vmap: one scalar per mapped entry
    np.testing.assert_array_equal(
        np.asarray(mapped.ok), ~np.asarray(eager.invalid_entries)
    )


@pytest.mark.step("A6")
def test_parameter_issues_is_a_pytree_round_trip():
    """spec 6.3, rev 28: `ParameterIssues` is a registered PyTree (flatten/unflatten round trip)."""
    issues = _mixed_scaled().parameter_issues()
    flat, tree = jax.tree_util.tree_flatten(issues)
    assert flat  # the masks are leaves, not opaque
    rebuilt = jax.tree_util.tree_unflatten(tree, flat)
    _assert_same_issues(rebuilt, issues)


# --------------------------------------- parameter_conditions() contract errors


def _conditions_family(make):
    """A ``capped_rate``-like family whose ``parameter_conditions()`` returns ``make(self)``."""

    @survjax.register_family
    @dataclasses.dataclass(frozen=True)
    class BadConditions(survjax.SurvivalDistribution):
        rate: jax.Array = survjax.param()

        def log_survival(self, t):
            return -self.rate * t

        def reference_time(self):
            return 1.0 / self.rate

        def parameter_conditions(self):
            return make(self)

    return BadConditions


_BAD_CONDITIONS = {
    "list-of-pairs": lambda d: [("too_big", d.rate > 1.0)],
    "empty-key": lambda d: {"": d.rate > 1.0},
    "dotted-key": lambda d: {"too.big": d.rate > 1.0},
    "int-key": lambda d: {1: d.rate > 1.0},
    "float-value": lambda d: {"too_big": jnp.where(d.rate > 1.0, 1.0, 0.0)},
    "int-value": lambda d: {"too_big": (d.rate > 1.0).astype(jnp.int32)},
    "scalar-value": lambda d: {"too_big": jnp.any(d.rate > 1.0)},
    "extra-axis-value": lambda d: {"too_big": (d.rate > 1.0)[..., None]},
}


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(_BAD_CONDITIONS))
def test_malformed_parameter_conditions_raise_type_error(case):
    """spec 4.7, rev 28: non-mapping results, keys that are empty, dotted or not strings, and non-bool or wrongly shaped values raise `TypeError`."""
    dist = _conditions_family(_BAD_CONDITIONS[case])(rate=f([0.5, 2.0, 3.0]))
    with pytest.raises(TypeError):
        dist.parameter_issues()


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(_BAD_CONDITIONS))
def test_malformed_parameter_conditions_raise_type_error_under_jit(case):
    """spec 4.7, rev 28: the check "uses shapes and dtypes only, so it also holds during tracing"."""
    dist = _conditions_family(_BAD_CONDITIONS[case])(rate=f([0.5, 2.0, 3.0]))
    with pytest.raises(TypeError):
        jax.jit(lambda d: d.parameter_issues())(dist)


@pytest.mark.step("A6")
def test_other_mapping_types_are_accepted_from_parameter_conditions():
    """spec 4.7, rev 28: `parameter_conditions()` may return "a `dict` (or other mapping)"."""
    cls = _conditions_family(
        lambda d: types.MappingProxyType({"too_big": d.rate > 1.0})
    )
    issues = cls(rate=f([0.5, 2.0])).parameter_issues()
    _assert_masks(_conditions(issues), {"too_big": [False, True]})


# ---------------------------------------------------------- raise_if_invalid


@pytest.mark.step("A6")
def test_invalid_parameter_error_is_a_public_value_error():
    """spec 4.7 and 10.2, rev 28: `survjax.InvalidParameterError` is exported and subclasses `ValueError`."""
    assert issubclass(survjax.InvalidParameterError, ValueError)


@pytest.mark.step("A6")
def test_raise_if_invalid_passes_without_violations():
    """spec 4.7, rev 28: `raise_if_invalid()` "returns normally when no violation is reported"."""
    _scaled(rate=[1.0, 1.5], cap=[2.0, 2.0], log_ratio=[0.0, 1.0]).parameter_issues().raise_if_invalid()


@pytest.mark.step("A6")
def test_raise_if_invalid_passes_with_nothing_declared():
    """spec 4.7, rev 28: undeclared NaN parameters report no violation, so the checkpoint passes."""
    fam.const_hazard()(rate=f([NAN, -1.0])).parameter_issues().raise_if_invalid()


@pytest.mark.step("A6")
def test_raise_if_invalid_raises_invalid_parameter_error_for_field_violation():
    """spec 4.7, rev 28: a violated declared constraint raises `survjax.InvalidParameterError`."""
    issues = _capped(rate=[1.0, -1.0], cap=[2.0, 2.0]).parameter_issues()
    with pytest.raises(survjax.InvalidParameterError):
        issues.raise_if_invalid()


@pytest.mark.step("A6")
def test_raise_if_invalid_raises_invalid_parameter_error_for_condition_violation():
    """spec 4.7, rev 28: a violated named condition raises `survjax.InvalidParameterError`."""
    issues = _capped(rate=[1.0, 3.0], cap=[2.0, 2.0]).parameter_issues()
    with pytest.raises(survjax.InvalidParameterError):
        issues.raise_if_invalid()


@pytest.mark.step("A6")
def test_raise_if_invalid_message_names_field_with_count_and_first_index():
    """spec 4.7, rev 28: the error names the violated fields "with counts and first offending batch indices"."""
    rate = [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 0.0]  # 2 violations, first at index 3
    issues = _capped(rate=rate, cap=[9.0] * 7).parameter_issues()
    with pytest.raises(survjax.InvalidParameterError) as caught:
        issues.raise_if_invalid()
    message = str(caught.value)
    assert "rate" in message
    assert re.search(r"(?<!\d)2(?!\d)", message)  # count
    assert re.search(r"(?<!\d)3(?!\d)", message)  # first offending index


@pytest.mark.step("A6")
def test_raise_if_invalid_message_names_condition_and_nested_field():
    """spec 4.7, rev 28: the error names violated conditions, and nested fields under their dotted keys."""
    issues = _scaled(
        rate=[1.0, 1.0, 1.0, 1.0, 1.0],
        cap=[2.0, 2.0, 2.0, 2.0, NAN],
        log_ratio=[0.0, 0.0, 5.0, 0.0, 0.0],
    ).parameter_issues()
    with pytest.raises(survjax.InvalidParameterError) as caught:
        issues.raise_if_invalid()
    message = str(caught.value)
    assert "ratio_too_large" in message
    assert "base.cap" in message


@pytest.mark.step("A6")
def test_raise_if_invalid_works_on_issues_returned_from_jit():
    """spec 4.7 and 6.2, rev 28: concrete diagnostics returned by compiled code can be checked on the host."""
    issues = jax.jit(lambda d: d.parameter_issues())(
        _capped(rate=[1.0, -1.0], cap=[2.0, 2.0])
    )
    with pytest.raises(survjax.InvalidParameterError):
        issues.raise_if_invalid()


@pytest.mark.step("A6")
@pytest.mark.parametrize("rate", [[1.0, 1.5], [1.0, -1.0]], ids=["valid", "invalid"])
def test_raise_if_invalid_on_traced_diagnostics_raises_plain_value_error(rate):
    """spec 4.7, rev 28: "On traced diagnostics it raises a plain `ValueError`", whatever the values."""

    def run(r):
        fam.capped_rate()(rate=r, cap=f([2.0, 2.0])).parameter_issues().raise_if_invalid()
        return r

    with pytest.raises(ValueError) as caught:
        jax.jit(run)(f(rate))
    assert not isinstance(caught.value, survjax.InvalidParameterError)


# ------------------------------------------------------- built-in declarations


@pytest.mark.step("A6")
def test_exponential_declares_rate_positive():
    """spec 4.1, rev 28: the built-in `rate` is `positive` (finite and `>0`)."""
    assert list(survjax.Exponential.parameter_constraints) == ["rate"]
    issues = survjax.Exponential(rate=f([1.0, 0.0, -1.0, NAN, INF])).parameter_issues()
    np.testing.assert_array_equal(
        _fields(issues)["rate"], [False, True, True, True, True]
    )
    np.testing.assert_array_equal(
        np.asarray(issues.invalid_entries), [False, True, True, True, True]
    )


@pytest.mark.step("A6")
def test_weibull_declares_shape_and_scale_positive():
    """spec 4.1, rev 28: the built-in Weibull `shape` and `scale` are `positive`."""
    assert set(survjax.Weibull.parameter_constraints) == {"shape", "scale"}
    issues = survjax.Weibull(
        shape=f([1.0, 0.0, 2.0, 2.0]), scale=f([1.0, 1.0, INF, 1.0])
    ).parameter_issues()
    fields = _fields(issues)
    np.testing.assert_array_equal(fields["shape"], [False, True, False, False])
    np.testing.assert_array_equal(fields["scale"], [False, False, True, False])
    np.testing.assert_array_equal(
        np.asarray(issues.invalid_entries), [False, True, True, False]
    )


@pytest.mark.step("A6")
def test_built_in_constraints_behave_as_positive():
    """spec 4.1, rev 28: each built-in declaration is `positive`: finite and `>0`."""
    x = f([1e-30, 1.0, 0.0, -1.0, NAN, INF])
    expected = [True, True, False, False, False, False]
    for cls in (survjax.Exponential, survjax.Weibull):
        for name, con in cls.parameter_constraints.items():
            np.testing.assert_array_equal(np.asarray(con(x)), expected, err_msg=name)


@pytest.mark.step("A6")
def test_from_log_overflow_is_reported_as_parameter_issue():
    """spec 4.1, rev 28: `from_log` stores natural parameters, "checked like any other"; `exp(1000)=inf` violates `positive`."""
    dist = survjax.Exponential.from_log(log_rate=f([0.0, 1000.0, -1000.0]))
    # exp(-1000) underflows to 0 in both dtypes: 0 is outside positive too
    np.testing.assert_array_equal(
        _fields(dist.parameter_issues())["rate"], [False, True, True]
    )


@pytest.mark.step("A6")
def test_weibull_from_log_overflow_is_reported_as_parameter_issue():
    """spec 4.1, rev 28: Weibull `from_log` natural parameters are checked; `exp(1000)=inf` violates `positive`."""
    dist = survjax.Weibull.from_log(log_shape=f([0.0, 0.0]), log_scale=f([0.0, 1000.0]))
    fields = _fields(dist.parameter_issues())
    np.testing.assert_array_equal(fields["shape"], [False, False])
    np.testing.assert_array_equal(fields["scale"], [False, True])


# --------------------------------------------- direct methods are unaffected


_POINTWISE = (
    "log_survival",
    "survival",
    "log_prob",
    "log_cdf",
    "cumulative_hazard",
    "log_hazard",
    "hazard",
)


@pytest.mark.step("A6")
@pytest.mark.parametrize("method", _POINTWISE)
def test_declared_violation_does_not_change_direct_method_values(method):
    """spec 4.5, rev 28: declared constraints "do not change direct-method values".

    The constrained and undeclared twins share one formula, so every pointwise
    method gives identical results (NaN positions included) on valid and
    invalid rates: ``-1`` (finite formula, ``log S = +t``), ``0``, NaN, ``inf``.
    """
    rate = f([0.5, -1.0, 0.0, NAN, INF])
    t = f([2.0, 2.0, 2.0, 2.0, 2.0])
    constrained = getattr(fam.constrained_rate()(rate=rate), method)(t)
    undeclared = getattr(fam.unconstrained_rate()(rate=rate), method)(t)
    np.testing.assert_array_equal(np.asarray(constrained), np.asarray(undeclared))


@pytest.mark.step("A6")
def test_declared_violation_keeps_finite_formula_value():
    """spec 4.5, rev 28: with `rate=-1` the formula `log S = -rate*t = 2` is returned, not NaN."""
    value = fam.constrained_rate()(rate=f(-1.0)).log_survival(f(2.0))
    np.testing.assert_array_equal(np.asarray(value), f(2.0))


@pytest.mark.step("A6")
def test_condition_violation_does_not_change_direct_method_values():
    """spec 4.5, rev 28: a violated named condition leaves direct-method values as the formula gives them."""
    dist = _capped(rate=[3.0], cap=[2.0])  # rate_above_cap violated
    # log S = -3 * 2 = -6; log f = log 3 - 6
    assert_close(dist.log_survival(f([2.0])), [-6.0])
    assert_close(dist.log_prob(f([2.0])), [np.log(3.0) - 6.0])
