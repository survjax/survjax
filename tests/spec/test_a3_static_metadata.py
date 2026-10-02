"""A3: static "may contain" metadata and compile counts (spec 3.2).

The flags live in PyTree auxiliary data, which has no public accessor, so the
tests observe them through tree structures and jit retracing: datasets with
different flags have different tree structures and compile separately.
"""

import jax
import pytest
from helpers import INF, NAN, b, f, treedef

import survjax


def _exact_only(times=(1.0, 2.0, 3.0, 4.0)):
    return survjax.SurvivalData(f(list(times)))


def _exact_and(kind):
    """Four rows: three exact and one row of the given kind (no truncation)."""
    flags = b([False, False, False, True])
    if kind == "right":
        return survjax.SurvivalData(f([1.0, 2.0, 3.0, 4.0]), right_censored=flags)
    if kind == "left":
        return survjax.SurvivalData(f([1.0, 2.0, 3.0, 4.0]), left_censored=flags)
    return survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 4.0]), interval_censored=flags, time2=f([NAN, NAN, NAN, 5.0])
    )


def _full_inputs():
    """Every kind plus left and right truncation, all rows valid.

    exact 1 in (0.5, 10]; right at 2 in (0.5, inf]; left at 3 in (0, 10];
    interval (1, 4] in (0, inf].
    """
    return {
        "time": f([1.0, 2.0, 3.0, 1.0]),
        "right_censored": b([False, True, False, False]),
        "left_censored": b([False, False, True, False]),
        "interval_censored": b([False, False, False, True]),
        "time2": f([NAN, NAN, NAN, 4.0]),
        "left_trunc": f([0.5, 0.5, 0.0, 0.0]),
        "right_trunc": f([10.0, INF, 10.0, INF]),
    }


def _full():
    kw = _full_inputs()
    return survjax.SurvivalData(kw.pop("time"), **kw)


def _traced(**kw):
    """Build data under jit from the given flag-constructor inputs, passed as traced jit arguments."""
    names = list(kw)

    def build(*values):
        args = dict(zip(names, values, strict=True))
        return survjax.SurvivalData(args.pop("time"), **args)

    return jax.jit(build)(*kw.values())


# ------------------------------------------------------- eager flags


@pytest.mark.step("A3")
def test_same_kinds_with_different_values_share_tree_structure():
    """spec 3.2, rev 4: flags record which kinds are present, not the values."""
    assert treedef(_exact_only((1.0, 2.0, 3.0, 4.0))) == treedef(
        _exact_only((5.0, 6.0, 7.0, 8.0))
    )


@pytest.mark.step("A3")
def test_adding_a_right_censored_row_changes_tree_structure():
    """spec 3.2, rev 4: auxiliary data records which derived kinds are present (right)."""
    assert treedef(_exact_and("right")) != treedef(_exact_only())


@pytest.mark.step("A3")
def test_adding_a_left_censored_row_changes_tree_structure():
    """spec 3.2, rev 4: auxiliary data records which derived kinds are present (left)."""
    assert treedef(_exact_and("left")) != treedef(_exact_only())


@pytest.mark.step("A3")
def test_adding_an_interval_row_changes_tree_structure():
    """spec 3.2, rev 4: auxiliary data records which derived kinds are present (interval)."""
    assert treedef(_exact_and("interval")) != treedef(_exact_only())


@pytest.mark.step("A3")
def test_each_censored_kind_gives_a_distinct_tree_structure():
    """spec 3.2, rev 4: right, left and interval are separate derived kinds."""
    structures = [treedef(_exact_and(kind)) for kind in ("right", "left", "interval")]
    assert structures[0] != structures[1]
    assert structures[0] != structures[2]
    assert structures[1] != structures[2]


@pytest.mark.step("A3")
def test_left_truncation_changes_tree_structure():
    """spec 3.2, rev 7: auxiliary data records whether any left truncation is present (any `left_trunc > 0`)."""
    truncated = survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 4.0]), left_trunc=f([0.0, 0.5, 0.0, 0.0])
    )
    assert treedef(truncated) != treedef(_exact_only())


@pytest.mark.step("A3")
def test_right_truncation_changes_tree_structure():
    """spec 3.2, rev 7: auxiliary data records whether any right truncation is present (any `right_trunc < +inf`)."""
    truncated = survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 4.0]), right_trunc=f([INF, 5.0, INF, INF])
    )
    assert treedef(truncated) != treedef(_exact_only())


@pytest.mark.step("A3")
def test_left_and_right_truncation_give_different_tree_structures():
    """spec 3.2, rev 7: left and right truncation are separate flags."""
    left_only = survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 4.0]), left_trunc=f([0.0, 0.5, 0.0, 0.0])
    )
    right_only = survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 4.0]), right_trunc=f([INF, 5.0, INF, INF])
    )
    assert treedef(left_only) != treedef(right_only)


@pytest.mark.step("A3")
def test_equal_flags_with_different_row_counts_share_tree_structure():
    """spec 3.2, rev 7: the auxiliary data holds only the flags, so equal flags give equal tree structures."""
    assert treedef(_exact_only((1.0, 2.0))) == treedef(
        _exact_only((1.0, 2.0, 3.0, 4.0, 5.0))
    )
    assert treedef(_exact_and("right")) == treedef(
        survjax.SurvivalData(f([1.0, 2.0]), right_censored=b([False, True]))
    )


@pytest.mark.step("A3")
def test_explicit_zero_and_infinite_truncation_give_the_same_flags_as_absent():
    """spec 3.2, rev 6: flags come from normalized fields; explicit L=0 and U=+inf normalize like absent truncation."""
    explicit = survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 4.0]), left_trunc=f([0.0] * 4), right_trunc=f([INF] * 4)
    )
    assert treedef(explicit) == treedef(_exact_only())


@pytest.mark.step("A3")
def test_both_constructors_give_the_same_tree_structure():
    """spec 3.2/12.1, rev 4: both constructors normalize identically, including the static metadata."""
    flags = survjax.SurvivalData(f([1.0, 2.0, 3.0]), b([True, False, True]))
    bounds = survjax.SurvivalData.from_bounds(
        f([1.0, 2.0, 3.0]), f([1.0, INF, 3.0]), exact=b([True, False, True])
    )
    assert treedef(flags) == treedef(bounds)


@pytest.mark.step("A3")
def test_eager_flag_constructor_computes_flags_from_values():
    """spec 3.2, rev 6: with no traced observation input, flags are computed from the normalized fields, by either route."""
    eager = survjax.SurvivalData(f([1.0, 2.0, 3.0, 4.0]))
    assert treedef(eager) == treedef(_exact_only())


@pytest.mark.step("A3")
def test_eager_bounds_constructor_computes_flags_from_values():
    """spec 3.2, rev 6: the bounds constructor with non-traced inputs computes flags from values."""
    eager = survjax.SurvivalData.from_bounds(
        f([1.0, 2.0, 3.0, 4.0]), f([NAN, NAN, NAN, NAN]), exact=b([True] * 4)
    )
    assert treedef(eager) == treedef(_exact_only())


@pytest.mark.step("A3")
def test_flags_include_rows_with_issue_bits():
    """spec 3.2, rev 6: flags are computed from the normalized fields of all rows, including rows with issue bits.

    Row 3 is right-censored at c=U=5 (empty intersection, invalid). Its inputs
    are individually valid; only its relation to U breaks a rule. Spec 3.2 says
    "There are no placeholder values: every field is meaningful on every row",
    and its normalization table gives a right-censored row only one meaning
    (lower=c, upper=+inf). So row 3 normalizes as right censoring with U=5, and
    the flags match a valid dataset with a right-censored row and right
    truncation.
    """
    invalid = survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 5.0]),
        right_censored=b([False, False, False, True]),
        right_trunc=f([INF, INF, INF, 5.0]),
    )
    assert not bool(invalid.issues.ok)
    valid = survjax.SurvivalData(
        f([1.0, 2.0, 3.0, 4.0]),
        right_censored=b([False, False, False, True]),
        right_trunc=f([INF, INF, INF, 5.0]),
    )
    assert treedef(invalid) == treedef(valid)


# ------------------------------------------- non-traced construction inside jit


@pytest.mark.step("A3")
def test_ordinary_construction_inside_jit_computes_flags_from_values():
    """spec 3.2, rev 6: with no traced observation input, flags are computed from values even inside a transformed function."""
    inside = jax.jit(lambda: survjax.SurvivalData(f([1.0, 2.0, 3.0, 4.0])))()
    assert treedef(inside) == treedef(_exact_only())
    assert treedef(inside) != treedef(_full())


@pytest.mark.step("A3")
def test_construction_from_concrete_inputs_inside_jit_computes_flags_from_values():
    """spec 3.2, rev 6: constructors classify inputs as in 6.2; non-traced inputs inside jit give value-based flags."""
    inside = jax.jit(lambda: survjax.SurvivalData(f([1.0, 2.0, 3.0, 4.0])))()
    assert treedef(inside) == treedef(_exact_only())


@pytest.mark.step("A3")
def test_ordinary_construction_inside_jit_with_other_traced_args_computes_flags_from_values():
    """spec 3.2, rev 6: traced arguments that are not observation inputs do not force all flags true."""
    structures = []

    def fn(scale):
        data = survjax.SurvivalData(f([1.0, 2.0, 3.0, 4.0]))
        structures.append(treedef(data))
        return data.lower * scale

    jax.jit(fn)(f(2.0))
    assert structures == [treedef(_exact_only())]


# ------------------------------------------------------- slices inherit


@pytest.mark.step("A3")
def test_slice_inherits_parent_flags():
    """spec 3.2, rev 4: slices inherit flags unchanged, even when the slice lacks a parent kind."""
    parent = _exact_and("right")
    exact_rows = parent[0:3]  # three exact rows only
    assert treedef(exact_rows) == treedef(parent)
    assert treedef(exact_rows) != treedef(_exact_only((1.0, 2.0, 3.0)))


@pytest.mark.step("A3")
def test_integer_index_inherits_parent_flags():
    """spec 3.2, rev 4: a scalar observation taken from a parent keeps the parent's flags."""
    parent = _full()
    assert treedef(parent[0]) == treedef(parent)


# ------------------------------------------------------- traced flags


@pytest.mark.step("A3")
def test_single_traced_observation_input_sets_all_flags():
    """spec 3.2, rev 6: any one traced observation input (here only `left_trunc`) makes all flags true."""
    data = jax.jit(
        lambda lt: survjax.SurvivalData(f([1.0, 2.0, 3.0, 4.0]), left_trunc=lt)
    )(f([0.0, 0.0, 0.0, 0.0]))
    assert treedef(data) == treedef(_full())


@pytest.mark.step("A3")
def test_traced_bounds_construction_sets_all_flags():
    """spec 3.2, rev 6: the bounds constructor with a traced observation input gives all flags true."""
    data = jax.jit(
        lambda lo: survjax.SurvivalData.from_bounds(
            lo, f([1.0, 2.0]), exact=b([True, True])
        )
    )(f([1.0, 2.0]))
    assert treedef(data) == treedef(_full())


@pytest.mark.step("A3")
def test_traced_construction_sets_all_flags():
    """spec 3.2, rev 9: when any observation input is traced (ordinary constructor), all flags are true, whatever the values."""
    full = _full()
    traced_exact_only = _traced(time=f([1.0, 2.0, 3.0, 4.0]))
    assert treedef(traced_exact_only) == treedef(full)


@pytest.mark.step("A3")
def test_traced_construction_of_full_data_matches_eager_structure():
    """spec 3.2, rev 6: eager data containing every kind and both truncations has all flags true, as traced data does."""
    assert treedef(_traced(**_full_inputs())) == treedef(_full())


# ------------------------------------------------------- compile counts


def _counting_jit():
    traces = []

    @jax.jit
    def total(data):
        traces.append(None)
        return data.upper.sum()

    return total, traces


@pytest.mark.step("A3")
def test_same_flags_compile_once():
    """spec 3.2, rev 4: datasets with the same flags, shapes and dtypes share one compilation."""
    total, traces = _counting_jit()
    total(_exact_only((1.0, 2.0, 3.0, 4.0)))
    total(_exact_only((5.0, 6.0, 7.0, 8.0)))
    assert len(traces) == 1


@pytest.mark.step("A3")
def test_different_flags_compile_separately():
    """spec 3.2, rev 4: datasets with different flags have different tree structures and compile separately."""
    total, traces = _counting_jit()
    total(_exact_only())
    total(_exact_and("right"))
    assert len(traces) == 2


@pytest.mark.step("A3")
def test_traced_and_eager_all_flag_data_share_one_compilation():
    """spec 3.2, rev 6: traced-built data (all flags true) and eager data with every kind share one compilation."""
    total, traces = _counting_jit()
    total(_full())
    total(_traced(time=f([1.0, 2.0, 3.0, 4.0])))
    assert len(traces) == 1


@pytest.mark.step("A3")
def test_slice_reuses_parent_flags_for_compilation():
    """spec 3.2, rev 4: equal-shape slices of one parent keep its flags, so they share one compilation."""
    parent = _full()
    total, traces = _counting_jit()
    total(parent[0:2])  # exact and right rows
    total(parent[2:4])  # left and interval rows
    assert len(traces) == 1
