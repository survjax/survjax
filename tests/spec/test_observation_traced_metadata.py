"""observation.traced_metadata: static metadata of traced construction follow the argument form (spec 3.2, 6.1, 6.2, 7.2, 12.1, 12.2; rev 62).

3.2 (rev 62): "When any observation input is traced, the flags follow the
constructor's argument form: only whether each observation argument is `None`
or supplied counts, never its values (traced or concrete). A flag is false
only in these cases; otherwise it is conservatively true:

- Flag form: with no `event` and no censoring flag, only the exact kind flag
  is true. With `event`, or with `right_censored` as the only censoring flag
  supplied, the left- and interval-censored flags are false. Supplying
  `left_censored` or `interval_censored` makes all four kind flags true.
- Bounds form: with `exact` omitted, the exact flag is false; the three
  censored kind flags are always true.
- Both forms: the left-truncation flag is false when `left_trunc` is omitted,
  and the right-truncation flag is false when `right_trunc` is omitted."

The flags have no public accessor (Q-014), so every test compares the tree
structure of traced-built data with that of an eagerly built *concrete*
dataset whose value-derived flags are the expected ones ("datasets with equal
flags have equal tree structures"), or counts `jit` traces of a consumer.
The concrete references (``_expected``) hold one valid row per expected kind:
exact ``t=1``; right ``(2, +inf]``; left ``(0, 3]``; interval ``(1, 4]``;
left truncation ``L=0.5`` and right truncation ``U=10`` on every row when
expected. Kinds are those of 3.2 ("Right takes precedence ... left-censored
rows are the remaining non-exact rows with `lower=0`").

The raw form ``dist.log_likelihood(time=..., event=...)`` "equals
`SurvivalData(time, event)`" (3.2, 12.1); its internal data are not public,
so its metadata cannot be observed and only values and derivatives are
compared.
"""

import jax
import numpy as np
import pytest
from helpers import INF, NAN, assert_close, b, f, flag_data, treedef

import survjax

FEATURE = pytest.mark.feature("observation.traced_metadata")

ALL_KINDS = ("exact", "right", "left", "interval")
EXACT_RIGHT = ("exact", "right")
CENSORED = ("right", "left", "interval")

# One valid row per kind: (time, time2) for helpers.flag_data.
_ROWS = {
    "exact": (1.0, NAN),
    "right": (2.0, NAN),
    "left": (3.0, NAN),
    "interval": (1.0, 4.0),
}

# Four observation rows used as traced inputs.
TIME = [1.0, 2.0, 0.5, 3.0]
EVENT = [True, False, True, False]


def _expected(*kinds, left_trunc=False, right_trunc=False):
    """Tree structure of concrete eager data whose value-derived flags are exactly ``kinds`` and the named truncations."""
    n = len(kinds)
    trunc = {}
    if left_trunc:
        trunc["left_trunc"] = [0.5] * n
    if right_trunc:
        trunc["right_trunc"] = [10.0] * n
    return treedef(
        flag_data(
            kinds, [_ROWS[k][0] for k in kinds], [_ROWS[k][1] for k in kinds], **trunc
        )
    )


def _flag(traced, concrete=None):
    """Flag-form ``SurvivalData`` built inside ``jit``.

    ``traced`` entries are ``jit`` arguments (tracers inside); ``concrete``
    entries are closed over (non-traced, spec 6.2), ``None`` included. ``time``
    and ``event`` are passed positionally (3.1: "The second positional
    argument is the event indicator"), the rest by keyword.
    """
    concrete = dict(concrete or {})
    names = list(traced)

    def build(*values):
        kw = {**concrete, **dict(zip(names, values, strict=True))}
        time = kw.pop("time")
        if "event" in kw:
            return survjax.SurvivalData(time, kw.pop("event"), **kw)
        return survjax.SurvivalData(time, **kw)

    return jax.jit(build)(*traced.values())


def _bounds(traced, concrete=None):
    """Bounds-form ``SurvivalData.from_bounds`` built inside ``jit``; ``lower`` and ``upper`` positional."""
    concrete = dict(concrete or {})
    names = list(traced)

    def build(*values):
        kw = {**concrete, **dict(zip(names, values, strict=True))}
        return survjax.SurvivalData.from_bounds(kw.pop("lower"), kw.pop("upper"), **kw)

    return jax.jit(build)(*traced.values())


def _counting_jit():
    traces = []

    @jax.jit
    def total(data):
        traces.append(None)
        return data.upper.sum()

    return total, traces


# ------------------------------------------------- flag form: kind flags


@FEATURE
def test_time_only_gives_only_the_exact_kind():
    """spec 3.2, rev 62: "with no `event` and no censoring flag, only the exact kind flag is true"; no truncation supplied."""
    assert treedef(_flag({"time": f(TIME)})) == _expected("exact")


@FEATURE
def test_time_with_unread_time2_gives_only_the_exact_kind():
    """spec 3.1/3.2, rev 62: `time2` is not a censoring flag; with no indicator every row is exact and `time2` is ignored."""
    data = _flag({"time": f(TIME), "time2": f([5.0, 5.0, 5.0, 5.0])})
    assert treedef(data) == _expected("exact")


@FEATURE
def test_list_containing_a_tracer_counts_as_traced():
    """spec 3.2/6.2, rev 62: a list holding a tracer is traced, so `SurvivalData([1.0, x])` has only the exact kind."""
    data = jax.jit(lambda x: survjax.SurvivalData([1.0, x]))(f(2.0))
    assert treedef(data) == _expected("exact")


@FEATURE
def test_event_gives_exact_and_right_kinds():
    """spec 3.2, rev 62: "With `event` ... the left- and interval-censored flags are false"."""
    assert treedef(_flag({"time": f(TIME), "event": b(EVENT)})) == _expected(
        *EXACT_RIGHT
    )


@FEATURE
def test_event_of_all_ones_still_has_the_right_kind():
    """spec 3.2, rev 62: values never count: a traced `event` of all ones keeps the right flag."""
    data = _flag({"time": f(TIME), "event": b([True] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT)


@FEATURE
def test_event_of_all_zeros_still_has_the_exact_kind():
    """spec 3.2, rev 62: values never count: a traced `event` of all zeros keeps the exact flag."""
    data = _flag({"time": f(TIME), "event": b([False] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT)


@FEATURE
def test_numeric_event_codes_follow_the_form():
    """spec 3.1/3.2, rev 62: numeric 0/1 event codes are the same argument form as booleans."""
    data = _flag({"time": f(TIME), "event": np.array([1, 1, 1, 1], dtype=np.int32)})
    assert treedef(data) == _expected(*EXACT_RIGHT)


@FEATURE
def test_concrete_event_beside_traced_time_follows_the_form():
    """spec 3.2, rev 62: once any input is traced, a concrete `event` of all ones still counts only as supplied."""
    data = _flag({"time": f(TIME)}, {"event": b([True] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT)


@FEATURE
def test_right_flag_alone_gives_exact_and_right_kinds():
    """spec 3.2, rev 62: "with `right_censored` as the only censoring flag supplied, the left- and interval-censored flags are false"."""
    data = _flag({"time": f(TIME), "right_censored": b([False, True, False, True])})
    assert treedef(data) == _expected(*EXACT_RIGHT)


@FEATURE
def test_right_flag_of_all_false_still_has_the_right_kind():
    """spec 3.2, rev 62: a supplied `right_censored` of all false keeps the right flag."""
    data = _flag({"time": f(TIME), "right_censored": b([False] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT)


@FEATURE
def test_explicit_none_indicators_count_as_omitted():
    """spec 3.2, rev 62: "only whether each observation argument is `None` or supplied counts": explicit `None` flags are omitted ones."""
    data = _flag(
        {"time": f(TIME), "right_censored": b([False, True, False, True])},
        {"event": None, "left_censored": None, "interval_censored": None},
    )
    assert treedef(data) == _expected(*EXACT_RIGHT)


@FEATURE
def test_explicit_none_everything_gives_only_the_exact_kind():
    """spec 3.2, rev 62: `event`, every censoring flag and `time2` passed as `None` are the no-indicator form."""
    data = _flag(
        {"time": f(TIME)},
        {
            "event": None,
            "right_censored": None,
            "left_censored": None,
            "interval_censored": None,
            "time2": None,
        },
    )
    assert treedef(data) == _expected("exact")


@FEATURE
def test_left_flag_makes_all_four_kinds():
    """spec 3.2, rev 62: "Supplying `left_censored` ... makes all four kind flags true", even when it is all false."""
    data = _flag({"time": f(TIME), "left_censored": b([False] * 4)})
    assert treedef(data) == _expected(*ALL_KINDS)


@FEATURE
def test_interval_flag_makes_all_four_kinds():
    """spec 3.2, rev 62: "Supplying ... `interval_censored` makes all four kind flags true", even when it is all false."""
    data = _flag(
        {"time": f(TIME), "interval_censored": b([False] * 4), "time2": f([NAN] * 4)}
    )
    assert treedef(data) == _expected(*ALL_KINDS)


@FEATURE
def test_right_and_left_flags_make_all_four_kinds():
    """spec 3.2, rev 62: `right_censored` is not the only censoring flag once `left_censored` is supplied too."""
    data = _flag(
        {
            "time": f(TIME),
            "right_censored": b([False, True, False, False]),
            "left_censored": b([False, False, False, False]),
        }
    )
    assert treedef(data) == _expected(*ALL_KINDS)


# ------------------------------------------------- flag form: truncation flags


@FEATURE
def test_traced_left_trunc_alone_gives_exact_kind_and_left_truncation():
    """spec 3.2, rev 62: a traced `left_trunc` (concrete `time`) makes the data traced; no indicator, `right_trunc` omitted."""
    data = _flag({"left_trunc": f([0.0, 0.5, 0.0, 0.0])}, {"time": f(TIME)})
    assert treedef(data) == _expected("exact", left_trunc=True)


@FEATURE
def test_left_trunc_supplied_sets_only_left_truncation():
    """spec 3.2, rev 62: "the right-truncation flag is false when `right_trunc` is omitted"."""
    data = _flag({"time": f(TIME), "event": b(EVENT), "left_trunc": f([0.2] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT, left_trunc=True)


@FEATURE
def test_right_trunc_supplied_sets_only_right_truncation():
    """spec 3.2, rev 62: "the left-truncation flag is false when `left_trunc` is omitted"."""
    data = _flag({"time": f(TIME), "event": b(EVENT), "right_trunc": f([5.0] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT, right_trunc=True)


@FEATURE
def test_both_truncation_bounds_supplied_set_both_flags():
    """spec 3.2, rev 62: both bounds supplied keep both truncation flags."""
    data = _flag(
        {
            "time": f(TIME),
            "event": b(EVENT),
            "left_trunc": f([0.2] * 4),
            "right_trunc": f([5.0] * 4),
        }
    )
    assert treedef(data) == _expected(*EXACT_RIGHT, left_trunc=True, right_trunc=True)


@FEATURE
def test_traced_zero_left_trunc_still_has_left_truncation():
    """spec 3.2, rev 62: values never count: a traced `left_trunc` of zeros keeps the left-truncation flag."""
    data = _flag({"time": f(TIME), "event": b(EVENT), "left_trunc": f([0.0] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT, left_trunc=True)


@FEATURE
def test_concrete_zero_left_trunc_beside_traced_input_still_has_left_truncation():
    """spec 3.2, rev 62: "never its values (traced or concrete)": concrete zeros alongside a traced input keep left truncation."""
    data = _flag({"time": f(TIME), "event": b(EVENT)}, {"left_trunc": f([0.0] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT, left_trunc=True)


@FEATURE
def test_concrete_infinite_right_trunc_beside_traced_input_still_has_right_truncation():
    """spec 3.2, rev 62: a concrete `right_trunc` of `+inf` alongside a traced input keeps the right-truncation flag."""
    data = _flag({"time": f(TIME), "event": b(EVENT)}, {"right_trunc": f([INF] * 4)})
    assert treedef(data) == _expected(*EXACT_RIGHT, right_trunc=True)


@FEATURE
def test_scalar_default_truncation_values_still_count_as_supplied():
    """spec 3.2, rev 62: Python scalars `left_trunc=0.0`, `right_trunc=inf` are supplied arguments, not omitted ones."""
    data = _flag(
        {"time": f(TIME), "event": b(EVENT)}, {"left_trunc": 0.0, "right_trunc": INF}
    )
    assert treedef(data) == _expected(*EXACT_RIGHT, left_trunc=True, right_trunc=True)


@FEATURE
def test_explicit_none_truncation_counts_as_omitted():
    """spec 3.2, rev 62: `left_trunc=None` and `right_trunc=None` are omitted bounds."""
    data = _flag(
        {"time": f(TIME), "event": b(EVENT)}, {"left_trunc": None, "right_trunc": None}
    )
    assert treedef(data) == _expected(*EXACT_RIGHT)


# ------------------------------------------------- bounds form


@FEATURE
def test_bounds_without_exact_have_the_three_censored_kinds_only():
    """spec 3.2, rev 62: "with `exact` omitted, the exact flag is false; the three censored kind flags are always true".

    The rows are all finite intervals, so values alone would give only the
    interval kind; the form still gives right and left.
    """
    data = _bounds({"lower": f([1.0, 0.5, 2.0, 1.0]), "upper": f([2.0, 3.0, 4.0, 5.0])})
    assert treedef(data) == _expected(*CENSORED)


@FEATURE
def test_bounds_with_exact_have_all_four_kinds():
    """spec 3.2, rev 62: with `exact` supplied, even all true, the exact flag and the three censored flags are true."""
    data = _bounds(
        {
            "lower": f([1.0, 2.0, 3.0, 4.0]),
            "upper": f([1.0, 2.0, 3.0, 4.0]),
            "exact": b([True] * 4),
        }
    )
    assert treedef(data) == _expected(*ALL_KINDS)


@FEATURE
def test_bounds_with_exact_of_all_false_still_have_the_exact_kind():
    """spec 3.2, rev 62: a supplied `exact` of all false keeps the exact flag (only omission clears it)."""
    data = _bounds(
        {
            "lower": f([1.0, 2.0, 0.0, 1.0]),
            "upper": f([INF, 3.0, 2.0, 4.0]),
            "exact": b([False] * 4),
        }
    )
    assert treedef(data) == _expected(*ALL_KINDS)


@FEATURE
def test_bounds_with_explicit_none_exact_have_no_exact_kind():
    """spec 3.2, rev 62: `exact=None` counts as omitted."""
    data = _bounds({"lower": f([1.0, 0.0]), "upper": f([INF, 2.0])}, {"exact": None})
    assert treedef(data) == _expected(*CENSORED)


@FEATURE
def test_bounds_concrete_exact_beside_traced_lower_follows_the_form():
    """spec 3.2, rev 62: a concrete `exact` alongside a traced `lower` counts only as supplied."""
    data = _bounds(
        {"lower": f([1.0, 2.0])}, {"upper": f([1.0, 2.0]), "exact": b([True, True])}
    )
    assert treedef(data) == _expected(*ALL_KINDS)


@FEATURE
def test_bounds_traced_truncation_alone_follows_the_form():
    """spec 3.2, rev 62: with only `left_trunc` traced, concrete right-censored bounds still give all three censored kinds and no exact."""
    data = _bounds(
        {"left_trunc": f([0.5, 0.5])}, {"lower": f([1.0, 2.0]), "upper": f([INF, INF])}
    )
    assert treedef(data) == _expected(*CENSORED, left_trunc=True)


@FEATURE
def test_bounds_left_trunc_supplied_sets_only_left_truncation():
    """spec 3.2, rev 62: bounds form, `left_trunc` supplied, `right_trunc` omitted."""
    data = _bounds(
        {
            "lower": f([1.0, 1.0]),
            "upper": f([1.0, INF]),
            "exact": b([True, False]),
            "left_trunc": f([0.5, 0.5]),
        }
    )
    assert treedef(data) == _expected(*ALL_KINDS, left_trunc=True)


@FEATURE
def test_bounds_right_trunc_supplied_sets_only_right_truncation():
    """spec 3.2, rev 62: bounds form, `right_trunc` supplied, `left_trunc` omitted."""
    data = _bounds(
        {
            "lower": f([1.0, 1.0]),
            "upper": f([1.0, 3.0]),
            "exact": b([True, False]),
            "right_trunc": f([5.0, 5.0]),
        }
    )
    assert treedef(data) == _expected(*ALL_KINDS, right_trunc=True)


@FEATURE
def test_bounds_concrete_default_truncation_values_still_count_as_supplied():
    """spec 3.2, rev 62: concrete `left_trunc` zeros and `right_trunc` `+inf` alongside traced bounds keep both truncation flags."""
    data = _bounds(
        {"lower": f([1.0, 0.0]), "upper": f([INF, 2.0])},
        {"left_trunc": f([0.0, 0.0]), "right_trunc": f([INF, INF])},
    )
    assert treedef(data) == _expected(*CENSORED, left_trunc=True, right_trunc=True)


@FEATURE
def test_flag_and_bounds_forms_with_equal_rule_flags_share_a_tree_structure():
    """spec 3.2, rev 62: flag form with `left_censored` and bounds form with `exact` both give all four kinds, so equal structures."""
    flags = _flag({"time": f([1.0, 2.0]), "left_censored": b([False, True])})
    bounds = _bounds(
        {"lower": f([1.0, 0.0]), "upper": f([1.0, 2.0]), "exact": b([True, False])}
    )
    assert treedef(flags) == treedef(bounds)


# ------------------------------------------------- rows with issue bits


@FEATURE
def test_invalid_event_rows_do_not_change_the_form_flags():
    """spec 3.2, rev 62: "These rules hold for every row, including rows with issue bits".

    Negative and NaN times, event code 2 and a NaN event are invalid; the
    flags are still those of the `event` form.
    """
    data = _flag({"time": f([1.0, -1.0, NAN, 2.0]), "event": f([1.0, 2.0, NAN, 0.0])})
    assert not bool(np.asarray(data.issues.ok))
    assert treedef(data) == _expected(*EXACT_RIGHT)


@FEATURE
def test_invalid_bounds_rows_do_not_change_the_form_flags():
    """spec 3.2, rev 62: reversed, negative and equal-endpoint non-exact bounds still give the three censored kinds only."""
    data = _bounds(
        {"lower": f([3.0, -1.0, 2.0, 1.0]), "upper": f([2.0, 1.0, 2.0, NAN])}
    )
    assert not bool(np.asarray(data.issues.ok))
    assert treedef(data) == _expected(*CENSORED)


@FEATURE
def test_invalid_interval_rows_do_not_change_the_form_flags():
    """spec 3.2, rev 62: an interval row with `time2 <= time` and conflicting flags still give the left/interval form's four kinds."""
    data = _flag(
        {
            "time": f([3.0, 2.0]),
            "interval_censored": b([True, True]),
            "left_censored": b([False, True]),
            "time2": f([1.0, 4.0]),
        }
    )
    assert not bool(np.asarray(data.issues.ok))
    assert treedef(data) == _expected(*ALL_KINDS)


@FEATURE
def test_invalid_left_trunc_sets_only_the_supplied_truncation_flag():
    """spec 3.2, rev 62: negative and NaN `left_trunc` are invalid; the left flag is set by supply, the right flag stays false."""
    data = _flag({"time": f(TIME), "left_trunc": f([-1.0, NAN, 0.0, 0.0])})
    assert not bool(np.asarray(data.issues.ok))
    assert treedef(data) == _expected("exact", left_trunc=True)


# ------------------------------------------------- slices, vmap


@FEATURE
def test_slice_of_traced_data_inherits_form_flags():
    """spec 3.2, rev 62: slices inherit the form-derived flags; a slice of all-exact rows keeps the right flag."""
    data = _flag({"time": f(TIME), "event": b([True] * 4)})
    sub = data[0:2]
    assert treedef(sub) == treedef(data)
    assert treedef(sub) == _expected(*EXACT_RIGHT)


@FEATURE
def test_integer_index_of_traced_bounds_data_inherits_form_flags():
    """spec 3.2, rev 62: a scalar observation taken from bounds-form traced data keeps no exact kind and the three censored kinds."""
    data = _bounds({"lower": f([1.0, 0.0]), "upper": f([INF, 2.0])})
    assert treedef(data[0]) == _expected(*CENSORED)


@FEATURE
def test_slice_inside_jit_inherits_form_flags():
    """spec 3.2 and 7.2, rev 62: slicing inside the compiled function keeps the parent's form-derived flags."""
    structures = []

    def fn(time, event, left_trunc):
        sub = survjax.SurvivalData(time, event, left_trunc=left_trunc)[1:3]
        structures.append(treedef(sub))
        return sub.lower

    jax.jit(fn)(f(TIME), b(EVENT), f([0.0] * 4))
    assert structures == [_expected(*EXACT_RIGHT, left_trunc=True)]


@FEATURE
def test_construction_under_vmap_follows_the_form():
    """spec 3.2/6.2, rev 62: inputs mapped by `vmap` are tracers, so batched construction carries the `event` form's flags."""
    data = jax.vmap(lambda t, e: survjax.SurvivalData(t, e))(
        f([TIME, TIME]), b([[True] * 4, [True] * 4])
    )
    assert treedef(data) == _expected(*EXACT_RIGHT)


# ------------------------------------------------- compile counts


@FEATURE
def test_equal_argument_forms_with_different_values_share_one_compilation():
    """spec 3.2 and 7.2, rev 62: traced data of one argument form have equal flags whatever the values, so one compilation."""
    total, traces = _counting_jit()
    total(_flag({"time": f(TIME), "event": b([True] * 4)}))
    total(_flag({"time": f([4.0, 3.0, 2.0, 1.0]), "event": b([False] * 4)}))
    assert len(traces) == 1


@FEATURE
def test_traced_and_concrete_data_with_equal_flags_share_one_compilation():
    """spec 3.2 and 7.2, rev 62: traced `event`-form data and concrete exact/right data have equal flags and share one compilation."""
    total, traces = _counting_jit()
    total(_flag({"time": f(TIME), "event": b(EVENT)}))
    total(survjax.SurvivalData(f([1.0, 2.0]), b([True, False])))
    assert len(traces) == 1


@FEATURE
def test_traced_time_only_shares_compilation_with_concrete_exact_data():
    """spec 3.2 and 7.2, rev 62: traced data without an indicator and concrete exact-only data share one compilation."""
    total, traces = _counting_jit()
    total(_flag({"time": f(TIME)}))
    total(survjax.SurvivalData(f([1.0, 2.0, 3.0, 4.0])))
    assert len(traces) == 1


@FEATURE
def test_different_argument_forms_compile_separately():
    """spec 3.2 and 7.2, rev 62: "datasets with different flags have different tree structures and compile separately"."""
    total, traces = _counting_jit()
    total(_flag({"time": f(TIME)}))
    total(_flag({"time": f(TIME), "event": b([True] * 4)}))
    total(_flag({"time": f(TIME), "event": b([True] * 4), "left_trunc": f([0.0] * 4)}))
    assert len(traces) == 3


@FEATURE
def test_compiled_construction_reuses_compilation_for_values_that_would_change_concrete_flags():
    """spec 7.2, rev 62: new observation values with one argument form reuse compilation, even values whose concrete flags differ."""
    traces = []

    @jax.jit
    def run(time, event, left_trunc):
        traces.append(None)
        data = survjax.SurvivalData(time, event, left_trunc=left_trunc)
        return data.upper.sum(), data.issues.ok

    run(f(TIME), b([True] * 4), f([0.0] * 4))  # concretely: exact only, untruncated
    run(
        f(TIME), b(EVENT), f([0.0, 0.5, 0.0, 0.5])
    )  # concretely: exact, right, left truncation
    assert len(traces) == 1


# ------------------------------------------------- raw time/event form


def _exponential_raw(rate, time, event):
    return survjax.Exponential(rate=rate).log_likelihood(time=time, event=event)


def _exponential_data(rate, time, event):
    return survjax.Exponential(rate=rate).log_likelihood(
        survjax.SurvivalData(time, event)
    )


def _exponential_hand(rate, time, event):
    """Exponential rows: exact ``log r - r t``, right-censored ``-r t`` (spec 2)."""
    r = float(np.asarray(f(rate)))
    t = np.asarray(f(time), dtype=np.float64)
    return np.where(np.asarray(event), np.log(r), 0.0) - r * t


@FEATURE
def test_raw_form_under_jit_matches_hand_values_and_data_form():
    """spec 3.2, 6.1 and 12.1, rev 62: the raw form "equals `SurvivalData(time, event)`" with traced inputs, in values."""
    args = (f(0.7), f(TIME), b(EVENT))
    raw = np.asarray(jax.jit(_exponential_raw)(*args))
    via_data = np.asarray(jax.jit(_exponential_data)(*args))
    assert_close(raw, _exponential_hand(0.7, TIME, EVENT))
    assert_close(raw, via_data)


@FEATURE
def test_raw_form_with_all_events_matches_data_form():
    """spec 3.2 and 6.1, rev 62: with traced `event` all ones (every row exact) the raw and data forms agree with the hand values."""
    args = (f(0.7), f(TIME), b([True] * 4))
    raw = np.asarray(jax.jit(_exponential_raw)(*args))
    assert_close(raw, _exponential_hand(0.7, TIME, [True] * 4))
    assert_close(raw, np.asarray(jax.jit(_exponential_data)(*args)))


@FEATURE
def test_raw_form_under_jit_matches_data_form_in_rate_derivatives():
    """spec 3.2, 6.1, 7.3 and 12.1, rev 62: first and second rate derivatives of the summed raw form equal the data form and the hand values.

    ``sum_i ll_i = d log r - r sum t`` with ``d`` events: gradient ``d/r -
    sum t = 2/r - 6.5``, second derivative ``-d/r**2 = -2/r**2``.
    """
    r = float(np.asarray(f(0.7)))

    def summed(fn):
        return lambda rate, time, event: fn(rate, time, event).sum()

    args = (f(0.7), f(TIME), b(EVENT))
    raw_grad = jax.jit(jax.grad(summed(_exponential_raw)))(*args)
    data_grad = jax.jit(jax.grad(summed(_exponential_data)))(*args)
    raw_hess = jax.jit(jax.hessian(summed(_exponential_raw)))(*args)
    data_hess = jax.jit(jax.hessian(summed(_exponential_data)))(*args)
    assert_close(np.asarray(raw_grad), 2.0 / r - 6.5)
    assert_close(np.asarray(raw_grad), np.asarray(data_grad))
    assert_close(np.asarray(raw_hess), -2.0 / r**2)
    assert_close(np.asarray(raw_hess), np.asarray(data_hess))


@FEATURE
def test_raw_form_under_jit_matches_data_form_in_weibull_derivatives():
    """spec 3.2, 6.1, 7.3 and 12.1, rev 62: Weibull values, gradients and Hessians in (shape, scale) agree between the raw and data forms under `jit`."""

    def raw(k, s, time, event):
        return (
            survjax.Weibull(shape=k, scale=s)
            .log_likelihood(time=time, event=event)
            .sum()
        )

    def via_data(k, s, time, event):
        return (
            survjax.Weibull(shape=k, scale=s)
            .log_likelihood(survjax.SurvivalData(time, event))
            .sum()
        )

    args = (f(1.4), f(2.2), f(TIME), b(EVENT))
    assert_close(np.asarray(jax.jit(raw)(*args)), np.asarray(jax.jit(via_data)(*args)))
    for transform in (
        lambda g: jax.grad(g, argnums=(0, 1)),
        lambda g: jax.hessian(g, argnums=(0, 1)),
    ):
        got = jax.tree.leaves(jax.jit(transform(raw))(*args))
        want = jax.tree.leaves(jax.jit(transform(via_data))(*args))
        assert len(got) == len(want)
        for a, c in zip(got, want, strict=True):
            assert_close(np.asarray(a), np.asarray(c))
