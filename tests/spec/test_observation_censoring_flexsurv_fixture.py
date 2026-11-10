"""Slice B: per-row log likelihoods against the committed flexsurv fixture (spec 2, 10.4, 12.1, 12.2; rev 60).

Spec 10.4 (rev 59/60), "Slice B likelihood fixture": "A synthetic dataset with a
recorded seed and per-row log likelihoods from `flexsurvreg(..., fixedpars =
TRUE)`, for Weibull (including shape 1), log-normal and log-logistic, each at
two or more fixed natural parameter vectors without covariates. It covers only
the cells where flexsurv's conventions match section 2":

==================  ==============================================================
window              rows (cell names of ``fixtures/README.md``)
==================  ==============================================================
no truncation       exact, right, left, interval (``untruncated_*``)
left only           exact; right with ``c>L`` (``left_truncated_*``)
right only          exact ``t<=U``; left ``c<U``; interval ``b<=U`` (``right_truncated_*``)
two-sided           exact (``two_sided_exact``)
==================  ==============================================================

Rev 60: "Each cell has at least five rows, spread over the domain and
including, where the cell allows, an interval ending at `U` (`b=U`) and an
exact time at `U` (`t=U`). Every finite bound, censoring time and truncation
limit has cumulative hazard at most 15 ... Finite intervals, including
two-sided truncation windows, are not narrower than `1e-4` times their upper
bound, and every censored set and every truncation window other than
`(0,+inf]` has probability at least `1e-6`, under every parameter vector. ... Targets: section 12.1 float64
tolerances, compared in float64 only". The other cells are covered by
independent mathematics (TESTMAP, "Slice B flexsurv fixture";
``test_observation_truncation_lognormal_loglogistic.py`` for the log-normal
and log-logistic families).

The fixture is ``fixtures/slice_b_likelihood.json``; its schema is in
``fixtures/README.md`` (Q-073, resolved by rev 60: one dataset shared by all
families and vectors). 10.4: "The spec tester defines the file name and
schema; the generator writes that schema." The spec tester never generates
fixture values: a missing fixture fails these tests.

Markers (role table, which now names the fixture's untruncated exact and
right-censored rows under ``observation.censoring``; Q-074 resolved by rev
60): untruncated rows ``observation.censoring``; truncated exact/right rows
``observation.truncation``; truncated left/interval rows both; log-normal and
log-logistic add ``family.lognormal`` / ``family.loglogistic``. The survjax
comparisons run in float64 only (10.4 rev 60).

The "fixture integrity" tests check only the committed file against 10.4 (the
ten cells with at least five rows each and their ``b=U`` / ``t=U`` rows,
families and vectors, the cumulative-hazard, width and minimum-probability
limits, and agreement with 50-digit mpmath of the spec 2 table); they do not
call survjax but stop a malformed fixture from passing silently.
"""

import math

import numpy as np
import observation_reference as oref
import pytest
from helpers import INF, NAN, X64, flag_data, ll
from tolerances import TOLERANCES

import survjax

FIXTURE = "slice_b_likelihood.json"
F64 = TOLERANCES[np.dtype(np.float64)]

CENSORING = "observation.censoring"
TRUNCATION = "observation.truncation"

# Spec 10.4 (rev 59): the ten comparable cells, as (window, kind).
CELLS = {
    "untruncated_exact": ("none", "exact"),
    "untruncated_right": ("none", "right"),
    "untruncated_left": ("none", "left"),
    "untruncated_interval": ("none", "interval"),
    "left_truncated_exact": ("left", "exact"),
    "left_truncated_right": ("left", "right"),
    "right_truncated_exact": ("right", "exact"),
    "right_truncated_left": ("right", "left"),
    "right_truncated_interval": ("right", "interval"),
    "two_sided_exact": ("two_sided", "exact"),
}
CELL_FEATURES = {
    "untruncated_exact": (CENSORING,),
    "untruncated_right": (CENSORING,),
    "untruncated_left": (CENSORING,),
    "untruncated_interval": (CENSORING,),
    "left_truncated_exact": (TRUNCATION,),
    "left_truncated_right": (TRUNCATION,),
    "right_truncated_exact": (TRUNCATION,),
    "right_truncated_left": (CENSORING, TRUNCATION),
    "right_truncated_interval": (CENSORING, TRUNCATION),
    "two_sided_exact": (TRUNCATION,),
}

# Spec 4.1 natural parameters, in the order of ``observation_reference``.
FAMILIES = {
    "weibull": ("shape", "scale"),
    "lognormal": ("meanlog", "sdlog"),
    "loglogistic": ("shape", "scale"),
}
FAMILY_FEATURES = {
    "weibull": (),
    "lognormal": ("family.lognormal",),
    "loglogistic": ("family.loglogistic",),
}
KINDS = ("exact", "right", "left", "interval")
MAX_CUMULATIVE_HAZARD = 15.0
MIN_RELATIVE_WIDTH = 1e-4
MIN_ROWS_PER_CELL = 5
MIN_PROBABILITY = 1e-6
# Spec 10.4 (rev 60): cells that allow an interval ending at ``U`` (``b=U``) or an exact time at ``U`` (``t=U``).
CELLS_WITH_INTERVAL_AT_UPPER = ("right_truncated_interval",)
CELLS_WITH_EXACT_AT_UPPER = ("right_truncated_exact", "two_sided_exact")

FLOAT64_ONLY = pytest.mark.skipif(
    not X64, reason="spec 10.4 (rev 60): section 12.1 float64 tolerances, compared in float64 only"
)
INTEGRITY = pytest.mark.feature(CENSORING)


def _optional(values):
    """JSON ``null`` (``None``) to ``nan``, as float64."""
    return np.array([NAN if v is None else float(v) for v in values], dtype=np.float64)


def _load(spec_fixture):
    """The fixture as NumPy float64 arrays, in the Q-073 schema of ``fixtures/README.md``."""
    raw = spec_fixture(FIXTURE)
    data = raw["data"]
    right_trunc = _optional(data["right_trunc"])
    out = {
        "cell": np.asarray(data["cell"], dtype=str),
        "kind": np.asarray(data["kind"], dtype=str),
        "time": np.asarray(data["time"], dtype=np.float64),
        "time2": _optional(data["time2"]),
        "left_trunc": np.asarray(data["left_trunc"], dtype=np.float64),
        # null is +inf (no upper truncation).
        "right_trunc": np.where(np.isnan(right_trunc), INF, right_trunc),
        "families": {},
    }
    for family, block in raw["families"].items():
        out["families"][family] = [
            {
                "name": str(vector["name"]),
                "parameters": {key: float(value) for key, value in vector["parameters"].items()},
                "loglik": np.asarray(vector["loglik"], dtype=np.float64),
            }
            for vector in block["vectors"]
        ]
    return out


@pytest.fixture(scope="module")
def fx(spec_fixture):
    return _load(spec_fixture)


def _params(family, vector):
    return tuple(vector["parameters"][key] for key in FAMILIES[family])


def _dist(family, vector):
    p = vector["parameters"]
    one = np.float64
    if family == "weibull":
        return survjax.Weibull(shape=one(p["shape"]), scale=one(p["scale"]))
    if family == "lognormal":
        return survjax.LogNormal(meanlog=one(p["meanlog"]), sdlog=one(p["sdlog"]))
    return survjax.LogLogistic(shape=one(p["shape"]), scale=one(p["scale"]))


def _data(fx, rows):
    """``SurvivalData`` (flag form) for the selected rows; truncation passed only where the rows use it."""
    window = {}
    if (fx["left_trunc"][rows] != 0).any():
        window["left_trunc"] = fx["left_trunc"][rows]
    if np.isfinite(fx["right_trunc"][rows]).any():
        window["right_trunc"] = fx["right_trunc"][rows]
    return flag_data(fx["kind"][rows], fx["time"][rows], fx["time2"][rows], **window)


def _row_set(fx, i):
    """Row ``i``'s ``(lower, upper)``: the censored set ``(lower, upper]``, or ``(t, None)`` for an exact row."""
    t, t2 = fx["time"][i], fx["time2"][i]
    return {"exact": (t, None), "right": (t, INF), "left": (0.0, t), "interval": (t, t2)}[fx["kind"][i]]


def _reference(fx, family, vector, i):
    """50-digit mpmath of the spec 2 table for row ``i`` at the fixture's float64 inputs."""
    kind = fx["kind"][i]
    lower, upper = _row_set(fx, i)
    return oref.contribution_float64(
        family,
        _params(family, vector),
        lower,
        upper,
        exact=kind == "exact",
        left=fx["left_trunc"][i],
        right=fx["right_trunc"][i],
    )


def _cases(features_of):
    return [
        pytest.param(
            family,
            cell,
            marks=pytest.mark.feature(*features_of(family, cell)),
            id=f"{family}-{cell}",
        )
        for family in FAMILIES
        for cell in CELLS
    ]


# ------------------------------------------------------- fixture integrity (no survjax)


@INTEGRITY
def test_fixture_covers_exactly_the_ten_comparable_cells(fx):
    """spec 10.4, rev 59: every row is in one of the ten comparable cells, and every cell has rows."""
    assert set(fx["cell"].tolist()) == set(CELLS)
    n = fx["cell"].shape[0]
    for name in ("kind", "time", "time2", "left_trunc", "right_trunc"):
        assert fx[name].shape == (n,), name


@INTEGRITY
@pytest.mark.parametrize("cell", list(CELLS))
def test_fixture_has_at_least_five_rows_per_cell(fx, cell):
    """spec 10.4, rev 60: "Each cell has at least five rows"."""
    assert int((fx["cell"] == cell).sum()) >= MIN_ROWS_PER_CELL, cell


@INTEGRITY
@pytest.mark.parametrize("cell", CELLS_WITH_INTERVAL_AT_UPPER)
def test_fixture_has_an_interval_ending_at_upper(fx, cell):
    """spec 10.4, rev 60: the cell includes "an interval ending at `U` (`b=U`)" (float64 equality of `time2` and `right_trunc`)."""
    rows = fx["cell"] == cell
    assert (fx["time2"][rows] == fx["right_trunc"][rows]).any(), cell


@INTEGRITY
@pytest.mark.parametrize("cell", CELLS_WITH_EXACT_AT_UPPER)
def test_fixture_has_an_exact_time_at_upper(fx, cell):
    """spec 10.4, rev 60: the cell includes "an exact time at `U` (`t=U`)" (float64 equality of `time` and `right_trunc`)."""
    rows = fx["cell"] == cell
    assert (fx["time"][rows] == fx["right_trunc"][rows]).any(), cell


@INTEGRITY
@pytest.mark.parametrize("cell", list(CELLS))
def test_fixture_rows_match_their_cell(fx, cell):
    """spec 2/10.4, rev 59: each row's kind and window are those its cell names, with the 10.4 membership conditions.

    No truncation is ``L=0, U=+inf``; left only ``L>0, U=+inf`` with right rows
    ``c>L``; right only ``L=0``, finite ``U``, exact ``t<=U``, left ``c<U``,
    interval ``b<=U``; two-sided ``L>0``, finite ``U``, exact ``L<t<=U``. Times
    are positive and finite, intervals have ``a<b``, ``time2`` is null on
    non-interval rows, and every window has ``L<U`` (spec 2).
    """
    window, kind = CELLS[cell]
    rows = fx["cell"] == cell
    t, t2 = fx["time"][rows], fx["time2"][rows]
    lt, rt = fx["left_trunc"][rows], fx["right_trunc"][rows]
    assert (fx["kind"][rows] == kind).all()
    assert np.isfinite(t).all() and (t > 0).all()
    assert np.isfinite(lt).all() and (lt >= 0).all() and (lt < rt).all()
    if kind == "interval":
        assert np.isfinite(t2).all() and (t < t2).all()
    else:
        assert np.isnan(t2).all()
    has_left = {"none": False, "left": True, "right": False, "two_sided": True}[window]
    has_right = {"none": False, "left": False, "right": True, "two_sided": True}[window]
    assert ((lt > 0) == has_left).all()
    assert (np.isfinite(rt) == has_right).all()
    if kind == "exact":
        assert (t > lt).all() and (t <= rt).all()
    if window == "left" and kind == "right":
        assert (t > lt).all()
    if window == "right" and kind == "left":
        assert (t < rt).all()
    if window == "right" and kind == "interval":
        assert (t2 <= rt).all()


@INTEGRITY
def test_fixture_has_the_three_families_with_two_or_more_vectors(fx):
    """spec 10.4, rev 59: Weibull, log-normal and log-logistic, "each at two or more fixed natural parameter vectors without covariates"."""
    assert set(fx["families"]) == set(FAMILIES)
    for family, vectors in fx["families"].items():
        assert len(vectors) >= 2, family
        names = [v["name"] for v in vectors]
        assert len(set(names)) == len(names), family
        for vector in vectors:
            assert set(vector["parameters"]) == set(FAMILIES[family]), (family, vector["name"])
            assert all(math.isfinite(x) for x in vector["parameters"].values())


@INTEGRITY
def test_fixture_weibull_includes_shape_one(fx):
    """spec 10.4, rev 59: "Weibull (including shape 1)"."""
    assert any(v["parameters"]["shape"] == 1.0 for v in fx["families"]["weibull"])


@INTEGRITY
def test_fixture_parameters_are_valid(fx):
    """spec 4.1/4.7, rev 59: `shape`, `scale` and `sdlog` positive, `meanlog` real (finite)."""
    for family, vectors in fx["families"].items():
        for vector in vectors:
            for key, value in vector["parameters"].items():
                if key != "meanlog":
                    assert value > 0, (family, vector["name"], key)


@INTEGRITY
def test_fixture_log_likelihoods_have_one_finite_value_per_row(fx):
    """spec 10.4, rev 59: per-row log likelihoods, one per row in row order, all finite."""
    n = fx["cell"].shape[0]
    for family, vectors in fx["families"].items():
        for vector in vectors:
            assert vector["loglik"].shape == (n,), (family, vector["name"])
            assert np.isfinite(vector["loglik"]).all(), (family, vector["name"])


@INTEGRITY
def test_fixture_cumulative_hazards_are_at_most_15(fx):
    """spec 10.4, rev 59: "Every finite bound, censoring time and truncation limit has cumulative hazard at most 15" (mpmath `H`)."""
    points = np.concatenate([fx["time"], fx["time2"], fx["left_trunc"], fx["right_trunc"]])
    points = np.unique(points[np.isfinite(points)])
    for family, vectors in fx["families"].items():
        for vector in vectors:
            params = _params(family, vector)
            worst = max(oref.cumulative_hazard_float64(family, params, t) for t in points)
            assert worst <= MAX_CUMULATIVE_HAZARD, (family, vector["name"], worst)


@INTEGRITY
def test_fixture_intervals_are_not_too_narrow(fx):
    """spec 10.4, rev 60: "Finite intervals, including two-sided truncation windows, are not narrower than `1e-4` times their upper bound" (interval rows and every window with finite `U`)."""
    interval = fx["kind"] == "interval"
    a, b_ = fx["time"][interval], fx["time2"][interval]
    assert (b_ - a >= MIN_RELATIVE_WIDTH * b_).all()
    windowed = np.isfinite(fx["right_trunc"])
    lt, rt = fx["left_trunc"][windowed], fx["right_trunc"][windowed]
    assert (rt - lt >= MIN_RELATIVE_WIDTH * rt).all()


@INTEGRITY
@pytest.mark.parametrize("family", list(FAMILIES))
def test_fixture_censored_sets_and_windows_have_probability_at_least_1e_minus_6(fx, family):
    """spec 10.4, rev 60: "every censored set and every truncation window other than `(0,+inf]` has probability at least `1e-6`, under every parameter vector".

    50-digit mpmath of the unconditional ``P(lower < T <= upper)`` of the
    section 2 table at the fixture's float64 inputs: for each censored row's
    set (right ``(c, +inf]``, left ``(0, c]``, interval ``(a, b]``) and for
    each window other than ``(0, +inf]``: left-only ``(L, +inf]``
    (probability ``S(L)``; flexsurv computes its normaliser ``1 - F(L)`` by
    subtraction), right-only ``(0, U]`` and two-sided ``(L, U]``.
    """
    n = fx["cell"].shape[0]
    sets = {_row_set(fx, i) for i in range(n) if fx["kind"][i] != "exact"}
    windowed = np.flatnonzero((fx["left_trunc"] > 0) | np.isfinite(fx["right_trunc"]))
    sets |= {(fx["left_trunc"][i], fx["right_trunc"][i]) for i in windowed}
    floor = math.log(MIN_PROBABILITY)
    for vector in fx["families"][family]:
        params = _params(family, vector)
        worst = min(sets, key=lambda s: oref.log_mass_float64(family, params, *s))
        log_p = oref.log_mass_float64(family, params, *worst)
        assert log_p >= floor, (vector["name"], worst, math.exp(log_p))


@INTEGRITY
def test_fixture_censored_rows_have_non_positive_log_likelihood(fx):
    """spec 2, rev 59: a censored row is a conditional probability, at most 1; in the comparable cells flexsurv's value cannot exceed 0 (10.4 contrasts sets reaching beyond `U`)."""
    censored = fx["kind"] != "exact"
    for family, vectors in fx["families"].items():
        for vector in vectors:
            assert (vector["loglik"][censored] <= F64["atol"]).all(), (family, vector["name"])


@INTEGRITY
@pytest.mark.parametrize("family", list(FAMILIES))
def test_fixture_agrees_with_mpmath(fx, family):
    """spec 2/10.4/12.1, rev 59: the flexsurv values equal 50-digit mpmath of the spec 2 table at the 12.1 float64 tolerances.

    In the comparable cells flexsurv's conventions match section 2, so a
    disagreement means a malformed fixture (wrong parameterisation, cell or
    precision), not a survjax error.
    """
    n = fx["cell"].shape[0]
    for vector in fx["families"][family]:
        expected = np.array([_reference(fx, family, vector, i) for i in range(n)])
        np.testing.assert_allclose(vector["loglik"], expected, err_msg=vector["name"], **F64)


# ------------------------------------------------------- survjax against flexsurv


@FLOAT64_ONLY
@pytest.mark.parametrize("family, cell", _cases(lambda family, cell: (*CELL_FEATURES[cell], *FAMILY_FEATURES[family])))
def test_per_row_log_likelihood_matches_flexsurv_in_each_cell(fx, family, cell):
    """spec 2/10.4/12.1, rev 60: `dist.log_likelihood` per row equals flexsurv's per-row value in each comparable cell, at every fixture vector (float64 only, `atol=1e-10, rtol=1e-8`)."""
    rows = np.flatnonzero(fx["cell"] == cell)
    assert rows.size > 0, cell
    data = _data(fx, rows)
    for vector in fx["families"][family]:
        value = ll(_dist(family, vector), data)
        assert value.shape == (rows.size,)
        assert value.dtype == np.float64
        np.testing.assert_allclose(value, vector["loglik"][rows], err_msg=vector["name"], **F64)


@FLOAT64_ONLY
@pytest.mark.parametrize(
    "family",
    [
        pytest.param(family, marks=pytest.mark.feature(CENSORING, TRUNCATION, *FAMILY_FEATURES[family]), id=family)
        for family in FAMILIES
    ],
)
def test_whole_fixture_in_one_batch_matches_flexsurv(fx, family):
    """spec 1/2/10.4/12.1, rev 60: all ten cells in one mixed batch (per-row kinds and windows) give flexsurv's per-row values (float64 only)."""
    rows = np.arange(fx["cell"].shape[0])
    data = _data(fx, rows)
    for vector in fx["families"][family]:
        value = ll(_dist(family, vector), data)
        np.testing.assert_allclose(value, vector["loglik"], err_msg=vector["name"], **F64)


@FLOAT64_ONLY
@pytest.mark.parametrize(
    "family",
    [
        pytest.param(family, marks=pytest.mark.feature(CENSORING, TRUNCATION, *FAMILY_FEATURES[family]), id=family)
        for family in FAMILIES
    ],
)
def test_summed_fixture_log_likelihood_matches_flexsurv(fx, family):
    """spec 7.1/10.4, rev 60: the summed objective matches flexsurv's sum (float64 only); `atol` scaled by the row count (each row within `atol`)."""
    rows = np.arange(fx["cell"].shape[0])
    data = _data(fx, rows)
    for vector in fx["families"][family]:
        total = float(ll(_dist(family, vector), data).sum(dtype=np.float64))
        expected = float(vector["loglik"].sum())
        np.testing.assert_allclose(total, expected, rtol=F64["rtol"], atol=rows.size * F64["atol"], err_msg=vector["name"])
