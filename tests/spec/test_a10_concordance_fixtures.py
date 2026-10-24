"""A10: Harrell's concordance against committed synthetic R fixtures (spec 9.2, 10.4, 12.1; rev 34).

9.2: "Cross-check against `survival::concordance(..., reverse = TRUE,
timewt = "n")`, documenting any convention difference." 10.4: synthetic
datasets with a recorded seed, R outputs and generating scripts committed
together. The maintainer's A10 outline adds ``timefix = FALSE`` so that R does
not merge nearly equal times when exact ties are checked (Q-046, Q-047).

The fixture is ``fixtures/concordance_synthetic.json``, schema in
``fixtures/README.md`` (Q-046). The spec tester never generates fixture
values: a missing fixture fails these tests.

R's ``count`` columns map to spec 9.2 as: ``concordant`` and ``discordant``
directly (``reverse = TRUE``: a larger score on the earlier event is
concordant); ``tied.x`` (comparable, tied score) is ``tied_score``;
``tied.y`` and ``tied.xy`` (tied on time) are not comparable; so
``comparable = concordant + discordant + tied.x``.
"""

import numpy as np
import pytest
from helpers import assert_close, b, count, f, harrell_reference

import survjax

FIXTURE = "concordance_synthetic.json"


def _load(spec_fixture):
    return spec_fixture(FIXTURE)


def _cases(spec_fixture):
    return _load(spec_fixture)["cases"]


def _data(case):
    return survjax.SurvivalData(f(case["time"]), b(np.asarray(case["event"]) == 1))


# ------------------------------------------------------------- fixture integrity


@pytest.mark.step("A10")
def test_fixture_records_the_r_call(spec_fixture):
    """spec 9.2 and 10.4, rev 34: the generator records the R call with `reverse = TRUE` and `timewt = "n"`."""
    call = _load(spec_fixture)["meta"]["call"].replace(" ", "")
    assert "reverse=TRUE" in call
    assert 'timewt="n"' in call


@pytest.mark.step("A10")
def test_fixture_covers_the_required_tie_patterns(spec_fixture):
    """spec 9.2, rev 34: the cases include event/event, event/censor and censor/censor time ties and score ties."""
    ee = ec = cc = score_ties = False
    for case in _cases(spec_fixture):
        time = np.asarray(case["time"], dtype=float)
        event = np.asarray(case["event"]) == 1
        for i in range(time.size):
            for j in range(i + 1, time.size):
                if time[i] == time[j]:
                    ee |= bool(event[i] and event[j])
                    cc |= bool(not event[i] and not event[j])
                    ec |= bool(event[i] != event[j])
        for scores in case["scores"]:
            score_ties |= np.unique(scores).size < len(scores)
    assert ee and ec and cc and score_ties


@pytest.mark.step("A10")
def test_fixture_rows_are_exact_or_right_censored_with_positive_finite_times(spec_fixture):
    """spec 9.1 and 10.4, rev 34: fixture data are supported rows: event `0` or `1`, finite positive times."""
    for case in _cases(spec_fixture):
        time = np.asarray(case["time"], dtype=float)
        assert np.all(np.isfinite(time)) and np.all(time > 0), case["name"]
        assert set(np.unique(case["event"])) <= {0, 1}, case["name"]
        assert len(case["scores"]) == len(case["r"]), case["name"]


@pytest.mark.step("A10")
def test_fixture_counts_agree_with_the_spec_pair_rules(spec_fixture):
    """spec 9.2, rev 34: R's counts equal the 9.2 rules enumerated in Python; a convention difference would show here, not as a survjax failure."""
    for case in _cases(spec_fixture):
        for scores, r in zip(case["scores"], case["r"], strict=True):
            conc, disc, tied, _ = harrell_reference(case["time"], np.asarray(case["event"]) == 1, scores)
            assert (r["concordant"], r["discordant"], r["tied_x"]) == (conc, disc, tied), case["name"]


# ---------------------------------------------------------------- survjax vs R


@pytest.mark.step("A10")
def test_concordance_matches_r(spec_fixture):
    """spec 9.2 and 12.1, rev 34: each score vector's `C` equals R's `concordance` at the 12.1 tolerances."""
    for case in _cases(spec_fixture):
        for scores, r in zip(case["scores"], case["r"], strict=True):
            value = np.asarray(survjax.concordance_index(f(scores), _data(case)))
            assert_close(value, r["concordance"], err_msg=case["name"])


@pytest.mark.step("A10")
def test_counts_match_r(spec_fixture):
    """spec 9.2, rev 34: concordant, discordant and tied-score counts equal R's `concordant`, `discordant`, `tied.x`; comparable their sum."""
    for case in _cases(spec_fixture):
        for scores, r in zip(case["scores"], case["r"], strict=True):
            _, info = survjax.concordance_index_with_info(f(scores), _data(case))
            assert count(info, "concordant") == r["concordant"], case["name"]
            assert count(info, "discordant") == r["discordant"], case["name"]
            assert count(info, "tied_score") == r["tied_x"], case["name"]
            expected = r["concordant"] + r["discordant"] + r["tied_x"]
            assert count(info, "comparable") == expected, case["name"]


@pytest.mark.step("A10")
def test_score_draws_match_r_in_one_call(spec_fixture):
    """spec 9.2, rev 34: all score vectors of a case as `(D,N)` scores give R's `D` values."""
    for case in _cases(spec_fixture):
        value = np.asarray(survjax.concordance_index(f(case["scores"]), _data(case)))
        assert value.shape == (len(case["scores"]),)
        assert_close(value, [r["concordance"] for r in case["r"]], err_msg=case["name"])


@pytest.mark.step("A10")
def test_raw_form_matches_r(spec_fixture):
    """spec 6.1 and 9.2, rev 34: the raw `time=`/`event=` form gives R's values."""
    for case in _cases(spec_fixture):
        value = np.asarray(
            survjax.concordance_index(
                f(case["scores"]), time=f(case["time"]), event=np.asarray(case["event"], dtype=np.int32)
            )
        )
        assert_close(value, [r["concordance"] for r in case["r"]], err_msg=case["name"])
