"""JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q <this file>."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from examples.weibull_simulation import run as runner  # noqa: E402
from examples.weibull_simulation._results import write_results  # noqa: E402
from examples.weibull_simulation.study import _assess_fit  # noqa: E402

if not jax.config.x64_enabled:
    raise RuntimeError("Run export checks with JAX_ENABLE_X64=1")


def test_complete_csv_roundtrip_preserves_every_fit_diagnostic_and_failure(tmp_path):
    """01 outputs: retain accepted, overlapping failures and unestimable rows."""
    models = {}
    # Fixed hand fixtures for the two specified model dimensions; no simulated fitting.
    for model, p in (("weibull", 3), ("exponential", 2)):
        theta = jnp.arange(p, dtype=float) / 10
        lower = jnp.tril(jnp.arange(1, p * p + 1, dtype=float).reshape(p, p))
        good = _assess_fit(
            theta,
            jnp.array(1.0),
            jnp.arange(p) * 1e-6,
            lower @ lower.T,
            jnp.array(True),
        )
        bad = _assess_fit(
            theta.at[0].set(jnp.nan),
            jnp.array(jnp.inf),
            jnp.full(p, -jnp.inf),
            jnp.full((p, p), jnp.nan),
            jnp.array(False),
        )
        missing = bad._replace(estimable=jnp.array(False))
        fits = jax.tree.map(lambda *xs: jnp.stack(xs), good, bad, missing)
        fits = jax.tree.map(lambda x: jnp.stack([x, x[::-1]]), fits)
        counts, measures, masks = runner._summarize(fits)
        models[model] = runner.ModelRun(fits, counts, measures, 10.0, 2.0, 3.0, masks)
    batch = jax.device_get(
        runner.ScenarioRun(np.array([40.0, 20.0]), jnp.array([0.25, 0.55]), models)
    )
    (tmp_path / "metadata.json").write_text("stale")
    (tmp_path / "evidence.json").write_text("stale")
    write_results(tmp_path, {0.25: 40.0, 0.55: 20.0}, [(np.array([4, 5]), 6, batch)])
    assert not (tmp_path / "metadata.json").exists()
    assert not (tmp_path / "evidence.json").exists()
    with (tmp_path / "performance.csv").open(newline="") as handle:
        summaries = list(csv.DictReader(handle))
    with (tmp_path / "replicates.csv").open(newline="") as handle:
        raw = list(csv.DictReader(handle))
    assert len(summaries) == 4 and len(raw) == 12
    assert len({(r["scenario"], r["model"], r["replicate"]) for r in raw}) == 12
    assert (
        list(summaries[0])[6:14]
        == "attempted accepted failed solver nonfinite gradient hessian unestimable".split()
    )
    assert (
        list(summaries[0])[14:23]
        == "bias bias_mcse empirical_se empirical_se_mcse model_se coverage coverage_mcse power power_mcse".split()
    )
    # Host round-trip assertions compare CSV values directly with the input records.
    for row in raw:
        level, r = int(row["scenario"]) - 4, int(row["replicate"])
        record = batch.models[row["model"]]
        fits, p = record.fits, int(row["parameter_count"])
        assert p == (3 if row["model"] == "weibull" else 2)
        for name in ("theta", "gradient"):
            np.testing.assert_array_equal(
                [float(row[f"{name}_{i}"]) for i in range(p)],
                getattr(fits, name)[level, r],
            )
        np.testing.assert_array_equal(
            [[float(row[f"hessian_{i}_{j}"]) for j in range(p)] for i in range(p)],
            fits.hessian[level, r],
        )
        for name in ("objective", "log_hr_se"):
            np.testing.assert_array_equal(
                float(row[name]), getattr(fits, name)[level, r]
            )
        np.testing.assert_array_equal(float(row["log_hr"]), fits.theta[level, r, -1])
        np.testing.assert_array_equal(
            [float(row["hr_lower"]), float(row["hr_upper"])], fits.hr_interval[level, r]
        )
        for name in ("solver_success", "hessian_ok", "estimable", "accepted"):
            assert int(row[name]) == int(getattr(fits, name)[level, r])
        for name, mask in record.failure_masks.items():
            assert int(row[name]) == int(mask[level, r])
        if p == 2:
            unused = "theta_2 gradient_2 hessian_0_2 hessian_1_2 hessian_2_0 hessian_2_1 hessian_2_2".split()
            assert all(row[name] == "" for name in unused)
    for row in summaries:
        cohort = [
            r
            for r in raw
            if (r["scenario"], r["model"]) == (row["scenario"], row["model"])
        ]
        assert (
            int(row["attempted"]) == 3
            and int(row["accepted"]) == 1
            and int(row["failed"]) == 2
        )
        for (
            name
        ) in "accepted failed solver nonfinite gradient hessian unestimable".split():
            assert sum(int(r[name]) for r in cohort) == int(row[name])
        assert (
            sum(
                int(row[name])
                for name in "solver nonfinite gradient hessian unestimable".split()
            )
            == 5
        )


def _compact_fixture(tmp_path):
    # Hand CSV: log(.5)=-log(2); accepted HRs .5 and 2. Middle ID fails
    # solver and gradient; Weibull/Exponential maxima are 3e-5/2e-5.
    raw = """scenario,model,replicate,parameter_count,log_hr,accepted,failed,solver,gradient,nonfinite,hessian,unestimable,theta_0,theta_1,theta_2,gradient_0,gradient_1,gradient_2
3,weibull,0,3,-0.6931471805599453,1,0,0,0,0,0,0,0,0,-0.6931471805599453,0,0,0
3,weibull,1,3,0,0,1,1,1,0,0,0,0,0,0,1e-5,2e-5,3e-5
3,weibull,2,3,0.6931471805599453,1,0,0,0,0,0,0,0,0,0.6931471805599453,0,0,0
3,exponential,0,2,-0.6931471805599453,1,0,0,0,0,0,0,0,-0.6931471805599453,,0,0,
3,exponential,1,2,0,0,1,1,1,0,0,0,0,0,,1e-5,2e-5,
3,exponential,2,2,0.6931471805599453,1,0,0,0,0,0,0,0,0.6931471805599453,,0,0,
"""
    summary = """scenario,n,target_censoring,model,attempted,accepted,failed,solver,gradient,nonfinite,hessian,unestimable
3,300,0.55,weibull,3,2,1,1,1,0,0,0
3,300,0.55,exponential,3,2,1,1,1,0,0,0
"""
    (tmp_path / "replicates.csv").write_text(raw)
    (tmp_path / "performance.csv").write_text(summary)
    (tmp_path / "metadata.json").write_text('{"source":{"commit":"fixture"}}')
    return list(csv.DictReader(raw.splitlines())), list(
        csv.DictReader(summary.splitlines())
    )


def test_compact_evidence_preserves_all_failures_histogram_and_shared_replicate(
    tmp_path,
):
    """D-035: complete histogram and failed-ID complement, not a fit subsample."""
    from examples.weibull_simulation._plots import _read_csv, write_evidence

    _compact_fixture(tmp_path)
    out = tmp_path / "compact"
    write_evidence(tmp_path, out)
    failures = _read_csv(out / "failures.csv")
    assert [(r["model"], r["replicate"]) for r in failures] == [
        ("weibull", "1"),
        ("exponential", "1"),
    ]
    assert all(r["failed"] == r["solver"] == r["gradient"] == "1" for r in failures)
    np.testing.assert_array_equal(
        [float(r["max_abs_gradient"]) for r in failures], [3e-5, 2e-5]
    )
    histogram = json.loads((out / "sampling_histograms.json").read_text())
    assert histogram["edges"][0] == 0.5 and histogram["edges"][-1] == 2.0
    assert all(s["accepted"] == sum(s["counts"]) == 2 for s in histogram["series"])
    assert all(s["counts"][0] == s["counts"][-1] == 1 for s in histogram["series"])
    assert [r["replicate"] for r in _read_csv(out / "illustration.csv")] == ["0", "0"]
    manifest = json.loads((out / "evidence.json").read_text())
    for name, expected in manifest["source_sha256"].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected
    for name, expected in manifest["output_sha256"].items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == expected
    assert not (out / "replicates.csv").exists()


@pytest.mark.parametrize(
    "corruption",
    ["duplicate", "missing", "reason", "accepted", "swap_accepted", "move_reasons"],
)
def test_inconsistent_raw_evidence_is_rejected_before_output(tmp_path, corruption):
    """D-035: compact artifacts cannot certify incomplete/inconsistent raw records."""
    from examples.weibull_simulation._plots import _csv, write_evidence

    raw, _ = _compact_fixture(tmp_path)
    if corruption == "duplicate":
        raw[-1]["replicate"] = 0
    elif corruption == "missing":
        raw.pop()
    elif corruption == "reason":
        raw[1]["gradient"] = 0
    elif corruption == "swap_accepted":
        # Preserve every group total: one row is in both sets and one in neither.
        raw[0]["accepted"], raw[1]["accepted"] = "0", "1"
    elif corruption == "move_reasons":
        # Preserve reason totals but put them on a row whose failed flag is zero.
        raw[0].update(solver="1", gradient="1")
        raw[1].update(solver="0", gradient="0")
    else:
        raw[0]["accepted"] = 0
    (tmp_path / "replicates.csv").write_text(_csv(raw, list(raw[0])))
    out = tmp_path / "absent"
    with pytest.raises(ValueError):
        write_evidence(tmp_path, out)
    assert not out.exists()


def test_no_accepted_estimates_produces_empty_histogram_and_illustration(tmp_path):
    """D-035: no accepted fits is represented explicitly, with every failure kept."""
    from examples.weibull_simulation._plots import _csv, _read_csv, write_evidence

    raw, summaries = _compact_fixture(tmp_path)
    for row in raw:
        row.update(accepted=0, failed=1, solver=1, gradient=1)
    for row in summaries:
        row.update(accepted=0, failed=3, solver=3, gradient=3)
    for name, rows in (("replicates.csv", raw), ("performance.csv", summaries)):
        (tmp_path / name).write_text(_csv(rows, list(rows[0])))
    write_evidence(tmp_path, tmp_path)
    histogram = json.loads((tmp_path / "sampling_histograms.json").read_text())
    assert all(s["accepted"] == sum(s["counts"]) == 0 for s in histogram["series"])
    assert _read_csv(tmp_path / "illustration.csv") == []
    assert len(_read_csv(tmp_path / "failures.csv")) == 6


def test_histogram_lattice_rounding_never_excludes_a_minimum(tmp_path):
    """17*.025 rounds above .425; move only the display edge outward."""
    from examples.weibull_simulation._plots import _csv, write_evidence

    raw, _ = _compact_fixture(tmp_path)
    raw[0]["log_hr"] = np.log(0.425)
    (tmp_path / "replicates.csv").write_text(_csv(raw, list(raw[0])))
    write_evidence(tmp_path, tmp_path)
    histogram = json.loads((tmp_path / "sampling_histograms.json").read_text())
    assert histogram["edges"][0] <= np.exp(float(raw[0]["log_hr"]))
    assert all(sum(s["counts"]) == 2 for s in histogram["series"])
