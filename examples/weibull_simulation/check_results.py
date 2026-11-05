"""JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q <this file>."""

import csv
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

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
    write_results(tmp_path, {0.25: 40.0, 0.55: 20.0}, [(np.array([4, 5]), 6, batch)])
    assert not (tmp_path / "metadata.json").exists()
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
