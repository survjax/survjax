"""JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q <this file>."""

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import impl_tolerances as tol  # noqa: E402

from examples.weibull_simulation import run as runner  # noqa: E402
from examples.weibull_simulation.study import _assess_fit  # noqa: E402

if not jax.config.x64_enabled:
    raise RuntimeError("Run performance checks with JAX_ENABLE_X64=1")

ATOL, RTOL = tol.ATOL[True], tol.RTOL[True]


def _hand_fits():
    # Inverse Hessian gamma variances are 1 and 1/4, respectively.
    fits = [
        _assess_fit(
            jnp.array([0.0, gamma]), jnp.array(1.0), jnp.zeros(2), h, jnp.array(True)
        )
        for gamma, h in (
            (0.0, jnp.array([[4.0, 2.0], [2.0, 2.0]])),
            (3.0, jnp.diag(jnp.array([1.0, 4.0]))),
        )
    ]
    # This raw estimate must not contaminate any reduction; its SE is NaN.
    fits.append(
        fits[0]._replace(
            theta=jnp.array([0.0, 1e30]),
            log_hr_se=jnp.array(jnp.nan),
            accepted=jnp.array(False),
            solver_success=jnp.array(False),
        )
    )
    fits.append(
        _assess_fit(
            jnp.array([0.0, jnp.nan]),
            jnp.array(1.0),
            jnp.zeros(2),
            jnp.eye(2),
            jnp.array(True),
        )
    )
    return jax.tree.map(lambda *xs: jnp.stack(xs), *fits)


def test_performance_matches_hand_measures_and_mcse_eager_and_jit():
    """01 performance: accepted gamma=[0,3], SE=[1,.5], truth=.5, R_acc=2."""
    expected = {
        "bias": 1.0,
        "bias_mcse": 1.5,
        "empirical_se": 3 / np.sqrt(2),
        "empirical_se_mcse": 1.5,
        "model_se": np.sqrt(0.625),
        "coverage": 0.5,
        "coverage_mcse": 1 / np.sqrt(8),
        "power": 0.5,
        "power_mcse": 1 / np.sqrt(8),
    }
    for fn in (runner.performance, jax.jit(runner.performance)):
        actual = fn(_hand_fits(), 0.5)
        assert actual.keys() == expected.keys()
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, atol=ATOL, rtol=RTOL)
            assert actual[name].dtype == jnp.float64


def test_rejected_nan_theta_is_excluded_and_has_nonfinite_reason():
    """SIM.5 N-3 / 01 performance: NaN theta with finite objective is rejected."""
    fits = _hand_fits()
    assert not fits.accepted[-1] and np.isfinite(fits.objective[-1])
    assert runner.failure_columns(fits)["nonfinite"][-1]
    assert np.isfinite(runner.performance(fits)["bias"])


def test_zero_accepted_has_undefined_statistics():
    """01 accepted-fit denominators: no accepted estimate defines no statistic."""
    fits = _hand_fits()._replace(accepted=jnp.zeros(4, dtype=bool))
    assert all(np.isnan(v) for v in jax.jit(runner.performance)(fits).values())
    assert runner.failure_counts(fits)["failed"] == 4


def test_singleton_retains_defined_measures_and_strict_wald_test():
    """01 Wald/performance: z=1.96 covers zero inclusively but does not reject."""
    fits = jax.tree.map(lambda x: x[:1], _hand_fits())
    fits = fits._replace(theta=jnp.array([[0.0, 1.96]]))
    actual = jax.jit(runner.performance)(fits, 0.0)
    for name in ("empirical_se", "empirical_se_mcse", "bias_mcse"):
        assert np.isnan(actual[name])
    expected = {
        "bias": 1.96,
        "model_se": 1.0,
        "coverage": 1.0,
        "power": 0.0,
        "coverage_mcse": 0.0,
        "power_mcse": 0.0,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name], value, atol=ATOL, rtol=RTOL)


def test_timing_synchronizes_whole_results_and_takes_median(monkeypatch):
    """01 timing: first 10s, warm [1,7,2] -> median 2s, all calls blocked."""
    data, arm = runner.prepare_scenario(0, 6, 4, 20.0)
    fits = _hand_fits()
    events = []
    clock = iter([0.0, 10.0, 20.0, 21.0, 30.0, 37.0, 40.0, 42.0])
    real_block = jax.block_until_ready

    def block(value):
        events.append("block")
        if isinstance(value, tuple) and value is not fits:
            assert value[0] is data and value[1] is arm
        else:
            assert value is fits
        return real_block(value)

    def tick():
        events.append("clock")
        return next(clock)

    def fit(d, x, *, model):
        events.append("fit")
        assert d is data and x is arm and model == "weibull"
        return fits

    monkeypatch.setattr(runner, "perf_counter", tick)
    monkeypatch.setattr(runner, "fit_batch", fit)
    monkeypatch.setattr(jax, "block_until_ready", block)
    actual, first, warm = runner._time_model(data, arm, "weibull")
    assert actual is fits and first == 10.0 and warm == 2.0
    assert events == ["block"] + ["clock", "fit", "block", "clock"] * 4
    monkeypatch.setattr(runner, "prepare_scenario", lambda *args: (data, arm))
    monkeypatch.setattr(runner, "_time_model", lambda *args: (fits, first, warm))
    result = runner.run_scenario(0, 6, 4, 20.0)
    for record in result.models.values():
        assert record.failures["accepted"] == 2
        assert record.failures["attempted"] == 4
        assert record.fits_per_second == 2.0  # requested 4 / 2s, not accepted 2


def test_real_scenario_retains_both_models_counts_censoring_and_timings():
    """01 orchestration: both models use one cohort; all R results are retained."""
    result = runner.run_scenario(0, 100, 2, 20.0)
    data, _ = runner.prepare_scenario(0, 100, 2, 20.0)
    expected = np.count_nonzero(~np.asarray(data.exact)) / 200
    np.testing.assert_allclose(
        result.realized_censoring, expected, atol=ATOL, rtol=RTOL
    )
    assert result.censor_limit == 20.0 and set(result.models) == {
        "weibull",
        "exponential",
    }
    for record in result.models.values():
        assert record.failures["attempted"] == 2
        assert record.failures["accepted"] + record.failures["failed"] == 2
        assert record.fits.theta.shape[0] == 2
        assert np.isfinite(record.first_call_seconds) and record.first_call_seconds > 0
        assert np.isfinite(record.warm_seconds) and record.warm_seconds > 0
        np.testing.assert_allclose(
            record.fits_per_second, 2 / record.warm_seconds, atol=ATOL, rtol=RTOL
        )


def test_three_accepted_distinguish_bias_and_empirical_se_mcse():
    """01 MCSE: gamma=[0,1,2] has SD=1; 1/sqrt(3) differs from 1/2."""
    rows = [
        _assess_fit(
            jnp.array([0.0, gamma]),
            jnp.array(1.0),
            jnp.zeros(2),
            jnp.eye(2),
            jnp.array(True),
        )
        for gamma in (0.0, 1.0, 2.0)
    ]
    fits = jax.tree.map(lambda *xs: jnp.stack(xs), *rows)
    # Hand derivation: mean=1, sum squared deviations=2, ddof=1 variance=1.
    expected = {
        "bias": 1.0,
        "empirical_se": 1.0,
        "bias_mcse": 1 / np.sqrt(3),
        "empirical_se_mcse": 0.5,
    }
    for fn in (runner.performance, jax.jit(runner.performance)):
        actual = fn(fits, 0.0)
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, atol=ATOL, rtol=RTOL)


def test_grouped_summaries_reduce_replicates_without_pooling_scenarios():
    """01 performance: retain C, with R_acc=2 in one level and 1 in the other."""
    second = jax.vmap(
        lambda gamma: _assess_fit(
            jnp.array([0.0, gamma]),
            jnp.array(1.0),
            jnp.zeros(2),
            jnp.eye(2),
            jnp.array(True),
        )
    )(jnp.array([1.0, jnp.nan, jnp.nan, jnp.nan]))
    fits = jax.tree.map(lambda a, b: jnp.stack([a, b]), _hand_fits(), second)
    counts = jax.jit(runner.failure_counts)(fits)
    np.testing.assert_array_equal(counts["attempted"], [4, 4])
    np.testing.assert_array_equal(counts["accepted"], [2, 1])
    np.testing.assert_array_equal(counts["failed"], [2, 3])
    np.testing.assert_array_equal(counts["solver"], [1, 0])
    np.testing.assert_array_equal(counts["nonfinite"], [1, 3])
    # Independent hand algebra: first scenario as above; second gamma=1, SE=1.
    expected = {
        "bias": [1.0, 0.5],
        "bias_mcse": [1.5, np.nan],
        "empirical_se": [3 / np.sqrt(2), np.nan],
        "empirical_se_mcse": [1.5, np.nan],
        "model_se": [np.sqrt(0.625), 1.0],
        "coverage": [0.5, 1.0],
        "coverage_mcse": [1 / np.sqrt(8), 0.0],
        "power": [0.5, 0.0],
        "power_mcse": [1 / np.sqrt(8), 0.0],
    }
    # Host assertions inspect both eager and compiled vector-valued summaries.
    for fn in (runner.performance, jax.jit(runner.performance)):
        actual = fn(fits, 0.5)
        for name, value in expected.items():
            np.testing.assert_allclose(actual[name], value, atol=ATOL, rtol=RTOL)


def test_full_figure_panels_map_to_each_scenario_and_model(tmp_path, monkeypatch):
    """01 outputs: six-panel sampling and coverage retain the correct grouping."""
    from examples.weibull_simulation import _plots
    from examples.weibull_simulation._evidence import _read_csv

    source = ROOT / "examples/weibull_simulation/results"
    summary = _read_csv(source / "performance.csv")
    histogram = json.loads((source / "sampling_histograms.json").read_text())
    saved = {}

    def capture(figure, target):
        saved[target.name] = figure

    monkeypatch.setattr(_plots, "_save_figure", capture)
    _plots.write_figures(source, tmp_path)
    axes = saved["fig2_sampling.svg"].axes
    assert len(axes) == 6
    # Inspect host artists only; the full-run fits are never recomputed.
    for scenario, axis in enumerate(axes):
        record = next(r for r in summary if int(r["scenario"]) == scenario)
        assert (
            axis.get_title()
            == f"n={record['n']}, target censoring {float(record['target_censoring']):.0%}"
        )
        assert len(axis.patches) == 2
        for patch in axis.patches:
            model = patch.get_label().split()[0].lower()
            series = next(
                r
                for r in histogram["series"]
                if r["scenario"] == scenario and r["model"] == model
            )
            expected = np.array(series["counts"]) / (
                series["accepted"] * np.diff(histogram["edges"])
            )
            np.testing.assert_allclose(
                patch.get_data().values, expected, atol=ATOL, rtol=RTOL
            )
    coverage = saved["fig3_coverage.svg"].axes[0]
    labels = [
        f"n={r['n']}\n{float(r['target_censoring']):.0%} target censoring"
        for r in summary
        if r["model"] == "weibull"
    ]
    assert [label.get_text() for label in coverage.get_xticklabels()] == labels
    containers = coverage.containers
    assert len(containers) == 2
    for container in containers:
        model = container.get_label().lower()
        offset = -0.08 if model == "weibull" else 0.08
        np.testing.assert_array_equal(
            container.lines[0].get_xdata(), np.arange(6) + offset
        )
        expected = [
            float(
                next(
                    r
                    for r in summary
                    if r["model"] == model and int(r["scenario"]) == s
                )["coverage"]
            )
            for s in range(6)
        ]
        np.testing.assert_array_equal(container.lines[0].get_ydata(), expected)
