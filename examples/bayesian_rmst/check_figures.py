"""Explicit checks: JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.check_figures INPUT."""

import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from examples.bayesian_rmst.figures import kaplan_meier, pointwise_band, render_figures
from examples.bayesian_rmst.model import predictive_trials


def check_figures(source: Path) -> None:
    """Check hand-derived KM, draw quantiles and plot semantics on full results."""
    # At t=1, two deaths among four at risk (including the tied censor): S=.5.
    # Last death at t=2 is the sole remaining subject: S=0.
    t, s = kaplan_meier(
        np.array([1.0, 1.0, 1.0, 2.0]), np.array([True, True, False, True])
    )
    np.testing.assert_array_equal(t, [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(s, [1.0, 0.5, 0.0])
    t, s = kaplan_meier(np.array([1.0, 3.0]), np.array([False, False]))
    np.testing.assert_array_equal(t, [0.0, 3.0])
    np.testing.assert_array_equal(s, [1.0, 1.0])
    # Same marginals but crossed arms: paired deltas [-2,2], not zero.
    arms = np.array([[[1.0, 3.0], [3.0, 1.0]], [[2.0, 6.0], [6.0, 2.0]]])
    band = pointwise_band(arms[:, :, 1] - arms[:, :, 0])
    np.testing.assert_allclose(band, [[-1.9, -3.8], [0.0, 0.0], [1.9, 3.8]], atol=1e-12)
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "figures"
        captured = []
        original_close = plt.close

        def capture(fig=None):
            if hasattr(fig, "axes"):
                captured.append(fig)
            original_close(fig)

        with patch("matplotlib.pyplot.close", capture):
            metadata = render_figures(source, output)
        assert len(captured) == 5
        assert metadata["prior_draws"] == 200
        assert metadata["predictive_draws"] == 100
        assert metadata["predictive_fold_in"] == [2, 3]
        with np.load(
            output / "predictive_trials.npz", allow_pickle=False
        ) as predictive:
            with np.load(source / "arrays.npz", allow_pickle=False) as saved:
                expected_indices = np.arange(100) * saved["sample_gamma"].size // 100
                np.testing.assert_array_equal(predictive["indices"], expected_indices)
                assert metadata["predictive_indices"] == expected_indices.tolist()
                assert predictive["time"].shape == (100, saved["arm"].size)
                source_record = json.loads((source / "metadata.json").read_text())
                c = source_record["censor_limit_years"]
                assert metadata["predictive_known_censor_limit"] == c
                # Pin renderer-to-generator parameter, censor-limit and key wiring.
                root = jax.random.fold_in(
                    jax.random.fold_in(jax.random.key(20261025), 2), 3
                )
                for replicate in (0, 99):
                    row = expected_indices[replicate]
                    k = np.exp(saved["sample_log_k"].reshape(-1)[row])
                    b = np.exp(saved["sample_log_b0"].reshape(-1)[row])
                    eta = saved["sample_gamma"].reshape(-1)[row] * saved["arm"]
                    ek, ck = jax.random.split(jax.random.fold_in(root, replicate))
                    u = np.asarray(
                        jax.random.uniform(ek, saved["arm"].shape, dtype=jnp.float64)
                    )
                    v = np.asarray(
                        jax.random.uniform(ck, saved["arm"].shape, dtype=jnp.float64)
                    )
                    expected_t = b * np.exp(-eta / k) * (-np.log(u)) ** (1 / k)
                    np.testing.assert_allclose(
                        predictive["event_time"][replicate],
                        expected_t,
                        atol=1e-12,
                        rtol=0,
                    )
                    np.testing.assert_allclose(
                        predictive["censor_time"][replicate], c * v, atol=1e-12, rtol=0
                    )

                np.testing.assert_array_equal(
                    predictive["time"],
                    np.minimum(predictive["event_time"], predictive["censor_time"]),
                )
                np.testing.assert_array_equal(
                    predictive["event"],
                    predictive["event_time"] <= predictive["censor_time"],
                )
                for group in (0, 1):
                    lines = captured[4].axes[group].lines
                    assert len(lines) == 101
                    assert "known synthetic" in captured[4]._suptitle.get_text()
                    selected = saved["arm"] == group
                    for replicate in range(100):
                        t, s = kaplan_meier(
                            predictive["time"][replicate, selected],
                            predictive["event"][replicate, selected],
                        )
                        np.testing.assert_array_equal(lines[replicate].get_xdata(), t)
                        np.testing.assert_array_equal(lines[replicate].get_ydata(), s)
                    t, s = kaplan_meier(
                        saved["time"][selected], saved["event"][selected]
                    )
                    np.testing.assert_array_equal(lines[-1].get_xdata(), t)
                    np.testing.assert_array_equal(lines[-1].get_ydata(), s)

        assert len(captured[0].axes[0].lines) == 201  # 200 priors plus observed KM.
        assert "Density" == captured[2].axes[0].get_ylabel()
        assert len(captured[3].axes[0].lines) == 3  # median, truth, zero.
        import mpmath as mp

        with mp.workdps(40):
            true5 = [
                float(
                    mp.quad(
                        lambda t, h=h: mp.exp(
                            -h * (t / mp.mpf("11.42")) ** mp.mpf("1.38")
                        ),
                        [0, 5],
                    )
                )
                for h in (mp.mpf(1), mp.mpf(".7"))
            ]
        np.testing.assert_allclose(
            captured[2].axes[0].lines[0].get_xdata(),
            true5[1] - true5[0],
            atol=1e-12,
            rtol=0,
        )
        with np.load(output / "prior_curves.npz", allow_pickle=False) as prior:
            for group in (0, 1):
                np.testing.assert_array_equal(
                    captured[0].axes[group].lines[0].get_ydata(),
                    prior["survival"][:, 0, group],
                )
        with np.load(source / "arrays.npz", allow_pickle=False) as saved:
            delta = saved["rmst_tau"][:, 1] - saved["rmst_tau"][:, 0]
            low, high = np.quantile(delta, [0.025, 0.975])
            interval = captured[2].axes[0].patches[-1]
            np.testing.assert_allclose(
                [interval.get_x(), interval.get_width()],
                [low, high - low],
                atol=1e-12,
                rtol=0,
            )
            for group, hr in enumerate((1.0, 0.7)):
                true_s = np.exp(-hr * (saved["t_grid"] / 11.42) ** 1.38)
                np.testing.assert_allclose(
                    captured[1].axes[group].lines[1].get_ydata(),
                    true_s,
                    atol=1e-12,
                    rtol=0,
                )
                selected = saved["arm"] == group
                km_t, km_s = kaplan_meier(
                    saved["time"][selected], saved["event"][selected]
                )
                for figure, index in ((0, 200), (1, 2)):
                    np.testing.assert_array_equal(
                        captured[figure].axes[group].lines[index].get_xdata(), km_t
                    )
                    np.testing.assert_array_equal(
                        captured[figure].axes[group].lines[index].get_ydata(), km_s
                    )
            for index in (0, 50, 100):
                cutoff = float(saved["tau_grid"][index])
                with mp.workdps(40):
                    true_delta = mp.quad(
                        lambda t: (
                            mp.exp(
                                -mp.mpf(".7") * (t / mp.mpf("11.42")) ** mp.mpf("1.38")
                            )
                            - mp.exp(-((t / mp.mpf("11.42")) ** mp.mpf("1.38")))
                        ),
                        [0, cutoff],
                    )
                np.testing.assert_allclose(
                    captured[3].axes[0].lines[1].get_ydata()[index],
                    float(true_delta),
                    atol=1e-12,
                    rtol=0,
                )
            paired = saved["rmst_curve"][:, :, 1] - saved["rmst_curve"][:, :, 0]
            np.testing.assert_allclose(
                captured[3].axes[0].lines[0].get_ydata(),
                np.median(paired, axis=1),
                atol=1e-12,
                rtol=0,
            )
            for group in (0, 1):
                np.testing.assert_allclose(
                    captured[1].axes[group].lines[0].get_ydata(),
                    np.median(saved["survival"][:, :, group], axis=1),
                    atol=1e-12,
                    rtol=0,
                )

        for number in range(1, 6):
            assert (output / f"fig{number}.svg").stat().st_size > 1000
            assert (output / f"fig{number}.png").stat().st_size > 1000
        try:
            render_figures(source, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing figures overwritten")
    print("KM, quantile, figure and overwrite checks passed")


def check_predictive_generation() -> None:
    """Independent inverse-CDF/key algebra checks all posterior predictive rows."""
    samples = {
        "log_k": jnp.log(jnp.linspace(1.0, 2.0, 200)),
        "log_b0": jnp.log(jnp.linspace(3.0, 6.0, 200)),
        "gamma": jnp.linspace(-0.8, 0.3, 200),
    }
    arm, c = jnp.array([0.0, 0.0, 1.0, 1.0]), 7.0
    indices, trials = predictive_trials(samples, arm, c)
    np.testing.assert_array_equal(indices, np.arange(100) * 2)
    _, promoted = predictive_trials(
        {name: value.astype(jnp.float32) for name, value in samples.items()}, arm, c
    )
    assert promoted.time.dtype == jnp.float64
    repeat_indices, repeated = predictive_trials(samples, arm, c)
    np.testing.assert_array_equal(repeat_indices, indices)
    for left, right in zip(trials, repeated, strict=True):
        np.testing.assert_array_equal(left, right)
    # 02-details: stream 3; shared generator splits U/C, PH b=b0*exp(-eta/k).
    root = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261025), 2), 3)
    for i in range(100):
        event_key, censor_key = jax.random.split(jax.random.fold_in(root, i))
        u = np.asarray(jax.random.uniform(event_key, (4,), dtype=jnp.float64))
        v = np.asarray(jax.random.uniform(censor_key, (4,), dtype=jnp.float64))
        k = np.exp(float(samples["log_k"][2 * i]))
        b = np.exp(float(samples["log_b0"][2 * i]))
        eta = float(samples["gamma"][2 * i]) * np.asarray(arm)
        expected_t = b * np.exp(-eta / k) * (-np.log(u)) ** (1 / k)
        expected_c = c * v
        np.testing.assert_allclose(trials.event_time[i], expected_t, atol=1e-12, rtol=0)
        np.testing.assert_allclose(
            trials.censor_time[i], expected_c, atol=1e-12, rtol=0
        )
        np.testing.assert_allclose(
            trials.time[i], np.minimum(expected_t, expected_c), atol=1e-12, rtol=0
        )
        np.testing.assert_array_equal(trials.event[i], expected_t <= expected_c)
    assert trials.time.shape == (100, 4) and trials.time.dtype == jnp.float64
    assert not np.array_equal(trials.censor_time[0], trials.censor_time[1])
    # Data and prior streams must not produce the predictive sequence.
    wrong = jax.random.fold_in(jax.random.fold_in(jax.random.key(20261025), 2), 2)
    _, wrong_c = jax.random.split(jax.random.fold_in(wrong, 0))
    assert not np.array_equal(
        trials.censor_time[0], c * jax.random.uniform(wrong_c, (4,), dtype=jnp.float64)
    )
    bad_cases = [
        (dict(samples, log_k=jnp.zeros(99)), arm, c),
        (samples, jnp.array([0.0, 2.0]), c),
        (samples, arm, -1.0),
        (dict(samples, log_k=jnp.full(200, 1000.0)), arm, c),
        (dict(samples, gamma=samples["gamma"].at[0].set(jnp.nan)), arm, c),
    ]
    for values, arms, limit in bad_cases:
        try:
            predictive_trials(values, arms, limit)
        except ValueError:
            pass
        else:
            raise AssertionError("Malformed predictive input accepted")
    print("predictive generation/key/domain checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    check_predictive_generation()
    check_figures(parser.parse_args().source)
