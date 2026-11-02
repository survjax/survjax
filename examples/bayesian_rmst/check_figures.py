"""Explicit checks: JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.check_figures INPUT."""

import argparse
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

from examples.bayesian_rmst.figures import kaplan_meier, pointwise_band, render_figures


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
        assert len(captured) == 4
        assert metadata["prior_draws"] == 200
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

        for number in range(1, 5):
            assert (output / f"fig{number}.svg").stat().st_size > 1000
            assert (output / f"fig{number}.png").stat().st_size > 1000
        try:
            render_figures(source, output)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Existing figures overwritten")
    print("KM, quantile, figure and overwrite checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    check_figures(parser.parse_args().source)
