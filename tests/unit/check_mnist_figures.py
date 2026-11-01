"""Offline checks for the exact three figures specified by use-case03."""

import sys
from pathlib import Path
from types import SimpleNamespace

import impl_tolerances as tol
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from examples.survival_mnist.run import _figures  # noqa: E402


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """Two images per digit in official-order stand-in, with different predictions."""
    out = tmp_path_factory.mktemp("mnist-figures")
    labels = np.concatenate((np.arange(9, -1, -1), np.arange(10)))
    pixels = np.broadcast_to(
        np.arange(20, dtype=np.float32)[:, None, None, None] / 19, (20, 1, 28, 28)
    )
    log_scales = np.log(2 + np.arange(20))
    history = SimpleNamespace(
        training_nll=(3.0, 2.0, 1.0), validation_nll=(4.0, 1.5, 2.0), selected_epoch=2
    )
    figures, groups = {}, []
    save, boxplot = Figure.savefig, Axes.boxplot

    def capture(fig, path, **kwargs):
        figures[Path(path).name] = fig
        return save(fig, path, **kwargs)

    def boxes(ax, values, **kwargs):
        groups.extend(values)
        return boxplot(ax, values, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Figure, "savefig", capture)
        patch.setattr(Axes, "boxplot", boxes)
        _figures(pixels, labels, np.log(np.float64(2)), log_scales, history, out)
    return out, figures, groups


def test_headline_selects_first_image_and_matches_analytic_survival(rendered):
    """03 Figures: first official image, t=0..20 in201 steps, true curve dashed."""
    _, figures, _ = rendered
    axes = figures["fig1.svg"].axes
    assert len(axes) == 20  # Image and curve for each of ten digits.
    times = np.arange(201) / 10
    for digit in range(10):
        image, curve = axes[2 * digit : 2 * digit + 2]
        # First digit d is at row9-d, so b=2+(9-d)=11-d, with non-unit shape k=2.
        np.testing.assert_array_equal(
            image.images[0].get_array(),
            np.full((28, 28), np.float32(9 - digit) / np.float32(19)),
        )
        prediction, truth = curve.lines
        np.testing.assert_allclose(
            prediction.get_xdata(), times, atol=tol.ATOL[False], rtol=tol.RTOL[False]
        )
        np.testing.assert_allclose(
            prediction.get_ydata(),
            np.exp(-((times / (11 - digit)) ** 2)),
            atol=tol.ATOL[False],
            rtol=tol.RTOL[False],
        )
        # Independent generating rule, not the plotted distribution method.
        scale = 10 * np.exp(-0.2 * digit)
        np.testing.assert_allclose(
            truth.get_ydata(),
            np.exp(-((times / scale) ** 1.38)),
            atol=tol.ATOL[False],
            rtol=tol.RTOL[False],
        )
        assert truth.get_linestyle() == "--"
        assert curve.get_xlim() == (0, 20) and curve.get_ylim() == (0, 1)
        assert image.get_title() == f"Digit {digit}"


def test_boxplots_include_every_prediction_and_true_digit_scales(rendered):
    """03 Figures: all20 rows grouped by digit; true log b_d markers."""
    _, figures, groups = rendered
    assert len(groups) == 10
    for digit, values in enumerate(groups):
        # Two official-order rows for d are9-d and10+d, with b=2+row.
        np.testing.assert_allclose(
            values,
            np.log([11 - digit, 12 + digit]),
            atol=tol.ATOL[False],
            rtol=tol.RTOL[False],
        )
    ax = figures["fig2.svg"].axes[0]
    np.testing.assert_allclose(
        ax.collections[0].get_offsets(),
        np.column_stack((np.arange(1, 11), np.log(10) - 0.2 * np.arange(10))),
        atol=tol.ATOL[False],
        rtol=tol.RTOL[False],
    )
    assert [label.get_text() for label in ax.get_xticklabels()] == [
        str(d) for d in range(10)
    ]


def test_history_uses_recorded_nll_and_one_based_selected_epoch(rendered):
    """03 Figures: epoch means and validation checkpoint, not last-epoch selection."""
    _, figures, _ = rendered
    training, validation, selected = figures["fig3.svg"].axes[0].lines
    np.testing.assert_array_equal(training.get_xdata(), [1, 2, 3])
    np.testing.assert_array_equal(training.get_ydata(), [3, 2, 1])
    np.testing.assert_array_equal(validation.get_ydata(), [4, 1.5, 2])
    np.testing.assert_array_equal(selected.get_xdata(), [2, 2])
    assert selected.get_label() == "Selected epoch"


def test_all_three_svgs_are_written_and_figures_closed(rendered):
    """03 Files: standalone fig1/2/3.svg, with no open figures left in the runner."""
    out, figures, _ = rendered
    assert set(figures) == {"fig1.svg", "fig2.svg", "fig3.svg"}
    assert all("<svg" in (out / name).read_text() for name in figures)
    assert not plt.get_fignums()


def test_missing_digit_fails_explicitly(tmp_path):
    """03 Figures: cannot silently substitute an image for a missing digit."""
    with pytest.raises(ValueError, match="every digit"):
        _figures(
            np.zeros((1, 1, 28, 28)), np.array([0]), 0, np.array([0]), None, tmp_path
        )
