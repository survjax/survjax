"""Figures 1–4: JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.figures INPUT OUTPUT."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from numpyro.infer import Predictive

import survjax
from examples.bayesian_rmst.model import model, use_case_key

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def kaplan_meier(time: np.ndarray, event: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return product-limit step coordinates through last observed follow-up.

    Args:
        time: Nonempty finite nonnegative observed times, shape (N,).
        event: Boolean event indicators, shape (N,).
    Returns:
        Times and survival starting at (0,1), for plotting with where='post'.
    Raises:
        ValueError: Inputs violate the stated shapes or value domains.
    """
    time, event = np.asarray(time), np.asarray(event)
    if (
        time.ndim != 1
        or not time.size
        or event.shape != time.shape
        or event.dtype != bool
        or not np.all(np.isfinite(time))
        or np.any(time < 0)
    ):
        raise ValueError("KM requires aligned nonempty times and boolean events")
    event_times = np.unique(time[event])
    survival = [1.0]
    for t in event_times:
        # 02-details KM: censorings tied at t stay in the risk set for events at t.
        deaths = np.count_nonzero((time == t) & event)
        at_risk = np.count_nonzero(time >= t)
        survival.append(survival[-1] * (1 - deaths / at_risk))
    times = np.r_[0.0, event_times]
    values = np.asarray(survival)
    if times[-1] < time.max():
        times, values = np.r_[times, time.max()], np.r_[values, values[-1]]
    return times, values


def pointwise_band(curves: np.ndarray) -> np.ndarray:
    """Return lower/median/upper pointwise curves, reducing draw axis 1.

    Args:
        curves: Time-first array (time, draw, ...).
    Returns:
        Quantiles (3, time, ...) at .025, .5 and .975.
    Raises:
        ValueError: Missing draw axis, empty draws or nonfinite values.
    """
    curves = np.asarray(curves)
    if curves.ndim < 2 or not curves.shape[1] or not np.all(np.isfinite(curves)):
        raise ValueError("Expected finite time-first curves with nonempty draw axis")
    return np.quantile(curves, [0.025, 0.5, 0.975], axis=1)


def prior_survival(
    data: survjax.SurvivalData, arm: jax.Array, grid: jax.Array
) -> np.ndarray:
    """Evaluate 200 unconditioned prior survival draws on stream 2.

    Args:
        data: Fixed synthetic observations passed to the model, not conditioned sites.
        arm: Observed arm indicators (N,).
        grid: Survival times (K,) in years.
    Returns:
        Finite prior curves (K,200,2). No RMST on extreme prior parameters.
    Raises:
        ValueError: Float64 disabled or survival values are nonfinite.
    """
    if not jax.config.x64_enabled:
        raise ValueError("Bayesian figures require JAX_ENABLE_X64=1")
    samples = Predictive(
        model, num_samples=200, return_sites=("log_k", "log_b0", "gamma")
    )(use_case_key(2), data, arm)
    base = survjax.Weibull.from_log(
        log_shape=samples["log_k"][:, None], log_scale=samples["log_b0"][:, None]
    )
    family = survjax.ProportionalHazards(
        base, log_hazard_ratio=samples["gamma"][:, None] * jnp.array([0.0, 1.0])
    )
    curves = np.asarray(family.survival_grid(grid))
    if not np.all(np.isfinite(curves)):
        raise ValueError(
            "Prior produced nonfinite survival; investigate without dropping draws"
        )
    return curves


def render_figures(source: Path, output: Path) -> dict:
    """Render saved full-run draws with truth and observed KM overlays.

    Args:
        source: Directory containing metadata.json and arrays.npz from run.py.
        output: New directory receiving four SVG/PNG pairs and figure metadata.
    Returns:
        Figure metadata including prior specification and source-run settings.
    Raises:
        ValueError: Source is not a passing full run or prior metadata differs.
        FileExistsError: Output already exists; prior results are protected.
    """
    if output.exists():
        raise FileExistsError(output)
    record = json.loads((source / "metadata.json").read_text())
    priors = {
        name: {"distribution": "Normal", "loc": 0.0, "scale": 5.0}
        for name in ("log_k", "log_b0", "gamma")
    }
    if (
        record["config"] != "full"
        or not record["audit"]["passed"]
        or record["priors"] != priors
    ):
        raise ValueError("Figures require a passing full run with the committed priors")
    with np.load(source / "arrays.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    time, event, arm = (arrays[name] for name in ("time", "event", "arm"))
    grid, tau_grid = arrays["t_grid"], arrays["tau_grid"]
    data = survjax.SurvivalData(time, event)
    prior = prior_survival(data, jnp.asarray(arm), jnp.asarray(grid))
    truth = survjax.ProportionalHazards(
        survjax.Weibull(shape=1.38, scale=11.42),
        log_hazard_ratio=jnp.log(0.7) * jnp.array([0.0, 1.0]),
    )
    true_survival = np.asarray(truth.survival_grid(grid))
    true_rmst = np.asarray(truth.rmst(tau_grid[:, None]))
    delta = arrays["rmst_tau"][:, 1] - arrays["rmst_tau"][:, 0]
    delta_curve = arrays["rmst_curve"][:, :, 1] - arrays["rmst_curve"][:, :, 0]
    survival_band, delta_band = (
        pointwise_band(arrays["survival"]),
        pointwise_band(delta_curve),
    )
    km = [kaplan_meier(time[arm == group], event[arm == group]) for group in (0, 1)]
    output.mkdir(parents=True, exist_ok=False)

    def save(fig: plt.Figure, number: int) -> None:
        fig.tight_layout()
        for extension in ("svg", "png"):
            fig.savefig(output / f"fig{number}.{extension}", dpi=150)
        plt.close(fig)

    labels = ("Control", "Treatment")
    titles = (
        "200 prior curves · log parameters ~ Normal(0, 5)",
        "Full posterior survival · 4 chains × 1000 draws",
    )
    for number in (1, 2):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for group, ax in enumerate(axes):
            if number == 1:
                ax.plot(grid, prior[:, :, group], color="C0", alpha=0.1, linewidth=0.6)
            else:
                low, median, high = survival_band[:, :, group]
                ax.fill_between(
                    grid, low, high, color="C0", alpha=0.25, label="95% pointwise band"
                )
                ax.plot(grid, median, color="C0", label="Posterior median")
                ax.plot(
                    grid,
                    true_survival[:, group],
                    "--",
                    color="darkorange",
                    label="Truth",
                )
            ax.step(*km[group], where="post", color="black", label="Observed KM")
            ax.set(
                title=labels[group], xlabel="Time (years)", ylim=(0, 1.02), xlim=(0, 20)
            )
            ax.legend(fontsize=8)
        axes[0].set_ylabel("Survival probability")
        fig.suptitle(titles[number - 1])
        save(fig, number)

    fig, ax = plt.subplots(figsize=(7, 4))
    low, high = np.quantile(delta, [0.025, 0.975])
    ax.hist(
        delta,
        bins=50,
        density=True,
        color="C0",
        alpha=0.65,
        label="Posterior histogram",
    )
    ax.axvspan(low, high, color="C0", alpha=0.15, label="95% credible interval")
    ax.axvline(
        record["audit"]["truth"]["delta_RMST(5)"],
        color="darkorange",
        linestyle="--",
        label="Truth",
    )
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set(
        xlabel="Treatment − control RMST at 5 years (years)",
        ylabel="Density",
        title="Full posterior treatment effect",
    )
    ax.legend(fontsize=8)
    save(fig, 3)

    fig, ax = plt.subplots(figsize=(7, 4))
    low, median, high = delta_band
    ax.fill_between(
        tau_grid, low, high, color="C0", alpha=0.25, label="95% pointwise band"
    )
    ax.plot(tau_grid, median, color="C0", label="Posterior median")
    ax.plot(
        tau_grid,
        true_rmst[:, 1] - true_rmst[:, 0],
        "--",
        color="darkorange",
        label="Truth",
    )
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set(
        xlabel="Restriction time τ (years)",
        ylabel="Treatment − control RMST (years)",
        title="Full posterior RMST difference",
    )
    ax.legend(fontsize=8)
    save(fig, 4)
    metadata = {
        "source": str(source.resolve()),
        "source_run": record,
        "prior_draws": 200,
        "matplotlib_version": matplotlib.__version__,
        "prior_root_seed": 20261025,
        "prior_fold_in": [2, 2],
        "prior_all_finite": True,
        "figures": [f"fig{i}.svg" for i in range(1, 5)],
    }
    (output / "figures.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )
    np.savez_compressed(output / "prior_curves.npz", survival=prior, t_grid=grid)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render_figures(args.source, args.output)
