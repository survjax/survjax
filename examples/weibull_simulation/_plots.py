"""Render the three simulation figures from compact evidence; never refit."""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


def _save_figure(figure, target: Path) -> None:
    """Keep SVG text/vector geometry and remove only generated XML whitespace."""
    from xml.etree import ElementTree as ET

    buffer = io.BytesIO()
    figure.savefig(buffer, format="svg", metadata={"Date": None})
    root = ET.fromstring(buffer.getvalue())
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    # Serialization only: preserve coordinates and label text, compact path whitespace.
    for node in root.iter():
        if node.text is not None and not node.text.strip():
            node.text = None
        if node.tail is not None and not node.tail.strip():
            node.tail = None
        if "d" in node.attrib:
            node.attrib["d"] = " ".join(node.attrib["d"].split())
    target.write_bytes(
        ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    )


def write_figures(results: Path, out: Path) -> None:
    """Write the three specified SVGs from compact evidence, without refitting.

    Args:
        results: Complete compact evidence directory (hashes verified before plotting).
        out: Destination for fig1_survival.svg, fig2_sampling.svg, fig3_coverage.svg.
    Raises:
        ValueError: Evidence hash mismatch or float64 disabled.
        OSError: Input/output failure. Empty accepted sets are labelled explicitly.
    """
    import jax
    import jax.numpy as jnp
    import matplotlib as mpl
    from matplotlib.figure import Figure

    import survjax

    if not jax.config.x64_enabled:
        raise ValueError("simulation figures require JAX_ENABLE_X64=1")
    manifest = json.loads((results / "evidence.json").read_text())
    # Host input checks and plot construction only; no scenario/generator/fit loop.
    for name, digest in manifest["output_sha256"].items():
        if hashlib.sha256((results / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"evidence hash mismatch: {name}")
    with (results / "performance.csv").open(newline="") as handle:
        summary = list(csv.DictReader(handle))
    with (results / "illustration.csv").open(newline="") as handle:
        illustration = list(csv.DictReader(handle))
    metadata = json.loads((results / "metadata.json").read_text())
    histogram = json.loads((results / "sampling_histograms.json").read_text())
    dgp = metadata["dgp"]
    scenarios = sorted({int(r["scenario"]) for r in summary})
    colors = {"weibull": "#0072B2", "exponential": "#D55E00"}
    out.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context(
        {"svg.fonttype": "none", "svg.hashsalt": "weibull-study", "font.size": 9}
    ):
        survival = Figure(figsize=(7.2, 4.2), layout="constrained")
        ax = survival.subplots()
        time = jnp.linspace(0.0, 40.0, 161)
        arm = jnp.array([0.0, 1.0])
        truth = survjax.Weibull(shape=dgp["shape"], scale=dgp["baseline_scale"])
        curves = [("True", truth, dgp["log_hr"], "-")]
        # Different baseline families need separate plot graphs, with vectorized grids.
        for row in illustration:
            theta = [
                float(row[f"theta_{i}"]) for i in range(int(row["parameter_count"]))
            ]
            base = (
                survjax.Weibull.from_log(log_shape=theta[0], log_scale=theta[1])
                if row["model"] == "weibull"
                else survjax.Exponential.from_log(log_rate=theta[0])
            )
            label = row["model"].capitalize()
            style = "--" if row["model"] == "weibull" else ":"
            curves.append((label, base, theta[-1], style))
        for name, base, gamma, style in curves:
            lp = gamma * arm
            dist = survjax.ProportionalHazards(base, log_hazard_ratio=lp)
            prediction = np.asarray(dist.survival_grid(time))
            for a, color in enumerate(("#0072B2", "#D55E00")):
                ax.plot(
                    time,
                    prediction[:, a],
                    color=color,
                    linestyle=style,
                    label=f"{name}, {'control' if a == 0 else 'treatment'}",
                )
        if not illustration:
            ax.text(
                0.5,
                0.5,
                "No mutually accepted replicate",
                transform=ax.transAxes,
                ha="center",
            )
        setting = next(
            (
                r
                for r in summary
                if illustration and r["scenario"] == illustration[0]["scenario"]
            ),
            None,
        )
        ax.set(
            xlabel="Time (years)",
            ylabel="Survival probability",
            ylim=(0, 1),
            title=f"Survival illustration: n={setting['n']}, target censoring {float(setting['target_censoring']):.0%}, replicate {illustration[0]['replicate']}"
            if setting
            else "True survival; fitted illustration unavailable",
        )
        ax.legend(ncol=2, fontsize=8)
        nrows = (len(scenarios) + 1) // 2
        sampling = Figure(figsize=(8.8, 2.4 * nrows), layout="constrained")
        axes = sampling.subplots(nrows, 2, squeeze=False).ravel()
        edges = np.asarray(histogram["edges"])
        for position, s in enumerate(scenarios):
            axis = axes[position]
            rows = [r for r in summary if int(r["scenario"]) == s]
            for series in (r for r in histogram["series"] if r["scenario"] == s):
                count = series["accepted"]
                density = (
                    np.asarray(series["counts"]) / (count * np.diff(edges))
                    if count
                    else np.zeros(len(edges) - 1)
                )
                axis.stairs(
                    density,
                    edges,
                    color=colors[series["model"]],
                    label=f"{series['model'].capitalize()} (accepted={count})",
                )
            axis.axvline(
                np.exp(dgp["log_hr"]), color="black", linestyle="--", label="True HR"
            )
            axis.set(
                title=f"n={rows[0]['n']}, target censoring {float(rows[0]['target_censoring']):.0%}",
                xlabel="Estimated HR",
                ylabel="Density (accepted fits)",
            )
            axis.legend(fontsize=7)
        for axis in axes[len(scenarios) :]:
            axis.set_visible(False)
        coverage = Figure(figsize=(8.8, 4.2), layout="constrained")
        ax = coverage.subplots()
        for model, offset in (("weibull", -0.08), ("exponential", 0.08)):
            rows = [
                next(
                    r
                    for r in summary
                    if int(r["scenario"]) == s and r["model"] == model
                )
                for s in scenarios
            ]
            ax.errorbar(
                np.arange(len(scenarios)) + offset,
                [float(r["coverage"]) for r in rows],
                yerr=[float(r["coverage_mcse"]) for r in rows],
                fmt="o",
                capsize=3,
                color=colors[model],
                label=model.capitalize(),
            )
        labels = [
            next(
                f"n={r['n']}\n{float(r['target_censoring']):.0%} target censoring"
                for r in summary
                if int(r["scenario"]) == s
            )
            for s in scenarios
        ]
        ax.axhline(0.95, color="black", linestyle="--", label="Nominal 95%")
        ax.set(
            xticks=np.arange(len(scenarios)),
            xticklabels=labels,
            ylabel="Coverage (accepted fits)",
            title="Wald 95% coverage ± 1 Monte Carlo SE",
        )
        ax.legend()
        for name, figure in (
            ("fig1_survival.svg", survival),
            ("fig2_sampling.svg", sampling),
            ("fig3_coverage.svg", coverage),
        ):
            _save_figure(figure, out / name)
