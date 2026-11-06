"""Compact, auditable plotting evidence; full fit diagnostics remain local (D-035)."""

import argparse
import csv
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np

_ILLUSTRATION_FIELDS = (
    "scenario model replicate parameter_count theta_0 theta_1 theta_2".split()
)
_FAILURE_FIELDS = "scenario model replicate failed solver nonfinite gradient hessian unestimable max_abs_gradient log_hr".split()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _csv(rows: list[dict], fields: list[str]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_evidence(source: Path, out: Path) -> None:
    """Reduce complete raw exports to the approved compact evidence bundle.

    Args:
        source: Directory containing performance.csv, replicates.csv, metadata.json.
        out: Destination (may equal source); existing compact evidence is replaced.
    Raises:
        ValueError: Missing/duplicate IDs, inconsistent counts/flags or invalid HRs.
        OSError: Input/output failure. No simulation or fitting occurs here.
        subprocess.CalledProcessError: Reducer Git provenance cannot be read.
    """
    summaries = _read_csv(source / "performance.csv")
    raw = _read_csv(source / "replicates.csv")

    def key(r):
        return r["scenario"], r["model"]

    groups = {key(r): [] for r in summaries}
    if len(groups) != len(summaries):
        raise ValueError("duplicate summary group")
    # Host serialization/auditing of existing CSV records; never numerical fitting.
    for row in raw:
        if key(row) not in groups:
            raise ValueError("unexpected replicate group")
        groups[key(row)].append(row)
    failures, series, accepted_hr = [], [], []
    reasons = "failed solver nonfinite gradient hessian unestimable".split()
    for summary in summaries:
        cohort = groups[key(summary)]
        attempted, accepted = int(summary["attempted"]), int(summary["accepted"])
        if len(cohort) != attempted or {int(r["replicate"]) for r in cohort} != set(
            range(attempted)
        ):
            raise ValueError("replicate IDs must be complete and unique")
        for row in cohort:
            flags = [row[name] for name in ("accepted", *reasons)]
            if any(value not in ("0", "1") for value in flags):
                raise ValueError("flags must be binary")
            failed = row["failed"] == "1"
            if failed == (row["accepted"] == "1") or failed != any(
                row[name] == "1" for name in reasons[1:]
            ):
                raise ValueError("inconsistent acceptance/failure flags")
            if failed:
                p = int(row["parameter_count"])
                score = np.array([float(row[f"gradient_{i}"]) for i in range(p)])
                failures.append(
                    {
                        **{
                            name: row[name]
                            for name in (
                                "scenario",
                                "model",
                                "replicate",
                                *reasons,
                                "log_hr",
                            )
                        },
                        "max_abs_gradient": np.max(np.abs(score)),
                    }
                )
        if sum(int(r["accepted"]) for r in cohort) != accepted:
            raise ValueError("accepted count mismatch")
        for reason in reasons:
            if sum(int(r[reason]) for r in cohort) != int(summary[reason]):
                raise ValueError("failure count mismatch")
        hr = np.exp([float(r["log_hr"]) for r in cohort if r["accepted"] == "1"])
        if not np.all(np.isfinite(hr) & (hr > 0)):
            raise ValueError("accepted HRs must be finite and positive")
        accepted_hr.append(hr)
    pooled = np.concatenate(accepted_hr)
    # Outward bin edges describe the histogram, never modify estimates/probabilities.
    lo = np.floor(pooled.min() / 0.025) * 0.025 if pooled.size else 0.0
    hi = (
        max(lo + 0.025, np.ceil(pooled.max() / 0.025) * 0.025) if pooled.size else 0.025
    )
    # Binary multiplication can round a lattice edge inward by an ulp.
    if pooled.size and lo > pooled.min():
        lo -= 0.025
    if pooled.size and hi < pooled.max():
        hi += 0.025
    edges = np.linspace(lo, hi, round((hi - lo) / 0.025) + 1)
    for summary, hr in zip(summaries, accepted_hr, strict=True):
        counts, _ = np.histogram(hr, bins=edges)
        if counts.sum() != len(hr):
            raise ValueError("histogram omitted accepted estimates")
        series.append(
            {
                "scenario": int(summary["scenario"]),
                "model": summary["model"],
                "accepted": len(hr),
                "counts": counts.tolist(),
            }
        )
    # One common accepted replicate, at the specified illustration setting.
    illustration = []
    target_n = "300" if any(r["n"] == "300" for r in summaries) else "100"
    selected = [
        r for r in summaries if r["n"] == target_n and float(r["censor_limit"]) == 20.0
    ]
    if len(selected) == 2:
        cohorts = [groups[key(r)] for r in selected]
        common = set.intersection(
            *({int(r["replicate"]) for r in c if r["accepted"] == "1"} for c in cohorts)
        )
        if common:
            illustration = [
                {name: row[name] for name in _ILLUSTRATION_FIELDS}
                for c in cohorts
                for row in c
                if int(row["replicate"]) == min(common)
            ]
    payloads = {
        "performance.csv": (source / "performance.csv").read_bytes(),
        "metadata.json": (source / "metadata.json").read_bytes(),
        "failures.csv": _csv(failures, _FAILURE_FIELDS).encode(),
        "illustration.csv": _csv(illustration, _ILLUSTRATION_FIELDS).encode(),
        "sampling_histograms.json": (
            json.dumps(
                {
                    "quantity": "HR = exp(log_hr), accepted fits only",
                    "edges": edges.tolist(),
                    "series": series,
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode(),
    }
    reducer_root = Path(__file__).resolve().parents[2]
    reducer_path = str(Path(__file__).resolve().relative_to(reducer_root))
    manifest = {
        "schema_version": 1,
        "reducer_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
        ).strip(),
        "reducer_path": reducer_path,
        "reducer_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--", reducer_path],
                cwd=reducer_root,
                text=True,
            ).strip()
        ),
        "reducer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("performance.csv", "metadata.json", "replicates.csv")
        },
        "output_sha256": {
            name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
        },
        "accepted_ids": "complement of failures within each summary's 0..attempted-1 IDs",
        "histogram": "common .025 HR bins; [left,right), final bin includes right edge; every accepted fit included",
        "raw_diagnostics": "local only; regenerate with run.py using recorded source/environment/seed; cross-platform bitwise equality is not promised",
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "evidence.json").unlink(missing_ok=True)
    for name, data in payloads.items():
        (out / name).write_bytes(data)
    (out / "evidence.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    write_evidence(args.results, args.out)
