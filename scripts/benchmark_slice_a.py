"""Reproduce the Slice A CPU baseline (spec 7.2/12.3), one dtype per process."""

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import runpy
import shlex
import subprocess
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import survjax

SEED = 20261029
SIZES = (1_000, 10_000, 100_000)
TOLERANCES = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tests/impl_tolerances.py")
)


def _timed(call):
    start = time.perf_counter()
    result = jax.block_until_ready(call())
    return result, time.perf_counter() - start


def _bytes(tree):
    return sum(leaf.nbytes for leaf in jax.tree.leaves(tree))


def _distribution(family, p):
    if family == "exponential":
        return survjax.Exponential(rate=p[0])
    return survjax.Weibull(shape=p[0], scale=p[1])


def _analytical(family, parameters, times, events, shared):
    # Independent exact/right formulas, differentiated in physical parameters.
    p, t, e = (np.asarray(a, dtype=np.float64) for a in (parameters, times, events))
    if family == "exponential":
        r = p[0]
        value = e * np.log(r) - r * t
        gradient = np.array([e / r - t])
    else:
        k, b = p
        u = np.log(t / b)
        h = np.exp(k * u)
        value = e * (np.log(k) - np.log(b) + (k - 1) * u) - h
        gradient = np.array([e / k + (e - h) * u, k * (h - e) / b])
    if shared:
        gradient = gradient.sum(axis=1)
    return value.sum(), gradient


def _check(result, expected):
    (value, (data_issues, parameter_issues)), gradient = result
    assert bool(data_issues.ok) and bool(parameter_issues.ok)
    x64 = bool(jax.config.x64_enabled)
    for actual, wanted in zip((value, gradient), expected, strict=True):
        np.testing.assert_allclose(
            actual,
            wanted,
            atol=TOLERANCES["ATOL"][x64],
            rtol=TOLERANCES["RTOL"][x64],
        )


def _host_inputs(family, n, shared, dtype):
    # Resetting the seed gives both paths/layouts the same observation cohort.
    rng = np.random.default_rng(SEED)
    times = rng.uniform(0.25, 4, n).astype(dtype)
    events = rng.random(n) < 0.65
    if shared:
        p = np.array([0.7] if family == "exponential" else [1.3, 2.0], dtype=dtype)
    else:
        p = np.array(
            [rng.uniform(0.5, 0.9, n)]
            if family == "exponential"
            else [rng.uniform(1.1, 1.5, n), rng.uniform(1.8, 2.2, n)],
            dtype=dtype,
        )
    return p, times, events


def _row(family, n, shared, path, device, warmups, repeats):
    dtype = np.float64 if jax.config.x64_enabled else np.float32
    host, generation = _timed(lambda: _host_inputs(family, n, shared, dtype))
    (p, t, e), placement = _timed(lambda: jax.device_put(host, device))
    data, construction = _timed(lambda: survjax.SurvivalData(t, e))
    traces, flags = [], []

    def objective(parameters, *inputs):
        traces.append(None)
        observations = (
            inputs[0] if path == "prepared" else survjax.SurvivalData(*inputs)
        )
        flags.append(observations._may_contain)
        dist = _distribution(family, parameters)
        value = dist.log_likelihood(observations).sum()
        return value, (observations.issues, dist.parameter_issues())

    compiled = jax.jit(jax.value_and_grad(objective, has_aux=True))
    inputs = (data,) if path == "prepared" else (t, e)
    expected = _analytical(family, *host, shared)
    result, first = _timed(lambda: compiled(p, *inputs))
    _check(result, expected)
    for _ in range(warmups):
        jax.block_until_ready(compiled(p, *inputs))
    samples = [_timed(lambda: compiled(p, *inputs))[1] for _ in range(repeats)]
    # spec 7.2: changed values and parent slices must retain shape/dtype/flags.
    parent = survjax.SurvivalData(
        jnp.concatenate((t, t * 1.05)), jnp.concatenate((e, ~e))
    )
    changed_p = p * 1.01
    for batch in (parent[:n], parent[n:]):
        batch_inputs = (
            (batch,)
            if path == "prepared"
            else (batch.lower, jnp.asarray(batch.exact, dtype=e.dtype))
        )
        changed = jax.block_until_ready(compiled(changed_p, *batch_inputs))
        _check(
            changed, _analytical(family, changed_p, batch.lower, batch.exact, shared)
        )
    assert len(traces) == 1, f"unexpected recompilation: {family}/{path}/{n}"
    assert flags[0] == (
        (True, True, False, False, False, False) if path == "prepared" else (True,) * 6
    )
    assert all(leaf.device == device for leaf in jax.tree.leaves(result))
    return {
        "family": family,
        "n": n,
        "parameters": "shared" if shared else "per_observation",
        "path": path,
        "parameter_shape": list(p.shape),
        "time_event_shape": list(t.shape),
        "static_flags": list(flags[0]),
        "host_generation_seconds": generation,
        "initial_placement_seconds": placement,
        "prepared_construction_seconds": construction,
        "compile_and_first_call_seconds": first,
        "warm_seconds": samples,
        "warm_min_seconds": min(samples),
        "warm_median_seconds": float(np.median(samples)),
        "warm_max_seconds": max(samples),
        "logical_input_bytes": _bytes((p, *inputs)),
        "logical_output_bytes": _bytes(result),
        "correctness": "analytical value/gradient passed",
        "reuse": "one trace for initial values and two changed parent minibatches",
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.warmups < 1 or args.repeats < 2:
        parser.error("require at least one warmup and two repetitions")
    device = jax.devices("cpu")[0]
    dtype = "float64" if jax.config.x64_enabled else "float32"
    processor = platform.processor()
    if sys.platform == "darwin":
        processor = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    run = {
        "library_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "library_worktree_status": subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", "src"],
            text=True,
        ).strip(),
        "utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "command": f"JAX_ENABLE_X64={int(jax.config.x64_enabled)} JAX_PLATFORMS=cpu uv run python {shlex.join(sys.argv)}",
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "numpy", "survjax")
        },
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": processor,
        "logical_cpus": os.cpu_count(),
        "device": str(device),
        "platform": device.platform,
        "dtype": dtype,
        "seed": SEED,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "timer": "time.perf_counter; block_until_ready on every output leaf",
        "peak_memory": "not measured; logical bytes are sums of array leaf nbytes, not peak memory",
        "first_call": "includes tracing, compilation and first synchronized execution",
        "preparation": "host generation, explicit placement, concrete construction timed separately for each row",
        "rows": [],
    }
    with jax.default_device(device):
        for family in ("exponential", "weibull"):
            for n in SIZES:
                for shared in (True, False):
                    for path in ("prepared", "traced"):
                        row = _row(
                            family, n, shared, path, device, args.warmups, args.repeats
                        )
                        run["rows"].append(row)
                        print(
                            f"{dtype} {family} N={n} {row['parameters']} {path}: {row['warm_median_seconds']:.6g}s",
                            flush=True,
                        )
    report = (
        json.loads(args.output.read_text()) if args.output.exists() else {"runs": {}}
    )
    report["runs"][dtype] = run
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"Recorded {len(run['rows'])} checked rows in {args.output}")


if __name__ == "__main__":
    _main()
