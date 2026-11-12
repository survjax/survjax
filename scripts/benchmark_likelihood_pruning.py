"""Compare prepared metadata and traced construction (spec 7.2/12.3)."""

import json
import platform
import runpy
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import survjax


def main():
    seed, repeats = 20261112, 30
    tolerances = runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "tests/impl_tolerances.py")
    )
    x64 = jax.config.x64_enabled
    print(
        json.dumps(
            {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python": platform.python_version(),
                "jax": jax.__version__,
                "devices": str(jax.devices()),
                "x64": x64,
                "seed": seed,
                "repeats": repeats,
                "peak_memory": "not measured",
            }
        )
    )
    rng = np.random.default_rng(seed)
    for size in (1000, 10000, 100000):
        times = rng.uniform(0.5, 4.0, size).astype(np.float64 if x64 else np.float32)
        for kind in ("exact", "censored", "mixed"):
            events = (
                np.ones(size, dtype=bool)
                if kind == "exact"
                else np.zeros(size, dtype=bool)
            )
            if kind == "mixed":
                events[::2] = True
            start = time.perf_counter()
            concrete = survjax.SurvivalData(times, events)
            # Rev 62: traced time/event now rules out left/interval/truncation.
            # Supplying exact and both windows in bounds form retains all flags.
            conservative = jax.jit(
                lambda d: survjax.SurvivalData.from_bounds(
                    d.lower,
                    d.upper,
                    exact=d.exact,
                    left_trunc=d.left_trunc,
                    right_trunc=d.right_trunc,
                )
            )(concrete)
            assert conservative.tree_flatten()[1] == (True,) * 6
            jax.block_until_ready((concrete, conservative))
            preparation_s = time.perf_counter() - start
            for family in ("exponential", "weibull"):
                start = time.perf_counter()
                parameters = jnp.asarray(
                    [0.75] if family == "exponential" else [1.5, 2.0]
                )
                jax.block_until_ready(parameters)
                parameter_preparation_s = time.perf_counter() - start

                def objective(p, data, family=family):
                    dist = (
                        survjax.Exponential(rate=p[0])
                        if family == "exponential"
                        else survjax.Weibull(shape=p[0], scale=p[1])
                    )
                    return dist.log_likelihood(data).sum()

                prepared_traces, construction_traces = [], []

                @jax.jit
                def evaluate(p, data, traces=prepared_traces, loss=objective):
                    traces.append(None)
                    return jax.value_and_grad(loss)(p, data), data.issues.ok

                @jax.jit
                def construct_and_evaluate(
                    p, time, event, traces=construction_traces, loss=objective
                ):
                    data = survjax.SurvivalData(time, event)
                    traces.append(data.tree_flatten()[1])
                    # Return diagnostics so compilation cannot remove that work.
                    return jax.value_and_grad(loss)(p, data), data.issues.ok

                routes = (
                    ("concrete", evaluate, (parameters, concrete)),
                    ("all_true", evaluate, (parameters, conservative)),
                    (
                        "traced_construction",
                        construct_and_evaluate,
                        (parameters, concrete.lower, concrete.exact),
                    ),
                )
                results, measurements = [], []
                for label, run, args in routes:
                    start = time.perf_counter()
                    result = jax.block_until_ready(run(*args))
                    first_s = time.perf_counter() - start
                    for _ in range(3):
                        jax.block_until_ready(run(*args))
                    results.append(result)
                    metadata = (
                        construction_traces[0]
                        if label == "traced_construction"
                        else args[1].tree_flatten()[1]
                    )
                    measurements.append(
                        {
                            "route": label,
                            "flags": metadata,
                            "first_s": first_s,
                            "input_bytes": sum(a.nbytes for a in jax.tree.leaves(args)),
                            "output_bytes": sum(
                                a.nbytes for a in jax.tree.leaves(result)
                            ),
                        }
                    )
                durations: list[list[float]] = [[], [], []]
                for repeat in range(repeats):
                    # Rotate the starting route and reverse order to limit drift.
                    order = [(repeat + i) % 3 for i in range(3)]
                    for index in order if repeat % 2 == 0 else order[::-1]:
                        _, run, args = routes[index]
                        start = time.perf_counter()
                        jax.block_until_ready(run(*args))
                        durations[index].append(time.perf_counter() - start)
                for record, samples in zip(measurements, durations, strict=True):
                    record["median_us"] = float(np.median(samples) * 1e6)
                for result in results:
                    assert bool(result[1])
                    for actual, expected in zip(
                        jax.tree.leaves(result),
                        jax.tree.leaves(results[0]),
                        strict=True,
                    ):
                        np.testing.assert_allclose(
                            actual,
                            expected,
                            atol=tolerances["ATOL"][x64],
                            rtol=tolerances["RTOL"][x64],
                        )
                trace_counts = (len(prepared_traces), len(construction_traces))
                # Changed values and same-shape subsets preserve static structure.
                for data in (concrete, conservative):
                    jax.block_until_ready(evaluate(parameters * 1.01, data[::-1]))
                jax.block_until_ready(
                    construct_and_evaluate(
                        parameters * 1.01,
                        concrete.lower * 1.01,
                        ~concrete.exact,
                    )
                )
                assert trace_counts == (len(prepared_traces), len(construction_traces))
                print(
                    json.dumps(
                        {
                            "size": size,
                            "kind": kind,
                            "family": family,
                            "preparation_s": preparation_s,
                            "parameter_preparation_s": parameter_preparation_s,
                            "compilation_reused": True,
                            "measurements": measurements,
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
