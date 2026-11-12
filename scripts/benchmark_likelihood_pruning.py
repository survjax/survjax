"""Compare prepared concrete/all-true metadata (spec 7.2); one dtype per process."""

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
            conservative = jax.jit(lambda t, e: survjax.SurvivalData(t, e))(
                concrete.lower, concrete.exact
            )
            jax.block_until_ready((concrete, conservative))
            preparation_s = time.perf_counter() - start
            for family in ("exponential", "weibull"):
                parameters = jnp.asarray(
                    [0.75] if family == "exponential" else [1.5, 2.0]
                )

                def objective(p, data, family=family):
                    dist = (
                        survjax.Exponential(rate=p[0])
                        if family == "exponential"
                        else survjax.Weibull(shape=p[0], scale=p[1])
                    )
                    return dist.log_likelihood(data).sum()

                evaluate = jax.jit(jax.value_and_grad(objective))
                results, measurements = [], []
                for label, data in (("concrete", concrete), ("all_true", conservative)):
                    start = time.perf_counter()
                    result = jax.block_until_ready(evaluate(parameters, data))
                    first_s = time.perf_counter() - start
                    for _ in range(3):
                        jax.block_until_ready(evaluate(parameters, data))
                    results.append(result)
                    measurements.append({"metadata": label, "first_s": first_s})
                durations: list[list[float]] = [[], []]
                for repeat in range(repeats):
                    # Interleave both routes and alternate order to limit timing drift.
                    for index in (0, 1) if repeat % 2 == 0 else (1, 0):
                        data = (concrete, conservative)[index]
                        start = time.perf_counter()
                        jax.block_until_ready(evaluate(parameters, data))
                        durations[index].append(time.perf_counter() - start)
                for record, samples in zip(measurements, durations, strict=True):
                    record["median_us"] = float(np.median(samples) * 1e6)
                for actual, expected in zip(results[0], results[1], strict=True):
                    np.testing.assert_allclose(
                        actual,
                        expected,
                        atol=tolerances["ATOL"][x64],
                        rtol=tolerances["RTOL"][x64],
                    )
                print(
                    json.dumps(
                        {
                            "size": size,
                            "kind": kind,
                            "family": family,
                            "preparation_s": preparation_s,
                            "measurements": measurements,
                            "input_bytes": sum(
                                a.nbytes for a in jax.tree.leaves(concrete)
                            )
                            + parameters.nbytes,
                            "output_bytes": sum(
                                a.nbytes for a in jax.tree.leaves(results[0])
                            ),
                        }
                    ),
                    flush=True,
                )


if __name__ == "__main__":
    main()
