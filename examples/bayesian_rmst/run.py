"""Run from checkout: uv run --group examples examples/bayesian_rmst/run.py."""

import argparse
import importlib.metadata
import json
import sys
import time
from pathlib import Path


def main() -> None:
    """Run approved smoke/full NUTS settings and print reproducibility metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=("smoke", "full"), default="smoke")
    args = parser.parse_args()

    import numpyro

    # Configure devices and precision before any JAX arrays/backends initialize.
    numpyro.set_host_device_count(4)
    import jax

    jax.config.update("jax_enable_x64", True)
    from numpyro.diagnostics import summary
    from numpyro.infer import MCMC, NUTS

    # File execution adds this directory, rather than the checkout root.
    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.bayesian_rmst.model import (
        check_rmst_routes,
        make_trial,
        model,
        posterior_draws,
        summarize_draws,
        time_grids,
        use_case_key,
    )

    trial = make_trial()
    draws = 100 if args.config == "smoke" else 1000
    chain_method = "parallel" if jax.local_device_count() >= 4 else "vectorized"
    sampler = MCMC(
        NUTS(model),
        num_warmup=draws,
        num_samples=draws,
        num_chains=4,
        chain_method=chain_method,
        progress_bar=False,
    )
    start = time.perf_counter()
    sampler.run(use_case_key(1), trial.data, trial.arm, extra_fields=("diverging",))
    samples = sampler.get_samples(group_by_chain=True)
    extra = sampler.get_extra_fields(group_by_chain=True)
    jax.block_until_ready((samples, extra))
    elapsed = time.perf_counter() - start
    diagnostics = summary(samples, group_by_chain=True)
    post_start = time.perf_counter()
    t_grid, tau_grid = time_grids(trial.data)
    flat_samples = sampler.get_samples()
    derived = posterior_draws(flat_samples, t_grid, tau_grid)
    route_error = check_rmst_routes(flat_samples, trial, tau_grid, derived)
    table, positive_probability = summarize_draws(derived)
    jax.block_until_ready(derived)
    post_elapsed = time.perf_counter() - post_start
    record = {
        "config": args.config,
        "posterior_table": table,
        "probability_delta_rmst_positive": positive_probability,
        "tau_max_years": float(tau_grid[-1]),
        "rmst_route_max_absolute_error": route_error,
        "postprocessing_seconds_including_compile": post_elapsed,
        "root_seed": 20261025,
        "use_case": 2,
        "nuts_stream": 1,
        "chains": 4,
        "warmup": draws,
        "draws_per_chain": draws,
        "chain_method": chain_method,
        "initialization": "NumPyro default init_to_uniform",
        "priors": {
            name: {"distribution": "Normal", "loc": 0.0, "scale": 5.0}
            for name in ("log_k", "log_b0", "gamma")
        },
        "x64": jax.config.x64_enabled,
        "backend": jax.default_backend(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("survjax", "jax", "jaxlib", "numpy", "numpyro")
        },
        "censor_limit_years": trial.censor_limit,
        "censored_fraction": float(jax.numpy.mean(~trial.data.exact)),
        "sampling_seconds_including_compile": elapsed,
        "divergences_per_chain": extra["diverging"].sum(axis=1).tolist(),
        "numpyro_diagnostics": {
            name: {key: float(value) for key, value in values.items()}
            for name, values in diagnostics.items()
        },
        "diagnostic_scope": "pilot only; full bulk/tail ESS and truth audit pending",
    }
    print(json.dumps(record, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
