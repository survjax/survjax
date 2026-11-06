"""Fit the trial with NUTS and save the draws: uv run --group examples python -m examples.bayesian_rmst.run OUTPUT."""

import argparse
import importlib.metadata
import json
from pathlib import Path

import numpyro

# One CPU device per chain; must run before JAX initializes its backend.
numpyro.set_host_device_count(4)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro.distributions as distributions  # noqa: E402
from numpyro.diagnostics import summary  # noqa: E402
from numpyro.infer import MCMC, NUTS, Predictive  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

import survjax  # noqa: E402
from examples.bayesian_rmst.model import (  # noqa: E402
    TRUE_HAZARD_RATIO,
    TRUE_SCALE,
    TRUE_SHAPE,
    make_trial,
    model,
    use_case_key,
)


def check_factor(trial) -> None:
    """Spec 11.3: NumPyro's log density is the priors plus one summed likelihood."""
    params = {"log_k": jnp.log(1.2), "log_b0": jnp.log(10.0), "gamma": -0.3}
    actual, _ = log_density(model, (trial.data, trial.arm), {}, params)
    base = survjax.Weibull.from_log(
        log_shape=params["log_k"], log_scale=params["log_b0"]
    )
    survival = survjax.ProportionalHazards(
        base, log_hazard_ratio=params["gamma"] * trial.arm
    )
    prior = sum(distributions.Normal(0.0, 5.0).log_prob(v) for v in params.values())
    expected = prior + survival.log_likelihood(trial.data).sum()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def run(output: Path, num_draws: int) -> dict:
    """Sample, save posterior.npz and metadata.json, then apply the acceptance checks.

    Args:
        output: New directory for the results.
        num_draws: Warmup and retained draws per chain (1000 for the recorded run).

    Returns:
        The metadata record.

    Raises:
        AssertionError: An acceptance check fails; the results are saved first.
    """
    trial = make_trial()
    check_factor(trial)

    mcmc = MCMC(
        NUTS(model),
        num_warmup=num_draws,
        num_samples=num_draws,
        num_chains=4,
        chain_method="parallel" if jax.local_device_count() >= 4 else "sequential",
        progress_bar=False,
    )
    mcmc.run(use_case_key(1), trial.data, trial.arm, extra_fields=("diverging",))
    chains = mcmc.get_samples(group_by_chain=True)  # each (4, num_draws)
    samples = mcmc.get_samples()  # each (S,), chain-major
    divergences = int(mcmc.get_extra_fields()["diverging"].sum())

    # Route A: the draws as one (S, 2) batch of distributions (draw, arm).
    base = survjax.Weibull.from_log(
        log_shape=samples["log_k"][:, None], log_scale=samples["log_b0"][:, None]
    )
    posterior = survjax.ProportionalHazards(
        base, log_hazard_ratio=samples["gamma"][:, None] * jnp.array([0.0, 1.0])
    )
    rmst5 = posterior.rmst(5.0)
    # Horizons up to the 90th percentile of observed times, rounded down to half a year.
    tau_max = float(np.floor(np.percentile(trial.data.lower, 90) * 2) / 2)
    tau_grid = jnp.linspace(0.0, tau_max, 101)
    rmst_curve = posterior.rmst(tau_grid[:, None, None])  # (m, S, 2)

    # Route B: the model's deterministic sites, filled by Predictive.
    sites = Predictive(model, posterior_samples=samples)(
        jax.random.key(0), trial.data, trial.arm, compute_rmst=True, tau_grid=tau_grid
    )
    route_difference = max(
        float(jnp.max(jnp.abs(sites["rmst_tau"] - rmst5))),
        float(jnp.max(jnp.abs(jnp.swapaxes(sites["rmst_curve"], 0, 1) - rmst_curve))),
    )

    derived = {
        "k": jnp.exp(samples["log_k"]),
        "HR": jnp.exp(samples["gamma"]),
        "RMST0(5)": rmst5[:, 0],
        "RMST1(5)": rmst5[:, 1],
        "delta_RMST(5)": rmst5[:, 1] - rmst5[:, 0],
    }
    grouped = dict(chains) | {name: v.reshape(4, -1) for name, v in derived.items()}
    stats = summary({name: np.asarray(v) for name, v in grouped.items()})
    diagnostics = {
        name: {"r_hat": float(s["r_hat"]), "ess": float(s["n_eff"])}
        for name, s in stats.items()
    }
    truth_model = survjax.ProportionalHazards(
        survjax.Weibull(shape=TRUE_SHAPE, scale=TRUE_SCALE),
        log_hazard_ratio=jnp.log(TRUE_HAZARD_RATIO) * jnp.array([0.0, 1.0]),
    )
    true_rmst5 = truth_model.rmst(5.0)
    truth = {
        "k": TRUE_SHAPE,
        "HR": TRUE_HAZARD_RATIO,
        "RMST0(5)": float(true_rmst5[0]),
        "RMST1(5)": float(true_rmst5[1]),
        "delta_RMST(5)": float(true_rmst5[1] - true_rmst5[0]),
    }
    intervals = {
        name: [float(q) for q in jnp.quantile(v, jnp.array([0.025, 0.975]))]
        for name, v in derived.items()
    }

    record = {
        "num_chains": 4,
        "num_warmup": num_draws,
        "num_samples": num_draws,
        "chain_method": mcmc.chain_method,
        "root_key": 20261025,
        "priors": dict.fromkeys(("log_k", "log_b0", "gamma"), "Normal(0, 5)"),
        "censor_limit_years": trial.censor_limit,
        "censored": int((~trial.data.exact).sum()),
        "divergences": divergences,
        "diagnostics": diagnostics,
        "truth": truth,
        "intervals_95": intervals,
        "rmst_route_max_abs_difference": route_difference,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("survjax", "jax", "jaxlib", "numpy", "numpyro")
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "posterior.npz",
        time=np.asarray(trial.data.lower),
        event=np.asarray(trial.data.exact),
        arm=np.asarray(trial.arm),
        tau_grid=np.asarray(tau_grid),
        **{f"sample_{name}": np.asarray(v) for name, v in chains.items()},
    )
    (output / "metadata.json").write_text(json.dumps(record, indent=2) + "\n")

    # Acceptance checks of the use case (02-details), after saving the evidence.
    assert route_difference <= 1e-12, f"RMST routes differ by {route_difference}"
    if num_draws < 1000:  # the thresholds are for the recorded configuration only
        return record
    assert divergences == 0, f"{divergences} divergent transitions"
    for name, d in diagnostics.items():
        assert d["r_hat"] < 1.01 and d["ess"] > 400, f"{name}: {d}"
    for name in ("HR", "delta_RMST(5)"):
        low, high = intervals[name]
        assert low <= truth[name] <= high, f"true {name} outside its 95% interval"
    return record


def main() -> None:
    """Parse the command line and run the example in float64."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new directory for the results")
    parser.add_argument("--num-draws", type=int, default=1000)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    record = run(args.output, args.num_draws)
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
