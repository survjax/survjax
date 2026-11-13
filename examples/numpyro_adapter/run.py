"""Run with uv run --with numpyro==0.22.0 python -m examples.numpyro_adapter.run.

Discussion example: inference targets the underlying event-time distribution;
the observed site carries exact/right-censoring information. No sampling of
event times or observation records is implemented by this prototype.
"""

import jax
import jax.numpy as jnp
import numpyro
from numpyro import distributions as dist
from numpyro.infer import MCMC, NUTS

import survjax
from examples.numpyro_adapter.adapter import SurvivalLikelihood, prototype_compatibility


def model(data: survjax.SurvivalData) -> None:
    """Fit a rate using a Gamma prior and structured survival observations."""
    rate = numpyro.sample("rate", dist.Gamma(2, 1))
    with numpyro.plate("observations", len(data)):
        numpyro.sample(
            "survival", SurvivalLikelihood(survjax.Exponential(rate=rate)), obs=data
        )


def main() -> None:
    """Compare a short NUTS run with the conjugate Gamma posterior mean."""
    data = survjax.SurvivalData(jnp.array([1.0, 2.0, 3.0]), [1, 0, 1])
    fit = MCMC(NUTS(model), num_warmup=300, num_samples=1000, progress_bar=False)
    fit.run(jax.random.key(42), data)
    print("Posterior: Gamma(shape=4, rate=7); analytical mean:", 4 / 7)
    print("NUTS mean:", float(fit.get_samples()["rate"].mean()))
    print("Divergences:", int(fit.get_extra_fields()["diverging"].sum()))
    fit.print_summary()


if __name__ == "__main__":
    with prototype_compatibility():
        main()
