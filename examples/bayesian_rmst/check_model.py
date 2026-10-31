"""Explicit optional checks: JAX_ENABLE_X64=1 uv run --group examples python -m examples.bayesian_rmst.check_model."""

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as distributions
from numpyro.infer import Predictive
from numpyro.infer.util import log_density

import survjax
from examples.bayesian_rmst.model import make_trial, model, use_case_key


def check_trial() -> None:
    """02-details: balanced fixed trial, valid float64 rows and stable keys."""
    trial = make_trial()
    repeated = make_trial()
    assert trial.data.shape == (300,)
    assert trial.data.lower.dtype == jnp.float64
    np.testing.assert_array_equal(trial.arm, np.repeat([0.0, 1.0], 150))
    for left, right in zip(
        jax.tree.leaves(trial.data), jax.tree.leaves(repeated.data), strict=True
    ):
        np.testing.assert_array_equal(left, right)
    assert bool(trial.data.issues.ok)
    assert trial.censor_limit == repeated.censor_limit
    root = jax.random.key(20261025)
    for stream in range(4):
        expected = jax.random.fold_in(jax.random.fold_in(root, 2), stream)
        np.testing.assert_array_equal(
            jax.random.key_data(use_case_key(stream)), jax.random.key_data(expected)
        )
    print(f"trial checks passed; c={trial.censor_limit:.15g} years")


def check_factor() -> None:
    """Spec 11.3: log density equals priors plus one full-data likelihood."""
    trial = make_trial()
    params = {"log_k": jnp.log(1.2), "log_b0": jnp.log(10.0), "gamma": -0.3}
    actual, trace = log_density(model, (trial.data, trial.arm), {}, params)
    base = survjax.Weibull.from_log(
        log_shape=params["log_k"], log_scale=params["log_b0"]
    )
    survival = survjax.ProportionalHazards(
        base, log_hazard_ratio=params["gamma"] * trial.arm
    )
    prior = (
        distributions.Normal(0.0, 0.5).log_prob(params["log_k"])
        + distributions.Normal(jnp.log(10.0), 1.5).log_prob(params["log_b0"])
        + distributions.Normal(0.0, 1.0).log_prob(params["gamma"])
    )
    expected = prior + survival.log_likelihood(trial.data).sum()
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)
    factor = trace["survival"]
    assert factor["fn"].batch_shape == (300,)
    assert [
        (frame.name, frame.dim, frame.size) for frame in factor["cond_indep_stack"]
    ] == [("observations", -1, 300)]
    assert "rmst_tau" not in trace and "rmst_curve" not in trace
    grid = jnp.linspace(0.0, 5.0, 3)
    result = Predictive(
        model,
        posterior_samples={
            name: jnp.asarray([value]) for name, value in params.items()
        },
    )(use_case_key(2), trial.data, trial.arm, compute_rmst=True, tau_grid=grid)
    assert result["rmst_tau"].shape == (1, 2)
    assert result["rmst_curve"].shape == (1, 3, 2)
    print(
        f"factor/plate/deterministic-site checks passed; density={float(actual):.15g}"
    )


if __name__ == "__main__":
    check_trial()
    check_factor()
