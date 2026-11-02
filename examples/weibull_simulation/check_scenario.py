"""JAX_ENABLE_X64=1 uv run --locked --group examples pytest -q <this file>."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
import impl_tolerances as tol  # noqa: E402

import survjax  # noqa: E402
from examples.weibull_simulation.run import (  # noqa: E402
    failure_columns,
    failure_counts,
    fit_batch,
    prepare_scenario,
)
from examples.weibull_simulation.study import _assess_fit, fit_replicate  # noqa: E402

if not jax.config.x64_enabled:
    raise RuntimeError("Run scenario checks with JAX_ENABLE_X64=1")

ATOL, RTOL = tol.ATOL[True], tol.RTOL[True]


def test_scenario_matches_prescribed_keys_and_inverse_cdf():
    """01 DGP/seeds: fold scenario, split R, then split U/C independently."""
    data, arm = prepare_scenario(2, 6, 2, 20.0)
    np.testing.assert_array_equal(arm, [0, 0, 0, 1, 1, 1])
    keys = jax.random.split(jax.random.fold_in(jax.random.key(20261025), 2), 2)
    for i, key in enumerate(keys):
        uk, ck = jax.random.split(key)
        u = np.asarray(jax.random.uniform(uk, (6,), dtype=jnp.float64))
        t = 11.42 * (-np.log(u) * np.exp(-np.log(0.7) * np.asarray(arm))) ** (1 / 1.38)
        c = 20 * np.asarray(jax.random.uniform(ck, (6,), dtype=jnp.float64))
        np.testing.assert_allclose(
            data.lower[i], np.minimum(t, c), atol=ATOL, rtol=RTOL
        )
        np.testing.assert_array_equal(data.exact[i], t <= c)
    assert data.shape == (2, 6) and data.lower.dtype == jnp.float64


@pytest.mark.parametrize("n,r,c", [(3, 2, 20.0), (6, 0, 20.0), (6, 2, 0.0)])
def test_scenario_rejects_invalid_configuration(n, r, c):
    """01 DGP: positive even allocation, positive R and a finite positive c."""
    with pytest.raises(ValueError):
        prepare_scenario(0, n, r, c)


@pytest.mark.parametrize("model", ["weibull", "exponential"])
def test_compiled_batch_agrees_with_single_replicate_fits(model):
    """01 fitting: compiled vmap retains the reviewed per-cohort fit semantics."""
    time = jnp.array([1.0, 2.0, 3.0, 2.0, 4.0, 6.0])
    event = jnp.array([1, 0, 1, 1, 1, 0], dtype=bool)
    arm = jnp.array([0, 0, 0, 1, 1, 1])
    data = survjax.SurvivalData(
        jnp.stack([time, time * 2]), event=jnp.stack([event, event])
    )
    data.issues.raise_if_invalid()
    batched = fit_batch(data, arm, model=model)
    for i in range(2):
        reference = fit_replicate(data[i], arm, model=model)
        assert batched.accepted[i] and reference.accepted
        for name in ("theta", "objective", "hessian", "log_hr_se", "hr_interval"):
            np.testing.assert_allclose(
                getattr(batched, name)[i],
                getattr(reference, name),
                atol=ATOL,
                rtol=RTOL,
            )
        # Scores belong to slightly different converged theta; compare each
        # against the exact PH score at its own point, not another fit's residue.
        t, d, x = map(np.asarray, (data.lower[i], event, arm))
        for p, score in (
            (batched.theta[i], batched.gradient[i]),
            (reference.theta, reference.gradient),
        ):
            p = np.asarray(p)
            if model == "weibull":
                k, log_b, gamma = np.exp(p[0]), p[1], p[2]
                z = np.log(t) - log_b
                delta = np.exp(k * z + gamma * x) - d
                expected_score = [
                    np.sum(k * z * delta - d),
                    -k * np.sum(delta),
                    np.sum(x * delta),
                ]
            else:
                delta = (
                    np.exp(p[0] + p[1] * np.asarray(arm)) * np.asarray(data.lower[i])
                    - d
                )
                expected_score = [np.sum(delta), np.sum(x * delta)]
            np.testing.assert_allclose(score, expected_score, atol=ATOL, rtol=RTOL)
            assert np.max(np.abs(score)) < 1e-5


@pytest.mark.parametrize("score", [2e-5, 1e-5, np.nan])
def test_failure_reasons_overlap_but_failed_total_is_unique(score):
    """D-033: one rejected attempt can fail four gates; totals count it once."""
    p, good_h = jnp.zeros(2), jnp.array([[4.0, 2.0], [2.0, 2.0]])
    good = _assess_fit(p, jnp.array(1.0), p, good_h, jnp.array(True))
    bad = _assess_fit(
        p,
        jnp.array(jnp.nan),
        jnp.array([score, 0.0]),
        jnp.array([[1.0, 2.0], [2.0, 1.0]]),
        jnp.array(False),
    )
    # Unestimable diagnostics are placeholders, not failed optimizer attempts.
    missing = bad._replace(estimable=jnp.array(False))
    fits = jax.tree.map(lambda *xs: jnp.stack(xs), good, bad, missing)
    columns = jax.jit(failure_columns)(fits)
    np.testing.assert_array_equal(columns["failed"], [False, True, True])
    np.testing.assert_array_equal(columns["unestimable"], [False, False, True])
    for reason in ("solver", "nonfinite", "gradient", "hessian"):
        np.testing.assert_array_equal(columns[reason], [False, True, False])
    assert failure_counts(fits) == {
        "attempted": 3,
        "accepted": 1,
        "failed": 2,
        "solver": 1,
        "nonfinite": 1,
        "gradient": 1,
        "hessian": 1,
        "unestimable": 1,
    }


def test_real_optimizer_failure_is_counted():
    """SIM.4 N-3: default BFGS failure must reach the solver failure column."""
    # Tied exact times have unbounded Weibull shape likelihood (log L~4 log k).
    # In the locked environment, this valid small-scale cohort causes a real
    # solver failure at extreme iterates; no claim of tail derivative accuracy.
    data = survjax.SurvivalData(jnp.full(4, 1e-50), event=jnp.ones(4, dtype=bool))
    data.issues.raise_if_invalid()
    arm = jnp.array([0, 0, 1, 1])
    one = jax.jit(lambda d: fit_replicate(d, arm, model="weibull"))(data)
    fits = jax.tree.map(lambda value: value[None], one)
    assert fits.estimable[0] and not fits.solver_success[0] and not fits.accepted[0]
    counts = failure_counts(fits)
    assert counts["attempted"] == counts["failed"] == counts["solver"] == 1
    assert counts["accepted"] == 0
