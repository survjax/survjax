"""Optional issue #1 probe: run with uv run --with numpyro==0.22.0 pytest ... ."""

import importlib
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax

numpyro = pytest.importorskip("numpyro")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
adapter = importlib.import_module("examples.numpyro_adapter.adapter")
util = importlib.import_module("numpyro.infer.util")


@pytest.fixture(autouse=True)
def compatibility():
    had_ndim = hasattr(survjax.SurvivalData, "ndim")
    with adapter.prototype_compatibility():
        yield
    assert hasattr(survjax.SurvivalData, "ndim") == had_ndim


def assert_close(actual, expected, x64):
    np.testing.assert_allclose(actual, expected, atol=ATOL[x64], rtol=RTOL[x64])


@pytest.mark.parametrize("events", [[1, 1, 1], [0, 0, 0], [1, 0, 1]])
def test_mask_preserves_metadata_and_analytical_derivatives(events, x64_enabled):
    """Spec 3.2/7.2–7.3: excluded invalid rows have no parameter derivatives."""
    data = survjax.SurvivalData(jnp.array([1.0, jnp.nan, 3.0]), events)
    assert data.ndim == 1 and data[0].ndim == 0
    keep = jnp.array([True, False, True])

    def score(log_rate):
        survival = survjax.Exponential(rate=jnp.exp(log_rate))
        return adapter.SurvivalLikelihood(survival).mask(keep).log_prob(data).sum()

    rate = 0.5
    log_rate = jnp.log(rate)
    events_kept = events[0] + events[2]
    # Exponential: d log L / d log(rate) = events - rate * total exposure.
    expected = events_kept * log_rate - 4 * rate
    assert_close(score(log_rate), expected, x64_enabled)
    assert_close(jax.jit(score)(log_rate), expected, x64_enabled)
    assert_close(jax.grad(score)(log_rate), events_kept - 4 * rate, x64_enabled)
    assert_close(jax.hessian(score)(log_rate), -4 * rate, x64_enabled)
    assert_close(
        jax.vmap(score)(jnp.array([log_rate, log_rate])), expected, x64_enabled
    )
    fn = adapter.SurvivalLikelihood(survjax.Exponential(rate=rate))
    safe = adapter._select_records(keep, data, fn.support.feasible_like(data))
    assert safe._may_contain == data._may_contain
    assert not np.any(safe.issues.invalid_rows)


@pytest.mark.parametrize("mask_outside", [False, True])
def test_plate_and_nested_masks_broadcast_in_either_order(mask_outside, x64_enabled):
    """Issue #1: handlers must retain structured masking through plate expansion."""
    data = survjax.SurvivalData(jnp.array([1.0, 2.0, 3.0]), [1, 7, 0])
    keep = jnp.array([[True, False, True], [False, False, True]])

    def model():
        contexts = [numpyro.plate("rows", 3), numpyro.handlers.mask(mask=keep)]
        if mask_outside:
            contexts.reverse()
        with (
            contexts[0],
            contexts[1],
            numpyro.handlers.mask(mask=jnp.array([1, 1, 0], bool)),
        ):
            numpyro.sample(
                "survival",
                adapter.SurvivalLikelihood(survjax.Exponential(rate=0.5)),
                obs=data,
            )

    site = numpyro.handlers.trace(model).get_trace()["survival"]
    # Only row 0 in batch 0 survives both masks: log(rate) - rate*time.
    expected = jnp.array([[jnp.log(0.5) - 0.5, 0, 0], [0, 0, 0]])
    assert_close(site["fn"].log_prob(site["value"]), expected, x64_enabled)


def test_factor_agreement_and_pointwise_extraction(x64_enabled):
    """Spec 6.3: the observed site contributes each row exactly once."""
    data = survjax.SurvivalData(jnp.array([1.0, 2.0, 3.0]), [1, 0, 1])

    def model(data, factor=False):
        rate = numpyro.sample("rate", numpyro.distributions.Gamma(2, 1))
        survival = survjax.Exponential(rate=rate)
        with numpyro.plate("rows", len(data)):
            if factor:
                numpyro.factor("survival", survival.log_likelihood(data))
            else:
                numpyro.sample(
                    "survival", adapter.SurvivalLikelihood(survival), obs=data
                )

    for subset in (data, data[jnp.array([0, 2])]):
        actual, _ = util.log_density(model, (subset,), {}, {"rate": 0.5})
        factor, _ = util.log_density(model, (subset,), {"factor": True}, {"rate": 0.5})
        assert_close(actual, factor, x64_enabled)
    terms = util.log_likelihood(model, {"rate": jnp.array([0.5, 1.0])}, data)[
        "survival"
    ]
    # Hand-derived: event*log(rate) - rate*time, one row per observation.
    expected = jnp.array([[jnp.log(0.5) - 0.5, -1, jnp.log(0.5) - 1.5], [-1, -2, -3]])
    assert_close(terms, expected, x64_enabled)


def test_arrays_validation_and_sampling_boundary(x64_enabled):
    """Spec 4.5/4.7: arrays score density; invalid records retain diagnostics."""
    fn = adapter.SurvivalLikelihood(survjax.Exponential(rate=0.5), validate_args=True)
    assert_close(
        fn.log_prob(jnp.array([1.0, 2.0])),
        jnp.log(0.5) - jnp.array([0.5, 1]),
        x64_enabled,
    )
    with pytest.raises(ValueError, match="survival"):
        adapter.SurvivalLikelihood(survjax.Exponential(rate=-1.0), validate_args=True)
    bad = survjax.SurvivalData(jnp.array([1.0]), [7])
    with pytest.warns(UserWarning, match="support"):
        assert np.isnan(fn.log_prob(bad)).all()
    assert_close(fn.mask(False).log_prob(bad), jnp.zeros(1), x64_enabled)
    with pytest.raises(NotImplementedError, match="sampling"):
        fn.sample(jax.random.key(0))
