"""Ten comparable likelihood cells against fixed-parameter flexsurv (spec 10.4)."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


def test_slice_b_payloads_load_and_acceptance_copy_is_identical(r_fixture):
    """Spec 10.4: committed, manifest-checked CSV and JSON describe the same rows."""
    fixture = r_fixture("slice_b")
    assert set(fixture["families"]) == {"weibull", "lognormal", "loglogistic"}
    tests = Path(__file__).parents[1]
    filename = "slice_b_likelihood.json"
    assert (tests / "fixtures/slice_b" / filename).read_bytes() == (
        tests / "spec/fixtures" / filename
    ).read_bytes()


@pytest.mark.parametrize("family", ("weibull", "lognormal", "loglogistic"))
@pytest.mark.parametrize("vector_index", (0, 1))
def test_mixed_censoring_and_truncation_rows_match_flexsurv(
    r_fixture, x64_enabled, family, vector_index
):
    """Spec 2/6.1/10.4/12.1: every R row agrees in eager, jit and row-vmap."""
    if not x64_enabled:
        pytest.skip("spec 10.4: Slice B R likelihood comparisons are float64 only")
    fixture = r_fixture("slice_b")
    rows = fixture["data"]
    kind = np.asarray(rows["kind"])
    data = survjax.SurvivalData(
        rows["time"],
        right_censored=kind == "right",
        left_censored=kind == "left",
        interval_censored=kind == "interval",
        time2=[0.0 if t is None else t for t in rows["time2"]],
        left_trunc=rows["left_trunc"],
        right_trunc=[np.inf if t is None else t for t in rows["right_trunc"]],
        dtype=jnp.float64,
    )
    vector = fixture["families"][family]["vectors"][vector_index]
    constructor = {
        "weibull": survjax.Weibull,
        "lognormal": survjax.LogNormal,
        "loglogistic": survjax.LogLogistic,
    }[family]
    dist = constructor(**vector["parameters"])
    assert data.issues.ok
    assert dist.parameter_issues().ok
    # Expected values are exclusively R fit$logliki, generated with fixedpars=TRUE.
    expected = np.asarray(vector["loglik"], dtype=np.float64)

    def evaluate(d, observations):
        return d.log_likelihood(observations)

    with jax.debug_nans(True):
        eager = evaluate(dist, data)
    for actual in (
        eager,
        jax.jit(evaluate)(dist, data),
        jax.vmap(evaluate, in_axes=(None, 0))(dist, data),
    ):
        assert actual.shape == (50,)
        assert actual.dtype == jnp.float64
        np.testing.assert_allclose(
            actual,
            expected,
            atol=ATOL[True],
            rtol=RTOL[True],
            err_msg=f"{family}: {vector['name']}",
        )
