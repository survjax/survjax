"""Two statistical checks: JAX_ENABLE_X64=1 uv run --group examples pytest -q this-file."""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import survjax

sys.path[:0] = [
    str(Path(__file__).resolve().parents[2]),
    str(Path(__file__).resolve().parents[2] / "tests"),
]
import impl_tolerances as tol  # noqa: E402

from examples.weibull_simulation import study  # noqa: E402


def test_exponential_fit_matches_event_count_over_exposure():
    """Use-case 01: exponential PH has an analytic two-arm MLE."""
    data = survjax.SurvivalData(
        [1.0, 2.0, 3.0, 2.0, 4.0, 6.0], event=[1, 0, 1, 1, 1, 0]
    )
    fit = jax.jit(
        lambda d: study.fit_replicate(
            d, jnp.repeat(jnp.arange(2), 3), model="exponential"
        )
    )(data)
    # Two events and exposure 6 or 12: rates 1/3 and 1/6, HR=1/2.
    assert fit.accepted
    np.testing.assert_allclose(
        np.exp(fit.theta), [1 / 3, 1 / 2], rtol=tol.RTOL[True], atol=tol.ATOL[True]
    )
    # Var(log HR) = 1/d0 + 1/d1 = 1 at the exponential MLE.
    np.testing.assert_allclose(
        fit.log_hr_se, 1.0, rtol=tol.RTOL[True], atol=tol.ATOL[True]
    )


def test_no_events_in_an_arm_is_not_reported_as_a_success():
    """Use-case 01: an unestimable treatment effect has no Wald interval."""
    data = survjax.SurvivalData([1.0, 2.0, 3.0, 4.0], event=[1, 1, 0, 0])
    fit = study.fit_replicate(data, jnp.array([0, 0, 1, 1]), model="weibull")
    assert not fit.accepted and not fit.estimable
    assert np.isnan(fit.hr_interval).all()
