"""Independent analytical checks of the runnable example (spec 7.3/12.2)."""

import runpy
from pathlib import Path

import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL


@pytest.fixture(scope="module")
def result():
    example = Path(__file__).resolve().parents[2] / "examples" / "slice_a.py"
    namespace = runpy.run_path(str(example))
    return namespace["run_example"]()


def test_example_contributions_match_exact_and_right_formulas(result, x64_enabled):
    """Spec 6.1/7.1/12.2: the example uses the supported likelihood subset."""
    np.testing.assert_array_equal(result["time"], [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(result["event"], [True, False, True])
    assert result["exponential_rate"] == 0.5
    np.testing.assert_array_equal(result["weibull_parameters"], [2.0, 2.0])
    # Exponential: log f=log(r)-r*t for events, log S=-r*t for censoring.
    np.testing.assert_allclose(
        result["exponential_log_likelihood"],
        [np.log(0.5) - 0.5, -1.0, np.log(0.5) - 1.5],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    # Weibull k=b=2: H=(t/2)^2, log h=log(t/2); log f=log h-H.
    np.testing.assert_allclose(
        result["weibull_log_likelihood"],
        [np.log(0.5) - 0.25, -1.0, np.log(1.5) - 2.25],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    dtype = np.dtype("float64" if x64_enabled else "float32")
    assert all(value.dtype == dtype for key, value in result.items() if key != "event")


def test_example_compiled_objective_has_analytical_derivatives(result, x64_enabled):
    """Spec 7.3/12.2: joint shape/scale derivatives include the mixed Hessian."""
    # L=sum e*(log(k)-log(b)+(k-1)*u)-H, u=log(t/b), H=(t/b)^k.
    # At k=b=2, H=[1/4,1,9/4], e=[1,0,1], sum H=7/2.
    u = np.log([0.5, 1.0, 1.5])
    h = np.array([0.25, 1.0, 2.25])
    e = np.array([1.0, 0.0, 1.0])
    gradient = [1.0 + np.sum((e - h) * u), 1.5]
    # L_kk=sum -e/k^2-H*u^2, L_kb=sum (H-e+k*H*u)/b,
    # L_bb=sum k*(e-(k+1)*H)/b^2.
    hkk = -0.5 - np.sum(h * u**2)
    hkb = 0.75 + np.sum(h * u)
    hessian = [[hkk, hkb], [hkb, -4.25]]
    assert result["weibull_total"].shape == ()
    assert result["weibull_gradient"].shape == (2,)
    assert result["weibull_hessian"].shape == (2, 2)
    for key, expected in (
        ("weibull_total", np.log(0.75) - 3.5),
        ("weibull_gradient", gradient),
        ("weibull_hessian", hessian),
    ):
        np.testing.assert_allclose(
            result[key], expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_example_grid_is_time_first_and_starts_at_zero(result, x64_enabled):
    """Spec 5/12.2: every grid row evaluates both scale parameters."""
    time = np.array([0.0, 1.0, 2.0, 3.0])
    scale = np.array([1.0, 2.0])
    np.testing.assert_array_equal(result["grid_time"], time)
    np.testing.assert_array_equal(result["grid_scale"], scale)
    curves = result["weibull_survival_grid"]
    assert curves.shape == (4, 2)
    np.testing.assert_array_equal(curves[0], [1.0, 1.0])
    # Closed-form Weibull survival at shape=2, separately for both scales.
    np.testing.assert_allclose(
        curves,
        np.exp(-((time[:, None] / scale) ** 2)),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
