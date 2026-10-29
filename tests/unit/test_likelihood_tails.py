"""Independent likelihood tail and overflow checks (spec 7/12.1–12.2)."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL, TAIL_ATOL, TAIL_RTOL

import survjax


def _contributions(family, parameters, time, event):
    if family is survjax.Exponential:
        return family(rate=parameters[0]).log_likelihood(time=time, event=event)
    return family(shape=parameters[0], scale=parameters[1]).log_likelihood(
        survjax.SurvivalData(time, event)
    )


def _mp_contribution(family, parameters, time, exact):
    # Spec 2/4.1: evaluate defining identities at the actual dtype-rounded inputs.
    # All inputs here are concrete; mpmath never receives a tracer.
    with mp.workdps(100):
        time = mp.mpf(float(time))
        parameters = [mp.mpf(float(p)) for p in parameters]
        if family is survjax.Exponential:
            (rate,) = parameters
            return float((mp.log(rate) if exact else 0) - rate * time)
        shape, scale = parameters
        log_ratio = mp.log(time / scale)
        log_density_factor = mp.log(shape) - mp.log(scale) + (shape - 1) * log_ratio
        return float((log_density_factor if exact else 0) - (time / scale) ** shape)


@pytest.mark.parametrize("exact", [True, False], ids=["exact", "right"])
@pytest.mark.parametrize(
    "rate,time",
    [(0.002, 400.0), (400.0, 0.003), (1e-10, 1e-10), (2.0, 500.0), (1e-30, 1.0)],
    ids=["small-rate", "large-rate", "tiny-hazard", "density-underflow", "tiny-rate"],
)
def test_exponential_tail_contribution_matches_mpmath(rate, time, exact, x64_enabled):
    """Spec 7.1/12.1: log(r)-r*t and -r*t retain relative tail precision."""
    parameters, time = jnp.asarray([rate]), jnp.asarray(time)
    expected = _mp_contribution(survjax.Exponential, parameters, time, exact)
    actual = _contributions(survjax.Exponential, parameters, time, exact)
    assert np.isfinite(actual)
    assert np.sign(actual) == np.sign(expected)
    np.testing.assert_allclose(
        actual, expected, atol=TAIL_ATOL[x64_enabled], rtol=TAIL_RTOL[x64_enabled]
    )


@pytest.mark.parametrize("exact", [True, False], ids=["exact", "right"])
@pytest.mark.parametrize(
    "shape,scale,time",
    [
        (0.4, 300.0, 2.0),
        (1.0, 7.0, 3.0),
        (2.5, 0.002, 0.004),
        (0.5, 2.0, 1e-30),
        (3.0, 1.0, 1e-10),
        (0.5, 1.0, 1e6),
        (2.0, 1.0, 100.0),
        (8.0, 1.0, 1e4),
    ],
    ids=[
        "large-scale",
        "unit-shape",
        "small-scale",
        "early-decreasing-hazard",
        "tiny-hazard",
        "late-decreasing-hazard",
        "density-underflow",
        "huge-hazard",
    ],
)
def test_weibull_tail_contribution_matches_mpmath(
    shape, scale, time, exact, x64_enabled
):
    """Spec 4.1/7.1/12.1: log h-H and -H stay finite while representable."""
    parameters, time = jnp.asarray([shape, scale]), jnp.asarray(time)
    expected = _mp_contribution(survjax.Weibull, parameters, time, exact)
    actual = _contributions(survjax.Weibull, parameters, time, exact)
    assert np.isfinite(actual)
    assert np.sign(actual) == np.sign(expected)
    np.testing.assert_allclose(
        actual, expected, atol=TAIL_ATOL[x64_enabled], rtol=TAIL_RTOL[x64_enabled]
    )


@pytest.mark.parametrize(
    "family,parameters",
    [(survjax.Exponential, [2.0]), (survjax.Weibull, [2.0, 1.0])],
    ids=["exponential-function-raw", "weibull-method-prepared"],
)
def test_mixed_tail_rows_match_mpmath_under_transformations(
    family, parameters, x64_enabled
):
    """Spec 5/7.2–7.3: jit and row-wise vmap preserve mixed tail contributions."""
    parameters = jnp.asarray(parameters)
    time = jnp.array([1e-15, 0.5, 100.0, 1e4, 1.0])
    event = jnp.array([False, True, True, False, True])
    expected = [
        _mp_contribution(family, parameters, t, bool(e))
        for t, e in zip(np.asarray(time), np.asarray(event), strict=True)
    ]

    def evaluate(t, e):
        return _contributions(family, parameters, t, e)

    for run in (evaluate, jax.jit(evaluate), jax.vmap(evaluate)):
        actual = run(time, event)
        assert actual.shape == time.shape
        assert np.isfinite(actual).all()
        np.testing.assert_allclose(
            actual, expected, atol=TAIL_ATOL[x64_enabled], rtol=TAIL_RTOL[x64_enabled]
        )


@pytest.mark.parametrize(
    "rate,time,exact",
    [(1e-10, 1e-10, False), (2.0, 500.0, True)],
    ids=["tiny-hazard", "density-underflow"],
)
def test_exponential_tail_rate_derivative_matches_algebra(
    rate, time, exact, x64_enabled
):
    """Spec 7.3/12.1: d/dr log L = event/r-t, including tiny derivatives."""
    parameters, time = jnp.asarray([rate]), jnp.asarray(time)
    with mp.workdps(100):
        rate, t = mp.mpf(float(parameters[0])), mp.mpf(float(time))
        expected = [float(int(exact) / rate - t)]

    def value(p):
        return _contributions(survjax.Exponential, p, time, exact)

    for differentiate in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            differentiate(value)(parameters),
            expected,
            atol=TAIL_ATOL[x64_enabled],
            rtol=TAIL_RTOL[x64_enabled],
        )


@pytest.mark.parametrize(
    "shape,scale,time,exact",
    [(3.0, 1.0, 1e-10, False), (2.0, 1.0, 100.0, True)],
    ids=["tiny-hazard", "density-underflow"],
)
def test_weibull_tail_parameter_derivatives_match_algebra(
    shape, scale, time, exact, x64_enabled
):
    """Spec 7.3/12.1: shape and scale derivatives retain nonzero tail terms."""
    parameters, time = jnp.asarray([shape, scale]), jnp.asarray(time)
    with mp.workdps(100):
        k, b = (mp.mpf(float(p)) for p in parameters)
        log_ratio = mp.log(mp.mpf(float(time)) / b)
        cumulative_hazard = mp.exp(k * log_ratio)
        # Differentiate event*(log k-log b+(k-1)*log(t/b)) - (t/b)**k.
        expected = [
            float(int(exact) / k + (int(exact) - cumulative_hazard) * log_ratio),
            float(k * (cumulative_hazard - int(exact)) / b),
        ]

    def value(p):
        return _contributions(survjax.Weibull, p, time, exact)

    for differentiate in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            differentiate(value)(parameters),
            expected,
            atol=TAIL_ATOL[x64_enabled],
            rtol=TAIL_RTOL[x64_enabled],
        )


@pytest.mark.parametrize("exact", [True, False], ids=["exact", "right"])
def test_overflowing_log_contribution_is_negative_infinity(exact, x64_enabled):
    """Spec 7.1 rev 34: H=1e500 gives -inf with no data/parameter issue."""
    parameters = jnp.array([50.0, 1.0])
    time, event = jnp.array([1.0, 1e10]), jnp.array([True, exact])
    data = survjax.SurvivalData(time, event)
    dist = survjax.Weibull(shape=parameters[0], scale=parameters[1])
    assert data.issues.ok
    assert dist.parameter_issues().ok
    # At t=b=1, log f=log(k)-1; at t=1e10, H=t**50=1e500.
    expected_interior = np.log(50.0) - 1

    def evaluate(t, e):
        return _contributions(survjax.Weibull, parameters, t, e)

    for run in (evaluate, jax.jit(evaluate), jax.vmap(evaluate)):
        actual = run(time, event)
        assert actual[1] == -jnp.inf
        np.testing.assert_allclose(
            actual[0], expected_interior, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_finite_neighbour_forward_derivative_survives_overflow(x64_enabled):
    """Spec 7.3 rev 34: a shared parameter's forward derivatives remain valid."""
    time, event = jnp.array([1.0, 1e10]), jnp.array([True, False])

    def values(p):
        return _contributions(survjax.Weibull, p, time, event)

    # At t=b=1: d/dk log f=1/k; d/db log f=k*(H-1)/b=0.
    actual = jax.jacfwd(values)(jnp.array([50.0, 1.0]))
    np.testing.assert_allclose(
        actual[0], [1 / 50, 0], atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


def test_finite_neighbour_reverse_derivative_survives_per_row_overflow(x64_enabled):
    """Spec 7.3 rev 34: per-row parameters keep the finite row's reverse derivative."""
    time, event = jnp.array([0.5, 1e10]), jnp.array([True, False])

    def first_row(p):
        return _contributions(survjax.Weibull, p, time, event)[0]

    # First row k=2,b=1,t=.5: H=.25, dk=.5+.75*log(.5), db=2*(.25-1).
    expected = [0.5 + 0.75 * np.log(0.5), -1.5]
    actual = jax.jacrev(first_row)(jnp.array([[2.0, 50.0], [1.0, 1.0]]))
    np.testing.assert_allclose(
        actual[:, 0], expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    # Spec 7.3: the infinite row's own derivatives are deliberately unasserted.
