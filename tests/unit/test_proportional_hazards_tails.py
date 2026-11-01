"""High-precision PH tails and total time-value policy (spec 4.4/4.5/7)."""

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL, TAIL_ATOL, TAIL_RTOL

from survjax import Exponential, Weibull
from survjax.distributions.wrappers import ProportionalHazards

METHODS = (
    "log_survival survival log_cdf cumulative_hazard log_hazard hazard log_prob".split()
)


def _distribution(p, family, eta=None):
    baseline = (
        Exponential(rate=p[0])
        if family == "exponential"
        else Weibull(shape=p[0], scale=p[1])
    )
    return ProportionalHazards(baseline, log_hazard_ratio=p[-1] if eta is None else eta)


def _inputs(family, x64):
    dtype = np.float64 if x64 else np.float32
    late_hazard = 600.0 if x64 else 60.0
    if family == "exponential":
        parameters = [
            [1e-200 if x64 else 1e-30, 500.0 if x64 else 40.0],
            [1e-300 if x64 else 1e-30, 720.0 if x64 else 90.0],
            [1e300 if x64 else 1e30, -720.0 if x64 else -100.0],
            [0.7, 0.3],
        ]
        times = [1e-200 if x64 else 1e-20, 1.0, 1.0, late_hazard / (0.7 * np.exp(0.3))]
    else:
        parameters = [
            [2.0, 1.0, 500.0 if x64 else 90.0],
            [2.0, 1.0, 720.0 if x64 else 90.0],
            [2.0, 1.0, -720.0 if x64 else -100.0],
            [1.4, 2.0, 0.3],
        ]
        times = [
            1e-200 if x64 else 1e-30,
            1e-150 if x64 else 1e-15,
            1e150 if x64 else 1e15,
            2 * (late_hazard / np.exp(0.3)) ** (1 / 1.4),
        ]
    # The oracle uses these actual dtype-rounded inputs, not decimal ideals.
    return np.asarray(parameters, dtype=dtype), np.asarray(times, dtype=dtype)


def _reference(p, time, family):
    with mp.workdps(100):
        q = [mp.mpf(float(value)) for value in p]
        t, eta = mp.mpf(float(time)), q[-1]
        if family == "exponential":
            log_hazard = mp.log(q[0]) + eta
            cumulative_hazard = mp.exp(log_hazard) * t
        else:
            k, b = q[:2]
            log_ratio = mp.log(t / b)
            log_hazard = mp.log(k / b) + (k - 1) * log_ratio + eta
            cumulative_hazard = mp.exp(eta + k * log_ratio)
        # Stable independent log F in BOTH tails, without subtracting from one.
        log_cdf = (
            mp.log(-mp.expm1(-cumulative_hazard))
            if cumulative_hazard < 1
            else mp.log1p(-mp.exp(-cumulative_hazard))
        )
        return dict(
            zip(
                METHODS,
                map(
                    float,
                    [
                        -cumulative_hazard,
                        mp.exp(-cumulative_hazard),
                        log_cdf,
                        cumulative_hazard,
                        log_hazard,
                        mp.exp(log_hazard),
                        log_hazard - cumulative_hazard,
                    ],
                ),
                strict=True,
            )
        )


@pytest.mark.parametrize("family", ["exponential", "weibull"])
@pytest.mark.parametrize("method", METHODS)
def test_tail_values_match_high_precision_ph_formulas(family, method, x64_enabled):
    """Spec 4.4/7.2: scale before exp, recover tiny H0, retain both log tails."""
    parameters, times = _inputs(family, x64_enabled)
    expected = np.array(
        [
            _reference(p, t, family)[method]
            for p, t in zip(parameters, times, strict=True)
        ]
    )

    def evaluate(p, time):
        return getattr(_distribution(p, family), method)(time)

    compiled = jax.jit(evaluate)
    for p, t, reference in zip(parameters, times, expected, strict=True):
        for value in (evaluate(p, t), compiled(p, t)):
            np.testing.assert_allclose(
                value,
                reference,
                atol=TAIL_ATOL[x64_enabled],
                rtol=TAIL_RTOL[x64_enabled],
                equal_nan=False,
            )
    for evaluate_many in (jax.vmap(evaluate), jax.jit(jax.vmap(evaluate))):
        np.testing.assert_allclose(
            evaluate_many(parameters, times),
            expected,
            atol=TAIL_ATOL[x64_enabled],
            rtol=TAIL_RTOL[x64_enabled],
            equal_nan=False,
        )


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_tail_eta_first_and_second_derivatives_equal_log_survival(family, x64_enabled):
    """Spec 7.3: d_eta log S=d_eta^2 log S=-H for finite smooth inputs."""
    parameters, times = _inputs(family, x64_enabled)
    expected = np.array(
        [
            _reference(p, t, family)["log_survival"]
            for p, t in zip(parameters, times, strict=True)
        ]
    )

    def evaluate(eta, p, time):
        # Differentiate eta directly; do not send zero tangents through tiny r.
        return _distribution(p, family, eta).log_survival(time)

    parameters, times = jnp.asarray(parameters), jnp.asarray(times)
    # log H spans approximately [-421, 30]; log reconstruction/AD rounding is
    # within the existing tail relative bounds. Zero atol cannot accept zero
    # for a representable tiny derivative. No natural-parameter tail Hessian.
    for orders in ("f", "r", "ff", "fr", "rf", "rr"):
        derivative = evaluate
        # "fr" = jacfwd(jacrev(f)): the outermost transform is named first.
        for order in reversed(orders):
            derivative = (jax.jacfwd if order == "f" else jax.jacrev)(derivative)
        for p, t, reference in zip(parameters, times, expected, strict=True):
            np.testing.assert_allclose(
                derivative(p[-1], p, t),
                reference,
                atol=TAIL_ATOL[x64_enabled],
                rtol=TAIL_RTOL[x64_enabled],
                equal_nan=False,
            )
        for evaluate_many in (jax.vmap(derivative), jax.jit(jax.vmap(derivative))):
            np.testing.assert_allclose(
                evaluate_many(parameters[:, -1], parameters, times),
                expected,
                atol=TAIL_ATOL[x64_enabled],
                rtol=TAIL_RTOL[x64_enabled],
                equal_nan=False,
            )


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_lower_tail_natural_parameter_gradients_match_analytic_chain_rule(
    family, x64_enabled
):
    """Spec 4.4/7.3: recovering a rounded-zero H0 retains baseline tangents."""
    parameters, times = _inputs(family, x64_enabled)
    p, time = parameters[0], times[0]
    with mp.workdps(100):
        q = [mp.mpf(float(value)) for value in p]
        t = mp.mpf(float(time))
        if family == "exponential":
            h = mp.exp(q[-1]) * q[0] * t
            expected = [-h / q[0], -h]
        else:
            k, b = q[:2]
            h = mp.exp(q[-1]) * (t / b) ** k
            expected = [-h * mp.log(t / b), h * k / b, -h]
        expected = np.array([float(value) for value in expected])

    def evaluate(p):
        return _distribution(p, family).log_survival(time)

    p = jnp.asarray(p)
    assert _distribution(p, family).baseline.log_survival(time) == 0
    for differentiate in (jax.jacfwd, jax.jacrev):
        for derivative in (differentiate(evaluate), jax.jit(differentiate(evaluate))):
            np.testing.assert_allclose(
                derivative(p),
                expected,
                atol=TAIL_ATOL[x64_enabled],
                rtol=TAIL_RTOL[x64_enabled],
                equal_nan=False,
            )


@pytest.mark.parametrize("family", ["exponential", "weibull"])
@pytest.mark.parametrize("method", METHODS)
def test_direct_and_grid_methods_follow_complete_time_value_policy(
    family, method, x64_enabled
):
    """Spec 4.5/5: negative and NaN times follow policy; overrides supply limits."""
    p = jnp.array([0.7, 0.3] if family == "exponential" else [1.4, 2.0, 0.3])
    times = jnp.array([-jnp.inf, -2.0, -0.0, 0.0, 2.0, jnp.inf, jnp.nan])
    # Hand boundary limits: Weibull k>1 has h(0)=0 and h(inf)=inf.
    h0 = float(np.exp(np.log(0.7) + float(p[-1]))) if family == "exponential" else 0.0
    hinf = h0 if family == "exponential" else np.inf
    negative = {
        "log_survival": 0,
        "survival": 1,
        "log_cdf": -np.inf,
        "cumulative_hazard": 0,
        "log_hazard": -np.inf,
        "hazard": 0,
        "log_prob": -np.inf,
    }
    zero = dict(
        negative,
        log_hazard=np.log(h0) if h0 else -np.inf,
        hazard=h0,
        log_prob=np.log(h0) if h0 else -np.inf,
    )
    infinity = {
        "log_survival": -np.inf,
        "survival": 0,
        "log_cdf": 0,
        "cumulative_hazard": np.inf,
        "log_hazard": np.log(hinf),
        "hazard": hinf,
        "log_prob": -np.inf,
    }
    interior = _reference(np.asarray(p), 2.0, family)[method]
    expected = [
        negative[method],
        negative[method],
        zero[method],
        zero[method],
        interior,
        infinity[method],
    ]
    dist = _distribution(p, family)
    direct, grid = getattr(dist, method), getattr(dist, method + "_grid")
    for value in (
        direct(times),
        jax.jit(direct)(times),
        grid(times),
        jax.jit(grid)(times),
        jax.vmap(direct)(times),
    ):
        np.testing.assert_array_equal(np.isnan(value), [False] * 6 + [True])
        np.testing.assert_allclose(
            value[:6],
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
        )


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_finite_policy_values_have_zero_parameter_gradients(family):
    """Spec 4.5/7.2: safe inactive inputs preserve constant policy tangents."""
    p = jnp.array([0.7, 0.3] if family == "exponential" else [1.4, 2.0, 0.3])

    def evaluate(p):
        dist = _distribution(p, family)
        before = jnp.array([-jnp.inf, -2.0, 0.0])
        return jnp.concatenate(
            [
                dist.log_survival(before),
                dist.survival(before),
                dist.cumulative_hazard(before),
                dist.survival(jnp.inf)[None],
                dist.log_cdf(jnp.inf)[None],
            ]
        )

    for differentiate in (jax.jacfwd, jax.jacrev):
        for derivative in (differentiate(evaluate), jax.jit(differentiate(evaluate))):
            np.testing.assert_array_equal(derivative(p), np.zeros((11, len(p))))
