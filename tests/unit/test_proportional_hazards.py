"""PH construction and analytical direct-method composition, spec 4.4."""

from dataclasses import FrozenInstanceError, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import (
    ATOL,
    DERIVATIVE_ATOL,
    DERIVATIVE_RTOL,
    LIKELIHOOD_GRAD_EPS,
    RTOL,
    TAIL_RTOL,
)
from jax.test_util import check_grads

from survjax import (
    Exponential,
    SurvivalData,
    SurvivalDistribution,
    param,
    register_family,
)
from survjax.distributions.wrappers import ProportionalHazards

METHODS = (
    "log_survival survival log_cdf cumulative_hazard log_hazard hazard log_prob".split()
)


def test_broadcast_pytree_indexing_and_dtypes(x64_enabled):
    """Spec 3.3/4.1/4.4: nested leaves broadcast without dtype normalization."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    baseline = Exponential(rate=jnp.array([[1], [2]], dtype=jnp.float32))
    dist = ProportionalHazards(
        baseline=baseline, log_hazard_ratio=jnp.zeros((2, 2), dtype=dtype)
    )
    assert dist.batch_shape == dist.baseline.batch_shape == (2, 2)
    assert dist.baseline.rate.dtype == jnp.float32
    assert dist.log_hazard_ratio.dtype == dist.reference_time().dtype == dtype
    assert baseline.batch_shape == (2, 1)
    leaves, structure = jax.tree.flatten(dist)
    assert len(leaves) == 2
    rebuilt = jax.tree.unflatten(structure, leaves)
    np.testing.assert_array_equal(rebuilt[1].baseline.rate, [2, 2])
    np.testing.assert_array_equal(
        jax.tree.map(lambda x: x[0], dist).baseline.rate, [1, 1]
    )
    with pytest.raises(FrozenInstanceError):
        dist.log_hazard_ratio = 0


def test_traced_diagnostics_include_predictor_and_prefixed_baseline():
    """Spec 4.7: real predictor and nested positive constraint keep row masks."""

    def issues(rate, eta):
        return ProportionalHazards(
            baseline=Exponential(rate=rate), log_hazard_ratio=eta
        ).parameter_issues()

    rate, eta = (
        jnp.array([1.0, -1.0, 1.0, 1.0]),
        jnp.array([0.0, 0.0, jnp.nan, jnp.inf]),
    )
    for result in (
        issues(rate, eta),
        jax.jit(issues)(rate, eta),
        jax.vmap(issues)(rate, eta),
    ):
        assert set(result.fields) == {"log_hazard_ratio", "baseline.rate"}
        np.testing.assert_array_equal(result.fields["baseline.rate"], [0, 1, 0, 0])
        np.testing.assert_array_equal(result.fields["log_hazard_ratio"], [0, 0, 1, 1])
        np.testing.assert_array_equal(result.invalid_entries, [0, 1, 1, 1])
    dist = ProportionalHazards(baseline=Exponential(rate=rate), log_hazard_ratio=eta)
    assert dist[1].parameter_issues().invalid_entries


def test_static_misuse_raises():
    """Spec 4.1–4.2 / D-028 decision: eta stays keyword-only."""
    with pytest.raises(TypeError):
        ProportionalHazards(Exponential(rate=1), 0)
    with pytest.raises(TypeError):
        ProportionalHazards(
            Exponential(rate=1), baseline=Exponential(rate=1), log_hazard_ratio=0
        )
    with pytest.raises(TypeError):
        ProportionalHazards(Exponential(rate=1))
    for baseline, eta in (
        (1, 0),
        (Exponential(rate=[1, 2]), [0, 0, 0]),
        (Exponential(rate=1), True),
    ):
        with pytest.raises(TypeError):
            ProportionalHazards(baseline=baseline, log_hazard_ratio=eta)


@pytest.mark.parametrize("name", METHODS)
def test_direct_and_grid_exponential_formulas(name, x64_enabled):
    """Spec 4.4/5: r=2, exp(eta)=[1/2,2] gives hazards [1,4]."""
    dist = ProportionalHazards(
        baseline=Exponential(rate=2), log_hazard_ratio=jnp.log(jnp.array([0.5, 2.0]))
    )
    time = jnp.array([0.0, 1.0, jnp.inf])
    hazard = np.array([1.0, 4.0])
    cumulative = np.array([[0.0, 0.0], [1.0, 4.0], [np.inf, np.inf]])
    expected = {
        "log_survival": -cumulative,
        "survival": np.exp(-cumulative),
        "log_cdf": np.array(
            [[-np.inf, -np.inf], np.log(-np.expm1(-hazard)), [0.0, 0.0]]
        ),
        "cumulative_hazard": cumulative,
        "log_hazard": np.broadcast_to(np.log(hazard), (3, 2)),
        "hazard": np.broadcast_to(hazard, (3, 2)),
        "log_prob": np.log(hazard) - cumulative,
    }[name]
    grid = getattr(dist, name + "_grid")
    for actual in (
        grid(time),
        jax.jit(grid)(time),
        jax.vmap(getattr(dist, name))(time),
    ):
        assert actual.shape == (3, 2)
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_likelihood_and_second_order_interior_derivatives(x64_enabled):
    """Spec 6–7/12.1: log L=[log(r)+eta-r*exp(eta), -2*r*exp(eta)]."""
    data = SurvivalData([1.0, 2.0], [1, 0])

    def objective(p):
        dist = ProportionalHazards(
            baseline=Exponential(rate=p[0]), log_hazard_ratio=p[1]
        )
        return dist.log_likelihood(data).mean()

    p = jnp.array([2.0, 0.0])
    dist = ProportionalHazards(baseline=Exponential(rate=p[0]), log_hazard_ratio=p[1])
    with jax.debug_nans(True):
        for actual in (
            dist.log_likelihood(data),
            jax.jit(dist.log_likelihood)(time=data.lower, event=data.exact),
        ):
            np.testing.assert_allclose(
                actual,
                [np.log(2) - 2, -4],
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
            )
        check_grads(
            objective,
            (p,),
            order=2,
            modes=("fwd", "rev"),
            eps=LIKELIHOOD_GRAD_EPS[x64_enabled],
            atol=DERIVATIVE_ATOL[x64_enabled],
            rtol=DERIVATIVE_RTOL[x64_enabled],
        )


def test_extreme_predictors_do_not_form_overflow_times_zero(x64_enabled):
    """Spec 7.2: exp(eta)*H0 is representable even if exp(eta) is not."""
    eta = 720.0 if x64_enabled else 90.0
    rate = 1e-300 if x64_enabled else 1e-30
    dist = ProportionalHazards(baseline=Exponential(rate=rate), log_hazard_ratio=eta)
    # Analytical log H = eta+log(rate), evaluated in host float64.
    expected = np.exp(eta + np.log(rate))
    np.testing.assert_allclose(
        jax.jit(dist.cumulative_hazard)(1.0),
        expected,
        atol=0,
        rtol=TAIL_RTOL[x64_enabled],
    )
    # Pin the overflow guard immediately above log(max), not only far beyond it.
    dtype = np.float64 if x64_enabled else np.float32
    for predictor in (eta, -eta, float(np.log(np.finfo(dtype).max)) + 0.25):

        def fn(e):
            return ProportionalHazards(
                baseline=Exponential(rate=1), log_hazard_ratio=e
            ).survival_grid(jnp.array([0.0, jnp.inf]))

        np.testing.assert_array_equal(jax.jit(fn)(predictor), [1, 0])
        np.testing.assert_array_equal(jax.jit(jax.jacrev(fn))(predictor), [0, 0])


def test_positional_baseline_matches_keyword_under_tracing():
    """D-028 maintainer decision: both baseline forms retain registration behavior."""

    def construct(rate, eta):
        baseline = Exponential(rate=rate)
        return (
            ProportionalHazards(baseline, log_hazard_ratio=eta),
            ProportionalHazards(baseline=baseline, log_hazard_ratio=eta),
        )

    for fn in (construct, jax.jit(construct), jax.vmap(construct)):
        positional, keyword = fn(jnp.array([1.0, 2.0]), jnp.array([0.0, 1.0]))
        for left, right in zip(
            jax.tree.leaves(positional), jax.tree.leaves(keyword), strict=True
        ):
            np.testing.assert_array_equal(left, right)


def test_lower_tail_recovers_value_and_first_derivatives(x64_enabled):
    """Spec 4.4/7.3: H=exp(eta)*r*t remains normal when r*t underflows."""
    rate, time, eta = (1e-200, 1e-200, 300.0) if x64_enabled else (1e-30, 1e-20, 40.0)
    expected = -np.exp(eta + np.log(rate) + np.log(time))
    gradient = [-np.exp(eta) * time, expected]

    def fn(r, e):
        return ProportionalHazards(
            Exponential(rate=r), log_hazard_ratio=e
        ).log_survival(time)

    for evaluate in (fn, jax.jit(fn)):
        np.testing.assert_allclose(
            evaluate(rate, eta), expected, atol=0, rtol=TAIL_RTOL[x64_enabled]
        )
    for derivative in (jax.jacfwd(fn, (0, 1)), jax.jacrev(fn, (0, 1))):
        np.testing.assert_allclose(
            jax.jit(derivative)(rate, eta),
            gradient,
            atol=0,
            rtol=TAIL_RTOL[x64_enabled],
        )


@register_family
@dataclass(frozen=True)
class CureBaseline(SurvivalDistribution):
    cure: jax.Array = param()

    def log_survival(self, t):
        return jnp.log(self.cure) * (-jnp.expm1(-jnp.asarray(t)))

    def log_survival_at_infinity(self):
        return jnp.log(self.cure)

    def reference_time(self):
        return jnp.ones_like(self.cure)


@pytest.mark.parametrize("cure", [0.3, 1.0])
def test_improper_infinity_value_and_derivatives(cure, x64_enabled):
    """Spec 4.4–4.5: S(inf)=p**exp(eta), including the zero-log-limit p=1."""
    eta = 0.5
    expected = cure ** np.exp(eta)
    gradient = [expected * np.exp(eta) / cure, expected * np.exp(eta) * np.log(cure)]

    def fn(p, e):
        return ProportionalHazards(CureBaseline(cure=p), log_hazard_ratio=e).survival(
            jnp.inf
        )

    np.testing.assert_allclose(
        jax.jit(fn)(cure, eta), expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    for derivative in (jax.jacfwd(fn, (0, 1)), jax.jacrev(fn, (0, 1))):
        np.testing.assert_allclose(
            jax.jit(derivative)(cure, eta),
            gradient,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )


def test_nan_predictor_propagates_to_infinity_hook_and_direct_values():
    """Spec 4.4–4.5: a NaN predictor cannot become the proper -inf hook default."""
    dist = ProportionalHazards(Exponential(rate=1), log_hazard_ratio=jnp.nan)
    assert jnp.isnan(jax.jit(lambda d: d.log_survival_at_infinity())(dist))
    for name in METHODS:
        assert jnp.isnan(jax.jit(lambda d, name=name: getattr(d, name)(jnp.inf))(dist))


def test_reference_retains_wider_baseline_dtype(x64_enabled):
    """Spec 3.3/4.2: the float64 baseline scale is not narrowed to float32."""
    if not x64_enabled:
        pytest.skip("mixed float64/float32 parameters require x64 mode")
    baseline = Exponential(rate=jnp.array(1e-60, dtype=jnp.float64))
    dist = ProportionalHazards(
        baseline, log_hazard_ratio=jnp.array(0.0, dtype=jnp.float32)
    )
    reference = jax.jit(lambda d: d.reference_time())(dist)
    assert reference.dtype == jnp.float64
    np.testing.assert_allclose(reference, 1e60, atol=0, rtol=RTOL[x64_enabled])


def test_zero_log_limit_avoids_overflowing_linear_factor():
    """Spec 4.4/7.2: with cure=1, S(inf)=1 is constant in every finite eta."""

    def fn(eta):
        dist = ProportionalHazards(CureBaseline(cure=1), log_hazard_ratio=eta)
        return dist.survival(jnp.inf)

    value, derivative = jax.jit(jax.value_and_grad(fn))(1000.0)
    assert value == 1 and derivative == 0
