"""PH summary closure, independent integrals and safe composition (spec 4.4/8.7)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import mpmath as mp
import numpy as np
import pytest
from impl_tolerances import ATOL, RMST_CUTOFF_ATOL, RMST_PARAMETER_ATOL, RTOL

from survjax import Exponential, SurvivalDistribution, Weibull, param, register_family
from survjax.distributions.wrappers import ProportionalHazards

CASES = [("exponential", [1.3, 0.2, 0.75]), ("weibull", [1.4, 2.0, 0.2, 0.75])]


def _law(p, family):
    base = (
        Exponential(rate=p[0])
        if family == "exponential"
        else Weibull(shape=p[0], scale=p[1])
    )
    return ProportionalHazards(baseline=base, log_hazard_ratio=p[-2])


def _integral(p, family):
    # Independent PH survival, integrated directly rather than through gamma.
    eta, tau = p[-2:]
    if family == "exponential":
        return mp.quad(lambda t: mp.exp(-mp.exp(eta) * p[0] * t), [0, tau])
    return mp.quad(lambda t: mp.exp(-mp.exp(eta) * (t / p[1]) ** p[0]), [0, tau])


@pytest.mark.parametrize("family,point", CASES)
def test_rmst_values_and_first_derivatives_match_independent_integrals(
    family, point, x64_enabled
):
    """Spec 4.4/8.7/12.1: closure chain rule in eta, every baseline field and tau."""
    p = jnp.array(point)
    points = jnp.stack([p, p.at[-2].add(0.1).at[-1].multiply(1.2)])
    values, gradients = [], []
    with mp.workdps(60):
        for row in np.asarray(points):
            q = tuple(mp.mpf(float(x)) for x in row)
            values.append(float(_integral(q, family)))
            gradients.append(
                [
                    float(
                        mp.diff(
                            lambda *x: _integral(x, family),
                            q,
                            tuple(int(j == i) for j in range(len(q))),
                        )
                    )
                    for i in range(len(q))
                ]
            )
    values, gradients = np.array(values), np.array(gradients)

    def fn(q):
        return _law(q, family).rmst(q[-1])

    with jax.debug_nans(True):
        for evaluate in (jax.vmap(fn), jax.jit(jax.vmap(fn))):
            np.testing.assert_allclose(
                evaluate(points),
                values,
                atol=0,
                rtol=RTOL[x64_enabled],
                equal_nan=False,
            )
    # Existing spec 12.1 log-parameter error units; eta already is a log ratio.
    # Value-scaled errors avoid asserting relative accuracy for tiny gradients.
    units = np.asarray(points[:, :-1]).copy()
    units[:, -1] = 1
    for differentiate in (jax.jacfwd, jax.jacrev):
        for evaluate in (
            jax.vmap(differentiate(fn)),
            jax.jit(jax.vmap(differentiate(fn))),
        ):
            error = np.asarray(evaluate(points)) - gradients
            np.testing.assert_allclose(
                error[:, :-1] * units / values[:, None],
                0,
                atol=RMST_PARAMETER_ATOL[x64_enabled],
                rtol=0,
                equal_nan=False,
            )
            np.testing.assert_allclose(
                error[:, -1],
                0,
                atol=RMST_CUTOFF_ATOL[x64_enabled],
                rtol=0,
                equal_nan=False,
            )
        np.testing.assert_allclose(
            differentiate(fn)(p),
            gradients[0],
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
        )


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_rmst_broadcast_closure_identity_and_explicit_cutoff_grid(family, x64_enabled):
    """Spec 4.4/5/8.7: baseline (S,1), eta (S,2), cutoffs (m,1,1)."""
    baseline = jnp.array([[0.7], [1.3], [2.0]], dtype=jnp.float32)
    eta = jnp.array([[0.2, -0.3], [0.4, 0.1], [-0.2, 0.3]])
    if family == "exponential":
        base = Exponential(rate=baseline)
        equivalent = Exponential(rate=baseline * jnp.exp(eta))
    else:
        base = Weibull(shape=1.4, scale=baseline)
        equivalent = Weibull(shape=1.4, scale=baseline * jnp.exp(-eta / 1.4))
    dist = ProportionalHazards(baseline=base, log_hazard_ratio=eta)
    for cutoff in (jnp.array(0.75), jnp.array([0.01, 1.0, 100.0])[:, None, None]):
        expected = equivalent.rmst(cutoff)
        for fn in (dist.rmst, jax.jit(dist.rmst)):
            np.testing.assert_allclose(
                fn(cutoff), expected, atol=0, rtol=RTOL[x64_enabled], equal_nan=False
            )
    assert dist.rmst(cutoff).shape == (3, 3, 2)
    np.testing.assert_allclose(
        jax.vmap(dist.rmst)(cutoff),
        expected,
        atol=0,
        rtol=RTOL[x64_enabled],
        equal_nan=False,
    )


@pytest.mark.parametrize("family,point", CASES)
@pytest.mark.parametrize("eta", [0.2, -2000.0, 2000.0])
def test_rmst_zero_value_and_first_derivatives(family, point, eta):
    """Spec 8.7: M(0)=0, baseline/eta tangents 0, cutoff tangent 1."""
    p = jnp.array(point).at[-1].set(0).at[-2].set(eta)

    def fn(q):
        return _law(q, family).rmst(q[-1])

    with jax.debug_nans(True):
        assert fn(p) == jax.jit(fn)(p) == 0
        for transform in (jax.jacfwd, jax.jacrev):
            np.testing.assert_array_equal(
                jax.jit(transform(fn))(p), [0] * (len(p) - 1) + [1]
            )


def test_weibull_shape_hessian_remains_explicitly_unsupported():
    """Spec 8.7: PH delegation preserves the native shape-Hessian restriction."""

    def fn(k):
        return ProportionalHazards(
            baseline=Weibull(shape=k, scale=1.25), log_hazard_ratio=0.2
        ).rmst(0.75)

    for outer in (jax.jacfwd, jax.jacrev):
        for inner in (jax.jacfwd, jax.jacrev):
            with pytest.raises(NotImplementedError):
                jax.jit(outer(inner(fn)))(2.0)


@pytest.mark.parametrize(
    "family,field",
    [(f, v) for f in ("exponential", "weibull") for v in ("baseline", "eta", "cutoff")]
    + [("weibull", "shape")],
)
def test_invalid_rows_preserve_shared_parameter_derivatives(family, field, x64_enabled):
    """Spec 7.2/8.7: mask original operands before constructing any closure."""
    constants = jnp.array([1.3, 0, -1, jnp.nan, jnp.inf, -jnp.inf])
    sign = jnp.array([1.0, -1.0, 0.0, 0.0, 0.0, 0.0])

    def fn(q, mixed=True):
        # No upstream NaN arithmetic: only finite signed entries depend on q.
        bad = jnp.where(jnp.arange(6) < 3, q * sign, constants)
        baseline = (
            bad if mixed and field == "baseline" else (1.3 if field == "shape" else q)
        )
        shape = bad if mixed and field == "shape" else (q if field == "shape" else 1.3)
        eta = (
            jnp.array([0.2, jnp.nan, jnp.inf, -jnp.inf, jnp.nan, jnp.inf])
            if mixed and field == "eta"
            else 0.2
        )
        tau = (
            jnp.array([0.75, -1, jnp.nan, jnp.inf, -jnp.inf, -2])
            if mixed and field == "cutoff"
            else 0.75
        )
        base = (
            Exponential(rate=baseline)
            if family == "exponential"
            else Weibull(shape=shape, scale=baseline)
        )
        return ProportionalHazards(baseline=base, log_hazard_ratio=eta).rmst(tau)

    q = 1.3
    for value in (fn(q), jax.jit(fn)(q)):
        assert np.isnan(value[1:]).all()
        np.testing.assert_allclose(
            value[0], fn(q, False), rtol=RTOL[x64_enabled], equal_nan=False
        )
    for transform in (jax.jacfwd, jax.jacrev):
        actual = transform(lambda x: fn(x)[0])
        expected = transform(lambda x: fn(x, False))(q)
        np.testing.assert_allclose(
            jax.jit(actual)(q),
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
        )


@register_family
@dataclass(frozen=True)
class CustomBaseline(SurvivalDistribution):
    weights: jax.Array = param(component_ndim=1)

    def log_survival(self, t):
        return -jnp.sum(self.weights, axis=-1) * t

    def reference_time(self):
        return jnp.ones(self.batch_shape)

    def parameter_conditions(self):
        raise AssertionError("unsupported family diagnostics must not run")


def test_custom_baseline_rmst_has_no_generic_fallback():
    """Spec 8.7: custom families remain unsupported, including component fields."""
    dist = ProportionalHazards(
        baseline=CustomBaseline(weights=[1.0, 2.0]), log_hazard_ratio=0.2
    )
    for cutoff in (0.0, 1.0, jnp.inf, jnp.nan):
        for fn in (dist.rmst, jax.jit(dist.rmst)):
            with pytest.raises(NotImplementedError):
                fn(cutoff)
