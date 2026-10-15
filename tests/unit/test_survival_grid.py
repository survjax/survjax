"""Time-first grid algebra and transformations (spec 4.5/5/7.3)."""

import warnings
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, DERIVATIVE_ATOL, DERIVATIVE_RTOL, RTOL
from jax.test_util import check_grads

import survjax

NAMES = (
    "survival",
    "log_survival",
    "log_prob",
    "log_cdf",
    "cumulative_hazard",
    "log_hazard",
    "hazard",
)


@pytest.mark.parametrize("name", NAMES)
def test_shared_values_order_transformations_and_trace_reuse(name, x64_enabled):
    """Spec 5/7.2/12.1: K=N is an outer grid; values reuse one compiled trace."""
    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    traces = []

    def evaluate(dist, time):
        traces.append(None)
        return getattr(dist, name + "_grid")(time)

    compiled = jax.jit(evaluate)
    for shift in (0.0, 0.25):
        rates = jnp.asarray([0.5, 1.0, 2.0]) + shift
        times = jnp.asarray([1.0, 0.5, 1.0]) + shift
        dist = survjax.Exponential(rate=rates)
        r, t = np.asarray(rates), np.asarray(times)[:, None]
        # Exponential: H=r*t, h=r, log f=log(r)-H, F=1-exp(-H).
        expected = {
            "survival": np.exp(-t * r),
            "log_survival": -t * r,
            "log_prob": np.log(r) - t * r,
            "log_cdf": np.log(-np.expm1(-t * r)),
            "cumulative_hazard": t * r,
            "log_hazard": np.broadcast_to(np.log(r), (3, 3)),
            "hazard": np.broadcast_to(r, (3, 3)),
        }[name]
        for out in (
            getattr(dist, name + "_grid")(t=times),
            compiled(dist, times),
            jax.vmap(lambda d, time=times: getattr(d, name + "_grid")(time))(dist).T,
        ):
            assert out.shape == (3, 3)
            np.testing.assert_allclose(out, expected, atol=atol, rtol=rtol)
    assert len(traces) == 1


@pytest.mark.parametrize(
    "batch,length", [((), 0), ((), 2), ((2, 1, 3), 0), ((2, 1, 3), 4), ((0,), 3)]
)
def test_empty_and_multiple_batch_axes(batch, length):
    """Spec 5: empty grids/batches and arbitrary batch rank retain time first."""
    dist = survjax.Exponential(rate=jnp.ones(batch))
    for name in NAMES:
        out = jax.jit(getattr(dist, name + "_grid"))(jnp.ones(length))
        assert out.shape == (length,) + batch


def test_components_and_density_override_dispatch(x64_enabled):
    """Spec 4.2/5: components are family-owned; grid dispatch honors overrides."""
    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    calls = []

    @survjax.register_family
    @dataclass(frozen=True)
    class Component(survjax.SurvivalDistribution):
        rates: jax.Array = survjax.param(component_ndim=1)

        def reference_time(self):
            return 1 / self.rates.sum(-1)

        def log_survival(self, t):
            return -self.rates.sum(-1) * t

        def log_prob(self, t):
            calls.append("density override")
            return jnp.log(self.rates.sum(-1)) - self.rates.sum(-1) * t

    dist = Component(rates=[[1.0, 2.0], [3.0, 4.0]])
    t, r = np.array([0.5, 1.0, 2.0])[:, None], np.array([3.0, 7.0])
    # Sum component rates: H_j(t)=r_j*t, so log f_j=log(r_j)-r_j*t.
    out = dist.log_prob_grid(t[:, 0])
    assert out.shape == (3, 2) and calls == ["density override"]
    np.testing.assert_allclose(out, np.log(r) - t * r, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        dist.survival_grid(t[:, 0]), np.exp(-t * r), atol=atol, rtol=rtol
    )


@pytest.mark.parametrize(
    "bad",
    [1.0, [[1.0, 2.0]], [True, False], [1j], None],
)
def test_grid_misuse_raises_eagerly_and_during_tracing(bad):
    """Spec 4.5/5: grids reject nonreal inputs and non-vector ranks statically."""
    for name in NAMES:
        method = getattr(survjax.Exponential(rate=1.0), name + "_grid")
        for evaluate in (method, jax.jit(method)):
            with pytest.raises(TypeError):
                evaluate(bad)


@pytest.mark.parametrize("bad", [["1"], np.array([1], dtype=object)])
def test_nonjax_grid_constants_reach_conversion_during_tracing(bad):
    """Spec 4.5: capture non-JAX inputs so the grid conversion itself is traced."""
    for name in NAMES:
        method = getattr(survjax.Exponential(rate=1.0), name + "_grid")
        with pytest.raises(TypeError, match="grid time must contain real numeric"):
            method(bad)
        with pytest.raises(TypeError, match="grid time must contain real numeric"):
            jax.jit(partial(method, bad))()


@pytest.mark.parametrize("name", NAMES)
def test_grid_dtype_conversion_and_integer_inputs(name, x64_enabled):
    """Spec 4.5: transient NumPy float64 grids convert silently in x32."""
    method = getattr(survjax.Exponential(rate=1.0), name + "_grid")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        out = method(np.array([0.5, 1.0], dtype=np.float64))
    assert not seen
    assert out.dtype == (jnp.float64 if x64_enabled else jnp.float32)
    method32 = getattr(survjax.Exponential(rate=jnp.float32(1)), name + "_grid")
    assert method32(jnp.ones(2, dtype=jnp.float32)).dtype == jnp.float32
    np.testing.assert_array_equal(
        method32(np.array([1, 2], np.int32)),
        method32(jnp.array([1.0, 2.0], dtype=jnp.float32)),
    )


@pytest.mark.parametrize("name", NAMES)
def test_grid_analytic_rate_gradient_hessian_and_order_two(name, x64_enabled):
    """Spec 7.3: differentiate shared grids in forward/reverse modes to order two."""
    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    t = np.array([0.5, 1.0])
    s, q = np.exp(-t), np.expm1(t)
    # At r=1: dS/dr=-t*S, dlogF/dr=t/(exp(t)-1); differentiate once more.
    first, second = {
        "survival": (-t * s, t * t * s),
        "log_survival": (-t, np.zeros(2)),
        "log_prob": (1 - t, -np.ones(2)),
        "log_cdf": (t / q, -t * t * np.exp(t) / q**2),
        "cumulative_hazard": (t, np.zeros(2)),
        "log_hazard": (np.ones(2), -np.ones(2)),
        "hazard": (np.ones(2), np.zeros(2)),
    }[name]

    def value(rate):
        return getattr(survjax.Exponential(rate=rate), name + "_grid")(
            jnp.asarray(t)
        ).sum()

    for derivative, expected in (
        (jax.jacfwd(value), first.sum()),
        (jax.jacrev(value), first.sum()),
        (jax.hessian(value), second.sum()),
    ):
        np.testing.assert_allclose(
            jax.jit(jax.vmap(derivative))(jnp.ones(2)), expected, atol=atol, rtol=rtol
        )
    check_grads(
        value,
        (jnp.asarray(1.0),),
        order=2,
        modes=("fwd", "rev"),
        atol=DERIVATIVE_ATOL[x64_enabled],
        rtol=DERIVATIVE_RTOL[x64_enabled],
    )


@pytest.mark.parametrize(
    "name,expected",
    [
        ("survival", [0, -np.exp(-1), 0]),
        ("log_survival", [0, -1, None]),
        ("log_prob", [1, 0, None]),
        ("log_cdf", [None, 1 / np.expm1(1), 0]),
        ("cumulative_hazard", [0, 1, None]),
        ("log_hazard", [1, 1, 1]),
        ("hazard", [1, 1, 1]),
    ],
)
def test_mixed_boundary_grid_jacobian(name, expected, x64_enabled):
    """Spec 4.5/7.2: full Jacobians retain finite-boundary and interior derivatives."""
    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    # At r=1: S'=-t*S, log f'=1-t, log F'=t/expm1(t), H'=t, log h'=h'=1.
    # None marks nonfinite outputs, whose boundary derivatives are not guaranteed.
    expected = np.asarray(expected, dtype=float)
    required = np.isfinite(expected)

    def value(rate):
        return getattr(survjax.Exponential(rate=rate), name + "_grid")(
            jnp.array([0.0, 1.0, jnp.inf])
        )

    for derivative in (jax.jacfwd(value), jax.jacrev(value)):
        jacobian = np.asarray(jax.jit(derivative)(1.0))
        assert jacobian.shape == (3,)
        assert np.isfinite(jacobian[required]).all()
        np.testing.assert_allclose(
            jacobian[required], expected[required], atol=atol, rtol=rtol
        )


def test_weibull_shape_one_zero_grid_and_nan_isolation(x64_enabled):
    """Spec 4.5/11.1: S=exp(-t/b), log f=-log(b)-t/b, h=1/b at k=1."""
    atol, rtol = ATOL[x64_enabled], RTOL[x64_enabled]
    scales = 10 * np.exp(np.array([0.0, -0.5, -1.0]))
    b = np.append(scales, np.nan)
    t = np.array([0.0, 1.0, np.inf, np.nan])[:, None]
    dist = survjax.Weibull(shape=1.0, scale=b.tolist())
    for name, expected in (
        ("survival", np.exp(-t / b)),
        ("log_prob", -np.log(b) - t / b),
        ("hazard", np.where(np.isnan(t), np.nan, np.broadcast_to(1 / b, (4, 4)))),
    ):
        out = jax.jit(getattr(dist, name + "_grid"))(jnp.asarray(t[:, 0]))
        np.testing.assert_allclose(out, expected, atol=atol, rtol=rtol)

    def zero_survival(k):
        return (
            survjax.Weibull(shape=k, scale=scales.tolist()).survival_grid([0.0]).sum()
        )

    np.testing.assert_array_equal(jax.jit(jax.grad(zero_survival))(1.0), 0.0)
