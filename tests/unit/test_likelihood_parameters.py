"""Likelihood parameter preparation, independent of the evaluator (spec 7.1)."""

from dataclasses import dataclass
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL
from jax.flatten_util import ravel_pytree

import survjax
from survjax._likelihood_parameters import stop_invalid_gradients
from survjax.likelihood import _log_likelihood


@survjax.register_family
@dataclass(frozen=True)
class ComponentRate(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)
    weights: jax.Array = survjax.param(2, survjax.constraints.positive)
    free: jax.Array = survjax.param()
    label: str = survjax.static(default="component")

    def intensity(self):
        return self.rate * jnp.sum(self.weights, axis=(-2, -1)) * jnp.exp(self.free)

    def log_survival(self, t):
        return -self.intensity() * t

    def log_prob(self, t):
        return jnp.log(self.intensity()) - self.intensity() * t

    def reference_time(self):
        return jnp.ones(self.batch_shape, dtype=self.rate.dtype)

    def parameter_conditions(self):
        return {"cap": jnp.broadcast_to(self.rate > 3, self.batch_shape)}


@survjax.register_family
@dataclass(frozen=True)
class NestedRate(survjax.SurvivalDistribution):
    base: survjax.SurvivalDistribution = survjax.distribution()
    factor: jax.Array = survjax.param(constraint=survjax.constraints.real)
    label: str = survjax.static(default="nested")

    def log_survival(self, t):
        return jnp.exp(self.factor) * self.base.log_survival(t)

    def log_prob(self, t):
        rate = jnp.exp(self.factor) * self.base.intensity()
        return jnp.log(rate) - rate * t

    def reference_time(self):
        return self.base.reference_time()

    def parameter_conditions(self):
        return {"cap": self.factor > 2}


def case(kind, bad):
    if kind == "rate":
        return {"rate": [0.5, bad, 2.0]}, ("rate",)
    if kind == "shape":
        return {"shape": [2.0, bad, 0.5], "scale": 1.5}, ("shape",)
    if kind == "scale":
        return {"shape": 1.5, "scale": [1.0, bad, 2.0]}, ("scale",)
    params = {"rate": [0.5, bad, 2.0], "weights": [[0.25, 0.75]], "free": 0.1}
    per_row = ("rate",)
    if kind == "component":
        params.update(rate=1.0, weights=[[[0.25, 0.75]], [[bad, 0.75]], [[0.5, 0.5]]])
        per_row = ("weights",)
    if kind in ("nested", "outer", "outer-condition"):
        params["factor"] = 0.2
    if kind in ("outer", "outer-condition"):
        params.update(rate=1.0, factor=[0.1, bad, 0.2])
        per_row = ("factor",)
    if kind == "outer-condition":
        # Only the outer condition declares this row invalid; free has no domain.
        params["free"] = [0.1, np.nan, 0.1]
        per_row += ("free",)
    return params, per_row


def build(kind, params):
    if kind == "rate":
        return survjax.Exponential(**params)
    if kind in ("shape", "scale"):
        return survjax.Weibull(**params)
    if kind in ("nested", "outer", "outer-condition"):
        own = {k: v for k, v in params.items() if k != "factor"}
        return NestedRate(base=ComponentRate(**own), factor=params["factor"])
    return ComponentRate(**params)


CASES = (
    [
        (kind, bad)
        for kind in ("rate", "shape", "scale", "custom", "nested", "component")
        for bad in (0.0, -1.0, np.nan, np.inf, -np.inf)
    ]
    + [("outer", bad) for bad in (np.nan, np.inf, -np.inf)]
    + [("condition", 4.0), ("outer-condition", 3.0)]
)


@pytest.mark.parametrize("kind,bad", CASES)
@pytest.mark.parametrize("evaluation", ["helper", "core"])
def test_declared_violations_isolate_valid_values_and_derivatives(
    kind, bad, evaluation, x64_enabled
):
    """Spec 7.1/7.3: unguarded families match valid rows alone, including shared leaves."""
    params, per_row = case(kind, bad)
    flat, unravel = ravel_pytree({k: jnp.asarray(v) for k, v in params.items()})
    keep = jnp.array([0, 2])
    time = jnp.array([1.0, 2.0, 3.0])
    data = survjax.SurvivalData(time, time == 1)

    def contributions(dist, times):
        return jnp.where(times == 1, dist.log_prob(times), dist.log_survival(times))

    def together(values):
        dist = build(kind, unravel(values))
        if evaluation == "core":
            return _log_likelihood(dist, data)[keep]
        prepared = stop_invalid_gradients(dist, dist.parameter_issues().invalid_entries)
        return contributions(prepared, time)[keep]

    def alone(values):
        p = unravel(values)
        p = {k: v[keep] if k in per_row else v for k, v in p.items()}
        return contributions(build(kind, p), time[keep])

    dist = build(kind, unravel(flat))
    if evaluation == "core":
        assert jnp.isnan(_log_likelihood(dist, data)[1])
    np.testing.assert_array_equal(
        dist.parameter_issues().invalid_entries, [False, True, False]
    )
    prepared = stop_invalid_gradients(dist, dist.parameter_issues().invalid_entries)
    for before, after in zip(
        jax.tree.leaves(dist), jax.tree.leaves(prepared), strict=True
    ):
        np.testing.assert_array_equal(after, before)
        assert after.dtype == before.dtype
    if kind == "rate":
        # Exact log(r)-r: gradient 1/r-1, Hessian -1/r²; right -3r: -3,0.
        # At r=0.5 these derivatives are exactly representable binary values.
        np.testing.assert_array_equal(
            jax.jacrev(together)(flat), [[1, 0, 0], [0, 0, -3]]
        )
        expected = np.zeros((2, 3, 3))
        expected[0, 0, 0] = -4
        np.testing.assert_array_equal(jax.hessian(together)(flat), expected)
    # Identity: removing the bad row cannot change valid per-row contributions.
    transforms = [
        ("identity", lambda f: f),
        ("jacfwd", jax.jacfwd),
        ("jacrev", jax.jacrev),
    ]
    transforms += [
        (f"{outer}-over-{inner}", lambda f, a=a, b=b: a(b(f)))
        for outer, a in (("fwd", jax.jacfwd), ("rev", jax.jacrev))
        for inner, b in (("fwd", jax.jacfwd), ("rev", jax.jacrev))
    ]
    transforms += [
        ("jit-jacrev", lambda f: jax.jit(jax.jacrev(f))),
        ("jit-hessian", lambda f: jax.jit(jax.hessian(f))),
    ]
    for name, transform in transforms:
        actual = transform(together)(flat)
        assert np.isfinite(actual).all(), name
        np.testing.assert_allclose(
            actual,
            transform(alone)(flat),
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=name,
        )
    mapped = jax.jit(jax.vmap(jax.jacrev(together)))(jnp.stack([flat, flat]))
    np.testing.assert_allclose(
        mapped,
        jnp.stack([jax.jacrev(alone)(flat)] * 2),
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
        err_msg="jit-vmap-jacrev",
    )


def test_rebuilt_nested_leaves_broadcast_before_masking_without_construction():
    """Spec 7.1: outer union expands across smaller nested leaves and two component axes."""
    dist = NestedRate(
        base=ComponentRate(rate=1.0, weights=[[0.25, 0.75]], free=0.1),
        factor=[[0.0], [3.0]],
    )
    tree = jax.tree.structure(dist)
    original = [
        jnp.array(1.0, dtype=jnp.float32),
        jnp.array([[0.25, 0.75]]),
        jnp.array([0.1]),
        dist.factor,
    ]
    rebuilt = jax.tree.unflatten(tree, original)
    shapes = [leaf.shape for leaf in jax.tree.leaves(rebuilt)]
    mask = rebuilt.parameter_issues().invalid_entries
    np.testing.assert_array_equal(mask, [[False], [True]])
    with (
        patch.object(
            ComponentRate, "__init__", side_effect=AssertionError("constructor")
        ),
        patch.object(NestedRate, "__init__", side_effect=AssertionError("constructor")),
    ):
        prepared = jax.jit(stop_invalid_gradients)(rebuilt, mask)
    assert type(prepared) is NestedRate and type(prepared.base) is ComponentRate
    assert prepared.label == "nested" and prepared.base.label == "component"
    assert prepared.batch_shape == prepared.base.batch_shape == (2, 1)
    assert prepared.base.weights.shape == (2, 1, 1, 2)
    for before, after in zip(original, jax.tree.leaves(prepared), strict=True):
        np.testing.assert_array_equal(after, np.broadcast_to(before, after.shape))
        assert after.dtype == before.dtype
    assert [leaf.shape for leaf in jax.tree.leaves(rebuilt)] == shapes

    def free_values(free):
        current = jax.tree.unflatten(tree, [*original[:2], free, original[3]])
        return stop_invalid_gradients(current, mask).base.free

    # A shared unconstrained leaf has derivative one only at valid outer entries.
    np.testing.assert_array_equal(
        jax.jacrev(free_values)(jnp.array(0.1)), [[1.0], [0.0]]
    )


def test_per_leaf_vmap_preserves_batch_shapes_and_masks_unconstrained_sibling():
    """Spec 7.1: per-leaf vmap broadcasts mapped scalars before masking shared leaves."""
    dist = ComponentRate(rate=[0.5, -1.0, 2.0], weights=[[0.25, 0.75]], free=0.1)
    tree = jax.tree.structure(dist)
    axes = jax.tree.unflatten(tree, [0, None, None])

    def prepare(current):
        return stop_invalid_gradients(
            current, current.parameter_issues().invalid_entries
        )

    def mapped(free):
        current = jax.tree.unflatten(tree, [dist.rate, dist.weights, free])
        return jax.vmap(prepare, in_axes=(axes,))(current)

    prepared = jax.jit(mapped)(dist.free)
    assert prepared.rate.shape == prepared.free.shape == (3, 3)
    assert prepared.weights.shape == (3, 3, 1, 2)
    expected = np.stack([np.eye(3), np.zeros((3, 3)), np.eye(3)])
    np.testing.assert_array_equal(
        jax.jacrev(lambda free: mapped(free).free)(dist.free), expected
    )


def test_draw_batch_preserves_values_and_derivative_shape():
    """Spec 5/7.1: a (draws,1) mask retains its column axis before observation broadcasting."""

    def rates(rate):
        dist = survjax.Exponential(rate=rate)
        return stop_invalid_gradients(
            dist, dist.parameter_issues().invalid_entries
        ).rate

    rate = jnp.array([[0.5], [np.nan]])
    prepared = jax.jit(rates)(rate)
    assert prepared.shape == (2, 1)
    np.testing.assert_array_equal(prepared, rate)
    # Identity leaf: valid output has derivative one w.r.t. its own entry.
    np.testing.assert_array_equal(
        jax.jacrev(lambda r: rates(r)[0])(rate), [[[1.0], [0.0]]]
    )
