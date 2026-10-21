"""Prepared-data likelihood contracts (spec 3.3/5/6.2/7.1–7.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax
from survjax.likelihood import _log_likelihood


@survjax.register_family
@dataclass(frozen=True)
class DefaultDensity(survjax.SurvivalDistribution):
    base: survjax.SurvivalDistribution = survjax.distribution()
    factor: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def log_survival(self, t: jax.Array) -> jax.Array:
        # A test probe: promotion must reach own/nested leaves before family code.
        assert all(leaf.dtype == t.dtype for leaf in jax.tree.leaves(self))
        return self.factor * self.base.log_survival(t)

    def reference_time(self) -> jax.Array:
        return jnp.ones(self.batch_shape, dtype=self.factor.dtype)


@survjax.register_family
@dataclass(frozen=True)
class Uniform(survjax.SurvivalDistribution):
    def log_survival(self, t: jax.Array) -> jax.Array:
        inside = t < 1
        safe = jnp.where(inside, t, 0.0)
        return jnp.where(inside, jnp.log1p(-safe), -jnp.inf)

    def log_prob(self, t: jax.Array) -> jax.Array:
        return jnp.where(t < 1, 0.0, -jnp.inf)

    def reference_time(self) -> jax.Array:
        return jnp.asarray(0.5)


@pytest.mark.parametrize("family", ["exponential", "weibull"])
def test_mixed_values_transform_without_nan_operations(family, x64_enabled):
    """Spec 7.1/7.2: hand-derived exact/right/zero-censoring contributions."""
    data = survjax.SurvivalData([1.0, 2.0, 0.0], [True, False, False])
    dist = (
        survjax.Exponential(rate=2.0)
        if family == "exponential"
        else survjax.Weibull(shape=2.0, scale=2.0)
    )
    # log f=log r-r*t, log S=-r*t; Weibull H=(t/2)^2, log h=log(t/2).
    expected = (
        [np.log(2) - 2, -4, 0]
        if family == "exponential"
        else [-np.log(2) - 0.25, -1, 0]
    )
    with jax.debug_nans(True):
        value = _log_likelihood(dist, data)
    np.testing.assert_allclose(
        value, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    np.testing.assert_array_equal(jax.jit(_log_likelihood)(dist, data), value)
    mapped = jax.vmap(_log_likelihood, in_axes=(0, None))(
        jax.tree.map(lambda x: jnp.stack([x, x]), dist), data
    )
    np.testing.assert_array_equal(mapped, jnp.stack([value, value]))


def test_custom_defaults_receive_common_dtype_on_own_and_nested_leaves(x64_enabled):
    """Spec 3.3/4.2: autodiff density dispatch sees promoted temporary parameters."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    base = survjax.Exponential(rate=jnp.array(1.0, dtype=jnp.float32))
    dist = DefaultDensity(base=base, factor=jnp.array(2.0, dtype=jnp.float32))
    data = survjax.SurvivalData(jnp.array([1.0, 2.0], dtype=dtype), [True, False])
    value = jax.jit(_log_likelihood)(dist, data)
    assert value.dtype == dtype
    assert all(leaf.dtype == jnp.float32 for leaf in jax.tree.leaves(dist))
    # The custom survival is -2*t, hence its default density is log(2)-2*t.
    np.testing.assert_allclose(
        value, [np.log(2) - 2, -4], atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


def test_true_zero_density_and_survival_remain_negative_infinity():
    """Spec 7.1: uniform (0,1) has no density or right-tail mass at t=2."""
    value = _log_likelihood(Uniform(), survjax.SurvivalData([2.0, 2.0], [True, False]))
    np.testing.assert_array_equal(value, [-np.inf, -np.inf])


@pytest.mark.parametrize(
    "rates,shape",
    [(2.0, (2,)), ([1.0, 2.0], (2,)), ([[1.0], [2.0]], (2, 2))],
    ids=["scalar", "paired", "draws"],
)
def test_trailing_axis_broadcasting(rates, shape, x64_enabled):
    """Spec 5: exact log(r)-r and right -2*r, with every broadcast axis retained."""
    data = survjax.SurvivalData([1.0, 2.0], [True, False])
    value = _log_likelihood(survjax.Exponential(rate=rates), data)
    rate = np.asarray(rates)
    time = np.array([1.0, 2.0])
    expected = np.where([True, False], np.log(rate) - rate * time, -rate * time)
    assert value.shape == shape
    np.testing.assert_allclose(
        value, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    scalar = _log_likelihood(survjax.Exponential(rate=2.0), data[0])
    assert scalar.shape == ()


def test_incompatible_batch_shapes_raise_type_error():
    """Spec 5/6.2: static mismatches fail eagerly and during tracing."""
    for evaluate in (_log_likelihood, jax.jit(_log_likelihood)):
        with pytest.raises(TypeError, match="shapes"):
            evaluate(survjax.Exponential(rate=[1, 2, 3]), survjax.SurvivalData([1, 2]))


def test_stored_input_issues_survive_indexing_and_reconstruction():
    """Spec 6.2/7.1: invalid raw event=2 remains invalid after bool normalization."""
    data = survjax.SurvivalData([1.0, 2.0], [1, 2])[jnp.array([1, 0])]
    rebuilt = jax.tree.unflatten(jax.tree.structure(data), jax.tree.leaves(data))
    value = jax.jit(_log_likelihood)(survjax.Exponential(rate=1.0), rebuilt)
    assert jnp.isnan(value[0]) and value[1] == -1
    np.testing.assert_array_equal(rebuilt.issues.invalid_rows, [True, False])


@pytest.mark.parametrize("field,bad", [(1, np.nan), (1, 0), (2, 3), (3, -1), (4, 0)])
def test_rebuilt_normalized_fields_are_rechecked(field, bad):
    """Spec 3.2/6.2/7.1: zero old issue bits cannot hide newly invalid fields."""
    data = survjax.SurvivalData([1.0, 2.0])
    leaves, tree = jax.tree.flatten(data)
    leaves[field] = leaves[field].at[1].set(bad)
    rebuilt = jax.tree.unflatten(tree, leaves)
    assert rebuilt.issues.ok
    for evaluate in (_log_likelihood, jax.jit(_log_likelihood)):
        value = evaluate(survjax.Exponential(rate=1.0), rebuilt)
        assert value[0] == -1 and jnp.isnan(value[1])


def test_zero_censoring_has_zero_parameter_gradient_and_hessian():
    """Spec 7.1: S(0)=1 makes this uninformative contribution structurally zero."""
    data = survjax.SurvivalData.from_bounds(0.0, np.inf)

    def value(rate):
        return _log_likelihood(survjax.Exponential(rate=rate), data)

    assert data.issues.uninformative and value(2.0) == 0
    assert jax.jit(jax.grad(value))(2.0) == jax.hessian(value)(2.0) == 0


@pytest.mark.parametrize("paired", [False, True], ids=["shared", "per-row"])
@pytest.mark.parametrize(
    "lower,upper,exact,left,right",
    [
        (0, 3, False, 0, np.inf),
        (1, 3, False, 0, np.inf),
        (2, 2, True, 1, np.inf),
        (2, np.inf, False, 1, np.inf),
        (2, 2, True, 0, 5),
    ],
)
def test_unsupported_modes_preserve_supported_values_and_derivatives(
    lower, upper, exact, left, right, paired, x64_enabled
):
    """Spec 7.1/7.3: mode support is separate from issues, with safe inactive pieces."""
    data = survjax.SurvivalData.from_bounds(
        [1.0, lower],
        [1.0, upper],
        exact=[True, exact],
        left_trunc=[0, left],
        right_trunc=[np.inf, right],
    )

    def supported(rate):
        return _log_likelihood(survjax.Exponential(rate=rate), data)[0]

    rate = jnp.array([0.5, 0.5] if paired else 0.5)
    assert data.issues.ok
    assert not survjax.Exponential(rate=rate).parameter_issues().invalid_entries.any()
    for evaluate in (_log_likelihood, jax.jit(_log_likelihood)):
        assert jnp.isnan(evaluate(survjax.Exponential(rate=rate), data)[1])
    # Exact log(r)-r at r=.5: gradient=1, Hessian=-4; other-row terms are zero.
    np.testing.assert_allclose(
        supported(rate),
        np.log(0.5) - 0.5,
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )
    gradient = [1, 0] if paired else 1
    hessian = [[-4, 0], [0, 0]] if paired else -4
    for first in (jax.jacfwd, jax.jacrev):
        np.testing.assert_allclose(
            jax.jit(first(supported))(rate),
            gradient,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            err_msg=first.__name__,
        )
        for second in (jax.jacfwd, jax.jacrev):
            np.testing.assert_allclose(
                second(first(supported))(rate),
                hessian,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
                err_msg=f"{second.__name__}-over-{first.__name__}",
            )


def test_traced_construction_uses_fields_instead_of_conservative_metadata():
    """Spec 3.2/7.2: traced may-contain flags are not evidence of unsupported modes."""

    def evaluate(time, event):
        data = survjax.SurvivalData(time, event)
        return _log_likelihood(survjax.Exponential(rate=1.0), data)

    value = jax.jit(evaluate)(jnp.array([1.0, 2.0]), jnp.array([True, False]))
    np.testing.assert_array_equal(value, [-1, -2])
