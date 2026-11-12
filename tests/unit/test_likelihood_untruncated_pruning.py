"""Exact/right specialization preserves spec 2, 3.2 and 7.1–7.3."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


def _conservative(data):
    return jax.jit(
        lambda d: survjax.SurvivalData.from_bounds(
            d.lower,
            d.upper,
            exact=d.exact,
            left_trunc=d.left_trunc,
            right_trunc=d.right_trunc,
        )
    )(data)


def test_untruncated_exact_right_omits_both_interval_hooks(monkeypatch):
    """Spec 2/7.2: Z=1; exact log f and right log S need no interval hooks."""
    data = survjax.SurvivalData([1.0, 2.0, 0.0], [True, False, False])

    def forbidden(*args):
        raise AssertionError("unneeded interval hook")

    monkeypatch.setattr(survjax.Exponential, "log_interval_prob", forbidden)
    monkeypatch.setattr(survjax.Exponential, "log_conditional_interval_prob", forbidden)
    for evaluate in (
        lambda d: d.log_likelihood(data),
        jax.jit(lambda d: d.log_likelihood(data)),
    ):
        # Unit-rate log f(t)=log S(t)=-t; right at zero is uninformative.
        np.testing.assert_array_equal(
            evaluate(survjax.Exponential(rate=1.0)), [-1, -2, 0]
        )


@pytest.mark.parametrize("mode", ["left", "interval", "left_trunc", "right_trunc"])
def test_each_disqualifying_flag_retains_conditional_hook(
    mode, monkeypatch, x64_enabled
):
    """Spec 2/3.2: each of flags 2–5 independently rules out the shortcut."""
    lower, upper, left, right = {
        "left": ([0.0, 0.0], [2.0, 3.0], 0.0, np.inf),
        "interval": ([0.5, 1.0], [2.0, 3.0], 0.0, np.inf),
        "left_trunc": ([2.0, 3.0], [np.inf, np.inf], 1.0, np.inf),
        "right_trunc": ([1.0, 2.0], [np.inf, np.inf], 0.0, 4.0),
    }[mode]
    data = survjax.SurvivalData.from_bounds(
        lower, upper, left_trunc=left, right_trunc=right
    )
    original = survjax.Exponential.log_conditional_interval_prob
    calls = []

    def record(self, *args):
        calls.append(None)
        return original(self, *args)

    monkeypatch.setattr(survjax.Exponential, "log_conditional_interval_prob", record)
    result = jax.jit(lambda d: survjax.Exponential(rate=1.0).log_likelihood(d))(data)
    # Unit-rate exponential conditional mass: (e^-a-e^-b)/(e^-L-e^-U).
    expected = np.log(
        (np.exp(-np.maximum(lower, left)) - np.exp(-np.minimum(upper, right)))
        / (np.exp(-left) - np.exp(-right))
    )
    assert calls
    np.testing.assert_allclose(
        result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


@survjax.register_family
@dataclass(frozen=True)
class Cure(survjax.SurvivalDistribution):
    rate: jax.Array = survjax.param(constraint=survjax.constraints.positive)
    cure: jax.Array = survjax.param(constraint=survjax.constraints.unit_interval)

    def reference_time(self):
        return 1 / self.rate

    def log_survival(self, t):
        return jnp.logaddexp(jnp.log(self.cure), jnp.log1p(-self.cure) - self.rate * t)

    def log_survival_at_infinity(self):
        return jnp.log(self.cure)


def _cure_reference(rate, cure):
    # Exact at 1: log((1-c)*r)-r. Right at 2: log(c+(1-c)*exp(-2r)).
    tail = np.exp(-2 * rate)
    survival = cure + (1 - cure) * tail
    ds = np.array([-2 * (1 - cure) * tail, 1 - tail])
    d2s = np.array([[4 * (1 - cure) * tail, 2 * tail], [2 * tail, 0]])
    values = [np.log1p(-cure) + np.log(rate) - rate, np.log(survival), 0]
    gradients = [[1 / rate - 1, -1 / (1 - cure)], ds / survival, [0, 0]]
    # Hess(log S) = Hess(S)/S - grad(S) outer grad(S)/S^2.
    hessians = [
        np.diag([-1 / rate**2, -1 / (1 - cure) ** 2]),
        d2s / survival - np.outer(ds, ds) / survival**2,
        np.zeros((2, 2)),
    ]
    return tuple(np.asarray(v) for v in (values, gradients, hessians))


def test_cure_mass_and_joint_derivatives_match_independent_algebra(x64_enabled):
    """Spec 2/7.3: closed-at-infinity sets include cure mass; Z remains one."""
    data = survjax.SurvivalData([1.0, 2.0, 0.0], [True, False, False])
    points = jnp.asarray([[0.5, 0.25], [0.75, 0.375]])
    references = [_cure_reference(*map(float, p)) for p in points]
    expected = [np.stack([r[order] for r in references]) for order in range(3)]
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}
    for observations in (data, _conservative(data)):

        def rows(p, observations=observations):
            return Cure(rate=p[0], cure=p[1]).log_likelihood(observations)

        with jax.debug_nans(True):
            np.testing.assert_allclose(rows(points[0]), expected[0][0], **tol)
        transforms = [(lambda f: f, 0), (jax.jacfwd, 1), (jax.jacrev, 1)]
        for outer in (jax.jacfwd, jax.jacrev):
            for inner in (jax.jacfwd, jax.jacrev):
                transforms.append((lambda f, o=outer, i=inner: o(i(f)), 2))
        for transform, order in transforms:
            evaluate = transform(rows)
            np.testing.assert_allclose(evaluate(points[0]), expected[order][0], **tol)
            mapped = jax.jit(jax.vmap(evaluate))(points)
            np.testing.assert_allclose(mapped, expected[order], **tol)
            np.testing.assert_array_equal(mapped[:, -1], 0)


def test_inherited_truncation_metadata_keeps_general_path(monkeypatch):
    """Spec 3.2/7.2: slicing preserves conservative flags and identical values."""
    parent = survjax.SurvivalData(
        [1.0, 2.0, 3.0], [True, False, False], left_trunc=[0.0, 0.0, 1.0]
    )
    data = parent[:2]
    calls = []
    original = survjax.Exponential.log_conditional_interval_prob

    def record(self, *args):
        calls.append(None)
        return original(self, *args)

    monkeypatch.setattr(survjax.Exponential, "log_conditional_interval_prob", record)
    np.testing.assert_array_equal(
        jax.jit(lambda d: survjax.Exponential(rate=1.0).log_likelihood(d))(data),
        [-1, -2],
    )
    assert calls


def test_fast_path_checks_invalid_current_fields_and_stored_issues():
    """Spec 7.1: skipping window arithmetic cannot skip window validation."""
    data = survjax.SurvivalData([1.0, 2.0], [True, False])
    evaluate = jax.jit(lambda d: survjax.Exponential(rate=1.0).log_likelihood(d))
    for field, invalid in ((3, np.nan), (4, np.nan), (4, -1.0)):
        leaves, tree = jax.tree.flatten(data)
        leaves[field] = leaves[field].at[1].set(invalid)
        rebuilt = jax.tree.unflatten(tree, leaves)
        assert rebuilt.issues.ok
        np.testing.assert_array_equal(evaluate(rebuilt), [-1, np.nan])
    np.testing.assert_array_equal(
        evaluate(survjax.SurvivalData([1.0, 2.0], [1, 2])), [-1, np.nan]
    )
