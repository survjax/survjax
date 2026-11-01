"""PH likelihood isolates declared-invalid rows (spec 4.4/4.7/7.1–7.3)."""

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax
from survjax.distributions.wrappers import ProportionalHazards

TIME = jnp.array([1.0, 2.0, 3.0])
EVENT = jnp.array([True, False, False])
KEEP = jnp.array([0, 2])
TEMPLATES = {
    "rate": [0.8, 1.0, 1.3, 0.2],
    "shape": [0.7, 1.2, 1.6, 2.0, 0.2],
    "scale": [1.5, 2.0, 2.7, 1.4, 0.2],
    "eta-exponential": [0.1, 0.0, -0.2, 1.2],
    "eta-weibull": [0.1, 0.0, -0.2, 1.4, 2.0],
    "condition": [1.0, 1.5, 2.0, 4.0, 0.2],
}


@survjax.register_family
@dataclass(frozen=True)
class GapBaseline(survjax.SurvivalDistribution):
    """Positive fields alone do not guarantee a positive square-root gap."""

    lower: jax.Array = survjax.param(constraint=survjax.constraints.positive)
    upper: jax.Array = survjax.param(constraint=survjax.constraints.positive)

    def log_survival(self, t):
        return -jnp.sqrt(self.upper - self.lower) * jnp.asarray(t)

    def reference_time(self):
        return jnp.ones(self.batch_shape, dtype=self.lower.dtype)

    def parameter_conditions(self):
        return {"nonpositive_gap": self.upper <= self.lower}


def _cases(kind):
    bad = [np.nan, np.inf, -np.inf]
    if kind == "condition":
        bad = [4.0, 5.0]  # Both fields stay positive; only the gap is invalid.
    elif not kind.startswith("eta"):
        bad = [0.0, -1.0, *bad]
    return jnp.stack([jnp.array(TEMPLATES[kind]).at[1].set(b) for b in bad])


def _distribution(p, kind):
    if kind == "rate":
        baseline, eta = survjax.Exponential(rate=p[:3]), p[3]
    elif kind == "shape":
        baseline, eta = survjax.Weibull(shape=p[:3], scale=p[3]), p[4]
    elif kind == "scale":
        baseline, eta = survjax.Weibull(shape=p[3], scale=p[:3]), p[4]
    elif kind == "eta-exponential":
        baseline, eta = survjax.Exponential(rate=p[3]), p[:3]
    elif kind == "eta-weibull":
        baseline, eta = survjax.Weibull(shape=p[3], scale=p[4]), p[:3]
    else:
        baseline, eta = GapBaseline(lower=p[:3], upper=p[3]), p[4]
    return ProportionalHazards(baseline, log_hazard_ratio=eta)


def _formula(p, kind, time=TIME, event=EVENT, indices=KEEP):
    # Hand derivation: ell=event*log(h)-H. Gather valid fields BEFORE arithmetic,
    # so the oracle never evaluates sqrt/log/powers at the invalid coordinate.
    t, d = time[indices], event[indices]
    if kind in ("rate", "eta-exponential", "condition"):
        if kind == "rate":
            log_rate = jnp.log(p[indices]) + p[3]
        elif kind == "eta-exponential":
            log_rate = jnp.log(p[3]) + p[indices]
        else:
            log_rate = 0.5 * jnp.log(p[3] - p[indices]) + p[4]
        return jnp.where(d, log_rate, 0.0) - jnp.exp(log_rate) * t
    if kind == "shape":
        k, b, eta = p[indices], p[3], p[4]
    elif kind == "scale":
        k, b, eta = p[3], p[indices], p[4]
    else:
        k, b, eta = p[3], p[4], p[indices]
    # Weibull PH: H=exp(eta+k*log(t/b)), h=exp(eta)*k/b*(t/b)**(k-1).
    log_ratio = jnp.log(t / b)
    log_hazard = jnp.log(k) - jnp.log(b) + (k - 1) * log_ratio + eta
    return jnp.where(d, log_hazard, 0.0) - jnp.exp(eta + k * log_ratio)


@pytest.mark.parametrize("kind", TEMPLATES)
def test_invalid_rows_have_nan_values_and_prefixed_diagnostics(kind, x64_enabled):
    """Spec 4.7/7.1: each declared violation masks precisely one row."""
    roots = _cases(kind)

    def evaluate(p):
        dist = _distribution(p, kind)
        return dist.log_likelihood(time=TIME, event=EVENT), dist.parameter_issues()

    expected = jax.vmap(lambda p: _formula(p, kind))(roots)
    for evaluate_many in (jax.vmap(evaluate), jax.jit(jax.vmap(evaluate))):
        values, issues = evaluate_many(roots)
        assert np.isnan(values[:, 1]).all()
        np.testing.assert_allclose(
            values[:, KEEP],
            expected,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
        )
        invalid = np.broadcast_to([False, True, False], values.shape)
        np.testing.assert_array_equal(issues.invalid_entries, invalid)
        assert not np.asarray(issues.ok).any()
        baseline_fields = (
            ["rate"] if kind in ("rate", "eta-exponential") else ["shape", "scale"]
        )
        if kind == "condition":
            baseline_fields = ["lower", "upper"]
        assert list(issues.fields) == [
            "log_hazard_ratio",
            *[f"baseline.{n}" for n in baseline_fields],
        ]
        broken = "log_hazard_ratio" if kind.startswith("eta") else f"baseline.{kind}"
        for name, mask in issues.fields.items():
            np.testing.assert_array_equal(
                mask, invalid if name == broken else np.zeros_like(invalid)
            )
        assert list(issues.conditions) == (
            ["baseline.nonpositive_gap"] if kind == "condition" else []
        )
        if kind == "condition":
            np.testing.assert_array_equal(
                issues.conditions["baseline.nonpositive_gap"], invalid
            )
        eager_values, eager_issues = evaluate(roots[0])
        assert np.isnan(eager_values[1])
        np.testing.assert_allclose(
            eager_values[KEEP],
            expected[0],
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
        )
        for eager, mapped in zip(
            jax.tree.leaves(eager_issues), jax.tree.leaves(issues), strict=True
        ):
            np.testing.assert_array_equal(eager, mapped[0])


@pytest.mark.parametrize("kind", TEMPLATES)
@pytest.mark.parametrize("orders", ["f", "r", "ff", "fr", "rf", "rr"])
def test_valid_contribution_derivatives_survive_invalid_neighbors(
    kind, orders, x64_enabled
):
    """Spec 7.1/7.3: only valid outputs are differentiated, through order two."""
    roots = _cases(kind)

    def actual(p):
        return _distribution(p, kind).log_likelihood(time=TIME, event=EVENT)[KEEP]

    def expected(p):
        return _formula(p, kind)

    for order in reversed(orders):
        differentiate = jax.jacfwd if order == "f" else jax.jacrev
        actual, expected = differentiate(actual), differentiate(expected)
    reference = jax.vmap(expected)(roots)
    # Entries are O(1–10); exact AD/analytic comparisons incur rounding only,
    # not finite-difference error or differentiation across invalid coordinates.
    for evaluate in (jax.vmap(actual), jax.jit(jax.vmap(actual))):
        np.testing.assert_allclose(
            evaluate(roots),
            reference,
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
        )


@pytest.mark.parametrize(
    "method",
    "log_survival survival log_cdf cumulative_hazard log_hazard hazard log_prob".split(),
)
def test_custom_baseline_grid_values_and_derivatives_match_scaled_rate(
    method, x64_enabled
):
    """Spec 4.2/4.4/7.3: generic baseline defaults compose with PH and grids."""
    roots = jnp.stack(
        [jnp.array(TEMPLATES["condition"]), jnp.array([1.1, 1.6, 2.1, 4.2, 0.3])]
    )
    times = jnp.array([0.4, 1.6])

    def actual(p):
        return getattr(_distribution(p, "condition"), method + "_grid")(times)

    def equivalent(p):
        # H0=sqrt(upper-lower)*t; PH multiplies its rate by exp(eta).
        dist = survjax.Exponential(rate=jnp.sqrt(p[3] - p[:3]) * jnp.exp(p[4]))
        return getattr(dist, method + "_grid")(times)

    for transform in (lambda f: f, jax.jacfwd, jax.jacrev):
        expected = jax.vmap(transform(equivalent))(roots)
        for evaluate in (
            jax.vmap(transform(actual)),
            jax.jit(jax.vmap(transform(actual))),
        ):
            np.testing.assert_allclose(
                evaluate(roots),
                expected,
                atol=ATOL[x64_enabled],
                rtol=RTOL[x64_enabled],
                equal_nan=False,
            )
        np.testing.assert_allclose(
            transform(actual)(roots[0]),
            expected[0],
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
        )


@pytest.mark.parametrize("kind", TEMPLATES)
def test_runtime_parameter_issues_and_raw_observations_reuse_one_trace(
    kind, x64_enabled
):
    """Spec 4.7/7.3: numeric validity is dynamic, with fixed shapes and dtypes."""
    traces = 0

    def evaluate(p, time, event):
        nonlocal traces
        traces += 1
        dist = _distribution(p, kind)
        return dist.log_likelihood(time=time, event=event), dist.parameter_issues()

    compiled = jax.jit(evaluate)
    cases = [
        (_cases(kind)[0], TIME, EVENT),
        (jnp.array(TEMPLATES[kind]), TIME * 0.8, ~EVENT),
        (_cases(kind)[-1], TIME * 1.1, EVENT),
    ]
    for i, (p, time, event) in enumerate(cases):
        values, issues = compiled(p, time, event)
        indices = jnp.arange(3) if i == 1 else KEEP
        np.testing.assert_allclose(
            values[indices],
            _formula(p, kind, time, event, indices),
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
            equal_nan=False,
        )
        np.testing.assert_array_equal(issues.invalid_entries, [False, i != 1, False])
        assert bool(issues.ok) == (i == 1)
        if i != 1:
            assert np.isnan(values[1])
    assert traces == 1
