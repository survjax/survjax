"""Static absence may remove work, never validation (spec 3.2/7.1–7.3)."""

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


@pytest.mark.parametrize("kind", ["exact", "censored", "empty"])
def test_absent_pieces_are_not_called_eagerly_or_while_tracing(kind, monkeypatch):
    """Spec 7.2: absence prevents hook invocation, rather than hiding outputs."""
    data = survjax.SurvivalData(
        [] if kind == "empty" else [1.0, 2.0],
        None if kind != "censored" else [False, False],
    )
    omitted = []
    if kind != "exact":
        omitted += ["log_prob", "log_interval_prob"]
    if kind == "empty":
        # A retained family hook may itself need reference_time for safe inputs.
        omitted += ["reference_time"]
    if kind != "censored":
        omitted += ["log_conditional_interval_prob"]

    def forbidden(*args):
        raise AssertionError("statically absent likelihood piece was evaluated")

    for name in omitted:
        monkeypatch.setattr(survjax.Exponential, name, forbidden)
    for evaluate in (
        lambda d: d.log_likelihood(data),
        jax.jit(lambda d: d.log_likelihood(data)),
    ):
        # At unit rate, both log f(t) and log S(t) equal -t.
        np.testing.assert_array_equal(
            evaluate(survjax.Exponential(rate=1.0)), -data.lower
        )


@pytest.mark.parametrize("exact", [True, False])
def test_pruned_derivatives_match_delayed_entry_algebra(exact, x64_enabled):
    """Spec 2/7.3: log(r)-r(t-L) or -r(c-L), including right at entry."""
    times = [2.0, 3.0] if exact else [1.0, 2.0]
    data = survjax.SurvivalData(times, [exact, exact], left_trunc=1.0)
    conservative = _conservative(data)
    rates = jnp.asarray([0.5, 0.75])
    distance = np.asarray(times) - 1
    expected = [
        exact * np.log(np.asarray(rates))[:, None]
        - np.asarray(rates)[:, None] * distance,
        exact / np.asarray(rates)[:, None] - np.broadcast_to(distance, (2, 2)),
        np.broadcast_to(-exact / np.asarray(rates)[:, None] ** 2, (2, 2)),
    ]
    tol = {"atol": ATOL[x64_enabled], "rtol": RTOL[x64_enabled]}
    for observations in (data, conservative):

        def rows(rate, observations=observations):
            return survjax.Exponential(rate=rate).log_likelihood(observations)

        with jax.debug_nans(True):
            np.testing.assert_allclose(rows(rates[0]), expected[0][0], **tol)
        transforms = [(lambda f: f, 0), (jax.jacfwd, 1), (jax.jacrev, 1)]
        for outer in (jax.jacfwd, jax.jacrev):
            for inner in (jax.jacfwd, jax.jacrev):
                transforms.append((lambda f, o=outer, i=inner: o(i(f)), 2))
        for transform, order in transforms:
            evaluate = transform(rows)
            np.testing.assert_allclose(evaluate(rates[0]), expected[order][0], **tol)
            np.testing.assert_allclose(
                jax.jit(jax.vmap(evaluate))(rates), expected[order], **tol
            )


def test_censored_only_mixed_kinds_and_windows_match_ratios(x64_enabled):
    """Spec 2/7.2: pruned left, interval and right rows retain the conditional hook."""
    data = survjax.SurvivalData.from_bounds(
        [0.0, 0.5, 1.0], [2.0, 3.0, np.inf], right_trunc=4.0
    )
    # Unit-rate exponential: P(a<T<=b | T<=4)=(exp(-a)-exp(-min(b,4)))/(1-exp(-4)).
    expected = np.log(
        (np.exp(-np.array([0.0, 0.5, 1.0])) - np.exp(-np.array([2.0, 3.0, 4.0])))
        / -np.expm1(-4.0)
    )
    for observations in (data, _conservative(data)):
        value = jax.jit(lambda d: survjax.Exponential(rate=1.0).log_likelihood(d))(
            observations
        )
        np.testing.assert_allclose(
            value, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


def test_slices_reuse_parent_metadata_and_compilation():
    """Spec 3.2/7.2: slices can omit a kind without changing the parent's flags."""
    parent = survjax.SurvivalData([1.0, 2.0, 3.0, 4.0], [True, True, False, False])
    traces = []

    @jax.jit
    def evaluate(data, rate):
        traces.append(None)
        return survjax.Exponential(rate=rate).log_likelihood(data)

    np.testing.assert_array_equal(evaluate(parent[:2], jnp.asarray(1.0)), [-1, -2])
    np.testing.assert_array_equal(evaluate(parent[2:], jnp.asarray(2.0)), [-6, -8])
    assert len(traces) == 1
    for event in (True, False):
        for times in ([1.0, 2.0], [2.0, 3.0]):
            evaluate(survjax.SurvivalData(times, [event, event]), jnp.asarray(1.0))
    assert len(traces) == 3


@pytest.mark.parametrize("exact", [True, False])
def test_pruning_preserves_invalid_data_parameters_and_shapes(exact):
    """Spec 3.3/7.1: stored/current faults and invalid parameters stay NaN."""
    data = survjax.SurvivalData([1.0, 2.0], [exact, exact])
    leaves, tree = jax.tree.flatten(data)
    leaves[3] = leaves[3].at[1].set(-1.0)
    rebuilt = jax.tree.unflatten(tree, leaves)
    assert rebuilt.issues.ok  # Current-field validation must still detect this.
    stored = survjax.SurvivalData([1.0, 2.0], [0, 2])
    evaluate = jax.jit(lambda d, rate: survjax.Exponential(rate=rate).log_likelihood(d))
    for observations in (rebuilt, stored):
        np.testing.assert_array_equal(
            evaluate(observations, jnp.asarray(1.0)), [-1, np.nan]
        )
    values = evaluate(data, jnp.asarray([[1.0], [-1.0]]))
    np.testing.assert_array_equal(values, [[-1, -2], [np.nan, np.nan]])

    # The valid first row is exact*log(r)-r; the invalid second rate is isolated.
    def objective(p):
        return evaluate(data, p)[0]

    rates = jnp.asarray([0.5, -1.0])
    np.testing.assert_array_equal(jax.grad(objective)(rates), [2 * exact - 1, 0])
    np.testing.assert_array_equal(
        jax.hessian(objective)(rates), [[-4 * exact, 0], [0, 0]]
    )
    empty = survjax.SurvivalData([])
    assert evaluate(empty, jnp.ones((2, 1))).shape == (2, 0)
    assert evaluate(data[0], jnp.asarray(1.0)).shape == ()
