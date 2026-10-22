"""A6: NaN likelihood rows for declared parameter violations (spec 4.7, 7.1; rev 29).

Hand-derived exponential contributions (spec 2, 4.1): with ``log S = -r t`` and
``log h = log r``, an exact row gives ``log f = log r - r t`` and a
right-censored row gives ``log S = -r t``. Parameter derivatives:
``d/dr (log r - r t) = 1/r - t``, ``d/dr (-r t) = -t``.

Since revision 29 there is no ``LikelihoodIssues``: rows marked by
``dist.parameter_issues().invalid_entries``, broadcast to the contribution
shape, are NaN, and the documented triage recipe (7.1) reads the causes of NaN
rows from ``data.issues`` and ``dist.parameter_issues()`` (``helpers.nan_causes``).
Revision 32 widens the isolation of rows without issues to second derivatives,
both modes, every violation class and custom and nested families; that suite is
``test_a6_likelihood_gradient_isolation.py``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, b, f, ll_method, nan_causes

import survjax

EXACT = True
RIGHT = False


def _ll(dist, data):
    return np.asarray(survjax.log_likelihood(dist, data))


def _data(time, event):
    return survjax.SurvivalData(f(time), b(event))


def _bad_params(dist, value):
    """``dist.parameter_issues().invalid_entries`` broadcast to the contribution shape (7.1)."""
    return np.broadcast_to(
        np.asarray(dist.parameter_issues().invalid_entries).astype(bool), np.shape(value)
    )


# ------------------------------------------------- declared violations force NaN


@pytest.mark.step("A6")
def test_declared_violation_is_nan_where_formula_is_finite():
    """spec 7.1, rev 29: rows marked by `parameter_issues().invalid_entries` are NaN "even where the family's formula is finite".

    Right-censored at ``t=2``: ``rate=-1`` gives the finite ``-r t = 2``, but
    ``rate`` is declared ``positive``.
    """
    dist = survjax.Exponential(rate=f([0.5, -1.0, 2.0]))
    value = _ll(dist, _data([2.0, 2.0, 2.0], [RIGHT, RIGHT, RIGHT]))
    assert np.isnan(value[1])
    assert_close(value[[0, 2]], [-1.0, -4.0])


@pytest.mark.step("A6")
def test_zero_rate_right_censored_row_is_nan():
    """spec 4.1 and 7.1, rev 28: `rate=0` is outside `positive`, so `log S = 0` is replaced by NaN."""
    dist = survjax.Exponential(rate=f([0.0, 1.0]))
    value = _ll(dist, _data([2.0, 2.0], [RIGHT, RIGHT]))
    assert np.isnan(value[0])
    assert_close(value[1], -2.0)


@pytest.mark.step("A6")
def test_infinite_weibull_scale_right_censored_row_is_nan():
    """spec 4.1 and 7.1, rev 28: `scale=inf` violates `positive`; the formula's `-(t/inf)**k = 0` is replaced by NaN."""
    dist = survjax.Weibull(shape=f([2.0, 2.0]), scale=f([INF, 1.0]))
    value = _ll(dist, _data([2.0, 2.0], [RIGHT, RIGHT]))
    assert np.isnan(value[0])
    assert_close(value[1], -4.0)  # -(2/1)**2


@pytest.mark.step("A6")
def test_negative_weibull_shape_right_censored_row_is_nan():
    """spec 4.1 and 7.1, rev 28: `shape=-1` violates `positive`; the formula's `-(2/1)**-1 = -0.5` is replaced by NaN."""
    dist = survjax.Weibull(shape=f([-1.0, 1.0]), scale=f([1.0, 1.0]))
    value = _ll(dist, _data([2.0, 2.0], [RIGHT, RIGHT]))
    assert np.isnan(value[0])
    assert_close(value[1], -2.0)


@pytest.mark.step("A6")
def test_from_log_overflow_row_is_nan():
    """spec 4.1 and 7.1, rev 29: `from_log` overflow (`rate=exp(1000)=inf`) is a declared violation, so NaN rather than `-inf`."""
    dist = survjax.Exponential.from_log(log_rate=f([1000.0, 0.0]))
    value = _ll(dist, _data([2.0, 2.0], [RIGHT, RIGHT]))
    assert np.isnan(value[0])
    assert_close(value[1], -2.0)
    np.testing.assert_array_equal(_bad_params(dist, value), [True, False])


@pytest.mark.step("A6")
def test_condition_violation_is_nan_on_exact_row():
    """spec 4.7 and 7.1, rev 29: a violated named condition marks `invalid_entries` and forces NaN on a finite exact row.

    ``capped_rate`` with ``rate=3 > cap=2``: ``log f(1) = log 3 - 3`` is finite.
    """
    dist = fam.capped_rate()(rate=f([3.0, 1.0]), cap=f([2.0, 2.0]))
    value = _ll(dist, _data([1.0, 1.0], [EXACT, EXACT]))
    assert np.isnan(value[0])
    assert_close(value[1], np.log(1.0) - 1.0)
    np.testing.assert_array_equal(_bad_params(dist, value), [True, False])


@pytest.mark.step("A6")
def test_nested_violation_is_nan():
    """spec 4.7 and 7.1, rev 29: `invalid_entries` includes nested masks, so a nested violation forces NaN.

    ``scaled_capped`` with ``log_ratio=0`` reproduces the baseline:
    right-censored ``log S(2) = -r * 2``.
    """
    base = fam.capped_rate()(rate=f([1.0, 1.0]), cap=f([2.0, NAN]))
    dist = fam.scaled_capped()(base=base, log_ratio=f([0.0, 0.0]))
    value = _ll(dist, _data([2.0, 2.0], [RIGHT, RIGHT]))
    assert_close(value[0], -2.0)
    assert np.isnan(value[1])
    np.testing.assert_array_equal(_bad_params(dist, value), [False, True])


# ------------------------------------ NaN rows are exactly the invalid_entries


@pytest.mark.step("A6")
def test_nan_rows_are_parameter_issues_invalid_entries():
    """spec 7.1, rev 29: with valid data, the NaN rows are `dist.parameter_issues().invalid_entries` (paired shapes)."""
    dist = survjax.Exponential(rate=f([0.5, -1.0, 2.0, NAN, 1.0]))
    value = _ll(dist, _data([1.0, 2.0, 1.0, 1.0, 3.0], [EXACT, RIGHT, EXACT, RIGHT, EXACT]))
    expected = np.asarray(dist.parameter_issues().invalid_entries)
    np.testing.assert_array_equal(expected, [False, True, False, True, False])
    np.testing.assert_array_equal(np.isnan(value), expected)


@pytest.mark.step("A6")
def test_scalar_invalid_distribution_marks_every_row():
    """spec 5 and 7.1, rev 29: a scalar invalid distribution's mask, broadcast to data `(N,)`, makes every row NaN."""
    dist = survjax.Exponential(rate=f(-1.0))
    value = _ll(dist, _data([1.0, 2.0, 3.0, 4.0], [EXACT, RIGHT, EXACT, RIGHT]))
    assert value.shape == (4,)
    assert np.isnan(value).all()
    np.testing.assert_array_equal(_bad_params(dist, value), [True] * 4)


@pytest.mark.step("A6")
def test_invalid_entries_broadcast_over_draws():
    """spec 5 and 7.1, rev 29: distribution `(D,1)` against data `(N,)` gives `(D,N)`; `invalid_entries` broadcast to it selects the NaN rows."""
    dist = survjax.Exponential(rate=f([[1.0], [-1.0]]))
    value = _ll(dist, _data([1.0, 2.0, 3.0], [EXACT, RIGHT, RIGHT]))
    assert value.shape == (2, 3)
    mask = _bad_params(dist, value)
    np.testing.assert_array_equal(mask, [[False] * 3, [True] * 3])
    np.testing.assert_array_equal(np.isnan(value), mask)
    assert_close(value[0], [np.log(1.0) - 1.0, -2.0, -3.0])


@pytest.mark.step("A6")
def test_valid_parameters_give_no_nan_rows():
    """spec 7.1, rev 29: valid declared parameters report no `invalid_entries` and give finite values."""
    dist = survjax.Weibull(shape=f([0.5, 1.0, 2.0]), scale=f([1.0, 2.0, 3.0]))
    value = _ll(dist, _data([1.0, 2.0, 3.0], [EXACT, RIGHT, EXACT]))
    assert np.isfinite(value).all()
    assert not _bad_params(dist, value).any()


@pytest.mark.step("A6")
def test_invalid_parameters_and_invalid_data_overlap():
    """spec 7.1, rev 29: a row with invalid data and invalid parameters is NaN; the recipe attributes it to both."""
    dist = survjax.Exponential(rate=f([-1.0, 1.0]))
    data = _data([0.0, 1.0], [EXACT, EXACT])  # exact t=0 is invalid data
    value = _ll(dist, data)
    assert np.isnan(value[0])
    assert_close(value[1], -1.0)
    bad_data, bad_params, other = nan_causes(dist, data, value)
    np.testing.assert_array_equal(bad_data, [True, False])
    np.testing.assert_array_equal(bad_params, [True, False])
    assert not other.any()


# ------------------------------------------------ method and function agree


@pytest.mark.step("A6")
def test_method_and_function_agree_on_parameter_violations():
    """spec 7.1 and 12.1, rev 29: `dist.log_likelihood` returns exactly what the function returns, NaN rows included."""
    dist = fam.capped_rate()(rate=f([3.0, 1.0, -1.0, 0.5]), cap=f([2.0, 2.0, 2.0, 2.0]))
    data = _data([1.0, 1.0, 2.0, 2.0], [EXACT, EXACT, RIGHT, RIGHT])
    plain = _ll(dist, data)
    method = ll_method(dist, data)
    np.testing.assert_array_equal(method, plain)
    np.testing.assert_array_equal(np.isnan(plain), [True, False, True, False])


@pytest.mark.step("A6")
def test_raw_form_forces_nan_on_parameter_violations():
    """spec 6.1 and 7.1, rev 29: the raw `time=`/`event=` form forces NaN too, in the function and the method."""
    dist = survjax.Exponential(rate=f([-1.0, 1.0]))
    time, event = f([2.0, 2.0]), b([RIGHT, RIGHT])
    value = np.asarray(survjax.log_likelihood(dist, time=time, event=event))
    assert np.isnan(value[0])
    assert_close(value[1], -2.0)
    np.testing.assert_array_equal(_bad_params(dist, value), [True, False])
    np.testing.assert_array_equal(ll_method(dist, time=time, event=event), value)


# ------------------------------------------ undeclared NaN: the recipe's remainder


@pytest.mark.step("A6")
def test_undeclared_nan_is_the_recipe_remainder():
    """spec 7.1, rev 29: a NaN row with no data or parameter issue is the recipe's `other` ("`Z=0` or numerical failure").

    ``const_hazard`` declares nothing; ``rate=NaN`` makes the exact row NaN.
    """
    dist = fam.const_hazard()(rate=f([1.0, NAN]))
    data = _data([1.0, 1.0], [EXACT, EXACT])
    value = _ll(dist, data)
    assert_close(value[0], -1.0)
    assert np.isnan(value[1])
    bad_data, bad_params, other = nan_causes(dist, data, value)
    assert not bad_data.any()
    assert not bad_params.any()
    np.testing.assert_array_equal(other, [False, True])


@pytest.mark.step("A6")
def test_undeclared_negative_rate_log_is_the_recipe_remainder():
    """spec 7.1, rev 29: an undeclared `rate=-1` gives `log f = log(-1) + 1 = NaN`, unexplained by either diagnostic."""
    dist = fam.unconstrained_rate()(rate=f([-1.0, 1.0]))
    data = _data([1.0, 1.0], [EXACT, EXACT])
    value = _ll(dist, data)
    assert np.isnan(value[0])
    _, bad_params, other = nan_causes(dist, data, value)
    assert not bad_params.any()
    np.testing.assert_array_equal(other, [True, False])


@pytest.mark.step("A6")
def test_declared_twin_of_finite_value_is_nan():
    """spec 7.1, rev 29: a custom family's declared `positive` forces NaN where its formula `-r t = 2` is finite."""
    dist = fam.constrained_rate()(rate=f([-1.0]))
    value = _ll(dist, _data([2.0], [RIGHT]))
    assert np.isnan(value).all()
    np.testing.assert_array_equal(_bad_params(dist, value), [True])


# ------------------------------------ other rows: values and gradients unchanged


ROWS_OK = [0, 2]


def _per_row_case():
    """Exact, exact, right rows; row 1 has ``rate=0``: ``log 0 = -inf`` and ``d/dr log r = 1/0``."""
    rate = f([0.5, 0.0, 2.0])
    data = _data([1.0, 1.0, 3.0], [EXACT, EXACT, RIGHT])
    return rate, data


def _expected_values():
    # row 0: log 0.5 - 0.5 * 1; row 2: -2 * 3
    return np.array([np.log(0.5) - 0.5, -6.0])


def _expected_jacobian():
    # d ll_i / d r_j: diagonal 1/r - t (exact) or -t (right); off-diagonal 0
    jac = np.zeros((2, 3))
    jac[0, 0] = 1.0 / 0.5 - 1.0
    jac[1, 2] = -3.0
    return jac


@pytest.mark.step("A6")
def test_rows_without_issues_keep_their_values():
    """spec 7.1, rev 28: "rows without issues give the same values ... as when evaluated alone"."""
    rate, data = _per_row_case()
    together = _ll(survjax.Exponential(rate=rate), data)
    alone = _ll(survjax.Exponential(rate=rate[::2]), data[::2])  # rows 0 and 2
    assert np.isnan(together[1])
    assert_close(together[ROWS_OK], alone)
    assert_close(together[ROWS_OK], _expected_values())


@pytest.mark.step("A6")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_rows_without_issues_keep_their_gradients(mode):
    """spec 7.1, rev 32: rows without issues give "the same first ... parameter derivatives as when evaluated alone" (per-row parameters).

    The Jacobian rows of the valid contributions are finite, equal the
    hand-derived diagonal, and carry exactly zero with respect to the invalid
    row's ``rate=0`` (no NaN or infinity leaks from ``1/0``).
    """
    rate, data = _per_row_case()

    def ll(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), data)

    jac_fn = jax.jacrev(ll) if mode == "rev" else jax.jacfwd(ll)
    jac = np.asarray(jac_fn(jnp.asarray(rate)))[ROWS_OK]
    assert np.isfinite(jac).all()
    assert_close(jac, _expected_jacobian())


@pytest.mark.step("A6")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_rows_without_issues_keep_their_gradients_under_jit(mode):
    """spec 7.1 and 7.3, rev 32: the derivative guarantee for rows without issues holds under `jit`."""
    rate, data = _per_row_case()

    def ll(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), data)

    jac_fn = jax.jacrev(ll) if mode == "rev" else jax.jacfwd(ll)
    jac = np.asarray(jax.jit(jac_fn)(jnp.asarray(rate)))[ROWS_OK]
    assert np.isfinite(jac).all()
    assert_close(jac, _expected_jacobian())


@pytest.mark.step("A6")
def test_sum_over_valid_rows_has_alone_gradient():
    """spec 7.1, rev 32: the gradient of the valid rows' sum equals that of the valid rows evaluated alone."""
    rate, data = _per_row_case()
    keep = jnp.asarray(ROWS_OK)

    def loss_together(r):
        return jnp.sum(survjax.log_likelihood(survjax.Exponential(rate=r), data)[keep])

    def loss_alone(r):
        sub = survjax.Exponential(rate=r[::2])  # rows 0 and 2
        return jnp.sum(survjax.log_likelihood(sub, data[::2]))

    g_together = np.asarray(jax.grad(loss_together)(jnp.asarray(rate)))[ROWS_OK]
    g_alone = np.asarray(jax.grad(loss_alone)(jnp.asarray(rate)))[ROWS_OK]
    assert np.isfinite(g_together).all()
    assert_close(g_together, g_alone)
    assert_close(g_together, [1.0 / 0.5 - 1.0, -3.0])


# ------------------------------------------------------------ transformations


@pytest.mark.step("A6")
def test_parameter_violations_under_jit_match_eager():
    """spec 6.3 and 7.1, rev 29: values and `invalid_entries` agree eagerly and under `jit`, with traced parameters."""
    data = _data([1.0, 2.0, 3.0], [EXACT, RIGHT, RIGHT])
    rate = f([1.0, -1.0, 0.0])

    def run(r):
        dist = survjax.Exponential(rate=r)
        return survjax.log_likelihood(dist, data), dist.parameter_issues().invalid_entries

    eager_value, eager_mask = run(rate)
    jit_value, jit_mask = jax.jit(run)(rate)
    assert_close(jit_value, eager_value)
    np.testing.assert_array_equal(np.asarray(jit_mask), np.asarray(eager_mask))
    np.testing.assert_array_equal(np.asarray(jit_mask), [False, True, True])
    np.testing.assert_array_equal(np.isnan(np.asarray(jit_value)), [False, True, True])


@pytest.mark.step("A6")
def test_parameter_violations_under_vmap_match_batched_call():
    """spec 5 and 7.1, rev 29: `vmap` over parameter draws equals the `(D,N)` batched call, NaN rows included."""
    data = _data([1.0, 2.0, 3.0], [EXACT, RIGHT, RIGHT])
    rates = f([[1.0, 2.0, 0.5], [1.0, -1.0, 0.5]])

    def run(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), data)

    mapped = np.asarray(jax.vmap(run)(rates))
    batched = np.asarray(run(rates))
    assert_close(mapped, batched)
    np.testing.assert_array_equal(np.isnan(batched), [[False] * 3, [False, True, False]])
