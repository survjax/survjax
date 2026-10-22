"""A6: NaN rows for observation modes Slice A does not implement (spec 7.1, 12.2; rev 31).

Slice A supports exact and right-censored rows without truncation.
"Unimplemented observation modes return NaN on those rows, eagerly and under
transformations, preserving supported rows in the same batch" (12.2); 7.1:
"Unsupported rows return NaN in both likelihood entry points, eagerly and
under `jit`/`vmap`, without a value-dependent exception", and such an
observation "does not add a data or parameter issue". The triage recipe
(``helpers.nan_causes``, ``helpers.unsupported_nan``) attributes those rows to
mode support, not to ``Z=0`` or numerical failure.

Row 0 of every case is a supported exact row at ``t=1``; row 1 is the
unsupported one. Exponential (spec 2, 4.1): ``log f(t) = log r - r t``, so with
``r = 0.5`` row 0 is ``log 0.5 - 0.5``; ``d/dr = 1/r - t = 1`` and
``d2/dr2 = -1/r**2 = -4``. Unsupported-row derivatives are unspecified and not
asserted.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import INF, assert_close, b, f, ll, ll_method, nan_causes, unsupported_nan

import survjax

RATE = 0.5
ROW0 = np.log(RATE) - RATE * 1.0
D_ROW0 = 1.0 / RATE - 1.0
D2_ROW0 = -1.0 / RATE**2


def _dist():
    return survjax.Exponential(rate=f(RATE))


def _left_censored():
    return survjax.SurvivalData(f([1.0, 3.0]), left_censored=b([False, True]))


def _interval_censored():
    return survjax.SurvivalData(
        f([1.0, 1.0]), interval_censored=b([False, True]), time2=f([1.0, 3.0])
    )


def _bounds_left():
    return survjax.SurvivalData.from_bounds(f([1.0, 0.0]), f([1.0, 3.0]), exact=b([True, False]))


def _bounds_interval():
    return survjax.SurvivalData.from_bounds(f([1.0, 1.0]), f([1.0, 3.0]), exact=b([True, False]))


def _left_truncated_exact():
    return survjax.SurvivalData(f([1.0, 2.0]), b([True, True]), left_trunc=f([0.0, 1.0]))


def _left_truncated_right():
    return survjax.SurvivalData(f([1.0, 2.0]), b([True, False]), left_trunc=f([0.0, 1.0]))


def _right_truncated_exact():
    return survjax.SurvivalData(f([1.0, 2.0]), b([True, True]), right_trunc=f([INF, 5.0]))


CASES = {
    "left-censored": _left_censored,
    "interval-censored": _interval_censored,
    "bounds-left": _bounds_left,
    "bounds-interval": _bounds_interval,
    "left-truncated-exact": _left_truncated_exact,
    "left-truncated-right": _left_truncated_right,
    "right-truncated-exact": _right_truncated_exact,
}


# ------------------------------------------------------------ input diagnostics


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_row_is_valid_data(case):
    """spec 3.2 and 7.1, rev 31: an unsupported mode "is not invalid data ... it does not add a data ... issue"."""
    data = CASES[case]()
    assert bool(np.asarray(data.issues.ok))
    assert not np.asarray(data.issues.invalid_rows).any()


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_row_adds_no_parameter_issue(case):
    """spec 7.1, rev 31: an unsupported mode "is not ... an invalid parameter; it does not add a ... parameter issue"."""
    data = CASES[case]()
    dist = survjax.Exponential(rate=f([RATE, RATE]))
    value = ll(dist, data)
    assert np.isnan(value[1])
    assert not np.asarray(dist.parameter_issues().invalid_entries).any()


# -------------------------------------------------- eager values, both entry points


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_row_is_nan(case):
    """spec 7.1 and 12.2, rev 31: `survjax.log_likelihood` returns NaN on the unsupported row, without an exception."""
    value = ll(_dist(), CASES[case]())
    assert value.shape == (2,)
    assert np.isnan(value[1]), f"unsupported row produced {value[1]!r}"


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_row_is_nan_in_method(case):
    """spec 7.1 and 12.2, rev 31: `dist.log_likelihood` also returns NaN on the unsupported row."""
    value = ll_method(_dist(), CASES[case]())
    assert value.shape == (2,)
    assert np.isnan(value[1]), f"unsupported row produced {value[1]!r}"


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_supported_row_keeps_its_value(case):
    """spec 7.1 and 12.2, rev 31: "preserving supported rows in the same batch": row 0 is `log 0.5 - 0.5`."""
    value = ll(_dist(), CASES[case]())
    assert_close(value[0], ROW0)


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_supported_row_keeps_its_value_in_method(case):
    """spec 7.1, rev 31: the method preserves supported rows and equals the function."""
    data = CASES[case]()
    value = ll_method(_dist(), data)
    assert_close(value[0], ROW0)
    np.testing.assert_array_equal(value, ll(_dist(), data))


# ------------------------------------------------------------- transformations


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_row_is_nan_under_jit(case):
    """spec 7.1, rev 31: prepared data passed into `jit` gives the eager result, NaN on the unsupported row."""
    data = CASES[case]()

    def run(d):
        return survjax.log_likelihood(_dist(), d), _dist().log_likelihood(d)

    plain, method = (np.asarray(v) for v in jax.jit(run)(data))
    for value in (plain, method):
        assert np.isnan(value[1])
        assert_close(value[0], ROW0)


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_mode_from_traced_construction_is_nan(case):
    """spec 3.2, 6.1 and 7.1, rev 31: with traced construction the unsupported row is NaN and row 0 keeps its value.

    The case's normalized fields are passed into ``jit`` and rebuilt with the
    bounds form (exact rows have ``upper == lower``), so construction sees
    traced inputs.
    """
    data = CASES[case]()

    def build_and_evaluate(lower, upper, exact, lt, rt):
        traced = survjax.SurvivalData.from_bounds(
            lower, upper, exact=exact, left_trunc=lt, right_trunc=rt
        )
        return survjax.log_likelihood(_dist(), traced)

    value = np.asarray(
        jax.jit(build_and_evaluate)(
            data.lower, data.upper, data.exact, data.left_trunc, data.right_trunc
        )
    )
    assert np.isnan(value[1])
    assert_close(value[0], ROW0)


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_unsupported_row_is_nan_under_vmap(case):
    """spec 7.1, rev 31: `vmap` over parameter draws: every draw has NaN on row 1 and `log r - r` on row 0."""
    data = CASES[case]()
    rates = f([0.5, 2.0])

    def run(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), data)

    value = np.asarray(jax.vmap(run)(rates))
    assert value.shape == (2, 2)
    assert np.isnan(value[:, 1]).all()
    assert_close(value[:, 0], [np.log(0.5) - 0.5, np.log(2.0) - 2.0])


# ------------------------------------------------------------------ triage recipe


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_triage_recipe_attributes_row_to_mode_support(case):
    """spec 7.1, rev 31: the Slice A recipe puts the row in `unsupported_nan`, not in `other` ("`Z=0` or numerical failure")."""
    data = CASES[case]()
    dist = _dist()
    value = ll(dist, data)
    bad_data, bad_params, other = nan_causes(dist, data, value)
    assert not bad_data.any()
    assert not bad_params.any()
    assert not other.any()
    np.testing.assert_array_equal(unsupported_nan(dist, data, value), [False, True])


# ------------------------------------------------- supported-row derivatives


def _per_row_jacobian_case(case, jac):
    data = CASES[case]()

    def row0(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), data)[0]

    return np.asarray(jac(row0)(jnp.asarray(f([RATE, RATE]))))


@pytest.mark.step("A6")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("case", list(CASES))
def test_supported_row_keeps_its_gradient_per_row(case, mode):
    """spec 7.1 and 7.3, rev 31: supported rows retain their parameter derivatives: `d ll_0 / d r = [1/r - t, 0]`."""
    jac = jax.jacrev if mode == "rev" else jax.jacfwd
    g = _per_row_jacobian_case(case, jac)
    assert np.isfinite(g).all()
    assert_close(g, [D_ROW0, 0.0])


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_supported_row_keeps_its_hessian_per_row(case):
    """spec 7.1 and 7.3, rev 31: second derivatives of row 0: `[[-1/r**2, 0], [0, 0]]` (fwd-over-rev and rev-over-rev)."""
    for hess in (jax.hessian, lambda fn: jax.jacrev(jax.jacrev(fn))):
        h = _per_row_jacobian_case(case, hess)
        assert np.isfinite(h).all()
        assert_close(h, [[D2_ROW0, 0.0], [0.0, 0.0]])


@pytest.mark.step("A6")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
@pytest.mark.parametrize("case", list(CASES))
def test_supported_row_keeps_its_gradient_with_shared_rate(case, mode):
    """spec 7.1 and 7.3, rev 31: with one shared `rate`, `d ll_0 / d r = 1/r - t` and `d2 = -1/r**2`, finite.

    The unsupported row's NaN must not reach the shared parameter's
    derivatives of the supported row.
    """
    data = CASES[case]()

    def row0(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), data)[0]

    grad = jax.grad(row0) if mode == "rev" else jax.jacfwd(row0)
    hess = jax.grad(jax.grad(row0)) if mode == "rev" else jax.jacfwd(jax.jacfwd(row0))
    r = jnp.asarray(f(RATE))
    assert_close(grad(r), D_ROW0)
    assert_close(hess(r), D2_ROW0)


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_supported_row_keeps_its_gradient_under_jit(case):
    """spec 7.1 and 7.3, rev 31: the supported-row gradient holds under `jit`."""
    data = CASES[case]()

    def row0(r):
        return survjax.log_likelihood(survjax.Exponential(rate=r), data)[0]

    g = np.asarray(jax.jit(jax.jacrev(row0))(jnp.asarray(f([RATE, RATE]))))
    assert np.isfinite(g).all()
    assert_close(g, [D_ROW0, 0.0])
