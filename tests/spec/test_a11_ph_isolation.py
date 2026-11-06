"""A11: PH rows without parameter issues keep values and derivatives (spec 4.4, 4.7, 7.1, 7.3, 12.1; rev 41).

7.1: before evaluating family code the likelihood applies `stop_gradient` to
every parameter leaf at the entries marked by `invalid_entries` ("This covers
the leaves of nested distributions"); "rows without issues give the same
values and the same first and second parameter derivatives as when evaluated
alone ... for every registered family". 7.3 keeps "invalid-entry isolation
within the supported derivative regime"; all valid rows here are ordinary
range. The scheme follows ``test_a6_likelihood_gradient_isolation.py``: three
rows, row 1 violating, rows 0 and 2 compared with the same wrapper built from
the per-row leaves at rows 0 and 2 alone, as functions of one flat parameter
vector (so the alone derivative is exactly zero in row 1's entries).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, b, f, ph
from jax.flatten_util import ravel_pytree

import survjax

EXACT, RIGHT = True, False
OK = [0, 2]
BAD = 1
TIME = (1.0, 2.0, 3.0)
EVENT = (EXACT, RIGHT, EXACT)

BASELINE_VIOLATIONS = {"zero": 0.0, "negative": -1.0, "nan": NAN, "posinf": INF, "neginf": -INF}
PREDICTOR_VIOLATIONS = {"nan": NAN, "posinf": INF, "neginf": -INF}


@dataclasses.dataclass(frozen=True)
class Case:
    build: object
    params: dict
    per_row: tuple


def _ph_weibull(p):
    return ph(survjax.Weibull(shape=p["shape"], scale=p["scale"]), p["eta"])


def _ph_exponential(p):
    return ph(survjax.Exponential(rate=p["rate"]), p["eta"])


def _ph_capped(p):
    return ph(fam.capped_rate()(rate=p["rate"], cap=p["cap"]), p["eta"])


def _cases():
    cases = {}
    for name, v in PREDICTOR_VIOLATIONS.items():
        cases[f"weibull-eta-{name}-shared-baseline"] = Case(
            _ph_weibull, {"shape": 1.5, "scale": 2.0, "eta": [0.3, v, -0.4]}, ("eta",)
        )
        cases[f"exponential-eta-{name}-shared-baseline"] = Case(
            _ph_exponential, {"rate": 0.7, "eta": [0.3, v, -0.4]}, ("eta",)
        )
    for name, v in BASELINE_VIOLATIONS.items():
        cases[f"exponential-rate-{name}-shared-eta"] = Case(
            _ph_exponential, {"rate": [0.5, v, 2.0], "eta": 0.2}, ("rate",)
        )
        cases[f"weibull-scale-{name}-shared-shape-and-eta"] = Case(
            _ph_weibull, {"shape": 1.5, "scale": [1.0, v, 2.0], "eta": 0.2}, ("scale",)
        )
        cases[f"weibull-shape-{name}-per-row-eta"] = Case(
            _ph_weibull, {"shape": [2.0, v, 0.8], "scale": 1.5, "eta": [0.1, 0.5, -0.3]}, ("shape", "eta")
        )
    cases["custom-baseline-condition-shared-eta"] = Case(
        _ph_capped, {"rate": [1.0, 3.0, 0.5], "cap": 2.0, "eta": 0.2}, ("rate",)
    )
    cases["custom-baseline-nan-rate-per-row-eta"] = Case(
        _ph_capped, {"rate": [1.0, NAN, 0.5], "cap": 2.0, "eta": [0.1, 0.2, 0.3]}, ("rate", "eta")
    )
    return cases


CASES = _cases()

DERIVATIVES = {
    "jacrev": jax.jacrev,
    "jacfwd": jax.jacfwd,
    "fwd-over-rev": lambda fn: jax.jacfwd(jax.jacrev(fn)),
    "rev-over-rev": lambda fn: jax.jacrev(jax.jacrev(fn)),
    "fwd-over-fwd": lambda fn: jax.jacfwd(jax.jacfwd(fn)),
    "rev-over-fwd": lambda fn: jax.jacrev(jax.jacfwd(fn)),
}


def _data(time, event):
    return survjax.SurvivalData(f(list(time)), b(list(event)))


def _setup(case):
    x, unravel = ravel_pytree({k: jnp.asarray(f(v)) for k, v in case.params.items()})
    data = _data(TIME, EVENT)
    data_ok = _data([TIME[i] for i in OK], [EVENT[i] for i in OK])
    keep = jnp.asarray(OK)

    def full(flat):
        return case.build(unravel(flat)).log_likelihood(data)

    def together(flat):
        return full(flat)[keep]

    def alone(flat):
        p = unravel(flat)
        p = {k: (v[keep] if k in case.per_row else v) for k, v in p.items()}
        return case.build(p).log_likelihood(data_ok)

    return x, full, together, alone


@pytest.mark.step("A11")
@pytest.mark.parametrize("case", list(CASES))
def test_case_reports_the_issue_on_the_bad_row(case):
    """spec 4.4/4.7, rev 41: the violation is declared (`log_hazard_ratio`, a `baseline.` field or condition) on row 1 only."""
    c = CASES[case]
    x, unravel = ravel_pytree({k: jnp.asarray(f(v)) for k, v in c.params.items()})
    mask = np.asarray(c.build(unravel(x)).parameter_issues().invalid_entries).astype(bool)
    np.testing.assert_array_equal(np.broadcast_to(mask, (3,)), [False, True, False])


@pytest.mark.step("A11")
@pytest.mark.parametrize("case", list(CASES))
def test_rows_without_issues_keep_values(case):
    """spec 7.1, rev 41: rows without issues give the same values as alone; row 1 is NaN."""
    x, full, together, alone = _setup(CASES[case])
    assert np.isnan(np.asarray(full(x))[BAD])
    value = np.asarray(together(x))
    assert np.isfinite(value).all()
    assert_close(value, alone(x))


@pytest.mark.step("A11")
@pytest.mark.parametrize("deriv", list(DERIVATIVES))
@pytest.mark.parametrize("case", list(CASES))
def test_rows_without_issues_keep_derivatives(case, deriv):
    """spec 7.1/7.3/12.1, rev 41: the same first and second derivatives as alone, forward and reverse, nested baseline leaves included."""
    x, _, together, alone = _setup(CASES[case])
    d = DERIVATIVES[deriv]
    got = np.asarray(d(together)(x))
    assert np.isfinite(got).all(), f"non-finite derivative leaked: {got!r}"
    assert_close(got, d(alone)(x))


@pytest.mark.step("A11")
@pytest.mark.parametrize("deriv", ["jacrev", "fwd-over-rev"])
@pytest.mark.parametrize("case", list(CASES))
def test_rows_without_issues_keep_derivatives_under_jit(case, deriv):
    """spec 7.1/7.3, rev 41: the isolation holds under `jit`."""
    x, _, together, alone = _setup(CASES[case])
    d = DERIVATIVES[deriv]
    got = np.asarray(jax.jit(d(together))(x))
    assert np.isfinite(got).all()
    assert_close(got, d(alone)(x))


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_shared_predictor_derivatives_are_hand_derived(mode):
    """spec 7.1/7.3, rev 41: baseline `rate = [0.5, NaN, 2]`, shared `eta = a = 0.3`; rows 0 (exact `t=1`) and 2 (right `t=3`).

    Row 0: ``log r + a - r e**a t = log 0.5 + a - 0.5 e**a``; row 2:
    ``-r e**a t = -6 e**a``. Sum: ``d/da = 1 - 6.5 e**a``,
    ``d2/da2 = -6.5 e**a``. The baseline violation must not reach ``a``.
    """
    data = _data([1.0, 1.0, 3.0], [EXACT, EXACT, RIGHT])
    rate = jnp.asarray(f([0.5, NAN, 2.0]))
    keep = jnp.asarray(OK)

    def loss(a):
        return jnp.sum(ph(survjax.Exponential(rate=rate), a).log_likelihood(data)[keep])

    if mode == "rev":
        grad, hess = jax.grad(loss), jax.grad(jax.grad(loss))
    else:
        grad, hess = jax.jacfwd(loss), jax.jacfwd(jax.jacfwd(loss))
    a = 0.3
    assert_close(grad(f(a)), 1.0 - 6.5 * np.exp(a))
    assert_close(hess(f(a)), -6.5 * np.exp(a))


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_shared_baseline_derivatives_with_infinite_predictor_row(mode):
    """spec 7.1/7.3, rev 41: shared Weibull `(k, b0) = (1.5, 2)`, `eta = [0, inf, 0]`; rows 0 (exact `t=2`) and 2 (right `t=2`).

    With ``z = 1``, ``w = 1``: row 0 ``log f`` has ``d/db0 = (k/b0)(w-1) = 0``,
    row 2 ``log S`` has ``d/db0 = k w/b0 = 0.75``; sum ``0.75``. Row 1's
    ``exp(inf)`` must not leak into the shared ``b0``.
    """
    data = _data([2.0, 2.0, 2.0], [EXACT, EXACT, RIGHT])
    eta = jnp.asarray(f([0.0, INF, 0.0]))
    keep = jnp.asarray(OK)

    def loss(b0):
        return jnp.sum(ph(survjax.Weibull(shape=f(1.5), scale=b0), eta).log_likelihood(data)[keep])

    grad = jax.grad(loss) if mode == "rev" else jax.jacfwd(loss)
    assert_close(grad(f(2.0)), 0.75)
