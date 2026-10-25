"""A6: rows without parameter issues keep values and first and second derivatives (spec 7.1, 7.3, 12.1; rev 32).

7.1: "With per-row parameters, rows without issues give the same values and the
same first and second parameter derivatives as when evaluated alone. This holds
for every declared violation, including NaN, infinite, zero and negative
parameters and named-condition violations, and for every registered family."
The mechanism covers "the leaves of nested distributions. Each leaf is first
broadcast to the distribution's current `batch_shape`", so a shared (scalar)
leaf of a distribution with per-row batch entries is covered too. 7.3 puts the
isolation inside the derivative guarantee "to second order, in forward and
reverse modes"; 12.1 asks for zero, negative, NaN and `±inf` violations and a
named-condition violation in a built-in, a custom and a nested custom family.
The custom families in ``spec_families`` do not guard their parameter operands
("declaring the constraint or condition is sufficient").

Every case has three rows; row 1 has the declared violation and rows 0 and 2
(``OK``) have none. "Evaluated alone" is the same family built from the per-row
leaves at ``OK`` (shared leaves unchanged) on the data rows ``OK``. Both
evaluations are written as functions of the same flat parameter vector, so the
alone derivative is exactly zero in the violating row's entries: evaluated
alone, the valid rows do not depend on them. Gradients of the violating row
itself are unspecified and never formed. Hand-derived anchors at the end tie
the identity to known values.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, assert_close, b, f
from jax.flatten_util import ravel_pytree

import survjax

EXACT = True
RIGHT = False
OK = [0, 2]
BAD = 1

VIOLATIONS = {"zero": 0.0, "negative": -1.0, "nan": NAN, "posinf": INF, "neginf": -INF}
NON_FINITE = {"nan": NAN, "posinf": INF, "neginf": -INF}


@dataclasses.dataclass(frozen=True)
class Case:
    build: object  # dict of parameter arrays -> distribution
    params: dict  # name -> values; per-row leaves have leading axis 3
    per_row: tuple  # names sliced to OK for "alone"
    time: tuple
    event: tuple


def _exponential(p):
    return survjax.Exponential(rate=p["rate"])


def _weibull(p):
    return survjax.Weibull(shape=p["shape"], scale=p["scale"])


def _constrained_rate(p):
    return fam.constrained_rate()(rate=p["rate"])


def _capped(p):
    return fam.capped_rate()(rate=p["rate"], cap=p["cap"])


def _component(p):
    return fam.component_constrained()(rate=p["rate"], free=p["free"], w=p["w"], knots=p["knots"])


def _nested(p):
    base = fam.capped_rate()(rate=p["rate"], cap=p["cap"])
    return fam.scaled_capped()(base=base, log_ratio=p["log_ratio"])


TIME = (1.0, 2.0, 3.0)
EVENT = (EXACT, RIGHT, EXACT)


def _cases():
    cases = {}
    for name, v in VIOLATIONS.items():
        cases[f"exponential-rate-{name}"] = Case(
            _exponential, {"rate": [0.5, v, 2.0]}, ("rate",), TIME, EVENT
        )
        cases[f"weibull-scale-{name}-shared-shape"] = Case(
            _weibull, {"shape": 1.5, "scale": [1.0, v, 2.0]}, ("scale",), TIME, EVENT
        )
        cases[f"weibull-shape-{name}-shared-scale"] = Case(
            _weibull, {"shape": [2.0, v, 0.5], "scale": 1.5}, ("shape",), TIME, EVENT
        )
        cases[f"custom-rate-{name}"] = Case(
            _constrained_rate, {"rate": [0.5, v, 2.0]}, ("rate",), TIME, EVENT
        )
        cases[f"nested-base-rate-{name}"] = Case(
            _nested,
            {"rate": [0.5, v, 2.0], "cap": 4.0, "log_ratio": 0.3},
            ("rate",),
            TIME,
            EVENT,
        )
        cases[f"component-rate-{name}-shared-w"] = Case(
            _component,
            {"rate": [0.5, v, 2.0], "free": 0.0, "w": [0.5, 1.5], "knots": [0.0, 1.0]},
            ("rate",),
            TIME,
            EVENT,
        )
        cases[f"component-w-{name}-per-row"] = Case(
            _component,
            {
                "rate": 1.0,
                "free": 0.0,
                "w": [[0.5, 1.5], [v, 1.0], [1.0, 0.25]],
                "knots": [0.0, 1.0],
            },
            ("w",),
            TIME,
            EVENT,
        )
    for name, v in NON_FINITE.items():
        cases[f"nested-log-ratio-{name}"] = Case(
            _nested,
            {"rate": 1.0, "cap": 4.0, "log_ratio": [0.1, v, 0.2]},
            ("log_ratio",),
            TIME,
            EVENT,
        )
    # Named conditions: rate > cap is finite in the formula, flagged only by the declaration.
    cases["custom-condition"] = Case(
        _capped, {"rate": [1.0, 3.0, 0.5], "cap": 2.0}, ("rate",), TIME, EVENT
    )
    cases["nested-base-condition"] = Case(
        _nested,
        {"rate": [1.0, 3.0, 0.5], "cap": 2.0, "log_ratio": 0.3},
        ("rate",),
        TIME,
        EVENT,
    )
    cases["nested-outer-condition"] = Case(
        _nested,
        {"rate": 1.0, "cap": 2.0, "log_ratio": [0.1, 3.0, 0.2]},
        ("log_ratio",),
        TIME,
        EVENT,
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
    """Flat parameter vector; all rows, the together and the alone contributions of rows ``OK`` as functions of it."""
    x, unravel = ravel_pytree({k: jnp.asarray(f(v)) for k, v in case.params.items()})
    data = _data(case.time, case.event)
    data_ok = _data([case.time[i] for i in OK], [case.event[i] for i in OK])
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


def _assert_case_has_issue_on_bad_row(case):
    x, unravel = ravel_pytree({k: jnp.asarray(f(v)) for k, v in case.params.items()})
    dist = case.build(unravel(x))
    mask = np.asarray(dist.parameter_issues().invalid_entries).astype(bool)
    np.testing.assert_array_equal(np.broadcast_to(mask, (3,)), [False, True, False])


@pytest.mark.step("A6")
@pytest.mark.parametrize("case", list(CASES))
def test_rows_without_issues_keep_values(case):
    """spec 7.1, rev 32: "rows without issues give the same values ... as when evaluated alone"; row 1 is NaN."""
    c = CASES[case]
    _assert_case_has_issue_on_bad_row(c)
    x, full, together, alone = _setup(c)
    assert np.isnan(np.asarray(full(x))[BAD])
    value = np.asarray(together(x))
    assert np.isfinite(value).all()
    assert_close(value, alone(x))


@pytest.mark.step("A6")
@pytest.mark.parametrize("deriv", list(DERIVATIVES))
@pytest.mark.parametrize("case", list(CASES))
def test_rows_without_issues_keep_derivatives(case, deriv):
    """spec 7.1, 7.3 and 12.1, rev 32: "the same first and second parameter derivatives as when evaluated alone", forward and reverse.

    Covers every leaf, shared leaves included, and the violating row's
    entries (exactly zero alone).
    """
    x, _, together, alone = _setup(CASES[case])
    d = DERIVATIVES[deriv]
    got = np.asarray(d(together)(x))
    want = np.asarray(d(alone)(x))
    assert np.isfinite(got).all(), f"non-finite derivative leaked: {got!r}"
    assert_close(got, want)


@pytest.mark.step("A6")
@pytest.mark.parametrize("deriv", ["jacrev", "fwd-over-rev"])
@pytest.mark.parametrize("case", list(CASES))
def test_rows_without_issues_keep_derivatives_under_jit(case, deriv):
    """spec 7.1 and 7.3, rev 32: the isolation holds under `jit` (parameters traced)."""
    x, _, together, alone = _setup(CASES[case])
    d = DERIVATIVES[deriv]
    got = np.asarray(jax.jit(d(together))(x))
    want = np.asarray(d(alone)(x))
    assert np.isfinite(got).all()
    assert_close(got, want)


# ------------------------------------------------------------ hand-derived anchors


@pytest.mark.step("A6")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_exponential_isolated_hessian_is_hand_derived(mode):
    """spec 7.1 and 7.3, rev 32: per-row `rate = [0.5, NaN, 2]`, rows exact `t=1` and right `t=3`.

    ``d2/dr2 (log r - r t) = -1/r**2 = -4`` on row 0; ``d2/dr2 (-r t) = 0`` on
    row 2; every other entry is zero.
    """
    data = _data([1.0, 1.0, 3.0], [EXACT, EXACT, RIGHT])
    keep = jnp.asarray(OK)

    def ll_ok(r):
        return survjax.Exponential(rate=r).log_likelihood(data)[keep]

    outer = jax.jacrev if mode == "rev" else jax.jacfwd
    h = np.asarray(outer(outer(ll_ok))(jnp.asarray(f([0.5, NAN, 2.0]))))
    expected = np.zeros((2, 3, 3))
    expected[0, 0, 0] = -1.0 / 0.5**2
    assert np.isfinite(h).all()
    assert_close(h, expected)


@pytest.mark.step("A6")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_weibull_shared_shape_derivatives_are_hand_derived(mode):
    """spec 7.1 and 7.3, rev 32: shared `shape k=1.5`, `scale = [1, -1, 1]`; the `log(t/b)` of row 1 must not leak.

    With ``z = t/b``, ``u = log z``: exact ``log f = log k - log b + (k-1) u - z**k``,
    ``d/dk = 1/k + u - z**k u``, ``d2/dk2 = -1/k**2 - z**k u**2``; right
    ``log S = -z**k``, ``d/dk = -z**k u``, ``d2/dk2 = -z**k u**2``. Rows 0
    (exact) and 2 (right) at ``t = e``, ``b = 1`` (``u = 1``, ``z**k = e**k``)
    sum to ``d/dk = 1/k + 1 - 2 e**k`` and ``d2/dk2 = -1/k**2 - 2 e**k``.
    """
    e = np.e
    data = _data([e, 2.0, e], [EXACT, EXACT, RIGHT])
    scale = jnp.asarray(f([1.0, -1.0, 1.0]))
    keep = jnp.asarray(OK)

    def loss(k):
        dist = survjax.Weibull(shape=k, scale=scale)
        return jnp.sum(dist.log_likelihood(data)[keep])

    if mode == "rev":
        grad, hess = jax.grad(loss), jax.grad(jax.grad(loss))
    else:
        grad, hess = jax.jacfwd(loss), jax.jacfwd(jax.jacfwd(loss))
    k = 1.5
    kk = jnp.asarray(f(k))
    assert_close(grad(kk), 1.0 / k + 1.0 - 2.0 * np.exp(k))
    assert_close(hess(kk), -1.0 / k**2 - 2.0 * np.exp(k))


@pytest.mark.step("A6")
@pytest.mark.parametrize("mode", ["rev", "fwd"])
def test_nested_shared_log_ratio_derivatives_are_hand_derived(mode):
    """spec 7.1 and 7.3, rev 32: nested `scaled_capped`, base `rate = [1, NaN, 2]`, shared outer `log_ratio a = 0.3`.

    ``log S = -e**a r t`` and ``log h = a + log r``: exact row 0 (``r=1, t=2``)
    gives ``a + log r - e**a r t`` with ``d/da = 1 - 2 e**a``; right row 2
    (``r=2, t=1``) gives ``-e**a r t`` with ``d/da = -2 e**a``. Sum:
    ``d/da = 1 - 4 e**a``, ``d2/da2 = -4 e**a``. The base violation must not
    reach the shared outer leaf.
    """
    data = _data([2.0, 2.0, 1.0], [EXACT, EXACT, RIGHT])
    rate = jnp.asarray(f([1.0, NAN, 2.0]))
    keep = jnp.asarray(OK)

    def loss(a):
        base = fam.capped_rate()(rate=rate, cap=jnp.asarray(f(4.0)))
        dist = fam.scaled_capped()(base=base, log_ratio=a)
        return jnp.sum(dist.log_likelihood(data)[keep])

    if mode == "rev":
        grad, hess = jax.grad(loss), jax.grad(jax.grad(loss))
    else:
        grad, hess = jax.jacfwd(loss), jax.jacfwd(jax.jacfwd(loss))
    a = 0.3
    aa = jnp.asarray(f(a))
    assert_close(grad(aa), 1.0 - 4.0 * np.exp(a))
    assert_close(hess(aa), -4.0 * np.exp(a))
