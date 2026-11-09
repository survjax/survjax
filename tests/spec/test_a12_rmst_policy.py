"""A12: staged RMST value policy, isolation, shapes and surface (spec 4.2, 4.5, 5, 8.1, 8.7, 10.2, 12.1; rev 41).

8.7 value policy: "`tau` follows section 4.5: NaN for negative, NaN or
infinite `tau`; `0` at `tau=0`. The result is also NaN wherever
`parameter_issues()` reports a violation ... NaN entries do not give NaN
derivatives to other entries, including through shared parameters ... The
tangent of a NaN RMST entry itself is outside this staged guarantee ...
Isolation of the other valid entries remains required." Shapes: "`tau`
broadcasts against `batch_shape` ... There is no `rmst_grid` ... `dist.rmst(5.0)`
has shape `(S, 2)` and `dist.rmst(tau_grid[:, None, None])` has shape
`(m, S, 2)`." Surface: `rmst` only on Exponential, Weibull and PH wrappers
of them; every other `rmst`, and `mean`, `median`, `quantile` on every
family, raise `NotImplementedError`; no `options`, `rmst_with_info`,
`SummaryOptions`, `SummaryInfo`, `sample` or `inverse_cumulative_hazard`.

Section 8.8 (rev 51) lifts these surface restrictions method by method. Each
surface test keeps its step marker and carries `until(<feature>)` for the
feature that lifts it, so it stops running once that feature is enabled.

Isolation follows ``test_a11_ph_isolation.py``: entry 1 is NaN, entries 0
and 2 are compared with the same distribution evaluated on entries 0 and 2
alone, as functions of one flat parameter vector. Only first derivatives
are compared for Weibull families (shape higher orders are restricted);
the Exponential case also compares Hessians. Values use the 12.1 ``rtol``
without absolute floor; first derivatives the ordinary 12.1 tolerances.
"""

import dataclasses
import importlib
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import rmst_reference as ref
import spec_families as fam
from helpers import (
    INF,
    NAN,
    X64,
    assert_close,
    assert_log_parameter_derivative,
    assert_rel_close,
    f,
    ph,
    rounded,
)
from jax.flatten_util import ravel_pytree

import survjax

INVALID_TAU = {"negative": -1.0, "nan": NAN, "posinf": INF, "neginf": -INF}


def _families():
    return {
        "exponential": lambda: survjax.Exponential(rate=f(0.7)),
        "weibull": lambda: survjax.Weibull(shape=f(2.5), scale=f(2.0)),
        "ph-exponential": lambda: ph(survjax.Exponential(rate=f(0.7)), f(0.4)),
        "ph-weibull": lambda: ph(survjax.Weibull(shape=f(2.5), scale=f(2.0)), f(-0.3)),
    }


FAMILIES = list(_families())


# --------------------------------------------------------------- invalid cutoffs


@pytest.mark.step("A12")
@pytest.mark.parametrize("bad", list(INVALID_TAU))
@pytest.mark.parametrize("family", FAMILIES)
def test_invalid_cutoff_gives_nan(family, bad):
    """spec 4.5/8.7, rev 41: "NaN for negative, NaN or infinite `tau`", eagerly and under `jit`."""
    d = _families()[family]()
    tau = f(INVALID_TAU[bad])
    assert np.isnan(np.asarray(d.rmst(tau)))
    assert np.isnan(np.asarray(jax.jit(lambda dd, t: dd.rmst(t))(d, tau)))


@pytest.mark.step("A12")
@pytest.mark.parametrize("family", FAMILIES)
def test_invalid_cutoff_entries_leave_valid_entries(family):
    """spec 8.7, rev 41: a cutoff vector with invalid entries: those are NaN, the rest keep their values."""
    d = _families()[family]()
    taus = f([1.0, -1.0, NAN, INF, 3.0, 0.0])
    out = np.asarray(d.rmst(taus))
    assert np.isnan(out[1:4]).all()
    assert out[5] == 0.0
    assert_rel_close(out[[0, 4]], [np.asarray(d.rmst(f(1.0))), np.asarray(d.rmst(f(3.0)))])


# --------------------------------------------------------------- parameter issues


def _issue_cases():
    cases = {}
    for name, v in {"zero": 0.0, "negative": -1.0, "nan": NAN, "posinf": INF}.items():
        cases[f"exponential-rate-{name}"] = survjax.Exponential(rate=f([0.5, v, 2.0]))
        cases[f"weibull-shape-{name}"] = survjax.Weibull(shape=f([2.0, v, 0.8]), scale=f(1.5))
        cases[f"weibull-scale-{name}"] = survjax.Weibull(shape=f(1.5), scale=f([1.0, v, 2.0]))
        cases[f"ph-baseline-scale-{name}"] = ph(survjax.Weibull(shape=f(1.5), scale=f([1.0, v, 2.0])), f(0.2))
        cases[f"ph-exponential-rate-{name}"] = ph(survjax.Exponential(rate=f([0.5, v, 2.0])), f(0.2))
    for name, v in {"nan": NAN, "posinf": INF, "neginf": -INF}.items():
        cases[f"ph-weibull-eta-{name}"] = ph(survjax.Weibull(shape=f(1.5), scale=f(2.0)), f([0.3, v, -0.4]))
        cases[f"ph-exponential-eta-{name}"] = ph(survjax.Exponential(rate=f(0.7)), f([0.3, v, -0.4]))
    return cases


@pytest.mark.step("A12")
@pytest.mark.parametrize("case", list(_issue_cases()))
def test_parameter_issue_entries_are_nan(case):
    """spec 8.7, rev 41: "The result is also NaN wherever `parameter_issues()` reports a violation"; other entries finite."""
    d = _issue_cases()[case]
    np.testing.assert_array_equal(np.asarray(d.parameter_issues().invalid_entries), [False, True, False])
    out = np.asarray(d.rmst(f(2.0)))
    assert np.isnan(out[1])
    assert np.isfinite(out[[0, 2]]).all()
    for i in (0, 2):
        assert_rel_close(out[i], np.asarray(d[i].rmst(f(2.0))))


@pytest.mark.step("A12")
def test_parameter_issue_nan_under_jit():
    """spec 8.7, rev 41: the NaN signal is the same under `jit` (no value-dependent exception)."""
    d = survjax.Weibull(shape=f(1.5), scale=f([1.0, -1.0, 2.0]))
    out = np.asarray(jax.jit(lambda dd: dd.rmst(f(2.0)))(d))
    assert np.isnan(out[1])
    assert np.isfinite(out[[0, 2]]).all()


# --------------------------------------------------------------- isolation of valid entries


@dataclasses.dataclass(frozen=True)
class Case:
    build: object  # params -> distribution
    params: dict  # parameter name -> values; "tau" included
    per_row: tuple


def _weibull(p):
    return survjax.Weibull(shape=p["shape"], scale=p["scale"])


def _exponential(p):
    return survjax.Exponential(rate=p["rate"])


def _ph_weibull(p):
    return ph(survjax.Weibull(shape=p["shape"], scale=p["scale"]), p["eta"])


def _ph_exponential(p):
    return ph(survjax.Exponential(rate=p["rate"]), p["eta"])


def _isolation_cases():
    cases = {}
    for name, v in INVALID_TAU.items():
        cases[f"weibull-shared-parameters-tau-{name}"] = Case(
            _weibull, {"shape": 2.5, "scale": 2.0, "tau": [1.0, v, 3.0]}, ("tau",)
        )
        cases[f"exponential-shared-rate-tau-{name}"] = Case(_exponential, {"rate": 0.7, "tau": [1.0, v, 3.0]}, ("tau",))
        cases[f"ph-weibull-shared-parameters-tau-{name}"] = Case(
            _ph_weibull, {"shape": 1.5, "scale": 2.0, "eta": 0.3, "tau": [1.0, v, 3.0]}, ("tau",)
        )
        cases[f"ph-exponential-shared-parameters-tau-{name}"] = Case(
            _ph_exponential, {"rate": 0.7, "eta": 0.3, "tau": [1.0, v, 3.0]}, ("tau",)
        )
    for name, v in {"zero": 0.0, "negative": -1.0, "nan": NAN, "posinf": INF}.items():
        cases[f"weibull-scale-{name}-shared-shape"] = Case(
            _weibull, {"shape": 1.5, "scale": [1.0, v, 2.0], "tau": 2.0}, ("scale",)
        )
        cases[f"weibull-shape-{name}-shared-scale"] = Case(
            _weibull, {"shape": [2.0, v, 0.8], "scale": 1.5, "tau": 2.0}, ("shape",)
        )
        cases[f"exponential-rate-{name}-shared-tau"] = Case(_exponential, {"rate": [0.5, v, 2.0], "tau": 2.0}, ("rate",))
        cases[f"ph-baseline-scale-{name}-shared-eta"] = Case(
            _ph_weibull, {"shape": 1.5, "scale": [1.0, v, 2.0], "eta": 0.2, "tau": 2.0}, ("scale",)
        )
    for name, v in {"nan": NAN, "posinf": INF, "neginf": -INF}.items():
        cases[f"ph-weibull-eta-{name}-shared-baseline"] = Case(
            _ph_weibull, {"shape": 1.5, "scale": 2.0, "eta": [0.3, v, -0.4], "tau": 2.0}, ("eta",)
        )
        cases[f"ph-exponential-eta-{name}-shared-baseline"] = Case(
            _ph_exponential, {"rate": 0.7, "eta": [0.3, v, -0.4], "tau": 2.0}, ("eta",)
        )
    return cases


ISOLATION = _isolation_cases()
OK = [0, 2]


def _setup(case):
    x, unravel = ravel_pytree({k: jnp.asarray(f(v)) for k, v in case.params.items()})
    keep = jnp.asarray(OK)

    def full(flat):
        p = unravel(flat)
        return case.build(p).rmst(p["tau"])

    def together(flat):
        return full(flat)[keep]

    def alone(flat):
        p = unravel(flat)
        p = {k: (v[keep] if k in case.per_row else v) for k, v in p.items()}
        return case.build(p).rmst(p["tau"])

    return x, full, together, alone


@pytest.mark.step("A12")
@pytest.mark.parametrize("case", list(ISOLATION))
def test_valid_entries_keep_values_beside_nan_entries(case):
    """spec 8.7, rev 41: entry 1 is NaN; entries 0 and 2 equal their values alone."""
    x, full, together, alone = _setup(ISOLATION[case])
    assert np.isnan(np.asarray(full(x))[1])
    assert_rel_close(together(x), alone(x))


@pytest.mark.step("A12")
@pytest.mark.parametrize("mode", ["jacrev", "jacfwd"])
@pytest.mark.parametrize("case", list(ISOLATION))
def test_valid_entries_keep_first_derivatives_beside_nan_entries(case, mode):
    """spec 8.7, rev 41: "NaN entries do not give NaN derivatives to other entries, including through shared parameters", forward and reverse.

    The derivative of the valid entries in every parameter (shared ones
    included) and in the cutoffs equals the alone derivative, which is exactly
    zero in entry 1's own leaves.
    """
    x, _, together, alone = _setup(ISOLATION[case])
    d = getattr(jax, mode)
    got = np.asarray(d(together)(x))
    assert np.isfinite(got).all(), f"NaN leaked into valid entries: {got!r}"
    assert_close(got, d(alone)(x))


@pytest.mark.step("A12")
@pytest.mark.parametrize("case", [c for c in ISOLATION if c.startswith("ph-weibull")])
def test_valid_entries_keep_first_derivatives_under_jit(case):
    """spec 8.7, rev 41: the isolation holds under `jit`."""
    x, _, together, alone = _setup(ISOLATION[case])
    got = np.asarray(jax.jit(jax.jacrev(together))(x))
    assert np.isfinite(got).all()
    assert_close(got, jax.jacrev(alone)(x))


@pytest.mark.step("A12")
@pytest.mark.parametrize("bad", list(INVALID_TAU))
def test_exponential_shared_rate_hessian_is_isolated(bad):
    """spec 8.7, rev 41: Exponential supports all orders; the shared-rate Hessian of the valid entries ignores the invalid cutoff."""
    taus = f([1.0, INVALID_TAU[bad], 3.0])
    keep = jnp.asarray(OK)

    def total(r):
        return jnp.sum(survjax.Exponential(rate=r).rmst(taus)[keep])

    def alone(r):
        return jnp.sum(survjax.Exponential(rate=r).rmst(taus[keep]))

    for outer in (jax.jacrev, jax.jacfwd):
        got = np.asarray(outer(outer(total))(f(0.7)))
        assert np.isfinite(got)
        assert_close(got, outer(outer(alone))(f(0.7)))


@pytest.mark.step("A12")
def test_shared_scale_gradient_beside_invalid_cutoff_is_hand_checked():
    """spec 8.7/12.1, rev 44: Weibull `(k, b) = (2.5, 2)`, `tau = [1, -1, 3]`: `d/db` of entries 0 and 2 equals the quadrature reference."""
    kk, bb = rounded(2.5, 2.0)
    taus = f([1.0, -1.0, 3.0])
    keep = jnp.asarray(OK)
    got = jax.grad(lambda s: jnp.sum(survjax.Weibull(shape=f(kk), scale=s).rmst(taus)[keep]))(f(bb))
    want = sum(ref.weibull(kk, bb, t)[2] for t in rounded(1.0, 3.0))
    total = sum(ref.weibull(kk, bb, t)[0] for t in rounded(1.0, 3.0))
    assert np.isfinite(np.asarray(got))
    assert_log_parameter_derivative(got, want, bb, total)


# --------------------------------------------------------------- shapes


def _draws_by_arm():
    k = np.array([[0.8], [2.0], [1.0]])
    b0 = np.array([[1.0], [3.0], [2.0]])
    eta = np.array([[0.0, 0.6], [-0.5, 1.0], [0.3, -0.3]])
    return ph(survjax.Weibull(shape=f(k), scale=f(b0)), f(eta)), k, b0, eta


def _ph_reference(k, b0, eta, tau):
    return ref.ph_weibull(*rounded(k, b0, eta, tau))[0]


@pytest.mark.step("A12")
def test_scalar_cutoff_against_draws_by_arms():
    """spec 8.7, rev 41: "With a batch of `(S, 2)` (draws × arms), `dist.rmst(5.0)` has shape `(S, 2)`"."""
    d, k, b0, eta = _draws_by_arm()
    out = d.rmst(5.0)
    assert np.shape(out) == (3, 2)
    assert_rel_close(out, [[_ph_reference(k[s, 0], b0[s, 0], eta[s, j], 5.0) for j in range(2)] for s in range(3)])


@pytest.mark.step("A12")
def test_explicit_cutoff_grid_against_draws_by_arms():
    """spec 8.7, rev 41: "`dist.rmst(tau_grid[:, None, None])` has shape `(m, S, 2)`"; row `i` belongs to `tau_grid[i]`."""
    d, k, b0, eta = _draws_by_arm()
    tau_grid = np.array([0.5, 2.0, 10.0, 0.0])
    out = d.rmst(f(tau_grid)[:, None, None])
    assert np.shape(out) == (4, 3, 2)
    want = [
        [[_ph_reference(k[s, 0], b0[s, 0], eta[s, j], t) for j in range(2)] for s in range(3)] for t in tau_grid[:3]
    ]
    assert_rel_close(np.asarray(out)[:3], want)
    np.testing.assert_array_equal(np.asarray(out)[3], 0.0)


@pytest.mark.step("A12")
def test_cutoff_vector_pairs_with_the_trailing_arm_axis():
    """spec 5/8.7, rev 41: ordinary trailing-axis broadcasting: `tau` of shape `(2,)` pairs with the arms."""
    d, k, b0, eta = _draws_by_arm()
    taus = np.array([1.0, 4.0])
    out = d.rmst(f(taus))
    assert np.shape(out) == (3, 2)
    assert_rel_close(out, [[_ph_reference(k[s, 0], b0[s, 0], eta[s, j], taus[j]) for j in range(2)] for s in range(3)])


@pytest.mark.step("A12")
def test_cutoff_grid_for_native_weibull_batch():
    """spec 8.7, rev 41: explicit grid `tau_grid[:, None]` against Weibull batch `(N,)` gives `(m, N)`."""
    k = np.array([0.5, 2.5])
    d = survjax.Weibull(shape=f(k), scale=f(2.0))
    tau_grid = np.array([0.1, 1.0, 30.0])
    out = d.rmst(f(tau_grid)[:, None])
    assert np.shape(out) == (3, 2)
    assert_rel_close(out, [[ref.weibull(*rounded(ki, 2.0, t))[0] for ki in k] for t in tau_grid])


@pytest.mark.step("A12")
@pytest.mark.parametrize("family", FAMILIES)
def test_there_is_no_rmst_grid(family):
    """spec 8.7, rev 46: "There is no `rmst_grid`"; "not provided" means absent."""
    assert not hasattr(_families()[family](), "rmst_grid")


def _pairs():
    """Each first-release family with batch `(2,)`."""
    return {
        "exponential": survjax.Exponential(rate=f([0.7, 1.2])),
        "weibull": survjax.Weibull(shape=f([2.5, 0.8]), scale=f(2.0)),
        "ph-exponential": ph(survjax.Exponential(rate=f(0.7)), f([0.4, -0.2])),
        "ph-weibull": ph(survjax.Weibull(shape=f(2.5), scale=f(2.0)), f([-0.3, 0.5])),
    }


@pytest.mark.step("A12")
@pytest.mark.parametrize("family", FAMILIES)
def test_cutoff_that_does_not_broadcast_is_rejected(family):
    """spec 8.7, rev 46: "a `tau` that does not broadcast against `batch_shape` is static misuse and raises `TypeError`"."""
    with pytest.raises(TypeError):
        _pairs()[family].rmst(f([1.0, 2.0, 3.0]))


@pytest.mark.step("A12")
@pytest.mark.parametrize(
    "bad",
    [np.array(True), np.array(1.0 + 0.5j), np.array("1.0"), np.array([1.0], dtype=object)],
    ids=["bool", "complex", "string", "object"],
)
@pytest.mark.parametrize("family", FAMILIES)
def test_cutoff_with_a_non_numeric_dtype_is_rejected(family, bad):
    """spec 4.5/8.7, rev 46: "boolean, complex, string and object `tau` raise `TypeError`"."""
    with pytest.raises(TypeError):
        _families()[family]().rmst(bad)


@pytest.mark.step("A12")
@pytest.mark.parametrize("family", FAMILIES)
def test_boolean_cutoff_is_rejected_during_tracing(family):
    """spec 8.7, rev 46: the dtype check holds "also during tracing"."""
    d = _families()[family]()
    with pytest.raises(TypeError):
        jax.jit(lambda dd, t: dd.rmst(t))(d, jnp.asarray(True))


@pytest.mark.step("A12")
@pytest.mark.skipif(X64, reason="float64 conversion applies in 32-bit mode")
@pytest.mark.parametrize("family", FAMILIES)
def test_float64_cutoff_converts_without_warning_in_32_bit_mode(family):
    """spec 4.5/8.7, rev 46: "a float64 `tau` with 64-bit mode disabled converts to float32 without a warning"."""
    d = _families()[family]()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        out = d.rmst(np.array([1.0, 3.0], dtype=np.float64))
    relevant = [w for w in record if not issubclass(w.category, (DeprecationWarning, PendingDeprecationWarning))]
    assert relevant == []
    assert np.dtype(jnp.asarray(out).dtype) == np.dtype(np.float32)
    assert_rel_close(out, [np.asarray(d.rmst(f(1.0))), np.asarray(d.rmst(f(3.0)))])


# --------------------------------------------------------------- transformations


@pytest.mark.step("A12")
@pytest.mark.parametrize("family", FAMILIES)
def test_rmst_agrees_under_jit(family):
    """spec 12.1, rev 41: eager and `jit` agree."""
    d = _families()[family]()
    taus = f([0.1, 1.0, 7.0])
    assert_rel_close(jax.jit(lambda dd, t: dd.rmst(t))(d, taus), d.rmst(taus))


@pytest.mark.step("A12")
def test_rmst_agrees_under_vmap_over_distribution_and_cutoff():
    """spec 12.1, rev 41: `vmap` over paired entries and cutoffs equals the batched call."""
    d, *_ = _draws_by_arm()
    d = d[:, 0]
    taus = f([0.5, 2.0, 6.0])
    assert_rel_close(jax.vmap(lambda dd, t: dd.rmst(t))(d, taus), d.rmst(taus))


@pytest.mark.step("A12")
def test_rmst_gradient_agrees_under_vmap_and_jit():
    """spec 12.1, rev 41: `jit(vmap(grad))` over predictor draws equals per-draw gradients."""
    taus = f(3.0)

    def one(e):
        return ph(survjax.Weibull(shape=f(1.5), scale=f(2.0)), e).rmst(taus)

    etas = f([-0.5, 0.0, 0.8])
    got = jax.jit(jax.vmap(jax.grad(one)))(etas)
    want = np.array([np.asarray(jax.grad(one)(e)) for e in etas])
    assert_close(got, want)


# --------------------------------------------------------------- surface


@pytest.mark.step("A12")
@pytest.mark.until("summary.rmst")
def test_rmst_on_a_custom_family_raises():
    """spec 8.7/12.1, rev 41: "On every other family ... `rmst` raises `NotImplementedError` at call or trace time"."""
    d = fam.const_hazard()(rate=f(0.5))
    with pytest.raises(NotImplementedError):
        d.rmst(f(1.0))
    with pytest.raises(NotImplementedError):
        jax.jit(lambda dd: dd.rmst(f(1.0)))(d)


@pytest.mark.step("A12")
@pytest.mark.until("summary.rmst")
def test_rmst_check_depends_on_the_type_not_values():
    """spec 8.7, rev 41: "the check depends on the type, not on values"; NaN parameters and `tau=0` still raise."""
    d = fam.const_hazard()(rate=f(NAN))
    with pytest.raises(NotImplementedError):
        d.rmst(f(0.0))


@pytest.mark.step("A12")
@pytest.mark.until("summary.rmst")
def test_rmst_on_a_ph_wrapper_of_a_custom_baseline_raises():
    """spec 4.4/8.7/12.1, rev 41: "`rmst` on a PH wrapper of a custom baseline raises `NotImplementedError`", also under `jit`."""
    d = ph(fam.power_hazard()(shape=f(1.5), scale=f(2.0)), f(0.3))
    with pytest.raises(NotImplementedError):
        d.rmst(f(1.0))
    with pytest.raises(NotImplementedError):
        jax.jit(lambda dd: dd.rmst(f(1.0)))(d)


def _every_family():
    out = _families()
    out["custom"] = lambda: fam.const_hazard()(rate=f(0.5))
    out["ph-custom"] = lambda: ph(fam.const_hazard()(rate=f(0.5)), f(0.1))
    return out


def _raises_eagerly_and_under_jit(d, call):
    with pytest.raises(NotImplementedError):
        call(d)
    with pytest.raises(NotImplementedError):
        jax.jit(call)(d)


@pytest.mark.step("A12")
@pytest.mark.until("summary.mean")
@pytest.mark.parametrize("family", list(_every_family()))
def test_mean_raises_on_every_family(family):
    """spec 8.7/12.1, rev 41: "`mean` ... raise[s] `NotImplementedError` on every family", eager and under `jit`; lifted by `summary.mean` (8.8, rev 51)."""
    _raises_eagerly_and_under_jit(_every_family()[family](), lambda d: d.mean())


@pytest.mark.step("A12")
@pytest.mark.until("summary.quantile")
@pytest.mark.parametrize("family", list(_every_family()))
def test_median_raises_on_every_family(family):
    """spec 8.7/12.1, rev 41: "`median` ... raise[s] `NotImplementedError` on every family", eager and under `jit`; lifted by `summary.quantile` (8.8, rev 51)."""
    _raises_eagerly_and_under_jit(_every_family()[family](), lambda d: d.median())


@pytest.mark.step("A12")
@pytest.mark.until("summary.quantile")
@pytest.mark.parametrize("family", list(_every_family()))
def test_quantile_raises_on_every_family(family):
    """spec 8.7/12.1, rev 41: "`quantile` ... raise[s] `NotImplementedError` on every family", eager and under `jit`; lifted by `summary.quantile` (8.8, rev 51)."""
    _raises_eagerly_and_under_jit(_every_family()[family](), lambda d: d.quantile(f(0.5)))


@pytest.mark.step("A12")
@pytest.mark.until("summary.rmst")
@pytest.mark.parametrize("family", FAMILIES)
def test_rmst_takes_no_options_argument(family):
    """spec 8.7, rev 41: "It has no `options` argument"."""
    with pytest.raises(TypeError):
        _families()[family]().rmst(f(1.0), options=None)


@pytest.mark.step("A12")
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize(
    "name",
    [
        pytest.param("rmst_with_info", marks=pytest.mark.until("summary.rmst")),
        pytest.param("inverse_cumulative_hazard", marks=pytest.mark.until("summary.quantile")),
        pytest.param("sample", marks=pytest.mark.until("sampling")),
    ],
)
def test_absent_first_release_methods(family, name):
    """spec 8.7, rev 46: "'Not provided' means absent: accessing such a method on a distribution raises `AttributeError`"; each case runs until the feature that provides the method (8.8, rev 51)."""
    with pytest.raises(AttributeError):
        getattr(_families()[family](), name)


@pytest.mark.step("A12")
@pytest.mark.until("summary.containers")
@pytest.mark.parametrize("name", ["SummaryOptions", "SummaryInfo"])
def test_summary_configuration_types_are_not_provided(name):
    """spec 8.7, rev 46: "`SummaryOptions` and `SummaryInfo` are not exported from any public module"."""
    assert not hasattr(survjax, name)
    for module in ("survjax.diagnostics", "survjax.distributions"):
        try:
            mod = importlib.import_module(module)
        except ImportError:
            continue
        assert not hasattr(mod, name), module
