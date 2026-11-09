"""Shared cases for the post-release summary feature tests (spec 8.1-8.5, 8.8; rev 51).

Used by ``test_summary_rmst_*``, ``test_summary_mean_*`` and
``test_summary_quantile_*``. Families come from the built-ins and from
``spec_families`` (public extension interface only); no expected value is
computed here.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, f, ph

import survjax

# Feature markers (spectest CLAUDE.md, "Post-release features").
CONTAINERS = pytest.mark.feature("summary.containers")
RMST = pytest.mark.feature("summary.rmst")
RMST_INFO = pytest.mark.feature("summary.containers", "summary.rmst")
MEAN = pytest.mark.feature("summary.mean")
MEAN_INFO = pytest.mark.feature("summary.containers", "summary.mean")
QUANTILE = pytest.mark.feature("summary.quantile")
QUANTILE_INFO = pytest.mark.feature("summary.containers", "summary.quantile")


def families():
    """One scalar distribution of each kind: built-ins, PH closures, custom and PH without closure."""
    return {
        "exponential": lambda: survjax.Exponential(rate=f(0.7)),
        "weibull": lambda: survjax.Weibull(shape=f(2.5), scale=f(2.0)),
        "ph-exponential": lambda: ph(survjax.Exponential(rate=f(0.7)), f(0.4)),
        "ph-weibull": lambda: ph(survjax.Weibull(shape=f(2.5), scale=f(2.0)), f(-0.3)),
        "custom": lambda: fam.const_hazard()(rate=f(0.5)),
        "ph-custom": lambda: ph(fam.power_hazard()(shape=f(1.5), scale=f(2.0)), f(0.3)),
    }


FAMILIES = list(families())
BUILT_IN = ["exponential", "weibull", "ph-exponential", "ph-weibull"]
GENERIC = ["custom", "ph-custom"]


def draws_by_arms(family):
    """A `(S, 2)` batch (S = 3) for each kind of family in ``families()``."""
    k = f([[1.5, 2.5], [0.8, 1.2], [3.0, 1.0]])
    rates = f([[0.5, 0.7], [1.0, 2.0], [0.1, 0.3]])
    k_col = f([[1.5], [0.8], [3.0]])
    eta = f([[0.0, 0.5], [-0.3, 0.2], [0.1, -1.0]])
    if family == "exponential":
        return survjax.Exponential(rate=rates)
    if family == "weibull":
        return survjax.Weibull(shape=k, scale=f(2.0))
    if family == "ph-exponential":
        return ph(survjax.Exponential(rate=f([[0.5], [1.0], [0.1]])), eta)
    if family == "ph-weibull":
        return ph(survjax.Weibull(shape=k_col, scale=f(2.0)), eta)
    if family == "custom":
        return fam.const_hazard()(rate=rates)
    return ph(fam.power_hazard()(shape=k_col, scale=f(2.0)), eta)


def violations():
    """Batches of 3 whose middle entry violates a declaration (spec 4.7); the others are valid."""
    return {
        "weibull-shape": survjax.Weibull(shape=f([2.5, -1.0, 1.5]), scale=f(2.0)),
        "exponential-rate": survjax.Exponential(rate=f([0.5, 0.0, 0.7])),
        "ph-baseline": ph(survjax.Weibull(shape=f(2.5), scale=f([2.0, NAN, 1.0])), f(0.3)),
        "ph-eta": ph(survjax.Exponential(rate=f(0.7)), f([0.1, INF, -0.2])),
        "custom-constraint": fam.constrained_rate()(rate=f([0.5, -1.0, 0.7])),
        "custom-condition": fam.capped_rate()(rate=f([0.5, 2.0, 0.7]), cap=f(1.0)),
        "ph-custom-baseline": ph(fam.constrained_rate()(rate=f([0.5, -1.0, 0.7])), f(0.2)),
    }


VIOLATIONS = list(violations())

# --------------------------------------------------------------- derivatives

MODES = ("rev", "fwd")
HESSIANS = {
    "rev-over-rev": lambda fn: jax.jacrev(jax.jacrev(fn)),
    "fwd-over-rev": lambda fn: jax.jacfwd(jax.jacrev(fn)),
    "fwd-over-fwd": lambda fn: jax.jacfwd(jax.jacfwd(fn)),
    "rev-over-fwd": lambda fn: jax.jacrev(jax.jacfwd(fn)),
}


def grad(fn, argnums, mode):
    return (jax.grad if mode == "rev" else jax.jacfwd)(fn, argnums=argnums)


def assert_hessian(got, variables, ref_fn, value, scale, err_msg):
    """Every entry ``H_ij`` with weight ``|x_i x_j|`` within ``scale * |value|`` of the reference (12.1 log units)."""
    got = np.asarray(got, dtype=np.float64)
    n = len(variables)
    assert got.shape == (n, n), got.shape
    assert np.isfinite(got).all(), f"{err_msg}: non-finite Hessian {got!r}"
    for i, j in itertools.product(range(n), repeat=2):
        orders = [0] * n
        orders[i] += 1
        orders[j] += 1
        want = ref_fn(tuple(orders))
        err = abs(variables[i] * variables[j]) * abs(got[i, j] - want)
        assert err <= scale * abs(value), f"{err_msg}[{i},{j}]: {got[i, j]} vs {want} (weighted error {err})"


def valid_subset_grad(full, alone, x, mode, idx=(0, 2)):
    """Gradient of the valid entries' sum from the full batch, and from the valid entries alone."""
    index = jnp.asarray(idx)
    g_full = grad(lambda y: jnp.sum(full(y)[index]), 0, mode)(x)
    g_alone = grad(lambda y: jnp.sum(alone(y)), 0, mode)(x)
    return g_full, g_alone
