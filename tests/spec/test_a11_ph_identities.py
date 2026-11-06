"""A11: PH identities with the equivalent Weibull and Exponential laws (spec 4.4, 7.3, 12.1; rev 41).

4.4: "every method of `ProportionalHazards(Weibull(k, b0), eta)` and
`ProportionalHazards(Exponential(r), eta)` available in the milestone under
test equals the same method of `Weibull(shape=k, scale=b0*exp(-eta/k))` and
`Exponential(rate=r*exp(eta))`: direct methods, the likelihood and delegated
summaries, in values and first derivatives with respect to `eta` and the
baseline parameters ... The tests include broadcasting between baseline
parameters and `eta`, for example a baseline of batch `(S, 1)` with `eta` of
shape `(S, 2)`." Derivatives stay in the ordinary range (7.3 initial-release
exception; 12.1). Summaries (RMST) are A12.

Both sides are survjax calculations; the equivalent side is written as a
function of ``(k, b0, eta)`` so its derivatives are the chain rule through
``b = b0 exp(-eta/k)`` (or ``rate = r exp(eta)``). The hand-derived values of
``test_a11_ph_direct_methods.py`` and ``test_a11_ph_derivatives.py`` anchor
the identity to independent references. Interior magnitudes are O(1), so the
12.1 ordinary tolerances apply to values and derivatives alike.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import assert_close, assert_rel_close, b, f, ph

import survjax

METHODS = (
    "log_survival",
    "survival",
    "log_cdf",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)
GRID_METHODS = tuple(m + "_grid" for m in METHODS)

# Ordinary-range paired cases (k<1, k=1, k>1).
K = [0.7, 1.0, 2.5]
B0 = [0.5, 2.0, 3.0]
ETA = [-1.5, 0.0, 0.8]
T = [0.3, 1.0, 4.0]
RATE = [0.5, 1.0, 2.0]

DERIVATIVES = {"jacrev": jax.jacrev, "jacfwd": jax.jacfwd}


def _ph_weibull(k, b0, eta):
    return ph(survjax.Weibull(shape=k, scale=b0), eta)


def _equivalent_weibull(k, b0, eta):
    return survjax.Weibull(shape=k * jnp.ones_like(eta), scale=b0 * jnp.exp(-eta / k))


def _ph_exponential(r, eta):
    return ph(survjax.Exponential(rate=r), eta)


def _equivalent_exponential(r, eta):
    return survjax.Exponential(rate=r * jnp.exp(eta))


def _compare(method, actual, expected):
    if method.startswith("log_cdf"):
        assert_rel_close(actual, expected, err_msg=method)
    else:
        assert_close(actual, expected, err_msg=method)


# --------------------------------------------------------------- values


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", METHODS)
def test_ph_weibull_values_equal_equivalent_weibull(method):
    """spec 4.4, rev 41: PH-Weibull equals `Weibull(shape=k, scale=b0*exp(-eta/k))` in values."""
    args = (f(K), f(B0), f(ETA))
    _compare(
        method,
        getattr(_ph_weibull(*args), method)(f(T)),
        getattr(_equivalent_weibull(*args), method)(f(T)),
    )


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", METHODS)
def test_ph_exponential_values_equal_equivalent_exponential(method):
    """spec 4.4, rev 41: PH-Exponential equals `Exponential(rate=r*exp(eta))` in values."""
    args = (f(RATE), f(ETA))
    _compare(
        method,
        getattr(_ph_exponential(*args), method)(f(T)),
        getattr(_equivalent_exponential(*args), method)(f(T)),
    )


@pytest.mark.step("A11")
@pytest.mark.parametrize("method", GRID_METHODS)
def test_ph_weibull_grids_equal_equivalent_weibull_with_draw_by_arm_broadcasting(method):
    """spec 4.4, rev 41: the identity with a baseline of batch `(S, 1)` and `eta` `(S, 2)`, through the grid twins."""
    k = f([[0.8], [1.0], [2.0]])
    b0 = f([[1.0], [2.0], [0.5]])
    eta = f([[0.0, 0.7], [-0.4, 0.3], [1.0, -1.0]])
    t = f([0.25, 1.0, 3.0])
    out = getattr(_ph_weibull(k, b0, eta), method)(t)
    assert np.shape(out) == (3, 3, 2)
    _compare(method, out, getattr(_equivalent_weibull(k, b0, eta), method)(t))


@pytest.mark.step("A11")
def test_ph_exponential_grid_equals_equivalent_with_draw_by_arm_broadcasting():
    """spec 4.4, rev 41: PH-Exponential identity for baseline `(S, 1)` against `eta` `(S, 2)`."""
    r = f([[0.5], [1.0], [2.0]])
    eta = f([[0.0, 0.7], [-0.4, 0.3], [1.0, -1.0]])
    t = f([0.25, 1.0, 3.0])
    for method in METHODS:
        _compare(
            method,
            getattr(_ph_exponential(r, eta), method + "_grid")(t),
            getattr(_equivalent_exponential(r, eta), method + "_grid")(t),
        )


def _data():
    return survjax.SurvivalData(f(T), b([True, False, True]))


@pytest.mark.step("A11")
def test_ph_weibull_likelihood_equals_equivalent_weibull():
    """spec 4.4, rev 41: the identity holds for "the likelihood" (exact and right-censored, untruncated)."""
    args = (f(K), f(B0), f(ETA))
    assert_close(
        _ph_weibull(*args).log_likelihood(_data()),
        _equivalent_weibull(*args).log_likelihood(_data()),
    )


@pytest.mark.step("A11")
def test_ph_exponential_likelihood_equals_equivalent_exponential():
    """spec 4.4, rev 41: the PH-Exponential likelihood identity."""
    args = (f(RATE), f(ETA))
    assert_close(
        _ph_exponential(*args).log_likelihood(_data()),
        _equivalent_exponential(*args).log_likelihood(_data()),
    )


@pytest.mark.step("A11")
def test_likelihood_identity_with_draw_by_arm_broadcasting():
    """spec 4.4/5, rev 41: baseline `(S, 1)`, `eta` `(S, 2)` against data `(2,)` (paired arms) gives `(S, 2)`."""
    k = f([[0.8], [1.0], [2.0]])
    b0 = f([[1.0], [2.0], [0.5]])
    eta = f([[0.0, 0.7], [-0.4, 0.3], [1.0, -1.0]])
    data = survjax.SurvivalData(f([1.2, 2.5]), b([True, False]))
    out = _ph_weibull(k, b0, eta).log_likelihood(data)
    assert np.shape(out) == (3, 2)
    assert_close(out, _equivalent_weibull(k, b0, eta).log_likelihood(data))


# --------------------------------------------------------------- first derivatives


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(DERIVATIVES))
@pytest.mark.parametrize("method", METHODS)
def test_ph_weibull_first_derivatives_equal_equivalent_weibull(method, mode):
    """spec 4.4/7.3, rev 41: first derivatives in `k`, `b0` and `eta` equal those through the equivalent Weibull, forward and reverse."""
    d = DERIVATIVES[mode]
    t = f(T)

    def lhs(k, b0, eta):
        return getattr(_ph_weibull(k, b0, eta), method)(t)

    def rhs(k, b0, eta):
        return getattr(_equivalent_weibull(k, b0, eta), method)(t)

    args = (f(K), f(B0), f(ETA))
    got = d(lhs, argnums=(0, 1, 2))(*args)
    want = d(rhs, argnums=(0, 1, 2))(*args)
    for name, g, w in zip(("k", "b0", "eta"), got, want, strict=True):
        assert np.isfinite(np.asarray(g)).all(), name
        assert_close(g, w, err_msg=f"{method} d/d{name}")


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(DERIVATIVES))
@pytest.mark.parametrize("method", METHODS)
def test_ph_exponential_first_derivatives_equal_equivalent_exponential(method, mode):
    """spec 4.4/7.3, rev 41: first derivatives in `r` and `eta` equal those through `Exponential(rate=r*exp(eta))`."""
    d = DERIVATIVES[mode]
    t = f(T)

    def lhs(r, eta):
        return getattr(_ph_exponential(r, eta), method)(t)

    def rhs(r, eta):
        return getattr(_equivalent_exponential(r, eta), method)(t)

    got = d(lhs, argnums=(0, 1))(f(RATE), f(ETA))
    want = d(rhs, argnums=(0, 1))(f(RATE), f(ETA))
    for name, g, w in zip(("r", "eta"), got, want, strict=True):
        assert_close(g, w, err_msg=f"{method} d/d{name}")


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(DERIVATIVES))
def test_likelihood_first_derivatives_equal_equivalent_weibull(mode):
    """spec 4.4/7.3, rev 41: the likelihood identity in first derivatives, shared baseline and per-row `eta`."""
    d = DERIVATIVES[mode]
    data = _data()

    def lhs(k, b0, eta):
        return _ph_weibull(k, b0, eta).log_likelihood(data)

    def rhs(k, b0, eta):
        return _equivalent_weibull(k, b0, eta).log_likelihood(data)

    args = (f(1.7), f(2.0), f(ETA))
    for g, w in zip(d(lhs, argnums=(0, 1, 2))(*args), d(rhs, argnums=(0, 1, 2))(*args), strict=True):
        assert_close(g, w)


@pytest.mark.step("A11")
@pytest.mark.parametrize("mode", list(DERIVATIVES))
def test_likelihood_first_derivatives_equal_equivalent_exponential(mode):
    """spec 4.4/7.3, rev 41: the PH-Exponential likelihood identity in first derivatives."""
    d = DERIVATIVES[mode]
    data = _data()

    def lhs(r, eta):
        return _ph_exponential(r, eta).log_likelihood(data)

    def rhs(r, eta):
        return _equivalent_exponential(r, eta).log_likelihood(data)

    for g, w in zip(
        d(lhs, argnums=(0, 1))(f(0.8), f(ETA)), d(rhs, argnums=(0, 1))(f(0.8), f(ETA)), strict=True
    ):
        assert_close(g, w)


@pytest.mark.step("A11")
def test_first_derivatives_with_draw_by_arm_broadcasting():
    """spec 4.4, rev 41: derivative identity for baseline `(S, 1)` and `eta` `(S, 2)`: gradients of a summed grid."""
    t = f([0.5, 2.0])

    def lhs(k, b0, eta):
        return jnp.sum(_ph_weibull(k, b0, eta).log_prob_grid(t))

    def rhs(k, b0, eta):
        return jnp.sum(_equivalent_weibull(k, b0, eta).log_prob_grid(t))

    args = (f([[0.8], [1.0], [2.0]]), f([[1.0], [2.0], [0.5]]), f([[0.0, 0.7], [-0.4, 0.3], [1.0, -1.0]]))
    for g, w in zip(
        jax.grad(lhs, argnums=(0, 1, 2))(*args), jax.grad(rhs, argnums=(0, 1, 2))(*args), strict=True
    ):
        assert np.shape(g) == np.shape(w)
        assert_close(g, w)
