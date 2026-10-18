"""A4: the base supplies survival limits at `t=0` and `t=+inf` (spec 2, 4.2, 4.5, 12.1, 12.2; rev 26, 27).

Rev 26 (decision 0040): "The base now supplies `log S`, `S`, `log F` and `H`
at `t=0` and `t=+inf`, the latter from a new optional
`log_survival_at_infinity()` (default `-inf`), and calls family code for these
methods only at finite `t>0`." At ``t=+inf`` the base also gives
``log_prob=-inf``. Base-supplied values "have parameter gradients equal to the
derivative of the value as a function of the parameters".

Custom families (``spec_families.py``):

- ``boundary_nan_hazard``: ``log S = -r t (t/t)``, NaN at ``0`` and ``+inf``,
  exact ``-r t`` for finite ``t>0``.
- ``softplus_log_logistic``: the spec 4.5 test case, plain
  ``log S = -softplus(beta (log t - log alpha))`` with NaN boundary gradients.
- ``mixture_cure``: improper, ``S(t) = c + (1-c) exp(-r t)``, overriding
  ``log_survival_at_infinity()`` with ``log c``.

Hand values for ``mixture_cure`` with ``c = 0.3``, ``r = 0.5``:
``log S(+inf) = log 0.3``, ``S(+inf) = 0.3``, ``log F(+inf) = log 0.7``,
``H(+inf) = -log 0.3``; ``d/dc`` of these: ``1/0.3``, ``1``, ``-1/0.7``,
``-1/0.3``; ``d/dr`` of each is ``0``.

Rev 27: "For valid parameters the substituted input is a finite `t>0`", so a
family correct only for finite ``t>0`` (``boundary_nan_hazard``) is finite at
the substituted input and its base-supplied values are the policy values, not
NaN; the ``boundary_nan_hazard`` boundary tests are tagged ``rev 27``.

The generic ``log_prob`` and ``log_hazard`` defaults at ``t=0`` are not tested:
"that result carries no value or gradient guarantee" (rev 26).
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import FLOAT, INF, NAN, assert_close, f

BASE_METHODS = ("log_survival", "survival", "log_cdf", "cumulative_hazard")

# Spec 4.5 (rev 26), base-supplied values with the proper default.
AT_ZERO = {
    "log_survival": 0.0,
    "survival": 1.0,
    "log_cdf": -INF,
    "cumulative_hazard": 0.0,
}
AT_INF_PROPER = {
    "log_survival": -INF,
    "survival": 0.0,
    "log_cdf": 0.0,
    "cumulative_hazard": INF,
    "log_prob": -INF,
}

C, R = 0.3, 0.5

# Mixture cure at t=+inf, hand-derived above.
AT_INF_CURE = {
    "log_survival": np.log(C),
    "survival": C,
    "log_cdf": np.log1p(-C),
    "cumulative_hazard": -np.log(C),
}
D_DC_AT_INF_CURE = {
    "log_survival": 1.0 / C,
    "survival": 1.0,
    "log_cdf": -1.0 / (1.0 - C),
    "cumulative_hazard": -1.0 / C,
}


def _nan_boundary(rate=0.5):
    return fam.boundary_nan_hazard()(rate=f(rate))


def _log_logistic(alpha=2.0, beta=3.0):
    return fam.softplus_log_logistic()(alpha=f(alpha), beta=f(beta))


def _cure(cure=C, rate=R):
    return fam.mixture_cure()(cure=f(cure), rate=f(rate))


# --------------------------------------------------------------- log_survival_at_infinity default


@pytest.mark.step("A4")
def test_default_log_survival_at_infinity_is_minus_infinity_for_scalar_family():
    """spec 4.2, rev 26: the default `log_survival_at_infinity()` is `-inf`, shape `batch_shape` `()`, parameter dtype."""
    out = fam.const_hazard()(rate=f(0.5)).log_survival_at_infinity()
    assert np.shape(out) == ()
    assert np.dtype(out.dtype) == FLOAT
    np.testing.assert_array_equal(np.asarray(out), -INF)


@pytest.mark.step("A4")
def test_default_log_survival_at_infinity_has_one_entry_per_batch_entry():
    """spec 4.2, rev 26: "`-inf` per batch entry ...: an array of shape `batch_shape` in the parameter dtype"."""
    out = fam.power_hazard()(
        shape=f([0.5, 1.0, 2.0]), scale=f(2.0)
    ).log_survival_at_infinity()
    assert np.shape(out) == (3,)
    assert np.dtype(out.dtype) == FLOAT
    np.testing.assert_array_equal(np.asarray(out), np.full(3, -INF))


@pytest.mark.step("A4")
def test_default_log_survival_at_infinity_excludes_component_axes():
    """spec 4.1/4.2, rev 26: the default has shape `batch_shape`, which excludes component axes (`rate (2,)`, `w (2, 3)`)."""
    dist = fam.component_rate()(
        rate=f([1.0, 2.0]), w=f([[1.0, 2.0, 3.0], [0.5, 0.5, 0.5]])
    )
    out = dist.log_survival_at_infinity()
    assert np.shape(out) == (2,)
    np.testing.assert_array_equal(np.asarray(out), [-INF, -INF])


# --------------------------------------------------------------- log_survival is called only for finite t > 0


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_zero_time_value_comes_from_the_base_when_family_is_nan_there(method):
    """spec 4.2/4.5, rev 27: `log_survival` is "Called for finite `t>0`; the base supplies the values at `t=0`".

    ``boundary_nan_hazard`` returns NaN at ``t=0``; the base still gives
    ``log S=0``, ``S=1``, ``log_cdf=-inf``, ``H=0``.
    """
    out = getattr(_nan_boundary(), method)(f(0.0))
    np.testing.assert_array_equal(np.asarray(out), AT_ZERO[method])


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", sorted(AT_INF_PROPER))
def test_infinite_time_value_comes_from_the_base_when_family_is_nan_there(method):
    """spec 4.2/4.5, rev 27: at `t=+inf`, `log S = log_survival_at_infinity() = -inf`; `S`, `log F`, `H` follow; `log_prob=-inf`.

    ``boundary_nan_hazard`` returns NaN at ``t=+inf``.
    """
    out = getattr(_nan_boundary(), method)(f(INF))
    np.testing.assert_array_equal(np.asarray(out), AT_INF_PROPER[method])


@pytest.mark.step("A4")
def test_nan_at_boundary_family_interior_values_are_its_formula():
    """spec 4.2, rev 26: for finite `t>0` the family formula is used: `log S = -r t`, `log f = log r - r t` (`r=0.5`)."""
    dist = _nan_boundary()
    t = np.array([1e-6, 0.5, 2.0, 40.0])
    assert_close(dist.log_survival(f(t)), -0.5 * t)
    assert_close(dist.log_prob(f(t)), np.log(0.5) - 0.5 * t)


@pytest.mark.step("A4")
def test_nan_at_boundary_family_values_elementwise_in_one_call():
    """spec 4.5, rev 27: per element: `log S` at `(-1, 0, 2, +inf, NaN)` is `(0, 0, -1, -inf, NaN)`."""
    out = np.asarray(_nan_boundary().log_survival(f([-1.0, 0.0, 2.0, INF, NAN])))
    np.testing.assert_array_equal(out[[0, 1, 3]], [0.0, 0.0, -INF])
    assert_close(out[2], -1.0)
    assert np.isnan(out[4])


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_nan_at_boundary_family_values_agree_under_jit(method):
    """spec 4.5/12.1, rev 27: base-supplied boundary values are the same eagerly and under `jit`."""
    dist = fam.boundary_nan_hazard()(rate=f([0.5, 2.0]))
    t = f([[0.0], [1.0], [INF]])
    eager = getattr(dist, method)(t)
    compiled = jax.jit(lambda d, s: getattr(d, method)(s))(dist, t)
    assert_close(compiled, eager)


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_nan_at_boundary_family_values_agree_under_vmap(method):
    """spec 4.5/12.1, rev 27: base-supplied boundary values are the same eagerly and under `vmap` over time."""
    dist = _nan_boundary()
    t = f([0.0, 1.0, INF])
    assert_close(jax.vmap(getattr(dist, method))(t), getattr(dist, method)(t))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_nan_at_boundary_family_zero_time_has_zero_rate_gradient(method):
    """spec 4.5, rev 27: base-supplied values at `t=0` have gradient `0` ("`0` for `S(0)`, `log F(0)`, `H(0)`"), not NaN."""
    g = jax.grad(lambda r: getattr(fam.boundary_nan_hazard()(rate=r), method)(f(0.0)))(
        f(0.5)
    )
    np.testing.assert_array_equal(np.asarray(g), 0.0)


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", ["survival", "log_cdf"])
def test_nan_at_boundary_family_infinite_time_has_zero_rate_gradient(method):
    """spec 4.5, rev 27: with the proper default, `S(+inf)` and `log F(+inf)` have gradient `0`, not NaN."""
    g = jax.grad(lambda r: getattr(fam.boundary_nan_hazard()(rate=r), method)(f(INF)))(
        f(0.5)
    )
    np.testing.assert_array_equal(np.asarray(g), 0.0)


# --------------------------------------------------------------- spec 4.5 test case: plain softplus log-logistic


@pytest.mark.step("A4")
def test_softplus_log_logistic_survival_at_zero_is_one():
    """spec 4.5/12.1, rev 26: the plain softplus log-logistic "gives `S(0)=1`"."""
    np.testing.assert_array_equal(np.asarray(_log_logistic().survival(f(0.0))), 1.0)


@pytest.mark.step("A4")
def test_softplus_log_logistic_survival_at_zero_has_zero_gradients():
    """spec 4.5/12.1, rev 26: `S(0)=1` "with zero gradient", although the naive `log_survival` has a NaN `beta` gradient at `t=0`."""
    ga, gb = jax.grad(
        lambda a, bb: fam.softplus_log_logistic()(alpha=a, beta=bb).survival(f(0.0)),
        argnums=(0, 1),
    )(f(2.0), f(3.0))
    np.testing.assert_array_equal(np.asarray(ga), 0.0)
    np.testing.assert_array_equal(np.asarray(gb), 0.0)


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_softplus_log_logistic_zero_time_values_have_zero_gradients(method):
    """spec 4.5, rev 26: `log S(0)=0`, `log F(0)=-inf`, `H(0)=0` from the base, with zero parameter gradients."""
    ga, gb = jax.grad(
        lambda a, bb: getattr(fam.softplus_log_logistic()(alpha=a, beta=bb), method)(
            f(0.0)
        ),
        argnums=(0, 1),
    )(f(2.0), f(3.0))
    np.testing.assert_array_equal(np.asarray(ga), 0.0)
    np.testing.assert_array_equal(np.asarray(gb), 0.0)


@pytest.mark.step("A4")
def test_softplus_log_logistic_survival_at_infinity_is_zero():
    """spec 4.5/12.1, rev 26: the plain softplus log-logistic gives "`S(+inf)=0`"."""
    np.testing.assert_array_equal(np.asarray(_log_logistic().survival(f(INF))), 0.0)


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", ["survival", "log_cdf"])
def test_softplus_log_logistic_infinite_time_values_have_zero_gradients(method):
    """spec 4.5, rev 26: `S(+inf)=0` and `log F(+inf)=0` have gradient `0` with the proper default (naive `exp(-softplus)` gives NaN)."""
    ga, gb = jax.grad(
        lambda a, bb: getattr(fam.softplus_log_logistic()(alpha=a, beta=bb), method)(
            f(INF)
        ),
        argnums=(0, 1),
    )(f(2.0), f(3.0))
    np.testing.assert_array_equal(np.asarray(ga), 0.0)
    np.testing.assert_array_equal(np.asarray(gb), 0.0)


@pytest.mark.step("A4")
def test_softplus_log_logistic_interior_values():
    """spec 4.2, rev 26: interior values use the family formula: `S(t) = 1/(1 + (t/alpha)**beta)`.

    ``alpha=2``, ``beta=3``: ``S(2) = 1/2``, ``S(4) = 1/9``, ``S(1) = 1/(1 + 1/8) = 8/9``.
    """
    assert_close(
        _log_logistic().survival(f([1.0, 2.0, 4.0])), [8.0 / 9.0, 0.5, 1.0 / 9.0]
    )


@pytest.mark.step("A4")
def test_softplus_log_logistic_interior_gradient_is_finite():
    """spec 4.5/7.3, rev 26: the base changes nothing at interior times; `d S(4)/d beta = -S(1-S) log(t/alpha)`.

    ``S = 1/(1+u)`` with ``u = (t/alpha)**beta``, ``dS/dbeta = -u log(t/alpha)/(1+u)**2``;
    ``t=4``, ``alpha=2``, ``beta=3``: ``u = 8``, ``-8 log 2 / 81``.
    """
    g = jax.grad(
        lambda bb: fam.softplus_log_logistic()(alpha=f(2.0), beta=bb).survival(f(4.0))
    )(f(3.0))
    assert_close(g, -8.0 * np.log(2.0) / 81.0)


# --------------------------------------------------------------- improper family: log_survival_at_infinity override


@pytest.mark.step("A4")
def test_improper_family_log_survival_at_infinity_override_is_used():
    """spec 4.2, rev 26: "Improper families return `log S(+inf)`, for example the log cure fraction"."""
    assert_close(_cure().log_survival_at_infinity(), np.log(C))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_improper_family_values_at_infinity(method):
    """spec 2/4.5, rev 26: `log S(+inf) = log_survival_at_infinity()`; `S`, `log_cdf = log1mexp(log S)`, `H = -log S` follow.

    ``c = 0.3``: ``(log 0.3, 0.3, log 0.7, -log 0.3)``.
    """
    assert_close(getattr(_cure(), method)(f(INF)), AT_INF_CURE[method])


@pytest.mark.step("A4")
def test_improper_family_log_prob_at_infinity_is_minus_infinity():
    """spec 4.5, rev 26: at `t=+inf`, `log_prob=-inf` also for an improper family."""
    np.testing.assert_array_equal(np.asarray(_cure().log_prob(f(INF))), -INF)


@pytest.mark.step("A4")
def test_improper_family_generic_log_hazard_at_infinity_is_nan():
    """spec 4.2/4.5, rev 26: "the generic default gives NaN" for `log_hazard` and `hazard` at `+inf`."""
    dist = _cure()
    assert np.isnan(np.asarray(dist.log_hazard(f(INF))))
    assert np.isnan(np.asarray(dist.hazard(f(INF))))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_improper_family_values_at_zero(method):
    """spec 2/4.5, rev 26: `S(0)=1` "for every family", proper or not: `log S=0`, `S=1`, `log F=-inf`, `H=0`."""
    np.testing.assert_array_equal(
        np.asarray(getattr(_cure(), method)(f(0.0))), AT_ZERO[method]
    )


@pytest.mark.step("A4")
def test_improper_family_interior_values():
    """spec 4.2, rev 26: finite `t>0` uses the family formula: `S(2) = 0.3 + 0.7 exp(-1)`; `log F(2) = log(0.7 (1 - exp(-1)))`."""
    dist = _cure()
    s2 = C + (1.0 - C) * np.exp(-1.0)
    assert_close(dist.survival(f(2.0)), s2)
    assert_close(dist.log_cdf(f(2.0)), np.log((1.0 - C) * -np.expm1(-1.0)))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_improper_family_gradients_at_infinity(method):
    """spec 4.2/4.5/12.1, rev 26: values at `+inf` are differentiated "through `log_survival_at_infinity()` for an improper family".

    ``d/dc``: ``log S`` ``1/c``, ``S`` ``1``, ``log F`` ``-1/(1-c)``, ``H`` ``-1/c``;
    ``d/dr = 0`` for each (``c = 0.3``).
    """
    gc, gr = jax.grad(
        lambda c, r: getattr(fam.mixture_cure()(cure=c, rate=r), method)(f(INF)),
        argnums=(0, 1),
    )(f(C), f(R))
    assert_close(gc, D_DC_AT_INF_CURE[method])
    np.testing.assert_array_equal(np.asarray(gr), 0.0)


@pytest.mark.step("A4")
def test_improper_family_values_at_infinity_per_batch_entry():
    """spec 4.2/4.5, rev 26: per batch entry: `cure = (0.3, 0.6)` gives `S(+inf) = (0.3, 0.6)`; `S(1)` uses the formula."""
    dist = fam.mixture_cure()(cure=f([0.3, 0.6]), rate=f(0.5))
    assert np.shape(dist.log_survival_at_infinity()) == (2,)
    out = np.asarray(dist.survival(f([[1.0], [INF]])))
    assert out.shape == (2, 2)
    c = np.array([0.3, 0.6])
    assert_close(out[0], c + (1.0 - c) * np.exp(-0.5))
    assert_close(out[1], c)


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_improper_family_values_at_infinity_agree_under_jit_and_vmap(method):
    """spec 4.5/12.1, rev 26: the improper `+inf` values are the same eagerly, under `jit` and under `vmap` over time."""
    dist = _cure()
    t = f([0.0, 2.0, INF])
    eager = getattr(dist, method)(t)
    assert_close(jax.jit(lambda d, s: getattr(d, method)(s))(dist, t), eager)
    assert_close(jax.vmap(getattr(dist, method))(t), eager)


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", [*BASE_METHODS, "log_prob"])
def test_nan_log_survival_at_infinity_gives_nan_at_infinity(method):
    """spec 4.5, rev 26: "if ... at `t=+inf` `log_survival_at_infinity()` is NaN, the result is NaN".

    ``cure = -1``: ``log(-1)`` is NaN, so ``log_survival_at_infinity()`` is
    NaN (and so is the family at every input, so the result is NaN by either
    clause).
    """
    out = getattr(_cure(cure=-1.0), method)(f(INF))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", [*BASE_METHODS, "log_prob"])
def test_nan_log_survival_at_infinity_wins_over_finite_family_value(method):
    """spec 4.5, rev 26: at `t=+inf` the result is NaN when `log_survival_at_infinity()` is NaN, even where the family is finite.

    ``cure = -0.5``, ``rate = 0.5``: ``log_survival_at_infinity() = log(-0.5)``
    is NaN, while ``log S(t) = log(-0.5 + 1.5 exp(-0.5 t))`` is finite for
    ``t < 2 log 3 = 2.197`` (at ``reference_time() = 2`` it is
    ``log(0.0518)``) and NaN beyond. Whatever substituted input the base
    uses, the second clause of the rule gives NaN. ``t=0`` is not tested:
    for these parameters the result there "may be NaN or the policy value".
    """
    out = getattr(_cure(cure=-0.5, rate=0.5), method)(f(INF))
    assert np.isnan(np.asarray(out))


# --------------------------------------------------------------- overrides at the boundaries


@pytest.mark.step("A4")
def test_log_hazard_override_is_called_at_zero():
    """spec 4.5, rev 26: at `t=0`, "an override of `log_prob` or `log_hazard` is called ... and supplies" the hazard limit (`log r`)."""
    dist = fam.log_hazard_override()(rate=f(0.5))
    assert_close(dist.log_hazard(f(0.0)), np.log(0.5))
    assert_close(dist.hazard(f(0.0)), 0.5)


@pytest.mark.step("A4")
def test_log_prob_override_at_infinity_gives_base_value():
    """spec 4.5, rev 26: at `t=+inf`, `log_prob=-inf` is base-supplied, also over a `log_prob` override (sentinel elsewhere)."""
    dist = fam.log_prob_override()(rate=f(0.5))
    np.testing.assert_array_equal(np.asarray(dist.log_prob(f(INF))), -INF)


# --------------------------------------------------------------- NaN parameters at base-supplied boundaries


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_nan_parameter_gives_nan_at_zero_time(method):
    """spec 4.5, rev 26: a base-supplied value at `t=0` is NaN when the family's value at the substituted input is NaN.

    ``rate = NaN``: ``-NaN * t`` is NaN at every input.
    """
    out = getattr(fam.const_hazard()(rate=f(NAN)), method)(f(0.0))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_nan_parameter_gives_nan_at_infinite_time(method):
    """spec 4.5, rev 26: the same for the base-supplied values at `t=+inf`."""
    out = getattr(fam.const_hazard()(rate=f(NAN)), method)(f(INF))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", BASE_METHODS)
def test_nan_batch_entry_keeps_valid_entry_boundary_values(method):
    """spec 4.5, rev 26: per batch entry: `rate = (NaN, 2)` at `t = (0, +inf)`; entry 0 NaN, entry 1 the base values."""
    out = np.asarray(
        getattr(fam.const_hazard()(rate=f([NAN, 2.0])), method)(f([[0.0], [INF]]))
    )
    assert out.shape == (2, 2)
    assert np.all(np.isnan(out[:, 0]))
    np.testing.assert_array_equal(out[:, 1], [AT_ZERO[method], AT_INF_PROPER[method]])
