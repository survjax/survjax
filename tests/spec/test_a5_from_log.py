"""A5: log-parameter constructors `Exponential.from_log` and `Weibull.from_log` (spec 4.1).

Spec 4.1 (rev 22): "They return the same class with `k=exp(log_shape)`,
`b=exp(log_scale)` (and `r=exp(log_rate)`), stored as the natural-parameter
leaves. Values and parameter gradients equal those of the ordinary constructor
wherever `exp` is finite." Log parameters whose `exp` overflows give infinite
parameters and fall under the invalid-parameters row of 4.5, which fixes no
value, so only finite cases are tested (Q-028, resolved).
"""

import jax
import numpy as np
import pytest
from helpers import assert_close, f

import survjax

# --------------------------------------------------------------- class and leaves


@pytest.mark.step("A5")
def test_exponential_from_log_returns_exponential():
    """spec 4.1, rev 22: `from_log` returns the same class."""
    assert type(survjax.Exponential.from_log(log_rate=f(0.0))) is survjax.Exponential


@pytest.mark.step("A5")
def test_weibull_from_log_returns_weibull():
    """spec 4.1, rev 22: `from_log` returns the same class."""
    dist = survjax.Weibull.from_log(log_shape=f(0.0), log_scale=f(0.0))
    assert type(dist) is survjax.Weibull


@pytest.mark.step("A5")
def test_exponential_from_log_stores_natural_rate():
    """spec 4.1, rev 22: natural parameters are the leaves; `rate = exp(log_rate)`."""
    a = [-2.0, 0.0, 1.5]
    dist = survjax.Exponential.from_log(log_rate=f(a))
    assert_close(dist.rate, np.exp(a))


@pytest.mark.step("A5")
def test_weibull_from_log_stores_natural_shape_and_scale():
    """spec 4.1/4.6, rev 22: `k=exp(log_shape)`, `b=exp(log_scale)`, stored under `shape` and `scale`."""
    ls, lb = [-0.7, 0.0, 1.1], [2.0, -1.0, 0.5]
    dist = survjax.Weibull.from_log(log_shape=f(ls), log_scale=f(lb))
    assert_close(dist.shape, np.exp(ls))
    assert_close(dist.scale, np.exp(lb))


@pytest.mark.step("A5")
def test_weibull_from_log_broadcasts_batch_dimensions():
    """spec 4.1, rev 22: `from_log` returns the same class, so batch dimensions are broadcast as at construction."""
    dist = survjax.Weibull.from_log(
        log_shape=f([0.0, 0.5, 1.0]), log_scale=f([[0.0], [1.0]])
    )
    assert dist.batch_shape == (2, 3)


# --------------------------------------------------------------- values


@pytest.mark.step("A5")
def test_exponential_from_log_values_match_natural_constructor():
    """spec 4.1, rev 22: `from_log(log_rate=a)` is the exponential with `r=exp(a)`; `log f = a - exp(a) t`."""
    a, t = f([-1.0, 0.3, 2.0]), f([0.5, 2.0, 0.1])
    out = survjax.Exponential.from_log(log_rate=a).log_prob(t)
    a64, t64 = np.asarray(a, np.float64), np.asarray(t, np.float64)
    assert_close(out, a64 - np.exp(a64) * t64)


@pytest.mark.step("A5")
def test_weibull_from_log_values_match_natural_constructor():
    """spec 4.1, rev 22: `from_log` gives the Weibull with `k=exp(log_shape)`, `b=exp(log_scale)`.

    ``log S = -(t/b)**k``; ``log_shape = log 2``, ``log_scale = log 4``, ``t = 2``: ``-(1/2)**2 = -0.25``.
    """
    dist = survjax.Weibull.from_log(log_shape=f(np.log(2.0)), log_scale=f(np.log(4.0)))
    assert_close(dist.log_survival(f(2.0)), -0.25)


# --------------------------------------------------------------- traced and derivatives


@pytest.mark.step("A5")
def test_from_log_with_traced_inputs_under_jit():
    """spec 4.1, rev 22: construction with traced parameters must work, including through `from_log`."""
    out = jax.jit(
        lambda a, c: survjax.Weibull.from_log(log_shape=a, log_scale=c).log_survival(
            f(2.0)
        )
    )(f(np.log(2.0)), f(np.log(4.0)))
    assert_close(out, -0.25)


@pytest.mark.step("A5")
def test_gradient_through_exponential_from_log():
    """spec 4.1/7.3, rev 22: `d/da log f = 1 - exp(a) t` for `log f = a - exp(a) t`."""
    t = 1.5
    grad = jax.grad(lambda a: survjax.Exponential.from_log(log_rate=a).log_prob(f(t)))(
        f(0.4)
    )
    assert_close(grad, 1.0 - np.exp(0.4) * t)


@pytest.mark.step("A5")
def test_gradient_through_weibull_from_log():
    """spec 4.1/7.3, rev 22: parameter gradients through `Weibull.from_log`.

    With ``u = log_shape``, ``v = log_scale``, ``k = e**u``, ``b = e**v``,
    ``z = t/b`` and ``log f = log k - log b + (k-1) log z - z**k``:

    - ``d/dk log f = 1/k + log z - z**k log z``, so ``d/du = k d/dk = 1 + k log z (1 - z**k)``;
    - ``d/db log f = (k/b)(z**k - 1)``, so ``d/dv = b d/db = k (z**k - 1)``.
    """
    u, v, t = 0.3, 0.6, 1.2
    gu, gv = jax.grad(
        lambda uu, vv: survjax.Weibull.from_log(log_shape=uu, log_scale=vv).log_prob(
            f(t)
        ),
        argnums=(0, 1),
    )(f(u), f(v))
    k, b = np.exp(u), np.exp(v)
    z = t / b
    assert_close(gu, 1.0 + k * np.log(z) * (1.0 - z**k))
    assert_close(gv, k * (z**k - 1.0))
