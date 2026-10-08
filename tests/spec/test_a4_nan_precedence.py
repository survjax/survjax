"""A4: a NaN family value takes precedence over the `t<0` policy value (spec 4.5, rev 20).

Rev 20: "For `t<0` and NaN `t`, if the family's value at the substituted input
is NaN, the result is NaN; otherwise the policy value applies. This is how
invalid parameters surface on these rows; no validity domain is inferred".

The constant-hazard family has ``log S(t) = -r t``. With ``r = NaN`` its value
is NaN at every substituted input in ``[0, +inf]`` (``-NaN * t`` is NaN for
``t = 0`` and ``t = +inf`` too), so every direct method derived from it is NaN
there, whatever safe input the base chooses.

Deliberately not tested: finite invalid parameters (e.g. ``r < 0``) at ``t<0``.
Rev 21: the substituted input is not specified, so for such parameters the
result "may be NaN or the policy value, and may differ between methods"
(Q-026, resolved).
"""

import jax
import numpy as np
import pytest
import spec_families as fam
from helpers import INF, NAN, f

DIRECT_METHODS = (
    "log_survival",
    "survival",
    "log_cdf",
    "cumulative_hazard",
    "log_prob",
    "log_hazard",
    "hazard",
)

# Spec 4.5 (rev 21) `t<0` row: `log S=0`, `S=1`, `log_prob=-inf`,
# `log_cdf=-inf`, `H=0`, `log_hazard=-inf`, `h=0`.
POLICY_AT_NEGATIVE_TIME = {
    "log_survival": 0.0,
    "survival": 1.0,
    "log_cdf": -INF,
    "cumulative_hazard": 0.0,
    "log_prob": -INF,
    "log_hazard": -INF,
    "hazard": 0.0,
}


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_nan_parameter_gives_nan_at_negative_time(method):
    """spec 4.5, rev 20: for `t<0`, a NaN family value at the substituted input gives NaN, not the policy value.

    ``r = NaN`` at ``t = -1``.
    """
    out = getattr(fam.const_hazard()(rate=f(NAN)), method)(f(-1.0))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_nan_parameter_gives_nan_at_minus_infinity(method):
    """spec 4.5, rev 20: `t=-inf` is `t<0`; a NaN family value still gives NaN."""
    out = getattr(fam.const_hazard()(rate=f(NAN)), method)(f(-INF))
    assert np.isnan(np.asarray(out))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", DIRECT_METHODS)
def test_nan_parameter_gives_nan_at_negative_time_for_unguarded_family(method):
    """spec 4.5, rev 20: the same for family code written only for `t>=0` (Weibull-like, `scale=NaN`).

    ``log S = -(t/b)**k`` with ``b = NaN`` is NaN at every substituted input.
    """
    dist = fam.naive_power()(shape=f(2.0), scale=f(NAN))
    assert np.isnan(np.asarray(getattr(dist, method)(f(-1.0))))


@pytest.mark.step("A4")
@pytest.mark.parametrize("method", sorted(POLICY_AT_NEGATIVE_TIME))
def test_nan_batch_entry_does_not_change_valid_entry_policy_value(method):
    """spec 4.5, rev 20/21: per batch entry, NaN where the family is NaN, "otherwise the policy value applies".

    ``rate = (NaN, 2)`` at ``t = -1``: entry 0 is NaN; entry 1 is the `t<0`
    policy value.
    """
    out = np.asarray(getattr(fam.const_hazard()(rate=f([NAN, 2.0])), method)(f(-1.0)))
    assert out.shape == (2,)
    assert np.isnan(out[0])
    np.testing.assert_array_equal(out[1], POLICY_AT_NEGATIVE_TIME[method])


@pytest.mark.step("A4")
def test_nan_precedence_applies_elementwise_over_times():
    """spec 4.5, rev 20: in one `(N, K)` result, only NaN-parameter entries at `t<0` are NaN.

    ``rate = (NaN, 2)``; times ``(-1, 1)`` as a column. ``log S``:
    row ``t=-1`` is ``(NaN, 0)``; row ``t=1`` is ``(NaN, -2)``.
    """
    out = np.asarray(
        fam.const_hazard()(rate=f([NAN, 2.0])).log_survival(f([[-1.0], [1.0]]))
    )
    assert out.shape == (2, 2)
    assert np.isnan(out[0, 0])
    assert np.isnan(out[1, 0])
    np.testing.assert_array_equal(out[0, 1], 0.0)
    np.testing.assert_array_equal(out[1, 1], -2.0)


@pytest.mark.step("A4")
def test_nan_precedence_agrees_under_jit():
    """spec 4.5/12.1, rev 20: the NaN-precedence values are the same eagerly and under `jit`."""
    dist = fam.const_hazard()(rate=f([NAN, 2.0]))
    t = f([[-INF], [-1.0], [NAN]])
    for name in DIRECT_METHODS:
        eager = np.asarray(getattr(dist, name)(t))
        compiled = np.asarray(jax.jit(lambda d, s, n=name: getattr(d, n)(s))(dist, t))
        np.testing.assert_array_equal(compiled, eager, err_msg=name)


@pytest.mark.step("A4")
def test_nan_parameter_with_nan_time_gives_nan():
    """spec 4.5, rev 20: the rule also covers NaN `t`; NaN `t` with a NaN parameter is NaN."""
    out = fam.const_hazard()(rate=f(NAN)).log_survival(f(NAN))
    assert np.isnan(np.asarray(out))
