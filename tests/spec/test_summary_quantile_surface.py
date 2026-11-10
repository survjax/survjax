"""Feature `summary.quantile`: quantile, median and inverse cumulative hazard surface (spec 4.2, 4.4, 8.1, 8.5, 8.8; rev 51).

8.8: "each method of section 8.1 (summaries, their `_with_info` variants,
`inverse_cumulative_hazard` and `sample`) ... is enabled by its own
implementation step"; once enabled, section 8.7's "absence,
`NotImplementedError`, missing `options` argument, and the derivative
restrictions" no longer apply. 8.1: `quantile(p)`, `median() = quantile(0.5)`,
`inverse_cumulative_hazard(e)` "the `t` with `H(t)=e`, for `e>=0`";
`quantile_with_info` and `median_with_info` return "`(values, SummaryInfo)`
and share the ordinary backend". 8.5: "Summary methods take `options`".
Signatures `quantile(p, *, options=None)` and `median(*, options=None)` follow
the 8.7 keyword-only `options` note (Q-056). Rev 54 (8.1; Q-063 resolved)
fixes `inverse_cumulative_hazard(e, *, options=None)` with no `_with_info`
twin; its options, absent twin and input policy are tested in
``test_summary_quantile_policy.py``, and these tests call it with `e` only.
"""

import numpy as np
import pytest
from helpers import COUNT_FIELDS, INFO_FIELDS, assert_close, assert_status, f, status
from summary_cases import BUILT_IN, FAMILIES, QUANTILE, QUANTILE_INFO, families

import survjax

# --------------------------------------------------------------- quantile


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_every_family_provides_quantile(family):
    """spec 4.2/4.4/8.8, rev 51: every family, including custom families and PH wrappers without a closure, provides `quantile`; finite here."""
    value = np.asarray(families()[family]().quantile(f(0.4)))
    assert value.shape == ()
    assert np.isfinite(value)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_quantile_with_info_returns_values_and_summary_info(family):
    """spec 8.1, rev 51: `quantile_with_info` returns `(values, SummaryInfo)`; values equal `quantile` (same backend)."""
    d = families()[family]()
    out = d.quantile_with_info(f(0.4))
    assert isinstance(out, tuple) and len(out) == 2
    values, info = out
    assert isinstance(info, survjax.SummaryInfo)
    assert_close(values, d.quantile(f(0.4)))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_valid_interior_quantile_reports_converged(family):
    """spec 8.5, rev 51: a valid interior quantile reports `converged`; status codes are integers."""
    _, info = families()[family]().quantile_with_info(f(0.4))
    assert np.issubdtype(status(info).dtype, np.integer)
    assert_status(info, "converged")


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_quantile_info_fields_are_shaped_like_the_values(family):
    """spec 8.5, rev 51: "all five fields as array leaves shaped like the returned values"; counts are nonnegative."""
    values, info = families()[family]().quantile_with_info(f([0.2, 0.4]))
    for name in INFO_FIELDS:
        assert np.shape(getattr(info, name)) == np.shape(values), name
    for name in COUNT_FIELDS:
        assert np.all(np.asarray(getattr(info, name)) >= 0), name


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_quantile_options_none_selects_the_defaults(family):
    """spec 8.5/8.7, rev 51: `options=None`, omitted and `SummaryOptions()` agree (Q-056)."""
    d = families()[family]()
    omitted = d.quantile(f(0.4))
    assert_close(d.quantile(f(0.4), options=None), omitted)
    assert_close(d.quantile(f(0.4), options=survjax.SummaryOptions()), omitted)


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_quantile_options_are_keyword_only(family):
    """spec 8.7, rev 51: keyword-only `options`: positional options raise `TypeError` for `quantile`, `quantile_with_info`, `median` and `median_with_info`."""
    d = families()[family]()
    opts = survjax.SummaryOptions()
    with pytest.raises(TypeError):
        d.quantile(f(0.5), opts)
    with pytest.raises(TypeError):
        d.quantile_with_info(f(0.5), opts)
    with pytest.raises(TypeError):
        d.median(opts)
    with pytest.raises(TypeError):
        d.median_with_info(opts)


@pytest.mark.parametrize("family", BUILT_IN)
@QUANTILE_INFO
def test_built_in_quantile_accepts_explicit_options(family):
    """spec 8.5, rev 51: built-in closed forms and PH closures take `options`; tighter-than-needed controls keep the value."""
    d = families()[family]()
    opts = survjax.SummaryOptions(rtol=f(1e-4), atol=f(1e-6), max_levels=6, max_solver_steps=32, max_bracket_steps=32)
    assert_close(d.quantile(f(0.4), options=opts), d.quantile(f(0.4)))
    assert_close(d.median(options=opts), d.median())


# --------------------------------------------------------------- median


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_median_is_the_half_quantile(family):
    """spec 8.1/8.8, rev 51: `median()` is `quantile(0.5)`; it no longer raises (8.7 restriction lifted)."""
    d = families()[family]()
    got = d.median()
    assert np.shape(got) == ()
    assert_close(got, d.quantile(f(0.5)))


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE_INFO
def test_median_with_info_is_the_half_quantile_with_info(family):
    """spec 8.1, rev 51: `median_with_info` returns `(values, SummaryInfo)` from the shared backend, equal to `quantile_with_info(0.5)`."""
    d = families()[family]()
    values, info = d.median_with_info()
    q_values, q_info = d.quantile_with_info(f(0.5))
    assert isinstance(info, survjax.SummaryInfo)
    assert_close(values, d.median())
    assert_close(values, q_values)
    np.testing.assert_array_equal(status(info), status(q_info))


# --------------------------------------------------------------- inverse cumulative hazard


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_every_family_provides_inverse_cumulative_hazard(family):
    """spec 8.1/8.8, rev 51: public `inverse_cumulative_hazard(e)` exists on every family once enabled (no longer absent); finite for `e > 0`."""
    value = np.asarray(families()[family]().inverse_cumulative_hazard(f(0.7)))
    assert value.shape == ()
    assert np.isfinite(value) and value > 0


@pytest.mark.parametrize("family", FAMILIES)
@QUANTILE
def test_quantile_is_the_inverse_cumulative_hazard_of_minus_log1p_minus_p(family):
    """spec 8.1, rev 51: `quantile(p) = H⁻¹(-log1p(-p))` for `0<p<1`."""
    d = families()[family]()
    p = f(0.4)
    e = -np.log1p(-np.asarray(p, dtype=np.float64))
    assert_close(d.quantile(p), d.inverse_cumulative_hazard(f(e)))
