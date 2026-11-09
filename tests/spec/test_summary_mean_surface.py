"""Feature `summary.mean`: mean surface (spec 4.2, 4.4, 8.1, 8.5, 8.8; rev 51).

8.8: once `mean` is enabled, section 8.7's "absence, `NotImplementedError`,
missing `options` argument, and the derivative restrictions" no longer apply.
8.1: `mean()` is `∫_0^inf S(t) dt`; `mean_with_info` returns "`(values,
SummaryInfo)` and share[s] the ordinary backend". 4.2: generic numerical
version, families may override. 4.4: PH uses the closure, "otherwise the
generic method". 8.5: "Summary methods take `options=survjax.SummaryOptions(...)`".
The signature `mean(*, options=None)` follows the 8.7 keyword-only `options`
note (Q-056).
"""

import numpy as np
import pytest
from helpers import COUNT_FIELDS, INFO_FIELDS, assert_close, assert_status, f, status
from summary_cases import BUILT_IN, FAMILIES, MEAN, MEAN_INFO, families

import survjax


@pytest.mark.parametrize("family", FAMILIES)
@MEAN
def test_every_family_provides_mean(family):
    """spec 4.2/4.4/8.8, rev 51: every family, including custom families and PH wrappers without a closure, provides `mean`; finite here."""
    value = np.asarray(families()[family]().mean())
    assert value.shape == ()
    assert np.isfinite(value)


@pytest.mark.parametrize("family", FAMILIES)
@MEAN_INFO
def test_mean_with_info_returns_values_and_summary_info(family):
    """spec 8.1, rev 51: `mean_with_info` returns `(values, SummaryInfo)`; values equal `mean` (same backend)."""
    d = families()[family]()
    out = d.mean_with_info()
    assert isinstance(out, tuple) and len(out) == 2
    values, info = out
    assert isinstance(info, survjax.SummaryInfo)
    assert_close(values, d.mean())


@pytest.mark.parametrize("family", FAMILIES)
@MEAN_INFO
def test_valid_mean_reports_converged(family):
    """spec 8.5, rev 51: a valid finite mean reports `converged`; status codes are integers."""
    _, info = families()[family]().mean_with_info()
    assert np.issubdtype(status(info).dtype, np.integer)
    assert_status(info, "converged")


@pytest.mark.parametrize("family", FAMILIES)
@MEAN_INFO
def test_mean_info_fields_are_shaped_like_the_values(family):
    """spec 8.5, rev 51: "all five fields as array leaves shaped like the returned values"; counts are nonnegative."""
    values, info = families()[family]().mean_with_info()
    for name in INFO_FIELDS:
        assert np.shape(getattr(info, name)) == np.shape(values), name
    for name in COUNT_FIELDS:
        assert np.all(np.asarray(getattr(info, name)) >= 0), name


@pytest.mark.parametrize("family", FAMILIES)
@MEAN_INFO
def test_mean_options_none_selects_the_defaults(family):
    """spec 8.5/8.7, rev 51: `options=None`, omitted and `SummaryOptions()` agree (Q-056)."""
    d = families()[family]()
    omitted = d.mean()
    assert_close(d.mean(options=None), omitted)
    assert_close(d.mean(options=survjax.SummaryOptions()), omitted)


@pytest.mark.parametrize("family", FAMILIES)
@MEAN_INFO
def test_mean_options_are_keyword_only(family):
    """spec 8.7, rev 51: keyword-only `options`: positional options raise `TypeError` for `mean` and `mean_with_info`."""
    d = families()[family]()
    opts = survjax.SummaryOptions()
    with pytest.raises(TypeError):
        d.mean(opts)
    with pytest.raises(TypeError):
        d.mean_with_info(opts)


@pytest.mark.parametrize("family", BUILT_IN)
@MEAN_INFO
def test_built_in_mean_accepts_explicit_options(family):
    """spec 8.5, rev 51: built-in closed forms and PH closures take `options`; tighter-than-needed controls keep the value."""
    d = families()[family]()
    opts = survjax.SummaryOptions(rtol=f(1e-4), atol=f(1e-6), max_levels=6, max_solver_steps=32, max_bracket_steps=32)
    assert_close(d.mean(options=opts), d.mean())
