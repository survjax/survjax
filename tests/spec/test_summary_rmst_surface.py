"""Feature `summary.rmst`: generic RMST surface (spec 4.2, 4.4, 8.1, 8.5, 8.7, 8.8; rev 51).

8.8: once `rmst` is enabled, "section 8.7's restrictions on it no longer apply:
its absence, `NotImplementedError`, missing `options` argument, and the
derivative restrictions"; "Section 8.7's ... value policy and shape rules
remain in force". 4.2: summaries are "Generic numerical versions (section 8).
Families may override". 4.4: PH evaluates `rmst(tau)` on the closure if the
baseline provides one, "otherwise the generic method". 8.1: `rmst_with_info`
returns "`(values, SummaryInfo)` and share[s] the ordinary backend". 8.5:
"Summary methods take `options=survjax.SummaryOptions(...)`"; info fields are
"array leaves shaped like the returned values". The signature
`rmst(tau, *, options=None)` with `None` meaning the 8.5 defaults is the 8.7
note "Slice D adds a keyword-only `options` with a default" (Q-056).
"""

import numpy as np
import pytest
from helpers import COUNT_FIELDS, INFO_FIELDS, assert_close, assert_status, f, status
from summary_cases import BUILT_IN, FAMILIES, RMST, RMST_INFO, families

import survjax


def _plain(d, **kw):
    return d.rmst(f(1.5), **kw)


def _with_info(d, **kw):
    return d.rmst_with_info(f(1.5), **kw)


@pytest.mark.parametrize("family", FAMILIES)
@RMST
def test_every_family_provides_rmst(family):
    """spec 4.2/4.4/8.8, rev 51: every family, including custom families and PH wrappers without a closure, provides `rmst`; finite here."""
    value = np.asarray(_plain(families()[family]()))
    assert value.shape == ()
    assert np.isfinite(value)


@pytest.mark.parametrize("family", FAMILIES)
@RMST_INFO
def test_rmst_with_info_returns_values_and_summary_info(family):
    """spec 8.1, rev 51: `rmst_with_info` returns `(values, SummaryInfo)`; values equal `rmst` (same backend)."""
    d = families()[family]()
    out = _with_info(d)
    assert isinstance(out, tuple) and len(out) == 2
    values, info = out
    assert isinstance(info, survjax.SummaryInfo)
    assert_close(values, _plain(d))


@pytest.mark.parametrize("family", FAMILIES)
@RMST_INFO
def test_valid_interior_rmst_reports_converged(family):
    """spec 8.5, rev 51: a valid interior RMST reports `converged`; status codes are integers."""
    _, info = _with_info(families()[family]())
    assert np.issubdtype(status(info).dtype, np.integer)
    assert_status(info, "converged")


@pytest.mark.parametrize("family", FAMILIES)
@RMST_INFO
def test_rmst_info_fields_are_shaped_like_the_values(family):
    """spec 8.5, rev 51: "all five fields as array leaves shaped like the returned values"; counts are nonnegative."""
    values, info = _with_info(families()[family]())
    for name in INFO_FIELDS:
        assert np.shape(getattr(info, name)) == np.shape(values), name
    for name in COUNT_FIELDS:
        assert np.all(np.asarray(getattr(info, name)) >= 0), name


@pytest.mark.parametrize("family", FAMILIES)
@RMST_INFO
def test_rmst_options_none_selects_the_defaults(family):
    """spec 8.5/8.7, rev 51: `options=None`, omitted and `SummaryOptions()` agree (Q-056)."""
    d = families()[family]()
    omitted = _plain(d)
    assert_close(_plain(d, options=None), omitted)
    assert_close(_plain(d, options=survjax.SummaryOptions()), omitted)


@pytest.mark.parametrize("family", FAMILIES)
@RMST_INFO
def test_rmst_options_are_keyword_only(family):
    """spec 8.7, rev 51: "a keyword-only `options`": positional options raise `TypeError` for `rmst` and `rmst_with_info`."""
    d = families()[family]()
    opts = survjax.SummaryOptions()
    with pytest.raises(TypeError):
        d.rmst(f(1.0), opts)
    with pytest.raises(TypeError):
        d.rmst_with_info(f(1.0), opts)


@pytest.mark.parametrize("family", BUILT_IN)
@RMST_INFO
def test_built_in_rmst_accepts_explicit_options(family):
    """spec 8.5, rev 51: "Summary methods take `options`", built-in closed forms and PH closures included; tighter-than-needed controls keep the value."""
    d = families()[family]()
    opts = survjax.SummaryOptions(rtol=f(1e-4), atol=f(1e-6), max_levels=6, max_solver_steps=32, max_bracket_steps=32)
    assert_close(d.rmst(f(1.5), options=opts), d.rmst(f(1.5)))


@pytest.mark.parametrize("family", FAMILIES)
@RMST
def test_there_is_still_no_rmst_grid(family):
    """spec 8.7/8.8, rev 51: 8.7 shape rules remain in force: "There is no `rmst_grid`"."""
    assert not hasattr(families()[family](), "rmst_grid")
