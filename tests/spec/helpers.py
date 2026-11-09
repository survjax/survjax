"""Shared helpers for the survjax spec tests.

No survjax behaviour is asserted here. Every guessed API detail lives in this
module and names its API_ASSUMPTIONS.md entry, so resolving a question means
editing one line.
"""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import survjax

# Active dtype mode, as JAX sees it (conftest checks it matches JAX_ENABLE_X64).
FLOAT = np.dtype(jnp.asarray(0.0).dtype)
X64 = FLOAT == np.dtype(np.float64)

INF = np.inf
NAN = np.nan


class _NoInvalidDataError(Exception):
    """Stand-in when survjax lacks InvalidDataError; never raised, so only tests expecting it fail."""


# Spec 6.2: invalid rows raise only at an explicit host checkpoint. Malformed
# calls and unavailable explicit dtype requests retain their exception policy.
INVALID_DATA = getattr(survjax, "InvalidDataError", _NoInvalidDataError)
MISUSE = TypeError
DTYPE_REQUEST = ValueError

# Q-005: assigning to a field of the immutable SurvivalData.
IMMUTABLE = AttributeError

# Spec 3.2: the five public normalized fields.
FIELDS = ("exact", "lower", "upper", "left_trunc", "right_trunc")
TIME_FIELDS = ("lower", "upper", "left_trunc", "right_trunc")

# Spec 6.2: the ten per-rule DataIssues masks.
RULE_FIELDS = (
    "invalid_indicator",
    "conflicting_indicators",
    "missing_time2",
    "invalid_time",
    "invalid_time2",
    "invalid_interval",
    "invalid_truncation",
    "exact_outside_truncation",
    "censored_before_entry",
    "empty_observation",
)
MASK_FIELDS = (*RULE_FIELDS, "invalid_rows", "uninformative")


def f(values, dtype=None):
    """Input times as a NumPy array in the active default float dtype."""
    return np.asarray(values, dtype=FLOAT if dtype is None else dtype)


def b(values):
    """Boolean indicator array."""
    return np.asarray(values, dtype=bool)


def flag_data_and_issues(*args, **kwargs):
    """Test helper: construct through the public flag form, then read diagnostics."""
    data = survjax.SurvivalData(*args, **kwargs)
    return data, data.issues


def bounds_data_and_issues(*args, **kwargs):
    """Test helper: construct through the public bounds form, then read diagnostics."""
    data = survjax.SurvivalData.from_bounds(*args, **kwargs)
    return data, data.issues


def field(data, name):
    """Q-002: normalized fields are read as attributes."""
    return np.asarray(getattr(data, name))


def assert_fields(data, **expected):
    """Compare named normalized fields with expected values (exact equality)."""
    for name, value in expected.items():
        np.testing.assert_array_equal(
            field(data, name), np.asarray(value), err_msg=name
        )


def assert_same_fields(left, right):
    """Two SurvivalData objects hold identical normalized fields and dtypes."""
    for name in FIELDS:
        a, c = getattr(left, name), getattr(right, name)
        assert a.shape == c.shape, name
        assert a.dtype == c.dtype, name
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c), err_msg=name)


def assert_same_leaves(left, right):
    """Identical tree structure and identical leaves, including the issue bitfield (spec 3.2)."""
    assert treedef(left) == treedef(right)
    for a, c in zip(leaves(left), leaves(right), strict=True):
        assert a.shape == c.shape
        assert a.dtype == c.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c))


def mask(issues, name):
    return np.asarray(getattr(issues, name)).astype(bool)


def assert_mask(issues, name, expected):
    np.testing.assert_array_equal(
        mask(issues, name), np.asarray(expected, dtype=bool), err_msg=name
    )


def one_hot(n, *rows):
    out = np.zeros(n, dtype=bool)
    out[list(rows)] = True
    return out


def treedef(tree):
    return jax.tree_util.tree_structure(tree)


def leaves(tree):
    return jax.tree_util.tree_leaves(tree)


@contextlib.contextmanager
def raises_dtype_request():
    """Spec 3.3/6.2: ValueError for an unavailable dtype request, and not InvalidDataError."""
    with pytest.raises(DTYPE_REQUEST) as info:
        yield info
    assert not isinstance(info.value, INVALID_DATA)


@contextlib.contextmanager
def raises_misuse():
    """Spec 6.2: structural argument misuse raises TypeError."""
    with pytest.raises(MISUSE) as info:
        yield info


def assert_close(actual, expected, err_msg=""):
    """Spec 12.1 ordinary finite-value tolerances of the active dtype mode.

    Non-finite entries must match exactly (``assert_allclose`` compares
    infinities and NaNs positionally).
    """
    from tolerances import TOLERANCES

    actual = np.asarray(actual)
    np.testing.assert_allclose(
        actual.astype(np.float64),
        np.asarray(expected, dtype=np.float64),
        err_msg=err_msg,
        **TOLERANCES[FLOAT],
    )


def assert_rel_close(actual, expected, err_msg=""):
    """Spec 12.1 relative tolerance of the active dtype mode, with no absolute floor.

    For tail quantities far below 1 (for example ``H = 1e-20``) the 12.1
    absolute tolerance would accept any value near zero; spec 4.2 asks for
    relative precision there, so only the 12.1 ``rtol`` is applied.
    """
    from tolerances import TOLERANCES

    np.testing.assert_allclose(
        np.asarray(actual).astype(np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=TOLERANCES[FLOAT]["rtol"],
        atol=0.0,
        err_msg=err_msg,
    )


def log1mexp(x):
    """``log(1 - exp(x))`` for ``x <= 0`` in float64 NumPy, Maechler's split (spec 7.2, rev 25).

    An analytical reference, not survjax behaviour: ``log(-expm1(x))`` for
    ``x > -log 2`` and ``log1p(-exp(x))`` otherwise, each accurate to a few
    ulps on its branch; ``log1mexp(0) = -inf``, ``log1mexp(-inf) = 0``.
    """
    x = np.asarray(x, dtype=np.float64)
    near = x > -np.log(2.0)
    with np.errstate(divide="ignore"):
        return np.where(
            near,
            np.log(-np.expm1(np.where(near, x, -1.0))),
            np.log1p(-np.exp(np.where(near, -1.0, x))),
        )


def mp_log_cdf(cumulative_hazard):
    """mpmath reference ``log F = log1p(-exp(-H))`` at 50 digits, as float64 (spec 12.1, rev 25)."""
    import mpmath

    with mpmath.workdps(50):
        h = mpmath.mpf(cumulative_hazard)
        return float(mpmath.log1p(-mpmath.exp(-h)))


def mp_weibull_cumulative_hazard(k, b, t):
    """mpmath ``H = (t/b)**k`` at the dtype-rounded inputs, 50 digits (an ``mpf``)."""
    import mpmath

    with mpmath.workdps(50):
        return (mpmath.mpf(float(t)) / mpmath.mpf(float(b))) ** mpmath.mpf(float(k))


def tail_rtol(cumulative_hazard):
    """Relative tolerance for an upper-tail ``log F ~ -S = -exp(-H)`` (spec 12.1, rev 25).

    ``S`` has condition number ``H`` with respect to the relative error of
    ``H``, so even an exact ``log1mexp`` applied to an ``H`` (or ``log H``)
    correct to a few ulps loses about ``H * eps`` relative. The scale is the
    12.1 ``rtol`` or ``16 * H * eps``, whichever is larger (float32, ``H=70``:
    ``1.3e-4``; float64, ``H=600``: ``2e-12``, so the 12.1 ``1e-8`` applies).
    """
    from tolerances import TOLERANCES

    eps = float(np.finfo(FLOAT).eps)
    return max(TOLERANCES[FLOAT]["rtol"], 16.0 * float(cumulative_hazard) * eps)


def assert_tail_close(actual, expected, rtol, err_msg=""):
    """Zero absolute tolerance, so an erroneous zero cannot pass (spec 12.1, rev 25)."""
    np.testing.assert_allclose(
        np.asarray(actual).astype(np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=rtol,
        atol=0.0,
        err_msg=err_msg,
    )


def ll(dist, data=None, **raw):
    """``dist.log_likelihood``, the only likelihood entry point, as a NumPy array (spec 7.1, rev 35)."""
    return np.asarray(dist.log_likelihood(data, **raw))


def supported_mode(data):
    """The Slice A mode-support view of the spec 7.1 (rev 31) triage recipe, as a NumPy bool array.

    Exact or right-censored (``upper=+inf``) rows without truncation; a view of
    normalized fields, not a public container.
    """
    return (
        (np.asarray(data.left_trunc) == 0)
        & np.isposinf(np.asarray(data.right_trunc))
        & (np.asarray(data.exact).astype(bool) | np.isposinf(np.asarray(data.upper)))
    )


def nan_causes(dist, data, value):
    """The spec 7.1 (rev 31) Slice A triage recipe, as NumPy bool arrays of ``value``'s shape.

    Returns ``(bad_data, bad_params, other)``: ``data.issues.invalid_rows`` and
    ``dist.parameter_issues().invalid_entries`` broadcast to the contribution
    shape, and the NaN rows that neither explains and whose mode is supported
    (``Z=0`` or numerical failure). ``unsupported_nan`` gives the rest.
    For raw-form inputs pass ``SurvivalData(time, event)``, as the recipe says.
    """
    value = np.asarray(value)
    bad_data = np.broadcast_to(np.asarray(data.issues.invalid_rows).astype(bool), value.shape)
    bad_params = np.broadcast_to(
        np.asarray(dist.parameter_issues().invalid_entries).astype(bool), value.shape
    )
    unsupported = np.broadcast_to(~supported_mode(data), value.shape)
    other = np.isnan(value) & ~bad_data & ~bad_params & ~unsupported
    return bad_data, bad_params, other


def unsupported_nan(dist, data, value):
    """The recipe's ``unsupported_nan`` (spec 7.1, rev 31): unexplained NaN rows in an unsupported mode."""
    value = np.asarray(value)
    bad_data, bad_params, _ = nan_causes(dist, data, value)
    unsupported = np.broadcast_to(~supported_mode(data), value.shape)
    return np.isnan(value) & ~bad_data & ~bad_params & unsupported


# ------------------------------------------------ likelihood references (A7, rev 33)


def _rounded(*values):
    """Inputs rounded to the active dtype first, so a reference measures only survjax's arithmetic."""
    return [float(np.asarray(v, dtype=FLOAT)) for v in values]


def _reference(terms):
    """Sum of ``(term, amplification)`` pairs in mpmath, and a relative tolerance.

    Each term computed in floating point carries a relative error of a few
    ulps times its amplification (``1`` for ``log`` and products; ``1 + k|log z|``
    for ``z**k = exp(k log z)``, whose relative error is that of ``k log z``
    times its size). Summed, the absolute error is about
    ``eps * sum(|term| * amplification)``. The returned tolerance is the 12.1
    ``rtol`` or ``16`` times that bound relative to the value, whichever is
    larger, so cancellation between terms widens it and an erroneous zero
    (compared with zero absolute tolerance) still fails.
    """
    from tolerances import TOLERANCES

    value = sum(term for term, _ in terms)
    scale = sum(abs(term) * amp for term, amp in terms)
    eps = float(np.finfo(FLOAT).eps)
    rtol = max(TOLERANCES[FLOAT]["rtol"], 16.0 * eps * float(scale / abs(value)))
    return float(value), rtol


def mp_exponential_contribution(rate, t, exact):
    """mpmath (50 digits) untruncated contribution of an exponential row: ``log r - r t`` or ``-r t``.

    Returns ``(value, rtol)``; see ``_reference``.
    """
    import mpmath

    r, t = _rounded(rate, t)
    with mpmath.workdps(50):
        r, t = mpmath.mpf(r), mpmath.mpf(t)
        terms = [(-r * t, 1)]
        if exact:
            terms.append((mpmath.log(r), 1))
        return _reference(terms)


def mp_weibull_contribution(shape, scale, t, exact):
    """mpmath (50 digits) untruncated contribution of a Weibull row (spec 2, 4.1).

    ``z = t/b``: exact ``log k - log b + (k-1) log z - z**k``; right-censored
    ``-z**k``. Returns ``(value, rtol)``; see ``_reference``.
    """
    import mpmath

    k, b_, t = _rounded(shape, scale, t)
    with mpmath.workdps(50):
        k, b_, t = mpmath.mpf(k), mpmath.mpf(b_), mpmath.mpf(t)
        log_z = mpmath.log(t / b_)
        terms = [(-mpmath.exp(k * log_z), 1 + k * abs(log_z))]
        if exact:
            terms += [
                (mpmath.log(k), 1),
                (-mpmath.log(b_), 1),
                ((k - 1) * log_z, 1 + abs(log_z)),
            ]
        return _reference(terms)


# ------------------------------------------------------ concordance (A10, rev 34)

# Q-043: field names of ConcordanceInfo (spec 9.2: "concordant, discordant,
# tied-score and comparable counts plus invalid, unsupported and
# no-comparable-pair masks").
CONCORDANCE_COUNTS = ("concordant", "discordant", "tied_score", "comparable")
CONCORDANCE_MASKS = ("invalid", "unsupported", "no_comparable_pairs")


def concordance_info_class():
    """Q-042: ``ConcordanceInfo`` is discoverable in ``survjax.metrics`` (10.2)."""
    import survjax.metrics

    return survjax.metrics.ConcordanceInfo


def count(info, name):
    """Exact integer count(s) of a ``ConcordanceInfo`` field as a NumPy ``object`` array of Python ints (Q-044).

    Spec 9.1: totals use "a two-word (hi/lo) accumulator". Guess: a field is
    either an integer array, or carries ``hi`` and ``lo`` integer arrays with
    value ``hi * 2**32 + (lo mod 2**32)``.
    """
    value = getattr(info, name)
    if hasattr(value, "hi") and hasattr(value, "lo"):
        hi = np.asarray(value.hi).astype(np.int64).astype(object)
        lo = (np.asarray(value.lo).astype(np.int64) % (2**32)).astype(object)
        return hi * (2**32) + lo
    arr = np.asarray(value)
    assert np.issubdtype(arr.dtype, np.integer), (name, arr.dtype)
    return arr.astype(np.int64).astype(object)


def result_mask(info, name, result_shape):
    """A ``ConcordanceInfo`` mask reduced to one flag per concordance value (Q-045).

    The spec does not say whether ``invalid`` and ``unsupported`` are per-row
    or per-result masks. A mask of the result shape is used as is; a mask with
    a trailing observation axis is reduced with ``any`` over it, then
    broadcast to the result shape.
    """
    arr = np.asarray(getattr(info, name)).astype(bool)
    if arr.ndim > len(result_shape):
        arr = arr.any(axis=-1)
    return np.broadcast_to(arr, result_shape)


def harrell_reference(time, event, scores):
    """Pair counts of spec 9.2 by direct enumeration, for valid exact/right-censored rows.

    A transcription of the spec's rules, independent of survjax: each unordered
    pair once; unequal times comparable when the earlier one is exact; equal
    times comparable only for an event/censor pair, the event earlier; a larger
    score on the earlier row is concordant, smaller discordant, equal tied.
    Returns ``(concordant, discordant, tied_score, comparable)`` as ints.
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    conc = disc = tied = 0
    n = time.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if time[i] == time[j]:
                if event[i] == event[j]:
                    continue
                first, second = (i, j) if event[i] else (j, i)
            else:
                first, second = (i, j) if time[i] < time[j] else (j, i)
                if not event[first]:
                    continue
            if scores[first] > scores[second]:
                conc += 1
            elif scores[first] < scores[second]:
                disc += 1
            else:
                tied += 1
    return conc, disc, tied, conc + disc + tied


# ------------------------------------------------- ProportionalHazards (A11, rev 41)


def ph(baseline, log_hazard_ratio):
    """``survjax.ProportionalHazards`` with its keyword-only fields ``baseline`` and ``log_hazard_ratio`` (spec 4.4, rev 42; Q-050 resolved)."""
    return survjax.ProportionalHazards(
        baseline=baseline, log_hazard_ratio=log_hazard_ratio
    )


# ------------------------------------------------------- staged RMST (A12, rev 41)


def rounded(*values):
    """Inputs rounded to the active dtype, as floats: references are evaluated at what survjax receives."""
    return [float(np.asarray(v, dtype=FLOAT)) for v in values]


def rmst_rtol():
    """Spec 12.1 (rev 44): the factor on every staged RMST derivative scale is the ordinary ``rtol`` (Q-052 resolved)."""
    from tolerances import TOLERANCES

    return TOLERANCES[FLOAT]["rtol"]


def assert_log_parameter_derivative(actual, expected, param, rmst, err_msg=""):
    """Spec 12.1 (rev 44): the error in ``theta * d rmst/d theta`` within ``rtol * |rmst|``.

    Equivalently the natural derivative is within ``rtol * |rmst| / |theta|``.
    Used for ``k``, ``b`` (``b0``) and ``r``; for ``eta`` pass ``param=1``
    (12.1: "errors in `d rmst/d eta` have scale `|rmst|`").
    """
    actual = np.asarray(actual, dtype=np.float64)
    assert np.isfinite(actual).all(), f"{err_msg}: non-finite derivative {actual!r}"
    err = np.abs(np.asarray(param, dtype=np.float64) * (actual - np.asarray(expected, dtype=np.float64)))
    bound = rmst_rtol() * np.abs(np.asarray(rmst, dtype=np.float64))
    assert np.all(err <= bound), f"{err_msg}: error {err!r} exceeds {bound!r}"


def assert_cutoff_derivative(actual, expected, err_msg=""):
    """Spec 12.1 (rev 44): "The `tau` derivative uses absolute scale 1"; tolerance ``rtol * 1``."""
    actual = np.asarray(actual, dtype=np.float64)
    assert np.isfinite(actual).all(), f"{err_msg}: non-finite derivative {actual!r}"
    np.testing.assert_allclose(actual, np.asarray(expected, dtype=np.float64), rtol=0.0, atol=rmst_rtol(), err_msg=err_msg)


# ------------------------------------------ post-release summary features (spec 8.5, 8.8; rev 50-51)

# Spec 8.5 (rev 50): the five SummaryInfo fields, read as attributes (Q-056).
INFO_FIELDS = ("status", "error_estimate", "num_levels", "num_solver_steps", "num_bracket_steps")
COUNT_FIELDS = ("num_levels", "num_solver_steps", "num_bracket_steps")

# Spec 8.5 (rev 50): ``SummaryInfo.Status`` codes.
STATUS_CODES = {
    "converged": 0,
    "known_infinite": 1,
    "invalid_input": 2,
    "non_finite_arithmetic": 3,
    "failed_bracket": 4,
    "work_exhausted": 5,
}

# Q-058: generic numerical summaries (quadrature and solver) are compared with
# a factor GENERIC_FACTOR on the 12.1 scales; see that entry for the reason.
GENERIC_FACTOR = 10.0


def status(info):
    """The integer status array of a ``SummaryInfo`` (spec 8.5, rev 50; Q-056)."""
    return np.asarray(info.status)


def assert_status(info, expected, err_msg=""):
    """Every entry's status equals the named code(s); ``expected`` is a name or an array-like of names."""
    want = np.vectorize(STATUS_CODES.__getitem__)(np.asarray(expected))
    got = status(info)
    np.testing.assert_array_equal(got, np.broadcast_to(want, got.shape), err_msg=err_msg)


def generic_rtol():
    """Q-058: ``GENERIC_FACTOR`` times the 12.1 ``rtol`` of the active dtype."""
    from tolerances import TOLERANCES

    return GENERIC_FACTOR * TOLERANCES[FLOAT]["rtol"]


def assert_generic_close(actual, expected, err_msg=""):
    """Relative comparison at ``generic_rtol()``, no absolute floor; non-finite entries must match."""
    np.testing.assert_allclose(
        np.asarray(actual).astype(np.float64),
        np.asarray(expected, dtype=np.float64),
        rtol=generic_rtol(),
        atol=0.0,
        err_msg=err_msg,
    )


def assert_generic_log_derivative(actual, expected, param, value, err_msg=""):
    """``|param * (actual - expected)| <= generic_rtol() * |value|`` (12.1 log units, Q-058)."""
    actual = np.asarray(actual, dtype=np.float64)
    assert np.isfinite(actual).all(), f"{err_msg}: non-finite derivative {actual!r}"
    err = np.abs(np.asarray(param, dtype=np.float64) * (actual - np.asarray(expected, dtype=np.float64)))
    bound = generic_rtol() * np.abs(np.asarray(value, dtype=np.float64))
    assert np.all(err <= bound), f"{err_msg}: error {err!r} exceeds {bound!r}"
