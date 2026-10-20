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
    """``survjax.log_likelihood`` as a NumPy array (spec 7.1)."""
    return np.asarray(survjax.log_likelihood(dist, data, **raw))


def ll_method(dist, data=None, **raw):
    """``dist.log_likelihood`` as a NumPy array (spec 7.1, rev 29)."""
    return np.asarray(dist.log_likelihood(data, **raw))


def nan_causes(dist, data, value):
    """The spec 7.1 (rev 29) triage recipe, as NumPy bool arrays of ``value``'s shape.

    Returns ``(bad_data, bad_params, other)``: ``data.issues.invalid_rows`` and
    ``dist.parameter_issues().invalid_entries`` broadcast to the contribution
    shape, and the NaN rows neither explains (``Z=0`` or numerical failure).
    For raw-form inputs pass ``SurvivalData(time, event)``, as the recipe says.
    """
    value = np.asarray(value)
    bad_data = np.broadcast_to(np.asarray(data.issues.invalid_rows).astype(bool), value.shape)
    bad_params = np.broadcast_to(
        np.asarray(dist.parameter_issues().invalid_entries).astype(bool), value.shape
    )
    other = np.isnan(value) & ~bad_data & ~bad_params
    return bad_data, bad_params, other
