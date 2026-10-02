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
