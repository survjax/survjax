"""Shared helpers for the Chronox spec tests.

No Chronox behaviour is asserted here. Every guessed API detail lives in this
module and names its API_ASSUMPTIONS.md entry, so resolving a question means
editing one line.
"""

import jax
import jax.numpy as jnp
import numpy as np

import chronox_jax as cx

# Active dtype mode, as JAX sees it (conftest checks it matches JAX_ENABLE_X64).
FLOAT = np.dtype(jnp.asarray(0.0).dtype)
X64 = FLOAT == np.dtype(np.float64)

INF = np.inf
NAN = np.nan

# Q-003: exception types of the eager constructors for invalid data or misuse.
REJECTED = (ValueError, TypeError)
# Q-007: the explicit unsupported-feature error for outside_interval_censored.
UNSUPPORTED = NotImplementedError
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


def with_issues(*args, **kwargs):
    """Q-009: ``SurvivalData.with_issues`` takes the flag constructor's arguments."""
    return cx.SurvivalData.with_issues(*args, **kwargs)


def field(data, name):
    """Q-002: normalized fields are read as attributes."""
    return np.asarray(getattr(data, name))


def assert_fields(data, **expected):
    """Compare named normalized fields with expected values (exact equality)."""
    for name, value in expected.items():
        np.testing.assert_array_equal(field(data, name), np.asarray(value), err_msg=name)


def assert_same_fields(left, right):
    """Two SurvivalData objects hold identical normalized fields and dtypes."""
    for name in FIELDS:
        a, c = getattr(left, name), getattr(right, name)
        assert a.shape == c.shape, name
        assert a.dtype == c.dtype, name
        np.testing.assert_array_equal(np.asarray(a), np.asarray(c), err_msg=name)


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
