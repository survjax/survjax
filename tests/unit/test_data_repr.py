"""Readable reprs for SurvivalData and DataIssues.

Expected strings are written by hand from the documented notation: exact
``t``, every censored row as its set ``(lower, upper]``, and a
``window`` column for non-default truncation.
"""

import jax
import jax.numpy as jnp
import pytest

from survjax import SurvivalData

NAN = float("nan")
INF = float("inf")


@pytest.fixture
def float_name(x64_enabled: bool) -> str:
    return "float64" if x64_enabled else "float32"


def test_repr_shows_each_observation_kind(float_name: str) -> None:
    """Spec 3.2: exact, right, left and interval rows print as their sets."""
    data = SurvivalData(
        [4.0, 12.0, 3.0, 6.0],
        right_censored=[False, True, False, False],
        left_censored=[False, False, True, False],
        interval_censored=[False, False, False, True],
        time2=[NAN, NAN, NAN, 9.0],
    )
    assert repr(data) == "\n".join(
        [
            f"SurvivalData: 4 observations, {float_name}",
            "  0  4",
            "  1  (12, inf]",
            "  2  (0, 3]",
            "  3  (6, 9]",
        ]
    )


def test_repr_shows_non_default_window(float_name: str) -> None:
    """Spec 3.2: rows with truncation show their window (L, U]."""
    data = SurvivalData(
        [4.0, 6.0], [1, 1], left_trunc=[0.0, 2.0], right_trunc=[INF, 8.0]
    )
    assert repr(data) == "\n".join(
        [
            f"SurvivalData: 2 observations, {float_name}",
            "  0  4",
            "  1  6   window (2, 8]",
        ]
    )


def test_repr_footer_counts_invalid_and_uninformative_rows(float_name: str) -> None:
    """Spec 6.2: invalid rows print as stored, with a footer to data.issues.

    Row 0 has equal endpoints without the exact mask (invalid interval); row 1
    is (0, inf], valid but uninformative.
    """
    data = SurvivalData.from_bounds([5.0, 0.0], [5.0, INF])
    assert repr(data) == "\n".join(
        [
            f"SurvivalData: 2 observations, {float_name}",
            "  0  (5, 5]",
            "  1  (0, inf]",
            "issues: 1 invalid row, 1 uninformative (see .issues)",
        ]
    )


def test_repr_elides_middle_of_long_data(float_name: str) -> None:
    """More than 20 rows print the first 10 and last 5."""
    data = SurvivalData(jnp.arange(1.0, 26.0))
    head = [f"  {i:>3}  {i + 1}" for i in range(10)]
    tail = [f"  {i:>3}  {i + 1}" for i in range(20, 25)]
    assert repr(data) == "\n".join(
        [f"SurvivalData: 25 observations, {float_name}", *head, "  ...", *tail]
    )


def test_repr_labels_two_dimensional_rows(float_name: str) -> None:
    """Spec 3.2: n-d observations show their shape and index tuples."""
    data = SurvivalData.from_bounds(
        [[1.0, 0.0], [2.0, 3.0]],
        [[1.0, 4.0], [INF, 5.0]],
        exact=[[True, False], [False, False]],
    )
    assert repr(data) == "\n".join(
        [
            f"SurvivalData: 4 observations, shape (2, 2), {float_name}",
            "  (0, 0)  1",
            "  (0, 1)  (0, 4]",
            "  (1, 0)  (2, inf]",
            "  (1, 1)  (3, 5]",
        ]
    )


def test_traced_reprs_show_only_shape(float_name: str) -> None:
    """Under jit the reprs read no values (no tracer-to-NumPy conversion)."""
    captured: list[str] = []

    @jax.jit
    def build(time: jax.Array) -> jax.Array:
        data = SurvivalData(time)
        captured.append(repr(data))
        captured.append(repr(data.issues))
        return data.lower

    build(jnp.array([2.0, 3.0]))
    assert captured == [
        f"SurvivalData(shape=(2,), dtype={float_name}, traced)",
        "DataIssues(shape=(2,), traced)",
    ]


def test_valid_issues_repr() -> None:
    """Spec 6.2: valid data print a one-line verdict."""
    assert (
        repr(SurvivalData([4.0, 12.0], [1, 0]).issues) == "DataIssues: all 2 rows valid"
    )


def test_invalid_issues_repr_lists_firing_rules() -> None:
    """Spec 6.2: each firing rule shows its row count and first index.

    Row 0: exact time 0 (invalid_time). Row 1: interval without time2
    (missing_time2). Row 2: (0, inf] is uninformative, not invalid.
    """
    data = SurvivalData(
        [0.0, 6.0, 0.0],
        interval_censored=[False, True, True],
        time2=[NAN, NAN, INF],
    )
    assert repr(data.issues) == "\n".join(
        [
            "DataIssues: 2 of 3 rows invalid",
            "  missing_time2             1 row, first index (1,)",
            "  invalid_time              1 row, first index (0,)",
            "  uninformative             1 row, first index (2,)  (not an error)",
        ]
    )


def test_repr_change_keeps_six_leaves() -> None:
    """Spec 3.2: the private issue bitfield remains the sixth leaf."""
    data = SurvivalData(jnp.array([2.0, 3.0]), [1, 0])
    assert len(jax.tree_util.tree_leaves(data)) == 6
