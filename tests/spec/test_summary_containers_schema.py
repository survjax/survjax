"""Feature `summary.containers`: `SummaryOptions` and `SummaryInfo` (spec 8.5, 8.8; rev 51).

8.5: "Summary methods take `options=survjax.SummaryOptions(...)`" (exported
at the package top level). "`SummaryOptions(rtol=None, atol=None,
max_levels=8, max_solver_steps=64, max_bracket_steps=64)` ... Explicit
tolerances are dynamic PyTree leaves; the three work limits are static
metadata. The container is frozen." "`SummaryInfo(status, error_estimate,
num_levels, num_solver_steps, num_bracket_steps)` is a frozen PyTree with all
five fields as array leaves ... `SummaryInfo.Status`, an `IntEnum`:
`converged=0`, `known_infinite=1`, `invalid_input=2`,
`non_finite_arithmetic=3`, `failed_bracket=4`, `work_exhausted=5` ...
Construction stores these fields." 8.8: `SummaryOptions` and `SummaryInfo`
are enabled by their own step; these tests need no summary method.

Fields are read as attributes and constructed by keyword (Q-056).
"""

import enum

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import INFO_FIELDS, STATUS_CODES, f

import survjax


def _info(shape=(3,)):
    return survjax.SummaryInfo(
        status=jnp.zeros(shape, dtype=jnp.int32),
        error_estimate=jnp.full(shape, 1e-12, dtype=jnp.asarray(f(0.0)).dtype),
        num_levels=jnp.full(shape, 4, dtype=jnp.int32),
        num_solver_steps=jnp.zeros(shape, dtype=jnp.int32),
        num_bracket_steps=jnp.zeros(shape, dtype=jnp.int32),
    )


# --------------------------------------------------------------- exports


@pytest.mark.feature("summary.containers")
@pytest.mark.parametrize("name", ["SummaryOptions", "SummaryInfo"])
def test_summary_containers_are_exported_at_top_level(name):
    """spec 8.5/8.8, rev 51: `options=survjax.SummaryOptions(...)`; both containers are reachable as `survjax.<name>` once enabled."""
    assert hasattr(survjax, name)


# --------------------------------------------------------------- SummaryOptions


@pytest.mark.feature("summary.containers")
def test_summary_options_defaults():
    """spec 8.5, rev 51: `SummaryOptions(rtol=None, atol=None, max_levels=8, max_solver_steps=64, max_bracket_steps=64)`."""
    opts = survjax.SummaryOptions()
    assert opts.rtol is None
    assert opts.atol is None
    assert opts.max_levels == 8
    assert opts.max_solver_steps == 64
    assert opts.max_bracket_steps == 64


@pytest.mark.feature("summary.containers")
def test_summary_options_stores_explicit_values():
    """spec 8.5, rev 51: explicit fields are stored as given."""
    opts = survjax.SummaryOptions(rtol=1e-6, atol=1e-9, max_levels=5, max_solver_steps=10, max_bracket_steps=20)
    assert float(opts.rtol) == pytest.approx(1e-6, rel=1e-6)
    assert float(opts.atol) == pytest.approx(1e-9, rel=1e-6)
    assert opts.max_levels == 5
    assert opts.max_solver_steps == 10
    assert opts.max_bracket_steps == 20


@pytest.mark.feature("summary.containers")
def test_summary_options_is_frozen():
    """spec 8.5, rev 51: "The container is frozen"."""
    opts = survjax.SummaryOptions()
    with pytest.raises(AttributeError):
        opts.rtol = 1e-3
    with pytest.raises(AttributeError):
        opts.max_levels = 3


@pytest.mark.feature("summary.containers")
def test_explicit_tolerances_are_the_pytree_leaves():
    """spec 8.5, rev 51: "Explicit tolerances are dynamic PyTree leaves; the three work limits are static metadata"."""
    leaves = jax.tree.leaves(survjax.SummaryOptions(rtol=1e-6, atol=1e-9))
    assert len(leaves) == 2
    got = sorted(float(np.asarray(x)) for x in leaves)
    np.testing.assert_allclose(got, [1e-9, 1e-6], rtol=1e-6)


@pytest.mark.feature("summary.containers")
def test_tolerance_values_do_not_change_the_tree_structure():
    """spec 8.5, rev 51: tolerances are dynamic leaves, so their values leave the structure unchanged."""
    a = jax.tree.structure(survjax.SummaryOptions(rtol=1e-6, atol=1e-9))
    b = jax.tree.structure(survjax.SummaryOptions(rtol=1e-7, atol=1e-11))
    assert a == b


@pytest.mark.feature("summary.containers")
@pytest.mark.parametrize("field", ["max_levels", "max_solver_steps", "max_bracket_steps"])
def test_work_limits_are_static_metadata(field):
    """spec 8.5, rev 51: "the three work limits are static metadata": they change the tree structure, not the leaves."""
    base = survjax.SummaryOptions(rtol=1e-6, atol=1e-9)
    other = survjax.SummaryOptions(rtol=1e-6, atol=1e-9, **{field: 3})
    assert jax.tree.structure(base) != jax.tree.structure(other)
    assert len(jax.tree.leaves(other)) == 2


@pytest.mark.feature("summary.containers")
def test_options_cross_jit_with_tolerances_traced_and_limits_static():
    """spec 8.5, rev 51: under `jit`, tolerance values reuse one trace and a changed work limit retraces."""
    traces = []

    @jax.jit
    def probe(opts):
        traces.append(opts.max_levels)
        assert isinstance(opts.max_levels, int)
        return opts.rtol * 2.0

    probe(survjax.SummaryOptions(rtol=f(1e-6), atol=f(1e-9)))
    probe(survjax.SummaryOptions(rtol=f(1e-7), atol=f(1e-10)))
    assert len(traces) == 1
    probe(survjax.SummaryOptions(rtol=f(1e-7), atol=f(1e-10), max_levels=4))
    assert traces == [8, 4]


# --------------------------------------------------------------- SummaryInfo


@pytest.mark.feature("summary.containers")
def test_status_is_an_int_enum_with_the_specified_codes():
    """spec 8.5, rev 51: `SummaryInfo.Status` is an `IntEnum` with `converged=0` ... `work_exhausted=5`."""
    status = survjax.SummaryInfo.Status
    assert issubclass(status, enum.IntEnum)
    assert {m.name: int(m) for m in status} == STATUS_CODES


@pytest.mark.feature("summary.containers")
def test_summary_info_construction_stores_its_fields():
    """spec 8.5, rev 51: "Construction stores these fields"."""
    parts = {
        "status": jnp.asarray([0, 2, 5], dtype=jnp.int32),
        "error_estimate": jnp.asarray(f([1e-12, np.nan, np.nan])),
        "num_levels": jnp.asarray([4, 0, 8], dtype=jnp.int32),
        "num_solver_steps": jnp.zeros(3, dtype=jnp.int32),
        "num_bracket_steps": jnp.zeros(3, dtype=jnp.int32),
    }
    info = survjax.SummaryInfo(**parts)
    for name in INFO_FIELDS:
        np.testing.assert_array_equal(np.asarray(getattr(info, name)), np.asarray(parts[name]), err_msg=name)


@pytest.mark.feature("summary.containers")
def test_summary_info_has_five_array_leaves():
    """spec 8.5, rev 51: "a frozen PyTree with all five fields as array leaves"."""
    leaves = jax.tree.leaves(_info())
    assert len(leaves) == 5
    assert all(np.shape(x) == (3,) for x in leaves)


@pytest.mark.feature("summary.containers")
def test_summary_info_round_trips_through_tree_map():
    """spec 8.5, rev 51: as a PyTree, `jax.tree.map` rebuilds a `SummaryInfo` with mapped fields."""
    info = _info()
    sliced = jax.tree.map(lambda x: x[:2], info)
    assert isinstance(sliced, survjax.SummaryInfo)
    for name in INFO_FIELDS:
        assert np.shape(getattr(sliced, name)) == (2,), name


@pytest.mark.feature("summary.containers")
def test_summary_info_passes_through_jit():
    """spec 8.5, rev 51: a `SummaryInfo` PyTree is a valid `jit` argument and output."""
    info = _info()
    out = jax.jit(lambda i: i)(info)
    assert isinstance(out, survjax.SummaryInfo)
    np.testing.assert_array_equal(np.asarray(out.num_levels), np.asarray(info.num_levels))


@pytest.mark.feature("summary.containers")
def test_summary_info_is_frozen():
    """spec 8.5, rev 51: `SummaryInfo` is frozen."""
    info = _info()
    with pytest.raises(AttributeError):
        info.status = jnp.ones(3, dtype=jnp.int32)
