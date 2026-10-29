"""Exact result words, floating ratios and pytrees (spec 3.3, 6.3, 9.1–9.2)."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

from survjax._concordance import _pair_counts
from survjax.metrics import ConcordanceCount, ConcordanceInfo, _concordance_result

NAMES = ("concordant", "discordant", "tied_score", "comparable")


def _counts(values):
    """Encode explicit Python integer inputs as uint32 words, independently."""
    return tuple(
        (
            jnp.asarray(value // 2**32, dtype=jnp.uint32),
            jnp.asarray(value % 2**32, dtype=jnp.uint32),
        )
        for value in values
    )


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ((3, 1, 2, 6), 2 / 3),
        ((0, 0, 6, 6), 0.5),
        ((6, 0, 0, 6), 1.0),
        ((0, 6, 0, 6), 0.0),
        ((2**32, 0, 0, 2**32), 1.0),
        ((2**32, 2**31, 0, 2**32 + 2**31), 2 / 3),
    ],
    ids=["half-credit", "all-tied", "perfect", "reversed", "high-only", "high-and-low"],
)
def test_ratio_and_exact_counts_match_hand_arithmetic(counts, expected, x64_enabled):
    """Spec 9.1–9.2: (C + T/2)/(C+D+T), including totals above uint32."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    value, info = _concordance_result(
        _counts(counts), jnp.asarray(False), jnp.asarray(False), dtype
    )
    # References above use the defining fraction; the final case is 2**32/(3*2**31).
    np.testing.assert_allclose(
        value, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )
    assert value.dtype == jnp.dtype(dtype)
    for name, total in zip(NAMES, counts, strict=True):
        count = getattr(info, name)
        assert isinstance(count, ConcordanceCount)
        assert count.hi.dtype == count.lo.dtype == jnp.dtype(jnp.uint32)
        assert count.hi.shape == count.lo.shape == ()
        assert int(count.hi) * 2**32 + int(count.lo) == total
    assert not bool(info.invalid | info.unsupported | info.no_comparable_pairs)


def test_ratio_uses_data_float_dtype_before_division(x64_enabled):
    """Spec 3.3: float64 data must not get an int32/float32 division result."""
    dtypes = (jnp.float32, jnp.float64) if x64_enabled else (jnp.float32,)
    for dtype in dtypes:
        value, _ = _concordance_result(
            _counts((1, 2, 0, 3)), jnp.asarray(False), jnp.asarray(False), dtype
        )
        assert value.dtype == jnp.dtype(dtype)
        is_float64 = dtype == jnp.float64
        np.testing.assert_allclose(
            value, 1 / 3, atol=ATOL[is_float64], rtol=RTOL[is_float64]
        )


@pytest.mark.parametrize(
    "invalid,unsupported", [(True, False), (False, True), (True, True)]
)
def test_failed_cohort_has_zero_counts_and_preserves_both_masks(invalid, unsupported):
    """Accepted D-027: failures are distinct from evaluated zero-pair cohorts."""
    value, info = _concordance_result(
        _counts((2**32, 2**31, 0, 2**32 + 2**31)),
        jnp.asarray(invalid),
        jnp.asarray(unsupported),
        jnp.float32,
    )
    assert np.isnan(value)
    assert bool(info.invalid) == invalid
    assert bool(info.unsupported) == unsupported
    assert not bool(info.no_comparable_pairs)
    for name in NAMES:
        count = getattr(info, name)
        assert int(count.hi) == int(count.lo) == 0


def test_evaluated_zero_pair_cohort_is_undefined():
    """Spec 9.2/D-027: zero comparable pairs give NaN with the undefined mask."""
    value, info = _concordance_result(
        _counts((0, 0, 0, 0)), jnp.asarray(False), jnp.asarray(False), jnp.float32
    )
    assert np.isnan(value)
    assert bool(info.no_comparable_pairs)
    assert not bool(info.invalid | info.unsupported)


def test_result_and_info_agree_under_jit_vmap_and_leading_batch_axes(x64_enabled):
    """Spec 6.3/9.1: independent results/masks retain their leading batch shape."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    counts = tuple(
        (jnp.zeros(3, dtype=jnp.uint32), jnp.array(low, dtype=jnp.uint32))
        for low in ([3, 0, 5], [1, 0, 0], [2, 0, 0], [6, 0, 5])
    )
    invalid = jnp.array([False, False, True])
    unsupported = jnp.zeros(3, dtype=bool)

    def run(c, i, u):
        return _concordance_result(c, i, u, dtype)

    for evaluate in (run, jax.jit(run), jax.vmap(run), jax.jit(jax.vmap(run))):
        value, info = evaluate(counts, invalid, unsupported)
        # First cohort: (3+2/2)/6=2/3; second has no pairs; third is invalid.
        np.testing.assert_allclose(
            value,
            [2 / 3, np.nan, np.nan],
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        assert isinstance(info, ConcordanceInfo)
        assert value.shape == (3,)
        np.testing.assert_array_equal(info.comparable.lo, [6, 0, 0])
        np.testing.assert_array_equal(info.invalid, [False, False, True])
        np.testing.assert_array_equal(info.no_comparable_pairs, [False, True, False])
        for leaf in jax.tree.leaves(info):
            assert leaf.shape == (3,)


def test_count_and_info_are_frozen_registered_pytrees():
    """Spec 6.3/D-027: named immutable fields survive pytree reconstruction."""
    _, info = _concordance_result(
        _counts((3, 1, 2, 6)), jnp.asarray(False), jnp.asarray(False), jnp.float32
    )
    with pytest.raises(FrozenInstanceError):
        info.comparable.hi = jnp.uint32(1)
    with pytest.raises(FrozenInstanceError):
        info.invalid = jnp.asarray(True)
    leaves, structure = jax.tree.flatten(info)
    rebuilt = jax.tree.unflatten(structure, leaves)
    assert isinstance(rebuilt, ConcordanceInfo)
    assert isinstance(rebuilt.comparable, ConcordanceCount)
    for original, restored in zip(leaves, jax.tree.leaves(rebuilt), strict=True):
        np.testing.assert_array_equal(original, restored)


def test_counting_and_result_layer_compose_under_eager_nan_debugging(x64_enabled):
    """Spec 7.2/9.2: a valid cohort composes without intermediate NaNs."""
    dtype = jnp.float64 if x64_enabled else jnp.float32
    scores = jnp.array([3.0, 1.0, 3.0, 4.0], dtype=dtype)
    time = jnp.array([1.0, 2.0, 3.0, 4.0], dtype=dtype)
    exact = jnp.array([True, False, True, True])

    def run(s, t, e):
        return _concordance_result(
            _pair_counts(s, t, e), jnp.asarray(False), jnp.asarray(False), dtype
        )

    # Pairs (0,1),(0,2),(0,3),(2,3) are C,T,D,D: (1+1/2)/4=3/8.
    with jax.debug_nans(True):
        for evaluate in (run, jax.jit(run)):
            value, info = evaluate(scores, time, exact)
            np.testing.assert_array_equal(value, 0.375)
            for name, total in zip(NAMES, (1, 2, 1, 4), strict=True):
                count = getattr(info, name)
                assert int(count.hi) == 0
                assert int(count.lo) == total
