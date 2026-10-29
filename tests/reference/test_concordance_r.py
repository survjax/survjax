"""Harrell-C values and exact counts from pinned synthetic R runs (spec 9.2)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


@pytest.mark.parametrize("case_index", (0, 1), ids=("heavy-ties", "untied"))
@pytest.mark.parametrize("prepared", (False, True), ids=("raw", "prepared"))
def test_concordance_values_counts_and_masks_match_r(
    r_fixture, x64_enabled, case_index, prepared
):
    """Spec 6.1/9.1–9.2/12.1: each draw matches R under eager/jit/vmap."""
    case = r_fixture("concordance")["cases"][case_index]
    dtype = jnp.float64 if x64_enabled else jnp.float32
    time = jnp.asarray(case["time"], dtype=dtype)
    event = jnp.asarray(case["event"], dtype=bool)
    scores = jnp.asarray(case["scores"], dtype=dtype)
    data = survjax.SurvivalData(time, event)

    def evaluate(s):
        if prepared:
            return survjax.concordance_index_with_info(s, data)
        return survjax.concordance_index_with_info(s, time=time, event=event)

    for run in (evaluate, jax.jit(evaluate), jax.vmap(evaluate)):
        value, info = run(scores)
        assert value.shape == (3,)
        assert value.dtype == jnp.dtype(dtype)
        # Expectations are the serialized fit$concordance and fit$count from R.
        np.testing.assert_allclose(
            value,
            [r["concordance"] for r in case["r"]],
            atol=ATOL[x64_enabled],
            rtol=RTOL[x64_enabled],
        )
        for name, key in (
            ("concordant", "concordant"),
            ("discordant", "discordant"),
            ("tied_score", "tied_x"),
            ("comparable", None),
        ):
            totals = [
                r[key] if key else r["concordant"] + r["discordant"] + r["tied_x"]
                for r in case["r"]
            ]
            words = getattr(info, name)
            np.testing.assert_array_equal(words.hi, np.zeros(3, np.uint32))
            np.testing.assert_array_equal(words.lo, totals)
        for mask in (info.invalid, info.unsupported, info.no_comparable_pairs):
            np.testing.assert_array_equal(mask, np.zeros(3, bool))
