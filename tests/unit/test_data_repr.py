"""The SurvivalData repr shows only public fields."""

import jax
import jax.numpy as jnp

from chronox_jax import SurvivalData

PUBLIC_FIELDS = ("exact=", "lower=", "upper=", "left_trunc=", "right_trunc=")


def test_eager_repr_names_public_fields_only() -> None:
    """Spec 3.2: the issue bitfield and static metadata are private."""
    text = repr(SurvivalData(jnp.array([2.0, 3.0]), [1, 0]))
    for name in PUBLIC_FIELDS:
        assert name in text
    assert "_issue_bits" not in text
    assert "_may_contain" not in text


def test_traced_repr_omits_private_fields() -> None:
    """Spec 3.2: repr of data built under jit does not raise."""
    captured: list[str] = []

    @jax.jit
    def build(time: jax.Array) -> jax.Array:
        data = SurvivalData(time)
        captured.append(repr(data))
        return data.lower

    build(jnp.array([2.0, 3.0]))
    assert "lower=" in captured[0]
    assert "_issue_bits" not in captured[0]
    assert "_may_contain" not in captured[0]


def test_repr_change_keeps_six_leaves() -> None:
    """Spec 3.2: the private issue bitfield remains the sixth leaf."""
    data = SurvivalData(jnp.array([2.0, 3.0]), [1, 0])
    assert len(jax.tree_util.tree_leaves(data)) == 6
