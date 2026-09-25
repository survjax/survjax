"""Public package import smoke test."""


def test_public_package_imports() -> None:
    """Spec 10.2: the distribution imports as chronox_jax."""
    import chronox_jax as cx

    assert cx.__all__ == ["SurvivalData"]
    assert cx.SurvivalData is not None
