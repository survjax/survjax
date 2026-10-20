"""Public package import smoke test."""


def test_public_package_imports() -> None:
    """Spec 10.2: the distribution imports as survjax."""
    import survjax

    assert set(survjax.__all__) == {
        "constraints",
        "Exponential",
        "Weibull",
        "SurvivalData",
        "InvalidDataError",
        "InvalidParameterError",
        "SurvivalDistribution",
        "register_family",
        "param",
        "distribution",
        "static",
    }
    assert survjax.SurvivalData is not None
    assert issubclass(survjax.InvalidDataError, ValueError)
    from survjax.diagnostics import InvalidParameterError

    assert survjax.InvalidParameterError is InvalidParameterError
    assert issubclass(InvalidParameterError, ValueError)
