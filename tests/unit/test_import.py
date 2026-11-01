"""Public package import smoke test."""


def test_public_package_imports() -> None:
    """Spec 10.2: the distribution imports as survjax."""
    import survjax

    assert set(survjax.__all__) == {
        "constraints",
        "Exponential",
        "Weibull",
        "ProportionalHazards",
        "SurvivalData",
        "InvalidDataError",
        "InvalidParameterError",
        "SurvivalDistribution",
        "register_family",
        "param",
        "distribution",
        "static",
        "concordance_index",
        "concordance_index_with_info",
    }
    assert survjax.SurvivalData is not None
    assert issubclass(survjax.InvalidDataError, ValueError)
    from survjax.diagnostics import InvalidParameterError

    assert survjax.InvalidParameterError is InvalidParameterError
    assert issubclass(InvalidParameterError, ValueError)


def test_likelihood_is_only_available_as_a_method() -> None:
    """Spec 7.1/10.2/12.1, rev 35: evaluator imports cannot restore the export."""
    import importlib

    import survjax

    assert not hasattr(survjax, "log_likelihood")
    evaluator = importlib.import_module("survjax.likelihood")
    assert not hasattr(evaluator, "log_likelihood")
    survjax.Exponential(rate=1.0).log_likelihood(time=1.0, event=True)
    assert not hasattr(survjax, "log_likelihood")


def test_ph_public_imports_are_the_same_registered_family() -> None:
    """Spec 4.4/10.2: root and distribution exports expose the implemented PH type."""
    import survjax
    import survjax.distributions as distributions
    from survjax.distributions.wrappers import ProportionalHazards

    assert survjax.ProportionalHazards is ProportionalHazards
    assert distributions.ProportionalHazards is ProportionalHazards
    assert set(distributions.__all__) == {
        "Exponential",
        "Weibull",
        "ProportionalHazards",
        "SurvivalDistribution",
        "register_family",
        "param",
        "distribution",
        "static",
    }
