"""Constraint declarations stay metadata (spec 4.1, 4.2, 4.7)."""

from dataclasses import dataclass

import jax
import numpy as np
import pytest

import survjax


def candidate(constraint, component_ndim=0):
    @dataclass(frozen=True)
    class Family(survjax.SurvivalDistribution):
        z: jax.Array = survjax.param(component_ndim, constraint=constraint)
        a: jax.Array = survjax.param(constraint=survjax.constraints.real)
        base: survjax.SurvivalDistribution = survjax.distribution()

    return Family


def test_registration_keeps_order_identity_and_numerical_leaves():
    """Spec 4.7: declarations do not run checks or alter construction/trees."""

    def forbidden(x):
        raise AssertionError("constraint evaluated during construction")

    family = survjax.register_family(candidate(forbidden))
    assert list(family.parameter_constraints) == ["z", "a"]
    assert family.parameter_constraints["z"] is forbidden
    with pytest.raises(TypeError):
        family.parameter_constraints["z"] = survjax.constraints.positive
    for build in (family, jax.jit(family)):
        dist = build(z=-1.0, a=0.0, base=survjax.Exponential(rate=2.0))
        np.testing.assert_array_equal(jax.tree.leaves(dist), [-1.0, 0.0, 2.0])


def test_registration_rejects_noncallable_or_excessive_event_axes():
    """Spec 4.7: declaration checks occur at registration, not param()."""
    for constraint in ("positive", 3, survjax.constraints.ordered):
        cls = candidate(constraint)
        with pytest.raises(TypeError):
            survjax.register_family(cls)
    survjax.register_family(candidate(survjax.constraints.ordered, 1))


def test_builtins_declare_positive_parameters():
    """Spec 4.1: Exponential rate and Weibull shape/scale are finite >0."""
    for cls, names in (
        (survjax.Exponential, ["rate"]),
        (survjax.Weibull, ["shape", "scale"]),
    ):
        assert list(cls.parameter_constraints) == names
        for constraint in cls.parameter_constraints.values():
            assert constraint is survjax.constraints.positive
