"""Survival distribution extension interface."""

from survjax.distributions.base import (
    SurvivalDistribution,
    distribution,
    param,
    register_family,
    static,
)

__all__ = ["SurvivalDistribution", "register_family", "param", "distribution", "static"]
