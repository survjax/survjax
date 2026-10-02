"""Survival distribution extension interface."""

from survjax.distributions.base import (
    SurvivalDistribution,
    param,
    register_family,
    static,
)

__all__ = ["SurvivalDistribution", "register_family", "param", "static"]
