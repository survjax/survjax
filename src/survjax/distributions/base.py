"""Flat frozen-family registration and construction (spec 4.1–4.2)."""

from dataclasses import MISSING, field, fields, is_dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax._dtypes import normalize_parameters


def param(component_ndim: int = 0) -> Any:
    """Declare a required array with leading batch and trailing component axes."""
    return field(metadata={"survjax_kind": "param", "component_ndim": component_ndim})


def static(*, default: Any = MISSING) -> Any:
    """Declare hashable configuration stored in PyTree auxiliary data."""
    return field(default=default, metadata={"survjax_kind": "static"})


class SurvivalDistribution:
    """Base for explicitly registered frozen dataclasses (spec 4.2).

    Parameters have leading batch axes and optional trailing component axes.
    Direct-method defaults and batch indexing are planned in later A4 steps.
    """

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Return the current leaves' batch shape, excluding component axes."""
        shapes = []
        for declared in fields(self):  # type: ignore[arg-type]
            if declared.metadata.get("survjax_kind") == "param":
                value = getattr(self, declared.name)
                count = declared.metadata["component_ndim"]
                shapes.append(value.shape[: value.ndim - count])
        return jnp.broadcast_shapes(*shapes)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Family primitive: log P(T>t), with time broadcast against batch axes."""
        raise NotImplementedError("families must implement log_survival")

    def reference_time(self) -> jax.Array:
        """Family primitive: positive finite reference time per batch entry."""
        raise NotImplementedError("families must implement reference_time")


def register_family[Family: SurvivalDistribution](cls: type[Family]) -> type[Family]:
    """Register a frozen dataclass family with keyword-only construction.

    Own parameters share a floating dtype and broadcast leading batch axes.
    Static fields are hashable configuration. Reconstruction stores leaves
    unchanged, bypassing initialization, conversion and broadcasting (spec 4.2).
    """
    if not issubclass(cls, SurvivalDistribution) or not is_dataclass(cls):
        raise TypeError("register_family requires a SurvivalDistribution dataclass")
    if not cls.__dataclass_params__.frozen:
        raise TypeError("registered families must be frozen dataclasses")
    declared_fields = fields(cls)
    parameters = tuple(
        declared
        for declared in declared_fields
        if declared.metadata.get("survjax_kind") == "param"
    )
    configuration = tuple(
        declared
        for declared in declared_fields
        if declared.metadata.get("survjax_kind") == "static"
    )
    original_init = cls.__init__

    def initialize(self: Family, *args: Any, **kwargs: Any) -> None:
        if args:
            raise TypeError("registered families require keyword arguments")
        original_init(self, **kwargs)
        values = normalize_parameters(
            {declared.name: getattr(self, declared.name) for declared in parameters},
            warning_stacklevel=3,
        )
        batch_shapes = []
        for declared in parameters:
            value = values[declared.name]
            count = declared.metadata["component_ndim"]
            if value.ndim < count:
                raise TypeError(f"{declared.name} has fewer axes than component_ndim")
            batch_shapes.append(value.shape[: value.ndim - count])
        try:
            batch_shape = jnp.broadcast_shapes(*batch_shapes)
        except ValueError as exc:
            raise TypeError("incompatible parameter batch shapes") from exc
        for declared in parameters:
            value = values[declared.name]
            count = declared.metadata["component_ndim"]
            component_shape = value.shape[value.ndim - count :]
            object.__setattr__(
                self,
                declared.name,
                jnp.broadcast_to(value, batch_shape + component_shape),
            )
        for declared in configuration:
            hash(getattr(self, declared.name))

    def flatten(self: Family) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = tuple(getattr(self, declared.name) for declared in parameters)
        auxiliary = tuple(getattr(self, declared.name) for declared in configuration)
        return children, auxiliary

    def unflatten(auxiliary: tuple[Any, ...], children: tuple[Any, ...]) -> Family:
        # spec 4.2: JAX may supply sentinel leaves, so do not normalize here.
        result = object.__new__(cls)
        for declared, value in zip(parameters, children, strict=True):
            object.__setattr__(result, declared.name, value)
        for declared, value in zip(configuration, auxiliary, strict=True):
            object.__setattr__(result, declared.name, value)
        return result

    cls.__init__ = initialize  # type: ignore[method-assign]
    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    return cls
