"""Flat frozen-family registration and construction (spec 4.1–4.2)."""

from dataclasses import MISSING, field, fields, is_dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax._dtypes import normalize_parameters


def param(component_ndim: int = 0) -> Any:
    """Declare a required numerical parameter in a family dataclass.

    Use ``rate: jax.Array = survjax.param()`` for a scalar per batch entry.
    For a vector per entry, use ``param(component_ndim=1)``: an input of
    shape ``(N, K)`` then has batch shape ``(N,)`` and K components.
    Component axes remain trailing and do not participate in batch broadcasting.

    ``component_ndim`` must be a non-negative Python int. Other types,
    including bool, raise TypeError; negative values raise ValueError.
    The returned dataclass field has no default.
    """
    if not isinstance(component_ndim, int) or isinstance(component_ndim, bool):
        raise TypeError("component_ndim must be a non-negative int, not bool")
    if component_ndim < 0:
        raise ValueError("component_ndim must be non-negative")
    return field(metadata={"survjax_kind": "param", "component_ndim": component_ndim})


def static(*, default: Any = MISSING) -> Any:
    """Declare family configuration that stays constant during JAX tracing.

    For example, ``method: str = survjax.static(default="sum")`` selects a
    method without treating its name as a numerical parameter. Changing the
    configuration may cause JAX to compile a new version of a function.

    ``default`` is optional; omit it to require the constructor argument.
    Configuration values must be hashable, such as strings or tuples of
    numbers. Construction raises TypeError naming the field otherwise.
    Returns a dataclass field for use with ``register_family``.
    """
    return field(default=default, metadata={"survjax_kind": "static"})


class SurvivalDistribution:
    """Base class for custom survival distributions.

    Define a frozen dataclass subclass, mark its fields with ``param()`` or
    ``static()``, then apply ``register_family`` above the dataclass decorator.
    Implement ``log_survival(t)`` and ``reference_time()`` in the subclass.

    Parameters describe one distribution per batch entry. Their leading axes
    broadcast at construction; any component axes declared with ``param``
    remain trailing. Use ``batch_shape`` for the batch dimensions: ``shape``
    is available as a parameter name and has no special meaning in the base.
    """

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch dimensions as a tuple of ints, excluding parameter components.

        A scalar distribution returns ``()``. A parameter of shape ``(N, K)``
        with one component axis contributes batch shape ``(N,)``. The property
        follows current parameter shapes after tree mapping and reports the
        per-entry batch shape inside ``jax.vmap``.
        """
        shapes = []
        for declared in fields(self):  # type: ignore[arg-type]
            if declared.metadata.get("survjax_kind") == "param":
                value = getattr(self, declared.name)
                count = declared.metadata["component_ndim"]
                shapes.append(value.shape[: value.ndim - count])
        return jnp.broadcast_shapes(*shapes)

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return log P(T > t); custom families must implement this method.

        ``t`` is a scalar or array of times. The result has the broadcast shape
        of ``t`` and ``batch_shape``, with no component axes. Equal-length time
        and parameter vectors pair entries; use singleton axes for all pairs.
        This is pointwise evaluation, not a time-first shared-grid operation.

        The family implementation must handle zero and positive infinity and
        retain precision near log survival zero. This base implementation
        raises NotImplementedError.
        """
        raise NotImplementedError("families must implement log_survival")

    def reference_time(self) -> jax.Array:
        """Return a positive finite evaluation time for each batch entry.

        Custom families must implement this hook, usually using a natural time
        scale of the distribution. Return a scalar or array broadcastable to
        ``batch_shape``. Choose a time in the distribution's smooth interior
        when its parameters are valid, so numerical routines have a safe point
        to evaluate the family. This base implementation raises NotImplementedError.
        """
        raise NotImplementedError("families must implement reference_time")


def register_family[Family: SurvivalDistribution](cls: type[Family]) -> type[Family]:
    """Make a custom family usable with JAX transformations.

    Apply above ``@dataclasses.dataclass(frozen=True)`` on a subclass of
    ``SurvivalDistribution``. Each field must use ``param()`` or ``static()``;
    subclasses must have their own dataclass-generated initializer. Returns
    the same class, registered as a PyTree with numerical parameter leaves.

    Instances accept keyword arguments only. Parameters are required, share
    one floating dtype (at least float32), and broadcast their leading batch
    axes while preserving trailing components. Values are stored without
    parameter-domain checks. There is no constructor ``dtype=`` argument.
    Float64 array inputs warn at the call site when JAX x64 is disabled.

    Registration raises TypeError for invalid class or field declarations.
    Construction raises TypeError for nonreal parameters, insufficient axes,
    incompatible batch shapes or unhashable static configuration. JAX tree
    reconstruction preserves supplied leaves without calling the constructor.
    """
    if not issubclass(cls, SurvivalDistribution) or not is_dataclass(cls):
        raise TypeError("register_family requires a SurvivalDistribution dataclass")
    if not cls.__dataclass_params__.frozen:
        raise TypeError("registered families must be frozen dataclasses")
    if "__init__" not in cls.__dict__:
        raise TypeError(
            "decorate the subclass itself with @dataclass before registration"
        )
    declared_fields = fields(cls)
    for declared in declared_fields:
        # D-014: losing an unmarked field on reconstruction changes user values.
        if declared.metadata.get("survjax_kind") not in ("param", "static"):
            raise TypeError(
                f"{declared.name} must be declared with param() or static()"
            )
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
            try:
                hash(getattr(self, declared.name))
            except TypeError as exc:
                raise TypeError(
                    f"{declared.name}: static fields must be hashable"
                ) from exc

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
