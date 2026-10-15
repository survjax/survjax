"""Frozen-family registration and construction (spec 4.1–4.2)."""

from dataclasses import MISSING, field, fields, is_dataclass
from typing import Any, ClassVar, Self

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from survjax._dtypes import normalize_parameters
from survjax.distributions import _grid_methods, _survival_methods


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


def distribution() -> Any:
    """Declare a required nested distribution field in a frozen family.

    A scalar constant-hazard baseline can be shared by two parameter sets.
    These custom families use the extension hooks already available; the
    example evaluates at a positive time with valid positive parameters.

    >>> from dataclasses import dataclass
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import survjax
    >>> @survjax.register_family
    ... @dataclass(frozen=True)
    ... class ConstantHazard(survjax.SurvivalDistribution):
    ...     rate: jax.Array = survjax.param()
    ...     def log_survival(self, t):
    ...         return -self.rate * jnp.asarray(t)
    ...     def reference_time(self):
    ...         return 1.0 / self.rate
    >>> @survjax.register_family
    ... @dataclass(frozen=True)
    ... class ScaledHazard(survjax.SurvivalDistribution):
    ...     base: survjax.SurvivalDistribution = survjax.distribution()
    ...     factor: jax.Array = survjax.param()
    ...     def log_survival(self, t):
    ...         return self.factor * self.base.log_survival(t)
    ...     def reference_time(self):
    ...         return self.base.reference_time() / self.factor
    >>> baseline = ConstantHazard(rate=2.0)
    >>> scaled = ScaledHazard(base=baseline, factor=[1.0, 3.0])
    >>> scaled.batch_shape
    (2,)
    >>> scaled.base.rate.tolist()
    [2.0, 2.0]
    >>> scaled.log_survival(1.0).tolist()
    [-2.0, -6.0]
    >>> scaled[1].log_survival(1.0).item()
    -6.0
    >>> baseline.batch_shape
    ()

    The nested family's batch axes broadcast with the outer parameters.
    Its parameter dtypes, component axes and static configuration are retained.
    This helper takes no arguments and returns a dataclass field with no default.
    """
    return field(metadata={"survjax_kind": "distribution"})


class SurvivalDistribution:
    """Base class for survival distributions, built-in and custom.

    A family describes the distribution of an event time T > 0 through its
    log survival function, log S(t) = log P(T > t). From it the base derives
    every other quantity: ``survival``, ``cumulative_hazard`` (H = -log S),
    ``hazard`` and ``log_hazard`` (h = dH/dt, by automatic differentiation),
    ``log_prob`` (the log density, log h + log S) and ``log_cdf``
    (log(1 - S)). Each of these pointwise methods has a ``_grid`` twin that
    evaluates one shared vector of times against the whole batch. Built-in
    families such as ``Weibull`` override the derived methods with closed
    forms.

    To write a family, define a frozen dataclass subclass, mark its fields
    with ``param()``, ``static()`` or ``distribution()``, apply
    ``register_family`` above the dataclass decorator, and implement
    ``log_survival(t)`` and ``reference_time()``.

    Parameters describe one distribution per batch entry. Their leading axes
    broadcast at construction; any component axes declared with ``param``
    remain trailing. Use ``batch_shape`` for the batch dimensions: ``shape``
    is available as a parameter name and has no special meaning in the base.

    Every method accepts any real time and never raises for a value. Times
    before the origin (t < 0) give S = 1 and the matching constants; NaN
    times give NaN; t = 0 and t = +inf give the family's limits, with
    finite parameter derivatives where the limit is finite and smooth.
    """

    _survjax_registered: ClassVar[bool] = False
    survival = _survival_methods.survival
    log_cdf = _survival_methods.log_cdf
    cumulative_hazard = _survival_methods.cumulative_hazard
    log_prob = _survival_methods.log_prob
    log_hazard = _survival_methods.log_hazard
    hazard = _survival_methods.hazard
    survival_grid = _grid_methods.survival_grid
    log_survival_grid = _grid_methods.log_survival_grid
    log_prob_grid = _grid_methods.log_prob_grid
    log_cdf_grid = _grid_methods.log_cdf_grid
    cumulative_hazard_grid = _grid_methods.cumulative_hazard_grid
    log_hazard_grid = _grid_methods.log_hazard_grid
    hazard_grid = _grid_methods.hazard_grid

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Batch dimensions as a tuple of ints, excluding parameter components.

        A scalar distribution returns ``()``. A parameter of shape ``(N, K)``
        with one component axis contributes batch shape ``(N,)``. The property
        follows current parameter shapes after tree mapping and reports the
        per-entry batch shape inside ``jax.vmap``. Nested distributions
        contribute their current batch shapes too.
        """
        shapes = []
        for declared in fields(self):  # type: ignore[arg-type]
            if declared.metadata.get("survjax_kind") == "param":
                value = getattr(self, declared.name)
                count = declared.metadata["component_ndim"]
                shapes.append(value.shape[: value.ndim - count])
            elif declared.metadata.get("survjax_kind") == "distribution":
                shapes.append(getattr(self, declared.name).batch_shape)
        return jnp.broadcast_shapes(*shapes)

    def __getitem__(self, index: Any) -> Self:
        """Select batch entries, preserving trailing parameter components.

        Integers, slices, index arrays and tuples follow JAX indexing rules.
        Ellipsis spans batch axes only; None inserts a batch axis. Static
        configuration and nested family types are preserved. A traced integer
        index works under jit; dynamically sized selections follow JAX's
        restrictions. Returns a reconstructed distribution, without construction.
        """
        rank = len(self.batch_shape)
        indices = index if isinstance(index, tuple) else (index,)
        # spec 4.1: every own/nested leaf has exactly rank leading batch axes.
        return jax.tree.map(
            lambda value: value[indices + (slice(None),) * (value.ndim - rank)],
            self,
        )

    def log_survival(self, t: ArrayLike) -> jax.Array:
        """Return the log survival probability, log S(t) = log P(T > t).

        Every family must implement this method; all other methods derive
        from it. Your implementation is only called with times in
        [0, +inf]: survjax answers negative and NaN times itself. It must
        return the right limits at t = 0 (usually 0) and t = +inf (usually
        -inf), and keep relative precision when log S is close to 0, for
        example by computing -H(t) directly rather than log(S(t)).

        If the limit at +inf is finite and depends on parameters, as in a
        cure model, substitute a safe time before any operation that would
        meet infinity, so that the limit's parameter derivatives stay
        finite. survjax handles the usual -inf limit itself.

        Args:
            t: Times, broadcast against ``batch_shape``. Times and batch
                pair element by element (trailing-axis broadcasting); use
                ``log_survival_grid`` to evaluate every time for every entry.

        Returns:
            Log survival with the broadcast shape of ``t`` and
            ``batch_shape``, without component axes.

        Raises:
            NotImplementedError: In the base class.
            TypeError: For nonreal times or times that do not broadcast.
        """
        raise NotImplementedError("families must implement log_survival")

    def reference_time(self) -> jax.Array:
        """Return a typical, well-behaved time for each distribution.

        Every family must implement this method. survjax evaluates the family
        at this time instead of at times it cannot use directly (negative,
        NaN or, in derived methods, infinite times), so that those entries
        give clean values and gradients; the value at the reference time is
        then discarded. A natural time scale such as the mean, the median or
        a scale parameter is a good choice.

        Returns:
            An array of exactly ``batch_shape`` in the parameter dtype, with
            positive finite times in the smooth interior of the distribution
            when the parameters are valid (anything for invalid ones).

        Raises:
            NotImplementedError: In the base class.
        """
        raise NotImplementedError("families must implement reference_time")


def register_family[Family: SurvivalDistribution](cls: type[Family]) -> type[Family]:
    """Make a custom family usable with JAX transformations.

    Apply above ``@dataclasses.dataclass(frozen=True)`` on a subclass of
    ``SurvivalDistribution``. Each field must use ``param()``, ``static()``
    or ``distribution()``;
    subclasses must have their own dataclass-generated initializer. Returns
    the same class, registered as a PyTree with numerical parameter leaves.

    Instances accept keyword arguments only. Parameters are required, share
    one floating dtype (at least float32), and broadcast their leading batch
    axes while preserving trailing components. Values are stored without
    parameter-domain checks. There is no constructor ``dtype=`` argument.
    Float64 array inputs warn at the call site when JAX x64 is disabled.
    Nested distributions broadcast with own parameters, preserving their dtypes.

    Registration raises TypeError for invalid class or field declarations.
    Construction raises TypeError for nonreal parameters, insufficient axes,
    incompatible batch shapes, unregistered nested distributions or unhashable
    static configuration. JAX tree
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
        if declared.metadata.get("survjax_kind") not in (
            "param",
            "static",
            "distribution",
        ):
            raise TypeError(
                f"{declared.name} must be declared with "
                "param() or static() or distribution()"
            )
    parameters = tuple(
        declared
        for declared in declared_fields
        if declared.metadata.get("survjax_kind") == "param"
    )
    nested = tuple(
        declared
        for declared in declared_fields
        if declared.metadata.get("survjax_kind") == "distribution"
    )
    dynamic = tuple(
        declared
        for declared in declared_fields
        if declared.metadata.get("survjax_kind") in ("param", "distribution")
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
        for declared in nested:
            value = getattr(self, declared.name)
            # spec 4.2: subclass registration is explicit, never inherited.
            registered = type(value).__dict__.get("_survjax_registered", False)
            if not isinstance(value, SurvivalDistribution) or not registered:
                raise TypeError(f"{declared.name} must be a registered distribution")
            batch_shapes.append(value.batch_shape)
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
        for declared in nested:
            value = getattr(self, declared.name)
            rank = len(value.batch_shape)
            # spec 3.3/4.2: broadcast nested leaves without dtype normalization.
            broadcast = jax.tree.map(
                lambda leaf, rank=rank: jnp.broadcast_to(
                    leaf, batch_shape + leaf.shape[rank:]
                ),
                value,
            )
            object.__setattr__(self, declared.name, broadcast)
        for declared in configuration:
            try:
                hash(getattr(self, declared.name))
            except TypeError as exc:
                raise TypeError(
                    f"{declared.name}: static fields must be hashable"
                ) from exc

    def flatten(self: Family) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
        children = tuple(getattr(self, declared.name) for declared in dynamic)
        auxiliary = tuple(getattr(self, declared.name) for declared in configuration)
        return children, auxiliary

    def unflatten(auxiliary: tuple[Any, ...], children: tuple[Any, ...]) -> Family:
        # spec 4.2: JAX may supply sentinel leaves, so do not normalize here.
        result = object.__new__(cls)
        for declared, value in zip(dynamic, children, strict=True):
            object.__setattr__(result, declared.name, value)
        for declared, value in zip(configuration, auxiliary, strict=True):
            object.__setattr__(result, declared.name, value)
        return result

    cls.__init__ = initialize  # type: ignore[method-assign]
    jax.tree_util.register_pytree_node(cls, flatten, unflatten)
    for name, negative_value in (
        ("log_survival", 0.0),
        ("survival", 1.0),
        ("log_cdf", -jnp.inf),
        ("cumulative_hazard", 0.0),
        ("log_prob", -jnp.inf),
        ("log_hazard", -jnp.inf),
        ("hazard", 0.0),
    ):
        setattr(
            cls,
            name,
            _survival_methods.apply_policy(
                getattr(cls, name),
                negative_value,
                density=name == "log_prob",
                primitive=name == "log_survival",
            ),
        )
    cls._survjax_registered = True
    return cls
