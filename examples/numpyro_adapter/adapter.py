"""Discussion prototype for issue #1; not part of the survjax package API."""

from collections.abc import Iterator
from contextlib import contextmanager

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints

import survjax


@contextmanager
def prototype_compatibility() -> Iterator[None]:
    """Temporarily supply ndim; a production change needs spec-test coordination.

    Use only around this standalone probe, not concurrently with other work.
    The package class is restored on exit, including after an exception.
    """
    added = not hasattr(survjax.SurvivalData, "ndim")
    if added:
        survjax.SurvivalData.ndim = property(lambda self: len(self.shape))
    try:
        yield
    finally:
        if added:
            del survjax.SurvivalData.ndim


class _Parameters(constraints.Constraint):
    def __call__(self, value):
        return ~value.parameter_issues().invalid_entries

    def tree_flatten(self):
        return (), ((), {})


class _Observations(constraints.Constraint):
    def __init__(self, survival):
        self.survival = survival

    def __call__(self, value):
        if isinstance(value, survjax.SurvivalData):
            return ~value.issues.invalid_rows
        return (value >= 0) & jnp.isfinite(value)

    def tree_flatten(self):
        return (self.survival,), (("survival",), {})

    def feasible_like(self, value):
        if not isinstance(value, survjax.SurvivalData):
            return jnp.broadcast_to(
                jax.lax.stop_gradient(self.survival.reference_time()),
                jnp.broadcast_shapes(jnp.shape(value), self.survival.batch_shape),
            )
        # Preserve the original possible kinds, including on right-only data.
        if value._may_contain[0] or value.lower.size == 0:
            time = jax.lax.stop_gradient(self.survival.reference_time())
            return survjax.SurvivalData(time)
        if value._may_contain[1]:
            return survjax.SurvivalData(jnp.zeros((), value.lower.dtype), event=False)
        raise NotImplementedError("Prototype masking supports exact/right observations")


def _select_records(keep, value, dummy):
    leaves, flags = value.tree_flatten()
    replacements, _ = dummy.tree_flatten()
    selected = tuple(
        jnp.where(keep, original, replacement)
        for original, replacement in zip(leaves, replacements, strict=True)
    )
    # feasible_like deliberately introduces no kinds absent from value.
    return survjax.SurvivalData.tree_unflatten(flags, selected)


class _Wrappers:
    def mask(self, mask):
        return _Masked(self, mask)

    def expand(self, batch_shape):
        batch_shape = tuple(batch_shape)
        return self if batch_shape == self.batch_shape else _Expanded(self, batch_shape)


class _Expanded(_Wrappers, dist.ExpandedDistribution):
    pass


class _Masked(_Wrappers, dist.MaskedDistribution):
    def log_prob(self, value, intermediates=None):
        if not isinstance(value, survjax.SurvivalData) or isinstance(self._mask, bool):
            return super().log_prob(value, intermediates)
        dummy = self.base_dist.support.feasible_like(value)
        safe_value = _select_records(self._mask, value, dummy)
        return jnp.where(self._mask, self.base_dist.log_prob(safe_value), 0.0)


class SurvivalLikelihood(_Wrappers, dist.Distribution):
    """Score observation records or event-time arrays using one survival model.

    Args:
        survival: A survjax distribution; its batch axes are observation axes.
        validate_args: NumPyro's eager parameter/support validation switch.

    Sampling is intentionally unavailable until core event-time sampling lands.
    This prototype supports scalar events, plates and likelihood masks. Generic
    distribution wrappers and structured obs_mask imputation are not promised.
    """

    arg_constraints = {"survival": _Parameters()}
    pytree_data_fields = ("survival",)

    def __init__(
        self,
        survival: survjax.SurvivalDistribution,
        *,
        validate_args: bool | None = None,
    ) -> None:
        self.survival = survival
        super().__init__(survival.batch_shape, (), validate_args=validate_args)

    @property
    def support(self) -> constraints.Constraint:
        return _Observations(self.survival)

    def log_prob(self, value: survjax.SurvivalData | jax.Array) -> jax.Array:
        """Return unsummed contributions; structured records retain issue flags."""
        if self._validate_args:
            self._validate_sample(value)
        if isinstance(value, survjax.SurvivalData):
            return self.survival.log_likelihood(value)
        return self.survival.log_prob(value)

    def sample(self, key: jax.Array, sample_shape: tuple[int, ...] = ()) -> jax.Array:
        """Raise until survjax supports draws of underlying event times."""
        raise NotImplementedError("Core event-time sampling is not available yet")
