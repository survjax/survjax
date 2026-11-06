"""Shared observation input forms for likelihoods and metrics."""

from jax.typing import ArrayLike

from survjax.data import SurvivalData


def _resolve_data(
    data: SurvivalData | None, time: ArrayLike | None, event: ArrayLike | None
) -> SurvivalData:
    """Resolve the two input forms without inspecting runtime values."""
    if data is not None:
        if not isinstance(data, SurvivalData):
            raise TypeError("data must be SurvivalData; use time=... and event=...")
        if time is not None or event is not None:
            raise TypeError("data cannot be combined with time or event")
        return data
    if time is None or event is None:
        raise TypeError("provide SurvivalData or both time=... and event=...")
    # spec 6.1/6.2: preserve raw-input evidence through the pure checked constructor.
    return SurvivalData(time, event)
