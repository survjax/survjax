"""Public likelihood input forms and delegation (spec 4.2/6.1/7.1–7.3)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from impl_tolerances import ATOL, RTOL

import survjax


@pytest.fixture
def evaluate():
    return lambda dist, *args, **kwargs: dist.log_likelihood(*args, **kwargs)


@pytest.mark.parametrize("form", ["raw", "prepared"])
def test_public_contributions_match_hand_derived_values(evaluate, form, x64_enabled):
    """Spec 6.1/7.1: both forms return each contribution, without a reduction."""
    time, event = jnp.array([1.0, 2.0, 3.0]), jnp.array([1, 0, 1])
    kwargs = (
        {"time": time, "event": event}
        if form == "raw"
        else {"data": survjax.SurvivalData(time, event)}
    )
    # Exponential log f=log(2)-2*t; log S=-2*t.
    expected = [np.log(2) - 2, -4, np.log(2) - 6]
    value = evaluate(survjax.Exponential(rate=2.0), **kwargs)
    assert value.shape == (3,)
    np.testing.assert_allclose(
        value, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
    )


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"time": [1.0]}, {"event": [1]}],
    ids=["neither-form", "missing-event", "missing-time"],
)
def test_incomplete_forms_raise_type_error(evaluate, kwargs):
    """Spec 6.1: all raw-form calls require both keywords."""
    with pytest.raises(TypeError, match="both time=.*event="):
        evaluate(survjax.Exponential(rate=1.0), **kwargs)


@pytest.mark.parametrize("keyword", ["time", "event"])
def test_prepared_data_rejects_either_raw_keyword(evaluate, keyword):
    """Spec 6.1: even one raw keyword conflicts with prepared data."""
    data = survjax.SurvivalData([1.0], [1])
    with pytest.raises(TypeError, match="cannot be combined"):
        evaluate(survjax.Exponential(rate=1.0), data, **{keyword: [1]})


def test_positional_raw_array_points_to_keyword_form(evaluate):
    """Spec 6.1: positional arrays never become implicit raw times."""
    with pytest.raises(TypeError, match="time=.*event="):
        evaluate(survjax.Exponential(rate=1.0), jnp.array([1.0]))


def test_raw_keywords_are_keyword_only(evaluate):
    """Spec 6.1: only data may be passed positionally."""
    with pytest.raises(TypeError):
        evaluate(survjax.Exponential(rate=1.0), None, [1.0], [1])


@pytest.mark.parametrize("keyword", ["time2", "left_trunc", "right_censored"])
def test_constructor_options_are_rejected(evaluate, keyword):
    """Spec 6.1: richer observations belong in SurvivalData."""
    with pytest.raises(TypeError):
        evaluate(survjax.Exponential(rate=1.0), time=[1.0], event=[1], **{keyword: [0]})


def test_incompatible_raw_shapes_raise_type_error(evaluate):
    """Spec 6.2: normalization rejects shapes that enlarge time.shape."""
    with pytest.raises(TypeError):
        evaluate(survjax.Exponential(rate=1.0), time=[1.0, 2.0], event=[1, 0, 1])


def test_static_misuse_is_rejected_during_tracing(evaluate):
    """Spec 6.1: tracing does not bypass static argument validation."""
    with pytest.raises(TypeError, match="both time=.*event="):
        jax.jit(lambda t: evaluate(survjax.Exponential(rate=1.0), time=t))(1.0)


@pytest.mark.parametrize("form", ["raw", "prepared"])
@pytest.mark.parametrize("mode", ["eager", "jit", "vmap"])
def test_invalid_raw_indicators_keep_evidence(evaluate, form, mode, x64_enabled):
    """Spec 6.2/7.1: finite normalized fields cannot erase an invalid raw code."""
    time, event = jnp.array([1.0, 2.0, 3.0]), jnp.array([1, 2, 0])

    def run(t, e):
        data = survjax.SurvivalData(t, e)
        kwargs = {"time": t, "event": e} if form == "raw" else {"data": data}
        return evaluate(
            survjax.Exponential(rate=2.0), **kwargs
        ), data.issues.invalid_rows

    if mode == "vmap":
        value, invalid = jax.vmap(run)(time[None, :], event[None, :])
        value, invalid = value[0], invalid[0]
    else:
        value, invalid = (jax.jit(run) if mode == "jit" else run)(time, event)
    np.testing.assert_array_equal(invalid, [False, True, False])
    np.testing.assert_array_equal(jnp.isnan(value), invalid)
    # For rate=2: exact at t=1 is log(2)-2, right at t=3 is -6.
    np.testing.assert_allclose(
        value[jnp.array([0, 2])],
        [np.log(2) - 2, -6],
        atol=ATOL[x64_enabled],
        rtol=RTOL[x64_enabled],
    )


@pytest.mark.parametrize("form", ["raw", "prepared"])
@pytest.mark.parametrize(
    "transform", [jax.jacfwd, jax.jacrev, jax.hessian], ids=["fwd", "rev", "hessian"]
)
def test_public_derivatives_match_analytical_anchors(
    evaluate, form, transform, x64_enabled
):
    """Spec 7.3/12.1: the method preserves first and second parameter derivatives."""
    time, event = jnp.array([1.0, 2.0, 3.0]), jnp.array([1, 0, 1])
    kwargs = (
        {"time": time, "event": event}
        if form == "raw"
        else {"data": survjax.SurvivalData(time, event)}
    )

    def total(rate):
        return evaluate(survjax.Exponential(rate=rate), **kwargs).sum()

    # Two exact events: total=2*log(r)-6*r, total'=2/r-6, total''=-2/r**2.
    expected = -0.5 if transform is jax.hessian else -5.0
    value = transform(total)(2.0)
    compiled = jax.jit(transform(total))(2.0)
    for result in (value, compiled):
        np.testing.assert_allclose(
            result, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )


@pytest.mark.parametrize("family", [survjax.Exponential, survjax.Weibull])
@pytest.mark.parametrize(
    "transform",
    [None, jax.jacfwd, jax.jacrev, jax.hessian],
    ids=["values", "fwd", "rev", "hessian"],
)
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_input_forms_agree_in_values_and_parameter_derivatives(
    family, transform, compiled, x64_enabled
):
    """Spec 6.1/12.1, rev 35: raw and prepared forms agree through order two."""
    time, event = jnp.array([0.5, 1.0, 1.5]), jnp.array([True, False, True])
    data = survjax.SurvivalData(time, event)
    exponential = family is survjax.Exponential
    parameters = jnp.array([2.0] if exponential else [1.5, 2.0])

    def rows(p, *, prepared):
        dist = family(rate=p[0]) if exponential else family(shape=p[0], scale=p[1])
        if prepared:
            return dist.log_likelihood(data)
        return dist.log_likelihood(time=time, event=event)

    def raw(p):
        return rows(p, prepared=False)

    def prepared(p):
        return rows(p, prepared=True)

    if transform is not None:
        raw, prepared = transform(raw), transform(prepared)
    if compiled:
        raw, prepared = jax.jit(raw), jax.jit(prepared)
    # This tests form equivalence; analytical and finite-difference tests above
    # and in test_likelihood_transformations.py independently check correctness.
    actual, expected = raw(parameters), prepared(parameters)
    assert jnp.all(jnp.isfinite(actual))
    if transform is None:
        np.testing.assert_array_equal(actual, expected)
    else:
        np.testing.assert_allclose(
            actual, expected, atol=ATOL[x64_enabled], rtol=RTOL[x64_enabled]
        )
