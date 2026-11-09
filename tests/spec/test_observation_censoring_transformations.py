"""observation.censoring: eager, `jit` and `vmap` agreement on left/interval rows (spec 3.2, 6.2, 7.1, 7.2, 12.1; rev 51).

12.1 (Transformations): "eager/`jit`/`vmap` agreement"; 7.2: pruning by
static metadata is "an optimization with identical results", and new values
with unchanged shapes, dtypes, tree structure and static configuration reuse
compilation. Expected values are hand-derived as in
``test_observation_censoring_values.py``: left ``log(-expm1(-H(c)))``,
interval ``-H(a) + log(-expm1(-(H(b)-H(a))))``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import NAN, assert_close, b, f, flag_data, ll

import survjax

FEATURE = pytest.mark.feature("observation.censoring")

KINDS = ["exact", "right", "left", "interval"]
TIME = [1.0, 2.0, 1.5, 0.5]
TIME2 = [NAN, NAN, NAN, 2.5]


def _expected_exponential(r):
    r = float(np.asarray(f(r)))
    return np.array(
        [
            np.log(r) - r,
            -2.0 * r,
            np.log(-np.expm1(-1.5 * r)),
            -0.5 * r + np.log(-np.expm1(-2.0 * r)),
        ]
    )


def _data():
    return flag_data(KINDS, TIME, TIME2)


@FEATURE
def test_prepared_data_under_jit_matches_eager():
    """spec 12.1, rev 51: data and distribution passed into `jit` give the eager contributions."""
    dist, data = survjax.Weibull(shape=f(1.4), scale=f(2.2)), _data()
    eager = ll(dist, data)
    jitted = np.asarray(jax.jit(lambda d, x: d.log_likelihood(x))(dist, data))
    assert_close(jitted, eager)
    assert np.isfinite(eager).all()


@FEATURE
def test_traced_flag_construction_inside_jit_matches_eager():
    """spec 3.2, 6.2 and 7.2, rev 51: flag-form construction from traced inputs (all metadata true) gives the same values as concrete data."""

    def run(rate, time, time2, right, left, interval):
        data = survjax.SurvivalData(
            time, right_censored=right, left_censored=left, interval_censored=interval, time2=time2
        )
        return survjax.Exponential(rate=rate).log_likelihood(data)

    value = jax.jit(run)(
        f(0.5),
        f(TIME),
        f(TIME2),
        b([k == "right" for k in KINDS]),
        b([k == "left" for k in KINDS]),
        b([k == "interval" for k in KINDS]),
    )
    assert_close(np.asarray(value), _expected_exponential(0.5))
    assert_close(np.asarray(value), ll(survjax.Exponential(rate=f(0.5)), _data()))


@FEATURE
def test_traced_bounds_construction_inside_jit_matches_eager():
    """spec 3.2, 6.2 and 7.2, rev 51: bounds-form construction from traced inputs gives the concrete-data values."""

    def run(rate, lower, upper, exact):
        data = survjax.SurvivalData.from_bounds(lower, upper, exact=exact)
        return survjax.Exponential(rate=rate).log_likelihood(data)

    value = jax.jit(run)(
        f(0.5), f([1.0, 2.0, 0.0, 0.5]), f([1.0, np.inf, 1.5, 2.5]), b([True, False, False, False])
    )
    assert_close(np.asarray(value), _expected_exponential(0.5))


@FEATURE
def test_concrete_metadata_without_left_rows_does_not_change_values():
    """spec 3.2 and 7.2, rev 51: data with only interval rows (no left kind in the metadata) and the same rows in a mixed batch agree.

    Pruning by metadata may differ between the two datasets but must give
    "identical results". Compared at the 12.1 tolerance rather than bitwise,
    since compiled code for different batches may round differently.
    """
    dist = survjax.Exponential(rate=f(0.5))
    only_interval = flag_data(["interval"], [0.5], [2.5])
    mixed = _data()
    assert_close(ll(dist, only_interval)[0], ll(dist, mixed)[3])
    only_left = flag_data(["left"], [1.5])
    assert_close(ll(dist, only_left)[0], ll(dist, mixed)[2])


@FEATURE
def test_vmap_over_parameter_draws():
    """spec 12.1, rev 51: `vmap` over rate draws equals per-draw hand values on the mixed rows."""
    data = _data()
    rates = f([0.5, 1.3, 2.0])
    out = np.asarray(jax.vmap(lambda r: survjax.Exponential(rate=r).log_likelihood(data))(rates))
    assert out.shape == (3, 4)
    assert_close(out, np.stack([_expected_exponential(r) for r in rates]))


@FEATURE
def test_vmap_over_datasets():
    """spec 3.2 and 12.1, rev 51: `vmap` over a leading data axis gives each dataset's contributions."""
    data = survjax.SurvivalData(
        f([[1.5, 0.5], [0.5, 1.5]]),
        left_censored=b([[True, False], [False, True]]),
        interval_censored=b([[False, True], [True, False]]),
        time2=f([[NAN, 2.5], [2.5, NAN]]),
    )
    out = np.asarray(jax.vmap(lambda d: survjax.Exponential(rate=f(0.5)).log_likelihood(d))(data))
    e = _expected_exponential(0.5)
    assert_close(out, [[e[2], e[3]], [e[3], e[2]]])


@FEATURE
def test_jit_of_vmap_over_draws_matches_eager_batch():
    """spec 5 and 12.1, rev 51: `jit(vmap)` over draws equals the eager `(D, 1)` batch."""
    data = _data()
    rates = f([0.5, 2.0])
    mapped = jax.jit(jax.vmap(lambda r: survjax.Exponential(rate=r).log_likelihood(data)))(rates)
    batch = ll(survjax.Exponential(rate=f([[0.5], [2.0]])), data)
    assert_close(np.asarray(mapped), batch)


@FEATURE
def test_indexed_censored_data_keeps_row_values():
    """spec 3.2, rev 51: `data[idx]` and `jax.tree.map` slicing select rows; contributions equal the parent's selected rows."""
    dist, data = survjax.Exponential(rate=f(0.5)), _data()
    full = ll(dist, data)
    assert_close(ll(dist, data[2:]), full[2:])
    assert_close(ll(dist, data[np.array([3, 0])]), full[[3, 0]])
    sliced = jax.tree.map(lambda leaf: leaf[1:3], data)
    assert_close(ll(dist, sliced), full[1:3])


@FEATURE
def test_new_censored_values_reuse_compilation():
    """spec 7.2, rev 51: new parameter and observation values with equal shapes, dtypes and metadata reuse compilation."""
    traces = []

    @jax.jit
    def run(dist, data):
        traces.append(None)
        return dist.log_likelihood(data)

    run(survjax.Exponential(rate=f(0.5)), flag_data(KINDS, TIME, TIME2))
    run(survjax.Exponential(rate=f(1.5)), flag_data(KINDS, [2.0, 1.0, 3.0, 1.0], [NAN, NAN, NAN, 4.0]))
    assert len(traces) == 1


@FEATURE
def test_traced_censored_construction_reuses_compilation():
    """spec 7.2, rev 51: repeated compiled construction of left/interval data from traced inputs does not retrace."""
    traces = []

    @jax.jit
    def run(rate, time, time2, left, interval):
        traces.append(None)
        data = survjax.SurvivalData(time, left_censored=left, interval_censored=interval, time2=time2)
        return survjax.Exponential(rate=rate).log_likelihood(data)

    run(f(0.5), f([1.5, 0.5]), f([NAN, 2.5]), b([True, False]), b([False, True]))
    run(f(2.0), f([0.5, 3.0]), f([1.0, NAN]), b([False, True]), b([True, False]))
    assert len(traces) == 1


@FEATURE
def test_censored_rows_compose_inside_one_compiled_objective():
    """spec 12.1 (Compiled composition), rev 51: construction, diagnostics and the likelihood with its gradient compose inside `jit`.

    Exponential summed objective over the four rows at ``r=0.5``: the
    gradient is ``1/r - 1`` (exact) ``- 2`` (right) ``+ 1.5/expm1(1.5 r)``
    (left) ``- 0.5 + 2/expm1(2 r)`` (interval), the per-kind derivatives of
    ``test_observation_censoring_derivatives.py``.
    """

    def run(rate, time, time2, right, left, interval):
        data = survjax.SurvivalData(
            time, right_censored=right, left_censored=left, interval_censored=interval, time2=time2
        )

        def loss(r):
            return jnp.sum(survjax.Exponential(rate=r).log_likelihood(data))

        return loss(rate), jax.grad(loss)(rate), data.issues.ok

    value, grad, ok = jax.jit(run)(
        f(0.5),
        f(TIME),
        f(TIME2),
        b([k == "right" for k in KINDS]),
        b([k == "left" for k in KINDS]),
        b([k == "interval" for k in KINDS]),
    )
    assert bool(ok)
    assert_close(value, _expected_exponential(0.5).sum())
    r = float(np.asarray(f(0.5)))
    expected_grad = (1 / r - 1) - 2.0 + 1.5 / np.expm1(1.5 * r) + (-0.5 + 2.0 / np.expm1(2.0 * r))
    assert_close(grad, expected_grad)
