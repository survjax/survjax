"""A8: per-row likelihood and survival against committed synthetic `bc`-like R fixtures (spec 10.4, 11.1, 12.1, 12.2; rev 33).

Spec 12.2 puts "committed synthetic `bc`-like per-row R fixtures" in the
Slice A gate; 11.1 defines them: "a dataset simulated from the fitted `bc`
model with the same size, group sizes, time units and a similar censoring
pattern, with per-row log likelihoods and survival values from flexsurv and
survreg at the three parameter vectors above. Targets: `atol=1e-10`,
`rtol=1e-8` in float64." Float32 uses the 12.1 float32 targets.

The fixture is ``fixtures/bc_synthetic_weibull.json``; its schema is in
``fixtures/README.md`` (Q-039). Since rev 34, 11.1 says "The spec tester
defines the file name and schema; the generator writes that schema." The spec tester never generates fixture
values: a missing fixture fails these tests.

Model (11.1): row ``i`` in group ``g`` has ``k = shape``,
``b_i = good_scale * exp(beta_g)`` with ``beta_Good = 0``, and
``S_i(t) = exp(-(t/b_i)**k)``. Without truncation an exact row contributes
``log f_i(t_i)`` and a right-censored row ``log S_i(t_i)`` (spec 2).

The "fixture integrity" tests check only the committed file against the
spec (sizes, groups, parameter vectors, the two R sources agreeing); they do
not exercise survjax but stop a malformed fixture from passing silently.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from helpers import FLOAT, assert_close, b, f
from tolerances import TOLERANCES

import survjax

FIXTURE = "bc_synthetic_weibull.json"
VECTORS = ("fixed_a", "fixed_b", "r_optimum")
SOURCES = ("flexsurv", "survreg")
GROUPS = ("Good", "Medium", "Poor")

# Spec 11.1: (k, Good-group scale, beta_Medium, beta_Poor).
SPEC_VECTORS = {
    "fixed_a": (1.0, 10.0, -0.5, -1.0),
    "fixed_b": (1.5, 12.0, -0.6, -1.2),
    "r_optimum": (
        1.3796517717622814,
        11.422861823436245,
        -0.6135891674792526,
        -1.2122137266943085,
    ),
}


def _load(spec_fixture):
    """The fixture as NumPy float64 arrays, in the Q-039 schema of ``fixtures/README.md``."""
    raw = spec_fixture(FIXTURE)
    data = raw["data"]
    out = {
        "time": np.asarray(data["time"], dtype=np.float64),
        "event": np.asarray(data["event"]).astype(np.int64),
        "group": np.asarray(data["group"], dtype=str),
        "parameters": {
            name: tuple(
                float(raw["parameters"][name][key])
                for key in ("shape", "good_scale", "beta_medium", "beta_poor")
            )
            for name in VECTORS
        },
        "results": {
            name: {
                src: {
                    q: np.asarray(raw["results"][name][src][q], dtype=np.float64)
                    for q in ("loglik", "survival")
                }
                for src in SOURCES
            }
            for name in VECTORS
        },
        "grid_time": np.asarray(raw["grid"]["time"], dtype=np.float64),
        "grid_survival": {
            name: np.asarray(raw["grid"]["survival"][name], dtype=np.float64)
            for name in VECTORS
        },
    }
    return out


@pytest.fixture(scope="module")
def bc(spec_fixture):
    return _load(spec_fixture)


def _row_scales(bc, name):
    """Per-row Weibull scale ``b_i = good_scale * exp(beta_g)`` in float64."""
    _, good, beta_m, beta_p = bc["parameters"][name]
    beta = {"Good": 0.0, "Medium": beta_m, "Poor": beta_p}
    return good * np.exp(np.array([beta[g] for g in bc["group"]]))


def _dist(bc, name):
    k = bc["parameters"][name][0]
    return survjax.Weibull(shape=f(k), scale=f(_row_scales(bc, name)))


def _data(bc):
    return survjax.SurvivalData(f(bc["time"]), b(bc["event"] == 1))


# ------------------------------------------------------- fixture integrity


@pytest.mark.step("A8")
def test_fixture_has_bc_size_and_group_sizes(bc):
    """spec 11.1, rev 33: the synthetic data have the `bc` size (686) and group sizes 229, 229, 228."""
    assert bc["time"].shape == (686,)
    assert bc["event"].shape == (686,)
    counts = {g: int((bc["group"] == g).sum()) for g in GROUPS}
    assert counts == {"Good": 229, "Medium": 229, "Poor": 228}


@pytest.mark.step("A8")
def test_fixture_rows_are_exact_or_right_censored_with_positive_finite_times(bc):
    """spec 11.1/12.2, rev 33: Slice A data: events coded 0/1, times positive and finite, both kinds present."""
    assert set(np.unique(bc["event"]).tolist()) == {0, 1}
    assert np.isfinite(bc["time"]).all()
    assert (bc["time"] > 0).all()


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_fixture_parameter_vectors_are_those_of_section_11_1(bc, name):
    """spec 11.1, rev 33: the fixture's three vectors are exactly the fixed vectors and the R optimum of 11.1."""
    assert bc["parameters"][name] == SPEC_VECTORS[name]


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_fixture_sources_agree_with_each_other(bc, name):
    """spec 11.1, rev 33: flexsurv and survreg give the same per-row values (they share a route; 11.1 notes this is not independence)."""
    fs, sr = (bc["results"][name][src] for src in SOURCES)
    for q in ("loglik", "survival"):
        assert fs[q].shape == (686,)
        np.testing.assert_allclose(fs[q], sr[q], rtol=1e-8, atol=1e-10, err_msg=q)


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_fixture_grid_has_shape_141_by_3(bc, name):
    """spec 11.1, rev 34: the flexsurv curves are on "the demo grid (141 times, three groups)", 0 to 7 in steps of 0.05 from 0 to 7 in steps of 0.05, three groups."""
    np.testing.assert_allclose(bc["grid_time"], np.arange(141) * 0.05, rtol=0, atol=1e-12)
    assert bc["grid_survival"][name].shape == (141, 3)


# ------------------------------------------------------- per-row likelihood


@pytest.mark.step("A8")
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", VECTORS)
def test_per_row_log_likelihood_matches_r(bc, name, source):
    """spec 11.1/12.1, rev 33: `survjax.log_likelihood` per row equals the R per-row log likelihood (float64 `atol=1e-10, rtol=1e-8`)."""
    value = np.asarray(survjax.log_likelihood(_dist(bc, name), _data(bc)))
    assert value.shape == (686,)
    assert_close(value, bc["results"][name][source]["loglik"])


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_method_log_likelihood_matches_r(bc, name):
    """spec 7.1/11.1, rev 33: `dist.log_likelihood` per row equals the flexsurv values."""
    value = np.asarray(_dist(bc, name).log_likelihood(_data(bc)))
    assert_close(value, bc["results"][name]["flexsurv"]["loglik"])


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_raw_form_log_likelihood_matches_r(bc, name):
    """spec 6.1/11.1, rev 33: the raw `time=`, `event=` form gives the same per-row values."""
    value = np.asarray(
        survjax.log_likelihood(
            _dist(bc, name), time=f(bc["time"]), event=b(bc["event"] == 1)
        )
    )
    assert_close(value, bc["results"][name]["flexsurv"]["loglik"])


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_design_matrix_construction_with_from_log_matches_r(bc, name):
    """spec 4.1/11.1, rev 33: the 11.1 objective's construction, `k=exp(theta[0])`, `b_i=exp(X_i beta)`, through `Weibull.from_log` under `jit`.

    ``X`` has columns intercept, Medium, Poor; ``theta = (log k, log good_scale,
    beta_Medium, beta_Poor)``.
    """
    k, good, beta_m, beta_p = bc["parameters"][name]
    theta = f([np.log(k), np.log(good), beta_m, beta_p])
    x = f(
        np.stack(
            [
                np.ones(686),
                (bc["group"] == "Medium").astype(float),
                (bc["group"] == "Poor").astype(float),
            ],
            axis=1,
        )
    )
    data = _data(bc)

    @jax.jit
    def per_row(theta):
        dist = survjax.Weibull.from_log(log_shape=theta[0], log_scale=x @ theta[1:])
        return survjax.log_likelihood(dist, data)

    assert_close(per_row(jnp.asarray(theta)), bc["results"][name]["flexsurv"]["loglik"])


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_summed_log_likelihood_matches_r(bc, name):
    """spec 7.1/11.1, rev 33: callers sum the per-row values (`-values.sum()` is the objective); the sum matches R's sum.

    The 12.1 ``rtol`` applies to the total; ``atol`` is scaled by the row
    count, since 686 rows each within ``atol`` can add up to ``686 * atol``.
    """
    tol = TOLERANCES[FLOAT]
    total = float(np.asarray(survjax.log_likelihood(_dist(bc, name), _data(bc))).sum(dtype=np.float64))
    expected = float(bc["results"][name]["flexsurv"]["loglik"].sum())
    np.testing.assert_allclose(total, expected, rtol=tol["rtol"], atol=686 * tol["atol"])


# ------------------------------------------------------- per-row survival and curves


@pytest.mark.step("A8")
@pytest.mark.parametrize("source", SOURCES)
@pytest.mark.parametrize("name", VECTORS)
def test_per_row_survival_matches_r(bc, name, source):
    """spec 8.1/11.1/12.1, rev 34: "survival values `S_i(t_i)` at each row's own time": `dist.survival(t_i)` per row (paired batch) equals R."""
    value = np.asarray(_dist(bc, name).survival(f(bc["time"])))
    assert value.shape == (686,)
    assert_close(value, bc["results"][name][source]["survival"])


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_per_row_log_survival_matches_right_censored_contributions(bc, name):
    """spec 2/11.1, rev 33: on right-censored rows the R contribution is `log S_i(t_i)`; `log_survival` matches it."""
    censored = bc["event"] == 0
    value = np.asarray(_dist(bc, name).log_survival(f(bc["time"])))
    assert_close(value[censored], bc["results"][name]["flexsurv"]["loglik"][censored])


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_per_row_log_prob_matches_exact_contributions(bc, name):
    """spec 2/11.1, rev 33: on exact rows the R contribution is `log f_i(t_i)`; `log_prob` matches it."""
    exact = bc["event"] == 1
    value = np.asarray(_dist(bc, name).log_prob(f(bc["time"])))
    assert_close(value[exact], bc["results"][name]["flexsurv"]["loglik"][exact])


@pytest.mark.step("A8")
@pytest.mark.parametrize("name", VECTORS)
def test_group_survival_curves_match_r(bc, name):
    """spec 5/11.1, rev 34: `survival_grid` on the 141-point demo grid with a three-group batch gives `(141, 3)` equal to R, with `S(0)=1`."""
    k, good, beta_m, beta_p = bc["parameters"][name]
    scales = good * np.exp(np.array([0.0, beta_m, beta_p]))
    dist = survjax.Weibull(shape=f(k), scale=f(scales))
    curves = np.asarray(dist.survival_grid(f(bc["grid_time"])))
    assert curves.shape == (141, 3)
    np.testing.assert_array_equal(curves[0], 1.0)
    assert_close(curves, bc["grid_survival"][name])
