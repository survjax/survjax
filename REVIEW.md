# Review: A6 docs rewrite — tutorial likelihood guide

## Fixes for review/slices/A6.docs.md (third follow-up)

- S1: aligned comments in the entry-point snippet now use Ruff's spacing;
  `ruff format --check .` reports 67 files already formatted.
- S2: restored the spec 7.1 Slice A triage recipe as "Telling the causes
  apart": one freshly constructed batch with a valid event, an invalid exact
  time, an interval row and a right-censored row; the recipe's masks show
  bad data in the second row, unsupported mode in the third and empty `other`. Limited
  explicitly to data as constructed; edited fields stay with D-026.
- N2: LR interpretation now describes the evidence in this sample only.
- Maintainer requests: the Weibull Newton step now has its vector formula
  (Hessian solve) in the text, and a short note says JAX derivatives make a
  hand-written Newton easy, with external-optimizer (Optimistix) examples
  planned for a later slice (spec 11.1, Slice C).
- N1 (size) not fixed: the docs diff against main is now about 375 changed
  lines. Needs the maintainer to accept or ask for a split (the Weibull
  section is the natural candidate, about 45 lines).

## Follow-up (second commit)

Reframed per review: the page is now "Log likelihoods" and is about how survjax
computes contributions; fitting is the illustration. New section "Two entry
points, two input forms": `dist.log_likelihood` vs `survjax.log_likelihood`,
positional `SurvivalData` vs keyword-only `time=`/`event=`, the call-form
rules, and a `raises-exception` cell for a raw positional array (`TypeError`).
"Fitting with JAX" is renamed "Using the likelihood to fit a model". Both
dtype builds pass again (`build succeeded.` twice).

## What to read, in order

1. `docs/likelihood.md` (rewritten). Read the rendered page:
   `open docs/_build/html-x64/likelihood.html`.
2. `docs/index.md`: one-sentence description of the guide.

## Structure of the guide

One synthetic 10-patient cohort (6 events, 103 months of follow-up) runs
through the whole page. Each section gives the mathematics first, then code:

1. Censored-data likelihood, $\ell_i = d_i \log f + (1-d_i)\log S$, and its
   exponential form $D\log\lambda - \lambda\sum t_i$.
2. Per-patient contributions from `log_likelihood`; caller-chosen reduction;
   raw `time=`/`event=` keywords and the functional form.
3. Closed-form exponential MLE $\hat\lambda = D/\sum t_i = 6/103$, plotted as
   a profile using `(m, 1)` parameter broadcasting.
4. Newton fit in $\eta = \log\lambda$ with `jax.grad`/`jax.hessian` under `jit`;
   standard error $1/\sqrt{D}$ from the observed information.
5. Weibull fit in $(\log k, \log b)$ with a 2×2 Newton step; likelihood-ratio
   statistic against the exponential.
6. NaN causes: invalid data (`data.issues`), invalid parameters
   (`parameter_issues()`), `raise_if_invalid()` at a host checkpoint, NaN vs
   `-inf`, and unsupported interval censoring.

## Where the numbers come from

- Row values at rate 0.05 and $\hat\lambda = 6/103$, $\operatorname{se} =
  1/\sqrt 6$: hand algebra shown in the prose.
- Weibull estimates (k ≈ 1.29, b ≈ 16.0) and LR ≈ 0.47 are quoted from the
  executed output; they have no closed form. They agree to the quoted digits
  in both dtype modes (prototype run and both builds).
- The old assert cells were removed by request; the page no longer checks
  values at build time, it only executes.

## Checks run

- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`: both
  `JAX_ENABLE_X64=0` and `=1` builds report `build succeeded.` (warnings fatal).
  The only `WARNING` lines are IPKernelApp TCP notices, which are not Sphinx
  warnings.
- Inspected the rendered x64 cell outputs; all match the prose.
- Not run: ruff, mypy, tests (no code changed).

## Unsure about

- Diff size: the rewrite is ~355 changed lines in `docs/likelihood.md`, above
  the 300-line guideline, because it replaces the whole page.
- The previous page's general NaN-triage recipe (mode mask built from fields)
  was dropped rather than rewritten, which also avoids the open D-026 wording.
  Say if you want it back in tutorial form.
- `parameter_issues()` prints as a raw dataclass repr, less readable than the
  `DataIssues` summary. That is a code matter, not changed here.
