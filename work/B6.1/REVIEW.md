# B6.1 — Censoring/truncation likelihood documentation

Independent review: round 1/3

## Read first

1. PLAN.md and spec 2, 3.3, 6.1–6.2, 7.1 and 10.5.
2. docs/likelihood.md: current scope, conditional-contribution table, new
   windowed example and NaN explanation.
3. docs/technical-notes/likelihood-diagnostics.md: corrected mixed-batch recipe.
4. The five short linked-page scope corrections in the diff.

This is one documentation topic across seven pages, 183 changed lines. No
source, tests, tolerances, dependencies, public API or protected paths change.
It covers B2.1/B2.2's likelihood and diagnostics docs to-do; B3.3–B3.5 supply
the existing matrix/wrapper evidence. No Slice B completion claim is made.

## Invariants and examples

The contributions condition on (L,U] with positive mass Z. Exact rows use
log density minus log Z; censored rows use log mass of the intersection minus
log Z. Empty intersections and right censoring before entry are invalid data.
Zero model numerator with positive Z differs from empty observation data;
zero Z is invalid conditioning, even for an uninformative observation.
Ordinary predictions are not silently conditioned by likelihood data.

The new unit-rate Exponential example uses window (1,4] and one row per kind:
exact at 2, right at 3, left at 3 and interval (2,5]. Independent algebra is
log P(a<T<=b) = -a + log(1-exp(-(b-a))); exact density is exp(-2).
Both eager and jit values are asserted against these formulas in each dtype.
The rendered float64 values are approximately
[-0.94893082, -2.40760596, -0.09434428, -1.09434428]; float32 agrees at its
precision. The at-entry example asserts valid + uninformative diagnostics and
an exactly zero contribution. These demonstrate spec 2/3.3/7.1 semantics.

The diagnostic example asserts that only the invalid negative exact time is
NaN, with data mask [False, True, False, False] and no parameter or unexplained
NaNs. Its valid interval row is finite. The text directs remaining NaNs to
window mass and numerical/family-method investigation; it does not promise
complete numerical diagnosis or infer validity from a finite formula.

The original untruncated fitting example is retained and explicitly scoped to
that cohort. Concordance's narrower mode support remains unchanged. The AFT
scope statement reflects the shared evaluator and prior wrapper tests without
adding summary/sampling claims. No new R or speed comparison is asserted.

## Validation and rendered-document evaluation

- scripts/build_docs.sh run directly twice, both exit 0 and end with
  "Verified both dtypes". The second run follows a final prose correction
  to the numerical guide's NaN table. Strict Sphinx builds and all runnable
  notebooks succeeded in float32 and float64, including all new assertions.
  Existing Jupyter transport notices appeared; no Sphinx warning failure.
- Final build identity: base 44b33b0, correctly labelled dirty with this docs
  diff; completed 2026-11-12T13:58:23Z, state success. Generated files stay
  ignored under docs/_build.
- Inspected generated HTML for all seven changed pages in both dtype builds:
  current scope prose, new section anchors, conditional example outputs,
  at-entry zero, diagnostic masks and final NaN table are present. The
  supported workflow is general censored/truncated likelihood evaluation;
  metric and summary restrictions are kept distinct.
- Browser visual inspection was attempted, but the computer-use browser host
  was unavailable. HTML content/output inspection was completed instead;
  no browser layout or MathJax visual check is claimed. The temporary local
  HTTP server was stopped afterward.
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Formatting:
  277 files already formatted; Ruff clean; mypy no issues in 25 source files.
  Implementation tests: float64 1212 passed / 8 skipped (14.55s), float32
  1214 passed / 6 skipped (14.40s). Source doctest stages passed.
  Acceptance tests: float64 4654 passed / 46 skipped (35.29s), float32
  4597 passed / 93 skipped (35.07s), two existing NumPy warnings per mode
  at test_a11_ph_direct_methods.py:78.
- git diff --check: clean. No full non-fast gate, R regeneration or benchmark
  rerun. The final prose-only NaN-table edit was additionally checked in the
  second complete docs build; all executable example code was unchanged.

## Commits and remaining work

Step starting / previous-step final commit:
44b33b0731ebb6521dd36b402e61167ca8e9affd (B4.3.CLEANUP).
Original worktree base: cc0e30f.

Follow-up docs still need trusted static metadata, rev-62 argument-form
specialization, the historical baseline explanation, and the ten flexsurv
comparison cells/excluded conventions. Final Slice B evidence audit and
integration follow those docs steps. Benchmark artifacts remain owned by the
experiments project. No shared progress file or other step record was edited.
