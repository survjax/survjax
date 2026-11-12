# B6.3 — Document flexsurv comparison coverage

Step starting commit: 2a8dec18198d66322fc5be489a00064d2e6d2f01
Previous step: [B6.2](../B6.2/PLAN.md), same commit.
Original worktree base: cc0e30f. Cumulative gate: A12, existing features.

One docs topic: the scope of the committed Slice B flexsurv evidence. Cover
the docs to-do in [B3.1](../B3.1/PLAN.md) and [B3.2](../B3.2/PLAN.md), using
their generator, fixture README, reference/acceptance tests and
[B3.3 matrix audit](../B3.3/REVIEW.md).

Expand docs/technical-notes/verification.md with the ten comparable cells,
their endpoint restrictions, synthetic provenance and numerical limits.
Explain the six excluded comparison categories and link to independent
mathematical coverage. Distinguish value-only float64 R comparisons from
float32 and derivative evidence. Add a downloadable fixture README. Correct
the page's stale opening assertion that these likelihood modes return NaN.
Add a short link from docs/likelihood.md to the comparison scope.

Spec sections 2, 10.4–10.5, 12.1–12.2. No source, tests, fixtures, public API,
dependency, tolerance or protected-path changes; no R regeneration or new
benchmark. Target below 150 changed documentation lines. No new standalone
tests or executable examples: the committed reference tests demonstrate the
claims, and the existing notebook examples remain unchanged.

Run the existing Slice B implementation and acceptance fixture tests in full
in both dtype modes, scripts/check.sh --fast --through A12, and
scripts/build_docs.sh directly. Inspect generated HTML for both changed pages,
including the matrix, exclusions, download and cross-links. Commit the step
handoff, run independent review and open the reviewed diff, then stop.

## Docs to-do and remaining work

This closes the flexsurv coverage item deferred by B6.1/B6.2. The final Slice B
evidence audit and maintainer integration remain separate steps; this docs
change makes no full Slice B completion claim. Benchmarks stay in experiments.
