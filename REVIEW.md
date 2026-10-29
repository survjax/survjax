Independent review: round 2/3

# A10.d2 — Concordance in the editable playground

Authorized after the maintainer reviewed A10.d1. Branch A10.4, clean base
fdce5db. Scope: docs/playground.ipynb only, plus PLAN.md and REVIEW.md.
No source, tests, dependency, protected paths or spec changes. The separate
8757c0f planning packet is not part of this step and remains unchanged.

## Reading order

1. docs/playground.ipynb: corrected introduction, then the new concordance
   cells before Things to try; existing data/distribution cells are preserved.
2. PLAN.md A10.d2: requirements, single-runner validation and remaining steps.
3. docs/concordance.md: accepted background for the same hand-enumerated
   cohort and the semantics; unchanged by this step.

## Validation ownership — reviewer is the sole runner

The maintainer explicitly requested “only you or the reviewer run them not
both as it would be exactly the same no?” The previous reviewer repeated the
implementer's docs build because its role requires one. For this step the
implementer deliberately runs NO docs build, Ruff, mypy, pytest or full gate.
The reviewer owns the fresh scripts/build_docs.sh run in both modes, Ruff
and notebook validation. Checks are pending at this commit, assigned to
review rather than omitted or claimed passed. Please record exact outcomes,
including the hand-derived C/count assertions and unexecuted source state.
The implementer reviews source and staged changes only. The final A10 audit
will inspect the complete rendered site; no browser QA claim for this step.

## Invariants and requirement mapping

Spec 6.1/9.1–9.2: finite caller-supplied higher-risk scores; exact/right only,
no truncation; N is last and must match; leading score draws evaluated
separately. Default cohort is the guide's time [1,2,2,3], events [T,T,F,T],
scores [4,3,1,4]. Hand pairs: (1,2),(1,3),(2,3) concordant; (1,4) tied;
(2,4) discordant; (3,4) excluded. Counts (3,1,1,5), C=3.5/5=7/10.
Reversal gives (1,3,1,5), C=3/10; constant scores give C=1/2. Assertions
check these expected fractions and transformation agreement on the defaults;
users editing defaults are told to recalculate their expectations.
Counts are reconstructed as Python ints from hi/lo on the host. No gradient
is taken through C. The existing mixed cohort is valid but unsupported;
all-censored data are evaluated with no comparable pairs. NaNs retain their
correct distinct masks and failed counts are zero, without dropping rows.
Spec 10.5/12.1: synthetic, public-API, network/R-free examples execute during
the two-mode build; notebook code cells are committed with empty outputs
and null execution counts. Remove the existing saved import error and stale
Python 3.14.5 metadata; the locked environment determines the interpreter.
Spec 12.2: editable playground coverage, not the held-out example/final audit.

## Checks and limits at commit

Implemented six new cells (three prose/code pairs) before Things to try.
Notebook diff: 138 additions, 23 deletions, 161 changed lines. Existing data
and distribution cell sources/IDs are preserved. Intro and exploration
prompts changed as planned; all code outputs/counts are cleared, including
the old ModuleNotFoundError output, and stale Python version metadata removed.
The new value assertions use exactly the correctly rounded hand-derived
7/10, 3/10 and 1/2 in the result dtype; they apply only to this default cohort,
not as a general numerical tolerance guarantee.

Source/diff inspection completed. No automated checks run by implementer.
Automated checks are pending the reviewer as sole runner. No claim of a
passing build or lint before review. Do not rerun the full through-A10 gate:
source/tests have not changed since the passing 7c4f57b wheel gate.
No browser inspection performed; held-out synthetic example and final
evidence/rendered-site audit remain. Commit and round 1 follow.

## Round 1 result and B-1 follow-up

Round 1 on 93dbca8 returned fix-first: B-1 is the sole blocker, a missing
Ruff line wrap in the new mapped_c call (91 characters). No should-fixes,
nits or spec questions. The reviewer confirmed the hand counts, C fractions,
mask prose, cleared outputs/counts and preserved existing cells/IDs.

Reviewer-owned checks (not repeated by implementer):
- scripts/build_docs.sh: sandboxed attempt exited 2 (uv cache permissions);
  permitted rerun exited 0, both builds succeeded with no Sphinx warnings.
  Playground execution: 8.73s float32 / 18.59s float64. Existing IPKernelApp
  TCP-transport warnings only. The same assertions passed in both modes.
- Separate float32 nbconvert execution: C=0.7, counts (3,1,1,5), all three
  draw evaluators [0.7,0.3,0.5], unsupported NaN/True and zero-pair NaN/True.
- Ruff lint passed; format check failed for B-1. Plain check.sh --through A10
  exited 1 at formatting, so no later gate stages ran. No full gate claim.
- Float32 notebook mutants: tie credit 0 caught by C assertion; failed
  cohorts incorrectly flagged no-pair caught by mask assertion. Dropping only
  delayed-entry detection survived as expected because this mixed cohort
  also has interval/left censoring; no new rule coverage claimed.

B-1 fix: manually wrap that one call exactly as the report's formatter
direction specifies. Two added notebook source lines; no numerical, prose,
metadata, output or assertion change. Final notebook delta from fdce5db:
140 additions / 23 deletions (163 changed lines). The implementer still ran
no build, lint, pytest or other automated validation. Commit this formatting
fix-up and request round 2, with the reviewer still the sole check runner.

Reviewer cleanup note: /tmp/survjax-mut-A10.d2 was reverted and reported
clean but removal was refused twice. It is reviewer-owned; the implementer
does not manipulate it under the repository's independent-review boundary.
