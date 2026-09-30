# A2 follow-up — align construction with specification revision 9

Status: proposed plan, awaiting maintainer approval before implementation.
Branch: `a2-constructors`. This file replaces the conflicting historical plans
while integrating `main`; their history remains in Git. Resolving these files
is not approval to begin the implementation below.

## Current state and target

The merged specification requires ordinary flag and bounds construction to be
pure for concrete and traced observations. Current A2 source still performs
implicit host validation, rejects traced construction and exposes `with_issues`.
The previous A2 test results are not evidence of revision 9 compliance.

Target public surface:

- `SurvivalData(...)` and `SurvivalData.from_bounds(...)` return normalized data
  with stored issue bits, even when observation values are invalid.
- `data.issues` returns `DataIssues`; masks and scalar `ok` work under JIT.
  The property adds no duplicate leaves to the data PyTree.
- `data.issues.raise_if_invalid()` is the explicit host checkpoint.
- Neither issue-returning convenience constructor is public.
- Static misuse and dtype errors still raise as specified. Preserve endpoint
  attribution, normalization, raw-input evidence and the six-leaf data tree.

## Proposed implementation steps

Each step needs approval, its own code/tests commit and an updated REVIEW.md.
Aim for approximately 300 changed lines per step; propose a narrower split if
needed before implementation grows beyond that review size.

1. Pure constructors and diagnostic property (spec 3.1–3.3, 6.2–6.3).
   - `src/chronox_jax/data.py`: remove automatic host checkpoints and
     `with_issues`; expose `issues` using existing decoding; revise docstrings.
     Do not add `from_bounds_with_issues`.
   - `src/chronox_jax/_data_validation.py`: remove obsolete constructor-only
     raising/traced-rejection plumbing and update helper descriptions while
     preserving shared normalization and issue calculations.
   - `tests/unit/test_data_construction.py`, `test_data_boundaries.py`,
     `test_data_issues.py`, `test_data_dtype.py`, and where needed
     `test_data_representation.py`: migrate own callers to ordinary construction
     and `data.issues`; assert invalid masks rather than automatic exceptions.
     Cover both forms on valid/invalid eager, JIT and vmap inputs, including
     nested traced inputs and concrete inputs captured inside JIT; property
     decoding after tree reconstruction; six unchanged data leaves; absence of
     both removed methods. Keep expected values hand-derived.
2. Exception and static-misuse alignment (spec 3.3, 6.2, 10.2, 12.1).
   - `src/chronox_jax/diagnostics.py` and `__init__.py`: introduce the specified
     top-level `InvalidDataError` for explicit host validation, retaining counts
     and first indices. Resolve the open traced-checkpoint exception question
     before assigning that error a new public type.
   - `src/chronox_jax/_data_validation.py` and `data.py`: align incompatible
     shapes and unknown/removed keywords with `TypeError`; retain explicit
     dtype-policy errors and warnings. Remove obsolete special handling of
     outside-interval censoring, which revision 7 removed from the design.
   - Own construction, issue, dtype and import tests: check precise error types,
     host-only raising, valid/uninformative checkpoints, and unchanged dtype
     behavior. Do not silently choose an outcome for open exception questions.

Indexing and value-derived metadata remain A3 work. The existing all-true
metadata is conservative, but does not complete revision 9's concrete metadata
requirement. Likelihood issue propagation remains A6 work. No distributions,
optimizer dependencies or later-slice implementation belong to this follow-up.

## Verification and independent tests

For each implementation step run Ruff format/lint, mypy, `scripts/run_tests.sh`
and `scripts/check.sh --through A2`, with both configured dtype processes.
Report exact results and any intermediate incompatibilities. The independent
author must align acceptance tests and step markers with revision 9 through the
normal sync process. Do not edit, skip or xfail `tests/spec/`; inspect failures
against their stated contract rather than assuming tests are stale.

## Follow-up before implementation

- The merged DEVIATIONS.md has two D-006 entries: legacy diagnostic compatibility
  and the earlier event/flag conflict question. The specification refers to the
  former. Reconcile identifiers and references with the maintainer; do not mark
  any entry resolved here.
- Review older D-007–D-009 against the new contract without carrying their old
  proposed APIs forward. Preserve the explicitly open exception-lifecycle and
  traced-host-checkpoint questions from spec 13.
- The maintainer completes the pending merge. Then approve the first proposed
  implementation step. This merge-resolution task changes no library code or
  acceptance tests.
