Independent review: round 1/3

# RMST.appendix — Permanent mathematical appendix and conservative guarantees

Read in order: PLAN.md RMST.appendix; spec 8.7/12.1 revision 38;
appendices/README.md and weibull-rmst.md/PDF; section pointers in
src/survjax/_numerics.py and Weibull.rmst's boundary docstring; D-028.

The maintainer approved conservative guarantees and requested a permanent
math-only appendix outside docs/, then the next implementation step. The
approved spec clarification is committed separately in the design repo on
release-rmst as e4c1d50bcf480ef8535dd79fce12e695e564302d. .spec-seen
records it. spec-since.sh dev-release-rmst --mark targets this dedicated
checkout; literal dev targets the different canonical development checkout.
D-028 records the decision without changing its maintainer-owned status.

Invariants: no numerical expression, signature, export or wrapper changes.
Weibull/PH-Weibull zero-cutoff guarantees cover values and first derivatives;
higher nonshape guarantees require smooth interior points. A NaN result's
own tangent is unclaimed; valid-entry isolation remains required. Exponential
keeps its existing all-orders contract. The appendix derives the integral,
series/remainder, mean-tail bound and derivative identities; bounds concern
real arithmetic and do not claim floating-point or arbitrary-derivative proofs.
The source links to the helper and the helper links to the relevant sections.
The historical approval report is replaced rather than duplicated; prior
versions remain in git. No normal user-guide pages were changed.

Validation:
- Pandoc/XeLaTeX rebuilt the adjacent four-page PDF successfully. Poppler
  rendered all pages and every page was visually inspected, including the
  final revised page break; equations are legible and within margins.
- Ruff format: 116 files already formatted; Ruff: all checks passed.
- mypy: success, no issues in 21 source files. git diff --check passed.
- Focused tests/unit/test_rmst.py + test_weibull_rmst.py, separate processes:
  JAX_ENABLE_X64=1: 21 passed; JAX_ENABLE_X64=0: 21 passed. These retain the
  value, zero/invalid, transformation and derivative-smoke evidence for 8.7.
- scripts/build_docs.sh: both dtype builds succeeded under strict Sphinx
  settings. Run with approved escalation for kernel process cleanup; kernel
  TCP warnings were emitted, with no Sphinx errors.

No new unresolved mathematical question. RMST.3 remains the next bounded
implementation/evidence step: extend independent first-derivative checks and
supported Hessian/isolation coverage before PH ownership handoff. The
maintainer explicitly authorized continuing to it after storing this note.
