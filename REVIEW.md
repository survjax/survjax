Independent review: round 2/3

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

## Independent review outcome — round 1/3

Review: ../review/slices/RMST.appendix.md, VERDICT: maintainer.
The reviewer independently verified every mathematical bound/identity and
rebuilt the PDF; extracted text matches. Their unchanged gate passed lint,
mypy, own tests (1069/7 x64, 1075/1 x32) and four doctests per mode. The
x64 spec suite had 2327 passed, 28 skipped and five wheel-setup errors because
SURVJAX_WHEEL was unset; x32 spec tests did not run in that review.

S-1a: the maintainer already explicitly approved the conservative
value-and-first-derivative-only zero boundary. Higher boundary derivatives
remain outside the guarantee; this is not a new question for the maintainer.
The suggested direction-of-comparison wording can be clarified on resumption.
S-1b: my interpretation also deferred NaN-entry own tangents, beyond D-028.
Asked the maintainer whether to retain that broader exclusion or restore the
NaN-tangent requirement. Valid-entry isolation remains mandatory either way.
S-1c: design integration is maintainer-owned; no merge is authorized.

Pending factual fix-ups after the required maintainer stop: F-1 add both
boundary/tangent limits (as finally decided) to Docs to-do; F-2 add the
missing gamma-middle appendix pointer; N-1 explain the wider-dtype 24b
example; N-2 label the deleted report path historical. No numerical defect
was reported. The RMST.3 plan is drafted, but no implementation/test file
has been changed. Preserve that unfinished plan until S-1b is decided.

## Round 2 fixes and maintainer confirmation

S-1b: maintainer explicitly replied, “yes, a NaN RMST result's own
derivatives may be unspecified”. Revision 38 is now explicitly approved on
both questioned boundaries; valid-entry isolation remains mandatory.
S-1a: the prior approval already accepts no higher-order zero-boundary
guarantee. “May differ” is a symmetric relation, not a reversal of mathematical
authority: the appendix identifies the true right derivatives explicitly.
No new spec edit is needed. S-1c is a maintainer integration handoff, not
an outstanding implementation decision; we must not merge either repository.
F-1: both RMST Docs to-do records now include the first-order-only zero
boundary and unspecified NaN-own tangents. F-2: added gamma-middle pointer.
N-1: explained why 24b can require float64 even with a float32 scale.
N-2: identified the deleted report path as historical in D-028.
No numerical expression or guarantee was added. Validation pending below.

Round 2 validation: Pandoc/XeLaTeX compiled successfully; Poppler rendered
all four pages and each was visually inspected without clipping. Ruff format
reported 116 files already formatted; Ruff passed; mypy found no issues in
21 source files; git diff --check passed. scripts/build_docs.sh completed
both dtype builds successfully (only kernel TCP warnings). Numerical tests
are unchanged and no executable expression changed in this fix-up; focused
21-per-mode evidence and round 1 regressions remain applicable.
