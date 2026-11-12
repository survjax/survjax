# B4.1 — Prune unused exact and censored likelihood pieces

Feature: Slice B static-metadata pruning. Base / previous step:
da4dd733aa92bf7040e918a5fc0e8ef82ba0b85f ([B3.5](../B3.5/PLAN.md)).
Original worktree base: cc0e30f. Dependencies: B2 censoring/truncation evaluator,
B3.4/B3.5 derivative evidence and [B3.3 audit](../B3.3/REVIEW.md).
Cumulative gate: A12, existing observation feature flags unchanged.

Split B4 into bounded steps. This step uses the existing observation-kind flags
to omit the conditional-interval branch when no censored kind may occur, and
the density/reference/normalization branch when no exact row may occur. Mixed
batches and conservative all-true metadata retain both branches. Empty batches
need neither piece. No new specialization, API, module or dependency.

Change src/survjax/likelihood.py only in _evaluate. Preserve current-field
validation, stored issues, parameter masking, common dtype, shape broadcasting,
safe inactive operands and the conditional hook. Do not assume an untruncated
normalizer is zero: custom families may have mass at infinity. Metadata and
constructors remain unchanged; slices retain their parent's flags.

Add tests/unit/test_likelihood_pruning.py: trace-time hook omission, analytical
values/first and second derivatives for exact-only and censored-only batches,
conservative/sliced metadata agreement, compilation reuse, empty/broadcast
shapes, stored and current invalid data, invalid parameter isolation. Use both
dtype modes, jit/vmap, and eager debug_nans on valid inputs. Existing likelihood
tests retain broader family and finite-difference coverage.

Add a small benchmark script under scripts/ to compare prepared concrete data
with identical fields from traced construction (all-true metadata), compiled
summed objective and gradient, synchronized warmed timings separate from first
call, both dtypes. Record commands/environment and measurements in REVIEW.md;
do not assert a speed threshold. Target about 300 implementation/test/script
changed lines; keep the timing-list refresh separate if required, as in B3.5.

Run focused tests in full in each dtype, the required slow-list update if any
new test exceeds one second, and scripts/check.sh --fast --through A12. Commit
the handoff and run independent review, then stop. Spec: 3.2–3.3, 4.2,
7.1–7.3, 12.1–12.3.

## Docs to-do and remaining steps

B4.2: mixed exact/right untruncated optimization and its measurements. Preserve
the general family boundary/override contract when deciding that path.
Later: family feature activation and likelihood documentation. Docs should
explain that traced construction conservatively retains all branches and that
pruning is only an optimization. Reviewer-workflow investigation stays deferred.
