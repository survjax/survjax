# AFT integration — spec difference audit

Reviewed design commits 213cfb0 (revision 47, original AFT checkpoint) through
0ea22df (revision 59, current during integration). The design checkout was
clean; no spec or acceptance-test content was edited by this integration.

- Section 4.4: AFT direct survival, density and hazard formulas, predictor
  constraint, nested diagnostics, broadcasting and sign convention unchanged.
- Sections 4.5, 5 and 7.1–7.3: the direct-method and likelihood contract used
  by the approved AFT implementation remains unchanged. No PH derivative
  exemption is introduced for AFT.
- Revisions 48–58 primarily enable post-release summary methods individually
  and settle summary containers, options, convergence, inverse-hazard endpoints,
  NaN-tangent isolation and the generic RMST tail-cut algorithm. These are
  owned by the separate summary backend worktree. They do not require an
  AFT-specific summary override in this direct-wrapper integration.
- Log-normal and log-logistic scope was brought forward. Their already merged
  implementations, docs and acceptance tests are imported intact from main;
  both family exports and AFT are retained in all export inventories.
- Revision 59 defines comparable flexsurv fixture cells for Slice B censoring
  and truncation. This AFT step implements neither those observation modes
  nor that fixture generator; no AFT doc/value-policy change is needed.

The inherited .spec-seen checkpoint is revision 58, d5cc410. Revision 59 was
reviewed for applicability here, but no broader Slice B compliance claim or
feature activation is made. The spec feature registry is inherited unchanged
from main; unimplemented post-release feature suites remain staged as designed.
The full A12 gate still runs all implementation tests, including both new
families and AFT, plus the cumulative A0–A12 acceptance suites.
