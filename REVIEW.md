# Review: specification revision 9 — remove issue-returning constructors

Read in this order:

1. `spec/SPECIFICATION.md` section 6.2: the public construction surface now
   contains only `SurvivalData(...)` and `SurvivalData.from_bounds(...)`.
2. Sections 3.2 and 12.1–12.2: metadata and acceptance requirements refer to
   those two constructors and diagnostics through `data.issues`.
3. Section 10.2: migration from the removed methods to ordinary construction
   followed by `issues = data.issues`.
4. Sections 11.4 and 13–15: examples use the retained API; decision history
   explicitly marks the old convenience methods as superseded.

## Invariants and acceptance handoff

- Both constructors accept eager and traced inputs, normalize observations and
  retain issue bits. Invalid values do not raise automatically. Static misuse,
  dtype rules and explicit host validation are unchanged.
- `data.issues` remains a pure diagnostic property, with no duplicate data
  PyTree leaves. Likelihood propagation of raw-input issues is unchanged.
- No public constructor returns an issues tuple. Acceptance criteria require
  only the flag and bounds constructors; convenience-method equivalence is no
  longer required. Existing agreement, diagnostic, metadata and host-checkpoint
  requirements remain in section 12.1, covering sections 3.2 and 6.2.
- Strict and compiled simulation examples already use ordinary construction and
  `data.issues`. The simulation policy and Newton example remain intact.

## Checks actually run

- `git diff --check`: passed, no output.
- Repository reference search and specification diff review: removed method
  names appear only in explicit migration guidance; historical descriptions
  identify the old surface as superseded. Prior plans remain labelled history.
- Python `ast.parse`: `Simulation guidance: 2 Python blocks parsed successfully
  (not executed).`
- Focused reference assertions: `Reference audit passed: removed method names
  occur only in explicit migration guidance.` An initial overly broad assertion
  failed on the unchanged phrase “All four observation types”; it was narrowed
  to constructor references and passed. No observation-type rule changed.
- No runtime tests were added or run for this documentation-only follow-up.
  The A1 gate passed in the preceding revision; it does not verify revision 9.
  The example was not executed because its Chronox APIs remain implementation
  work. No dependency installation, R checks or wheel builds were performed.

## Scope and open questions

Implementation and independent acceptance tests remain separate work. No source,
independent tests, dependencies, goals or role instructions changed. D-006's
exception lifecycle and diagnostic-method compatibility questions remain open;
this follow-up does not decide them or introduce new ambiguities.

The maintainer's follow-up authorizes this specification fix-up on
`spec-constructor`. Stop after its commit for review.
