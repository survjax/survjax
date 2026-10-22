# Integrate main through A6 into the A8 checkout — 2026-10-23

The maintainer started the merge and authorized resolving its conflicts.
Incoming main: `d4d4752`; A8 parent: `15b0a16`.

## Read in this order

1. `PLAN.md`: A8 remains active; the incoming A6 plan is retained as history.
2. The merge diff for `PLAN.md` and `REVIEW.md`: these are the only conflicts.
3. Prior review records remain available at `15b0a16:REVIEW.md` (A8.3 nits)
   and `d4d4752:REVIEW.md` (A6 docs).

## Invariants and scope

- All cleanly merged files retain Git's merge result, including source, tests,
  docs, DEVIATIONS.md and `.spec-seen`. No independent test edits.
- A8 generators, manifests and committed R payloads remain unchanged.
- A6 makes the likelihood API available here. This merge adds no A8.4 loader
  or reference tests and makes no A7 or through-A8 completion claim.
- No conflicting API or numerical decision required maintainer input.

## Verification

- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/check.sh --through A6`:
  exit 0, `check: all gates passed through A6`.
  - Guard: OK; Ruff: 69 files formatted and all checks passed.
  - mypy: no issues in 17 source files.
  - Own tests: float64 722 passed / 7 skipped; float32 728 passed / 1 skipped.
  - Source doctests: 4 passed in each dtype mode.
  - Independent tests: float64 2023 passed, 28 skipped in 95.42s (0:01:35);
    float32 2027 passed, 24 skipped in 90.99s (0:01:30).
- Both fixture manifests: `data.csv: OK`, `expected.json: OK` (four hashes).
- Confirmed the incoming A6 plan and A8 fixture plan are preserved verbatim;
  no conflict markers or unmerged paths remain.
- Incoming source, docs, independent tests, spec marker, deviations and
  tolerances match MERGE_HEAD exactly. Fixture files match the A8 parent.
- `git diff --cached --check -- PLAN.md REVIEW.md`: exit 0, no output.
  The full staged whitespace check reports one inherited blank line at EOF
  in `tests/spec/API_ASSUMPTIONS.md:480`; it is identical to incoming main
  and is left unchanged under the read-only spec-test rule.

No R regeneration or docs build was run; neither generators nor docs were
changed during conflict resolution. No A8.4 implementation or new tests.
