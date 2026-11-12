# B6.4 — Verification page organization

Independent review: round 2/3

## Read first

1. PLAN.md: maintainer's requested structural review, spec 10.4–10.5.
2. docs/technical-notes/verification.md: complete heading outline, then the
   consolidated R section and regeneration guidance.
3. B6.3's handoff for the unchanged Slice B comparison evidence and the fixture
   READMEs for each generator's invocation. No new comparison claim is added.

One page/topic, 256 changed documentation lines including moves. No source,
tests, fixtures, dependencies, API or protected paths change.

## Findings addressed and resulting structure

- R text previously appeared under the wheel heading, in a later separate R
  section, and in a top-level concordance section. All now belongs to one H2
  Comparisons with R, directly after the executable mathematical checks.
- H3 children separate exact/right references, the Slice B matrix, concordance,
  and shared integrity/regeneration guidance. Flexsurv exclusions are H4 below
  the matrix. The Slice B generator/test pointers precede the exclusions so
  they describe the fixture as a whole.
- Wheel installation, historical CPU baseline and documentation builds are
  sibling H2 topics. The long build section has H3 verification/viewing and
  interactive-exploration subsections. No skipped heading levels.
- Duplicate general R provenance was consolidated. The old exact/right
  statement about "unimplemented" modes is scoped to those two fixtures.
  Historical decimal/hex read-back evidence remains scoped to the older
  exponential/bc-like fixtures. Regeneration wording now distinguishes default
  destinations from Slice B's required destination and links all four READMEs.

All existing heading texts/anchors are retained despite level/order changes.
All Python code cells are byte-identical and in the same order as step start.
The numerical examples, fixture tables, tolerances and benchmark measurements
are retained; this is organization and scope clarification, not new evidence.

## Validation

- Source checks: single Comparisons with R heading, no skipped levels,
  unchanged ordered Python code cells; git diff --check clean.
- scripts/build_docs.sh run directly: exit 0, "Verified both dtypes".
  Existing code cells executed successfully in both modes. Existing Jupyter
  transport notices appeared, with no Sphinx warning failure.
  Build identity: base 1c47d6a, dirty with this step, state success,
  completed 2026-11-12T17:59:58Z.
- HTMLParser inspection in both generated versions: 13 correctly nested
  headings and section parents, unique IDs, all 10 previous heading anchors
  preserved, 68 local links and 36 inbound links resolved per mode. All four
  fixture README downloads match their source bytes. No browser layout check
  claimed; hierarchy was checked in actual rendered HTML section elements.
- scripts/check.sh --fast --through A12: exit 0, all gates passed. Format:
  283 files already formatted; Ruff clean; mypy clean on 25 source files.
  Implementation float64: 1212 passed / 8 skipped (26.84s); float32:
  1214 passed / 6 skipped (30.50s). Source doctests: 4 passed per mode.
  Spec float64: 4654 passed / 46 skipped (58.60s); float32:
  4597 passed / 93 skipped (44.46s). Two existing NumPy warnings per mode
  at test_a11_ph_direct_methods.py:78.
- No new tests, fixture generation, benchmark rerun or non-fast gate. Existing
  B6.3 focused reference evidence remains applicable to the unchanged claims.

## Commits and remaining work

Round 1 reviewed 850ec2bfd345485dd4820716bb06f512a9e67b47 and returned
fix-first. F-1: the shared regeneration paragraph now explicitly lists
concordance alongside exponential and bc-like as defaulting to committed
directories. Checked all four generators' argument/output handling.
N-1: rewrapped the cited paragraphs. N-2: named the Slice B R comparison
explicitly in the exclusions introduction. No heading, code cell or link
changed in the fix-up.

After the fixes, scripts/build_docs.sh again exited 0 with both dtypes verified;
final build identity is base 850ec2b, dirty with the fix-up, completed
2026-11-12T18:06:58Z, state success. Final HTML in both modes contains the
corrected destination sentence and exclusions scope with the same 13 headings
and parent relationships. Python cells remain byte-identical to step start.
The fast A12 gate passed again: format 284 files, Ruff/mypy clean, the same
pass/skip counts as above in both modes, and the same two known spec warnings.
Implementation times were 14.60s/15.22s; spec times 35.85s/35.87s. No unresolved
finding remains from round 1 in the implementer's assessment.

Step starting / previous-step final:
1c47d6a0acce393cf57ff4727f78bf80baa6f398 (B6.3).
Original worktree base: cc0e30f.

The final Slice B evidence audit and maintainer integration remain. This step
addresses the page organization request and makes no Slice B completion claim.
