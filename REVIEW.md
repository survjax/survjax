# A6 documentation review

## Read in this order

1. `docs/likelihood.md`: one executable guide covering the implemented
   exact/right-censored likelihood subset, inputs, reduction, broadcasting,
   JAX composition and diagnostic triage.
2. `docs/index.md`: available likelihood subset and guide navigation.
3. `docs/api.md`: inherited likelihood method and parameter diagnostics.
4. The A6 docs entry at the top of `PLAN.md`.

## Invariants and requirement coverage

- Scope remains Slice A without truncation (spec 12.2); no API or numerical
  implementation changes, spec edits, dependencies or independent-test edits.
- All examples use synthetic data and the public API. Exact binary-valued
  analytical anchors need no numerical tolerance changes.
- Event/censoring cells check the hand-derived exponential contributions and
  all four input/entry-point combinations (6.1, 7.1).
- Batch cell checks `(models, observations)` shape and its first model's
  contributions (5); common dtype behavior is described (3.3).
- Compiled objective checks the analytical loss, log-rate gradient and second
  derivative at zero: 6, 4, 6 (7.3). Both builds execute construction in jit.
- Diagnostic cells check stored invalid indicators, parameter violations and
  NaN rows (4.7, 6.2, 7.1). A valid mixed event/interval/right batch checks
  unsupported-mode NaN and unchanged supported contributions (12.2).
- Docs explain the array result without an implicit reduction (6.3/7.1),
  explicit host raising and limits on derivatives of invalid rows (7.2–7.3).

## Checks actually run

- `UV_CACHE_DIR=/private/tmp/survjax-uv-cache scripts/build_docs.sh`: initial
  sandbox attempt exited 2 with `PermissionError: [Errno 1] Operation not
  permitted` while starting notebook execution. Approved unrestricted retry
  exited 0: `build succeeded.` in both float32 and float64, with warnings
  fatal at the Sphinx level. All six executable pages ran in each mode.
  Kernel startup emitted its TCP-without-encryption warning; no Sphinx
  warnings or notebook assertion failures occurred.
  Full retry log: `/private/tmp/survjax-docs-a6.log`.
- Parsed both generated HTML outputs: guide section anchors, index link and
  new likelihood/parameter-diagnostics API anchors are present. Output:
  `x32: guide sections, navigation and three new API anchors present` and
  the same for `x64`. This is HTML inspection, not a browser visual review.
- `git diff --check`: exit 0, no output.

## Limits and open questions

D-026 remains open. The diagnostic recipe is expressly limited to data whose
normalized fields have not been edited after construction; this step does not
resolve the disputed wording for reconstructed/edited data. No code gate,
R fixtures, wheel tests or performance checks were run for this docs-only
step. No full Slice A completion claim. Broader constraint-authoring docs,
simulation-guide activation and playground expansion remain separate work.
