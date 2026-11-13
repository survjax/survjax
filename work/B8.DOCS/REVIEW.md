# B8.DOCS — Current documentation sync

Independent review: round 1/3 completed; verdict maintainer; maintainer-accepted 2026-11-13

## Reading order and scope

Read PLAN.md, README.md/CHANGELOG.md, docs/api.md, then CONTRIBUTING.md and
 docs/releasing.md. Compare current claims with docs/index.md, likelihood.md,
working-with-distributions.md, src/survjax/__init__.py and the CI/release workflows.
B7.AUDIT records Slice B completion evidence; B6.1–B6.4 cover the detailed guides.

The invariant is that current-checkout support is distinct from historical
published-release support. Concordance and RMST retain their narrower subsets.
No numerical behavior, API, tests, dependencies or release versions changed.
Existing executable guides supply examples; API additions render existing
method docstrings. Existing wheel metadata coverage checks README portability.

## Validation

- Initial fast gate exposed relative README links in wheel metadata:
  test_wheel_metadata_contains_readme_and_release_links failed on CHANGELOG.md.
  Converted all three new README links to absolute GitHub URLs; no test edit.
- Final scripts/check.sh --fast --through A12: exit 0, all gates passed.
  Ruff: 287 files already formatted, all checks passed. Mypy: no issues in
  25 source files. Both implementation suites passed (float32: 1214 passed,
  6 skipped). Source doctests: 4 passed per dtype. Acceptance: float64
  4654 passed/46 skipped; float32 4597 passed/93 skipped. Each acceptance run
  emitted two NumPy boundary warnings from test_a11_ph_direct_methods.py.
  Full non-fast gate was not run; it remains the maintainer's merge check.
- scripts/build_docs.sh: exit 0 in both dtypes, repeated after the README link
  correction. Final build identity: this checkout, starting commit c908176,
  dirty documentation snapshot, completed 2026-11-13T10:14:35Z, state success.
- HTMLParser checks in both outputs: both new API IDs occur exactly once;
  the working-with-distributions link resolves to an existing page; the rendered
  release command names A12; changelog and contributor updates are included.
  Initial raw-HTML text assertion was corrected to compare parsed text because
  syntax highlighting splits command text into spans. No source change needed.
- git diff --check passed. No browser screenshot/layout inspection performed.

## Remaining items and integration

No remaining substeps for this documentation sync. Protected AGENTS.md and the
specification status still mention Slice A; left untouched for their owners.
Historical release entries, historical benchmark caveats and example-specific
exact/right-censoring descriptions remain accurate in context.
The pre-existing empty untracked nvim.log is temporarily preserved outside the
checkout for the clean-worktree review prerequisite and will be restored after
review. No other user's files were changed. No publication or integration done.

Starting/previous-step commit: c908176edefe18797e1d1b81eeca55801be7b6b0.
Original worktree base: cc0e30f89af61d963c8b91b5e2beb6b08a1af810.
Roadmap handoff: documentation entry-point sync merged, review completed and
maintainer-accepted on 2026-11-13; no further B8 review is pending.

## Initial independent-review attempt (historical)

Requested review of d4f6f1f615ec99f79f06754bf72dc169145c6495 using the exact
skill command. Both the initial attempt and the single permitted retry outside
the sandbox exited 1: `Failed to authenticate. API Error: 403 Attention Required! | Cloudflare`.
At that point no review file was written and no verdict or findings were
available. Review remained pending; no further retry was attempted then. The optional diff-display
hook is skipped because there is no reviewed SHA. This record-only follow-up
does not change the validated documentation snapshot.


## Review completion and maintainer disposition — 2026-11-13

The subsequent independent review is recorded in
[review/slices/B8.DOCS.md](../../../review/slices/B8.DOCS.md), round 1, for
`d4f6f1f615ec99f79f06754bf72dc169145c6495` against `c908176`.
It found no blockers or should-fix findings. N-1 is a nonblocking README
wording nit; the reviewer explicitly found the text acceptable and accurate.
The reviewer independently passed the fast A12 gate with all five enabled
features in both dtypes.

The formal verdict is `maintainer`, not `merge`: the reviewer could not run
the docs build because the sandbox denied the Jupyter kernel's local port
binding. It did not independently verify rendered HTML or browser layout.
The successful implementer builds and HTML checks remain recorded above.

The maintainer reported the reviewer was happy and instructed closure of the
roadmap and outstanding-review tracking. Record this as maintainer acceptance
of the review's stated verification limitation and N-1, with no required B8
follow-up or pending review. Preserve the independent verdict unchanged.
Full release-commit checks remain separate; no new build or test run is claimed
by this bookkeeping update. The earlier failed authentication attempts are
historical and no longer block B8.
