# META.FEATURE-GATE — Default to baseline plus completed features

Step starting commit: 65340045dce1e894c25b1fd7a78086775572138f.
Previous step: B8.DOCS; original branch base: same commit.
Branch: meta-feature-gate. Maintainer explicitly authorizes protected runner
and role changes, bookkeeping closure and commits on 2026-11-13.

Default check.sh to all baseline tests plus scripts/spec-features.txt, clearing
inherited SPEC_THROUGH. Retain optional --through for historical diagnostics;
reject malformed/missing values before running checks. Update CI/release,
wheel and slow-inventory helpers, current contributor/release/role guidance.
The slow inventory must select the committed features too. No acceptance test
or numerical contract changes. Spec sections 8.8 and 12.1–12.3.

Add subprocess regression tests with stubbed external commands to inspect the
actual pytest environment, default versus explicit step selection, malformed
arguments and committed feature precedence. Run focused tests, full gate without
--through, docs build, whitespace checks; commit and independent review.
Close maintainer-authorized B2.1/B4.3 deviations and B8 handoff bookkeeping.
Record design/status and Q-066 disposition separately in their owning repos.
