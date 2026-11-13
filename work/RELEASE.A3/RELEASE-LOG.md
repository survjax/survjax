# Alpha 3 operator execution log

Maintainer authorized proceeding through release after approving preparation.
Scheduled version/date: 0.0.1a3 / 2026-11-13. Publication is still pending.
Starting worktree/previous-step/base: cd383be231739ee8851d2dac540b00289128f622.
Preparation commits: 66c3781, f40fb07 (round-2 merge verdict).

Final dated metadata has passed the full local exact-wheel gate and both
executable docs builds; see REVIEW.md. Final distributions in
/private/tmp/survjax-release-a3-final/dist have the same hashes as the initial
pair listed there. These are rehearsal artifacts, not an upload claim.

Read-only operator checks on 2026-11-13:

- GitHub CLI authenticated as vanAmsterdam; remote main matches local cd383be.
- CPU CI 43415394442 succeeded on that starting main; this does not certify
  the new dated release commit. Fresh final-commit CI remains required.
- Existing pypi environment permits main branches and v* tags; no account
  protection/credential changes made. Its API reports branch-policy rules
  and no required-reviewer rule. The release guide recommends a trusted
  publication reviewer; this discrepancy needs maintainer interaction before
  publishing if that requirement is to be enforced. Do not weaken protections.
- Public PyPI JSON contains only 0.0.1a2. No 0.0.1a3 upload yet.
- Public RTD project is survjax, repository vanamsterdam/survjax, default latest;
  latest build 35865862 succeeded at cd383be. Public latest returns HTTP 200.
- Safari has the logged-in RTD account. The dashboard warns that it could not
  attach a webhook; existing successful builds are observed via the public API.
  No GitHub app permissions were changed. New-version setup/build remains.

Next: final independent review, fast-forward integration and main push,
observe full final-commit CPU CI, tag exact validated commit, activate/build
versioned docs and inspect public downloads, then GitHub prerelease publication
and PyPI workflow/artifact verification subject to existing protections and
the reviewer requirement above. Record further actual results separately;
this log does not certify any unperformed operator action.
