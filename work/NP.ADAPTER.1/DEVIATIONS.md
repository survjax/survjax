# NP.ADAPTER.1 design checkpoints

## D-NP.ADAPTER.1-001

- Step: NP.ADAPTER.1.
- Spec: 3.2, observation shape interface.
- Requirement: SurvivalData exposes shape/len but explicitly has no ndim.
  tests/spec/test_a3_shape_and_len.py independently enforces this exclusion.
- Proposal: NumPyro integration needs ndim for ordinary expansion and shape
  queries. Add len(data.shape) as a property after an authorized spec change,
  or keep this discussion prototype isolated with an explicit compatibility
  shim. The maintainer agreed to the prototype-only shim in response to the
  explicit permission question. No core/spec edits are made; the shim is scoped
  to a context manager and restored on exit. A production change remains future
  coordination work. This records the decision without changing entry status.
- Status: open.

## D-NP.ADAPTER.1-002

- Step: NP.ADAPTER.1.
- Spec: 1, adapter deferred beyond v0.
- Requirement: NumPyro is optional; the production adapter is outside v0.
- Action: The maintainer explicitly requested demonstration code for issue #1
  after discussing the proposed observation/sample contract. Implement only an
  experimental example and optional tests, with no package export, dependency
  declaration, core sampling implementation, or production compatibility claim.
- Status: open (maintainer owns disposition).
