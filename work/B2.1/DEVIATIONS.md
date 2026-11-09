# D-B2.1-001 — Tiny left-tail acceptance reference loses positive mass

- Step: B2.1; status: open.
- Spec: 4.1, 7.1, 7.2 and 12.1 require representable finite log CDFs even
  when probability/cumulative hazard underflows; precision loss is not zero.
- Evidence: test_observation_censoring_numerics.py::
  test_weibull_left_row_where_the_cumulative_hazard_underflows first asserts
  finiteness, then compares against observation_reference.contribution, which
  returns -inf. Its _log_mass subtracts survivals at DPS=50. For shape=2.5,
  scale=2, upper=1e-30, H is about 1.77e-76; that subtraction rounds to zero.
- Implementation returns about -174.4267499259533, as required by log F.
  Independent mpmath probe: log(1-exp(-H)) at 50 digits is -inf; at 100
  digits it is -174.4267499259532895748924393988. Stable log(-expm1(-H))
  at 50 digits is -174.4267499259532895748924394050.
- Proposal: independent spec-test author uses a stable reference expression
  (or enough precision) for this tiny-tail case. No protected test, tolerance,
  marker or feature flag has been weakened; observation.censoring stays enabled.
- Stop for maintainer resolution. Do not claim the B2.1 gate passed.
