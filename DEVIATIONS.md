# Deviations and spec questions

Entries are added by the implementer and resolved by the maintainer. Format:

```
## D-001 <short title>
- Step: A1
- Spec: section 3.3
- Spec says: ...
- Did / propose: ...
- Status: open
```

## D-001 Inactive `time2` entries in mixed observations
- Step: A1
- Spec: section 3.1
- Spec says: `time2` is an interval's upper endpoint, but does not state what
  supplied `time2` arrays must contain on non-interval rows.
- Did / propose: Ignore `time2` values on non-interval rows during
  normalization, including NaN; validate `time2` only where the interval flag
  is true. This allows one `time2` array to accompany mixed observation kinds
  without treating unused entries as endpoints. The maintainer approved this
  handling in conversation on 2026-09-25; the entry remains formally open.
- Status: open

## D-002 JAX weak Python inputs and float64 warnings
- Step: A1 review fix-up
- Spec: section 3.3
- Spec says: time and bound fields use JAX promotion; float64 input under
  disabled 64-bit mode converts with one warning per call site. It does not
  distinguish Python scalars and lists from dtype-bearing arrays.
- Did / propose: Follow the maintainer's 2026-09-25 decision recorded in the
  independent A1 review: Python `int`/`float` scalars are weak and never warn;
  Python lists/tuples take the dtype JAX gives `jnp.asarray` and never warn;
  NumPy/pandas/JAX values with a float64 dtype warn once per call site in
  32-bit mode; explicit `dtype=` controls the result without a conversion
  warning. Integer-only or scalar-only input uses the configured default
  floating dtype. A clarifying spec edit belongs on a separate `spec-*` branch.
- Status: open
