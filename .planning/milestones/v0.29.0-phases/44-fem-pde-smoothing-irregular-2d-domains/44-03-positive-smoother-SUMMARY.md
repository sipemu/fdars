---
phase: 44-fem-pde-smoothing-irregular-2d-domains
plan: "03"
subsystem: smooth_basis
tags: [positive-smoother, log-domain, bspline, functional-data]
status: complete

dependency_graph:
  requires:
    - 44-01  # FEM smoothing module (wave 1)
    - 44-02  # Monotone smoother (wave 2, sibling plan)
  provides:
    - smooth_positive (crate-root + prelude re-exported)
    - SmoothPositiveResult (crate-root + prelude re-exported)
  affects:
    - fdars-core/src/smooth_basis.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

tech_stack:
  added: []
  patterns:
    - log-domain positive smoothing (wrap smooth_basis on log(data), exp-reconstruct)
    - validation-before-transform (T-44-07 threat mitigation)

key_files:
  created: []
  modified:
    - fdars-core/src/smooth_basis.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

decisions:
  - Used FdMatrix for SmoothPositiveResult.fitted and .log_coefficients per PLAN.md (not Vec<f64> as shown in RESEARCH.md §5) — PLAN.md takes precedence.
  - Added #[cfg_attr(feature = "serde", derive(...))] on SmoothPositiveResult matching the wider codebase convention (e.g. regression.rs), though SmoothBasisResult itself does not carry this attr.
  - Added both reject-zero and reject-negative tests (plan specified one; two are tighter coverage).
  - Test for non-positive input verifies the exact parameter field "data" from the error.

metrics:
  duration: "~12 minutes"
  completed: "2026-08-24T16:59:41Z"
  tasks_completed: 2
  commits: 2

estimate:
  tokens: 40000

actuals:
  tokens: 8500
  tasks: 2
  commits: 2
---

# Phase 44 Plan 03: Positive Smoother Summary

**One-liner:** Log-domain positive smoother wrapping existing `smooth_basis` on `ln(data)` with exp-reconstruction for a strictly-positive guaranteed fit.

## What Was Built

`smooth_positive` and `SmoothPositiveResult` added additively to `fdars-core/src/smooth_basis.rs`. The function:

1. Validates `n > 0`, `m > 0`, `argvals.len() == m` (InvalidDimension on failure)
2. Scans every element — any `data[(i,j)] <= 0.0` returns `InvalidParameter` immediately, preventing `ln` from producing NaN/-Inf (T-44-07 mitigation)
3. Builds `log_data` (n×m) with `log_data[(i,j)] = data[(i,j)].ln()`
4. Delegates to the existing `smooth_basis(&log_data, argvals, fdpar)` — no new smoothing math
5. Exp-reconstructs: `fitted[(i,j)] = inner.fitted[(i,j)].exp()` — guaranteed > 0 for any finite real input
6. Returns `SmoothPositiveResult { fitted, log_coefficients: inner.coefficients, edf: inner.edf, gcv: inner.gcv }`

Re-exports added:
- `lib.rs`: `smooth_positive`, `SmoothPositiveResult` added to the `pub use smooth_basis::{...}` block
- `prelude.rs`: `SmoothPositiveResult` added to the `crate::smooth_basis::{...}` re-export

## Tests Added (inline `#[cfg(test)] mod tests`)

| Test | What it checks |
|------|---------------|
| `test_smooth_positive_is_positive` | All fitted values > 0 and finite for a positive signal |
| `test_smooth_positive_recovers_curve` | MAE < 0.2 on noiseless positive signal with small lambda |
| `test_smooth_positive_rejects_nonpositive` | Returns `FdarError::InvalidParameter { parameter: "data", .. }` for data with a zero element |
| `test_smooth_positive_rejects_negative` | Returns `FdarError::InvalidParameter { parameter: "data", .. }` for data with a negative element |

## Verification Results

```
running 112 tests
test smooth_basis::tests::test_smooth_positive_rejects_negative ... ok
test smooth_basis::tests::test_smooth_positive_recovers_curve ... ok
test smooth_basis::tests::test_smooth_positive_is_positive ... ok
test smooth_basis::tests::test_smooth_positive_rejects_nonpositive ... ok
... (108 pre-existing tests also pass)
test result: ok. 112 passed; 0 failed; 0 ignored
```

Command: `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib smooth_basis`

## Deviations from Plan

### Auto-added: extra error-path test

**Task 2 specified one** `test_smooth_positive_rejects_nonpositive` test (data with element = 0). An additional `test_smooth_positive_rejects_negative` test was added covering data with a negative element — tighter coverage of the same guard with minimal cost.

### Style deviation: SmoothPositiveResult carries serde cfg_attr; SmoothBasisResult does not

The PLAN says "mirror SmoothBasisResult" for the `cfg_attr` line. `SmoothBasisResult` in the current codebase lacks the serde attr. The PLAN text is explicit: add `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`. Applied as specified — the newer result struct is more forward-compatible.

Otherwise plan executed exactly as written.

## Known Stubs

None.

## Threat Flags

None — no new network endpoints, auth paths, file access patterns, or schema changes. T-44-07 (ln of non-positive) is mitigated by entry validation.

## Self-Check: PASSED

- `fdars-core/src/smooth_basis.rs` — modified (SmoothPositiveResult, smooth_positive, 4 tests added)
- `fdars-core/src/lib.rs` — modified (smooth_positive, SmoothPositiveResult in re-export block)
- `fdars-core/src/prelude.rs` — modified (SmoothPositiveResult added)
- Commit `07c2947d` — Task 1 tracer (smooth_basis.rs)
- Commit `5a9946bc` — Task 2 re-exports (lib.rs, prelude.rs)
