---
phase: 66-core-peer-estimator-penalty-families
verified: 2026-09-04T12:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 66: Core PEER Estimator & Penalty Families — Verification Report

**Phase Goal:** A public `peer()` estimator for structured-penalty scalar-on-function regression that estimates β(t) via the partially-empirical-eigenvector decomposition (null-space + range-space of the penalty operator), choosing among three a-priori penalty families — the estimator that distinguishes PEER from plain FPCR/pfr. Fixed λ (auto-selection is Phase 67; lpeer/predict/exports are Phase 68).
**Verified:** 2026-09-04
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `peer()` recovers a known β(t) within tolerance on synthetic data and returns a `PeerResult` carrying beta, intercept, fitted_values, effective_df, lambda, penalty_type (PER-01) | ✓ VERIFIED | `test_peer_difference_beta_recovery`: max abs error < 0.1 on sin(π·t) fixture (n=200, m=40, λ=1e-4). `test_peer_result_shape`: all fields present with correct types and values. Both tests pass. |
| 2 | Penalty family is selectable via `PeerPenalty`; Ridge, Difference{order:2} (reusing `penalty_matrix`), and caller-supplied Decree(Q) each fit without error (PER-02) | ✓ VERIFIED | `test_peer_ridge_fits`: Ridge fit succeeds, finite β. `test_peer_decree_fits`: Decree with penalty_matrix(m) fits without error. `test_peer_difference_order_rejected`: order 3 returns `FdarError::InvalidParameter`. All 3 tests pass. |
| 3 | Decree(Q) with a partition-boundary structure yields a β(t) demonstrably different from plain Difference{2} roughness on the same data (PER-02) | ✓ VERIFIED | `test_peer_decree_distinct_from_roughness`: partition-aware Q (second-difference with zeroed cross-boundary stencil rows + tiny ridge for PSD stability) vs. full Difference{2}, same data and λ=1.0; max pointwise |β_decree − β_diff| > 1e-3; both β(t) all-finite. Test passes. |
| 4 | Wrong-dimension Q (or invalid penalty input) returns a descriptive `FdarError`, never panics; no NaN β(t) across all three families (PER-01, PER-02) | ✓ VERIFIED | `test_peer_decree_wrong_dim`: Decree(m-1 × m-1) returns `InvalidDimension`. `test_peer_argvals_mismatch`: argvals.len()=m+1 returns `InvalidDimension`. `test_peer_no_nan_all_families`: all three families at λ=1.0 yield finite beta, fitted_values, effective_df. All 3 tests pass. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/peer.rs` | New file: `peer`, `PeerConfig`, `PeerResult`, `PeerPenalty` + inline `#[cfg(test)] mod tests` | ✓ VERIFIED | 659 lines; all four public symbols present with correct derives, `#[non_exhaustive]` on `PeerResult`, `#[must_use]`, conditional serde. 9 inline tests. |
| `fdars-core/src/lib.rs` | Adds `pub mod peer;` between `outliers` and `regression` | ✓ VERIFIED | Line 111: `pub mod peer;` — confirmed between `pub mod outliers;` (line 110) and `pub mod regression;` (line 112). |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `peer.rs` | `crate::helpers::simpsons_weights` | `use crate::helpers::simpsons_weights;` at line 29 | ✓ WIRED | Called in `peer()` body at line 174. |
| `peer.rs` | `crate::function_on_scalar::penalty_matrix` | `use crate::function_on_scalar::penalty_matrix;` at line 28 | ✓ WIRED | Called in `build_q` for Difference{2} arm. |
| `peer.rs` | `crate::linalg::{cholesky_factor, cholesky_forward_back, cholesky_solve}` | `use crate::linalg::{...};` at line 30 | ✓ WIRED | `cholesky_solve` called in `peer()` body; `cholesky_factor` and `cholesky_forward_back` called in `compute_peer_trace_hat`. |
| `lib.rs` | `peer` module | `pub mod peer;` at line 111 | ✓ WIRED | Module registered; crate-root `pub use` re-exports correctly DEFERRED to Phase 68. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `peer.rs::peer()` | `beta` | Penalized normal equations `(W_c'W_c + λQ)β = W_c'y_c` solved via `cholesky_solve` on real `data` + `y` inputs | Yes — no static fallback; returns `ComputationFailed` on non-finite result | ✓ FLOWING |
| `peer.rs::peer()` | `fitted_values` | Derived from `y_bar + Σ_j wc[i,j]·beta[j]` using real β(t) and centered design | Yes — real computation on caller-supplied data | ✓ FLOWING |
| `peer.rs::compute_peer_trace_hat()` | `effective_df` | Column-solve loop on `A = W'W + λQ` via Cholesky; falls back to `m as f64` only on Cholesky failure | Yes — real trace computation | ✓ FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All 9 peer module tests pass | `cargo test -p fdars-core --lib --features linalg,parallel peer:: 2>&1 \| tail -20` | `test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 2689 filtered out; finished in 0.01s` | ✓ PASS |
| Clippy clean on all targets | `cargo clippy -p fdars-core --all-targets --features linalg,parallel 2>&1 \| tail -5` | `Finished dev profile` — no warnings emitted | ✓ PASS |

---

### Scope Discipline Checks

| Check | Command | Result | Status |
|-------|---------|--------|--------|
| `pub mod peer;` registered | `grep -n "pub mod peer;" fdars-core/src/lib.rs` | `111:pub mod peer;` | ✓ PASS |
| No crate-root `pub use peer` re-export | `grep -rn "pub use.*peer" fdars-core/src/lib.rs` | No output | ✓ PASS |
| No prelude entry for peer | `grep -rn "peer" fdars-core/src/prelude.rs` | No file / no output | ✓ PASS |
| No `lpeer` or `fn predict` in peer.rs | `grep -rn "lpeer\|fn predict" fdars-core/src/peer.rs` | No output | ✓ PASS (correctly deferred to Phase 68) |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| PER-01 | 66-01 | Public `peer()` estimator; β(t) recovery; `PeerResult` with diagnostics | ✓ SATISFIED | SC-1 and SC-4 verified by 5 passing tests |
| PER-02 | 66-01 | Three penalty families; structured Q partition-awareness; error surface | ✓ SATISFIED | SC-2, SC-3, and SC-4 verified by 7 passing tests |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | No debt markers, no stubs, no unreachable TODOs found in peer.rs |

Scan confirmed: no `TBD`, `FIXME`, `XXX`, `todo!()`, or `unimplemented!()` markers remain in `fdars-core/src/peer.rs`. The Task 1 plan called for temporary stubs in `build_q` to be filled by Task 2 — the final committed file has all arms fully implemented.

---

### Human Verification Required

None. All phase behaviors have automated numerical verification with passing tests. The VALIDATION.md explicitly records "All phase behaviors have automated verification."

---

## Gaps Summary

No gaps. All four ROADMAP success criteria are verified by passing behavioral tests. The implementation is substantive (659-line file), correctly wired to existing crate infrastructure (Cholesky, penalty_matrix, simpsons_weights), and scope discipline is maintained (no premature Phase 68 exports).

---

_Verified: 2026-09-04T12:00:00Z_
_Verifier: Claude (gsd-verifier)_
