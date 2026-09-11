---
phase: 98-differentiable-depth-curve-distances
verified: 2026-09-11T00:00:00Z
status: passed
score: 3/3 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 98: Differentiable Depth & Curve Distances (DOP-04) Verification Report

**Phase Goal:** ≥1 functional depth + ≥1 curve distance (beyond soft-DTW) generic over Scalar and differentiable, f64 parity preserved.
**Verified:** 2026-09-11
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | At least one functional depth measure (modal) is generic over Scalar and differentiable w.r.t. curve values; f64 depth values unchanged (within 1e-12 of modal_1d). | VERIFIED | `pub fn modal_depth_generic<T: Scalar>(curve: &[T], reference: &FdMatrix, h: f64) -> T` present in `depth/modal.rs:96`. Test `test_modal_depth_generic_f64_parity` asserts parity within 1e-12 vs `modal_1d`; 3/3 tests pass live. |
| 2 | At least one curve distance beyond soft-DTW (l2_distance, Phase 95) is generic over Scalar and differentiable. | VERIFIED | `pub fn l2_distance<T: Scalar>(curve1: &[T], curve2: &[T], weights: &[f64]) -> T` present in `helpers.rs:66`. Dual and Var FD tests present in `helpers.rs:1222–1300`; full test suite passes (0 failed). |
| 3 | Gradients of the modal-depth path match central FD at both Dual and reverse-mode Var (1e-6*(1+|fd|)). | VERIFIED | Tests `test_modal_depth_generic_dual_fd_check` and `test_modal_depth_generic_var_fd_check` exercise off-reference queries and assert tolerance 1e-6*(1+|fd|); both pass live (confirmed in gate run). |

**Score:** 3/3 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/depth/modal.rs` | `modal_depth_generic<T: Scalar>` function | VERIFIED | Function present at line 96; generic over `Scalar`; full doctest + 3 tests. |
| `fdars-core/src/depth/mod.rs` | Re-export `modal_depth_generic` | VERIFIED | `pub use modal::{modal, modal_depth_generic};` at line 38. |
| `fdars-core/src/lib.rs` | Re-export `modal_depth_generic` at crate root | VERIFIED | Listed in depth pub use block at line 648. |
| `fdars-core/src/prelude.rs` | Re-export `modal_depth_generic` in prelude | VERIFIED | Listed in prelude re-exports at line 43. |
| `fdars-core/src/depth/tests.rs` | 3 tests: f64_parity, dual_fd_check, var_fd_check | VERIFIED | Tests present at lines 593–684; all 3 pass live. |
| `fdars-core/src/helpers.rs` | `l2_distance<T: Scalar>` (Phase 95, criterion #2) | VERIFIED | Function at line 66; Dual/Var tests at lines 1222–1300; passes full suite. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `depth/modal.rs` | `depth/mod.rs` | `pub use modal::{modal, modal_depth_generic}` | WIRED | Confirmed grep line 38. |
| `depth/mod.rs` | `lib.rs` (crate root) | depth module pub use block | WIRED | Confirmed grep line 648. |
| `lib.rs` | `prelude.rs` | prelude re-export | WIRED | Confirmed grep line 43. |
| `modal_depth_generic<Dual>` | `autodiff::Dual` | `use crate::autodiff::Scalar` import line 3 of modal.rs; `Dual::seed` / `.extract()` in tests | WIRED | Tests import `crate::autodiff::Dual` directly and pass. |
| `modal_depth_generic<Var>` | `autodiff::vjp` + `Var` | `vjp` call in `test_modal_depth_generic_var_fd_check` | WIRED | Test imports `crate::autodiff::{vjp, Var}` and passes. |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|--------------|--------|--------------------|--------|
| `modal_depth_generic<T>` | `depth` (return value) | Live computation over `curve: &[T]` and `reference: &FdMatrix` via Gaussian kernel | Yes — arithmetic on caller-provided data, no static literals in return path | FLOWING |
| `l2_distance<T>` | return value | Live computation over `curve1: &[T]`, `curve2: &[T]`, `weights: &[f64]` | Yes | FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 3 modal_depth_generic tests (parity + Dual + Var) | `cargo test --lib modal_depth_generic --features linalg,parallel` | 3 passed; 0 failed | PASS |
| Full test suite | `cargo test -p fdars-core --features linalg,parallel` | 0 failed (all tests pass) | PASS |
| modal_depth_generic doctest | `cargo test --doc modal_depth_generic` | 1 passed; 0 failed | PASS |

---

### Probe Execution

No probe scripts declared for this phase. Standard gate suite substitutes.

| Gate | Command | Result | Status |
|------|---------|--------|--------|
| lib tests (all) | `cargo test -p fdars-core --features linalg,parallel` | 0 failed | PASS |
| doctest | `cargo test --doc modal_depth_generic` | 1 passed | PASS |
| examples build | `cargo build --examples -p fdars-core` | Finished (no errors) | PASS |
| serde feature build | `cargo build -p fdars-core --features serde` | Finished (no errors) | PASS |
| wasm build | `cargo build -p fdars-core --target wasm32-unknown-unknown --features js` | Finished (no errors) | PASS |
| clippy all-targets | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` | Finished (no warnings) | PASS |
| Cargo.toml diff | `git diff HEAD -- fdars-core/Cargo.toml` | empty — no new dependencies | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| DOP-04 | 98-01-PLAN.md | ≥1 functional depth + ≥1 curve distance generic over Scalar and differentiable, f64 parity preserved | SATISFIED | `modal_depth_generic<T>` + `l2_distance<T>`; parity/Dual/Var tests all pass. |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

No TBD/FIXME/XXX/TODO/HACK debt markers in any of the 5 modified files. No empty implementations, no hardcoded stubs, no orphaned code.

**Prohibition check:**

| Prohibition | Status | Evidence |
|-------------|--------|----------|
| No new crate dependency added to Cargo.toml | VERIFIED | `git diff HEAD -- fdars-core/Cargo.toml` produced empty output. |
| `modal_1d`/`modal` signatures unchanged; churn confined to depth/ + additive re-exports | VERIFIED | `modal_1d` is `pub(crate)` and unchanged (line 19 of modal.rs); `modal` is `pub fn modal(data_obj: &FdMatrix, data_ori: &FdMatrix, h: f64, dim: Dim) -> Vec<f64>` unchanged (line 56). Changes are additive: new function + re-exports only. |

**Note on sqrt kink at self-match:** Modal depth is non-differentiable exactly when `curve == reference_row_j` (dist=0, sqrt kink — analogous to `|x|` at 0). The FD tests correctly use off-reference queries (`data[(0,t)] + 0.05 + 0.01*t`), which ensures all `dist_j > 0`. This is expected behavior, documented in the PLAN and SUMMARY, and is not a gap.

---

### Human Verification Required

None. All three success criteria are verified by live gate output with zero ambiguity.

---

### Gaps Summary

No gaps. All must-haves satisfied with live evidence.

---

_Verified: 2026-09-11_
_Verifier: Claude (gsd-verifier)_
