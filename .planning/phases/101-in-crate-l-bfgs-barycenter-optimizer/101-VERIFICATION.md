---
phase: 101-in-crate-l-bfgs-barycenter-optimizer
verified: 2026-09-13T02:00:00Z
status: passed
score: 5/5 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/phases/101-in-crate-l-bfgs-barycenter-optimizer/101-01-PLAN.md
  - .planning/phases/101-in-crate-l-bfgs-barycenter-optimizer/101-01-SUMMARY.md
  - fdars-core/src/metric/soft_dtw.rs
  - fdars-core/src/metric/tests.rs
covered_digest: "v1:sha256:70ae9bede08c752eaf7dbf5bc41359b3f3f2fb8d02d6e99d06f541af943fb8c8"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 101: In-Crate L-BFGS Barycenter Optimizer Verification Report

**Phase Goal:** The soft-DTW barycenter is refined by a proper local optimizer — an in-crate limited-memory L-BFGS (two-loop recursion + backtracking/Armijo line search) driven by the exact soft-DTW gradient the crate already computes — replacing the fixed-step MM/soft-DBA descent, and it works in the default feature set and on `wasm32`.
**Verified:** 2026-09-13T02:00:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `soft_dtw_barycenter(data, gamma, max_iter, tol)` compiles with its exact 4-arg signature returning `SoftDtwBarycenterResult { barycenter, n_iter, converged }` unchanged (non-breaking) | ✓ VERIFIED | `pub fn soft_dtw_barycenter(data: &FdMatrix, gamma: f64, max_iter: usize, tol: f64) -> SoftDtwBarycenterResult` at soft_dtw.rs:585; `#[non_exhaustive]` struct at lines 18–26; `#[must_use]` added per project convention (non-breaking); re-export chain in metric/mod.rs:152–154 and lib.rs:636–638 unchanged |
| 2 | `update_barycenter` (MM descent) is GONE; `eval_obj_and_grad`, `two_loop_lbfgs`, `backtrack_armijo` present; m=10 VecDeque history; curvature guard `sᵀy > 1e-20`; non-descent fallback `d = -g` + clear history | ✓ VERIFIED | Git diff over phase commits shows `-fn update_barycenter` removed, `+fn eval_obj_and_grad`, `+fn two_loop_lbfgs`, `+fn backtrack_armijo` added; `m_lbfgs = 10` at line 618; `VecDeque::with_capacity(m_lbfgs)` at lines 626–627; curvature guard at line 695; non-descent fallback at lines 664–669; `mm_step` is `#[cfg(test)]`-only |
| 3 | Converges to bounded barycenter, honors max_iter (stops at max_iter) and tol (stops on grad inf-norm < tol OR relative cost-change < tol) | ✓ VERIFIED | `lbfgs_max_iter_and_tol_stop` and `lbfgs_boundedness_convex_hull` behavioral tests both pass (5/5 `barycenter_lbfgs_tests` green; `cargo test --lib barycenter_lbfgs` exit 0) |
| 4 | Compiles and runs in the default feature set and on wasm32-unknown-unknown; no new crate dependency | ✓ VERIFIED | `Cargo.toml` unchanged (no new entries in `[dependencies]`); L-BFGS body imports only `std::collections::VecDeque` plus scalar `f64`; `rayon` usage is `#[cfg(feature = "parallel")]` gated (line 10–11), not in L-BFGS body; SUMMARY records `cargo check --target wasm32-unknown-unknown`, `cargo check`, `cargo build --features serde` all pass |
| 5 | Two identical `soft_dtw_barycenter` calls produce bit-identical barycenter (pure-f64 determinism) | ✓ VERIFIED | `lbfgs_deterministic` test uses `.to_bits()` equality; passes in `cargo test --lib barycenter_lbfgs` |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/metric/soft_dtw.rs` | In-crate L-BFGS (two-loop, m=10 VecDeque, Armijo), `update_barycenter` removed | ✓ VERIFIED | File exists at 1210 lines; all three helpers present at lines 421, 447, 502; loop body at lines 636–707; `update_barycenter` absent from production code |
| `fdars-core/src/metric/soft_dtw.rs` | `#[cfg(test)] mod barycenter_lbfgs_tests` with 5 SDTW-01 tests | ✓ VERIFIED | Module at line 964–1209; 5 named tests: `cost_monotone`, `lbfgs_boundedness_convex_hull`, `lbfgs_max_iter_and_tol_stop`, `lbfgs_deterministic`, `lbfgs_cost_le_mm` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `soft_dtw_barycenter` | `eval_obj_and_grad` | Direct call at line 632 and 689 | ✓ WIRED | `eval_obj_and_grad` calls `soft_dtw_distance` + `soft_dtw_accumulate_gradient_and_weight` |
| `soft_dtw_barycenter` | `init_barycenter_mean` | Direct call at lines 609, 615 | ✓ WIRED | Function unchanged at lines 547–560 |
| `metric/mod.rs` | `lib.rs` | Re-export at mod.rs:152–154 and lib.rs:636–638 | ✓ WIRED | `soft_dtw_barycenter` and `SoftDtwBarycenterResult` both re-exported; prelude not checked separately (lib.rs is the public surface) |

### Data-Flow Trace (Level 4)

Not applicable — this phase produces no new data-fetching or rendering artifacts. The optimization loop operates on in-memory `Vec<f64>` slices derived from the caller's `FdMatrix`. The data-flow is: caller `data` -> `rows` (Vec<Vec<f64>>) -> `eval_obj_and_grad` -> `soft_dtw_distance` + `soft_dtw_accumulate_gradient_and_weight` -> gradient `g` and objective `f` -> L-BFGS update -> `bary`. No static returns or hollow props.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 5 SDTW-01 unit tests exist and pass | `cargo test -p fdars-core --features linalg,parallel --lib barycenter_lbfgs` | `running 5 tests … test result: ok. 5 passed; 0 failed` | ✓ PASS |
| Regression test still passes | `cargo test -p fdars-core --features linalg,parallel soft_dtw` (integration) | `test test_soft_dtw_barycenter_vs_tslearn … ok; 9 passed, 0 failed` | ✓ PASS |

### Probe Execution

No probes declared in PLAN.md or found under `scripts/*/tests/probe-*.sh`. Skipped.

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| SDTW-01 | 101-01-PLAN.md | In-crate L-BFGS optimizer with two-loop recursion + Armijo line search, no new dep, wasm32-safe | ✓ SATISFIED | All 5 unit tests pass; `update_barycenter` removed; Cargo.toml unchanged; behavioral tests green |

REQUIREMENTS.md traceability table marks SDTW-01 as `Complete` for Phase 101. SDTW-02 through REL-01 remain pending (phases 102–105) — not in scope for this phase.

### Anti-Patterns Found

No `TBD`, `FIXME`, or `XXX` markers found in `fdars-core/src/metric/soft_dtw.rs`.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None found | — | — |

### Out-of-Scope Functions — Not Modified

Explicitly confirmed unchanged (git diff over phase commits shows no `+`/`-` lines touching these function bodies):

- `soft_dtw_forward` (lines 281–295)
- `soft_dtw_backward` (lines 300–351)
- `soft_dtw_accumulate_gradient_and_weight` (lines 378–398)
- `SoftDtwBarycenterResult` struct definition (lines 18–26, only doc comment updated, no field changes)

### Human Verification Required

None. All must-haves are verifiable programmatically and behavioral tests confirm the runtime behavior.

### Gaps Summary

No gaps. All five must-have truths are VERIFIED against the actual codebase:

- `update_barycenter` is confirmed removed from production code (only referenced in comments and the `#[cfg(test)]` `mm_step` reconstruction)
- All three L-BFGS helpers (`eval_obj_and_grad`, `two_loop_lbfgs`, `backtrack_armijo`) are present and wired into `soft_dtw_barycenter`
- m=10 VecDeque history with curvature guard and non-descent fallback are present as specified
- 5 SDTW-01 unit tests pass; existing regression test passes
- No new crate dependency; wasm32 safety is structural (only `std::collections::VecDeque` + `f64`)

---

_Verified: 2026-09-13T02:00:00Z_
_Verifier: Claude (gsd-verifier)_
