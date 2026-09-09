---
phase: 90-golden-flake-root-cause-deterministic-fix
verified: 2026-09-09T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 90: Golden-Flake Root-Cause & Deterministic Fix — Verification Report

**Phase Goal:** The three known flaky golden tests (`golden_co_cluster_parallel`, `golden_co_cluster_below_threshold` in equivalence_phase48; `svd_sign_fpca_two_matrix_bit_identical` in equivalence_phase49) pass reliably under repeated full parallel `cargo test`, backed by an evidence-based diagnosis of the root cause.
**Verified:** 2026-09-09
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Diagnosis artifact records evidence-backed root cause AND reconciles intermittent-under-full-parallel symptom against deterministic-when-linalg-off reproduction, naming the fdata_to_pc SVD-branch divergence point | VERIFIED | `90-DIAGNOSIS.md` (commit f2327d9b): contains verbatim deltas (469.177 vs 434.23/415.59; 4.18e-16 vs -0.0), classifies all as categorical backend divergence, section 4 reconciles the intermittent appearance as a feature-config artifact (not disk pressure), section 5 names `fdata_to_pc` `#[cfg(feature = "linalg")]` / `#[cfg(not(feature = "linalg"))]` branch as the single divergence point, section 7 records PACE exclusion. All PLAN.md Task 1 acceptance criteria pass. |
| 2 | Three named golden tests pass reliably under full parallel `cargo test --features linalg,parallel` AND each affected binary passes in isolation | VERIFIED | Spot-check per-binary: `cargo test --features linalg,parallel --test equivalence_phase48 --test equivalence_phase49 -- golden_co_cluster svd_sign_fpca_two_matrix_bit_identical` → `2 passed` + `1 passed`, 0 FAILED. Full-suite run: all test-result lines show 0 failed (2857 lib tests + integration tests, 0 failures). SUMMARY records 10 consecutive ALL_10_GREEN full-parallel runs during execution (commit d75d4f3d). |
| 3 | Under default features (no linalg) the three tests are reported ignored, not failed | VERIFIED | `cargo test --features parallel --test equivalence_phase48 -- golden_co_cluster` → `2 ignored`; `cargo test --features parallel --test equivalence_phase49 -- svd_sign_fpca_two_matrix_bit_identical` → `1 ignored`. PACE sibling `svd_sign_pace_eigenfunctions_single_matrix_bit_identical` passes unguarded under default features (`1 passed`). |
| 4 | No new non-dev crate dependency introduced; fix is test-attribute-only | VERIFIED | `git diff --name-only fce65843..HEAD -- fdars-core/src fdars-core/Cargo.toml Cargo.toml Cargo.lock` produced empty output — no src/ or Cargo changes since the plan commit. The fix consists exclusively of `#[cfg_attr(not(feature = "linalg"), ignore)]` attribute lines on the three named test functions. |

**Score:** 4/4 truths verified (0 present, behavior-unverified)

---

## Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `.planning/phases/90-.../90-DIAGNOSIS.md` | Evidence-backed diagnosis with deltas, reconciliation, divergence point, fix justification | VERIFIED | Non-empty (83 lines); contains all required sections; committed at f2327d9b. grep confirms: `469`, `4.18e-16`/`4.178e-16`, `-0.0`, reconciliation keywords, `fdata_to_pc`, PACE exclusion. |
| `fdars-core/tests/equivalence_phase48.rs` | `#[cfg_attr(not(feature = "linalg"), ignore)]` on `golden_co_cluster_parallel` and `golden_co_cluster_below_threshold` | VERIFIED | 2 occurrences confirmed (grep -c = 2); attribute at lines 56 and 72, immediately above the `#[test]` attribute for each function. Module doc updated to explain the backend requirement. |
| `fdars-core/tests/equivalence_phase49.rs` | `#[cfg_attr(not(feature = "linalg"), ignore)]` on `svd_sign_fpca_two_matrix_bit_identical` only; PACE sibling unguarded | VERIFIED | Exactly 1 occurrence (grep -c = 1); attribute at line 384 above `svd_sign_fpca_two_matrix_bit_identical`. `svd_sign_pace_eigenfunctions_single_matrix_bit_identical` (line 418) has no cfg_attr guard and passes under default features. |

---

## Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `fdata_to_pc` SVD feature branch (src/regression.rs) | co_cluster global FPCA rotation | `#[cfg(feature = "linalg")]` faer vs `#[cfg(not(feature = "linalg"))]` nalgebra | VERIFIED | Diagnosis section 5 names this exact branch and explains propagation: FPCA rotation computed once before n_init loop → both parallel and sequential n_init variants land at identical wrong value 469.177 when linalg absent. |
| cfg_attr guard on three tests | Tests ignored (not failed) when linalg absent | `#[cfg_attr(not(feature = "linalg"), ignore)]` | VERIFIED | Behavioral confirmation: 2 ignored + 1 ignored observed under `--features parallel`. |

---

## Data-Flow Trace (Level 4)

N/A — this phase modifies test attributes and adds a documentation artifact only. No dynamic data rendering.

---

## Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Three golden tests pass under linalg,parallel per-binary | `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase48 --test equivalence_phase49 -- golden_co_cluster svd_sign_fpca_two_matrix_bit_identical` | `2 passed`; `1 passed`; 0 FAILED | PASS |
| Three tests ignored (not failed) under default features — co_cluster pair | `cargo test -p fdars-core --features parallel --test equivalence_phase48 -- golden_co_cluster` | `0 passed; 0 failed; 2 ignored` | PASS |
| Three tests ignored (not failed) under default features — svd_sign | `cargo test -p fdars-core --features parallel --test equivalence_phase49 -- svd_sign_fpca_two_matrix_bit_identical` | `0 passed; 0 failed; 1 ignored` | PASS |
| PACE sibling unguarded and passes under default features | `cargo test -p fdars-core --features parallel --test equivalence_phase49 -- svd_sign_pace_eigenfunctions_single_matrix_bit_identical` | `1 passed; 0 failed; 0 ignored` | PASS |
| Full-suite under linalg,parallel shows no failures | `cargo test -p fdars-core --features linalg,parallel` (tail of output) | All `test result: ok` lines; 0 failed across all binaries | PASS |
| cfg_attr guard count in phase49 is exactly 1 | `grep -c 'cfg_attr(not(feature = "linalg"), ignore)' fdars-core/tests/equivalence_phase49.rs` | `1` | PASS |
| cfg_attr guard count in phase48 is exactly 2 | `grep -c 'cfg_attr(not(feature = "linalg"), ignore)' fdars-core/tests/equivalence_phase48.rs` | `2` | PASS |
| No src/ or Cargo changes since plan commit | `git diff --name-only fce65843..HEAD -- fdars-core/src fdars-core/Cargo.toml Cargo.toml Cargo.lock` | empty output | PASS |

---

## Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FLAKE-01 | 90-01-PLAN.md | Evidence-backed diagnosis of the three golden-test flakes | SATISFIED | `90-DIAGNOSIS.md` (f2327d9b): reproduction commands, verbatim deltas, categorical-vs-ULP classification, intermittent-vs-deterministic reconciliation, fdata_to_pc named as divergence point, PACE exclusion recorded. |
| FLAKE-02 | 90-01-PLAN.md | Deterministic fix chosen from FLAKE-01's diagnosis; three tests pass reliably under full parallel runs | SATISFIED | `#[cfg_attr(not(feature = "linalg"), ignore)]` applied to exactly the three named tests (d75d4f3d); per-binary green; full-suite green; 10× ALL_10_GREEN recorded in SUMMARY. |

---

## Test Quality Audit

| Test File | Linked Req | Active | Skipped (off-linalg) | Circular | Assertion Level | Verdict |
|-----------|-----------|--------|----------------------|----------|-----------------|---------|
| `equivalence_phase48.rs::golden_co_cluster_parallel` | FLAKE-02 | Yes (under linalg) | Yes (under no-linalg, by design) | No | Value (`assert_eq!` bit-identity) | PASS |
| `equivalence_phase48.rs::golden_co_cluster_below_threshold` | FLAKE-02 | Yes (under linalg) | Yes (under no-linalg, by design) | No | Value (`assert_eq!` bit-identity) | PASS |
| `equivalence_phase49.rs::svd_sign_fpca_two_matrix_bit_identical` | FLAKE-02 | Yes (under linalg) | Yes (under no-linalg, by design) | No | Value (`assert_eq!` bit-identity on rotation and scores) | PASS |

No disabled tests on requirements (the `cfg_attr` guard is a conditional skip, not a blanket disable — tests run and pass under their documented feature set). No circular patterns detected. No tolerance relaxations (bit-identity `assert_eq!` preserved throughout).

---

## Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | — | — | None found |

No TBD/FIXME/XXX/TODO markers in either modified test file. No empty implementations. No hardcoded stubs. No assert_eq conversions to tolerance comparisons.

---

## Human Verification

N/A — Infrastructure/foundation phase with no user-facing elements.

This phase modifies only test attributes and adds a documentation artifact. All acceptance criteria are verifiable programmatically via cargo test output and grep. The phase is a test-determinism fix for a Rust library — no UI, CLI output visible to end users, or real-time behavior to observe.

---

## Gaps Summary

None. All four must-haves are verified. The three golden tests pass under the correct feature set, are ignored (not failed) under the unsupported feature set, the diagnosis artifact is complete and committed, and no new dependency was introduced.

---

_Verified: 2026-09-09_
_Verifier: Claude (gsd-verifier)_
