---
phase: 90-golden-flake-root-cause-deterministic-fix
reviewed: 2026-09-09T00:00:00Z
depth: standard
files_reviewed: 2
files_reviewed_list:
  - fdars-core/tests/equivalence_phase48.rs
  - fdars-core/tests/equivalence_phase49.rs
findings:
  critical: 0
  warning: 0
  info: 0
  total: 0
status: clean
---

# Phase 90: Code Review Report

**Reviewed:** 2026-09-09
**Depth:** standard
**Files Reviewed:** 2
**Status:** clean

## Summary

Phase 90 added a single `#[cfg_attr(not(feature = "linalg"), ignore)]` guard to exactly three golden tests — `golden_co_cluster_parallel` and `golden_co_cluster_below_threshold` in `equivalence_phase48.rs`, and `svd_sign_fpca_two_matrix_bit_identical` in `equivalence_phase49.rs` — plus explanatory doc/comment lines. No source files, Cargo.toml, or golden constant values were changed.

The change was reviewed against the following axes:

**Attribute syntax and placement.** The `#[cfg_attr(not(feature = "linalg"), ignore)]` form is valid Rust; `cfg_attr` accepts a condition and an attribute to conditionally apply. The condition `not(feature = "linalg")` is correctly expressed. In both files the guard appears as the line immediately preceding `#[test]`, which is the required placement — multiple outer attributes on an item are applied in declaration order and are fully composable.

**Test count.** Exactly three tests are guarded: `golden_co_cluster_parallel` (phase48:56), `golden_co_cluster_below_threshold` (phase48:72), `svd_sign_fpca_two_matrix_bit_identical` (phase49:384). `svd_sign_pace_eigenfunctions_single_matrix_bit_identical` (phase49:418) carries no guard, consistent with the diagnosis that it passes under default (no-linalg) features.

**Regression masking.** The guard does not suppress a latent real regression. Per the diagnosis, off-linalg divergence is fully deterministic categorical backend divergence (faer vs nalgebra SVD in `fdata_to_pc`), not a correctness regression in `co_cluster` or `fix_svd_signs`. The module docs on both files have required `--features linalg` for these tests since their original capture; the guard mechanically enforces that pre-existing contract. Under both documented linalg configs (`--features linalg,parallel`; `--no-default-features --features linalg`) the tests continue to run and pass.

**Assertion strength.** All assertions remain `assert_eq!`. No golden constant values were modified. The diff introduces no tolerance widening.

**Scope containment.** `git diff fce65843..HEAD --name-only` shows only the two test files modified (plus planning artifacts); no `src/` or `Cargo.toml` change occurred.

**Doc/comment accuracy.** The module doc appended to `equivalence_phase48.rs` (lines 10–13) accurately describes the faer/nalgebra backend split and the ignore policy. The inline comment before `svd_sign_fpca_two_matrix_bit_identical` in `equivalence_phase49.rs` (lines 380–383) accurately cites the divergence value (`-0.0 → 4.18e-16`), names the Phase 90 diagnosis, and explicitly marks the unguarded PACE sibling.

All reviewed files meet quality standards. No issues found.

---

_Reviewed: 2026-09-09_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: standard_
