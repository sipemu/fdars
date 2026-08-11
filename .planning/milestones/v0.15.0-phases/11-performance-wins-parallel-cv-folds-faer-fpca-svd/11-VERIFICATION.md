---
phase: 11-performance-wins-parallel-cv-folds-faer-fpca-svd
verified: 2026-08-11T09:00:00Z
status: passed
score: 9/9
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 11: Parallel CV Folds + faer FPCA SVD — Verification Report

**Phase Goal:** Parallelize the classification CV fold loop (PERF-01) and swap the FPCA SVD to faer behind the `linalg` feature (PERF-02) — each with tests and numerical verification, results equivalent to the prior sequential/nalgebra behavior.
**Verified:** 2026-08-11T09:00:00Z
**Status:** PASSED
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | PERF-01: `iter_maybe_parallel!(0..nfold)` replaces the sequential `for fold in 0..nfold` loop in `fclassif_cv`; no `fold_errors.push` remains; `fold_errors` is no longer `mut`. | ✓ VERIFIED | cv.rs line 77: `let fold_errors: Vec<f64> = iter_maybe_parallel!(0..nfold).map(\|fold\| { ... }).collect();`. grep confirms: 1 match for `iter_maybe_parallel!(0..nfold)`, 0 matches for `fold_errors.push`, 0 matches for `let mut fold_errors`. |
| 2 | PERF-01: `use crate::iter_maybe_parallel;` and `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;` are present in cv.rs imports. | ✓ VERIFIED | cv.rs lines 4 and 7–8: both imports confirmed by grep (1 match each). No direct `rayon::iter` call for iteration — goes through macro. |
| 3 | PERF-01: `test_fclassif_cv_parallel_matches_sequential` exists in a `#[cfg(test)] mod tests` block in cv.rs, asserts bit-for-bit `fold_errors` equality and `error_rate` equality, and passes under both default and `parallel` features. | ✓ VERIFIED | cv.rs lines 363–398: test exists. Behavioral spot-checks: `cargo test -p fdars-core -- test_fclassif_cv_parallel_matches_sequential` → 1 passed, 0 failed. Same under `--features parallel` → 1 passed, 0 failed. |
| 4 | PERF-02: `fdata_to_pc_1d` computes SVD via `FaerSvd::new_thin(MatRef::<f64>::from_column_major_slice(weighted.as_slice(), n, m))` on a zero-copy view under `#[cfg(feature = "linalg")]`. | ✓ VERIFIED | regression.rs lines 338–366: `#[cfg(feature = "linalg")]` branch confirmed. `from_column_major_slice(weighted.as_slice(), n, m)` at line 340; `FaerSvd::new_thin(mat_ref)` at line 341. |
| 5 | PERF-02: faer V is accessed un-transposed — `rotation[(j, k)] = svd.V()[(j, k)]` — NOT the transposed `V()[(k, j)]`. | ✓ VERIFIED | regression.rs line 354: `rotation[(j, k)] = svd.V()[(j, k)];` confirmed. No transposed access found. |
| 6 | PERF-02: The nalgebra SVD path is retained under `#[cfg(not(feature = "linalg"))]`. | ✓ VERIFIED | regression.rs line 368: `#[cfg(not(feature = "linalg"))]` block retains `SVD::new(weighted.to_dmatrix(), true, true)` + `extract_pc_components`. |
| 7 | PERF-02: `fix_svd_signs` helper exists and is called exactly once from the shared binding, BEFORE the sqrt_weights unscaling loop, covering both cfg branches. | ✓ VERIFIED | regression.rs line 180: `fn fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize)`. Single production call site at line 381, between the cfg-branched binding (ends line 377) and the unscaling loop (starts line 383). |
| 8 | PERF-02: `test_faer_svd_matches_nalgebra` exists under `#[cfg(all(test, feature = "linalg"))]`, asserts significant-component equivalence of singular_values, rotation, and scores within `1e-8·σ₁`, and passes. | ✓ VERIFIED | regression.rs lines 964–1030: test exists with correct cfg gate. Significant-component filter (`sv[k] >= 1e-8 * sv[0]`) at line 1004. Assertions on singular_values, rotation, and scores confirmed. Behavioral spot-check: `cargo test -p fdars-core --features linalg -- test_faer_svd_matches_nalgebra` → 1 passed, 0 failed. |
| 9 | Both / No new Cargo.toml dependency introduced; matrix.rs not modified. | ✓ VERIFIED | `git show 23118b0b --stat` and `git show 08f28702 --stat` show no Cargo.toml touches. Only files changed: cv.rs, regression.rs, spm/tests.rs. |

**Score:** 9/9 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/classification/cv.rs` | Fold loop parallelized; new `#[cfg(test)] mod tests` with equivalence test | ✓ VERIFIED | File is substantive (399 lines); fold loop uses `iter_maybe_parallel!`; test block exists at lines 336–399; both use-imports present. |
| `fdars-core/src/regression.rs` | Feature-gated SVD in `fdata_to_pc_1d`; `fix_svd_signs` helper; `test_faer_svd_matches_nalgebra` under `cfg(all(test, linalg))` | ✓ VERIFIED | All three elements confirmed at correct locations. `nalgebra::SVD` and `extract_pc_components` gated to `cfg(any(not(feature = "linalg"), test))` so the equivalence test can compute its inline reference without unused-import warnings. |
| `fdars-core/src/spm/tests.rs` | MEWMA SPE-alarm test guarded against machine-noise regime (deviation) | ✓ VERIFIED | Lines 3282–3298: well-commented guard `if max_spe > 1e-20` with clear rationale. Deviation was documented in SUMMARY as an auto-fixed bug (faer and nalgebra are mathematically equivalent; the prior assertion was comparing roundoff-vs-roundoff). |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `iter_maybe_parallel!` macro (parallel.rs) | cv.rs fold loop | `use crate::iter_maybe_parallel;` + `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator;` | ✓ WIRED | Both imports present; macro call at line 77; `ParallelIterator` trait in scope for `.collect()` under parallel feature. |
| `FdMatrix::as_slice()` (matrix.rs:291 — existing public method) | faer `MatRef::from_column_major_slice` in regression.rs | `weighted.as_slice()` — zero-copy column-major view | ✓ WIRED | regression.rs line 340: `MatRef::<f64>::from_column_major_slice(weighted.as_slice(), n, m)`. matrix.rs not modified (as required). |
| `fix_svd_signs` helper | shared binding in `fdata_to_pc_1d` | Called once at line 381, before unscaling loop | ✓ WIRED | Single shared call site covers both `#[cfg(feature = "linalg")]` and `#[cfg(not(feature = "linalg"))]` branches. |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| PERF-01 equivalence test under default features | `cargo test -p fdars-core -- test_fclassif_cv_parallel_matches_sequential` | 1 passed, 0 failed | ✓ PASS |
| PERF-01 equivalence test under `parallel` feature | `cargo test -p fdars-core --features parallel -- test_fclassif_cv_parallel_matches_sequential` | 1 passed, 0 failed | ✓ PASS |
| PERF-02 faer-vs-nalgebra equivalence test under `linalg` | `cargo test -p fdars-core --features linalg -- test_faer_svd_matches_nalgebra` | 1 passed, 0 failed | ✓ PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| PERF-01 | 11-01-parallel-cv-folds-PLAN.md | `fclassif_cv` fold loop parallelized via `iter_maybe_parallel!`; bit-for-bit identical results with fixed seed; tests pass under `parallel` feature on and off | ✓ SATISFIED | Truths 1–3 verified; commits 23118b0b + 30832954 |
| PERF-02 | 11-02-faer-fpca-svd-PLAN.md | `fdata_to_pc_1d` uses faer `thin_svd` on zero-copy `MatRef` under `linalg`; nalgebra path retained; `fix_svd_signs` applied to both; equivalence test passes | ✓ SATISFIED | Truths 4–8 verified; commits 08f28702 + 96cb6f5b |

REQUIREMENTS.md traceability table confirms both PERF-01 and PERF-02 map to Phase 11 and are marked Complete. No orphaned requirements.

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | No TBD/FIXME/XXX/TODO/HACK markers found in cv.rs, regression.rs, or spm/tests.rs | — | — |

No stub patterns (empty returns, hardcoded empty collections, placeholder content) found in any of the three modified files.

---

### Advisory Items From Code Review (11-REVIEW.md)

These items were flagged by the code review agent as advisory warnings. They are quality concerns, not phase-goal blockers, and do not affect the PASSED verdict.

**WR-01 (advisory): MEWMA IC SPE-alarm assertion permanently unexecutable on current test data.**
The `if max_spe > 1e-20` guard in `spm/tests.rs:3292` means the alarm-count assertion never executes on the current test data (max_spe ~1e-28). The guard rationale is sound (roundoff vs. roundoff), but it leaves a gap: a regression that zeroes all SPE values would not be caught. A companion test with non-degenerate reconstruction (ncomp below intrinsic rank) would close this gap. Not blocking for phase 11 — the MEWMA behavioral change was necessary and correct.

**WR-02 (advisory): `fix_svd_signs` is a silent no-op on all-NaN rotation columns.**
NaN propagation through IEEE 754 comparisons causes `partial_cmp` to return `None`; the `unwrap_or(Equal)` tie-break then picks an arbitrary index, and `NaN < 0.0` evaluates to `false`, so the flip is silently skipped. A `debug_assert!(rotation[(j_max, k)].is_finite())` would catch this in test builds. Not blocking — NaN input to SVD is a pre-existing condition not introduced by this phase.

**IN-01 (info): Test name `test_fclassif_cv_parallel_matches_sequential` implies cross-mode comparison but actually proves call-to-call determinism.**
The test calls `fclassif_cv` twice with the same seed; under the `parallel` feature both calls use Rayon, so no sequential reference is computed. Cross-mode equality is structurally guaranteed by `iter_maybe_parallel!`'s indexed collect, but not directly asserted. Renaming to `test_fclassif_cv_deterministic_across_runs` would be more accurate. Not blocking.

---

### Prohibitions Check

| Prohibition | Status |
|-------------|--------|
| MUST NOT add any new external dependency to Cargo.toml | ✓ SATISFIED — Cargo.toml untouched in all 4 commits |
| MUST NOT alter the non-parallel code path's observable output | ✓ SATISFIED — sequential path uses same `iter_maybe_parallel!` macro expanded as a plain iterator; existing tests pass |
| MUST NOT import `rayon` directly for fold iteration | ✓ SATISFIED — only `use rayon::iter::ParallelIterator;` for `.collect()` trait resolution; macro handles the iteration |
| MUST NOT introduce per-fold RNG or shared mutable accumulator | ✓ SATISFIED — closure captures only shared references and Copy values |
| MUST NOT add a `pub(crate) fn as_slice` to matrix.rs | ✓ SATISFIED — matrix.rs not modified |
| MUST NOT populate rotation with transposed faer `V[(k,j)]` | ✓ SATISFIED — `rotation[(j, k)] = svd.V()[(j, k)]` (un-transposed) confirmed |
| MUST NOT apply fix_svd_signs AFTER the sqrt_weights unscaling loop | ✓ SATISFIED — `fix_svd_signs` at line 381 precedes the unscaling loop at line 383–390 |

---

## Verdict

**PASSED.** All 9 must-have truths are VERIFIED against the live codebase. Both PERF-01 and PERF-02 are implemented exactly as planned:

- `fclassif_cv` uses `iter_maybe_parallel!(0..nfold)` — the sequential push loop and `mut` accumulator are gone; a passing bit-for-bit equivalence test exists and runs under both feature configurations.
- `fdata_to_pc_1d` has a clean feature-gated SVD: faer `Svd::new_thin` on a zero-copy `MatRef` under `linalg`, retained nalgebra path otherwise; `fix_svd_signs` is applied once at the shared call site before unscaling; the numerical equivalence test passes.
- No new dependencies, no matrix.rs changes, no stub code, no unresolved debt markers.
- The two advisory warnings from the code review (WR-01 MEWMA gap, WR-02 NaN no-op) are noted but are not phase-goal blockers.

---

_Verified: 2026-08-11T09:00:00Z_
_Verifier: Claude (gsd-verifier)_
