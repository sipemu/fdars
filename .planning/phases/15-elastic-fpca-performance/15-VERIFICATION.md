---
phase: 15-elastic-fpca-performance
verified: 2026-08-12T00:00:00Z
status: passed
score: 4/4 must-haves verified
behavior_unverified: 0
overrides_applied: 0
re_verification: false
---

# Phase 15: Elastic-FPCA Performance — Verification Report

**Phase Goal:** The elastic-FPCA critical path runs in parallel under the `parallel` feature, cutting wall-clock for N≥50 while producing output numerically equivalent to the sequential path.
**Verified:** 2026-08-12
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

The four ROADMAP success criteria are evaluated against the live codebase and confirmed by running the test suite.

| #  | Truth | Status | Evidence |
|----|-------|--------|----------|
| 1  | Three per-curve loops in `elastic_fpca.rs` execute via `iter_maybe_parallel!(0..n)` under `parallel` feature and compile/run sequentially when feature is off | VERIFIED | `iter_maybe_parallel!` present at lines 715, 738, 800; `#[cfg(feature = "parallel")] use rayon::iter::ParallelIterator` at line 17-18 gates sequential fallback; 20/20 tests pass with `--features linalg,parallel` AND `--features linalg` (parallel off) |
| 2  | Inline `#[cfg(test)]` equivalence tests confirm `vert_fpca`/`joint_fpca` scores and eigenvalues match within numerical tolerance | VERIFIED | Five named tests present and green: `test_vert_fpca_parallel_equiv` (line 1335), `test_joint_fpca_parallel_equiv` (line 1371); pure disjoint writes yield bit-identical output asserted via `assert_eq!` on `f64`; 20/20 elastic_fpca tests green in both feature configurations |
| 3  | Light `:764` body guarded by a named N≥50 size threshold, parallel only where dispatch pays back | VERIFIED | `const SCORES_PARALLEL_THRESHOLD: usize = 50` at line 28 with rationale doc comment; `if n >= SCORES_PARALLEL_THRESHOLD` outer-if at line 796 (single branch, not per-k); `test_scores_threshold` (line 1250) exercises both the N=10 sequential branch and N=51 parallel branch with bit-exact assertions |
| 4  | Change is additive and non-breaking: `vert_fpca`/`joint_fpca` signatures unchanged, no new dependencies, feasibility demonstrated | VERIFIED | Public signatures confirmed unchanged at lines 101, 286; `rayon` was pre-existing optional dep (`parallel = ["rayon"]` in Cargo.toml); clippy clean under `--all-targets --features linalg,parallel -- -D warnings` |

**Score:** 4/4 truths verified

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/elastic_fpca.rs` | Three loops parallelized + 6 named equivalence tests | VERIFIED | 1498 lines; `iter_maybe_parallel!` at lines 715, 738, 800; `SCORES_PARALLEL_THRESHOLD` at line 28; all 6 test functions present and green |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `shooting_vectors_from_psis` (line 715) | `iter_maybe_parallel!` macro | `iter_maybe_parallel!(0..n).map(...).collect()` then sequential assign | WIRED | Confirmed in source; collect-then-assign pattern mirrors `alignment/set.rs::align_to_target` |
| `build_augmented_srsfs` (line 738) | `iter_maybe_parallel!` macro | Same collect-then-assign pattern | WIRED | Confirmed in source |
| `svd_scores_and_eigenvalues` (line 796) | `SCORES_PARALLEL_THRESHOLD` guard + `iter_maybe_parallel!` | `if n >= 50` outer-if, parallel arm at line 800 | WIRED | Confirmed; threshold outer-if at function level (not per-k) |
| `vert_fpca` (line 101) | `shooting_vectors_from_psis`, `build_augmented_srsfs`, `svd_scores_and_eigenvalues` | Called internally at lines 221, 128/406, 236/339 | WIRED | Public entry points call all three parallelized internal functions |
| `joint_fpca` (line 286) | Same three internal functions | Called internally | WIRED | Confirmed |

---

### Per-Criterion Test Map (PERF-04-A…F)

| Criterion | Test Name | Parallel ON | Parallel OFF |
|-----------|-----------|-------------|--------------|
| PERF-04-A | `test_shooting_vectors_parallel_equiv` | PASS | PASS |
| PERF-04-B | `test_augmented_srsfs_parallel_equiv` | PASS | PASS |
| PERF-04-C | `test_scores_threshold` | PASS | PASS |
| PERF-04-D | `test_vert_fpca_parallel_equiv` | PASS | PASS |
| PERF-04-E | `test_joint_fpca_parallel_equiv` | PASS | PASS |
| PERF-04-F | Full elastic_fpca suite (existing 15 + 5 new) | 20/20 PASS | 20/20 PASS |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 20 elastic_fpca tests green with parallel ON | `cargo test -p fdars-core --features linalg,parallel -- elastic_fpca` | 20 passed; 0 failed | PASS |
| 20 elastic_fpca tests green with parallel OFF | `cargo test -p fdars-core --features linalg -- elastic_fpca` | 20 passed; 0 failed | PASS |
| Clippy clean with parallel ON | `cargo clippy --all-targets -p fdars-core --features linalg,parallel -- -D warnings` | Finished with no warnings | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|---------|
| PERF-04 | 15-01-PLAN.md | Three per-curve elastic-FPCA loops parallelized via `iter_maybe_parallel!` with N≥50 guard, equivalence-tested | SATISFIED | All three loops wired; `SCORES_PARALLEL_THRESHOLD = 50`; 6 named tests green in both feature configurations |

---

### Anti-Patterns Found

No debt markers (`TBD`, `FIXME`, `XXX`, `TODO`, `HACK`, `PLACEHOLDER`) found in `fdars-core/src/elastic_fpca.rs`.

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| — | — | None | — | — |

---

### Commits Verified

All four commits declared in SUMMARY are present in git history:

| Hash | Description |
|------|-------------|
| `9d5618db` | feat(15-01): parallelize shooting_vectors_from_psis via iter_maybe_parallel! (PERF-04-A) |
| `5e413f9a` | feat(15-01): parallelize build_augmented_srsfs via iter_maybe_parallel! (PERF-04-B) |
| `849be577` | feat(15-01): guard svd_scores_and_eigenvalues with N>=50 threshold (PERF-04-C) |
| `57ea8556` | test(15-01): add vert_fpca and joint_fpca end-to-end equivalence tests (PERF-04-D/E/F) |

---

### Human Verification Required

None. All success criteria were verifiable programmatically. The wall-clock speedup figure (~4–5x at N≥50) is explicitly scoped as manual-only in the VALIDATION.md (governor unpinned, LOW-CONFIDENCE per audit); the gate is feasibility + numerical equivalence, not a pinned speedup number.

---

### Summary

Phase 15 fully achieved its goal. The three per-curve loops in `elastic_fpca.rs` — `shooting_vectors_from_psis` (line 715), `build_augmented_srsfs` (line 738), and `svd_scores_and_eigenvalues` (line 796) — are parallelized via the existing `iter_maybe_parallel!` macro using the collect-then-assign pattern that avoids any parallel write into the shared column-major `FdMatrix`. The light `:764` body is correctly guarded by the named constant `SCORES_PARALLEL_THRESHOLD = 50`. All six PERF-04-A…F equivalence tests pass under both `--features linalg,parallel` (parallel ON) and `--features linalg` (parallel OFF). Clippy is clean. No public signatures changed, no new dependencies added. No anti-patterns or debt markers found.

---

_Verified: 2026-08-12_
_Verifier: Claude (gsd-verifier)_
