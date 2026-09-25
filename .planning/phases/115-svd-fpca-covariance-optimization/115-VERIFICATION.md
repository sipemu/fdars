---
phase: 115-svd-fpca-covariance-optimization
verified: 2026-09-23T15:30:00Z
status: passed
score: 6/6 must-haves verified
covered_files:
  - .planning/REQUIREMENTS.md
  - .planning/phases/115-svd-fpca-covariance-optimization/115-01-PLAN.md
  - .planning/phases/115-svd-fpca-covariance-optimization/115-01-SUMMARY.md
  - .planning/phases/115-svd-fpca-covariance-optimization/115-02-PLAN.md
  - .planning/phases/115-svd-fpca-covariance-optimization/115-02-SUMMARY.md
  - .planning/phases/115-svd-fpca-covariance-optimization/115-03-PLAN.md
  - .planning/phases/115-svd-fpca-covariance-optimization/115-03-SUMMARY.md
  - fdars-core/src/irreg_fdata/face.rs
  - fdars-core/src/irreg_fdata/smoothing.rs
  - fdars-core/tests/equivalence_phase115.rs
re_verification: null
covered_digest: "v1:sha256:4330cce9224ba87b927a2f65370549810a929b4216bd5acb232e8d51a79626f9"
behavior_unverified: 0
overrides_applied: 0
---

# Phase 115: SVD/FPCA + Covariance Optimization Verification Report

**Phase Goal:** Optimize the next-ranked hot paths (SVD/FPCA + covariance surface) numeric-identically, and land the consolidated cross-cutting proof (OPT-03).
**Verified:** 2026-09-23T15:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `sym_cov_irreg` exists in `smoothing.rs`, `face.rs:149` calls it (not `cov_irreg`), and public `cov_irreg` is unchanged with no public API change | ✓ VERIFIED | `grep -n "sym_cov_irreg"` → line 182 (`pub(crate) fn`); `face.rs:149` → `let raw_cov = sym_cov_irreg(...)`; `cov_irreg` still `pub fn` at line 111; `mod.rs` re-exports only `cov_irreg`, not `sym_cov_irreg` |
| 2 | Golden fixture `equivalence_phase115.rs` passes UNCHANGED at ≤1e-10 in debug and release | ✓ VERIFIED | `cargo test --test equivalence_phase115 --features linalg,parallel` → `test result: ok. 1 passed` (run live); `cargo test --release --features linalg --test equivalence_phase115` → `test result: ok. 1 passed` (run live) |
| 3 | Committed before/after criterion evidence exists: `115-baseline-criterion/` + `115-after-criterion/` + `COMPARISON.md` with ~3.45× non-overlapping-CI verdict | ✓ VERIFIED | Both `estimates.json` present and valid (BEFORE mean=268.2ms CI=[260.1,275.9]; AFTER mean=77.8ms CI=[76.7,79.4]); AFTER CI upper (79.4ms) < BEFORE CI lower (260.1ms) — non-overlapping; COMPARISON.md records ~3.45× speedup, 71.0% reduction |
| 4 | OPT-03 consolidated proof committed: `115-opt03-proof/OPT03-PROOF.md` exists and records all battery gates green | ✓ VERIFIED | File exists at `.planning/phases/115-svd-fpca-covariance-optimization/115-opt03-proof/OPT03-PROOF.md`; committed at `e1f3093d`; records release test (215 passed), 6-module determinism guardrail, 5 build variants, pmat 0 new violations, fmt+clippy clean |
| 5 | `sym_cov_irreg` is substantive: upper-triangle loop, single kernel table, verbatim accumulation body, bit-copy mirror — not a stub | ✓ VERIFIED | Read `smoothing.rs:182-235`: single `w` table built at lines 197-202; upper-triangle `for si in 0..m { for ti in si..m { ... } }` at lines 207-208; inner body lines 213-221 word-for-word matches `cov_irreg` order; `cov[ti+si*m] = val` bit-copy mirror at line 230 |
| 6 | OPT-03: release test green (215), determinism guardrail (6 modules), 5 build variants, pmat 0 new fdars-core/src violations, fmt+clippy clean | ✓ VERIFIED | OPT03-PROOF.md records each gate with exit status 0; commit `e1f3093d` present in git log |

**Score:** 6/6 truths verified (0 present, behavior-unverified)

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/irreg_fdata/smoothing.rs` | New private `sym_cov_irreg` (upper-triangle + bit-copy mirror, single kernel table) | ✓ VERIFIED | `pub(crate) fn sym_cov_irreg` at line 182; 54-line substantive implementation; not exported publicly |
| `fdars-core/src/irreg_fdata/face.rs` | Line 148/149 calls `sym_cov_irreg` not `cov_irreg`; no public API change | ✓ VERIFIED | Line 28: `use crate::irreg_fdata::smoothing::sym_cov_irreg`; line 149: `let raw_cov = sym_cov_irreg(ifd, grid, bandwidth)` |
| `fdars-core/tests/equivalence_phase115.rs` | Golden-value fixture, n=10 m=11, tol 1e-10, symmetry invariant | ✓ VERIFIED | File exists; 1 test `golden_face_covariance_n10_m11` passes green in debug and release |
| `115-baseline-criterion/perf_face_covariance__n200_m30/estimates.json` | Positive mean + valid CI from un-optimized code | ✓ VERIFIED | mean=268,166,197.6ns; CI=[260,142,509.9, 275,883,662.2]; hi>lo; valid |
| `115-baseline-criterion/perf_face_covariance__n200_m30/BASELINE.md` | Command, mean_ns, CI bounds, commit SHA, features, machine | ✓ VERIFIED | File present; records SHA 328215e7, bench command, mean/CI, machine (i9-13900H) |
| `115-after-criterion/perf_face_covariance__n200_m30/estimates.json` | Positive mean + valid CI from optimized code | ✓ VERIFIED | mean=77,786,252.2ns; CI=[76,709,968.0, 79,372,180.8]; valid |
| `115-after-criterion/perf_face_covariance__n200_m30/COMPARISON.md` | BEFORE vs AFTER mean_ns, both CIs, percent, overlap verdict | ✓ VERIFIED | Present; records ~3.45× speedup, non-overlapping CI verdict, honest methodology note |
| `115-svd-rebench/SVD-VERDICT.md` | SVD near-optimal verdict with tightened CI widths | ✓ VERIFIED | Present; CI widths 0.15%–1.2% across 3 cells; concludes faer already production path |
| `115-svd-rebench/*.json` | SVD re-benchmark estimates (at least 2 cells) | ✓ VERIFIED | 3 files: `svd_n500_m50_estimates.json`, `svd_n1000_m200_estimates.json`, `svd_n500_m500_estimates.json` |
| `115-opt03-proof/OPT03-PROOF.md` | Committed consolidated OPT-03 proof record | ✓ VERIFIED | Present; commit `e1f3093d`; records all 10 gates with exit status + output tails |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `face.rs:149` | `smoothing.rs:sym_cov_irreg` | `use crate::irreg_fdata::smoothing::sym_cov_irreg` (line 28) | ✓ WIRED | Import present; call site confirmed at line 149 |
| `sym_cov_irreg` (private) | NOT `mod.rs` public re-exports | `pub(crate)` visibility only | ✓ VERIFIED | `mod.rs` re-exports only `cov_irreg`; `sym_cov_irreg` absent from public surface |
| `BEFORE baseline` → `AFTER criterion` | `COMPARISON.md` | Shared cell `perf_face_covariance/n200_m30`; same bench command | ✓ WIRED | COMPARISON.md cites BEFORE SHA 328215e7, AFTER SHA c0e84bd8; same cell reproduced |
| `equivalence_phase115.rs` | `face_covariance` via `sym_cov_irreg` | Public `face_covariance` entry point in test | ✓ WIRED | Golden fixture exercises the optimized code path; passes in debug and release |

---

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `sym_cov_irreg` | `cov` (m×m matrix) | Computes from `ifd.values` / `ifd.argvals` via `mean_irreg` + kernel accumulation | Yes — full computation, no static return | ✓ FLOWING |
| `face_covariance` | `raw_cov` | `sym_cov_irreg(ifd, grid, bandwidth)` real result | Yes — downstream sandwich smoother applied | ✓ FLOWING |
| BEFORE `estimates.json` | `mean.point_estimate` | Criterion benchmark of un-optimized code | Yes — 268.2ms measured value | ✓ FLOWING |
| AFTER `estimates.json` | `mean.point_estimate` | Criterion benchmark of optimized code | Yes — 77.8ms measured value | ✓ FLOWING |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Golden fixture green in debug (numeric identity at ≤1e-10) | `cargo test -p fdars-core --features linalg,parallel --test equivalence_phase115` | `test result: ok. 1 passed` | ✓ PASS |
| Golden fixture green in release (float-reassociation guard) | `cargo test -p fdars-core --release --features linalg --test equivalence_phase115` | `test result: ok. 1 passed` | ✓ PASS |

---

### Probe Execution

No `probe-*.sh` files declared for this phase. Behavioral spot-checks above serve as the phase's runnable evidence.

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| OPT-02 | 115-01, 115-02 | Optimize next-ranked hot paths (SVD/FPCA + covariance surface), non-breaking + numeric-identical, with committed before/after criterion evidence | ✓ SATISFIED | `sym_cov_irreg` implemented in smoothing.rs; face.rs:149 calls it; golden fixture passes ≤1e-10 debug+release; BEFORE baseline 268ms, AFTER 77.8ms, ~3.45× non-overlapping CI speedup; SVD documented near-optimal (faer already production path) |
| OPT-03 | 115-03 | Every optimization commit preserves numeric identity and all build variants — release test green, determinism guardrail green, all five builds compile, pmat quality-gate stays green | ✓ SATISFIED | OPT03-PROOF.md committed at e1f3093d; 215 release tests pass; 6-module guardrail green; 5 build variants green (incl. serde + WASM); pmat 0 new fdars-core/src violations; fmt+clippy clean |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| (none) | — | No TBD/FIXME/XXX/TODO/HACK/PLACEHOLDER markers found in modified files | — | — |

---

### Human Verification Required

None. All must-haves are observable from the codebase and verified programmatically. The criterion speedup claim (~3.45×, non-overlapping CIs) is verified by the committed `estimates.json` values and `COMPARISON.md` — no live bench re-run required for verification.

---

### Gaps Summary

No gaps. All 6 must-have truths verified, all required artifacts present and substantive, all key links wired, both requirements (OPT-02, OPT-03) satisfied.

---

_Verified: 2026-09-23T15:30:00Z_
_Verifier: Claude (gsd-verifier)_

## Re-fingerprint 2026-09-25

Covered-file digest recomputed by the v0.47.0 autonomous orchestrator before the milestone audit. Cause of staleness: only .planning/REQUIREMENTS.md changed (later phases marked their requirements complete) — false-stale. No verified truth was invalidated.
