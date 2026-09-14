---
phase: 106-top-level-complexity-hotspots
plan: "04"
subsystem: irreg_fdata
tags: [pmat, complexity-refactor, pace-fpca, sparse-functional-data, blup]

requires:
  - phase: 106-01
    provides: "Proven extract->test->pmat->commit loop and split-validate pattern"

provides:
  - "pace_fpca decomposed into 9 private helpers; main function drops from cyc 37/cog 78 to cyc 4/cog 5"
  - "All extracted helpers below pmat cognitive <=25 / cyclomatic <=30 error thresholds"
  - "BLUP closure extracted as named fn (pace_fpca_blup_curve) with all captures as explicit Send params"
  - "Both parallel (--features linalg,parallel) and sequential (--features linalg) builds confirmed clean"
  - "QUAL-01 pace_fpca hotspot cleared"

affects: [106-05-if-any, 109]

actuals:
  tokens: 13000
  tasks: 2
  commits: 1
  plan_head_before: 91eb600dbf5cc7d186121fd9b9c860479150c137

tech-stack:
  added: []
  patterns:
    - "Split-validate pattern (Phase 106 established): pace_fpca_validate delegates to pace_fpca_validate_dims (cog 15) + pace_fpca_validate_config (cog 24), both below error threshold 25"
    - "Named-fn BLUP extraction: BLUP closure body extracted as pace_fpca_blup_curve with every capture as explicit parameter -- makes fn Send-compatible with iter_maybe_parallel! rayon branch"
    - "Moment pipeline helper: pace_fpca_smooth_moments wraps steps 1-3 (mean + cov + eigendecomp) into one Result-returning fn, reducing parent nesting"
    - "Decompose sigma_yi assembly: pace_fpca_build_sigma_yi + pace_fpca_resolve_sigma_yi split the Phi_i diag(lambda) Phi_i^T + sigma2*I build from the ridge-stabilise probe, keeping each helper simple"
    - "Band helper: pace_fpca_confidence_bands handles all three Yao 2005 eq.3.2 steps (sigma_inv_phi_lam + A_i + Omega_i pointwise variance)"

key-files:
  created: []
  modified:
    - fdars-core/src/pace_fpca.rs

key-decisions:
  - "Split-validate pattern applied again: pace_fpca_validate_dims groups the 4 dimension/count guards (n==0, per-curve n_pts<2, m<2, ncomp==0); pace_fpca_validate_config groups the 3 scalar guards (bandwidth, sigma2, alpha) plus work_grid finite+sorted. Combined cog 39 reduced to max 24 / 15."
  - "BLUP closure extracted verbatim as pace_fpca_blup_curve(i, data, work_grid, mean, ef_cols, eigenvalues, eigenfunctions, actual_ncomp, m, z, sigma2) -- every captured variable becomes an explicit parameter, matching RESEARCH.md Pitfall 4. No algorithmic change."
  - "iter_maybe_parallel! macro kept in parent pace_fpca body with .map(|i| pace_fpca_blup_curve(i, ...)); only the closure body moved out. This preserves rayon feature gating and makes the parent function trivially short."
  - "pace_fpca_confidence_bands annotated with #[allow(clippy::too_many_arguments)] because the 10-parameter signature is necessary for the Send constraint -- an intermediate struct would require either interior mutability or Clone copies."

patterns-established:
  - "BLUP-closure extraction: when a closure passed to iter_maybe_parallel! exceeds threshold, extract it as a named fn with all captures as explicit &T parameters -- the fn is then Send automatically"
  - "Moment-pipeline helper: group pure-computation steps (mean + cov + eigen) that return a single aggregate tuple into one helper -- keeps the entry-point function under 20 lines"

requirements-completed: [QUAL-01]

coverage:
  - id: D1
    description: "pace_fpca cyclomatic complexity reduced from 37 to 4 and cognitive from 78 to 5 via 8 private helpers; all 15 pace_fpca::tests pass and confirm numeric-identical BLUP scores, fitted trajectories, and confidence bands"
    requirement: QUAL-01
    verification:
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel --lib -- pace_fpca::tests (15 tests)"
        status: pass
      - kind: other
        ref: "pmat analyze complexity --path fdars-core/src/pace_fpca.rs: pace_fpca cyc=4 cog=5; max any fn cyc=11 cog=24 (no errors)"
        status: pass
    human_judgment: false
  - id: D2
    description: "Both parallel and sequential feature gates compile and test clean; clippy --all-targets clean"
    requirement: QUAL-01
    verification:
      - kind: other
        ref: "cargo build --features linalg (sequential): clean"
        status: pass
      - kind: other
        ref: "cargo test --features linalg -- pace_fpca::tests: 15/15 pass"
        status: pass
      - kind: other
        ref: "cargo clippy --all-targets --features linalg,parallel -- -D warnings: clean"
        status: pass
      - kind: other
        ref: "cargo fmt --check: clean"
        status: pass
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel (full suite: 2942 lib + 215 doc tests)"
        status: pass
    human_judgment: false

duration: 18min
completed: 2026-09-14
status: complete
---

# Phase 106 Plan 04: pace_fpca Complexity Decomposition Summary

**pace_fpca decomposed from cyclomatic 37/cognitive 78 into 8+1 private helpers, dropping the entry-point to cyc 4/cog 5 with numeric-identical output verified by 15 tests under both feature gates.**

## Performance

- **Duration:** ~18 min
- **Tasks completed:** 2/2
- **Commits:** 1 (refactor commit; Task 2 was verification-only, no code change)

## pmat Before/After

| Function | Before (cyc/cog) | After (cyc/cog) | Status |
|----------|------------------|-----------------|--------|
| `pace_fpca` | 37 / 78 | 4 / 5 | CLEARED |
| `pace_fpca_validate` | — | 1 / 0 | NEW, clean |
| `pace_fpca_validate_dims` | — | 6 / 15 | NEW, clean |
| `pace_fpca_validate_config` | — | 11 / 24 | NEW, warning only (< error threshold 25) |
| `pace_fpca_smooth_moments` | — | 3 / 6 | NEW, clean |
| `pace_fpca_blup_curve` | — | 4 / 4 | NEW, clean |
| `pace_fpca_build_sigma_yi` | — | 4 / 6 | NEW, clean |
| `pace_fpca_resolve_sigma_yi` | — | 3 / 3 | NEW, clean |
| `pace_fpca_confidence_bands` | — | 9 / 15 | NEW, clean |
| `eigendecompose_cov` | 9 / 15 | 9 / 15 | UNCHANGED, clean |
| `standard_normal_quantile` | 2 / 1 | 2 / 1 | UNCHANGED, clean |

File max: cyclomatic 11, cognitive 24 — **no pmat errors** (thresholds: cyc >30 error, cog >25 error).
Two warnings at cognitive 23-24 range — acceptable (warning threshold >23; error threshold >25).

## Accomplishments

- `pace_fpca` entry point reduced to 4 lines of delegation + setup + parallel map + result assembly
- The large BLUP closure (135 lines) extracted as `pace_fpca_blup_curve` with all 11 captures as explicit parameters — confirmed `Send` under both rayon and sequential `iter_maybe_parallel!` branches
- Validation split into `_dims` (dimension/count guards) + `_config` (scalar param + work-grid ordering), mirroring the split-validate pattern established in Plan 106-01
- Full test suite (2942 lib + 215 doctests) green; no public API/signature/visibility change

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] pace_fpca_validate itself still scored cognitive 39 after first extraction**

- **Found during:** Task 1 pmat check after initial extraction
- **Issue:** The single `pace_fpca_validate` holding all 7 guards scored cog 39 — above the error threshold 25. The plan anticipated this (RESEARCH.md Pitfall 3) but listed "if needed" for the smooth-moments helper; the validate split was the actual fix needed.
- **Fix:** Applied the split-validate pattern (established in Plan 106-01) — split into `pace_fpca_validate_dims` (4 dimension/count guards, cog 15) and `pace_fpca_validate_config` (3 scalar guards + work_grid loops, cog 24). Both below error threshold.
- **Files modified:** `fdars-core/src/pace_fpca.rs`
- **Commit:** 4a8f999c (included in refactor commit)

## Self-Check: PASSED

- `fdars-core/src/pace_fpca.rs`: FOUND
- Commit `4a8f999c`: FOUND
- `106-04-SUMMARY.md`: FOUND
- All 15 pace_fpca::tests: pass (parallel + sequential)
- Full suite (2942 lib + 215 doc): pass
- clippy --all-targets: clean
- fmt --check: clean
- pmat: no error-level violations in pace_fpca.rs
