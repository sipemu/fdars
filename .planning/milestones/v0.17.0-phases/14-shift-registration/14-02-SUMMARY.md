---
phase: 14-shift-registration
plan: "02"
subsystem: alignment
tags: [shift-registration, registration-quality, functional-data, feat-07, least-squares, pairwise-correlation, sobolev]
status: complete

dependency_graph:
  requires:
    - phase: 14-01
      provides:
        - least_squares_shift_registration
        - ShiftRegistrationResult
        - shift.rs (needed for direction tests in Task 2)
  provides:
    - least_squares_score
    - pairwise_correlation_score
    - sobolev_least_squares_score
    - crate-root re-exports for all 5 plan-14 public items
  affects:
    - fdars-core/src/alignment/quality.rs
    - fdars-core/src/alignment/mod.rs
    - fdars-core/src/lib.rs

actuals:
  tokens: 11200
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - standalone-energy scoring (absolute L2 spread, not ratio-to-original)
    - Result<f64, FdarError> return on new quality functions (departs from raw-f64 neighbors)
    - precomputed row norms for O(n*m) pairwise correlation (not O(n^2*m))
    - gradient_uniform for Sobolev derivative term (same idiom as warp_smoothness)

key-files:
  created: []
  modified:
    - fdars-core/src/alignment/quality.rs
    - fdars-core/src/alignment/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "Standalone-energy form: scores measure absolute L2 spread, NOT ratio to unregistered data; avoids division-by-zero on constant original data (per CONTEXT.md D-Area2)"
  - "Result<f64, FdarError> on all three score functions — intentional deviation from raw-f64 neighbors (warp_complexity, warp_smoothness) to enable dimension validation"
  - "Precomputed row norms for pairwise_correlation_score: O(n*m) instead of O(n^2*m) per pair; NaN guard when norm product < 1e-15"
  - "All 6 FEAT-07-A...F tests placed inline in quality.rs #[cfg(test)] mod tests (matches project convention)"
  - "Tasks 1+2 committed together (single quality.rs change) then Task 3 (mod.rs + lib.rs) separately to keep re-export changes isolated"
  - "mod.rs shift re-export already present from plan 14-01 deviation — not duplicated; only quality score names added to existing pub use quality::{...} block"

patterns-established:
  - "Registration quality scores return Result<f64, FdarError>; doc note explains the deliberate divergence from raw-f64 neighbors"
  - "Pairwise correlation: precompute norms outside the pair loop to avoid redundant sqrt on every (i,k) pair"
  - "FdarError::InvalidParameter { parameter: 'n', message: '...' } for n<2 on pairwise scorer (Pitfall 5 pattern)"

requirements-completed: [FEAT-07]

coverage:
  - id: D1
    description: "least_squares_score: standalone L2 spread of registered curves around mean (Result<f64, FdarError>)"
    requirement: FEAT-07
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/quality.rs#test_ls_score_identical_curves"
        status: pass
      - kind: unit
        ref: "fdars-core/src/alignment/quality.rs#test_ls_score_drops_after_registration"
        status: pass
    human_judgment: false
  - id: D2
    description: "sobolev_least_squares_score: LS term + lambda * derivative-penalty, lambda=0 reproduces LS score"
    requirement: FEAT-07
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/quality.rs#test_sobolev_score_lambda_zero"
        status: pass
      - kind: unit
        ref: "fdars-core/src/alignment/quality.rs#test_sobolev_score_lambda_positive"
        status: pass
    human_judgment: false
  - id: D3
    description: "pairwise_correlation_score: mean functional Pearson correlation over n(n-1)/2 pairs, n<2 returns Err"
    requirement: FEAT-07
    verification:
      - kind: unit
        ref: "fdars-core/src/alignment/quality.rs#test_pairwise_corr_rises_after_registration"
        status: pass
      - kind: unit
        ref: "fdars-core/src/alignment/quality.rs#test_pairwise_corr_n1_error"
        status: pass
    human_judgment: false
  - id: D4
    description: "All five plan-14 public items re-exported at crate root (fdars_core::least_squares_shift_registration etc.)"
    requirement: FEAT-07
    verification:
      - kind: unit
        ref: "cargo build -p fdars-core --features linalg (clean)"
        status: pass
    human_judgment: false

duration: ~25min
completed: "2026-08-12"
---

# Phase 14 Plan 02: Registration-Quality Scores + Crate-Root Re-exports (FEAT-07) Summary

**Three standalone-energy registration-quality scorers added to alignment/quality.rs (Result-returning, Simpson-weighted), with all five plan-14 items re-exported at the crate root.**

## Performance

- **Duration:** ~25 min
- **Started:** 2026-08-12T12:22:10Z
- **Completed:** 2026-08-12T12:47:00Z
- **Tasks:** 3
- **Files modified:** 3

## Accomplishments

- `least_squares_score(registered, argvals)` — absolute L2 spread `(1/n) Σᵢ ∫(fᵢ − mean)² dt`, Simpson-weighted, returns `Result<f64, FdarError>` (FEAT-07-A, -B verified)
- `pairwise_correlation_score(registered, argvals)` — mean functional Pearson correlation over all n(n−1)/2 pairs; NaN guard; n<2 returns `Err(InvalidParameter)` (FEAT-07-C, -D verified)
- `sobolev_least_squares_score(registered, argvals, lambda)` — LS term + λ·derivative-spread; lambda=0 exactly equals LS score; lambda>0 adds non-negative penalty (FEAT-07-E, -F verified)
- All 6 FEAT-07-A…F tests green; `cargo test`, `cargo clippy -D warnings`, `cargo build` all pass
- Five new public items accessible at crate root: `fdars_core::{least_squares_score, least_squares_shift_registration, pairwise_correlation_score, sobolev_least_squares_score, ShiftRegistrationResult}`

## Task Commits

1. **Tasks 1+2: quality score functions + all FEAT-07 tests** — `3a876610` (feat)
2. **Task 3: re-exports in alignment/mod.rs and lib.rs** — `e73d474e` (feat)

## Files Created/Modified

- `fdars-core/src/alignment/quality.rs` — added `FdarError` import; appended `least_squares_score`, `sobolev_least_squares_score`, `pairwise_correlation_score`; appended `#[cfg(test)] mod tests` with 6 FEAT-07 tests
- `fdars-core/src/alignment/mod.rs` — extended `pub use quality::{...}` with three new score names (shift items already present from plan 14-01)
- `fdars-core/src/lib.rs` — inserted five new items alphabetically into the flat alignment `pub use alignment::{...}` block

## Decisions Made

- **Standalone-energy form** (per CONTEXT.md D-Area2): scores return absolute L2 spread, not a ratio to unregistered-data spread. Diverges from scikit-fda's `LeastSquares`/`PairwiseCorrelation`/`SobolevLeastSquares` scorers (which return ratios); documented in each function's rustdoc.
- **`Result<f64, FdarError>` returns** on all three functions — unlike the raw-`f64` neighbors (`warp_complexity`, `warp_smoothness`). Enables dimension validation; rustdoc notes the deliberate departure.
- **Precomputed row norms** for `pairwise_correlation_score`: compute each curve's L2 norm once (O(n·m)), then reuse across O(n²) pairs — avoids redundant `sqrt` per pair.
- **Tasks 1+2 in one commit**: both tasks modify `quality.rs` only; collapsing to one commit avoids a staging-only intermediate that wouldn't pass clippy (functions unused until re-exported).
- **mod.rs not re-adding shift**: plan 14-01 already added `mod shift;` and `pub use shift::{...}` as a Rule-3 deviation. Extending only the `pub use quality::{...}` block, not duplicating the shift line.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Tasks 1 and 2 committed together to satisfy clippy gate**

- **Found during:** Task 1 commit attempt
- **Issue:** The three new public functions in `quality.rs` are flagged as `dead_code` by clippy -D warnings until they are re-exported via `mod.rs`. The plan specifies separate commits for Task 1 (LS + Sobolev), Task 2 (pairwise + direction tests), and Task 3 (re-exports). Committing Task 1 alone with functions not yet re-exported triggers `error: function 'least_squares_score' is never used` — blocking the pre-commit hook.
- **Fix:** Implemented all quality.rs additions (Tasks 1+2) in one edit, then added the re-exports (Task 3) before making any commit. Result: two commits rather than three — `quality.rs` changes as one unit, `mod.rs + lib.rs` changes as the second.
- **Impact:** No functionality omitted. All 6 FEAT-07 tests and all three functions are in the single Task 1+2 commit. Task 3 re-exports remain a separate, clean commit.
- **Files modified:** `fdars-core/src/alignment/quality.rs` (Tasks 1+2), `fdars-core/src/alignment/mod.rs` + `fdars-core/src/lib.rs` (Task 3)
- **Commits:** `3a876610` (Tasks 1+2), `e73d474e` (Task 3)

---

**Total deviations:** 1 auto-fixed (Rule 3 — blocking clippy gate)
**Impact on plan:** One commit collapsed two tasks; all artifacts and tests delivered as specified.

## Issues Encountered

- First two commit attempts timed out at the default 120s bash tool limit — the pre-commit hook runs the full test suite (~2000 tests). Resolved by increasing the bash timeout to 600s.
- `cargo fmt` reformatted the staged `quality.rs` after initial staging; re-staged the post-fmt version before committing.

## Known Stubs

None — all three score functions are fully implemented and wired to real data.

## Threat Surface Scan

No new network endpoints, auth paths, file access patterns, or schema changes. The three new functions are pure-numeric (slice inputs, f64 output) with no I/O. Threat mitigations T-14-03, T-14-04, T-14-05 all implemented per plan:
- T-14-03: dimension/parameter validation at entry (`InvalidDimension` / `InvalidParameter`)
- T-14-04: NaN guard in `pairwise_correlation_score` (`if denom < 1e-15 { 0.0 }`)
- T-14-05: n≥2 guard in `pairwise_correlation_score` (`Err(InvalidParameter)` for n<2)

## Self-Check: PASSED

### Files exist
- FOUND: `fdars-core/src/alignment/quality.rs` (modified)
- FOUND: `fdars-core/src/alignment/mod.rs` (modified)
- FOUND: `fdars-core/src/lib.rs` (modified)

### Commits exist
- FOUND: `3a876610` — feat(14-02): implement registration-quality score functions (FEAT-07)
- FOUND: `e73d474e` — feat(14-02): re-export quality scores + shift items at alignment and crate root

## Next Phase Readiness

- Phase 14 (shift-registration) milestone is now complete: FEAT-06 (plan 14-01) + FEAT-07 (plan 14-02) both delivered
- All five plan-14 public items are accessible at the crate root
- Registration diagnostic workflow: `least_squares_shift_registration` → pass `registered_data` to `least_squares_score` / `pairwise_correlation_score` / `sobolev_least_squares_score` to measure quality improvement

---
*Phase: 14-shift-registration*
*Completed: 2026-08-12*
