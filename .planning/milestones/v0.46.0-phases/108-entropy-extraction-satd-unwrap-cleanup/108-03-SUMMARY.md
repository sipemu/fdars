---
phase: 108-entropy-extraction-satd-unwrap-cleanup
plan: "03"
subsystem: code-quality
tags: [pmat, entropy, refactor, private-helpers, rust]

requires:
  - phase: 108-01
    provides: tracer proof that pmat entropy -> commit loop works

provides:
  - private fn node_bbox in fem_smoothing.rs (single-fold bounding box)
  - private fn sin_sin_surface in fem_smoothing tests
  - private fn extract_scalar_row in explain_generic/friedman.rs
  - private fn subtract_fitted in elastic_regression/scalar_on_shape.rs
  - crate::helpers::extract_curves reused in metric/soft_dtw.rs (5 sites)
  - private fn extract_score_row in classification/fit.rs
  - pmat entropy: E-04 (fem_smoothing), E-05 (friedman), E-06 (soft_dtw), E-07 (scalar_on_shape), E-12 (classification/fit) all cleared

affects: [108-04, 109-complexity-tail, phase-109]

actuals:
  tokens: 4500
  tasks: 2
  commits: 3

tech-stack:
  added: []
  patterns:
    - "subtract_fitted / extract_score_row / extract_scalar_row: private per-file helper pattern for repeated transformation idioms"
    - "node_bbox: single-fold accumulation over (xmin,xmax,ymin,ymax) in one pass"
    - "sin_sin_surface: cfg(test) helper for repeated ground-truth surface pattern in test module"
    - "helpers::extract_curves reused in soft_dtw (no new helper needed — equivalence verified via source: FdMatrix::rows() = (0..nrows).map(|i| row(i)).collect())"

key-files:
  created: []
  modified:
    - fdars-core/src/fem_smoothing.rs
    - fdars-core/src/explain_generic/friedman.rs
    - fdars-core/src/elastic_regression/scalar_on_shape.rs
    - fdars-core/src/metric/soft_dtw.rs
    - fdars-core/src/classification/fit.rs

key-decisions:
  - "subtract_fitted chosen over compute_residuals because compute_residuals(y, fitted, y_mean) with 3 args already existed at line 636 — naming collision; the 2-arg version gets the distinct name"
  - "node_bbox refactored from 4 separate fold calls to a single-pass fold accumulating (xmin,xmax,ymin,ymax) — pmat still flagged the 4 structurally-identical fold lines even inside the helper"
  - "sin_sin_surface added in cfg(test) to remove 5 repeated obs_xy.iter().map(closure).collect() patterns in fem_smoothing tests — necessary to clear pmat flag after the production bbox fold was collapsed"
  - "helpers::extract_curves reused for soft_dtw without a new extract_rows — FdMatrix::rows() is literally (0..nrows).map(|i| self.row(i)).collect(); equivalence to (0..n).map(|i| data.row(i)) proven by source inspection"

patterns-established:
  - "When a helper name conflicts with an existing function in the same file, choose a semantically-distinct name (subtract_fitted vs compute_residuals)"
  - "pmat entropy counts repeated structural lines even inside a helper: if 4 lines share the same fold structure, collapse them to one multi-accumulator fold"

requirements-completed: [QUAL-04]

coverage:
  - id: D1
    description: "Production entropy idioms (E-04/05/06/07/E-12) factored into private per-file helpers; pmat entropy no longer flags fem_smoothing, friedman, scalar_on_shape, soft_dtw, classification/fit across 3 consecutive runs"
    requirement: QUAL-04
    verification:
      - kind: other
        ref: "cd fdars-core && for i in 1 2 3; do pmat analyze entropy | grep -iE 'fem_smoothing|friedman|scalar_on_shape|soft_dtw|classification/fit' || echo no flags; done"
        status: pass
      - kind: unit
        ref: "cargo test -p fdars-core --features linalg,parallel (2942 lib tests, 215 doctests)"
        status: pass
    human_judgment: false
  - id: D2
    description: "No public API/signature/visibility change — all new helpers are private fn"
    requirement: QUAL-04
    verification:
      - kind: other
        ref: "git diff 5b8fc7db..HEAD -- fdars-core/src/ | grep '^+.*pub ' (empty output)"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-09-14
plan_head_before: 5b8fc7db30cce6cd87e56049d9c6f15b183318ec
status: complete
---

# Phase 108 Plan 03: Production Entropy Extraction Summary

**Private helpers extracted for 5 production files (E-04 through E-12): node_bbox, extract_scalar_row, subtract_fitted, extract_score_row, plus helpers::extract_curves reuse in soft_dtw — pmat entropy flags cleared across 3x consecutive runs**

## Performance

- **Duration:** 12 min
- **Started:** 2026-09-14T10:53:06Z
- **Completed:** 2026-09-14T11:05:42Z
- **Tasks:** 2 (+ 1 additional fem_smoothing fix commit)
- **Files modified:** 5

## Accomplishments

- `fem_smoothing.rs` (E-04): `node_bbox` extracted as a single-fold helper computing (xmin,xmax,ymin,ymax) in one pass; `sin_sin_surface` test helper added to clear remaining pmat flag from repeated test patterns
- `explain_generic/friedman.rs` (E-05): `extract_scalar_row(sc, i, p_scalar)` private fn replaces 4 identical `scalar_covariates.map(|sc| (0..p_scalar).map(|j| sc[(i,j)]).collect())` sites
- `elastic_regression/scalar_on_shape.rs` (E-07): `subtract_fitted(y, fitted)` private fn replaces 6 repeated zip-map residual idioms (name chosen to avoid collision with existing `compute_residuals` at line 636)
- `metric/soft_dtw.rs` (E-06): `helpers::extract_curves` reused at 5 call sites replacing `(0..n).map(|i| data.row(i)).collect()` — equivalence proven by source inspection (`FdMatrix::rows()` = same expression)
- `classification/fit.rs` (E-12): `extract_score_row(scores, i, d)` private fn replaces 3 repeated `(0..d).map(|j| scores[(i,j)]).collect()` calls in `classif_predict_probs`
- Full test suite green: 2942 lib + 215 doctests; clippy and fmt clean
- pmat entropy: 3x consecutive runs — none of the 5 production files addressed here appear

## Task Commits

1. **Task 1: fem_smoothing bbox, friedman scalar-row, scalar_on_shape residuals** — `265a4d01` (refactor)
2. **Task 2: soft_dtw row extraction + classification/fit score-row** — `074de617` (refactor)
3. **Deviation fix: fem_smoothing single-fold + sin_sin_surface test helper** — `dfe9d2e3` (refactor)

## Files Created/Modified

- `fdars-core/src/fem_smoothing.rs` — `node_bbox` (single-fold), `sin_sin_surface` (cfg(test)), 5 test call sites simplified
- `fdars-core/src/explain_generic/friedman.rs` — `extract_scalar_row` private fn, 4 call sites
- `fdars-core/src/elastic_regression/scalar_on_shape.rs` — `subtract_fitted` private fn, 6 call sites
- `fdars-core/src/metric/soft_dtw.rs` — 5 sites replaced with `crate::helpers::extract_curves`
- `fdars-core/src/classification/fit.rs` — `extract_score_row` private fn, 3 call sites

## Decisions Made

- **subtract_fitted vs compute_residuals**: `compute_residuals(y, fitted, y_mean) -> (Vec, f64, f64)` already exists at line 636 of scalar_on_shape.rs; added `subtract_fitted(y, fitted) -> Vec<f64>` to avoid the naming collision and compile error
- **node_bbox refactored to single-fold**: The initial 4-line extraction into `node_bbox` did not clear the pmat flag — pmat still saw 4 structurally-identical fold calls inside the helper. Collapsed to a single fold accumulating all 4 bounds as a tuple, eliminating the pattern
- **sin_sin_surface test helper added**: After collapsing the bbox fold, pmat still flagged `fem_smoothing.rs` because 5 tests repeated `obs_xy.iter().map(|&[x,y]| g(x,y)).collect()` (same ground-truth surface). Extracted to `sin_sin_surface(&obs_xy)` in the `#[cfg(test)]` module to clear the file entirely
- **helpers::extract_curves equivalence proven by source**: `FdMatrix::rows()` is implemented as `(0..self.nrows).map(|i| self.row(i)).collect()` (matrix.rs:271); `helpers::extract_curves` calls `data.rows()` (helpers.rs:46); the flagged idiom `(0..n).map(|i| data.row(i)).collect()` where `n = data.nrows()` is byte-identical. No scratch test needed.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Naming collision between new compute_residuals and existing 3-arg compute_residuals**
- **Found during:** Task 1 (scalar_on_shape.rs)
- **Issue:** A 3-arg `compute_residuals(y, fitted, y_mean) -> (Vec<f64>, f64, f64)` already exists at line 636; adding a 2-arg helper with the same name caused compile error E0428 (duplicate function definition)
- **Fix:** Renamed new helper to `subtract_fitted(y, fitted) -> Vec<f64>`
- **Files modified:** `fdars-core/src/elastic_regression/scalar_on_shape.rs`
- **Verification:** Compiled and tests passed
- **Committed in:** `265a4d01`

**2. [Rule 2 - Missing Critical] pmat still flagged fem_smoothing after initial node_bbox extraction**
- **Found during:** Post-Task-1 pmat entropy run
- **Issue:** Extracting 4 fold lines into `node_bbox` helper did not clear the pmat flag — pmat still counted the 4 structurally-identical fold lines inside the helper as a repeated DataTransformation. Additionally 5 tests repeated `obs_xy.iter().map(g).collect()`.
- **Fix:** (a) Collapsed `node_bbox` to a single-accumulator fold; (b) added `sin_sin_surface` cfg(test) helper to deduplicate test patterns
- **Files modified:** `fdars-core/src/fem_smoothing.rs`
- **Verification:** pmat entropy 3x consecutive runs — fem_smoothing no longer appears
- **Committed in:** `dfe9d2e3`

---

**Total deviations:** 2 auto-fixed (1 naming collision, 1 insufficient extraction)
**Impact on plan:** Both required for pmat entropy clearance. No scope creep. Helper name change (subtract_fitted) is an internal-only difference with no public API impact.

## Issues Encountered

- pmat's DataTransformation detector counts structurally-identical lines within a helper as well as at call sites — simply extracting repeated lines into a named function is not always sufficient; the helper body itself must be non-repetitive (collapsed to a single expression).

## Self-Check Results

PASSED:
- `fdars-core/src/fem_smoothing.rs` — exists with node_bbox + sin_sin_surface
- `fdars-core/src/explain_generic/friedman.rs` — exists with extract_scalar_row
- `fdars-core/src/elastic_regression/scalar_on_shape.rs` — exists with subtract_fitted
- `fdars-core/src/metric/soft_dtw.rs` — 5 sites use crate::helpers::extract_curves
- `fdars-core/src/classification/fit.rs` — exists with extract_score_row
- Commits `265a4d01`, `074de617`, `dfe9d2e3` all exist in git log
- 2942 lib tests pass, 215 doctests pass
- clippy --all-targets clean, fmt clean
- pmat entropy 3x: none of addressed production files flagged
- No new pub signatures in diff

## Next Phase Readiness

- Plan 108-04 (doc-comment simplification) is the next independent plan in phase 108
- Phase 109 (complexity tail + quality-gate green) depends on 106+107+108 all landing

---
*Phase: 108-entropy-extraction-satd-unwrap-cleanup*
*Completed: 2026-09-14*
