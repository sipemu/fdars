---
phase: 22-constant-basis-aic-smoothing
plan: 01
subsystem: api
tags: [basis, smoothing, aic, gcv, bandwidth, functional-data, r-parity]

# Dependency graph
requires: []
provides:
  - "constant_basis(t) intercept-column basis constructor (column-major m×1 ones)"
  - "CvCriterion::Aic variant + aic_smoother for kernel-bandwidth selection (AIC = n·ln(RSS/n) + 2·tr(S))"
  - "smooth_basis_aic AIC-based λ selector for basis-penalized smoothing"
affects: [23-depth-boxplot, r-bindings, smoothing, basis]

# Actuals (#2632)
actuals:
  tokens: 4400
  tasks: 3
  commits: 3

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "AIC criterion reuses the GCV hat-matrix trace as df (df = tr(S) / tr(H)) across both the kernel-bandwidth and basis-penalty smoothing paths"
    - "CvCriterion made #[non_exhaustive] to allow additive criteria without a breaking change"

key-files:
  created:
    - fdars-core/src/basis/constant.rs
  modified:
    - fdars-core/src/basis/mod.rs
    - fdars-core/src/smoothing.rs
    - fdars-core/src/smooth_basis.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "constant_basis returns Vec<f64> directly (infallible, mirrors bspline_basis/fourier_basis) — empty grid yields empty matrix, no Result"
  - "CvCriterion marked #[non_exhaustive] (executor discretion per D — additive minor-release change for 0.19→0.20)"
  - "aic_smoother uses the standard smoother AIC = n·ln(RSS/n) + 2·tr(S), sharing gcv_smoother's Nadaraya-Watson matrix and hat-matrix trace"
  - "Basis-path AIC implemented as a new smooth_basis_aic function (not a criterion arg), reusing the aic field smooth_basis already populates — no inline AIC recomputation"

patterns-established:
  - "AIC-vs-GCV divergence test uses a custom small-bandwidth range where GCV's (1−tr/n)⁻² penalty blows up but AIC's additive 2·tr penalty under-smooths — the two criteria genuinely diverge"

requirements-completed: [T-01]

coverage:
  - id: D1
    description: "constant_basis(t) returns a column-major m×1 all-ones matrix; intercept-only fit reproduces the response mean; empty input yields empty Vec"
    requirement: "T-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/basis/constant.rs#ones_column_shape"
        status: pass
      - kind: unit
        ref: "fdars-core/src/basis/constant.rs#empty_input_no_panic"
        status: pass
      - kind: unit
        ref: "fdars-core/src/basis/constant.rs#intercept_only_fit_reproduces_mean"
        status: pass
    human_judgment: false
  - id: D2
    description: "CvCriterion::Aic + aic_smoother: AIC = n·ln(RSS/n)+2·tr(S), reusing the GCV hat-matrix trace; optim_bandwidth Aic argmin matches a brute-force grid and diverges from GCV; GCV/CV paths unchanged"
    requirement: "T-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/smoothing.rs#test_aic_smoother_matches_hand_computed"
        status: pass
      - kind: unit
        ref: "fdars-core/src/smoothing.rs#test_aic_smoother_invalid_inputs"
        status: pass
      - kind: unit
        ref: "fdars-core/src/smoothing.rs#test_optim_bandwidth_aic_matches_brute_force_grid"
        status: pass
      - kind: unit
        ref: "fdars-core/src/smoothing.rs#test_optim_bandwidth_aic_diverges_from_gcv"
        status: pass
      - kind: unit
        ref: "fdars-core/src/smoothing.rs#test_gcv_cv_unchanged_by_aic_addition"
        status: pass
    human_judgment: false
  - id: D3
    description: "smooth_basis_aic minimizes AIC over the same log-λ grid as smooth_basis_gcv, matches a brute-force AIC grid, and prefers a smoother (lower-edf) fit on noisy data; smooth_basis_gcv unchanged"
    requirement: "T-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/smooth_basis.rs#test_smooth_basis_aic_matches_brute_force_grid"
        status: pass
      - kind: unit
        ref: "fdars-core/src/smooth_basis.rs#test_smooth_basis_aic_prefers_smoother_fit_than_smallest_lambda"
        status: pass
    human_judgment: false

# Metrics
duration: 18min
completed: 2026-08-16
status: complete
---

# Phase 22 Plan 01: Constant Basis & AIC Smoothing Selection Summary

**Additive R-parity table-stakes: `constant_basis` intercept column, `CvCriterion::Aic` + `aic_smoother` kernel-bandwidth selection, and `smooth_basis_aic` λ selector — all reusing existing GCV hat-matrix-trace infrastructure with GCV/CV paths byte-for-byte unchanged.**

## Performance

- **Duration:** ~18 min
- **Completed:** 2026-08-16
- **Tasks:** 3 (tracer-first)
- **Files modified:** 5 (1 created, 4 modified)

## Accomplishments
- `constant_basis(t) -> Vec<f64>` in new `basis/constant.rs` — column-major m×1 all-ones intercept column, mirroring the `bspline_basis`/`fourier_basis` convention; re-exported at `basis/mod.rs` and the crate root.
- `CvCriterion::Aic` variant (enum now `#[non_exhaustive]`) + `aic_smoother(x, y, bandwidth, kernel)` computing `AIC = n·ln(RSS/n) + 2·tr(S)`, reusing `gcv_smoother`'s Nadaraya-Watson smoother matrix and hat-matrix trace as df; wired into `optim_bandwidth`'s dispatch.
- `smooth_basis_aic` λ selector alongside the unchanged `smooth_basis_gcv` — same log-λ grid, minimizing the `SmoothBasisResult.aic` field `smooth_basis` already computes.
- 10 new tests added; full suite green (2049 lib tests, all integration + 137 doctests passing); clippy `--all-targets` clean.

## Task Commits

Each task was committed atomically (all passed the full pre-commit gate: fmt + clippy + test + doc):

1. **Task 1: constant_basis constructor (tracer)** — `5fc5de7c` (feat)
2. **Task 2: CvCriterion::Aic + aic_smoother** — `f8f15cce` (feat)
3. **Task 3: smooth_basis_aic λ selector** — `691423dc` (feat)

## Files Created/Modified
- `fdars-core/src/basis/constant.rs` (created) — `constant_basis` + 3 inline tests
- `fdars-core/src/basis/mod.rs` — `pub mod constant;` + `pub use constant::constant_basis;`
- `fdars-core/src/smoothing.rs` — `CvCriterion::Aic` variant, `aic_smoother`, `optim_bandwidth` dispatch arm, 5 inline tests
- `fdars-core/src/smooth_basis.rs` — `smooth_basis_aic` function + 2 inline tests
- `fdars-core/src/lib.rs` — crate-root re-exports for `constant_basis`, `aic_smoother`, `smooth_basis_aic`

## Test Coverage (10 new tests, all pass)
- **constant_basis:** ones-column shape, empty input (no panic), intercept-mean identity (β = mean(y))
- **aic_smoother:** hand-computed AIC fixture match, invalid-input guards (n<2 / length mismatch / bandwidth≤0 → INFINITY)
- **optim_bandwidth AIC:** brute-force grid argmin equality (exact first-minimum tie-break), AIC-vs-GCV divergence (AIC picks smaller h)
- **GCV/CV regression guard:** `gcv_smoother` still equals the classic `(RSS/n)/(1−tr/n)²` formula
- **smooth_basis_aic:** brute-force AIC grid argmin match, smoother-fit (lower edf) than smallest-λ candidate on noisy data

## Decisions Made
- `constant_basis` returns `Vec<f64>` directly (infallible) — matches its `basis/` neighbors; empty grid → empty Vec, no `Result` needed.
- `CvCriterion` marked `#[non_exhaustive]` (per D — additive minor-release change is acceptable for 0.19→0.20). No in-crate exhaustive match over `CvCriterion` exists outside the `optim_bandwidth` dispatch, so the addition is safe.
- Standard smoother AIC formula `n·ln(RSS/n) + 2·tr(S)` chosen (per D — trace-of-smoother df form, validated against a brute-force grid).
- Basis-path AIC implemented as a new `smooth_basis_aic` function (not a criterion arg on `smooth_basis_gcv`), reusing the pre-computed `aic` field rather than recomputing AIC inline.

## Deviations from Plan
None - plan executed exactly as written. All three tasks implemented as specified; the AIC-vs-GCV divergence test required a custom small-bandwidth range (`(0.01, 0.5)`) to expose genuine divergence, which is within the plan's stated latitude ("a case where AIC and GCV diverge").

## Signatures Confirmed Unchanged
- `cv_smoother`, `gcv_smoother`, `optim_bandwidth` signatures unchanged; `CvCriterion::Cv`/`Gcv` dispatch untouched (regression guard test passing).
- `smooth_basis`, `smooth_basis_gcv` signatures and behavior unchanged.
- All existing basis constructors untouched.
- Only additive changes: 3 new public items + 1 new enum variant.

## Issues Encountered
- **`cargo fmt --check` gate on Task 2 commit:** the pre-commit hook rejected a single-line `assert!` that exceeded the width; resolved by running `cargo fmt -p fdars-core` and re-committing. No code logic changed.
- **AIC/GCV divergence with default range:** on the default bandwidth range both criteria selected the same grid optimum; probed via a scratch harness and confirmed the theoretically-expected divergence appears when the range reaches small bandwidths (GCV's `(1−tr/n)⁻²` blows up while AIC under-smooths). Test now uses `(0.01, 0.5)`.

## Next Phase Readiness
- T-01 complete. Phase 23 (T-02: depth-fence functional boxplot) is independent (disjoint modules) and unblocked.
- New public API surface (`constant_basis`, `aic_smoother`, `CvCriterion::Aic`, `smooth_basis_aic`) is available at the crate root and via `prelude` re-exports for downstream R/WASM bindings.

## Self-Check: PASSED

- All created/modified files present on disk.
- All 3 task commits (`5fc5de7c`, `f8f15cce`, `691423dc`) exist in git history.
- All 3 new public items (`constant_basis`, `aic_smoother`, `smooth_basis_aic`) re-exported at the crate root.

---
*Phase: 22-constant-basis-aic-smoothing*
*Completed: 2026-08-16*
