---
phase: 71-prediction-diagnostics-integration
plan: 01
subsystem: regression
tags: [wavelet, wcr, wnet, prediction, prelude, dwt, scalar-on-function]

requires:
  - phase: 69-dwt
    provides: DWT primitive (decompose/reconstruct/decompose_matrix/max_level, WaveletFamily/BoundaryMode/WaveletCoeffs)
  - phase: 70-wavelet-regressors
    provides: wcr/wnet regressors, WcrResult/WnetResult with intercept/coeff_weights/family/mode/level, curves_to_coeff_design + compute_fitted_affine seams
provides:
  - WcrResult::predict + WnetResult::predict (out-of-sample, affine coefficient-space dot)
  - beta_t()/coefficient_function()/fitted_values() accessors on both result structs
  - crate-root + prelude re-exports of the full 14-symbol wavelet surface
  - end-to-end running module doctest for the wavelet regressors
affects: [wavelet, prelude, r-bindings, wasm-bindings]

actuals:
  tokens: 9500
  tasks: 4
  commits: 4

tech-stack:
  added: []
  patterns:
    - "Affine coefficient-space intercept convention: fitted == intercept + design·coeff_weights (wcr now matches wnet), enabling a stored-config predict that reproduces fitted_values exactly"

key-files:
  created: []
  modified:
    - fdars-core/src/wavelet/regression.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

key-decisions:
  - "Re-expressed wcr's stored intercept in the affine coefficient-space convention (folding the centered-recovery offset Σ col_mean·w) so predict reproduces fitted_values to <=1e-8 WITHOUT new fields or loosened tolerance; slope beta_t and residuals unchanged."
  - "predict reuses curves_to_coeff_design(new, self.family, self.mode, Some(self.level)) with the STORED config; grid-length gate against beta_t.len() returns FdarError::InvalidDimension, never panics."
  - "Re-exports are serde-independent (no entanglement with the pre-existing --features serde shapelet break); default/linalg,parallel build is the gate."

patterns-established:
  - "Wavelet-domain result structs expose predict + coefficient/fitted accessors mirroring FregreLmResult/WaveletCoeffs getter conventions (#[must_use], &[f64])."

requirements-completed: [WAV-05, WAV-06]

coverage:
  - id: D1
    description: "WcrResult::predict reproduces stored training fitted_values within 1e-8 abs; fresh-curve predictions finite; grid-length mismatch returns FdarError"
    requirement: "WAV-05"
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wcr_predict_reproduces_training_fitted"
        status: pass
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wcr_predict_on_new_curves_is_finite_and_rejects_grid_mismatch"
        status: pass
    human_judgment: false
  - id: D2
    description: "WnetResult::predict reproduces stored training fitted_values within 1e-8 abs; fresh-curve finite; grid mismatch -> FdarError"
    requirement: "WAV-05"
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wnet_predict_reproduces_training_fitted"
        status: pass
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wnet_predict_on_new_curves_is_finite_and_rejects_grid_mismatch"
        status: pass
    human_judgment: false
  - id: D3
    description: "beta_t()/coefficient_function()/fitted_values() accessors on both structs return the stored slices with lengths m and n"
    requirement: "WAV-05"
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#accessors_return_stored_slices"
        status: pass
    human_judgment: false
  - id: D4
    description: "Full wavelet surface (14 symbols) reachable via both crate-root paths and fdars_core::prelude::*"
    requirement: "WAV-06"
    verification:
      - kind: integration
        ref: "cargo build -p fdars-core --features linalg,parallel (crate-root pub use) + doctest uses prelude"
        status: pass
    human_judgment: false
  - id: D5
    description: "Running end-to-end module doctest (fit -> predict -> beta_t/fitted via prelude) passes under cargo test --doc"
    requirement: "WAV-06"
    verification:
      - kind: e2e
        ref: "fdars-core/src/wavelet/regression.rs - wavelet::regression (line 27) doctest"
        status: pass
    human_judgment: false
  - id: D6
    description: "Whole-crate cargo test + clippy --all-targets + fmt --check green; 28 examples + R/WASM bindings still build"
    requirement: "WAV-06"
    verification:
      - kind: integration
        ref: "cargo test -p fdars-core --features linalg,parallel (2794 lib + 199 doc + integration, 0 fail); clippy --all-targets -D warnings clean; fmt --check clean"
        status: pass
    human_judgment: false

duration: 21min
completed: 2026-09-04
status: complete
---

# Phase 71 Plan 01: Prediction, Diagnostics & Integration Summary

**Out-of-sample `predict` + coefficient/fitted accessors on `WcrResult`/`WnetResult`, the full wavelet surface re-exported at crate root and prelude, and a running end-to-end module doctest — the v0.37.0 WAV milestone is now code-complete.**

## Performance

- **Duration:** 21 min
- **Started:** 2026-09-04
- **Completed:** 2026-09-04
- **Tasks:** 4
- **Files modified:** 3

## Accomplishments
- `WcrResult::predict` and `WnetResult::predict` (`#[must_use]`, `Result<Vec<f64>, FdarError>`) re-transform new curves with the STORED DWT config and reproduce training `fitted_values` within 1e-8 absolute; grid-length mismatch returns `FdarError::InvalidDimension` (never panics).
- `beta_t()` / `coefficient_function()` / `fitted_values()` accessors on both result structs.
- The 14-symbol wavelet surface (DWT primitives + `wcr`/`wnet` + `Wcr*`/`Wnet*` config/result types) re-exported at the crate root (`lib.rs`) AND `prelude.rs`, serde-independent.
- A running `//!` module doctest fits `wcr`, calls `predict`, and reads `beta_t()`/`fitted_values()` through `fdars_core::prelude::*`, executing under `cargo test --doc`.

## Task Commits

1. **Task 1: WcrResult::predict + accessors (affine-consistent intercept)** - `3f49d578` (feat)
2. **Task 2: WnetResult::predict + accessors on both structs** - `aedbb338` (feat)
3. **Task 3: crate-root + prelude re-exports of full wavelet surface** - `abb6b537` (feat)
4. **Task 4: end-to-end wavelet module doctest via prelude** - `295a3d80` (feat)

## Files Created/Modified
- `fdars-core/src/wavelet/regression.rs` - `impl WcrResult`/`impl WnetResult` (predict + 3 accessors each); affine-intercept re-expression in `wcr`; module doctest; 5 new tests.
- `fdars-core/src/lib.rs` - crate-root `pub use wavelet::{...}` + `pub use wavelet::regression::{...}`.
- `fdars-core/src/prelude.rs` - mirrored `// Wavelet-domain regression (v0.37.0)` group.

## Decisions Made
- **Affine intercept re-expression in `wcr` (Deviation, Rule 1).** The stored `coeff_weights` come from a column-*centered* plain-dot recovery, so `intercept + Σ design·coeff_weights` overshot `fitted_values` by the constant `Σ col_mean·coeff_weights`. Rather than loosen the 1e-8 self-consistency tolerance (forbidden) or add a field (forbidden), `wcr` now folds that centering offset into the stored `intercept` (the affine coefficient-space convention `wnet` already uses). This makes `predict`'s simple `compute_fitted_affine` reproduce `fitted_values` exactly. The slope `beta_t` (inverse DWT of `coeff_weights`) and `residuals` (`y − fitted_values`) are numerically unchanged; only the intercept's numeric value is re-expressed. All pre-existing recovery tests (`wcr_pcr_recovers_known_beta_t`, `wcr_pls_...`) still pass unchanged.
- `predict` reuses `curves_to_coeff_design(new, self.family.clone(), self.mode, Some(self.level))` — the STORED config — not a reconstruction from `beta_t`.
- Re-exports kept serde-clean; the default/`linalg,parallel` build is the gate (the pre-existing `--features serde` shapelet break was not entangled).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] wcr predict self-consistency required an affine-intercept fix**
- **Found during:** Task 1 (`wcr_predict_reproduces_training_fitted` initially failed with |Δ| ≈ 0.005)
- **Issue:** `recover_coeff_weights` centers the design, so `fitted_i = intercept + Σ(design_ij − col_mean_j)·w_j`, whereas the affine predict formula `intercept + Σ design_ij·w_j` differs by the constant `Σ col_mean_j·w_j`. `col_mean` is training-specific and (correctly) not stored.
- **Fix:** In `wcr`, after recovering `coeff_weights`, re-express the stored `intercept` as `intercept − Σ col_mean_j·w_j` (affine convention). This keeps the predict formula in the plan (`compute_fitted_affine` with stored `intercept`), touches no fields, and preserves the 1e-8 tolerance.
- **Files modified:** `fdars-core/src/wavelet/regression.rs`
- **Verification:** `wcr_predict_reproduces_training_fitted` passes (≤1e-8); all 33 pre-existing + new wavelet::regression tests green; whole-crate suite green.
- **Committed in:** `3f49d578` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 bug).
**Impact on plan:** Necessary for the SC1 self-consistency contract; no scope creep, no field/signature change, no tolerance loosening. `wnet` needed no analogous fix — it already stores the affine intercept.

## Issues Encountered
None beyond the deviation above. Disk sat at 94% throughout with no "No space left"/link failures; no recovery needed.

## Gate Results (SC4)
- **Whole-crate `cargo test -p fdars-core --features linalg,parallel`:** 2794 lib + 199 doc (4 pre-existing ignored) + all integration suites — **0 failures**.
- **`cargo clippy --all-targets --features linalg,parallel -- -D warnings`:** clean.
- **`cargo fmt --check`:** clean (exit 0).
- **Module doctest under `--doc`:** `wavelet::regression (line 27) ... ok` (1 passed).

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- v0.37.0 WAV milestone is code-complete (Phases 69–71 done). Ready for `/gsd-complete-milestone` / operator ship (version bump is the operator step; no `Cargo.toml` change here).

---
*Phase: 71-prediction-diagnostics-integration*
*Completed: 2026-09-04*

## Self-Check: PASSED
