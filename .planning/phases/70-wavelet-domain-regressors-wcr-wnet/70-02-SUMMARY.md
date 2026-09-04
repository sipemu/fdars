---
phase: 70-wavelet-domain-regressors-wcr-wnet
plan: 02
subsystem: regression
tags: [wavelet, elastic-net, coordinate-descent, cross-validation, scalar-on-function, dwt]

requires:
  - phase: 70-01 (wcr)
    provides: curves_to_coeff_design, coeff_weights_to_beta_t, CoeffLayout shared seams in wavelet/regression.rs
provides:
  - WnetConfig / WnetResult (wavelet-domain elastic-net config + result)
  - wnet() entry point (WAV-04)
  - elastic_net_cd() per-coefficient L1+L2 coordinate-descent adapter (pub(crate))
  - wnet_cv_lambda() deterministic K-fold CV-lambda helper (pub(crate))
affects: [71 (predict + crate-root/prelude re-exports for wcr/wnet)]

actuals:
  tokens: 15000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Per-coefficient elastic-net coordinate descent (L1 soft-threshold + L2 ridge) modeled on additive.rs group-lasso soft-threshold pattern, but scalar-per-coefficient"
    - "Deterministic CV-lambda: fixed create_folds seed + geometric grid (descending) + min-CV-MSE with tie->larger/sparser lambda"

key-files:
  created: []
  modified:
    - fdars-core/src/wavelet/regression.rs

key-decisions:
  - "elastic_net_cd centers columns (not unit-variance standardized) to keep coefficient-space geometry faithful to the DWT; penalty denominator is ||X_c,j||^2/n + lambda(1-alpha)"
  - "lambda grid built descending (lambda_max first) so a strictly-less-minus-epsilon MSE comparison naturally keeps the larger/sparser lambda on ties"
  - "Single combined commit for all 3 tasks (interleaved edits in one file); tasks are logically distinct but code is cohesive"

patterns-established:
  - "Running-fit vector (Sigma_j beta_j X_c,j) maintained across CD sweeps for O(nP)/sweep partial-residual updates instead of O(nP^2)"

requirements-completed: [WAV-04]

coverage:
  - id: D1
    description: "wnet fits per-coefficient elastic-net (L1+L2) on wavelet coefficients and recovers a sparse coefficient pattern concentrating on the true support (SC2)"
    requirement: WAV-04
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wnet_elastic_net_cd_recovers_sparse_support"
        status: pass
    human_judgment: false
  - id: D2
    description: "wnet CV-selected lambda is deterministic across two independent runs (fixed fold partition + seed) (SC3)"
    requirement: WAV-04
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wnet_cv_lambda_is_deterministic_across_runs"
        status: pass
    human_judgment: false
  - id: D3
    description: "at the CV lambda, wnet yields a non-degenerate fit whose beta(t) tracks the injected signal on SNR data (SC3)"
    requirement: WAV-04
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wnet_recovers_beta_t_on_snr_data"
        status: pass
    human_judgment: false
  - id: D4
    description: "wnet validates dim/param mismatch -> descriptive FdarError, never panics; beta(t)/fitted/selected outputs finite/NaN-free (SC4)"
    requirement: WAV-04
    verification:
      - kind: unit
        ref: "fdars-core/src/wavelet/regression.rs#wnet_rejects_* / wnet_finite_outputs_on_larger_snr_design"
        status: pass
    human_judgment: false

duration: 18min
completed: 2026-09-04
status: complete
---

# Phase 70 Plan 02: wnet Summary

**Wavelet-domain elastic-net scalar-on-function regressor (WAV-04): per-coefficient L1+L2 coordinate descent with deterministic cross-validated lambda, reusing Plan 01's curves->coeff and inverse-DWT beta(t) seams.**

## Performance

- **Duration:** ~18 min
- **Tasks:** 3 (implemented as one cohesive addition)
- **Files modified:** 1 (`fdars-core/src/wavelet/regression.rs`, +921 lines)

## Accomplishments
- `elastic_net_cd` per-coefficient coordinate-descent adapter: partial-residual sweep via a maintained running-fit vector (O(nP)/sweep), per-coordinate L1 soft-threshold (`sign(z)·max(|z|-lambda·alpha,0)`) with L2 ridge denominator (`||X_c,j||^2/n + lambda(1-alpha)`), intercept recovered on un-centered column means. Modeled on additive.rs's group-lasso soft-threshold pattern (NOT `variable_selection`).
- `wnet_cv_lambda`: deterministic K-fold (`crate::cv::create_folds` fixed seed) + auto geometric lambda grid (lambda_max = max_j |X_c,j·y_c|/(n·alpha) down to lambda_max·1e-3) or explicit grid, min-CV-MSE with tie->larger/sparser lambda.
- `wnet` entry point: validate -> `curves_to_coeff_design` (shared seam) -> `wnet_cv_lambda` -> refit `elastic_net_cd` on full data -> `coeff_weights_to_beta_t` (shared seam) -> assemble `WnetResult` carrying intercept, beta_t, fitted/residuals, sparse coeff_weights, selected nonzero indices, CV lambda, alpha, and DWT config for Phase 71 predict.
- `WnetConfig`/`WnetResult` structs: `#[non_exhaustive]`, `Debug/Clone/PartialEq`, serde-gated, `#[must_use]` producer; builder-style `Default` (db4/Periodic/auto, alpha=0.5, n_lambda=50, n_folds=5, seed=0).

## Task Commits

Tasks 1-3 landed as one cohesive commit (interleaved edits to a single file):

1. **Tasks 1-3: wnet regressor (CD adapter + CV-lambda + validation/tests)** - `cebb6df7` (feat)

## Files Created/Modified
- `fdars-core/src/wavelet/regression.rs` - Added the entire wnet surface (config, result, `elastic_net_cd`, `build_lambda_grid`, `wnet_cv_lambda`, `wnet`, `soft_threshold`, `compute_fitted_affine`) + 15 inline wnet tests.

## Gate Results
- **Tests:** `cargo test -p fdars-core --features linalg,parallel --lib wavelet::regression` -> **27 passed, 0 failed** (13 pre-existing wcr + 14 new wnet tests: sparse-support recovery, CV-lambda determinism across two runs + direct-helper determinism, SNR beta(t) recovery, default-config, fixed-lambda end-to-end finite, explicit-grid, larger-SNR finite outputs, and 7 validation-gate tests).
- **Clippy:** `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` -> **clean**.
- **fmt:** `cargo fmt --check` -> **no diff**.

## Success-criteria verification
- **SC2 sparse recovery:** `wnet_elastic_net_cd_recovers_sparse_support` asserts the true-support coefficients are all selected and the selected set is < P/2 (sparse). GREEN.
- **SC3 determinism:** `wnet_cv_lambda_is_deterministic_across_runs` asserts identical CV lambda across two `wnet` fits and two direct `wnet_cv_lambda` calls. GREEN.
- **SC3 beta(t) recovery:** `wnet_recovers_beta_t_on_snr_data` (300x32 spanning full-rank design, ~20:1 SNR) asserts non-degenerate fit and rel-L2 beta(t) error < 0.35. GREEN.
- **SC4 validation/finite:** 7 rejection tests (too-few-rows, zero-cols, y-mismatch, alpha out of range, n_folds<2, empty lambda_grid, unsupported family, level out of range) + finite-output test. GREEN, no panics.

## Decisions Made
- Combined the three tasks into a single commit because the edits interleave within one file and are mutually dependent (CV path and fixed-lambda path share `elastic_net_cd`). Each task's acceptance criteria are individually verified by dedicated tests.
- Columns are centered (not standardized to unit variance) in `elastic_net_cd` to keep the coefficient-space penalty geometry faithful to the DWT.
- Descending lambda grid + strictly-less-minus-epsilon MSE comparison implements the "tie -> larger/sparser lambda" rule cleanly.

## Deviations from Plan
- **Commit granularity:** plan implies one commit per task; delivered as a single `feat(70-02)` commit covering all three tasks (see rationale above). No scope change — all task acceptance criteria met and tested.
- No other deviations. Guardrails honored: additive-only edits to `wavelet/regression.rs`, shared seams reused (not duplicated), NEW per-coefficient CD adapter (not `variable_selection`), no crate-root/prelude re-exports, no Cargo.toml change, additive.rs untouched.

## Issues Encountered
- `cargo build --features serde` fails with a **pre-existing** breakage (`ClassifFit: serde::Serialize/Deserialize` not satisfied in `shapelet/classifier.rs`, Phase 60) — unrelated to this work; the new `WnetConfig`/`WnetResult` serde derives produce no errors. Documented in project memory (serde-feature-build-broken-shapelet-classiffit).
- No disk events. `/home` at 94% but no space failures during build/test/clippy.

## Next Phase Readiness
- `wnet` result carries all DWT config (`family`/`mode`/`level`) + `coeff_weights` needed for Phase 71 predict + crate-root/prelude re-exports.
- Shared seams from Plan 01 remain the single curves->coeff and inverse-DWT integration points for both wcr and wnet.

---
*Phase: 70-wavelet-domain-regressors-wcr-wnet*
*Completed: 2026-09-04*
