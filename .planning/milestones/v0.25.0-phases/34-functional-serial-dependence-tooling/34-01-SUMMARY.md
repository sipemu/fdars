---
phase: 34-functional-serial-dependence-tooling
plan: "01"
subsystem: fts
tags: [functional-time-series, acf, pacf, white-noise-band, nalgebra, monte-carlo]
requirements: [FTS-02]

dependency_graph:
  requires: []
  provides:
    - fdars_core::fts::functional_acf
    - fdars_core::fts::functional_pacf
    - fdars_core::fts::FacfResult
    - fdars_core::fts::StationarityResult
    - fdars_core::fts::LongRunCovResult
    - fdars_core::fts (module barrel)
  affects:
    - fdars-core/src/lib.rs (pub mod fts; + pub use re-exports)

tech_stack:
  added: []
  patterns:
    - nalgebra::SymmetricEigen for weight-scaled C_0 eigendecomposition
    - rand_distr::ChiSquared for MC chi-squared mixture band
    - autocovariance_matrix pub(crate) helper as shared spine for plan 34-03 long-run covariance
    - weight-scaled eigenvalues (W^{1/2} C_0 W^{1/2}) to align MC band units with HS norm

key_files:
  created:
    - fdars-core/src/fts/mod.rs
    - fdars-core/src/fts/acf.rs
  modified:
    - fdars-core/src/lib.rs

decisions:
  - "Use weight-scaled C_0 for eigendecomposition: eigenvalues of W^{1/2}C_0W^{1/2} match the quadrature-weighted HS norm units, preventing band over-inflation seen with raw matrix eigenvalues (AR(1) test would fail otherwise)"
  - "Default n_sim=999 matching project permutation-test convention (fdaACF defaults to 10 000; document in rustdoc)"
  - "autocovariance_matrix is pub(crate) to serve as shared spine for plan 34-03 long-run covariance without forcing a public API commitment"
  - "StationarityResult and LongRunCovResult declared in mod.rs now (struct home stable) even though their producing functions arrive in plan 34-02"

metrics:
  duration: "35m"
  completed: "2026-08-21"
  tasks_completed: 3
  tasks_total: 3

status: complete

actuals:
  tokens: 11000
  tasks: 3
  commits: 1
---

# Phase 34 Plan 01: fts Module — Functional ACF/PACF Tracer Summary

Delivered the full functional-ACF vertical slice end-to-end in a new `fts/` module: lag-h autocovariance operator, Hilbert-Schmidt L2 fACF, weight-scaled MC chi-squared mixture white-noise band via nalgebra eigendecomposition, and scalar Durbin-Levinson fPACF — all wired through `FacfResult` and crate-root re-exported.

## What Was Built

| Artifact | Description |
|----------|-------------|
| `fdars-core/src/fts/mod.rs` | Module barrel (mirrors `inference/mod.rs`): `mod acf`, `pub use acf::{...}`, plus `FacfResult`, `StationarityResult`, `LongRunCovResult` structs |
| `fdars-core/src/fts/acf.rs` | Full implementation: `validate_fts_input`, `mean_curve`, `autocovariance_matrix` (pub(crate)), `hs_norm_sq`, `acf_normalization`, `mc_band_threshold`, `durbin_levinson_pacf`, `functional_acf`, `functional_pacf` + 11 inline tests |
| `fdars-core/src/lib.rs` | Added `pub mod fts;` (alphabetical position after `fof_regression`) and `pub use fts::{functional_acf, functional_pacf, FacfResult}` |

## Algorithm Correctness Notes

- **fACF formula**: `ρ_h = sqrt(Σ_{j1,j2} Ĉ_h[j1,j2]² · w[j1] · w[j2]) / ∫ Ĉ_0(t,t)dt` — exactly matching fdaACF convention; 1/N normalisation throughout.
- **MC band eigenvalues**: eigendecomposition runs on the weight-scaled matrix `W^{1/2} C_0 W^{1/2}` (not raw C_0). This aligns the eigenvalue units with the quadrature-weighted HS norm. The naive raw-matrix eigenvalues produce an over-inflated band that the AR(1) test cannot beat.
- **fPACF divergence from fdaACF**: documented in rustdoc. fdaACF uses an ARH(p-1) residual-cross-covariance approach; this implementation uses the classical scalar Durbin-Levinson recursion over {ρ_h}, which is valid for AR/MA order diagnosis.
- **Band divergence from fdaACF**: documented in rustdoc. fdaACF offers both Imhof exact and MC paths; this implementation provides MC only.

## Tests Delivered and Results

All 11 tests pass (seeded, reproducible):

| Test | Purpose | Result |
|------|---------|--------|
| `facf_lags_start_at_one_and_finite` | Lags start at 1, all ρ_h finite and non-negative | PASS |
| `autocovariance_c0_is_symmetric` | C_0 symmetry within 1e-12 | PASS |
| `error_empty_data` | InvalidDimension on empty matrix | PASS |
| `error_argvals_mismatch` | InvalidDimension on length mismatch | PASS |
| `error_too_few_curves` | InvalidDimension when max_lag >= n | PASS |
| `deterministic_seed` | Bit-identical output on same seed | PASS |
| `facf_whitenoise_inside_band` | 80 i.i.d. white-noise curves: all lags inside 95% band | PASS |
| `facf_ar1_exceeds_band` | AR(1) coefficient 0.8, n=120: lag-1 fACF exceeds band | PASS |
| `dl_pacf_single_rho` | Durbin-Levinson: pacf[0] == rho[0] for single input | PASS |
| `fpacf_ar1_cutoff` | AR(1) fPACF: large lag-1, small lags 2+ | PASS |
| `fpacf_returns_populated_pacf` | functional_pacf populates pacf Vec | PASS |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Weight-scaled eigenvalues for MC band**

- **Found during:** Task 2 (MC white-noise band) — `facf_ar1_exceeds_band` failed with lag-1 fACF 0.59 < band 2.72
- **Issue:** The plan and RESEARCH.md specify eigendecomposing the raw m×m C_0 matrix. This produces eigenvalues of order O(N·variance/m²) unscaled relative to the quadrature-weighted HS norm, inflating the band by a factor unrelated to N. The band converged to ~2.72 while lag-1 fACF was ~0.59, making the AR(1) power test impossible at n=120.
- **Fix:** Eigendecompose `W^{1/2} C_0 W^{1/2}` (weight-scaled covariance matrix) instead of raw C_0. The eigenvalues then match the units of the HS norm `Σ c_h[j1,j2]² · w[j1] · w[j2]`, giving a well-calibrated band (~0.25 for i.i.d. data, correctly inside which white-noise lags fall and correctly exceeded by AR(1) lag-1).
- **Files modified:** `fdars-core/src/fts/acf.rs` (mc_band_threshold caller in `functional_acf`)
- **Commit:** 75a766df (incorporated before final commit)

## Threat Mitigations Applied

| ID | Threat | Mitigation |
|----|--------|-----------|
| T-34-01 | Index `i + h` out-of-bounds | `max_lag + 1 > n` returns `InvalidParameter`; loop bound `i in 0..(n-h)` never exceeds n |
| T-34-02 | Near-zero normalization → NaN | `acf_normalization` checks `< NUMERICAL_EPS` → `ComputationFailed` before division |
| T-34-03 | Large phi matrix (Durbin-Levinson) | `phi` heap-allocated `Vec<Vec<f64>>` |

## Known Stubs

None — all FacfResult fields (acf, pacf, upper_band, lags) are fully populated.

`StationarityResult` and `LongRunCovResult` struct definitions are declared in `fts/mod.rs` but their producing functions (`stationarity_test`, `long_run_covariance`, `functional_difference`) are delivered by plan 34-02. This is intentional (struct home stable before plan 34-02).

## Self-Check

- `fdars-core/src/fts/mod.rs` exists: FOUND
- `fdars-core/src/fts/acf.rs` exists: FOUND
- `lib.rs` has `pub mod fts;` and `pub use fts::{functional_acf, functional_pacf, FacfResult}`: FOUND
- Commit 75a766df exists: FOUND
- All 11 tests pass: CONFIRMED
- `git diff --exit-code fdars-core/Cargo.toml`: clean (no new deps)

## Self-Check: PASSED
