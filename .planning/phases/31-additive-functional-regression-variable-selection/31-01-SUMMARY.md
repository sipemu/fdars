---
phase: 31-additive-functional-regression-variable-selection
plan: "01"
subsystem: scalar_on_function
tags: [fam, gkam, gsam, additive-regression, nonparametric, fpc-scores]
dependency_graph:
  requires:
    - regression::fdata_to_pc_1d
    - smoothing::{nadaraya_watson, optim_bandwidth, CvCriterion}
    - scalar_on_function::nonparametric::{compute_pairwise_distances, gaussian_kernel, select_bandwidth_loo}
  provides:
    - scalar_on_function::additive::{fam, fregre_gkam, fregre_gsam}
    - scalar_on_function::additive::{FamConfig, FamResult, GkamConfig, GkamResult, GsamConfig, GsamResult}
  affects:
    - fdars-core public API (additive/non-breaking)
    - lib.rs scalar_on_function re-export block
    - scalar_on_function/mod.rs
tech_stack:
  added: []
  patterns:
    - One-pass FPC additive smooth (Müller & Yao 2008 uncorrelatedness)
    - Iterative backfitting over Nadaraya-Watson on L2 curve distances (GKAM)
    - Shared fpc_additive_smooth helper reused by fam and fregre_gsam
    - resolve_ncomp_additive with GCV-based auto-selection (cap at min(n,m,10))
key_files:
  created:
    - fdars-core/src/scalar_on_function/additive.rs
  modified:
    - fdars-core/src/scalar_on_function/mod.rs
    - fdars-core/src/lib.rs
decisions:
  - FAM single forward pass sufficient (FPC scores uncorrelated per Müller & Yao 2008) — no backfitting loop
  - GKAM applies NW weights inline (O(n) per point) instead of materializing n×n hat matrix
  - GSAM reuses fpc_additive_smooth helper (numerically identical to FAM for Gaussian identity link)
  - Shared private helper fpc_additive_smooth() to avoid code duplication between fam() and fregre_gsam()
  - resolve_ncomp_additive() caps auto-selection at min(n,m,10) for speed; uses per-component GCV
  - Gaussian identity link only for GKAM/GSAM (IRLS for non-Gaussian links documented as known gap)
metrics:
  duration: "~35 minutes"
  completed: "2026-08-20"
  tasks: 3
  commits: 2
estimate:
  tokens: 78000
  tasks: 3
actuals:
  tokens: 11971
  tasks: 3
  commits: 2
status: complete
---

# Phase 31 Plan 01: FAM / GKAM / GSAM Additive Functional Regression Summary

New file `fdars-core/src/scalar_on_function/additive.rs` delivers three
nonparametric additive scalar-on-function regression estimators — FAM, GKAM,
and GSAM — wired end-to-end via module re-exports in `mod.rs` and crate-root
re-exports in `lib.rs`. All 10 new tests pass; clippy `--all-targets` is green.

## What Was Built

### `fam` — Functional Additive Model (Müller & Yao 2008)

`E[Y|X] = μ_Y + Σ_k f_k(ξ_k)` where ξ_k are FPC scores. Implementation uses
a single forward pass over k = 0..K, building partial residuals and fitting
each f_k via `nadaraya_watson` with per-component GCV bandwidth from
`optim_bandwidth`. No backfitting loop — FPC uncorrelatedness makes one pass
equivalent to infinite-iteration convergence.

Config: `FamConfig { ncomp, bandwidth, kernel, n_grid_bandwidth }`
Result: `FamResult { fitted_values, residuals, component_fits, intercept, bandwidths, ncomp, r_squared, fpca }`

### `fregre_gkam` — Generalized Kernel Additive Model

Iterative backfitting over NW smoothers on functional L2 distances. Accepts
multiple functional predictors (`&[&FdMatrix]`). For each predictor, computes
pairwise L2 distances once via `compute_pairwise_distances`, selects bandwidth
by LOO-CV, then iterates: adjusted response → inline NW kernel loop → update
component fit → check max-delta convergence.

Implements only Gaussian identity link; logit/log require IRLS (documented gap).
Does NOT materialise the n×n hat matrix inside the backfitting loop (O(n) per
point, O(n²) per covariate per iteration).

Config: `GkamConfig { bandwidth, kernel, max_iter, epsilon }`
Result: `GkamResult { fitted_values, residuals, component_fits, intercept, bandwidths, iterations, converged, r_squared }`

### `fregre_gsam` — Generalized Spectral Additive Model

FPC-score basis additive smooth — same computation path as FAM, reusing
`fpc_additive_smooth`. Under Gaussian identity link, numerically equivalent to
FAM within 1e-6 (verified by `gsam_matches_fam_identity`). Non-Gaussian links
documented as known gap.

Config: `GsamConfig { ncomp, bandwidth, kernel, n_grid_bandwidth }`
Result: `GsamResult { fitted_values, residuals, component_fits, intercept, bandwidths, ncomp, r_squared, fpca }`

## Test Coverage (10 new tests)

| Test | Purpose | Result |
|------|---------|--------|
| `fam_synthetic_recovery` | y=sin(ξ₁)+ξ₂²+noise → R²>0.75 | PASS |
| `fam_decomposition_identity` | fitted+residuals==y within 1e-9 | PASS |
| `fam_output_shapes` | component_fits.len()==ncomp, bandwidths.len()==ncomp | PASS |
| `fam_invalid_dimension` | empty data/wrong y.len/wrong argvals.len → FdarError | PASS |
| `gkam_r2_synthetic` | monotone-amplitude curves → R²>0.70 | PASS |
| `gkam_convergence` | smooth data → converged==true, iterations≤max_iter | PASS |
| `gkam_invalid_inputs` | empty predictors/mismatched n/argvals_list mismatch → FdarError | PASS |
| `gsam_matches_fam_identity` | fixed bandwidth → fitted values within 1e-6 of fam | PASS |
| `gsam_ncomp_too_large` | ncomp > min(n,m) → InvalidParameter | PASS |
| `gsam_output_shapes` | component_fits.len()==ncomp, fitted_values.len()==n | PASS |

## Deviations from Plan

### Auto-fix: Clippy `manual_clamp` (Rule 1 — Bug)

- **Found during:** Post-implementation clippy gate
- **Issue:** `max_ncomp.min(10).max(1)` triggers `clippy::manual_clamp`
- **Fix:** Replaced with `max_ncomp.clamp(1, 10)` in `resolve_ncomp_additive`
- **Files modified:** `fdars-core/src/scalar_on_function/additive.rs`
- **Commit:** b0b35917

### Implementation pattern: Tasks 2+3 implemented alongside Task 1

The plan specified a TDD sequence (Task 1 FAM → Task 2 GKAM → Task 3 GSAM).
In practice the shared infrastructure (`fpc_additive_smooth`, `resolve_ncomp_additive`,
config/result types) was designed to serve all three estimators simultaneously,
so all three were implemented and committed in a single initial commit (6cb6016b).
The per-estimator tests (`gkam_*`, `gsam_*`) all pass as per plan requirements.

## Known Gaps (Documented)

These are known limitations documented in rustdoc, not implementation stubs:

1. **GKAM non-Gaussian links**: logit/log links require IRLS wrapping — noted
   as a documented gap consistent with milestone convention.
2. **GSAM non-Gaussian links**: same as GKAM.
3. **FAM ncomp auto-selection**: uses single-component GCV heuristic (cheapest
   proxy); a proper forward-stepwise approach is more accurate but O(K²) more
   expensive.

## Threat Mitigations Applied (T-31-01, T-31-02, T-31-03)

| Threat | Mitigation in Code |
|--------|-------------------|
| T-31-01 Tamper (input validation) | InvalidDimension at entry for n/m/y.len()/argvals mismatch; InvalidParameter for ncomp > min(n,m) |
| T-31-02 Tamper (NW denominator) | `if den > 1e-15 { num/den } else { fallback }` at every NW evaluation site in GKAM inner loop |
| T-31-03 DoS (GKAM loop) | GkamConfig.max_iter (default 50) hard-bounds backfitting iterations |

## Module Wiring

```
scalar_on_function/mod.rs
  mod additive;
  pub use additive::{ fam, fregre_gkam, fregre_gsam, FamConfig, FamResult,
                      GkamConfig, GkamResult, GsamConfig, GsamResult };

src/lib.rs
  pub use scalar_on_function::{ ..existing.., fam, fregre_gkam, fregre_gsam,
                                 FamConfig, FamResult, GkamConfig, GkamResult,
                                 GsamConfig, GsamResult };
```

## R Baseline Divergences

All three divergences are documented in module-level rustdoc:
- **FAM**: fdars uses `optim_bandwidth` GCV (not fdapace PACE); no backfitting loop
- **GKAM**: fdars avoids materialising n×n hat matrix; Gaussian identity link only
- **GSAM**: fdars uses NW on FPC scores (not mgcv penalised splines); identity link only

## Commits

| Hash | Message |
|------|---------|
| 6cb6016b | feat(31-01): Task 1 — FAM tracer (fam, FamConfig, FamResult) wired end-to-end |
| b0b35917 | fix(31-01): clippy manual_clamp in resolve_ncomp_additive |

## Self-Check

Files created/modified:
- `fdars-core/src/scalar_on_function/additive.rs` — FOUND
- `fdars-core/src/scalar_on_function/mod.rs` — FOUND (mod additive + pub use)
- `fdars-core/src/lib.rs` — FOUND (fam, fregre_gkam, fregre_gsam + types)

Commits:
- 6cb6016b — FOUND
- b0b35917 — FOUND

Gate results:
- `cargo test -p fdars-core --features linalg,parallel additive` — 10 pass, 0 fail
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — PASS

## Self-Check: PASSED
