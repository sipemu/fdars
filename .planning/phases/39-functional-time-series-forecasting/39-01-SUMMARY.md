---
phase: 39-functional-time-series-forecasting
plan: 01
subsystem: fts
tags: [fpca, time-series, forecasting, yule-walker, ar-model, ftsm]

requires:
  - phase: 34-functional-time-series
    provides: fts module scaffolding (fts/mod.rs, fts/acf.rs validation + Levinson-Durbin patterns)
  - phase: (regression)
    provides: fdata_to_pc_1d dense FPCA + FpcaResult::reconstruct
provides:
  - "ftsm: FPCA-based functional time-series model fit (mean + loadings + score-series + fitted curves + per-component AR diagnostics)"
  - "ftsm_forecast: h-step FPC-score AR forecast reconstructed into forecast curves"
  - "Private Yule-Walker AR machinery: scalar_acov, levinson_durbin_yw, ArModel (fit/forecast)"
  - "Result structs FtsmResult, FtsmForecastResult, ArModelResult (crate-root re-exported)"
affects: [39-02, 39-03]

actuals:
  tokens: 17000
  tasks: 3
  commits: 1

tech-stack:
  added: []
  patterns:
    - "FPCA-delegate + per-score univariate AR(p) forecasting"
    - "Deterministic Yule-Walker (Levinson-Durbin) AR estimation with AIC order selection, no new dependency"

key-files:
  created:
    - fdars-core/src/fts/forecast.rs
  modified:
    - fdars-core/src/fts/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "ftsm delegates FPCA to fdata_to_pc_1d and reconstruction to FpcaResult::reconstruct — no custom SVD."
  - "Each FPC-score sequence gets an independent univariate AR(p) via Yule-Walker + AIC (p_max = min(floor(10*log10(n)), n-1, n/4))."
  - "ftsm_forecast implements the full h-horizon iterative plug-in (h rows); h=1 is the single-step case. Plan 02 adds ftsm_forecast_multistep + h=1 consistency test."
  - "ar_models exposed as public ArModelResult diagnostics; the private ArModel is re-derived in ftsm_forecast from diagnostics + score-column history (scores are FPCA-centered so column mean ≈ 0)."

patterns-established:
  - "Yule-Walker AR fit reusing acf.rs's Levinson-Durbin structural pattern (nu[k-1].abs()<1e-12 early-exit guard) on raw autocovariances."
  - "Deterministic LCG pseudo-white-noise for AR synthetic-series tests (no RNG dependency)."

requirements-completed: [FTS-01-01, FTS-01-02]

coverage:
  - id: D1
    description: "ftsm fits an FPCA-based functional time-series model and reconstructs fitted curves recovering the input within 1% relative-L2."
    requirement: "FTS-01-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/forecast.rs::tests::ftsm_fitted_recovers_input"
        status: pass
    human_judgment: false
  - id: D2
    description: "ftsm_forecast recovers the AR one-step prediction on a known AR(1) score series and beats a naive last-curve baseline."
    requirement: "FTS-01-02"
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/forecast.rs::tests::forecast_recovers_ar_one_step, forecast_beats_naive_baseline"
        status: pass
    human_judgment: false
  - id: D3
    description: "Invalid inputs return FdarError (never panic); ftsm is deterministic."
    requirement: "FTS-01-01"
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/forecast.rs::tests::ftsm_rejects_ncomp_ge_n, ftsm_rejects_empty, ftsm_rejects_argvals_mismatch, forecast_rejects_h_zero, ftsm_deterministic"
        status: pass
    human_judgment: false

duration: 30min
completed: 2026-08-22
status: complete
---

# Phase 39 Plan 01: Functional Time-Series Tracer Summary

**A time-ordered curve series can now be decomposed into an FPCA-based `ftsm` model and forecast one (or more) steps ahead via per-component Yule-Walker AR score models, reconstructed back into curves — the end-to-end tracer through module wiring, FPCA delegation, AR estimation, and reconstruction.**

## Performance

- **Duration:** ~30 min
- **Tasks:** 3/3
- **Tests:** 12 inline `#[cfg(test)]` tests, all passing

## Accomplishments

- New `fdars-core/src/fts/forecast.rs` with the `ftsm` fit and `ftsm_forecast` entry points.
- Private deterministic Yule-Walker AR machinery: `scalar_acov` (1/n convention), `levinson_durbin_yw` (with near-unit-root early-exit guard), and `ArModel` with AIC order selection + iterative plug-in forecasting.
- `FtsmResult`, `FtsmForecastResult`, `ArModelResult` declared in `fts/mod.rs`, re-exported at the crate root.
- Additive/non-breaking: no existing signature changed, no new crate dependency; full clippy `--all-targets --features linalg,parallel -- -D warnings` clean.

## Verification

- `cargo test -p fdars-core --features linalg,parallel fts::forecast` → 12 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` → clean.
- `cargo build` green.

## Notes for Plan 02/03

- `ftsm_forecast` already forecasts the full requested `h` horizons; `ftsm_forecast_multistep` (Plan 02) should delegate to / share this core so the h=1 consistency invariant is trivially guaranteed.
- Dynamic update (Plan 02) should store/reuse the frozen mean/rotation/weights already on `FtsmResult` and project new obs (mirror `FpcaResult::project`) then re-fit `ArModel` per component.
- `fplsr` (Plan 03) will add a `FplsrResult` struct in `fts/mod.rs` + crate-root re-export, and per-point PLS via `fdata_to_pls_1d`.
