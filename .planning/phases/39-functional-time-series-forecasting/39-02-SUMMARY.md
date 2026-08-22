---
phase: 39-functional-time-series-forecasting
plan: 02
subsystem: fts
tags: [time-series, forecasting, multi-step, dynamic-update, ftsm]

requires:
  - phase: 39-01
    provides: ftsm fit, ArModel + reconstruction arithmetic, FtsmResult/FtsmForecastResult
provides:
  - "ftsm_forecast_multistep: iterative multi-step (h>1) per-horizon forecast curves"
  - "ftsm_update: dynamic update projecting new obs onto frozen loadings + AR re-fit (no FPCA refit)"
  - "ftsm_forecast refactored to delegate to ftsm_forecast_multistep (h=1 bit-identical)"
affects: [39-03]

actuals:
  tokens: 12000
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Single-arithmetic-path delegation (ftsm_forecast → ftsm_forecast_multistep) to guarantee h=1 consistency"
    - "Frozen-loadings projection update (no FPCA refit) for online-style dynamic forecasting"

key-files:
  created: []
  modified:
    - fdars-core/src/fts/forecast.rs
    - fdars-core/src/fts/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "ftsm_forecast is a thin wrapper over ftsm_forecast_multistep so the h=1 output is bit-identical by construction (no duplicated reconstruction arithmetic)."
  - "ftsm_update freezes mean/rotation/weights and re-projects new rows via FpcaResult::project arithmetic — never calls fdata_to_pc_1d."
  - "Documented divergence: the frozen mean can drift over long update sequences → periodic full ftsm refit recommended; hence update-vs-refit agreement is within 1% relative-L2, not machine epsilon."

patterns-established:
  - "Dynamic update returns a new FtsmResult with extended scores/fitted and rebuilt AR diagnostics, preserving frozen FPCA state."

requirements-completed: [FTS-01-04, FTS-01-05]

coverage:
  - id: D1
    description: "ftsm_forecast_multistep returns per-horizon forecast curves for h>1; h=1 is bit-identical to ftsm_forecast."
    requirement: "FTS-01-05"
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/forecast.rs::tests::multistep_h1_equals_single_step, multistep_returns_h_rows, multistep_rejects_h_zero"
        status: pass
    human_judgment: false
  - id: D2
    description: "ftsm_update updates the fit without refitting FPCA and agrees with a full refit within 1% relative-L2."
    requirement: "FTS-01-04"
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/forecast.rs::tests::update_agrees_with_refit, update_freezes_loadings, update_extends_scores, update_rejects_bad_shape"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-08-22
status: complete
---

# Phase 39 Plan 02: Multi-step + Dynamic Update Summary

**Forecasts now extend to arbitrary horizons via iterative AR plug-in (`ftsm_forecast_multistep`), and an existing fit can be updated in place as new curves arrive (`ftsm_update`) by projecting onto frozen FPC loadings and re-fitting the score AR models — no FPCA recomputation.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 2/2
- **Tests:** 7 new inline tests (19 total in module), all passing

## Accomplishments

- `ftsm_forecast_multistep(fit, h, argvals)` → `h × m` per-horizon forecast curves; `ftsm_forecast` refactored to delegate (h=1 bit-identical, verified < 1e-12).
- `ftsm_update(fit, new_curve, argvals)` → updated `FtsmResult` with extended scores/fitted and re-fit AR diagnostics; mean/rotation/weights frozen (asserted bitwise equal).
- Both exported via `fts/mod.rs` + crate root.

## Verification

- `cargo test -p fdars-core --features linalg,parallel fts::forecast` → 19 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` → clean.
- No existing signature changed; no new dependency.
