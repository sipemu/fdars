---
phase: 39-functional-time-series-forecasting
plan: 03
subsystem: fts
tags: [time-series, forecasting, pls, fplsr]

requires:
  - phase: 39-01
    provides: forecast.rs module scaffolding, validate_fts_input, FTS result-struct conventions
  - phase: (scalar_on_function)
    provides: fregre_pls / predict_fregre_pls PLS regression machinery
provides:
  - "fplsr: functional PLS forecasting variant (lag-1 per-point PLS) — 1-step forecast + in-sample fitted curves"
  - "FplsrResult struct (forecast 1 x m, fitted (n-1) x m, ncomp) — crate-root re-exported"

actuals:
  tokens: 11000
  tasks: 2
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Per-evaluation-point scalar PLS to emulate a functional-response PLS forecast (Option A) reusing fregre_pls"

key-files:
  created: []
  modified:
    - fdars-core/src/fts/forecast.rs
    - fdars-core/src/fts/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "fplsr uses a lag-1 design (X_cur = rows 0..n-2, X_next = rows 1..n-1) and fits one scalar PLS regression per evaluation point via fregre_pls (which internally uses fdata_to_pls_1d) — reuses shipped PLS machinery, no new NIPALS subsystem."
  - "ncomp clamped to min(ncomp, n-1, m) to avoid overfitting / rank deficiency on the lag-1 design."
  - "Documented divergence: per-point scalar PLS rather than ftsa's unified NIPALS/SIMPLS functional operator (functionally equivalent for point prediction)."

patterns-established:
  - "Dedicated full-rank AR-driven test generator (3 components + broadband noise) so per-point PLS OLS is well-conditioned (rank-2 synthetic data made Cholesky singular)."

requirements-completed: [FTS-01-03]

coverage:
  - id: D1
    description: "fplsr produces a finite lag-1 PLS 1-step forecast + (n-1) x m in-sample fitted curves, deterministic, no worse than a naive last-curve baseline."
    requirement: "FTS-01-03"
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/forecast.rs::tests::fplsr_produces_finite_forecast, fplsr_no_worse_than_naive, fplsr_deterministic"
        status: pass
    human_judgment: false
  - id: D2
    description: "Invalid inputs (empty, ncomp==0, n<3, argvals mismatch) return the documented FdarError."
    requirement: "FTS-01-03"
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/forecast.rs::tests::fplsr_rejects_bad_input"
        status: pass
    human_judgment: false

duration: 20min
completed: 2026-08-22
status: complete
---

# Phase 39 Plan 03: Functional PLS Forecasting Summary

**`fplsr` adds a PLS-score alternative to the FPC-score AR path: a lag-1 design (regress next curve on current curve) solved as one scalar PLS regression per evaluation point, reusing the shipped `fregre_pls` machinery, producing a one-step forecast curve plus in-sample fitted curves.**

## Performance

- **Duration:** ~20 min
- **Tasks:** 2/2
- **Tests:** 4 new inline tests (23 total in module), all passing

## Accomplishments

- `fplsr(data, ncomp, argvals)` and `FplsrResult` (forecast 1×m, fitted (n-1)×m, ncomp), crate-root re-exported.
- Reuses `fregre_pls`/`predict_fregre_pls` per evaluation point over the lag-1 design; no new PLS subsystem, no new dependency.
- Handles rank/overfitting via `ncomp = min(ncomp, n-1, m)`; all invalid inputs return `FdarError`.

## Verification

- `cargo test -p fdars-core --features linalg,parallel fts::forecast` → 23 passed.
- `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings` → clean.
- No existing signature changed; no new crate dependency.

## Note

- Rank-2 synthetic curves made the per-point OLS Cholesky singular; a dedicated full-rank generator (3 AR components + broadband noise) is used for the fplsr tests. All five FTS-01 requirements (FTS-01-01..05) are now implemented across Plans 01–03.
