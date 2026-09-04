---
gsd_state_version: 1.0
milestone: v0.37.0
milestone_name: "WAV: Wavelet-Domain Functional Regression"
status: Awaiting next milestone
stopped_at: Phase 71 complete — all phases complete
last_updated: "2026-09-04T21:38:30.726Z"
last_activity: 2026-09-04
last_activity_desc: Milestone v0.37.0 completed and archived
state_head: b1b4006ceb9cbabb09a5cc4011f278e7b3c3704f
progress:
  total_phases: 3
  completed_phases: 3
  total_plans: 5
  completed_plans: 5
current_phase: 71
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-03)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against reference ecosystems — this milestone promotes GAP-07 (wavelet-domain functional regression, `wcr`/`wnet`), rank 6 in the v0.31.0 `GAP-BACKLOG.md`.
**Current focus:** Phase 69 — Discrete Wavelet Transform Primitive

## Current Position

Phase: Milestone v0.37.0 complete
Plan: —
Status: Awaiting next milestone
Last activity: 2026-09-04 — Milestone v0.37.0 completed and archived

## Milestone Roadmap (v0.37.0)

Three phases, 6 requirements (WAV-01..WAV-06) — an implementation milestone promoting GAP-07 (score 1.73, L-effort), adding a discrete wavelet transform (DWT) primitive plus wavelet-domain regularized scalar-on-function regression (`wcr` PCR/PLS + `wnet` elastic-net), for spiky/localized functional predictors in a sparse wavelet basis. Gaussian response only (binomial deferred to WAV-F1). Reference baseline refund@0.1-38 (`wcr`, `wnet`); numeric-output parity is the goal, not basis-internal parity. Likely a new DWT module (`wavelet.rs` or `wavelet/`) + a `wavelet_regression.rs` (or `wcr`/`wnet` submodule) with additive crate-root/prelude re-exports. Additive/non-breaking (protects R + WASM bindings + 28 examples), **no new crate dependency**; **publishes to crates.io on the `v0.37.0` tag** (crate bump 0.36.0 → 0.37.0). Strict dependency chain: DWT primitive → wavelet-domain regressors (`wcr`, `wnet`) → prediction + integration. Fine granularity; the chain cannot be reordered (the DWT is foundational to both regressors; predict + exports consume the fitted regressors). Phase numbering continues from v0.36.0 (ended at 68) → Phase 69.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 69 — Discrete Wavelet Transform Primitive | WAV-01, WAV-02 | New in-crate DWT (crate has `rustfft` but no discrete wavelet transform). Forward/inverse orthogonal DWT — Haar (db1) + Daubechies db2–dbN filter families, multi-level decomposition, perfect reconstruction. Selectable periodic + symmetric boundary extension; handles arbitrary (non-power-of-2) lengths; descriptive `FdarError` on invalid inputs (never panic). Foundational — precedes Phases 70/71. Gates (known-answer testable): forward→inverse round-trip ≤1e-10 rel for Haar + db2–dbN across levels; both boundary modes reconstruct on non-power-of-2 lengths; Haar single-level matches hand-computed sum/difference-over-√2 coefficients; invalid family/order/level/empty → descriptive `FdarError`, no NaN. |
| 70 — Wavelet-Domain Regressors (`wcr` + `wnet`) | WAV-03, WAV-04 | Two Gaussian-response wavelet-domain scalar-on-function regressors, each transforming curves to wavelet coefficients via the Phase 69 DWT. `wcr`: PCR/PLS in wavelet-coefficient space, reusing fdars' FPCR/PLS patterns (`scalar_on_function/`). `wnet`: elastic-net (lasso + ridge) via coordinate descent + soft-thresholding reusing `scalar_on_function/additive.rs` GroupLasso machinery, with sparse coefficient recovery + cross-validated λ. Both return β(t), intercept, fitted values (`wnet` also selected coefficients). `wcr`/`wnet` independent of each other once the DWT exists. Gates: `wcr` (PCR + PLS) recovers known β(t) on spanning full-rank synthetic data; `wnet` recovers sparse support + deterministic CV-λ + non-degenerate fit tracking injected signal; both validate inputs (→ `FdarError`, never panic) with finite/NaN-free outputs. |
| 71 — Prediction, Diagnostics & Integration | WAV-05, WAV-06 | Out-of-sample `predict` on new curves for both `wcr` and `wnet` fitted results (self-consistent: re-passed training curves reproduce training fitted values within tolerance); coefficient-function + fitted-value accessors. Full crate-root + prelude re-exports of the public surface (DWT + `wcr` + `wnet` + config/result types + predict); running end-to-end module doctest under `cargo test --doc`. Additive/non-breaking. Gates: `predict` self-consistent + finite on new curves for both fits; accessors return expected outputs; full surface crate-root/prelude reachable; module doctest passes under `cargo test --doc`; whole-crate `cargo test` + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` + `cargo fmt --check` green (R + WASM + 28 examples unaffected). |

**Execution order (dependency-driven — strict chain):** 69 → 70 → 71. No reordering: the DWT primitive (Phase 69) is foundational to both wavelet-domain regressors (Phase 70); prediction + integration/exports (Phase 71) consume the fitted regressors. Phase 69 front-loads the DWT numerical risk (perfect-reconstruction round-trip); Phase 70 adds the two regressors atop the shared DWT + reuse patterns; Phase 71 adds prediction, diagnostics, the public surface + doctest. `wcr` and `wnet` are independent of each other within Phase 70.

## Performance Metrics

**Velocity:**

- Total plans completed: 108+ (across v0.14.0–v0.35.0)
- Average duration: — min
- Total execution time: — hours

**By Phase (prior milestones):**

| Phase | Milestone | Plans |
|-------|-----------|-------|
| 01–09 | v0.14.0 | 21 |
| 10–45 | v0.15.0–v0.29.0 | 63 |
| 46–51 | v0.30.0 | 23 |
| 52–53 | v0.31.0 | 7 |
| 54–56 | v0.32.0 | 3 |
| 57–60 | v0.33.0 | 4 |
| 61–63 | v0.34.0 | 3 |
| 64–65 | v0.35.0 | 4 |
| 66–68 | v0.36.0 | 3/3 |
| 69–71 | v0.37.0 | 0/? (pending) |

**Recent Trend:**

- Last milestone: v0.36.0 phases 66–68 (3 plans) — audit PASSED 5/5, crate 0.35.0 → 0.36.0. Promoted GAP-06 (PEER / longitudinal PEER).
- Trend: v0.37.0 stays in implementation shape — real code, normal test/clippy/fmt gates, crate publish on the `v0.37.0` tag. Reuse-heavy (`scalar_on_function/` FPCR/PLS for `wcr`, `scalar_on_function/additive.rs` GroupLasso coordinate-descent/soft-threshold for `wnet`, `linalg::cholesky_solve`, `helpers::simpsons_weights`, `regression.rs::fdata_to_pc_1d`), but WAV-01/02 add a genuinely new in-crate DWT primitive (no existing discrete wavelet transform to reuse). Effort L for a mature codebase; likely a new DWT module (`wavelet.rs`/`wavelet/`) + `wavelet_regression.rs`. Three phases driven by a strict DWT → regressors → prediction/integration dependency chain, not padding.

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| (none yet — v0.37.0) | — | — | — |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.37.0 WAV):

- **Implementation milestone, publishes on tag** — v0.37.0 makes real `fdars-core/src/` changes and **will** bump the crate version 0.36.0 → 0.37.0 + publish to crates.io on the `v0.37.0` tag. Normal test/clippy(`--all-targets --features linalg,parallel`)/fmt gates apply. (audit-milestone-no-tag does NOT apply.)
- **New in-crate DWT primitive** — the crate has `rustfft` and a Morlet CWT (`seasonal/strength.rs`) but NO discrete orthogonal wavelet transform. WAV-01/02 build one from scratch (Haar/db1 + Daubechies db2–dbN filter banks, multi-level Mallat pyramid, periodic + symmetric boundary extension, arbitrary lengths). Perfect-reconstruction round-trip is the make-or-break numerical gate. No new crate dependency — implemented in-crate.
- **`wcr` reuses FPCR/PLS; `wnet` reuses GroupLasso** — `wcr` transforms curves → wavelet coefficients, then fits via PCR (and PLS) in coefficient space reusing `scalar_on_function/` FPCR/PLS patterns + `regression.rs::fdata_to_pc_1d`. `wnet` applies lasso + ridge on wavelet coefficients via coordinate descent + soft-thresholding, reusing `scalar_on_function/additive.rs` GroupLasso machinery, with cross-validated λ. Numeric-output parity with refund `wcr`/`wnet` is the goal — NOT basis-internal parity with refund's `wavethresh`/`wmtsa` internals (documented divergences acceptable).
- **Gaussian response only** — binomial/logistic (refund's `family` argument) deferred to WAV-F1. Keeps the numerical-gate surface bounded.
- **DWT scope bounded** — Haar + Daubechies db2–dbN only; Symlets/Coiflets/biorthogonal families + wavelet packets deferred to WAV-F2; 2D/surface DWT deferred to WAV-F3.
- **Reuse-first, no new dependency** — `scalar_on_function/` (FPCR/PLS for `wcr`), `scalar_on_function/additive.rs` (GroupLasso coordinate-descent/soft-threshold for `wnet`), `linalg::cholesky_solve`, `helpers::simpsons_weights`, `regression.rs::fdata_to_pc_1d`. No `Cargo.toml` change; MSRV stays 1.81 (confirm at plan time whether any faer/`linalg`-gated path is needed).
- **Likely a new DWT module + `wavelet_regression.rs`** — DWT in `wavelet.rs` (or `wavelet/`); `wcr`/`wnet` in `wavelet_regression.rs` (or a `wcr`/`wnet` submodule). Peer of `kshape.rs`/`kernel_kmeans.rs`/`optimal_design.rs`. Additive crate-root/prelude re-exports of the full surface (DWT + `wcr` + `wnet` + config/result types + predict) land in the final phase (71) to avoid exposing a partial public API mid-milestone.
- **Additive/non-breaking** — zero changes to existing public signatures (protects R + WASM bindings + 28 examples); only the new wavelet module(s) + additive `lib.rs`/`prelude.rs` re-exports.
- **Phase numbering continues** — v0.36.0 ended at Phase 68 → v0.37.0 starts at Phase 69. No reset.
- **6 requirements → 3 phases** (fine granularity, strict dependency chain): Phase 69 WAV-01/02, Phase 70 WAV-03/04, Phase 71 WAV-05/06. All 6 mapped, no orphans, no duplicates. (WAV-01 forward/inverse DWT + WAV-02 boundary handling paired as the DWT primitive; WAV-03 `wcr` + WAV-04 `wnet` paired as the wavelet-domain regressor layer, independent of each other; WAV-05 predict + WAV-06 integration paired as the prediction/exposure layer.)

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; the additive wavelet surface (DWT + `wcr`/`wnet`) should be exposed to R/WASM bindings in a follow-up, not this milestone.

### Blockers/Concerns

- **No research/SUMMARY.md** — research was intentionally skipped for this milestone (method well-scoped from the refund `wcr`/`wnet` baseline + a standard orthogonal DWT). Numerical make-or-break gates warrant known-answer tests: DWT perfect-reconstruction round-trip, Haar hand-computed coefficients, `wcr` β(t) recovery, `wnet` sparse-support recovery + deterministic CV-λ, `predict` self-consistency, module doctest under `cargo test --doc`. Non-blocking for the roadmap.
- **DWT is genuinely new code** — unlike prior implementation milestones that mostly re-wired existing machinery, WAV-01/02 add a from-scratch filter-bank DWT (no in-crate discrete-wavelet code to reuse). Front-loaded as Phase 69 so the perfect-reconstruction numerical risk is retired before the regressors build on it. Beware boundary-mode edge cases (non-power-of-2 lengths, symmetric vs periodic extension) and Daubechies filter-coefficient correctness.
- **`wnet` GroupLasso reuse fit** — confirm at plan time that `scalar_on_function/additive.rs`'s coordinate-descent/soft-threshold machinery accepts the wavelet-coefficient design (per-coefficient lasso + ridge, not group structure) without modification, else scope a thin additive adapter (still additive/non-breaking).
- Historical build/CI hazards (MEMORY.md) apply this implementation milestone: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds; prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds; per-phase impl-subagent pattern (plan→code→test→--no-verify commit→artifacts) dodged executor stalls on v0.32.0 GAK; audit-milestone-no-tag does NOT apply (this ships code → tag as normal).
- Pre-existing serde build break (NOT v0.37.0): `fdars-core/src/shapelet/classifier.rs` `ShapeletTransformClassifier` (Phase 60, commit ea39c623) derives serde but embeds `ClassifFit` which lacks serde derives → `cargo build --features serde` fails. Independent of WAV; new wavelet types should be serde-clean (or serde-gated) if they derive serde. Candidate GSD-ready backlog fix: add serde-gated derives to `ClassifFit` + fitted sub-structs.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Backlog | GAP-08 (autodiff-compatible / differentiable FDA core — invasive generics refactor, score 1.73, L) — the last remaining backlog item after GAP-07 | Deferred | v0.36.0 | future milestone |
| Wavelet-regression | WAV-F1 (binomial/logistic + GLM-family wavelet-domain regression — refund `family`); WAV-F2 (Symlets/Coiflets/biorthogonal families + wavelet packets); WAV-F3 (2D/surface DWT + regression) | Deferred | v0.37.0 | future milestone |
| PEER | Non-Gaussian / GLM PEER families; basis-expansion PEER matching refund's exact internal `pentype`/basis representation | Deferred | v0.36.0 | future milestone |
| Optimal-design | FOD-BREADTH (SR-criterion, exhaustive/branch-and-bound, CV-ridge, rank-1 Cholesky update, off-grid interpolated candidates) | Deferred | v0.35.0 | future milestone |
| Shape-clustering | KSH-BREADTH (multivariate/variable-length SBD, hierarchical/other clustering families) | Deferred | v0.34.0 | future milestone |
| Shapelets | LSH-01 (gradient learning-shapelets) — needs autodiff through the distance; ties to GAP-08 | Deferred | v0.33.0 | future milestone |
| Kernel-methods | SVM-01 (native in-crate kernel-SVM / QP solver) — Gram export (GAK-05/06) covers the use case in the interim | Deferred | v0.32.0 | future milestone |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-04T15:30:00.000Z
Stopped at: Phase 71 complete — all phases complete
Resume file: None

## Operator Next Steps

- Start the next milestone with /gsd-new-milestone
