---
gsd_state_version: 1.0
milestone: v0.36.0
milestone_name: PEER — Structured-Penalty & Longitudinal Scalar-on-Function Regression
status: planning
last_updated: "2026-09-03T21:07:08.749Z"
last_activity: 2026-09-03
progress:
  total_phases: 3
  completed_phases: 0
  total_plans: 0
  completed_plans: 0
  percent: 0
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-09-03)

**Core value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against reference ecosystems — this milestone promotes GAP-06 (PEER / longitudinal PEER, structured-penalty scalar-on-function regression), rank 5 in the v0.31.0 `GAP-BACKLOG.md`.
**Current focus:** Phase 66 — Core PEER Estimator & Penalty Families

## Current Position

Phase: Not started (roadmap created)
Plan: —
Status: Roadmap created — ready to plan Phase 66
Last activity: 2026-09-03 — Roadmap created for v0.36.0 (Phases 66–68)

## Milestone Roadmap (v0.36.0)

Three phases, 5 requirements (PER-01..PER-05) — an implementation milestone promoting GAP-06 (score 2.12, M-effort), adding PEER (Partially Empirical Eigenvectors for Regression) structured-penalty scalar-on-function regression + the longitudinal `lpeer` variant. Reference baseline refund@0.1-38 (`peer`, `lpeer`), NOT captured in the v0.18.0 R audit. Likely a new top-level `src/peer.rs` (or `peer/` submodule) with additive crate-root/prelude re-exports. Additive/non-breaking (protects R + WASM bindings + 28 examples), **no new crate dependency**; **publishes to crates.io on the `v0.36.0` tag** (crate bump 0.35.0 → 0.36.0). Strict dependency chain: core estimator + penalties → λ-selection → longitudinal extension + prediction/exports. Fine granularity; the chain cannot be reordered (λ-selection wraps the core penalized fit; lpeer reuses PEER's penalty machinery + REML path; predict consumes the fitted result). Phase numbering continues from v0.35.0 (ended at 65) → Phase 66.

| Phase | Requirements | Notes |
|-------|--------------|-------|
| 66 — Core PEER Estimator & Penalty Families | PER-01, PER-02 | Public `peer()` estimator estimating β(t) via the partially-empirical-eigenvector decomposition (null-space + range-space of the penalty operator) — distinct from FPCR/`pfr` — returning a result struct (β(t), intercept, fitted values, df/selection diagnostics). Three penalty families via a penalty-type parameter/enum: ridge/identity (classical), 2nd-difference/roughness (reuse `smooth_basis::penalty_matrix`), caller-supplied structured/"decree" Q (PEER's signature partitioned-domain a-priori structure). Reuse-first: `scalar_on_function/`, `function_on_scalar.rs` (`penalized_solve`), `smooth_basis.rs`/`function_on_scalar_2d.rs` penalty builders, `regression.rs::fdata_to_pc_1d` (FPC basis), `linalg::cholesky_solve`, `helpers::simpsons_weights`. Fixed λ here (auto-selection lands Phase 67). Gates (known-answer testable): β(t) recovery on synthetic data; all three penalty families fit without error; structured Q yields a partition-aware β(t) distinct from plain roughness; wrong-dimension Q → descriptive `FdarError` (never panic), no NaN β(t). |
| 67 — Automatic λ Selection — GCV + REML | PER-03 | Smoothing parameter λ chosen automatically by GCV grid search (reuse the `penalized_solve` + GCV hat-matrix-trace pattern) OR REML/mixed-model estimation (reuse `famm::fit_scalar_mixed_model`), selectable via config; explicit λ honored verbatim when supplied. Matches refund's default. Wraps the Phase 66 core fit. Gates: GCV picks the GCV-minimizing λ, deterministic across runs; REML picks a sensible positive λ and REML-vs-GCV β(t) agree within tolerance; explicit λ used verbatim (no search) in diagnostics; both selectors land a non-degenerate λ recovering the known β(t) on synthetic SNR data. |
| 68 — Longitudinal PEER, Prediction & Integration | PER-04, PER-05 | Public `lpeer()` longitudinal extension — repeated per-subject measurements with subject-level random effects fitted through `famm::fit_scalar_mixed_model` (REML EM) — returning a result struct with the (time-varying) coefficient function + variance components. Out-of-sample `predict` on new curves from a fitted `peer`/`lpeer` result; coefficient function + fitted values accessors; full crate-root + prelude re-exports; module doctest of the end-to-end workflow. Gates: variance components non-negative; lpeer recovers β(t) + tracks injected between-subject variance on synthetic longitudinal data; `predict` self-consistent (re-passed training curves == training fitted values) + finite on new curves; full public surface crate-root/prelude reachable; module doctest passes under `cargo test --doc`. |

**Execution order (dependency-driven — strict chain):** 66 → 67 → 68. No reordering or parallelization: Phase 67's λ-selection wraps Phase 66's core penalized fit; Phase 68's `lpeer` reuses PEER's penalty machinery and the REML λ-selection path, and `predict` consumes the fitted result. Phase 66 front-loads the core-estimator + penalty-family numerical risk; Phase 67 adds automatic λ selection; Phase 68 adds the longitudinal layer + prediction + integration/exports + doctest.

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
| 66–68 | v0.36.0 | 0/TBD |

**Recent Trend:**

- Last milestone: v0.35.0 phases 64–65 (4 plans) — audit PASSED 5/5, shipped `v0.35.0` (crate 0.34.0 → 0.35.0). Promoted GAP-05 (Optimal Experimental Design / FOptDes).
- Trend: v0.36.0 stays in implementation shape — real code, normal test/clippy/fmt gates, crate publish on tag. Reuse-heavy (`scalar_on_function/`, `function_on_scalar.rs` `penalized_solve` + GCV, `smooth_basis`/`function_on_scalar_2d` penalty builders, `famm::fit_scalar_mixed_model` REML EM, `regression.rs::fdata_to_pc_1d`, `linalg::cholesky_solve`, `helpers::simpsons_weights`), effort M for a mature codebase, likely ONE new file (`peer.rs`) or a small `peer/` submodule. Three phases driven by a strict core-estimator → λ-selection → longitudinal+prediction dependency chain, not padding.

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.

Relevant to current work (v0.36.0):

- **Implementation milestone, publishes on tag** — v0.36.0 makes real `fdars-core/src/` changes and **will** bump the crate version + publish to crates.io on the `v0.36.0` tag. Normal test/clippy(`--all-targets`)/fmt gates apply. (audit-milestone-no-tag does NOT apply.)
- **PEER, not FPCR/`pfr`** — the distinguishing feature is the partially-empirical-eigenvector decomposition: the penalty operator's null-space (unpenalized) + range-space (penalized) partition, letting the caller inject a-priori signal structure. Numeric-output parity with refund `peer`/`lpeer` is the goal — NOT implementation/basis-internal parity (fdars models β on its own FPC/penalty basis; documented divergences from refund's `pentype`/basis internals are acceptable, per REQUIREMENTS Out-of-Scope).
- **Three penalty families via one enum** — ridge/identity (classical), 2nd-difference/roughness (reuse `smooth_basis::penalty_matrix`), caller-supplied structured/"decree" Q (partitioned-domain a-priori structure — the signature feature). A single penalty-type parameter/enum selects among them; Q is validated for dimension (wrong-dim → `FdarError`, never panic).
- **λ selection: GCV + REML, config-selectable** — GCV grid search reusing the `function_on_scalar.rs` `penalized_solve` + hat-matrix-trace GCV pattern; REML/mixed-model estimation reusing `famm::fit_scalar_mixed_model` (REML EM). Explicit λ honored verbatim (no search). Matches refund's default. λ-selection wraps the core fit (Phase 67 depends on Phase 66).
- **Longitudinal `lpeer` via `famm`** — repeated per-subject measurements with subject-level random effects fitted through `famm::fit_scalar_mixed_model` (REML EM). Variance components must be non-negative (structural gate). Reuses PEER's penalty machinery + the REML λ-selection path.
- **Reuse-first, no new dependency** — `scalar_on_function/` (SoF regression family), `function_on_scalar.rs` (`penalized_solve` + GCV), `smooth_basis.rs` / `function_on_scalar_2d.rs` (`penalty_matrix` 2nd-difference roughness, `bspline_penalty_matrix`, `fourier_penalty_matrix`), `famm.rs` (`fit_scalar_mixed_model` REML EM), `regression.rs::fdata_to_pc_1d` (FPC scores), `linalg::cholesky_solve`, `helpers::simpsons_weights`. No `Cargo.toml` change; MSRV stays 1.81 (confirm whether any faer/`linalg`-gated path is needed at plan time).
- **New top-level `peer.rs` (or `peer/` submodule)** — self-contained algorithm; peer of `kshape.rs`/`kernel_kmeans.rs`/`optimal_design.rs`. If a `peer/` submodule, keep to the natural core/longitudinal split. Additive crate-root/prelude re-exports of the full surface (estimators, result structs, penalty enum, predict) land in the final phase (68) to avoid exposing a partial public API mid-milestone.
- **Additive/non-breaking** — zero changes to existing public signatures (protects R + WASM bindings + 28 examples); only the new `peer` module + additive `lib.rs`/`prelude.rs` re-exports.
- **Phase numbering continues** — v0.35.0 ended at Phase 65 → v0.36.0 starts at Phase 66. No reset.
- **5 requirements → 3 phases** (fine granularity, strict dependency chain): Phase 66 PER-01/02, Phase 67 PER-03, Phase 68 PER-04/05. All 5 mapped, no orphans, no duplicates. (PER-01 core estimator and PER-02 penalty families paired since the penalty is a parameter of the estimator; PER-04 lpeer + PER-05 predict/exports paired as the longitudinal + integration layer.)

### Pending Todos

- **Migrate `fdars-r` R wrapper to use the `FdMatrix` API** (issue `fdars-j75`) — carried forward; the additive PEER/lpeer surface should be exposed to R/WASM bindings in a follow-up, not this milestone.

### Blockers/Concerns

- **No research/SUMMARY.md** — research was skipped for this milestone (method well-scoped from the refund `peer`/`lpeer` baseline). Numerical make-or-break gates warrant known-answer tests cross-checked against the PEER decomposition + penalty formulas (β(t) recovery, penalty-family well-formedness, structured-Q partition awareness, GCV/REML λ agreement, variance-component non-negativity, predict self-consistency). Non-blocking for the roadmap.
- **REML/famm reuse fit** — Phase 67 (REML λ) and Phase 68 (lpeer) both lean on `famm::fit_scalar_mixed_model`; confirm at plan time that its REML EM interface accepts the PEER penalty structure (mapping the range-space penalty to a random-effect precision) without modification, else scope a thin additive adapter (still additive/non-breaking).
- Historical build/CI hazards (MEMORY.md) apply this implementation milestone: run clippy with `--all-targets --features linalg,parallel -- -D warnings` (CI lints test/bench code); run `cargo fmt` per commit (`--no-verify` commits leave fmt drift); watch `/tmp` and `target/` disk pressure on full builds; prefer inline execution + `commit --no-verify` after out-of-band gates if executor subagents stall on long cargo builds; audit-milestone-no-tag does NOT apply (this ships code → tag as normal).
- Pre-existing serde build break (NOT v0.36.0): `fdars-core/src/shapelet/classifier.rs` `ShapeletTransformClassifier` (Phase 60, commit ea39c623) derives serde but embeds `ClassifFit` which lacks serde derives → `cargo build --features serde` fails. Independent of PEER; new PEER types should be serde-clean (or serde-gated) if they derive serde. Candidate GSD-ready backlog fix: add serde-gated derives to `ClassifFit` + fitted sub-structs.

## Deferred Items

Items acknowledged and deferred, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Backlog | GAP-07 (wavelet-domain functional regression, `wcr`/`wnet` — needs a DWT built first, score 1.73, L) | Deferred | v0.36.0 | future milestone |
| Backlog | GAP-08 (autodiff-compatible / differentiable FDA core — invasive generics refactor, score 1.73, L) | Deferred | v0.36.0 | future milestone |
| PEER | Non-Gaussian / GLM PEER families; basis-expansion PEER matching refund's exact internal `pentype`/basis representation | Deferred | v0.36.0 | future milestone |
| Optimal-design | FOD-BREADTH (SR-criterion, exhaustive/branch-and-bound, CV-ridge, rank-1 Cholesky update, off-grid interpolated candidates) | Deferred | v0.35.0 | future milestone |
| Shape-clustering | KSH-BREADTH (multivariate/variable-length SBD, hierarchical/other clustering families) | Deferred | v0.34.0 | future milestone |
| Shapelets | LSH-01 (gradient learning-shapelets) — needs autodiff through the distance; ties to GAP-08 | Deferred | v0.33.0 | future milestone |
| Kernel-methods | SVM-01 (native in-crate kernel-SVM / QP solver) — Gram export (GAK-05/06) covers the use case in the interim | Deferred | v0.32.0 | future milestone |
| API-breaking | APIB-01 — breaking removal of the 6 `#[deprecated]` forms from v0.30.0 | Deferred | v0.30.0 | future 1.0-readiness |

## Session Continuity

Last session: 2026-09-03T21:07:08.749Z
Stopped at: Roadmap created for v0.36.0 (Phases 66–68); 5 requirements (PER-01..PER-05) mapped, traceability updated
Resume file: None

## Operator Next Steps

- Plan the first phase with `/gsd-plan-phase 66`
