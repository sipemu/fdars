# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (shipped 2026-08-12, PR #40) — [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 — Registration Parity & Elastic-FPCA Performance** — Phases 14–15 (shipped 2026-08-12, PR #41) — [archive](milestones/v0.17.0-ROADMAP.md)
- ✅ **v0.18.0 — R-Ecosystem Gap Audit** — Phases 16–19 (shipped 2026-08-15) — [archive](milestones/v0.18.0-ROADMAP.md)
- ✅ **v0.19.0 — Functional Inference Suite** — Phases 20–21 (shipped 2026-08-16) — [archive](milestones/v0.19.0-ROADMAP.md)
- ✅ **v0.20.0 — Table-Stakes Quick Wins** — Phases 22–23 (shipped 2026-08-16) — [archive](milestones/v0.20.0-ROADMAP.md)
- ✅ **v0.21.0 — Functional Regression Completeness** — Phases 24–25 (shipped 2026-08-17) — [archive](milestones/v0.21.0-ROADMAP.md)
- ✅ **v0.22.0 — PACE Sparse FPCA & Elastic Multinomial** — Phases 26–27 (shipped 2026-08-19) — [archive](milestones/v0.22.0-ROADMAP.md)
- ✅ **v0.23.0 — Depth, Outliers & Interval Inference** — Phases 28–30 (shipped 2026-08-20) — [archive](milestones/v0.23.0-ROADMAP.md)
- ✅ **v0.24.0 — Functional Regression & Clustering Breadth** — Phases 31–33 (shipped 2026-08-20) — [archive](milestones/v0.24.0-ROADMAP.md)
- ✅ **v0.25.0 — Serial Dependence, Representation & Density Breadth** — Phases 34–36 (shipped 2026-08-21) — [archive](milestones/v0.25.0-ROADMAP.md)
- ✅ **v0.26.0 — FPCA Breadth & Sparse Covariance** — Phases 37–38 (shipped 2026-08-21) — [archive](milestones/v0.26.0-ROADMAP.md)
- ✅ **v0.27.0 — Functional Time Series & Fréchet Regression** — Phases 39–40 (shipped 2026-08-22) — [archive](milestones/v0.27.0-ROADMAP.md)
- ✅ **v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression** — Phases 41–42 (shipped 2026-08-23) — [archive](milestones/v0.28.0-ROADMAP.md)
- ✅ **v0.29.0 — Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering** — Phases 43–45 (shipped 2026-08-30) — [archive](milestones/v0.29.0-ROADMAP.md)
- ✅ **v0.30.0 — Performance & Consolidation Pass** — Phases 46–51 (shipped 2026-09-01) — [archive](milestones/v0.30.0-ROADMAP.md)
- ✅ **v0.31.0 — Multi-Ecosystem Gap Audit** — Phases 52–53 (shipped 2026-09-02) — [archive](milestones/v0.31.0-ROADMAP.md)
- ✅ **v0.32.0 — Global Alignment Kernel & Kernel Clustering** — Phases 54–56 (shipped 2026-09-02) — [archive](milestones/v0.32.0-ROADMAP.md)
- ✅ **v0.33.0 — Shapelet Transform & Classification** — Phases 57–60 (shipped 2026-09-02) — [archive](milestones/v0.33.0-ROADMAP.md)
- ✅ **v0.34.0 — k-Shape Clustering & Shape-Based Distance** — Phases 61–63 (shipped 2026-09-02) — [archive](milestones/v0.34.0-ROADMAP.md)
- ✅ **v0.35.0 — Optimal Experimental Design for Sparse FDA (FOptDes)** — Phases 64–65 (shipped 2026-09-03) — [archive](milestones/v0.35.0-ROADMAP.md)
- 🚧 **v0.36.0 — PEER: Structured-Penalty & Longitudinal Scalar-on-Function Regression** — Phases 66–68 (in progress)

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (66.1, 66.2): Urgent insertions (marked with INSERTED)

### v0.36.0 — PEER: Structured-Penalty & Longitudinal Scalar-on-Function Regression (Phases 66–68)

Add PEER (Partially Empirical Eigenvectors for Regression) — structured a-priori-penalty scalar-on-function regression — and its longitudinal variant `lpeer`, letting users inject prior signal structure into the coefficient-function penalty. Promotes **GAP-06** (score 2.12, M-effort) from the v0.31.0 `GAP-BACKLOG.md`. Reference baseline refund@0.1-38 (`peer`, `lpeer`). Implementation milestone — real `fdars-core/src/` changes, additive/non-breaking (protects R + WASM bindings + 28 examples), no new crate dependency; likely a new top-level `peer.rs` (or `peer/` submodule). Strict dependency chain: core estimator + penalties → λ-selection → longitudinal extension + prediction/exports. Crate bumps 0.35.0 → **0.36.0**, published on the `v0.36.0` tag.

- [x] **Phase 66: Core PEER Estimator & Penalty Families (PER-01, PER-02)** — the public `peer()` estimator with the partially-empirical-eigenvector decomposition and the three a-priori penalty families (ridge/identity, 2nd-difference/roughness, caller-supplied structured Q) (completed 2026-09-04)
- [ ] **Phase 67: Automatic λ Selection — GCV + REML (PER-03)** — smoothing-parameter λ chosen automatically by GCV grid search or REML/mixed-model estimation, selectable via config; explicit λ honored when supplied
- [ ] **Phase 68: Longitudinal PEER, Prediction & Integration (PER-04, PER-05)** — the `lpeer()` longitudinal extension with subject-level random effects via `famm`, out-of-sample `predict`, crate-root + prelude re-exports, and an end-to-end module doctest

## Phase Details

### Phase 66: Core PEER Estimator & Penalty Families

**Goal**: A user can fit structured-penalty scalar-on-function regression via a public `peer()` estimator that estimates the coefficient function β(t) through the partially-empirical-eigenvector decomposition (null-space + range-space of the penalty operator), choosing among three a-priori penalty families — the estimator that distinguishes PEER from plain FPCR/`pfr`.
**Depends on**: Nothing (first phase of milestone; builds on shipped `scalar_on_function/`, `function_on_scalar.rs`, `smooth_basis.rs`)
**Requirements**: PER-01, PER-02
**Success Criteria** (what must be TRUE):

  1. `peer()` recovers a known β(t) within tolerance on synthetic data where the response was generated from a specified coefficient function, and returns a result struct carrying β(t), intercept, fitted values, and df/selection diagnostics.
  2. A user can select the penalty family via a penalty-type parameter/enum, and the ridge/identity, 2nd-difference/roughness (reusing the existing `penalty_matrix` builder), and caller-supplied structured "decree" Q families each produce a well-formed fit without error.
  3. The structured/"decree" Q penalty produces a β(t) whose shape reflects the caller-supplied partitioned-domain structure (e.g. a partition boundary in Q yields a partition-aware β(t)), demonstrably different from the plain-roughness fit on the same data.
  4. A caller-supplied Q of the wrong dimension (or otherwise invalid penalty input) returns a descriptive `FdarError` rather than panicking, and fitting is stable (no NaN β(t)) across the three penalty families.

**Plans**: 1 plan

- [x] 66-01-core-peer-tracer-PLAN.md — public `peer()` estimator in new `src/peer.rs`: tracer β(t) recovery (Difference{2}) → all three penalty families (Ridge/Difference/Decree) → partition-aware Decree distinctness → wrong-dim/NaN error surface

### Phase 67: Automatic λ Selection — GCV + REML

**Goal**: A user can have the PEER smoothing parameter λ chosen automatically — by GCV grid search (reusing the `penalized_solve` + GCV pattern) or by REML/mixed-model estimation (reusing `famm`) — selectable via config, matching refund's default, with an explicit λ honored when supplied.
**Depends on**: Phase 66 (λ selection wraps the core `peer()` penalized fit)
**Requirements**: PER-03
**Success Criteria** (what must be TRUE):

  1. With GCV selected, `peer()` runs an automatic λ search over a grid and returns the selected λ (recorded in the result struct) that minimizes the GCV score; two runs on the same data pick the same λ (deterministic).
  2. With REML selected, `peer()` fits λ via mixed-model estimation and returns a sensible positive λ on the same synthetic data, and the REML-selected and GCV-selected fits agree on β(t) within a documented tolerance.
  3. When the user supplies an explicit λ, that value is used verbatim (no search runs) and appears unchanged in the result diagnostics.
  4. On synthetic data with a known signal-to-noise level, both selectors pick a λ in a sensible range (neither degenerate-zero nor over-smoothing to a flat β(t)), recovering the known β(t) within tolerance.

**Plans**: 1 plan

- [ ] 67-01-lambda-selection-tracer-PLAN.md — tracer: evolve `PeerConfig.lambda` to `LambdaChoice` + wire `Fixed(λ)` end-to-end (Phase 66 tests migrated) → GCV grid-search selector → self-contained REML EM selector → full-suite/clippy/fmt phase gate

### Phase 68: Longitudinal PEER, Prediction & Integration

**Goal**: A user can fit longitudinal PEER via a public `lpeer()` estimator that extends PEER to repeated per-subject measurements with subject-level random effects (fitted through `famm::fit_scalar_mixed_model`, REML EM), predict out-of-sample from a fitted PEER/lpeer result, and reach the whole surface from the crate root + prelude — demonstrated by an end-to-end module doctest.
**Depends on**: Phase 67 (lpeer reuses PEER's penalty machinery and the REML λ-selection path; prediction consumes the fitted result)
**Requirements**: PER-04, PER-05
**Success Criteria** (what must be TRUE):

  1. `lpeer()` fits repeated per-subject measurements with subject-level random effects and returns a result struct carrying the (time-varying) coefficient function and variance components; the estimated variance components are non-negative.
  2. On synthetic longitudinal data with a known subject-random-effect structure, `lpeer()` recovers the coefficient function within tolerance and the fitted variance components track the injected between-subject variance.
  3. Calling `predict` on new curves from a fitted `peer`/`lpeer` result yields fitted values that match the training-time fitted values when the training curves are re-passed (self-consistency), and produces finite predictions on genuinely new curves.
  4. The full PEER/lpeer public surface (estimators, result structs, penalty enum, predict) is reachable from the crate root and the prelude, and a module doctest demonstrates the end-to-end fit → coefficient function → predict workflow and passes under `cargo test --doc`.

**Plans**: TBD

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 66. Core PEER Estimator & Penalty Families | 1/1 | Complete    | 2026-09-04 |
| 67. Automatic λ Selection — GCV + REML | 0/1 | Not started | - |
| 68. Longitudinal PEER, Prediction & Integration | 0/TBD | Not started | - |

## Status

Milestones through **v0.35.0 are shipped and archived** under `milestones/`. **v0.36.0 (Phases 66–68)** is the active milestone — promoting GAP-06 (PEER / longitudinal PEER, structured-penalty scalar-on-function regression). 5 requirements (PER-01..PER-05) mapped across 3 phases along a strict dependency chain (core estimator + penalties → λ-selection → longitudinal + prediction/exports). Remaining backlog items (GAP-07 wavelet regression, GAP-08 differentiable core) carry forward, drawn top-first.

Next: `/gsd-plan-phase 66`
