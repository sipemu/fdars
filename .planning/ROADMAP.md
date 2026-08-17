# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (shipped 2026-08-12, PR #40) — [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 — Registration Parity & Elastic-FPCA Performance** — Phases 14–15 (shipped 2026-08-12, PR #41) — [archive](milestones/v0.17.0-ROADMAP.md)
- ✅ **v0.18.0 — R-Ecosystem Gap Audit** — Phases 16–19 (shipped 2026-08-15) — [archive](milestones/v0.18.0-ROADMAP.md)
- ✅ **v0.19.0 — Functional Inference Suite** — Phases 20–21 (shipped 2026-08-16) — [archive](milestones/v0.19.0-ROADMAP.md)
- ✅ **v0.20.0 — Table-Stakes Quick Wins** — Phases 22–23 (shipped 2026-08-16) — [archive](milestones/v0.20.0-ROADMAP.md)
- 🚧 **v0.21.0 — Functional Regression Completeness** — Phases 24–25 (in progress)

## Phases

<details>
<summary>✅ v0.14.0 Performance & scikit-fda Gap Audit (Phases 1–9) — SHIPPED 2026-08-09</summary>

Audit-only milestone — every phase produced analysis artifacts, zero `fdars-core/src/` edits. Deliverables: `.planning/research/AUDIT-REPORT.md` (consolidated report) + `.planning/research/BACKLOG.md` (32-item value-ranked backlog).

- [x] Phase 1: Measurement Discipline & Baselines (2/2 plans) — completed 2026-08-07
- [x] Phase 2: Static Hot-Path Analysis (2/2 plans) — completed 2026-08-07
- [x] Phase 3: Elastic Alignment Hot Path (2/2 plans) — completed 2026-08-08
- [x] Phase 4: FPCA/SVD & Allocation Audit (3/3 plans) — completed 2026-08-08
- [x] Phase 5: Parallelism Gap Assessment (3/3 plans) — completed 2026-08-08
- [x] Phase 6: Conditional SVD Library Comparison (1/1 plans) — completed 2026-08-09
- [x] Phase 7: scikit-fda Capability Enumeration (2/2 plans) — completed 2026-08-09
- [x] Phase 8: Capability Parity Matrix & Categorization (3/3 plans) — completed 2026-08-09
- [x] Phase 9: Consolidated Report & Prioritized Backlog (3/3 plans) — completed 2026-08-09

Full phase detail: [milestones/v0.14.0-ROADMAP.md](milestones/v0.14.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.15.0 Top-Backlog Quick Wins (Phases 10–11) — SHIPPED 2026-08-11</summary>

First implementation milestone — the top-4 audit-backlog quick wins delivered as real `fdars-core/src/` code, each with inline tests and numerical verification. Full suite green; milestone audit passed (4/4); shipped via PR #38, `fdars-core` 0.15.0 on crates.io.

- [x] Phase 10: Capability Gaps — Spline Interpolation & Functional Summary Statistics (2/2 plans) — completed 2026-08-10 (FEAT-01, FEAT-02)
- [x] Phase 11: Performance Wins — Parallel CV Folds & faer FPCA SVD (2/2 plans) — completed 2026-08-11 (PERF-01, PERF-02)

Full phase detail: [milestones/v0.15.0-ROADMAP.md](milestones/v0.15.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.16.0 Elastic Feasibility + Parity Quick Wins (Phases 12–13) — SHIPPED 2026-08-12 (PR #40)</summary>

Second implementation milestone — the next tier of the v0.14.0 audit backlog: the P1 elastic-feasibility headline plus three effort-S scikit-fda parity gaps, all additive/non-breaking. Milestone audit passed (4/4 requirements, cross-phase integration clean, 2663 tests green). Released via PR #40 (crate 0.16.0, tag v0.16.0).

- [x] Phase 12: Elastic Feasibility — Banded Alignment Default & `band_frac` (1/1 plans) — completed 2026-08-12 (PERF-03: opt-in `*_with_band` wrappers, large grids feasible)
- [x] Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics (2 plans + 1 gap-closure) — completed 2026-08-12 (FEAT-03 imputation, FEAT-04 `ExtrapolationPolicy` both interp paths, FEAT-05 five scoring metrics)

Full phase detail: [milestones/v0.16.0-ROADMAP.md](milestones/v0.16.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.17.0 Registration Parity & Elastic-FPCA Performance (Phases 14–15) — SHIPPED 2026-08-12 (PR #41)</summary>

Third implementation milestone — the next tier of the v0.14.0 audit backlog: the P1 shift-registration gap + its scikit-fda quality diagnostics, plus a targeted elastic-FPCA parallelization. All additive/non-breaking. Milestone audit passed (3/3 requirements, integration clean; full suite green: 2727 tests `linalg,parallel` / 2718 default). Released via PR #41 (crate 0.17.0, tag v0.17.0).

- [x] Phase 14: Shift Registration (2/2 plans) — completed 2026-08-12 (FEAT-06 `least_squares_shift_registration` + `ShiftRegistrationResult` in new `alignment/shift.rs`; FEAT-07 three registration-quality scores in `alignment/quality.rs`)
- [x] Phase 15: Elastic-FPCA Performance (1/1 plans) — completed 2026-08-12 (PERF-04 parallelize `:701/:720/:764` via `iter_maybe_parallel!` collect-then-assign, N≥50 guard, bit-identical equivalence)

Full phase detail: [milestones/v0.17.0-ROADMAP.md](milestones/v0.17.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.18.0 R-Ecosystem Gap Audit (Phases 16–19) — SHIPPED 2026-08-15</summary>

Audit-only milestone — the R-ecosystem analog of v0.14.0: zero `fdars-core/src/` edits. The R FDA ecosystem (35 packages surveyed) replaced scikit-fda as the yardstick now that the actionable scikit-fda backlog is exhausted. Milestone audit passed (7/7 requirements). Deliverables: `.planning/research/R-AUDIT-REPORT.md` (inventory → parity matrix → strengths → consolidated findings) + `.planning/research/R-BACKLOG.md` (26-item value-ranked, promotion-ready backlog).

- [x] Phase 16: R Ecosystem Inventory (2/2 plans) — completed 2026-08-15 (INV-01, INV-02: 35 pkgs, 275 caps, 248 in-scope)
- [x] Phase 17: Parity Matrix & Categorization (1/1 plans) — completed 2026-08-15 (GAP-01, GAP-02: 250 rows mapped, 162 actionable gaps, 18 table-stakes / 144 differentiator)
- [x] Phase 18: Reverse-Parity Strengths Sweep (1/1 plans) — completed 2026-08-15 (GAP-03: 42 modules walked, 12 R-honest strengths)
- [x] Phase 19: Consolidated Report & Ranked Backlog (1/1 plans) — completed 2026-08-15 (RPT-01, RPT-02: 26 ranked GSD-ready items, completeness gate PASS)

Full phase detail: [milestones/v0.18.0-ROADMAP.md](milestones/v0.18.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.19.0 Functional Inference Suite (Phases 20–21) — SHIPPED 2026-08-16</summary>

First implementation milestone from the v0.18.0 R-ecosystem backlog — the two P1 table-stakes inference items (INF-01, INF-02) closing fdars' dominant table-stakes deficit: R-parity Area 5 (Inference), previously 0/22 present. New `fdars-core/src/inference/` module (8 public entry points), additive/non-breaking, 29 inline tests; full suite green (2039 lib tests), clippy clean. Milestone audit passed (2/2). No new crate dependency (self-contained χ²/F survival functions).

- [x] Phase 20: Two-Sample Functional Tests & `inference/` Module (1/1 plans) — completed 2026-08-16 (INF-01: `t_perm_test`, `f_perm_test`, `two_sample_mean_test`, `mean_scb`, `scb_two_sample_test` + `TestResult`)
- [x] Phase 21: Functional-Linear-Model Inference (1/1 plans) — completed 2026-08-16 (INF-02: `flm_f_test`, `flm_gof_test`, `oneway_anova_vstat` alongside unchanged `fanova`)

Full phase detail: [milestones/v0.19.0-ROADMAP.md](milestones/v0.19.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.20.0 Table-Stakes Quick Wins (Phases 22–23) — SHIPPED 2026-08-16</summary>

Second batch of R-backlog items — the two top-ranked (score 5.00, P1 table-stakes, S-effort) quick wins, each wrapping existing infrastructure. Additive/non-breaking; full suite green (2061 lib tests), clippy clean, no new dependency. Milestone audit passed (2/2). Phases independent (disjoint modules).

- [x] Phase 22: Constant Basis & AIC Smoothing Selection (1/1 plans) — completed 2026-08-16 (T-01: `constant_basis`, `CvCriterion::Aic` + `aic_smoother`, `smooth_basis_aic`)
- [x] Phase 23: Functional Boxplot & Depth Dispatcher (1/1 plans) — completed 2026-08-16 (T-02: `DepthMethod` + `functional_depth` dispatcher, `functional_boxplot` + `FunctionalBoxplotResult`)

Full phase detail: [milestones/v0.20.0-ROADMAP.md](milestones/v0.20.0-ROADMAP.md)

</details>

### 🚧 v0.21.0 Functional Regression Completeness (In Progress)

**Milestone Goal:** Close the two remaining P1 table-stakes functional-regression gaps against the R ecosystem, drawn top-first from `.planning/research/R-BACKLOG.md` (REG-01 rank 6, REG-02 rank 7) — dense concurrent / varying-coefficient regression and exponential-family functional GLMs. Both are additive/non-breaking, `Result`-returning, reuse existing scalar-on-function design machinery, and introduce no new algorithm subsystem or crate dependency. **Zero changes to existing public signatures.** The two phases touch disjoint modules with no cross-phase dependency — they may execute in either order or in parallel.

**Phase numbering continues** from v0.20.0 (ended at Phase 23) → v0.21.0 starts at Phase 24. No reset.

#### Phase 24: Concurrent / Varying-Coefficient Regression
**Goal**: Users can fit a dense functional concurrent (varying-coefficient) regression relating a functional response to one or more functional predictors sampled on the same shared grid, recovering a smooth time-varying coefficient curve β(t).
**Depends on**: Nothing (independent of Phase 25 — disjoint module: new `concurrent_regression.rs` + `smoothing.rs` kernels)
**Requirements**: REG-01
**Success Criteria** (what must be TRUE):
  1. User can call a new public entry point in `fdars-core/src/concurrent_regression.rs` (re-exported at the crate root) with a functional response, one or more functional predictors on a shared grid, and a roughness-penalty parameter, and receive a `Result` carrying `{ beta_curve, fitted, residuals }`.
  2. On synthetic data generated from a known β(t) with low noise, the recovered `beta_curve` reproduces the true coefficient curve within a stated tolerance (regression-recovery inline test).
  3. β(t) is estimated by penalized pointwise / local-linear least squares over the shared dense grid, and increasing the roughness penalty produces a demonstrably smoother `beta_curve` (verifiable via a monotone roughness/curvature check).
  4. `fitted` reconstructs the response from the estimated β(t) and predictors, `residuals == response − fitted` pointwise (consistency test); invalid inputs (mismatched grids/dimensions, empty data) return the appropriate `FdarError` rather than panicking.
  5. All existing regression APIs are unchanged (additive/non-breaking): no existing public signature is modified, and the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.
**Plans**: 1 plan

Plans:
- [x] 24-01: Implement dense concurrent/varying-coefficient regression in `concurrent_regression.rs` (penalized pointwise LS β(t) via `smoothing.rs` kernels, result struct, crate-root re-export, inline `#[cfg(test)]` tests) — completed 2026-08-17 (commit 5480ee25)

#### Phase 25: Functional GLM (Exponential Family)
**Goal**: Users can fit a functional GLM for a scalar response over functional predictors across the four mainstream exponential-family families, generalizing the existing logistic path without breaking it.
**Depends on**: Nothing (independent of Phase 24 — disjoint module: `scalar_on_function/`, reusing the `functional_logistic` IRLS loop + `fdata_to_pc_1d`)
**Requirements**: REG-02
**Success Criteria** (what must be TRUE):
  1. User can call `functional_glm(data, y, family)` with `family: GlmFamily` and receive a `Result`-returning fit (re-exported at the crate root), where `GlmFamily { Binomial, Poisson, Gamma, Gaussian }` each carries its canonical link and variance function.
  2. `functional_glm(data, y, GlmFamily::Binomial)` reproduces the existing `functional_logistic` fit on the same data (coefficients / fitted values agree within tolerance), and the existing `functional_logistic` public signature is retained unchanged (additive/non-breaking).
  3. On synthetic data drawn from a known generative model per non-Gaussian family (e.g. Poisson counts under a log link, Gamma responses), the fitted coefficients/predictions recover the true signal within a stated tolerance (per-family inline tests).
  4. The estimator runs IRLS over FPC/basis scores from `fdata_to_pc_1d` (reusing the `functional_logistic` IRLS loop), and invalid inputs (dimension mismatch, out-of-domain response for a family such as negative Poisson counts) return the appropriate `FdarError` rather than panicking.
  5. The full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green; no new crate dependency is introduced.
**Plans**: 1 plan

Plans:
- [ ] 25-01: Generalize `functional_logistic` into `functional_glm` + `GlmFamily` enum in `scalar_on_function/` (IRLS over FPC scores, four families with canonical link+variance, crate-root re-export, inline `#[cfg(test)]` tests)

## Progress

**Execution Order:**
Phases 24 and 25 are mutually independent (disjoint modules) and may execute in either order or in parallel. Default order 24 → 25 by backlog rank (REG-01 rank 6 before REG-02 rank 7).

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 24. Concurrent / Varying-Coefficient Regression | v0.21.0 | 0/1 | Not started | - |
| 25. Functional GLM (Exponential Family) | v0.21.0 | 0/1 | Not started | - |
