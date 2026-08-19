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
- 🔨 **v0.22.0 — PACE Sparse FPCA & Elastic Multinomial** — Phases 26–27 (in progress)

## Phases

- [x] **Phase 26: PACE Sparse FPCA** - Unified PACE estimator for sparse/irregular curves — smoothed mean + covariance-surface eigendecomposition + conditional-expectation (BLUP) scores + fitted trajectories with bands (FPCA-01) (completed 2026-08-19)
- [x] **Phase 27: Elastic Multinomial Regression** - Multi-class (K ≥ 2) elastic logistic over SRSF/SRVF space + `predict_elastic_multinomial`, completing fdars' elastic-regression family (REG-03) (completed 2026-08-19)

## Phase Details

### Phase 26: PACE Sparse FPCA

**Goal**: A user with sparse / irregularly-sampled functional data can fit a complete PACE FPCA in a single call — recovering a smoothed mean, eigenfunctions/eigenvalues, per-curve conditional-expectation (BLUP/PACE) FPC scores, and fitted continuous trajectories with pointwise confidence bands — via a new crate-root entry point, without any of that user's existing FPCA code changing.
**Depends on**: Nothing (independent of Phase 27; disjoint modules)
**Requirements**: FPCA-01
**Success Criteria** (what must be TRUE):

  1. User can call a new public `pace_fpca`-style entry point in `fdars-core/src/pace_fpca.rs` (re-exported at the crate root) on sparse/irregular input and receive a `Result` carrying the smoothed mean, eigenvalues + eigenfunctions, per-curve conditional-expectation FPC scores, and fitted trajectories with pointwise confidence bands.
  2. On synthetic sparse data drawn from a known generative model (known mean + eigenfunctions + score distribution + sampling density), the recovered eigenstructure and per-curve BLUP scores/trajectories match the ground truth within a documented tolerance (inline `#[cfg(test)]` recovery test).
  3. The estimator is built by orchestrating existing pieces — `irreg_fdata` (+ `cov_irreg`) covariance surface, `spm::partial::conditional_expectation`, and `regression::fdata_to_pc_1d` eigendecomposition — adding no new crate dependency.
  4. Invalid inputs (empty / degenerate / mismatched argvals vs values / too-few observations) return `FdarError` rather than panicking, with dimension/parameter checks at the entry point.
  5. Existing FPCA APIs (`fdata_to_pc_1d`, `FpcaResult`, `irreg_fdata`, `spm::partial`) keep their current public signatures unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 1 plan

- [x] 26-01-PLAN.md — New `pace_fpca.rs`: PACE sparse-FPCA estimator (smoothed mean + covariance-surface eigendecomposition + per-curve BLUP scores + fitted trajectories with prediction-variance bands), `Result`-returning, crate-root re-exported, reuse-only (no new dependency)

### Phase 27: Elastic Multinomial Regression

**Goal**: A user can fit an elastic multinomial (multi-class, K ≥ 2) logistic regression over SRSF/SRVF space and predict class probabilities / labels for new curves via a new crate-root entry point — completing fdars' elastic-regression family — while the existing binary `elastic_logistic` continues to work exactly as before.
**Depends on**: Nothing (independent of Phase 26; disjoint modules)
**Requirements**: REG-03
**Success Criteria** (what must be TRUE):

  1. User can call a new multinomial elastic-logistic entry point in `fdars-core/src/elastic_regression/logistic.rs` (re-exported at the crate root) with K ≥ 2 class labels and receive a `Result`-wrapped fitted model over SRSF/SRVF space (one-vs-rest or softmax).
  2. User can call a companion `predict_elastic_multinomial` on new curves and receive class probabilities / predicted labels.
  3. On synthetic data with well-separated per-class shape templates, the fitted model recovers the correct labels within a documented accuracy threshold (inline `#[cfg(test)]` classification test), and the K = 2 multinomial path agrees with the existing binary `elastic_logistic` within tolerance.
  4. The extension reuses the existing SRVF representation + warping machinery, adds no new crate dependency, and invalid inputs (fewer than 2 classes / label-curve count mismatch / empty input) return `FdarError` rather than panicking.
  5. The existing binary `elastic_logistic` public signature is retained unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 1 plan

Plans:

- [x] 27-01-PLAN.md — OvR elastic multinomial (elastic_multinomial + predict_elastic_multinomial + ElasticMultinomialResult), reusing binary elastic_logistic K times; re-exports + input guards + tests

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 26. PACE Sparse FPCA | 1/1 | Complete    | 2026-08-19 |
| 27. Elastic Multinomial Regression | 1/1 | Complete    | 2026-08-19 |

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

<details>
<summary>✅ v0.21.0 Functional Regression Completeness (Phases 24–25) — SHIPPED 2026-08-17</summary>

Third batch of R-backlog items — the two remaining P1 table-stakes functional-regression gaps (REG-01 rank 6, REG-02 rank 7), each reusing existing scalar-on-function design machinery. Additive/non-breaking, `Result`-returning, **zero changes to existing public signatures**, no new crate dependency. Full suite green (2081 lib tests + doctests), clippy `--all-targets` clean. Milestone audit passed (2/2). Both phases independent (disjoint modules). Code review caught + fixed real bugs in both phases (Phase 24: NaN/Inf-bandwidth + `n≤p` guards; Phase 25: Gamma IRLS-weight inversion, Poisson factorial overflow, predict/covariate dimension guards). Crate release (version bump + PR + tag) deferred to an operator-driven ship-time step.

- [x] Phase 24: Concurrent / Varying-Coefficient Regression (1/1 plans) — completed 2026-08-17 (REG-01: `concurrent_regression` + `ConcurrentRegrResult` in new `concurrent_regression.rs`, pointwise-OLS-then-kernel-smooth β(t) reusing `smoothing.rs`, commit 5480ee25)
- [x] Phase 25: Functional GLM (Exponential Family) (1/1 plans) — completed 2026-08-17 (REG-02: `functional_glm` + `GlmFamily{Binomial,Poisson,Gamma,Gaussian}` + `FunctionalGlmResult` in new `scalar_on_function/glm.rs`, IRLS over FPC scores generalizing `functional_logistic`, commit cb839d52)

Full phase detail: [milestones/v0.21.0-ROADMAP.md](milestones/v0.21.0-ROADMAP.md)

</details>
