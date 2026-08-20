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
- 🔨 **v0.24.0 — Functional Regression & Clustering Breadth** — Phases 31–33 (in progress)

## Phases

- [x] **Phase 31: Additive Functional Regression & Variable Selection** - Add FAM/GKAM/GSAM additive scalar-on-function models, a group-penalized `variable_selection` helper, a permutation-test wrapper, and a history-index (lagged) estimator in new `scalar_on_function/additive.rs` (REG-04, independent) (completed 2026-08-20)
- [x] **Phase 32: Flexible Mixed-Effects Regression** - Extend `famm.rs` beyond fixed-effect testing to random-effects estimation (denseFLMM/multiFAMM/fastFMM) and wire a flexible-RE function-on-function path into `fof_regression.rs` (REG-05, independent) (completed 2026-08-20)
- [ ] **Phase 33: Model-Based & Density Functional Clustering** - Add funHDDC/funFEM subspace models, DBSCAN over functional distances, a kCFC subspace-embedding loop, and a joint align-and-cluster estimator extending `clustering.rs` + `gmm/` (CLUS-01, independent)

## Phase Details

### Phase 31: Additive Functional Regression & Variable Selection

**Goal**: A user can fit nonparametric additive scalar-on-function regression models that `fdapace`/`fda.usc`/`refund` expose but fdars was missing — a functional additive model (FAM, backfitting over FPC-score components), generalized kernel and spectral additive variants (GKAM/GSAM), a group-penalized scalar-on-function `variable_selection` helper, a permutation-test significance wrapper, and a history-index (lagged-predictor-window) estimator — all in a new `scalar_on_function/additive.rs`, reusing `smoothing.rs` kernels and `fdata_to_pc_1d`, without any existing regression code changing.
**Depends on**: Nothing (independent of Phases 32/33; may run in any order or in parallel)
**Requirements**: REG-04
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points in `fdars-core/src/scalar_on_function/additive.rs` (crate-root re-exported) — a functional additive model (FAM, backfitting over FPC-score components), a generalized kernel additive model (GKAM) and a generalized spectral additive model (GSAM) variant, a group-penalized scalar-on-function `variable_selection` helper, a permutation-test wrapper, and a history-index (lagged-predictor-window) estimator — each fitting/predicting from a column-major `FdMatrix` and returning a structured result.
  2. On synthetic data generated from a known additive signal (e.g. a sum of nonlinear component functions of FPC scores, plus a lagged/history-index effect), the fitted model recovers that signal — fitted values track the truth and residuals shrink relative to a mean-only baseline — within a documented tolerance (inline `#[cfg(test)]` tests).
  3. The `variable_selection` helper identifies the truly-active predictors and drops the inert ones on data with a known active subset, and the permutation-test wrapper returns a small p-value when a real effect is present and a non-significant p-value under the null (inline `#[cfg(test)]` tests, seeded for reproducibility).
  4. The additive family reuses `smoothing.rs` kernels and `fdata_to_pc_1d` rather than adding a new subsystem, adds no new crate dependency, and invalid inputs (empty matrix / mismatched response length / mismatched argvals vs values / degenerate columns / invalid lag window) return `FdarError` rather than panicking.
  5. Existing `scalar_on_function/` public signatures and `fdata_to_pc_1d` keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 2 plans

Plans:

- [x] 31-01-PLAN.md — FAM tracer (new module wired end-to-end) + GKAM + GSAM estimators
- [x] 31-02-PLAN.md — variable_selection (GroupLasso) + permutation_test_fam + history_index

### Phase 32: Flexible Mixed-Effects Regression

**Goal**: A user can estimate flexible functional mixed-effects models that `denseFLMM`/`multifamm`/`fastFMM`/`refund` (pffr) expose but fdars was missing — a dense functional linear mixed model (denseFLMM-style mixed-model equations over FPC scores / basis coefficients), a multivariate functional additive mixed variant (multiFAMM), fast functional mixed-model inference (fastFMM), and a flexible random-effects function-on-function path — by extending `famm.rs` (today only `fmm_test_fixed`) and wiring the flexible-RE path into the already-present `fof_regression.rs`, without any existing mixed-model or FoF code changing.
**Depends on**: Nothing (independent of Phases 31/33; may run in any order or in parallel)
**Requirements**: REG-05
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points that extend `fdars-core/src/famm.rs` (crate-root re-exported) — a dense functional linear mixed model estimating both fixed and random effects (mixed-model equations over FPC scores / basis coefficients), a multivariate functional additive mixed variant, and a fast functional mixed-model inference path — each returning fixed-effect estimates, random-effect / variance-component estimates, and fitted functional curves.
  2. A flexible random-effects function-on-function estimator is wired into `fdars-core/src/fof_regression.rs` (extending only the flexible/RE variant — the base function-on-function capability is already present at parity and its signatures are untouched) and returns a structured result over the functional response.
  3. On synthetic data generated from a known mixed model (fixed effect + grouped random intercepts/slopes with known variance components), the estimators recover the fixed effects and the random-effect / variance-component structure, and fitted curves track the truth, within a documented tolerance (inline `#[cfg(test)]` tests).
  4. The mixed-model family reuses the existing `famm.rs` fixed-effect machinery and FPC-score / basis-coefficient infrastructure rather than adding a new subsystem, adds no new crate dependency, and invalid inputs (empty data / mismatched grouping factor length / singular design / mismatched dimensions) return `FdarError` rather than panicking.
  5. Existing `famm::fmm_test_fixed` and the base `fof_regression.rs` public signatures keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 2 plans

Plans:

- [x] 32-01-PLAN.md — denseFLMM tracer + multiFAMM + fastFMM in `famm.rs` (+ 6 pub(crate) promotions, re-exports)
- [x] 32-02-PLAN.md — flexible-RE function-on-function (`fof_re_regression` + `predict_fof_re`) in `fof_regression.rs` (Wave 2)

### Phase 33: Model-Based & Density Functional Clustering

**Goal**: A user can cluster functional data with the paradigms `funHDDC`/`funFEM`/`fdacluster`/`fdapace`/`fdasrvf` expose but fdars was missing beyond its existing k-means/GMM/hierarchical/k-medoids — a funHDDC-style per-group subspace covariance model (extending `gmm/`), a funFEM discriminative-subspace clustering variant, a DBSCAN density clusterer over functional distances (reusing `distance.rs`), a kCFC subspace-embedding loop, and a joint align-and-cluster estimator (reusing `alignment/`) — as numeric cluster assignments and model outputs only (no rendering), without any existing clustering code changing.
**Depends on**: Nothing (independent of Phases 31/32; may run in any order or in parallel)
**Requirements**: CLUS-01
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points (crate-root re-exported) extending `fdars-core/src/clustering.rs` + `gmm/` — a funHDDC-style per-group subspace covariance clusterer, a funFEM discriminative-subspace clusterer, a DBSCAN density clusterer over functional distances, a kCFC subspace-embedding clusterer, and a joint align-and-cluster estimator — each returning numeric cluster assignments and model outputs (no plotting/rendering).
  2. On synthetic data with a known cluster structure (well-separated functional groups, plus a shape-shifted group for the align-and-cluster path), each clusterer recovers the true grouping up to label permutation within a documented agreement tolerance (e.g. adjusted-Rand / accuracy threshold), and DBSCAN correctly flags injected noise curves as unassigned (inline `#[cfg(test)]` tests).
  3. The DBSCAN clusterer computes neighborhoods from `distance.rs` functional distances and the joint align-and-cluster estimator reuses `alignment/`, so the new paradigms share fdars' existing distance/GMM/alignment infrastructure rather than reimplementing it — adding no new crate dependency.
  4. Invalid inputs (empty matrix / fewer curves than requested clusters / mismatched argvals vs values / invalid DBSCAN eps or min-points / degenerate columns) return `FdarError` rather than panicking, with checks at each entry point.
  5. Existing clustering entry points (`kmeans_fd`, `fuzzy_cmeans_fd`, `gmm_cluster`, hierarchical / k-medoids) and `gmm/` / `distance.rs` / `alignment/` public signatures keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 3 plans

Plans:

- [x] 33-00-PLAN.md — tracer: `adjusted_rand_index` test helper + funHDDC per-group subspace EM in `gmm/subspace.rs` + crate-root re-exports (Wave 1)
- [x] 33-02-PLAN.md — DBSCAN (density, None-is-noise) + kCFC (per-cluster FPCA loop) in new `clustering_advanced.rs` + re-exports (Wave 2)
- [x] 33-03-PLAN.md — funFEM (Fisher-EM discriminative subspace) + joint align-and-cluster (elastic k-means) appended to `clustering_advanced.rs` + re-exports (Wave 3)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 31. Additive Functional Regression & Variable Selection | 2/2 | Complete    | 2026-08-20 |
| 32. Flexible Mixed-Effects Regression | 2/2 | Complete    | 2026-08-20 |
| 33. Model-Based & Density Functional Clustering | 0/3 | Planned | - |

**Execution order:** All three phases are **independent** — REG-04 (Phase 31), REG-05 (Phase 32), and CLUS-01 (Phase 33) have no cross-phase hard dependency (unlike v0.23.0's DEPTH→OUT chain), so they may be planned and executed in any order or in parallel. Each extends a disjoint area of the codebase (`scalar_on_function/` / `famm.rs`+`fof_regression.rs` / `clustering.rs`+`gmm/`). The next three top-ranked P2 differentiators in `R-BACKLOG.md` (REG-04 rank 12, REG-05 rank 13, CLUS-01 rank 15 — all score 1.73, M-effort). Additive/non-breaking, reuse-first, **no new crate dependency**; numeric outputs only (plotting out of scope).

<details>
<summary>✅ v0.23.0 Depth, Outliers & Interval Inference (Phases 28–30) — SHIPPED 2026-08-20</summary>

The top three P2 differentiator gaps from the R-ecosystem backlog (score 2.31 each), all additive/non-breaking to `fdars-core` with zero changes to existing public signatures and no new crate dependency. Milestone audit passed (3/3). DEPTH→OUT chain (tvdmss reuses DEPTH-01's TVD+MSSI); INF-03 independent.

- [x] Phase 28: Depth-Measure Long Tail (3/3 plans) — completed 2026-08-20 (DEPTH-01: 9 batch depth measures HRD/MHRD, HI/MHI/EI, extremal, ERL, L∞, TVD+MSSI in `depth/`, each registered in `DepthMethod`)
- [x] Phase 29: Outlier-Detector Suite (2/2 plans) — completed 2026-08-20 (OUT-01: `tvdmss`, `muod`, `sequential_transform_outliers`, `depthgram` in `outliers.rs`)
- [x] Phase 30: Interval Testing Procedure Family (2/2 plans) — completed 2026-08-20 (INF-03: `itp_one_pop`, `itp_two_pop`, `itp_flm` in new `inference/itp.rs`)

Full phase detail: [milestones/v0.23.0-ROADMAP.md](milestones/v0.23.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.22.0 PACE Sparse FPCA & Elastic Multinomial (Phases 26–27) — SHIPPED 2026-08-19</summary>

Final P1 table-stakes item (FPCA-01) + elastic-family completion (REG-03), each reuse-first (orchestrate/extend existing code). Additive/non-breaking, no new dependency; full suite 2107 lib tests green, clippy `--all-targets` clean, serde-feature build verified. Milestone audit passed (2/2). Disjoint independent modules. Code review caught + fixed real bugs both phases (Phase 26: NaN-mean guard + BLUP/band ridge/error consistency + n_i≥2; Phase 27: serde-feature compile break + normalization Inf/NaN guard). After this milestone the P1 table-stakes tier is exhausted.

- [x] Phase 26: PACE Sparse FPCA (1/1 plans) — completed 2026-08-19 (FPCA-01: `pace_fpca` + `PaceFpcaConfig` + `PaceFpcaResult` in new `pace_fpca.rs`)
- [x] Phase 27: Elastic Multinomial Regression (1/1 plans) — completed 2026-08-19 (REG-03: `elastic_multinomial` + `predict_elastic_multinomial` + `ElasticMultinomialResult` in `elastic_regression/logistic.rs`)

Full phase detail: [milestones/v0.22.0-ROADMAP.md](milestones/v0.22.0-ROADMAP.md)

</details>
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

Third batch of R-backlog items — the two remaining P1 table-stakes functional-regression gaps (REG-01 rank 6, REG-02 rank 7), each reusing existing scalar-on-function design machinery. Additive/non-breaking, `Result`-returning, **zero changes to existing public signatures**, no new crate dependency. Full suite green (2081 lib tests + doctests), clippy `--all-targets` clean. Milestone audit passed (2/2). Both phases independent (disjoint modules).

- [x] Phase 24: Concurrent / Varying-Coefficient Regression (1/1 plans) — completed 2026-08-17 (REG-01: `concurrent_regression` + `ConcurrentRegrResult` in new `concurrent_regression.rs`)
- [x] Phase 25: Functional GLM (Exponential Family) (1/1 plans) — completed 2026-08-17 (REG-02: `functional_glm` + `GlmFamily{Binomial,Poisson,Gamma,Gaussian}` + `FunctionalGlmResult` in new `scalar_on_function/glm.rs`)

Full phase detail: [milestones/v0.21.0-ROADMAP.md](milestones/v0.21.0-ROADMAP.md)

</details>
