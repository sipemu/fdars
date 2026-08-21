# Milestones

## v0.25.0 Serial Dependence, Representation & Density Breadth (Shipped: 2026-08-21)

**Phases completed:** 3 phases, 10 plans, 11 tasks

**Key accomplishments:**

- `BasisSystem` struct + `monomial_basis` analytic-Gram factory end-to-end, with `pub(crate)` numeric-Gram helpers promoted for downstream reuse — tracer slice for REP-01 basis-family architecture.
- Three remaining basis factories (exponential, power, polygonal) completing the four-factory REP-01 family — each returning `BasisSystem` with eval matrix and penalty, all crate-root re-exported, clippy gate clean.
- DENS-01 · **Wave:** 1 · **Status:** complete
- DENS-01 · **Wave:** 2 · **Status:** complete
- DENS-01 · **Wave:** 3 · **Status:** complete

---

## v0.24.0 Functional Regression & Clustering Breadth (Shipped: 2026-08-20)

**Phases completed:** 3 phases, 7 plans, 9 tasks

**Key accomplishments:**

- funHDDC AkBk per-group subspace EM (FPCA init + nalgebra SVD M-step + BIC/ICL) plus `adjusted_rand_index` test helper in `gmm/subspace.rs`, wired to crate root.
- funFEM (Fisher-EM discriminative-subspace GMM) and elastic k-means (Karcher-mean templates + amplitude_distance reassignment) appended to clustering_advanced.rs, completing the five-clusterer CLUS-01 set and re-exported at crate root

---

## v0.23.0 Depth, Outliers & Interval Inference (Shipped: 2026-08-20)

**Phases completed:** 3 phases (28–30), 7 plans

**Milestone audit:** PASSED — 3/3 requirements satisfied, cross-phase integration verified (`.planning/milestones/v0.23.0-MILESTONE-AUDIT.md`)

**Delivered:** The top three P2 differentiator gaps from the R-ecosystem backlog (score 2.31 each), all additive/non-breaking to `fdars-core` with zero changes to existing public signatures and no new crate dependency.

**Key accomplishments:**

- **DEPTH-01 — Depth-Measure Long Tail (Phase 28):** 9 canonical batch depth measures added to `depth/` — hypograph/modified-hypograph/epigraph indices (HI/MHI/EI), half-region & modified half-region depth (HRD/MHRD), extremal depth, extreme-rank-length depth (ERL), L∞ depth, and total-variation depth with MSSI — each a `Result`-returning fn registered in the `DepthMethod` dispatcher. `TvdMssResult { tvd, mss }` pinned as a forward contract.
- **OUT-01 — Outlier-Detector Suite (Phase 29):** `tvdmss` (two-stage TVD+MSSI, consuming DEPTH-01's `total_variation_depth_1d`), `muod` (Fast-MUOD regression-vs-mean indices), `sequential_transform_outliers` (T0/T1/T2/D1 cumulative + functional boxplot), and the `depthgram` statistic — numeric outputs reusing the MS-plot / outliergram machinery + a shared `iqr_fence` helper.
- **INF-03 — Interval Testing Procedure Family (Phase 30):** new `inference/itp.rs` with `itp_one_pop`, `itp_two_pop`, and `itp_flm` — B-spline/Fourier basis-wise permutation tests with the Pini & Vantini interval-wise closure adjustment (`pval_correct`) for domain-selective adjusted p-values, reusing the INF-01 permutation pattern + `basis/` projection.

**Tech debt (non-blocking):** VALIDATION.md files left `status: draft` (Nyquist NOT-VALIDATED — run `/gsd-validate-phase 28|29|30` to reconcile); intentional R-baseline divergences documented in rustdoc (Fast-MUOD, univariate depthgram, response-permutation FLM, symmetric extremal/ERL). Crate release (version bump 0.22.0→0.23.0 + tag + crates.io) is a pending operator ship-time step — NOT performed by this run (Cargo.toml still 0.22.0).

## v0.22.0 PACE Sparse FPCA & Elastic Multinomial (Shipped: 2026-08-19)

**Phases completed:** 2 phases, 2 plans, 0 tasks

**Key accomplishments:**

- Yao-Muller-Wang (2005) PACE FPCA estimator for sparse irregular functional data: six-step pipeline (mean→covariance surface→eigendecomposition→BLUP scores→fitted trajectories→confidence bands), 1243-line module, 13 tests, zero new dependencies.
- 1. [Rule 1 - Bug] Clippy: redundant pattern matching in reject tests

---

## v0.21.0 Functional Regression Completeness (Shipped: 2026-08-17)

**Phases completed:** 2 phases, 2 plans, 4 tasks

**Key accomplishments:**

- Dense functional concurrent regression via pointwise OLS + local-linear smoothing: new `concurrent_regression` entry point with `ConcurrentRegrResult`, zero new dependencies, all SC1–SC5 verified by inline tests
- `link_deriv(mu) = -1/mu²` (negative). Working response uses

---

## v0.20.0 Table-Stakes Quick Wins (Shipped: 2026-08-16)

**Phases completed:** 2 phases, 2 plans, 3 tasks

**Key accomplishments:**

- Additive R-parity table-stakes: `constant_basis` intercept column, `CvCriterion::Aic` + `aic_smoother` kernel-bandwidth selection, and `smooth_basis_aic` λ selector — all reusing existing GCV hat-matrix-trace infrastructure with GCV/CV paths byte-for-byte unchanged.
- None functionally.

---

## v0.19.0 Functional Inference Suite (Shipped: 2026-08-16)

**Phases completed:** 2 phases, 2 plans, 0 tasks

**Key accomplishments:**

- 1. [Rule 3 - Blocking] `f_perm_test` merged forward from Task 2 into Task 1's commit
- F-form residual lack-of-fit (Ramsey-RESET style).

---

## v0.18.0 R-Ecosystem Gap Audit (Shipped: 2026-08-15)

**Phases completed:** 4 phases, 5 plans, 0 tasks

**Key accomplishments:**

- `.planning/research/R-AUDIT-REPORT.md` §Phase 16 — R Ecosystem Inventory (INV-01 + INV-02), consolidated from the completed web-sourced survey `16-RESEARCH.md`.
- the `§Design-Goal Filter` section of `.planning/research/R-AUDIT-REPORT.md` (INV-02).
- `.planning/research/R-AUDIT-REPORT.md` §Phase 17 — Parity Matrix & Categorization (GAP-01 + GAP-02), +430 lines. Audit-only; zero `fdars-core/src/` edits.
- `.planning/research/R-AUDIT-REPORT.md` §Phase 18 — Reverse-Parity Strengths Sweep (GAP-03). Audit-only; zero `fdars-core/src/` edits.
- RPT-01 (consolidated report) + RPT-02 (ranked backlog) — the milestone's final deliverables. Audit-only; zero `fdars-core/src/` edits. Committed `c857532f`.

---

## v0.17.0 Registration Parity & Elastic-FPCA Performance (Shipped: 2026-08-12)

**Phases completed:** 2 phases, 3 plans, 3 tasks

**Key accomplishments:**

- New file: `fdars-core/src/alignment/shift.rs`
- Three standalone-energy registration-quality scorers added to alignment/quality.rs (Result-returning, Simpson-weighted), with all five plan-14 items re-exported at the crate root.

---

## v0.16.0 Elastic Feasibility + Parity Quick Wins (Shipped: 2026-08-12)

**Phases completed:** 2 phases, 3 plans, 0 tasks

**Key accomplishments:**

- API surfacing only — no new algorithm.
- 1. [Rule 1 - Bug] Removed `#[must_use]` from `fdata_interpolate_with_policy`
- Five functional scoring metrics (MAE/MSE/MAPE/MSLE/explained-variance) integrated over argvals via Simpson's rule with per-curve averaging, domain validation, and crate-root re-export.
- Fix four code-review findings (CR-01 Periodic NaN, CR-02 EV logic error, WR-01 misleading m=0 error, IN-01 redundant cfg-test) and add `spline_interpolate_with_policy` to close the VERIFICATION gap against ROADMAP SC#2.

---

## v0.15.0 Top-Backlog Quick Wins (Shipped: 2026-08-11)

**Phases completed:** 2 phases, 4 plans, 0 tasks

**Key accomplishments:**

- Adds `spline_interpolate` — order-k B-spline fit-then-evaluate interpolation using the existing `basis/bspline` system, resolving FEAT-01 (REPR-02) with full input validation and 5 inline tests covering exact reproduction and off-grid accuracy.
- Adds five public functional descriptive-statistics functions to `fdata.rs` — Bessel-corrected pointwise variance/std/covariance and FM-depth-based median/trim_mean — closing FEAT-02 (EXPL-02 gap vs scikit-fda).
- Task 1 (tracer): Parallelize the fclassif_cv fold loop
- `fdata_to_pc_1d` now decomposes its weighted matrix with faer `Svd::new_thin` on a zero-copy `MatRef` view under the `linalg` feature — eliminating the dense `to_dmatrix()` copy — while a shared `fix_svd_signs` helper reconciles singular-vector sign conventions so the faer and nalgebra paths produce equivalent `FpcaResult`s within `1e-8·σ₁`.

---

## v0.14.0 Performance & scikit-fda Gap Audit (Shipped: 2026-08-09)

**Phases completed:** 9 phases, 21 plans, 25 tasks
**Milestone audit:** PASSED — 13/13 requirements satisfied, cross-phase integration sound (`.planning/milestones/v0.14.0-MILESTONE-AUDIT.md`)

**Delivered:** An evidence-backed audit of fdars' performance and scikit-fda functionality gaps, consolidated into `.planning/research/AUDIT-REPORT.md` and a value-ranked, promotion-ready `.planning/research/BACKLOG.md`. Audit-only — zero `fdars-core/src/` edits across all 9 phases.

**Key accomplishments:**

- **Measurement discipline (Phases 1–2):** Built a criterion audit-bench harness across the 4-combo feature matrix (`""`/`parallel`/`linalg`/`linalg,parallel`), recorded 12 release baselines over an N×M workload matrix, wrote the methodology + infra-vs-code failure-triage rule, and produced a zero-cost static hot-path map (complexity in N/M, 8 SVD-copy + 14 basis allocation sites, parallelism gaps).
- **Elastic alignment is the top bottleneck (Phase 3):** Full criterion grid confirmed the O(N²·M²) cost — infeasible at N=500,M=200 on the default path — with a measured 4–6× banded-vs-unbanded penalty; root-caused `karcher_mean()` defaulting to `band = None`.
- **FPCA/SVD split (Phase 4):** dhat allocation audit proved the `FdMatrix→DMatrix` SVD-copy is only ~0.14–0.17% of wall-clock; SVD compute dominates (~99.8%), triggering the Phase-6 GO.
- **Parallelism + SVD library (Phases 5–6):** rayon thread-scaling (~4.73× at 8 threads) with 5 safe-to-parallelize loops identified; faer `thin_svd` measured **1.8–4.1× faster** than nalgebra with zero-copy conversion (P6-1).
- **scikit-fda parity (Phases 7–8):** Versioned capability inventory (skfda 0.10.1, 161 rows) → 141-row parity matrix (59 present / 19 partial / 63 absent) → **82 actionable in-scope gaps** (36 table-stakes, 46 differentiator) + a 30-item reverse-parity strengths sweep.
- **Consolidation (Phase 9):** Final report (5 performance findings, 82 gaps, 30 strengths) + a **32-item value-ranked backlog** (`score = value/√effort`, 34 seven-field promotion-ready blocks), all three completeness assertions passed.

---
