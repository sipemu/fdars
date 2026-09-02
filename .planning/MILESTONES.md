# Milestones

## v0.33.0 Shapelet Transform & Classification (Shipped: 2026-09-02)

**Phases completed:** 4 phases (57–60), 4 plans. Milestone audit PASSED 7/7 requirements. Promoted GAP-02 (score 2.89, the only backlog gap corroborated across sktime + pyts + tslearn) from the v0.31.0 `GAP-BACKLOG.md`. Discovery-based shapelet transform (Ye & Keogh / Hills–Lines / sktime STC) — learning-shapelets deferred. New `src/shapelet/` submodule (strict compile-time dependency chain 57→58→59→60).

**Key accomplishments:**

- **Phase 57 — Shapelet Distance Core (SHP-01/02):** `src/shapelet/distance.rs` — per-window z-normalization (population std, constant-window guard, never NaN), `shapelet_distance` = min over sliding windows of z-normalized Euclidean with explicit `best_so_far` early-abandon; scale/offset-invariant to 1e-10; the `Shapelet` type (z-normed values + provenance).
- **Phase 58 — Discovery & Ranking (SHP-03/04/05):** `src/shapelet/discovery.rs` — candidate generation (exhaustive or seeded `max_candidates`-contracted), quality via `QualityMeasure` (information gain on the optimal distance-split threshold, default; or F-statistic), top-K selection with self-similarity pruning → `ShapeletSet`; byte-identical across runs (seed + `total_cmp` tie-break), sequential==parallel. Reference: sktime `RandomShapeletTransform`, pyts.
- **Phase 59 — Shapelet Transform (SHP-06):** `src/shapelet/transform.rs` — `shapelet_transform` (n×K distance features) + `shapelet_transform_fit` (discover+transform) + `ShapeletTransformFit::transform` (out-of-sample), transform consistency within 1e-12.
- **Phase 60 — Bundled Classifier (SHP-07):** `src/shapelet/classifier.rs` — end-to-end `shapelet_classifier_fit` (discover → transform → classify; kNN default, LDA optional via `ShapeletClassifier`) + `ShapeletClassifierFit::predict`, reusing the existing `classification/` module; finalized crate-root re-exports; criterion `shapelet` bench. Reference: sktime `ShapeletTransformClassifier`.

**Verification:** additive/non-breaking, no new crate dependency. Whole-crate gates green — `cargo fmt --check` clean, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean, 2638 lib tests + 192 doctests pass (26 new shapelet tests). Crate bumped 0.32.0 → 0.33.0 + CHANGELOG. Phase 60 was authored by its execution agent then finished inline by the orchestrator after an account session limit killed the agent pre-commit (no code rewrite). **Deferred (operator ship-time step):** `git tag v0.33.0` + push → crates.io publish via `release.yml`.

---

## v0.32.0 Global Alignment Kernel & Kernel Clustering (Shipped: 2026-09-02)

**Phases completed:** 3 phases (54–56), 3 plans. Milestone audit PASSED 8/8 requirements. First implementation milestone after three audit/consolidation cycles; promoted GAP-01 (top-ranked, score 3.00) from the v0.31.0 `GAP-BACKLOG.md`.

**Key accomplishments:**

- **Phase 54 — GAK Kernel Core (GAK-01/02/03/04):** new `metric/gak.rs` — Cuturi Triangular Global Alignment Kernel via a log-domain (log-sum-exp) forward DP atop the existing soft-DTW lattice; triangular normalization → PSD similarity in `[0,1]` with unit diagonal; `gak_gram_matrix` (symmetric-by-assignment, PSD, parallel); `sigma_gak` median-distance bandwidth heuristic. Reference: tslearn@0.9.0 `gak`.
- **Phase 55 — Gram-Matrix Export (GAK-05/06):** split `gak_gram_train` (n×n, carries training self-kernels + σ) / `gak_gram_predict` (n_test×n_train, cross-normalized against stored training diagonals) for external `SVC(kernel='precomputed')` handoff — the split API closes the silent self-kernel-normalization bug. No native SVM (deferred SVM-01).
- **Phase 56 — Kernel-k-means (GAK-07/08):** new `kernel_kmeans.rs` — kernel-trick clustering on the GAK Gram (no explicit centroid), `n_init` random-partition restarts, empty-cluster recovery, deterministic seeding, and out-of-sample `predict` reusing the fitted σ/normalization. Reference: tslearn@0.9.0 `KernelKMeans`.

**Verification:** additive/non-breaking, no new crate dependency. Whole-crate gates green — `cargo fmt --check` clean, `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean, 2612 lib tests + 186 doctests pass (25 new tests: 17 GAK + 8 kernel-k-means). Crate bumped 0.30.0 → 0.32.0 + CHANGELOG. **Deferred (operator ship-time step):** `git tag v0.32.0` + push → crates.io publish via `release.yml`.

---

## v0.31.0 Multi-Ecosystem Gap Audit (Shipped: 2026-09-02)

**Phases completed:** 2 phases, 7 plans, 0 tasks

**Key accomplishments:**

- (none recorded)

---

## v0.30.0 Performance & Consolidation Pass (Shipped: 2026-09-01)

**Phases completed:** 6 phases, 23 plans, 12 tasks

**Key accomplishments:**

- End-to-end throwaway-probe pipeline proven on fpca_variants::fsvd — 4 criterion cells + a dhat allocation probe feed the first grounded row of PROF-01, with baseline suite green and zero src edits.
- All 9 reuse-first subsystems profiled — face_covariance (984ms) and fem_smooth (452ms) top the compute-bound ranking, fts::dpca (42MB churn) tops allocations — then every throwaway probe removed and the full suite proven green.
- Ranked, anchored duplication inventory: χ²/F survival kernels (2 impls) is the top dedup target; permutation loops, seeded-RNG, and SVD sign-fix follow; simpsons/Cholesky/FPCA-scoring confirmed already consolidated.
- Ranked API-inconsistency inventory: 4 configs missing Default and a non-seedable fanova are the top additive-safe targets; field renames and bulk _1d/_2d unification classified breaking and deferred.
- PROF-00 index ties the three ranked inventories to their consumer phases; crate-wide zero-behavior-change gate passes (suite green, clippy --all-targets clean) and 46-VALIDATION.md is signed off.
- fts::dpca allocation cut 54% (17,739→8,139 blocks) via index-sort eigenvector materialization, behavior-preserving (golden 1e-12), with the permanent proof pipeline (bench + golden + dhat + ledger) established end-to-end.
- fsvd/ssvd/functional_acf now build their eigen matrices via DMatrix::from_fn (no Vec staging + no m×m copy); functional_acf also precomputes sqrt(w). Byte-equivalent, proven by pre-edit golden captures at rel 1e-12.
- cov_irreg precomputes per-observation Gaussian kernel-weight tables (w_s/w_t) once instead of recomputing them per (s,t) grid cell — ~98% fewer exp() calls, cutting face_covariance wall-time 80.7% (983.8→189.8ms) with byte-equivalent output.
- fem_smooth builds phi_t_phi and a_mat in a single assembly pass (drops the phi_t_phi.clone() N×N copy), byte-equivalent; the O(N³) Cholesky/GCV bottleneck is documented+deferred; PERF-RESULTS.md consolidated and Phase 47 validation signed off.

---

## v0.29.0 Boosting/Bayesian, FEM/PDE & Co-Clustering (Shipped: 2026-08-30)

**Phases completed:** 3 phases, 11 plans, 3 tasks

**Key accomplishments:**

- Component-wise B-spline-boosted FOSR (REG-06-01) with penalized Cholesky base-learners, plus full module scaffold (all 5 config structs, all 5 result structs, 4 compiling skeletons) registered in lib.rs and prelude.rs.
- Boosted FoFR via FPC-score signal compression (bfpc): per-predictor FPCA design, amortised Cholesky base-learner solve, and rotation-matrix coefficient-surface reconstruction.
- Log-domain positive smoother wrapping existing `smooth_basis` on `ln(data)` with exp-reconstruction for a strictly-positive guaranteed fit.
- Ramsay integral-of-exp monotone smoother with Gauss-Newton + cumulative-trapezoid integration, direction auto-detect, structural monotonicity guarantee.

---

## v0.28.0 Spectral Functional Time Series & Object-Data Fréchet Regression (Shipped: 2026-08-23)

**Phases completed:** 2 phases, 5 plans, 0 tasks

**Key accomplishments:**

- (none recorded)

---

## v0.27.0 Functional Time Series & Fréchet Regression (Shipped: 2026-08-22)

**Phases completed:** 2 phases, 6 plans, 15 tasks

**Key accomplishments:**

- A time-ordered curve series can now be decomposed into an FPCA-based `ftsm` model and forecast one (or more) steps ahead via per-component Yule-Walker AR score models, reconstructed back into curves — the end-to-end tracer through module wiring, FPCA delegation, AR estimation, and reconstruction.
- Forecasts now extend to arbitrary horizons via iterative AR plug-in (`ftsm_forecast_multistep`), and an existing fit can be updated in place as new curves arrive (`ftsm_update`) by projecting onto frozen FPC loadings and re-fitting the score AR models — no FPCA recomputation.
- `fplsr` adds a PLS-score alternative to the FPC-score AR path: a lag-1 design (regress next curve on current curve) solved as one scalar PLS regression per evaluation point, reusing the shipped `fregre_pls` machinery, producing a one-step forecast curve plus in-sample fitted curves.
- The `frechet` module now provides a generic `MetricSpace` abstraction with a 1D-Wasserstein density backend, the public `wasserstein2_distance`, and the sample `frechet_mean`/`frechet_variance` — the end-to-end tracer proving trait → density backend → DENS-01 reuse → generic statistics, plus the signed-weight quantile-average helper Wave 2 regression depends on.
- `frechet_global_reg` (Petersen–Müller global linear weights) and `frechet_local_reg` (local-linear Gaussian-kernel weights) predict conditional density responses at new Euclidean predictor values, both routing signed weights through the sort-based `signed_quantile_average` — never `wasserstein_barycenter`.
- `frechet_anova` completes FRE-01: a Dubey–Müller `Tₙ` group-difference test over the Wasserstein density space with a primary seeded-permutation p-value and a secondary asymptotic χ²(k−1) p-value, reusing the Wave-1 Fréchet mean/variance machinery and the in-crate chi-square survival function — no new dependency.

---

## v0.26.0 FPCA Breadth & Sparse Covariance (Shipped: 2026-08-21)

**Phases completed:** 2 phases, 4 plans, 11 tasks

**Key accomplishments:**

- Established the additive `fpca_variants.rs` module and landed the two simplest FPCA variants end-to-end (cross-covariance surface + derivative FPCA), plus the `FsvdResult` scaffold for Plan 02.
- Completed FPCA-02 with the three normalization-sensitive variants — the Dubin–Müller dynamical correlation, the functional cross-SVD, and the sandwich-smoother FPCA path — all crate-root reachable.
- Landed the FACE fast-sandwich sparse-covariance estimator as the phase tracer — a new `irreg_fdata/face.rs` reusing `cov_irreg` + the Phase-37 sandwich smoother, producing a symmetric PSD covariance surface.
- Completed SPARSE-01 with the multivariate FACE block covariance (`mface_covariance` + `MfaceCovResult`) and the sparse trajectory-band entry point (`face_trajectory`, a thin PACE delegation) — all four FACE symbols crate-root reachable.

---

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
