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

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (64.1, 64.2): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v0.30.0 — Performance & Consolidation Pass (Phases 46–51) — SHIPPED 2026-09-01</summary>

First internally-driven milestone (both parity backlogs exhausted): measure-first, behavior-preserving depth work. Phase 46 profiling produced three ranked inventories driving 47–51.

- [x] Phase 46: Whole-Crate Profiling & Measurement (PROF-01/02/03, 5 plans) — ranked hot-path/dedup/API inventories
- [x] Phase 47: Hot-Path & Allocation Performance (PERF-01/02, 4 plans) — face_covariance −80.7% wall, dpca −54% alloc blocks; bit-identical
- [x] Phase 48: Parallelism-Gap Closure (PERF-03, 3 plans) — frechet_anova 9.9×, co_cluster 6.4× thread-scaling; payback guards
- [x] Phase 49: Code Consolidation / Dedup (CONS-01/02, 5 plans) — χ²/gamma → distributions.rs, seed_for_thread, permutation_pvalue, SVD sign-core; −358 LOC; bit-identical
- [x] Phase 50: Additive API-Surface Consolidation (API-01/02/03, 3 plans) — 3 Default impls, fanova_seeded, Dim + 5 dispatchers, 6 #[deprecated]; 28 examples + wasm compile
- [x] Phase 51: Benchmark Coverage & Regression Guards (BENCH-01/02, 4 plans) — 9 new module benches + BENCH-RESULTS.md ledger

Milestone audit: **tech_debt** (13/13 requirements satisfied, 6/6 phases verified passed). Full detail: [milestones/v0.30.0-ROADMAP.md](milestones/v0.30.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.31.0 — Multi-Ecosystem Gap Audit (Phases 52–53) — SHIPPED 2026-09-02</summary>

Next-yardstick audit (both prior parity backlogs exhausted): map fdars against four fresh ecosystems and produce a single ranked, de-duplicated, GSD-ready backlog. **Audit-only** — zero `fdars-core/src/` edits, no crate change, no git tag.

- [x] Phase 52: Ecosystem Surveys (MAT-01/JUL-01/TDY-01/PYX-01, 4 plans) — capability-first surveys of MATLAB FDA, Julia FDA, tidyfun/refund, Python-beyond-scikit-fda → four `survey-*.md` with net-new gap lists (completed 2026-09-02)
- [x] Phase 53: Consolidation & Backlog (RPT-01/02/03, 3 plans) — `GAP-AUDIT-REPORT.md` + ranked `GAP-BACKLOG.md` (7 net-new, value/√effort) + RPT-03 completeness gate PASS (completed 2026-09-02)

Milestone audit PASSED 7/7 requirements. Outcome: 7 ranked net-new gaps (top: GAK, shapelets) + 3 recorded out-of-scope; headline = fdars is exceptionally comprehensive, cross-ecosystem convergence LOW. Deliverables in `.planning/research/GAP-AUDIT-REPORT.md` + `GAP-BACKLOG.md`. Full detail: [milestones/v0.31.0-ROADMAP.md](milestones/v0.31.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.32.0 — Global Alignment Kernel & Kernel Clustering (Phases 54–56) — SHIPPED 2026-09-02</summary>

First implementation milestone after three audit/consolidation cycles: a PSD Global Alignment Kernel for curve sets + the kernel machinery it unlocks. Promoted GAP-01 (top-ranked, score 3.00). Strictly sequential dependency spine; all algorithmic risk front-loaded into Phase 54. Real `fdars-core/src/` changes, additive/non-breaking, no new dependency; crate bumped 0.30.0 → 0.32.0, published on the `v0.32.0` tag.

- [x] Phase 54: GAK Kernel Core (GAK-01/02/03/04) — new `metric/gak.rs`; log-domain PSD Triangular GAK + `gak_gram_matrix` + `sigma_gak` (completed 2026-09-02)
- [x] Phase 55: Gram-Matrix Export (GAK-05/06) — split `gak_gram_train`/`gak_gram_predict` for external precomputed-kernel SVM (completed 2026-09-02)
- [x] Phase 56: Kernel-k-means Clustering (GAK-07/08) — new `kernel_kmeans.rs`; kernel-k-means on curves + out-of-sample `predict` (completed 2026-09-02)

Milestone audit PASSED 8/8 requirements. Full detail: [milestones/v0.32.0-ROADMAP.md](milestones/v0.32.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.33.0 — Shapelet Transform & Classification (Phases 57–60) — SHIPPED 2026-09-02</summary>

Interpretable, discovery-based shapelet classification for curves. Promoted GAP-02 (score 2.89, the only backlog gap corroborated across sktime + pyts + tslearn). New `src/shapelet/` submodule along a strict compile-time dependency chain (distance core → discovery → transform → classifier). Real `fdars-core/src/` changes, additive/non-breaking, no new crate dependency; crate bumped 0.32.0 → 0.33.0, published on the `v0.33.0` tag.

- [x] Phase 57: Shapelet Distance Core (SHP-01/02) — new `src/shapelet/distance.rs`; per-window z-normalization + min sliding-window `sdist` with early-abandon + the `Shapelet` type (completed 2026-09-02)
- [x] Phase 58: Discovery & Ranking (SHP-03/04/05) — candidate generation (exhaustive + contracted/seeded), info-gain / F-statistic quality, top-K + self-similarity pruning → `ShapeletSet` (completed 2026-09-02)
- [x] Phase 59: Shapelet Transform (SHP-06) — fitted `ShapeletSet` → n×K distance-feature matrix (train + out-of-sample), transform consistency (completed 2026-09-02)
- [x] Phase 60: Bundled ShapeletTransformClassifier (SHP-07) — end-to-end `fit` (discover → transform → classify; kNN default, LDA optional) + `predict`; crate-root re-exports (completed 2026-09-02)

Milestone audit PASSED 7/7 requirements. Full detail: [milestones/v0.33.0-ROADMAP.md](milestones/v0.33.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.34.0 — k-Shape Clustering & Shape-Based Distance (Phases 61–63) — SHIPPED 2026-09-02</summary>

Shape-based curve clustering — the SBD (Shape-Based Distance) primitive and the k-Shape algorithm built on it — plus out-of-sample assignment and SBD as a distance backend for existing k-medoids. Promoted GAP-03 (score 2.12, M-effort). Strict SBD → k-Shape → k-medoids dependency chain (non-reorderable). Real `fdars-core/src/` changes, additive/non-breaking, no new crate dependency; crate bumped 0.33.0 → 0.34.0, published on the `v0.34.0` tag.

- [x] Phase 61: SBD Distance Core (KSH-01/02) — new `src/metric/sbd.rs`; FFT normalized-cross-correlation `sbd(x,y) -> (distance, shift)` + public n×n `sbd_distance_matrix` (completed 2026-09-02)
- [x] Phase 62: k-Shape Clustering & Predict (KSH-03/04) — new top-level `src/kshape.rs`; `kshape_fd` (SBD assignment + shape-extraction centroids, n_init restarts, empty-cluster recovery, deterministic seeding) + `KShapeResult::predict` (completed 2026-09-02)
- [x] Phase 63: SBD-based k-medoids & Wrap-up (KSH-05) — `sbd_kmedoids` convenience over the existing `kmedoids_from_distances`; crate-root re-exports + `prelude` + criterion benchmark (completed 2026-09-02)

Milestone audit PASSED 5/5 requirements. Full detail: [milestones/v0.34.0-ROADMAP.md](milestones/v0.34.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.35.0 — Optimal Experimental Design for Sparse FDA (FOptDes) (Phases 64–65) — SHIPPED 2026-09-03</summary>

Optimal sparse-measurement design over an already-fitted PACE model — choose the measurement locations that minimize PACE trajectory-reconstruction error or the posterior variance of predicted FPC scores. Promoted GAP-05 (score 2.12, M-effort), the first milestone drawing from the *design* front of the backlog. One new file `src/optimal_design.rs` (peer of `kshape.rs`/`kernel_kmeans.rs`), additive/non-breaking, no new crate dependency (MSRV 1.81, `linalg` not required); crate bumped 0.34.0 → 0.35.0, published on the `v0.35.0` tag.

- [x] Phase 64: Criterion Machinery Core (FOD-01/02/03) — new `src/optimal_design.rs`; shared `build_sigma_design` + trajectory-reconstruction BLUP-MSE (Simpson-weighted) + A-/D-optimality posterior score-covariance + public `#[must_use] design_criterion` with `DesignCriterion`/`OptimalityKind` enums; 16 known-answer tests (completed 2026-09-03)
- [x] Phase 65: Greedy Selection & Integration (FOD-04/05) — deterministic greedy forward-selection `optimal_design` (parallel-evaluate + sequential smallest-index tie-break) + `OptDesConfig`/`OptDesResult` + full crate-root/prelude re-exports + module doctest + criterion benchmark; 32 module tests (completed 2026-09-03)

Milestone audit PASSED 5/5 requirements (2/2 phases verified passed; integration + E2E doctest verified). Tech debt noted: pre-existing `--features serde` build break in `shapelet/classifier.rs` (Phase 60), unrelated to FOptDes. Full detail: [milestones/v0.35.0-ROADMAP.md](milestones/v0.35.0-ROADMAP.md)

</details>

## Status

All milestones through **v0.35.0 are shipped and archived** under `milestones/`. v0.35.0 promoted GAP-05 (Optimal Experimental Design for Sparse FDA / FOptDes) — the FOptDes public surface (`design_criterion`, `optimal_design`, `OptDesConfig`, `OptDesResult`, `DesignCriterion`, `OptimalityKind`) is live in `fdars-core::optimal_design` and the prelude. The remaining backlog items (GAP-06/07/08 — PEER/lpeer, wavelet regression, differentiable core) carry forward, drawn top-first.

Next: `/gsd-new-milestone`
