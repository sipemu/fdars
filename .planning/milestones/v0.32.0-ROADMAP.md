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
- 🚧 **v0.32.0 — Global Alignment Kernel & Kernel Clustering** — Phases 54–56 (in progress)

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (54.1, 54.2): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v0.29.0 — Boosting/Bayesian Regression, FEM/PDE Smoothing & Functional Co-Clustering (Phases 43–45) — SHIPPED 2026-08-30</summary>

- [x] Phase 43: Boosting / Bayesian Functional Regression (REG-06, 5 plans) — new `boosting_regression.rs`
- [x] Phase 44: FEM/PDE Smoothing on Irregular 2D Domains (REP-02) — new `fem_smoothing.rs` + additive `smooth_basis.rs` smoothers
- [x] Phase 45: Functional Co-Clustering (funLBM latent-block) (CLUS-02, 2 plans) — new `coclustering.rs`

Milestone audit PASSED 12/12 requirements. Full detail: [milestones/v0.29.0-ROADMAP.md](milestones/v0.29.0-ROADMAP.md)

</details>

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

### 🚧 v0.32.0 — Global Alignment Kernel & Kernel Clustering (In Progress)

**Milestone Goal:** Add a PSD Global Alignment Kernel (GAK) for curve sets and the kernel machinery it unlocks — clustering natively on curves and enabling external precomputed-kernel SVMs via an exported Gram matrix. Promotes GAP-01 (top-ranked, score 3.00) from the v0.31.0 `GAP-BACKLOG.md`. First **implementation** milestone after three audit/consolidation cycles — real `fdars-core/src/` changes, additive/non-breaking, no new crate dependency; **publishes to crates.io on the `v0.32.0` tag**.

**Phase shape (three phases — dependency spine):** The four researchers converged on a strictly sequential build: GAK kernel core → Gram-matrix export → kernel-k-means. Three small phases (rather than one phase / three plans) were chosen because the milestone is `granularity: fine`, each phase has a distinct set of correctness gates (Phase 54 front-loads all algorithmic risk — log-domain stability, PSD-ness, symmetry, σ-degeneracy, NaN; Phase 55 owns the train/predict split-normalization contract; Phase 56 owns restart/empty-cluster/seeding), and disjoint verification checkpoints match the fine-grained v0.14.0/v0.18.0-era convention. The phases are not thin: Phase 54 alone resolves 7+ named pitfalls. All algorithmic risk is in Phase 54; 55/56 are mechanical wrapping + a standard clustering loop.

- [x] **Phase 54: GAK Kernel Core** - Log-domain PSD Global Alignment Kernel with σ heuristic (GAK-01/02/03/04) (completed 2026-09-02)
- [x] **Phase 55: Gram-Matrix Export (external precomputed-kernel SVM)** - Split train/predict Gram matrices with correct cross-normalization (GAK-05/06) (completed 2026-09-02)
- [x] **Phase 56: Kernel-k-means Clustering** - Cluster curve sets through GAK with restarts, empty-cluster recovery, and out-of-sample predict (GAK-07/08) (completed 2026-09-02)

## Phase Details

### Phase 54: GAK Kernel Core

**Goal**: Users can compute a numerically-stable, positive-semi-definite Global Alignment Kernel between curves — the correctness foundation every downstream feature depends on. New `src/metric/gak.rs` (sibling to `soft_dtw.rs`), re-exported at the crate root; additive/non-breaking, no new dependency.
**Depends on**: Nothing (first phase of the milestone; builds only on existing `metric/soft_dtw.rs` + `distance.rs`)
**Requirements**: GAK-01, GAK-02, GAK-03, GAK-04
**Success Criteria** (what must be TRUE):

  1. User can compute the pairwise GAK similarity between two curves and it returns a non-zero value for long series (a `test_gak_no_underflow` with m ≥ 100–400 points asserts off-diagonal > 1e-10) — proving the forward DP is log-domain (log-sum-exp), never the raw-product recursion.
  2. User gets a normalized similarity in `[0,1]` with unit self-similarity: `k(x,x) == 1.0` (asserted within 1e-12) and no entry exceeds 1.0 or falls below 0.0, via `sqrt(k(x,x)·k(y,y))` triangular normalization — and the result is NaN/Inf-free even for wholly dissimilar curves.
  3. User can build an n×n GAK Gram matrix over a curve set (`cdist_gak`-equivalent) that is **symmetric by assignment** (bit-exact `G[i][j] == G[j][i]`) and **PSD** (minimum eigenvalue ≥ −1e-8 via symmetric eigendecomposition), parallelized under the `parallel` feature via `iter_maybe_parallel!`.
  4. User can auto-select bandwidth σ from a curve set via a median-distance heuristic (`sigma_gak`-equivalent), and with that σ the off-diagonal Gram entries land in a healthy (≈0.05–0.95) range rather than degenerating to near-identity or near-constant.
  5. GAK matches the tslearn@0.9.0 reference within 1e-6 on a small hand-checked dataset (`test_gak_vs_tslearn_reference`).

**Plans**: TBD

### Phase 55: Gram-Matrix Export (external precomputed-kernel SVM)

**Goal**: Users can export GAK Gram matrices suitable for an external precomputed-kernel SVM (`SVC(kernel='precomputed')` convention) — a split train/predict API that makes the cross-normalization bug impossible to hit. Functions live in `metric/gak.rs`, re-exported at the crate root; fdars ships no SVM of its own (native kernel-SVM is deferred to SVM-01).
**Depends on**: Phase 54
**Requirements**: GAK-05, GAK-06
**Success Criteria** (what must be TRUE):

  1. User can export a training Gram matrix (n_train × n_train, symmetric, PSD, unit diagonal) directly consumable as `SVC(kernel='precomputed')` training input, whose result carries the precomputed training self-kernels needed for prediction.
  2. User can export a prediction Gram matrix of shape **n_test × n_train** (asserted, not n_train × n_test) whose entries use the correct cross-normalization against the **stored training** self-kernels — so an external SVM trained on the GAK-05 matrix scores new curves in the same feature space.
  3. The train/predict split is enforced by the API: prediction reuses the identical σ and the stored training self-kernels (not test-set self-kernels alone), and every prediction-matrix entry lies in `[0,1]` — closing the silent self-kernel-normalization bug.
  4. A rustdoc example demonstrates the end-to-end handoff (train Gram + cross Gram → external precomputed-kernel SVM), and Gram construction stays O(n²) with the diagonal self-kernels computed once (no 2× recomputation).

**Plans**: TBD

### Phase 56: Kernel-k-means Clustering

**Goal**: Users can cluster a curve set natively through the GAK kernel — the headline consumer — and assign out-of-sample curves to a fitted model. New top-level `src/kernel_kmeans.rs` (peer of `clustering.rs`), re-exported at the crate root; operates purely on the Gram matrix with no explicit centroid curve.
**Depends on**: Phase 54, Phase 55
**Requirements**: GAK-07, GAK-08
**Success Criteria** (what must be TRUE):

  1. User can cluster a curve set with kernel-k-means through GAK and recover two well-separated synthetic groups with purity 1.0 — assignments computed from Gram-matrix kernel distances, with **no centroid-curve field** in the result struct (kernel-k-means has no centroid).
  2. Clustering is robust: `n_init` random-partition restarts (best-inertia run returned), empty-cluster recovery (a `k > natural clusters` test does not panic and returns valid labels), and the Gram computed once and reused across all restarts.
  3. Results are reproducible: deterministic per-restart RNG seeding (`seed + restart_idx`) means two fits with the same seed produce identical label assignments.
  4. User can assign new (out-of-sample) curves to a fitted model via a `predict` path that reuses the same GAK kernel and normalization as the fit, correctly routing new curves to their group.

**Plans**: TBD

## Progress

**Execution Order (dependency-driven):**
Phases execute in numeric order: 54 → 55 → 56. All algorithmic risk is front-loaded into Phase 54; Phase 55 is mechanical API wrapping over the proven kernel; Phase 56 is a standard kernel-k-means loop consuming the Gram.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 54. GAK Kernel Core | v0.32.0 | 1/1 | Complete    | 2026-09-02 |
| 55. Gram-Matrix Export | v0.32.0 | 1/1 | Complete    | 2026-09-02 |
| 56. Kernel-k-means Clustering | v0.32.0 | 1/1 | Complete    | 2026-09-02 |

## Status

All milestones through **v0.31.0 are shipped and archived** under `milestones/`. The crate is at version 0.30.0. Milestone **v0.32.0** (Phases 54–56) is the active implementation milestone — it promotes GAP-01 (GAK) out of `.planning/research/GAP-BACKLOG.md` and **will** bump the crate + publish on the `v0.32.0` tag. The remaining six backlog items (GAP-02/03/05/06/07/08) carry forward.

Next: `/gsd-plan-phase 54`
