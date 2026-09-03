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
- 🚧 **v0.35.0 — Optimal Experimental Design for Sparse FDA (FOptDes)** — Phases 64–65 (in progress)

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

### 🚧 v0.35.0 — Optimal Experimental Design for Sparse FDA (FOptDes) (In Progress)

**Milestone Goal:** Add optimal sparse-measurement design — choose the measurement locations that minimize the prediction error of PACE-recovered curves (or the posterior variance of predicted FPC scores), given an already-fitted PACE model. A pure, two-stage design step over a supplied `PaceFpcaResult` (no re-estimation). Promotes GAP-05 (score 2.12, M-effort) from the v0.31.0 `GAP-BACKLOG.md` — the first milestone drawing from the *design* front of the backlog, built directly on the shipped `pace_fpca` estimator. Reference baseline: PACE@2.17 (MATLAB) `FOptDes` / fdapace; Ji & Müller (2017), Yao–Müller–Wang (2005). Implementation milestone — real `fdars-core/src/` changes, additive/non-breaking (zero edits to existing public signatures; protects R + WASM bindings + 28 examples), normal test/clippy/fmt gates; **publishes to crates.io on the `v0.35.0` tag** (ships code → tag as normal; the audit-milestone-no-tag convention does NOT apply).

**Phase shape (two phases — a strict sequential dependency chain):** All four researchers converged (HIGH confidence) on a clean two-phase build mirroring the v0.34.0 SBD-core → k-Shape and the general criterion-primitive → greedy-wrapper precedent. All new code lives in **ONE** new top-level file `src/optimal_design.rs` (peer of `kshape.rs`/`kernel_kmeans.rs`), plus additive `lib.rs`/`prelude.rs` re-exports; **no new crate dependency** (MSRV 1.81 preserved, `linalg` feature NOT required — `cholesky_solve` is always available). All math reuses `pace_fpca.rs` (the Σ_yi assembly at lines 461–474 and the A_mat/Ω_i posterior-covariance pattern at 547–558), `linalg::cholesky_solve`, `helpers::simpsons_weights`, `helpers::linear_interp`, and `iter_maybe_parallel!`. **Phase 64** builds the pure, isolated, known-answer-testable criterion machinery — the shared `Σ_design`/`Γ*(S,S)` builder, the trajectory-reconstruction criterion (FOD-01), the FPC-score A-/D-optimality criterion (FOD-02), and the public `design_criterion` evaluator with `DesignCriterion`/`OptimalityKind` enums (FOD-03) — with no greedy loop yet. **Phase 65** wraps it in the greedy sequential forward-selection loop (FOD-04) + `OptDesConfig`/`OptDesResult` + the two-stage `&PaceFpcaResult` entry point (FOD-05) + additive crate-root/prelude re-exports + criterion benchmark. Execution is a strict chain (64 → 65): the greedy loop delegates entirely to Phase 64's `design_criterion`, so no parallelization is possible. Neither phase needs a `--research-phase` pass (well-documented Ji & Müller / fdapace patterns), though the numerical make-or-break gates warrant known-answer tests cross-checked against the formulas. Scalar-response (SR) design, exhaustive/global search, CV-ridge selection, rank-1 Cholesky updates, and off-grid interpolated candidates (FOD-BREADTH) are deferred.

- [x] **Phase 64: Criterion Machinery Core** - Shared `Σ_design` builder + trajectory-reconstruction criterion + FPC-score A-/D-optimality criterion + public `design_criterion` evaluator with `DesignCriterion`/`OptimalityKind` enums; new `src/optimal_design.rs` (FOD-01/02/03)
- [x] **Phase 65: Greedy Selection & Integration** - Greedy sequential forward-selection `optimal_design` wrapper + `OptDesConfig`/`OptDesResult` + two-stage `&PaceFpcaResult` entry point + additive crate-root/prelude re-exports + criterion benchmark (FOD-04/05)

## Phase Details

### Phase 64: Criterion Machinery Core

**Goal**: Users (and the Phase 65 greedy loop) can score any caller-supplied design point set against a fitted PACE model through a single public `design_criterion` evaluator — computing either the integrated trajectory-reconstruction BLUP-MSE or the A-/D-optimal posterior score-covariance summary. This is the pure, isolated, known-answer-testable math core in a new `src/optimal_design.rs`: the shared `Σ_design` builder plus both criterion branches, with NO greedy loop yet. It front-loads every numerical make-or-break gate — every downstream selection result inherits any bug here.
**Depends on**: Nothing new (first phase of the milestone; builds only on the existing, unchanged `pace_fpca::PaceFpcaResult`, `linalg::cholesky_solve`, `helpers::{simpsons_weights, linear_interp}`, `matrix::FdMatrix` column-major indexing, and `error::FdarError`)
**Requirements**: FOD-01, FOD-02, FOD-03
**Success Criteria** (what must be TRUE):

  1. User can compute the **trajectory-reconstruction** criterion for a design index set via `design_criterion(model, indices, DesignCriterion::Trajectory, _)`: the shared private `build_sigma_design` assembles the p×p `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p` (row-major, mirroring `pace_fpca.rs:461–474`, verified shape `|S|×|S|` not `K×K`), the integrated conditional BLUP-MSE `Σ_j w_j · (Σ_k λ_k φ_k(t_j)² − φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j))` is computed over the work grid with **Simpson weights** from `simpsons_weights(&model.argvals)` (never a uniform `1.0`/`1/m`), and the known-answer identity `MSE(∅) ≈ Σ_k λ_k` holds (grid-size-invariant) on a synthetic orthonormal 2-eigenfunction model.
  2. User can compute the **FPC-score-prediction** criterion via `design_criterion(model, indices, DesignCriterion::Score, OptimalityKind::A | D)`: the K×K posterior score covariance `Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ` (the `pace_fpca.rs:547–558` A_mat/Ω_i pattern generalized to a prospective design set) is formed via `cholesky_solve`, returning **A-optimality** `trace(Cov)` or **D-optimality** `log det(Cov)` (negative — all posterior eigenvalues ≤ prior λ_k); the known-answer `Cov(ξ|∅) = diag(λ)` holds, so `A(∅) = Σ_k λ_k` and `D(∅) = Σ_k log λ_k`.
  3. Both criteria are **monotone non-increasing** as design points are added (more information never raises posterior uncertainty): adding any point to `indices` satisfies `criterion(S ∪ {t}) ≤ criterion(S) + 1e-12` for Trajectory, A-opt, and D-opt — the optimality-sign gate that guarantees the Phase 65 greedy loop minimizes (never maximizes) the objective.
  4. `design_criterion` is a public, `#[must_use]`, reusable evaluator (independent of any selection loop) exposing `DesignCriterion` (`Trajectory` | `Score`) and `OptimalityKind` (`A` | `D`) enums (both serde-gated), and it is numerically robust: candidate indices are validated in-range, σ² > 0 is asserted, a near-singular `Σ_d` triggers the `pace_fpca.rs` `1e-8` ridge-retry (returns `FdarError::ComputationFailed`, never panics), and no duplicate/near-duplicate index silently corrupts the result. Enums + `design_criterion` are additively re-exported from `lib.rs` (existing signatures untouched).

**Plans**: 2 plans
- [ ] 64-01-sigma-design-and-trajectory-PLAN.md — Wave 1: new `optimal_design.rs` with `build_sigma_design`, validation, ridge-retry, and the trajectory BLUP-MSE branch (FOD-01) with known-answer/grid-invariance/monotonicity tests
- [ ] 64-02-score-criterion-and-reexport-PLAN.md — Wave 2: Score A-/D-optimality posterior-covariance branch (FOD-02), enum dispatch, and additive `lib.rs` re-export (FOD-03) with CI gates

### Phase 65: Greedy Selection & Integration

**Goal**: Users can run the whole two-stage FOptDes workflow — pass an already-estimated `PaceFpcaResult` and an `OptDesConfig` to `optimal_design`, and get back the greedily-selected sparse design (point indices, argvals, and the achieved-criterion trace). This phase wraps Phase 64's validated `design_criterion` in a deterministic greedy sequential forward-selection loop, adds the `OptDesConfig`/`OptDesResult` types, finalizes the additive crate-root + prelude re-exports, and lands the criterion benchmark. A thin orchestration layer over Phase 64 — it adds no new math.
**Depends on**: Phase 64 (`design_criterion` public + validated; the greedy loop delegates entirely to it for every candidate at every step). Reuses `iter_maybe_parallel!` (parallel candidate evaluation), `linalg::cholesky_solve`, and `pace_fpca::PaceFpcaResult` unchanged.
**Requirements**: FOD-04, FOD-05
**Success Criteria** (what must be TRUE):

  1. User can call `optimal_design(model: &PaceFpcaResult, config: &OptDesConfig) -> Result<OptDesResult, FdarError>` to obtain the optimal sparse design under a point budget via **greedy sequential forward selection**: start empty, and at each of `config.budget` steps add the not-yet-selected candidate that most reduces `config.criterion` (evaluated through `design_criterion`), until the budget is reached. The two-stage contract holds — the supplied PACE model is consumed read-only, with **no re-estimation** of eigenstructure or σ².
  2. The greedy selection is **deterministic and duplicate-free**: candidate evaluation may parallelize via `iter_maybe_parallel!` but the argmin is a sequential smallest-index tie-break, so two identical calls produce byte-identical `selected_indices` **and the result is identical with and without `--features parallel`**; already-selected indices are excluded each step (no index appears twice); and the achieved-criterion trace is monotone non-increasing across steps.
  3. `OptDesConfig` (candidate grid, budget `p`, `DesignCriterion`, `OptimalityKind`; `Default` impl, no `#[non_exhaustive]`) and `OptDesResult` (`selected_indices`, `selected_argvals`, achieved-criterion trace; `#[non_exhaustive]`) follow the `PaceFpcaConfig`/`PaceFpcaResult` precedent (Debug/Clone/PartialEq + serde-gated derives), and input validation returns `Err(FdarError::...)` for `budget == 0`, `budget > candidate_grid.len()`, any candidate not in `model.argvals`, `model.ncomp == 0`, or `model.sigma2 <= 0`.
  4. The full FOptDes public surface is re-exported at the crate root additively (`pub mod optimal_design`; `optimal_design`, `design_criterion`, `OptDesConfig`, `OptDesResult`, `DesignCriterion`, `OptimalityKind`) with `prelude` additions, a module-level doctest demonstrates the end-to-end workflow (fit PACE → `optimal_design` → read `selected_argvals`), a criterion benchmark covers `design_criterion` + `optimal_design` (Trajectory and Score) on a representative grid/budget, and whole-crate gates pass: `cargo fmt --check`, `cargo clippy --all-targets --features linalg,parallel -- -D warnings`, and the full lib + doctest suite — 28 examples + WASM + R bindings unaffected, no existing public signature changed, no new crate dependency.

**Plans**: 2 plans
- [ ] 65-01-greedy-loop-and-config-types-PLAN.md — Wave 1 (tracer): extend `optimal_design.rs` with `OptDesConfig`/`OptDesResult` + the deterministic greedy `optimal_design` loop + candidate→index mapping + validation + 13 inline tests (FOD-04, FOD-05 algorithmic half)
- [ ] 65-02-reexports-doctest-benchmark-PLAN.md — Wave 2: additive `lib.rs`/`prelude.rs` full-surface re-exports + module-level end-to-end doctest + new criterion benchmark + `[[bench]]` stanza + whole-crate gates (FOD-05)

## Progress

**Execution Order (dependency-driven — strict chain, no reordering or parallelization):**
Phases execute in numeric order: 64 → 65. Phase 65 cannot begin before Phase 64's `design_criterion` (and the `DesignCriterion`/`OptimalityKind` enums) compile and pass their known-answer gates — the greedy loop delegates entirely to it. Phase 64 front-loads every numerical make-or-break gate (Σ_yi assembly, score posterior covariance, Simpson-weighted integration, optimality sign/monotonicity); Phase 65 is a thin deterministic greedy wrapper + config/result types + re-exports + benchmark.

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 64. Criterion Machinery Core | v0.35.0 | 0/TBD | Not started | - |
| 65. Greedy Selection & Integration | v0.35.0 | 0/2 | Not started | - |

## Status

All milestones through **v0.34.0 are shipped and archived** under `milestones/`. Milestone **v0.35.0** (Phases 64–65) is the active implementation milestone — it promotes GAP-05 (Optimal Experimental Design for Sparse FDA / FOptDes) out of `.planning/research/GAP-BACKLOG.md` and **will** bump the crate + publish on the `v0.35.0` tag. The remaining three backlog items (GAP-06/07/08 — PEER/lpeer, wavelet regression, differentiable core) carry forward, drawn top-first.

Next: `/gsd-plan-phase 64`
