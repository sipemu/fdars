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
- 🔨 **v0.26.0 — FPCA Breadth & Sparse Covariance** — Phases 37–38 (in progress)

## Phases

- [x] **Phase 37: Specialized FPCA Variants** (2/2 plans) — completed 2026-08-21 (FPCA-02) - Add FPCA of derivatives (`fpca_der`), functional SVD / cross-FPCA (`fsvd`), cross-covariance surfaces (`cross_covariance`), dynamical/functional correlation (`dynamical_correlation`), and a sandwich-smoother / sparse-SVD (ssvd) FPCA path — extend `regression.rs` (or new `fpca_variants.rs`), reusing `fdata_to_pc_1d` + `covariance.rs` (FPCA-02, independent)
- [ ] **Phase 38: Sparse Fast Covariance & Trajectory Bands** - Add the FACE fast-sandwich sparse-data covariance estimator (`face_covariance`), its multivariate `mfaces` extension (`mface_covariance`), and fitted continuous trajectories with pointwise confidence bands — extend `irreg_fdata/`, reusing `cov_irreg` and integrating with the shipped PACE `pace_fpca` (SPARSE-01, independent)

## Phase Details

### Phase 37: Specialized FPCA Variants

**Goal**: A user can run the specialized FPCA variants that `fdapace`/`refund` expose but fdars was missing — FPCA of curve derivatives (`fpca_der`), a functional SVD / cross-FPCA between two functional samples (`fsvd`), a cross-covariance surface between two samples (`cross_covariance`), a dynamical/functional correlation scalar (`dynamical_correlation`), and a sandwich-smoother / sparse-SVD (ssvd) FPCA path — all by extending `fdars-core/src/regression.rs` (or a new `fpca_variants.rs`), reusing the existing dense FPCA (`fdata_to_pc_1d`) and `covariance.rs`, without any existing code changing. Completes the FPCA long tail alongside the already-shipped PACE core (FPCA-01).
**Depends on**: Nothing (independent of Phase 38; may run in any order or in parallel). Reuses shipped `fdata_to_pc_1d` + `covariance.rs`.
**Requirements**: FPCA-02 (FPCA-02-01, FPCA-02-02, FPCA-02-03, FPCA-02-04, FPCA-02-05)
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points (crate-root re-exported) in `fdars-core/src/regression.rs` (or a new `fpca_variants.rs`), each consuming functional data in column-major `FdMatrix` form and returning structured numeric output: FPCA of derivatives (`fpca_der`), a functional SVD / cross-FPCA (`fsvd`), a cross-covariance surface (`cross_covariance`), a dynamical/functional correlation (`dynamical_correlation`), and a sandwich-smoother / sparse-SVD (ssvd) FPCA path.
  2. `fpca_der` returns derivative loadings and scores of the differentiated process such that, on a synthetic sample whose curves differ along a known smooth mode of variation, the leading derivative component reconstructs the differentiated curves within a documented tolerance (inline `#[cfg(test)]` tests).
  3. `fsvd` returns paired left/right singular functions and singular values whose bivariate reconstruction recovers a known low-rank cross-covariance structure between two functional samples within a documented tolerance; `cross_covariance` returns a symmetric-in-construction surface over the two argument grids that agrees with the empirical cross-covariance on a hand-computed reference; and `dynamical_correlation` returns a scalar in a documented range that is 1 (within tolerance) for perfectly co-varying samples and near 0 for independent samples (inline `#[cfg(test)]` tests).
  4. The sandwich-smoother / sparse-SVD (ssvd) path estimates loadings/scores via a smoothed-covariance (sandwich) estimator as an alternative to the raw thin-SVD decomposition, agreeing with the dense `fdata_to_pc_1d` result in the dense/no-smoothing limit within a documented tolerance; all variants reuse `fdata_to_pc_1d` + `covariance.rs` rather than adding a new subsystem, add no new crate dependency, and invalid inputs (empty matrix, mismatched argvals vs values, mismatched sample sizes between the two samples for `fsvd`/`cross_covariance`/`dynamical_correlation`, `ncomp` out of range, degenerate columns) return `FdarError` rather than panicking.
  5. Existing public signatures across `fdars-core` (including `fdata_to_pc_1d` and everything in `covariance.rs`) keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 2 plans
- [ ] 37-01-PLAN.md — Create fpca_variants module + FsvdResult; land cross_covariance (tracer) + fpca_der (Wave 1)
- [ ] 37-02-PLAN.md — dynamical_correlation, fsvd, ssvd + crate-root smoke re-export test (Wave 2)

### Phase 38: Sparse Fast Covariance & Trajectory Bands

**Goal**: A user can estimate sparse/irregular functional covariance with the fast-sandwich smoother `face`/`mfaces` expose but fdars was missing — the FACE fast-sandwich covariance surface for sparse/irregular data (`face_covariance`), its multivariate extension for multiple simultaneously-observed sparse variables (`mface_covariance`), and integrated fitted continuous trajectories with pointwise confidence bands for sparse curves — all by extending `fdars-core/src/irreg_fdata/`, building on the existing `cov_irreg` and integrating with the shipped PACE `pace_fpca` (FPCA-01) machinery where applicable, without any existing code changing.
**Depends on**: Nothing (independent of Phase 37; may run in any order or in parallel). Builds on shipped `irreg_fdata::cov_irreg` + `pace_fpca`.
**Requirements**: SPARSE-01 (SPARSE-01-01, SPARSE-01-02, SPARSE-01-03)
**Success Criteria** (what must be TRUE):

  1. User can call new `Result`-returning public entry points (crate-root re-exported) in `fdars-core/src/irreg_fdata/`, each consuming sparse/irregular functional data and returning structured numeric output: a FACE fast-sandwich covariance surface (`face_covariance`), a multivariate `mfaces` covariance (`mface_covariance`), and fitted trajectories with pointwise confidence bands integrated with the FACE covariance path.
  2. `face_covariance` returns a symmetric covariance surface that recovers a known covariance surface on dense-limit synthetic data (curves sampled densely enough to approximate the regular case) within a documented tolerance, and reuses `cov_irreg` / the existing sparse-covariance machinery rather than adding a new subsystem (inline `#[cfg(test)]` tests).
  3. `mface_covariance` estimates the joint (block) covariance across multiple simultaneously-observed sparse functional variables, recovering the correct within-variable and cross-variable covariance blocks on a synthetic two-variable sample with a known cross-structure within a documented tolerance (inline `#[cfg(test)]` tests).
  4. The fitted-trajectory entry point returns, per sparse curve, a fitted continuous trajectory on the requested grid together with pointwise confidence bands, integrating with the FACE covariance path (and reusing `pace_fpca` machinery where applicable) such that on densely-sampled synthetic curves the fitted trajectory tracks the true curve within its bands within a documented tolerance; the module adds no new crate dependency, and invalid inputs (empty sample, mismatched variable counts / observation counts for `mface_covariance`, non-monotone or mismatched argvals, degenerate/all-missing curves, invalid bandwidth) return `FdarError` rather than panicking.
  5. Existing public signatures across `fdars-core` (including `irreg_fdata::cov_irreg` and `pace_fpca`) keep working unchanged (additive/non-breaking); the full suite plus `cargo clippy --all-targets --features linalg,parallel -- -D warnings` stays green.

**Plans**: 2 plans
- [ ] 38-01-PLAN.md — Create face.rs + module wiring + `face_covariance` (SPARSE-01-01) end-to-end tracer + full gate (Wave 1)
- [ ] 38-02-PLAN.md — `mface_covariance` + `MfaceCovResult` (SPARSE-01-02), `face_trajectory` (SPARSE-01-03), crate-root re-export smoke test (Wave 2)

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 37. Specialized FPCA Variants | 2/2 | ✅ Complete | 2026-08-21 |
| 38. Sparse Fast Covariance & Trajectory Bands | 0/2 | Not started | - |

**Execution order:** Both phases are **independent** — FPCA-02 (Phase 37) and SPARSE-01 (Phase 38) have **no cross-phase hard dependency** (as with prior implementation milestones), so they may be planned and executed in **any order or in parallel**. Each extends a disjoint area of the codebase (extend `regression.rs` / new `fpca_variants.rs` vs extend `irreg_fdata/`). The two remaining top-ranked `R-BACKLOG.md` items in the 1.73 tier (FPCA-02 rank 18, SPARSE-01 rank 19 — both score 1.73, M-effort). Both are P3 differentiators completing the FPCA/covariance cluster and complementing the already-shipped PACE core (FPCA-01); SPARSE-01's trajectory-band output integrates with `pace_fpca`. Additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports, **zero changes to existing public signatures**; reuse-first, **no new crate dependency**; numeric outputs only (plotting/rendering out of scope). R baselines matched by capability, not R's exact signatures.

<details>
<summary>✅ v0.25.0 Serial Dependence, Representation & Density Breadth (Phases 34–36) — SHIPPED 2026-08-21</summary>

The next three top-ranked `R-BACKLOG.md` items (score 1.73 each, M-effort): FTS-02 (Phase 34), REP-01 (Phase 35), DENS-01 (Phase 36). All additive/non-breaking, reuse-first, no new crate dependency; whole-crate 2385 lib + 166 doc tests green, `cargo clippy --all-targets` clean. All three phases independent (disjoint modules).

- [x] Phase 34: Functional Serial-Dependence Tooling (3/3 plans) — completed 2026-08-21 (FTS-02)
- [x] Phase 35: Basis-System Completions (4/4 plans) — completed 2026-08-21 (REP-01)
- [x] Phase 36: Density Object-Data FDA (3/3 plans) — completed 2026-08-21 (DENS-01)

Full phase detail: [milestones/v0.25.0-ROADMAP.md](milestones/v0.25.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.24.0 Functional Regression & Clustering Breadth (Phases 31–33) — SHIPPED 2026-08-20</summary>

The three top-ranked P2 differentiators from the R-ecosystem backlog (score 1.73 each), all additive/non-breaking to `fdars-core` with zero changes to existing public signatures and no new crate dependency. Milestone audit passed; whole-crate 2268-test suite + `cargo clippy --all-targets` green. All three phases independent (disjoint modules).

- [x] Phase 31: Additive Functional Regression & Variable Selection (2/2 plans) — completed 2026-08-20 (REG-04)
- [x] Phase 32: Flexible Mixed-Effects Regression (2/2 plans) — completed 2026-08-20 (REG-05)
- [x] Phase 33: Model-Based & Density Functional Clustering (3/3 plans) — completed 2026-08-20 (CLUS-01)

Full phase detail: [milestones/v0.24.0-ROADMAP.md](milestones/v0.24.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.23.0 Depth, Outliers & Interval Inference (Phases 28–30) — SHIPPED 2026-08-20</summary>

The top three P2 differentiator gaps from the R-ecosystem backlog (score 2.31 each), all additive/non-breaking. Milestone audit passed (3/3). DEPTH→OUT chain; INF-03 independent.

- [x] Phase 28: Depth-Measure Long Tail (3/3 plans) — completed 2026-08-20 (DEPTH-01)
- [x] Phase 29: Outlier-Detector Suite (2/2 plans) — completed 2026-08-20 (OUT-01)
- [x] Phase 30: Interval Testing Procedure Family (2/2 plans) — completed 2026-08-20 (INF-03)

Full phase detail: [milestones/v0.23.0-ROADMAP.md](milestones/v0.23.0-ROADMAP.md)

</details>

<details>
<summary>✅ v0.14.0–v0.22.0 (Phases 1–27) — SHIPPED</summary>

- ✅ **v0.14.0 Performance & scikit-fda Gap Audit** (Phases 1–9) — audit-only; `AUDIT-REPORT.md` + `BACKLOG.md`. [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 Top-Backlog Quick Wins** (Phases 10–11, PR #38) — FEAT-01/02, PERF-01/02. [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 Elastic Feasibility + Parity Quick Wins** (Phases 12–13, PR #40) — PERF-03, FEAT-03/04/05. [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 Registration Parity & Elastic-FPCA Performance** (Phases 14–15, PR #41) — FEAT-06/07, PERF-04. [archive](milestones/v0.17.0-ROADMAP.md)
- ✅ **v0.18.0 R-Ecosystem Gap Audit** (Phases 16–19) — audit-only; `R-AUDIT-REPORT.md` + `R-BACKLOG.md`. [archive](milestones/v0.18.0-ROADMAP.md)
- ✅ **v0.19.0 Functional Inference Suite** (Phases 20–21) — INF-01, INF-02; new `inference/` module. [archive](milestones/v0.19.0-ROADMAP.md)
- ✅ **v0.20.0 Table-Stakes Quick Wins** (Phases 22–23) — T-01, T-02. [archive](milestones/v0.20.0-ROADMAP.md)
- ✅ **v0.21.0 Functional Regression Completeness** (Phases 24–25) — REG-01, REG-02. [archive](milestones/v0.21.0-ROADMAP.md)
- ✅ **v0.22.0 PACE Sparse FPCA & Elastic Multinomial** (Phases 26–27) — FPCA-01, REG-03. [archive](milestones/v0.22.0-ROADMAP.md)

</details>
