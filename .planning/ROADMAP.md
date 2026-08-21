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

## Phases

<details>
<summary>✅ v0.26.0 FPCA Breadth & Sparse Covariance (Phases 37–38) — SHIPPED 2026-08-21</summary>

The two remaining top-ranked `R-BACKLOG.md` items (score 1.73, M-effort), exhausting the 1.73 tier: FPCA-02 (Phase 37), SPARSE-01 (Phase 38). Both additive/non-breaking, reuse-first, no new crate dependency; whole-crate 2414 lib + doc tests green, `cargo clippy --all-targets` clean. Both phases independent (disjoint modules: `fpca_variants.rs` vs `irreg_fdata/face.rs`), each verified 5/5, milestone audit PASSED 8/8.

- [x] Phase 37: Specialized FPCA Variants (2/2 plans) — completed 2026-08-21 (FPCA-02) — `fpca_der`, `fsvd`, `cross_covariance`, `dynamical_correlation`, `ssvd`
- [x] Phase 38: Sparse Fast Covariance & Trajectory Bands (2/2 plans) — completed 2026-08-21 (SPARSE-01) — `face_covariance`, `mface_covariance`, `face_trajectory`

Full phase detail: [milestones/v0.26.0-ROADMAP.md](milestones/v0.26.0-ROADMAP.md)

</details>

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
