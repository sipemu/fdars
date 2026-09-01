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
- 🚧 **v0.31.0 — Multi-Ecosystem Gap Audit** — Phases 52–53 (in progress)

## Phases

**Phase Numbering:**

- Integer phases (1, 2, 3, …): Planned milestone work — numbering continues across milestones (never resets)
- Decimal phases (52.1, 52.2): Urgent insertions (marked with INSERTED)

<details>
<summary>✅ v0.28.0 — Spectral Functional Time Series & Object-Data Fréchet Regression (Phases 41–42) — SHIPPED 2026-08-23</summary>

- [x] Phase 41: Spectral Functional Time Series (FTS-03, 2 plans) — new `fts/spectral.rs` (`spectral_density`, `dpca`, `dpca_reconstruct`) + `simulation.rs` (`sim_fvarma`, `sim_farma`)
- [x] Phase 42: Object-Data Fréchet Regression (FRE-02, 3 plans) — new `frechet/spaces/` + generic `frechet_*_space` solvers

Full detail: [milestones/v0.28.0-ROADMAP.md](milestones/v0.28.0-ROADMAP.md)

</details>

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

Milestone audit: **tech_debt** (13/13 requirements satisfied, 6/6 phases verified passed, cross-phase integration SOUND; deferred: REL-01 version bump/publish, APIB-01 breaking removals, Nyquist validate-phase reconciliation for 49/50/51). Full detail: [milestones/v0.30.0-ROADMAP.md](milestones/v0.30.0-ROADMAP.md)

</details>

### 🚧 v0.31.0 — Multi-Ecosystem Gap Audit (In Progress)

**Milestone Goal:** Map fdars' functionality gaps against four fresh reference ecosystems — MATLAB FDA, Julia FDA, tidyfun/refund (R), and Python-beyond-scikit-fda — and produce a single prioritized, de-duplicated, GSD-ready backlog for future implementation milestones. Both prior parity backlogs (scikit-fda v0.14.0, R core v0.18.0) are exhausted; this is the next-yardstick audit.

**Audit fences (apply to every phase):**

- **Audit-only** — zero `fdars-core/src/` edits across the entire milestone. Deliverables are markdown documents only.
- **Net-new gaps only** — hard de-dup against shipped fdars capabilities AND both prior backlogs (`BACKLOG.md` v0.14.0, `R-BACKLOG.md` v0.18.0). Anything fdars already ships or that already sits in a prior backlog is excluded.
- **No git tag / no crate publish** — the crate is unchanged; a `v*` tag would publish a phantom version (project convention for audit milestones).
- **Distinct filenames** — new deliverables land in `.planning/research/` as `GAP-AUDIT-REPORT.md` and `GAP-BACKLOG.md`. Do NOT overwrite the existing `AUDIT-REPORT.md` / `BACKLOG.md` / `R-AUDIT-REPORT.md` / `R-BACKLOG.md`.
- **Scope exclusions** — no plotting/visualization parity, no data/IO parity; no re-audit of scikit-fda or the core R FDA ecosystem (refund only where NOT captured in v0.18.0).

- [ ] **Phase 52: Ecosystem Surveys** - Enumerate, map, and de-dup net-new gaps against MATLAB FDA, Julia FDA, tidyfun/refund, and Python-beyond-scikit-fda (four independent parallel surveys)
- [ ] **Phase 53: Consolidation & Backlog** - Merge the four survey gap-lists into a single cross-ecosystem gap report, a value-ranked GSD-ready backlog, and pass a de-dup + completeness gate

## Phase Details

### Phase 52: Ecosystem Surveys
**Goal**: Four fresh reference ecosystems are surveyed capability-first, fdars is mapped present/partial/absent against each, and each survey emits a de-duplicated net-new gap list — the raw material Phase 53 consolidates.
**Depends on**: Nothing (first phase of milestone; prior milestones shipped)
**Requirements**: MAT-01, JUL-01, TDY-01, PYX-01
**Success Criteria** (what must be TRUE):
  1. Each of the four ecosystems has a versioned capability inventory (package + version pinned) organized capability-first, recorded in `.planning/research/GAP-AUDIT-REPORT.md`.
  2. Each inventory has an fdars present/partial/absent parity mapping, with an explicit "searched fdars for:" note per absent/partial row (mapped by capability, not API name).
  3. Each survey emits a net-new gap list where every listed gap is verified absent from shipped fdars AND absent from both `BACKLOG.md` (v0.14.0) and `R-BACKLOG.md` (v0.18.0).
  4. The TDY-01 survey covers refund methods ONLY where not already captured in v0.18.0, and the PYX-01 survey explicitly excludes scikit-fda (covered by v0.14.0).
  5. All four surveys complete with zero `fdars-core/src/` edits (audit-only fence verified).
**Plans**: 4 plans (one parallel plan per ecosystem — MAT-01, JUL-01, TDY-01, PYX-01; mutually independent)

Plans:
- [ ] 52-01-PLAN.md: MATLAB FDA survey (MAT-01) — Ramsay `fda` MATLAB toolbox + PACE (MATLAB) → `survey-matlab.md`
- [ ] 52-02-PLAN.md: Julia FDA survey (JUL-01) — JuliaStats / functional-data packages (modern/perf-oriented patterns) → `survey-julia.md`
- [ ] 52-03-PLAN.md: tidyfun/refund (R) survey (TDY-01) — tidyfun representation/workflow slice + refund not-in-v0.18.0 → `survey-tidyfun.md`
- [ ] 52-04-PLAN.md: Python-beyond-scikit-fda survey (PYX-01) — FDApy / tslearn / sktime + other Python FDA/ML libs → `survey-pyx.md`

### Phase 53: Consolidation & Backlog
**Goal**: The four per-ecosystem gap lists are merged into a single cross-ecosystem gap report, a value-ranked GSD-ready backlog is produced, and a de-dup + completeness gate confirms every backlog item is genuinely net-new and every surveyed gap is accounted for.
**Depends on**: Phase 52 (all four surveys must be complete — this phase merges/ranks/de-dups their outputs)
**Requirements**: RPT-01, RPT-02, RPT-03
**Success Criteria** (what must be TRUE):
  1. `.planning/research/GAP-AUDIT-REPORT.md` contains methodology, per-ecosystem findings, a cross-ecosystem overlap/convergence analysis (which gaps recur across ≥2 ecosystems), and a reverse-parity strengths sweep (where fdars leads these ecosystems).
  2. `.planning/research/GAP-BACKLOG.md` contains N ranked net-new items sorted strictly non-increasing by `score = value / √effort` (consistent with v0.14.0/v0.18.0), each a promotion-ready block with candidate requirement/phase, effort estimate, reference baseline, and rationale.
  3. Every `GAP-BACKLOG.md` item is verified genuinely net-new — absent from shipped fdars, from `BACKLOG.md`, and from `R-BACKLOG.md` (de-dup gate PASS).
  4. Every surveyed capability gap from Phase 52 is either ranked in `GAP-BACKLOG.md` or explicitly recorded as out-of-scope with reasoning (completeness gate PASS).
  5. Both deliverables are written with zero `fdars-core/src/` edits, use the distinct `GAP-*` filenames (existing audit reports/backlogs untouched), and no git tag is created.
**Plans**: 3 plans (RPT-01 report, then RPT-02 backlog, then RPT-03 de-dup + completeness gate — internal order: gate last)

Plans:
- [ ] 53-01: Consolidated multi-ecosystem gap report (RPT-01) — `GAP-AUDIT-REPORT.md`: methodology, per-ecosystem findings, cross-ecosystem convergence, reverse-parity strengths
- [ ] 53-02: Ranked GSD-ready backlog (RPT-02) — `GAP-BACKLOG.md`: value/√effort ranking, promotion-ready item blocks
- [ ] 53-03: De-dup & completeness gate (RPT-03) — verify every item net-new vs shipped + `BACKLOG.md` + `R-BACKLOG.md`; every gap ranked or out-of-scope-with-reason

## Progress

**Execution Order:**
Phases execute in numeric order: 52 → 53

Phases through **v0.30.0 are shipped and archived** under `milestones/`. The crate remains at version 0.29.0 — the v0.30.0 version bump + `cargo publish` + tag is the deferred operator ship step (REL-01). **v0.31.0 is audit-only: no crate change, no version bump, no git tag.**

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 52. Ecosystem Surveys | v0.31.0 | 0/4 | Not started | - |
| 53. Consolidation & Backlog | v0.31.0 | 0/3 | Not started | - |

Next: `/gsd-plan-phase 52`
