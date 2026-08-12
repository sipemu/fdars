# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (complete 2026-08-12, release pending) — [archive](milestones/v0.16.0-ROADMAP.md)

## Phase Details

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
<summary>✅ v0.16.0 Elastic Feasibility + Parity Quick Wins (Phases 12–13) — COMPLETE 2026-08-12 (release pending)</summary>

Second implementation milestone — the next tier of the v0.14.0 audit backlog: the P1 elastic-feasibility headline plus three effort-S scikit-fda parity gaps, all additive/non-breaking. Milestone audit passed (4/4 requirements, cross-phase integration clean, 2663 tests green). Not yet released — needs a version bump (0.15.0 → 0.16.0) + PR to protected `main` + tag.

- [x] Phase 12: Elastic Feasibility — Banded Alignment Default & `band_frac` (1/1 plans) — completed 2026-08-12 (PERF-03: opt-in `*_with_band` wrappers, large grids feasible)
- [x] Phase 13: Parity Quick Wins — Imputation, Extrapolation Policy & Scoring Metrics (2 plans + 1 gap-closure) — completed 2026-08-12 (FEAT-03 imputation, FEAT-04 `ExtrapolationPolicy` both interp paths, FEAT-05 five scoring metrics)

Full phase detail: [milestones/v0.16.0-ROADMAP.md](milestones/v0.16.0-ROADMAP.md)

</details>

---
*Latest: v0.16.0 code-complete 2026-08-12 — 2 phases, 4/4 requirements, milestone audit passed (2663 tests green); release pending (version bump + PR + tag). Prior: v0.15.0 shipped 2026-08-11 (crates.io 0.15.0); v0.14.0 audit shipped 2026-08-09. Next milestone starts with `/gsd-new-milestone` (fresh REQUIREMENTS.md).*
