# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (shipped 2026-08-12, PR #40) — [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 — Registration Parity & Elastic-FPCA Performance** — Phases 14–15 (shipped 2026-08-12, PR #41) — [archive](milestones/v0.17.0-ROADMAP.md)
- ✅ **v0.18.0 — R-Ecosystem Gap Audit** — Phases 16–19 (shipped 2026-08-15) — [archive](milestones/v0.18.0-ROADMAP.md)

## Phases

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
