# Roadmap: fdars

## Milestones

- ✅ **v0.14.0 — Performance & scikit-fda Gap Audit** — Phases 1–9 (shipped 2026-08-09) — [archive](milestones/v0.14.0-ROADMAP.md)
- ✅ **v0.15.0 — Top-Backlog Quick Wins** — Phases 10–11 (shipped 2026-08-11) — [archive](milestones/v0.15.0-ROADMAP.md)
- ✅ **v0.16.0 — Elastic Feasibility + Parity Quick Wins** — Phases 12–13 (shipped 2026-08-12, PR #40) — [archive](milestones/v0.16.0-ROADMAP.md)
- ✅ **v0.17.0 — Registration Parity & Elastic-FPCA Performance** — Phases 14–15 (shipped 2026-08-12, PR #41) — [archive](milestones/v0.17.0-ROADMAP.md)
- 🚧 **v0.18.0 — R-Ecosystem Gap Audit** — Phases 16–19 (in progress)

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

### 🚧 v0.18.0 R-Ecosystem Gap Audit (Phases 16–19) — IN PROGRESS

**Milestone Goal:** Map fdars' functionality gaps against the R functional-data-analysis package ecosystem — producing a consolidated gap report and a fresh GSD-ready ranked backlog, so future milestones target the highest-leverage R-parity work first. Audit-only — zero `fdars-core/src/` edits (mirrors v0.14.0). The R ecosystem replaces scikit-fda as this milestone's sole yardstick now that the actionable scikit-fda backlog is exhausted (v0.15.0–v0.17.0). Numeric algorithms + API ergonomics in scope; plotting/IO out of scope.

New deliverables land in `.planning/research/`, named distinctly from the archived scikit-fda audit (do **not** overwrite `AUDIT-REPORT.md` / `BACKLOG.md`):

- `.planning/research/R-AUDIT-REPORT.md` — the consolidated R-ecosystem inventory + gap report (mirrors the `AUDIT-REPORT.md` structure/methodology).
- `.planning/research/R-BACKLOG.md` — the fresh `score = value/√effort` ranked backlog (mirrors the `BACKLOG.md` 7-field promotion-block format).

- [x] **Phase 16: R Ecosystem Inventory** - Versioned, area-organized, capability-first inventory of the R FDA ecosystem, then design-goal filtered into in-scope / out-of-scope with per-area counts (completed 2026-08-15)
- [x] **Phase 17: Parity Matrix & Categorization** - Per-capability fdars-vs-R present/partial/absent verdicts (matched by capability, evidence-noted), then every gap categorized table-stakes / differentiator / out-of-scope (completed 2026-08-15)
- [x] **Phase 18: Reverse-Parity Strengths Sweep** - Module-map walk of fdars cataloguing capabilities with no R equivalent or where fdars is ahead of its closest R analog (completed 2026-08-15)
- [ ] **Phase 19: Consolidated Report & Ranked Backlog** - Consolidated R-ecosystem gap report + GSD-ready value-ranked (`score = value/√effort`) promotion-ready backlog

## Phase Details

### Phase 16: R Ecosystem Inventory

**Goal**: A versioned, area-organized capability inventory of the R FDA ecosystem exists, then filtered into the actionable (in-scope) comparison surface — establishing the R side of the parity comparison.
**Depends on**: Phase 15 (prior milestone; no new-milestone dependency — this is the first phase of v0.18.0)
**Requirements**: INV-01, INV-02
**Success Criteria** (what must be TRUE):

  1. `R-AUDIT-REPORT.md` §R-Inventory lists every enumerated capability capability-first (fit/predict/transform collapsed per row, not one row per API name), each row tagged with its source package **and** package version, across the core ecosystem (`fda`, `fda.usc`, `refund`, `fdapace`, `roahd`, `fdaoutlier`, `ftsa`, `MFPCA`/`funData`, `fdasrvf`, `fdatest`/`fdANOVA`, `frechet`/`fdadensity`, `funHDDC`/`FDboost`, plus any further packages surfaced during research)
  2. Capabilities are grouped into named areas (e.g. representation, preprocessing/registration, exploratory/depth-outlier, ML/regression-classification, inference, density/manifold, misc) with a per-area capability count
  3. Every inventoried capability carries a design-goal-filter tag of **in-scope** (numeric algorithm or API-ergonomics) or **out-of-scope** (plotting/visualization or data/IO), with an explicit rationale rule documented once
  4. A per-area in-scope vs out-of-scope count table exists, yielding the total actionable comparison surface the parity matrix will map against

**Plans**: 2 plans

Plans:

- [x] 16-01-PLAN.md — Enumerate the R FDA ecosystem capability-first, versioned + package-tagged, area-organized into 9 areas with per-area counts (INV-01)
- [x] 16-02-PLAN.md — Apply the in-scope / out-of-scope design-goal filter with per-area count table → 248 actionable surface (INV-02)

### Phase 17: Parity Matrix & Categorization

**Goal**: An evidence-backed fdars-vs-R parity matrix exists for every in-scope R capability, with each gap categorized to drive value ranking.
**Depends on**: Phase 16 (the in-scope inventory is the set of rows to map)
**Requirements**: GAP-01, GAP-02
**Success Criteria** (what must be TRUE):

  1. `R-AUDIT-REPORT.md` §Parity-Matrix contains one row per in-scope R capability with a **present / partial / absent** verdict, matched by capability (not API name), driven by a single documented verdict rubric
  2. Every parity row carries a "searched fdars for:" evidence note and a closest-match fdars reference (function/module) or an explicit "no match found"
  3. Present/partial/absent verdict counts are totalled per area and overall, so the actionable-gap count (absent + partial, in-scope) is explicit
  4. Every absent/partial gap is categorized **table-stakes / differentiator / out-of-scope** with a one-line rationale, driven by a single documented category rubric

**Plans**: TBD

Plans:

- [x] 17-01: Build the per-capability fdars-vs-R parity matrix with verdict rubric + evidence notes (GAP-01)
- [ ] 17-02: Categorize every gap table-stakes / differentiator / out-of-scope with rationale (GAP-02)

### Phase 18: Reverse-Parity Strengths Sweep

**Goal**: A full module-map walk of fdars catalogues where fdars is unique or ahead of its closest R analog — the reverse-parity picture that keeps the backlog honest about existing strengths.
**Depends on**: Phase 15 (walks the existing fdars codebase / `.planning/codebase/` module map; independent of the R-side enumeration and may run in parallel with Phases 16–17)
**Requirements**: GAP-03
**Success Criteria** (what must be TRUE):

  1. `R-AUDIT-REPORT.md` §Reverse-Parity-Strengths catalogues fdars capabilities that have **no R equivalent** (e.g. SPM, explainability, streaming depth, conformal prediction, tolerance bands) with the closest-R "none found" note per row
  2. Capabilities where fdars is **ahead** of its closest R analog are listed with the R analog named and the nature of the lead stated (e.g. elastic/shape vs `fdasrvf`)
  3. The sweep is derived from a full module-map walk of `fdars-core` (per-module coverage documented, so the catalogue is demonstrably exhaustive rather than cherry-picked)

**Plans**: TBD

Plans:

- [x] 18-01: Module-map walk of fdars-core cataloguing R-unique and fdars-ahead capabilities (GAP-03)

### Phase 19: Consolidated Report & Ranked Backlog

**Goal**: The R-ecosystem audit is consolidated into a single report and a fresh GSD-ready, value-ranked backlog ready to promote via `/gsd-new-milestone`.
**Depends on**: Phase 16, Phase 17, Phase 18 (consumes the inventory, parity matrix + categorization, and strengths sweep)
**Requirements**: RPT-01, RPT-02
**Success Criteria** (what must be TRUE):

  1. `R-AUDIT-REPORT.md` carries a Methodology section (packages + versions surveyed, in/out-of-scope rule, verdict rubric, category rubric) and a Consolidated Findings section (gap counts by area and category + the fdars-strengths summary)
  2. `R-BACKLOG.md` documents the `score = value / √effort` methodology (value 1–5, effort S/M/L, severity P1/P2/P3) matching the v0.14.0 `BACKLOG.md` convention
  3. `R-BACKLOG.md` contains a master ranked table sorted by strictly non-increasing score, with same-score ties sub-ordered by severity (P1 before P2 before P3)
  4. Every candidate item in the ranked table has a matching 7-field promotion-ready block (candidate requirement/phrasing, location/area, current gap, closest R reference, proposed direction, value+effort+severity, score), so the backlog is directly promotable via `/gsd-new-milestone`

**Plans**: TBD

Plans:

- [ ] 19-01: Consolidate the R-ecosystem gap report — methodology + findings + strengths summary (RPT-01)
- [ ] 19-02: Produce the GSD-ready value-ranked backlog with 7-field promotion blocks (RPT-02)

## Progress

**Execution Order:**
Phases execute in numeric order: 16 → 17 → 19, with 18 parallelizable alongside 16–17 (18 walks the fdars codebase, independent of the R-side enumeration; it must complete before 19).

| Phase | Milestone | Plans Complete | Status | Completed |
|-------|-----------|----------------|--------|-----------|
| 16. R Ecosystem Inventory | v0.18.0 | 2/2 | Complete    | 2026-08-15 |
| 17. Parity Matrix & Categorization | v0.18.0 | 1/1 | Complete    | 2026-08-15 |
| 18. Reverse-Parity Strengths Sweep | v0.18.0 | 1/1 | Complete    | 2026-08-15 |
| 19. Consolidated Report & Ranked Backlog | v0.18.0 | 0/2 | Not started | - |

---
*Latest: v0.18.0 R-Ecosystem Gap Audit started 2026-08-13 — Phases 16–19, 7 requirements (INV-01/INV-02 → Phase 16; GAP-01/GAP-02 → Phase 17; GAP-03 → Phase 18; RPT-01/RPT-02 → Phase 19). Audit-only (zero `fdars-core/src/` edits); R FDA ecosystem is the sole yardstick. Deliverables: `.planning/research/R-AUDIT-REPORT.md` + `.planning/research/R-BACKLOG.md` (distinct from the archived scikit-fda `AUDIT-REPORT.md`/`BACKLOG.md`). Prior: v0.17.0 shipped via PR #41 (crate 0.17.0, tag v0.17.0); v0.16.0 PR #40; v0.15.0 crates.io 0.15.0; v0.14.0 audit shipped 2026-08-09.*
