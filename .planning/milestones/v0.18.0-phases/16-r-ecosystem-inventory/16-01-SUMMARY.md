---
phase: 16-r-ecosystem-inventory
plan: "16-01"
status: complete
requirements: [INV-01, INV-02]
deliverable: .planning/research/R-AUDIT-REPORT.md
audit_only: true
completed: 2026-08-14
---

# Phase 16 · Plan 01 — Summary

**Delivered:** `.planning/research/R-AUDIT-REPORT.md` §Phase 16 — R Ecosystem Inventory (INV-01 + INV-02), consolidated from the completed web-sourced survey `16-RESEARCH.md`.

## What shipped

- **Methodology preamble** — the knowledge+CRAN sourcing convention (no local R install, no `packageVersion()`), the survey-month convention ("as of 2026-08"), the capability-first collapse rule, and the in/out-of-scope design-goal rule stated once.
- **Packages Surveyed** — 35 active R FDA packages, each with CRAN version + date (verified 2026-08-14) + primary area; plus the considered-and-excluded list (`classiFunc`/`FRegSigCom`/`fpca`/`warpMix` archived; `rainbow` in-but-out-of-scope; `refund.shiny`/`mlr3fda`/`tidyfun` excluded).
- **§R-Inventory** — 275 capability rows across 9 named areas (Representation/Basis/Smoothing 45; Preprocessing/Registration 22; Exploratory/Depth/Outlier 38; ML Regression+Classification+Clustering 61; Inference/Testing 25; Functional Time Series 26; Density/Object-Data/Manifold 25; SPM/Control Charts 12; FPCA Sparse/Longitudinal 21). Every row is capability-first (fit/predict/transform collapsed), tagged with source package(s) + version, and given a relevance tag. 326 versioned package tags total.
- **§Design-Goal Filter** — per-area in-scope vs out-of-scope count table: **248 in-scope / 27 out-of-scope (24 plotting + 3 IO) / 275 total**. The 248 in-scope capabilities are the actionable comparison surface for Phase 17.
- **Forward notes** — architectural responsibility map, Phase-17 parity-mapping pitfalls, and open questions.

## Corrections applied during consolidation

- Stale draft figures in the survey's early Summary ("210+ capabilities", "181 in-scope") were **not** carried over — the authoritative §Design-Goal Filter totals (275/248/27) are used.
- TOTAL out-of-scope breakdown corrected from "25 plotting, 3 IO" to **"24 plotting, 3 IO"** (the per-area rows sum to 24 plotting; the 248/27/275 totals were already consistent).

## Verification

All 4 ROADMAP Phase-16 success criteria pass (see `16-VERIFICATION.md`). Zero `fdars-core/src/` edits. Deliverable is a distinct file from the archived scikit-fda `AUDIT-REPORT.md`, which is untouched.

## Notes / deviations

- The R-ecosystem survey (the hard work) was produced by a web-enabled `gsd-phase-researcher` (RESEARCH.md); the planner then produced this plan. Due to repeated background-agent losses on process exit during this session, the consolidation execution (this plan's Task 1/2) and the phase bookkeeping were performed inline by the orchestrator rather than a spawned `gsd-executor`. Output verified against the plan's must-haves and all ROADMAP success criteria.
