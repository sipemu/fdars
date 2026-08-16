# Milestones

## v0.19.0 Functional Inference Suite (Shipped: 2026-08-16)

**Phases completed:** 2 phases, 2 plans, 0 tasks

**Key accomplishments:**

- 1. [Rule 3 - Blocking] `f_perm_test` merged forward from Task 2 into Task 1's commit
- F-form residual lack-of-fit (Ramsey-RESET style).

---

## v0.18.0 R-Ecosystem Gap Audit (Shipped: 2026-08-15)

**Phases completed:** 4 phases, 5 plans, 0 tasks

**Key accomplishments:**

- `.planning/research/R-AUDIT-REPORT.md` §Phase 16 — R Ecosystem Inventory (INV-01 + INV-02), consolidated from the completed web-sourced survey `16-RESEARCH.md`.
- the `§Design-Goal Filter` section of `.planning/research/R-AUDIT-REPORT.md` (INV-02).
- `.planning/research/R-AUDIT-REPORT.md` §Phase 17 — Parity Matrix & Categorization (GAP-01 + GAP-02), +430 lines. Audit-only; zero `fdars-core/src/` edits.
- `.planning/research/R-AUDIT-REPORT.md` §Phase 18 — Reverse-Parity Strengths Sweep (GAP-03). Audit-only; zero `fdars-core/src/` edits.
- RPT-01 (consolidated report) + RPT-02 (ranked backlog) — the milestone's final deliverables. Audit-only; zero `fdars-core/src/` edits. Committed `c857532f`.

---

## v0.17.0 Registration Parity & Elastic-FPCA Performance (Shipped: 2026-08-12)

**Phases completed:** 2 phases, 3 plans, 3 tasks

**Key accomplishments:**

- New file: `fdars-core/src/alignment/shift.rs`
- Three standalone-energy registration-quality scorers added to alignment/quality.rs (Result-returning, Simpson-weighted), with all five plan-14 items re-exported at the crate root.

---

## v0.16.0 Elastic Feasibility + Parity Quick Wins (Shipped: 2026-08-12)

**Phases completed:** 2 phases, 3 plans, 0 tasks

**Key accomplishments:**

- API surfacing only — no new algorithm.
- 1. [Rule 1 - Bug] Removed `#[must_use]` from `fdata_interpolate_with_policy`
- Five functional scoring metrics (MAE/MSE/MAPE/MSLE/explained-variance) integrated over argvals via Simpson's rule with per-curve averaging, domain validation, and crate-root re-export.
- Fix four code-review findings (CR-01 Periodic NaN, CR-02 EV logic error, WR-01 misleading m=0 error, IN-01 redundant cfg-test) and add `spline_interpolate_with_policy` to close the VERIFICATION gap against ROADMAP SC#2.

---

## v0.15.0 Top-Backlog Quick Wins (Shipped: 2026-08-11)

**Phases completed:** 2 phases, 4 plans, 0 tasks

**Key accomplishments:**

- Adds `spline_interpolate` — order-k B-spline fit-then-evaluate interpolation using the existing `basis/bspline` system, resolving FEAT-01 (REPR-02) with full input validation and 5 inline tests covering exact reproduction and off-grid accuracy.
- Adds five public functional descriptive-statistics functions to `fdata.rs` — Bessel-corrected pointwise variance/std/covariance and FM-depth-based median/trim_mean — closing FEAT-02 (EXPL-02 gap vs scikit-fda).
- Task 1 (tracer): Parallelize the fclassif_cv fold loop
- `fdata_to_pc_1d` now decomposes its weighted matrix with faer `Svd::new_thin` on a zero-copy `MatRef` view under the `linalg` feature — eliminating the dense `to_dmatrix()` copy — while a shared `fix_svd_signs` helper reconciles singular-vector sign conventions so the faer and nalgebra paths produce equivalent `FpcaResult`s within `1e-8·σ₁`.

---

## v0.14.0 Performance & scikit-fda Gap Audit (Shipped: 2026-08-09)

**Phases completed:** 9 phases, 21 plans, 25 tasks
**Milestone audit:** PASSED — 13/13 requirements satisfied, cross-phase integration sound (`.planning/milestones/v0.14.0-MILESTONE-AUDIT.md`)

**Delivered:** An evidence-backed audit of fdars' performance and scikit-fda functionality gaps, consolidated into `.planning/research/AUDIT-REPORT.md` and a value-ranked, promotion-ready `.planning/research/BACKLOG.md`. Audit-only — zero `fdars-core/src/` edits across all 9 phases.

**Key accomplishments:**

- **Measurement discipline (Phases 1–2):** Built a criterion audit-bench harness across the 4-combo feature matrix (`""`/`parallel`/`linalg`/`linalg,parallel`), recorded 12 release baselines over an N×M workload matrix, wrote the methodology + infra-vs-code failure-triage rule, and produced a zero-cost static hot-path map (complexity in N/M, 8 SVD-copy + 14 basis allocation sites, parallelism gaps).
- **Elastic alignment is the top bottleneck (Phase 3):** Full criterion grid confirmed the O(N²·M²) cost — infeasible at N=500,M=200 on the default path — with a measured 4–6× banded-vs-unbanded penalty; root-caused `karcher_mean()` defaulting to `band = None`.
- **FPCA/SVD split (Phase 4):** dhat allocation audit proved the `FdMatrix→DMatrix` SVD-copy is only ~0.14–0.17% of wall-clock; SVD compute dominates (~99.8%), triggering the Phase-6 GO.
- **Parallelism + SVD library (Phases 5–6):** rayon thread-scaling (~4.73× at 8 threads) with 5 safe-to-parallelize loops identified; faer `thin_svd` measured **1.8–4.1× faster** than nalgebra with zero-copy conversion (P6-1).
- **scikit-fda parity (Phases 7–8):** Versioned capability inventory (skfda 0.10.1, 161 rows) → 141-row parity matrix (59 present / 19 partial / 63 absent) → **82 actionable in-scope gaps** (36 table-stakes, 46 differentiator) + a 30-item reverse-parity strengths sweep.
- **Consolidation (Phase 9):** Final report (5 performance findings, 82 gaps, 30 strengths) + a **32-item value-ranked backlog** (`score = value/√effort`, 34 seven-field promotion-ready blocks), all three completeness assertions passed.

---
