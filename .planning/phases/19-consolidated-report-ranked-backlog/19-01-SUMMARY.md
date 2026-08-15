---
phase: 19-consolidated-report-ranked-backlog
plan: "19-01"
status: complete
requirements: [RPT-01, RPT-02]
deliverables:
  - .planning/research/R-AUDIT-REPORT.md
  - .planning/research/R-BACKLOG.md
audit_only: true
completed: 2026-08-15
---

# Phase 19 · Plan 01 — Summary

**Delivered:** RPT-01 (consolidated report) + RPT-02 (ranked backlog) — the milestone's final deliverables. Audit-only; zero `fdars-core/src/` edits. Committed `c857532f`.

## RPT-01 — Consolidated Report (appended to R-AUDIT-REPORT.md)

- **§Methodology (Consolidated)** — 35 packages + versions (as of 2026-08), in/out-of-scope rule, D-01 verdict rubric, D-03 category rubric, and the **`Rfssa` inventory-completeness caveat** (functional SSA missed by the 35-package survey; actionable-gap total may under-count by a few `Rfssa`-specific capabilities; does not affect the table-stakes signal).
- **§Consolidated Findings** — headline table (35 pkgs · 275 caps · 248 in-scope · **162 actionable gaps** · 18 table-stakes / 144 differentiator · 88/250 present); gap counts by area + category; largest-gap zones (Area 7 Fréchet/object-data 0/25, Area 6 functional time series 2/25, Area 5 inference 0/25); the 12 R-honest strengths summary (headliners: model explainability for functional models, streaming depth).

## RPT-02 — R-BACKLOG.md (new file, 451 lines)

- `score = value/√effort` methodology (value 1–5, effort S/M/L → 1.0/1.732/3.0, severity P1/P2/P3), mirroring v0.14.0.
- **162 actionable gaps clustered into 26 GSD-ready items** (candidate requirements/phases), with a Gap-to-Item Coverage Map (all 18 table-stakes covered by 7 P1 items, all in the top 8).
- Master ranked table **strictly non-increasing** (5.00 → 0.67, 26 items), ties broken by severity; each item carries a full 7-field promotion block.
- Existing fdars strengths (Phase 18) excluded by construction.

### Top 8 by score
1. **T-01** Constant basis + AIC smoothing selection — 5.00 · P1 · table-stakes
2. **T-02** Depth-fence functional boxplot + depth dispatcher — 5.00 · P1 · table-stakes
3. **REG-03** Elastic multinomial regression — 3.00 · P2 · differentiator
4. **INF-01** Two-sample functional tests (t/F-perm, mean/cov, SCB) — 2.89 · P1 · table-stakes
5. **INF-02** FLM inference suite (GoF, F-test, ANOVA V-stat) — 2.89 · P1 · table-stakes
6. **REG-01** Concurrent/varying-coefficient regression — 2.89 · P1 · table-stakes
7. **REG-02** Functional GLM exponential families — 2.31 · P1 · table-stakes
8. **FPCA-01** Unified PACE sparse FPCA + conditional-expectation scores — 2.31 · P1 · table-stakes

## Completeness gate: PASS

Table strictly descending; every ranked item has a matching 7-field block; top items non-cosmetic (7 of top 8 are P1 table-stakes); all 18 table-stakes gaps sit in top-8 P1 items; no fdars-strength work proposed. `R-BACKLOG.md` is promotion-ready for the next `/gsd-new-milestone`.

## Verification

All 4 ROADMAP Phase-19 success criteria pass (see `19-VERIFICATION.md`). Zero `fdars-core/src/` edits; scikit-fda `AUDIT-REPORT.md`/`BACKLOG.md` untouched.
