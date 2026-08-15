---
phase: 17-parity-matrix-categorization
plan: "17-01"
status: complete
requirements: [GAP-01, GAP-02]
deliverable: .planning/research/R-AUDIT-REPORT.md
audit_only: true
completed: 2026-08-15
---

# Phase 17 · Plan 01 — Summary

**Delivered:** `.planning/research/R-AUDIT-REPORT.md` §Phase 17 — Parity Matrix & Categorization (GAP-01 + GAP-02), +430 lines. Audit-only; zero `fdars-core/src/` edits.

## What shipped

- **§Parity-Matrix (GAP-01)** — 9 per-area verdict tables mapping every in-scope R capability to fdars. **250 literal in-scope rows** mapped (reconciled from the Phase-16 header's 248: Areas 3/6/7 +1 each, Area 4 −2 header over-count; recount note documented). Each row: capability · source package(s) · verdict (present/partial/absent, D-01 rubric) · "searched fdars for:" evidence note · closest-match `module::function` or "no match found". Matched by capability semantics, confirmed by grepping `fdars-core/src/`.
- **Verdict counts** — per-area + overall table: **present 88 · partial 49 · absent 113 · 250 rows**. **Headline actionable-gap count (in-scope, partial+absent): 162** (49 "add-a-variant" partials + 113 "implement-from-scratch" absents). Present = 35%.
- **§Categorization (GAP-02)** — every actionable gap categorized (D-03 rubric): **table-stakes 18 · differentiator 144 · out-of-scope 0**. 89% of gaps are specialized differentiators.

## Key findings (Phase-19 input signal)

- **Inference (Area 5) is the dominant table-stakes deficit** — fdars has 0 present in this area; 8 table-stakes gaps (two-sample permutation/SCB tests, FLM goodness-of-fit + F-test, one-way ANOVA V-stat, mean/covariance equality).
- **Largest gap zones:** Area 7 Density/Object-Data (0 present — no Fréchet/metric-space regression, LQD density-FPCA, multi-domain MFPCA), Area 6 Functional Time Series (2/25 — no FTS forecasting, functional ACF, spectral DPCA, FARMA), Area 5 Inference (0 present).
- **Strongest fdars areas:** SPM 9/10, Preprocessing/Registration 16/22, Representation 20/38.
- Other table-stakes gaps: ML concurrent/varying-coefficient regression + GLM families (4), PACE sparse FPCA + conditional-expectation scores (2), constant basis + AIC smoothing selection (2), depth dispatcher + depth-fence boxplot (2).
- All v0.15.0–v0.17.0 fdars additions credited **present**.

## Verification

All 4 ROADMAP Phase-17 success criteria pass (see `17-VERIFICATION.md`). Counts internally consistent (88+49+113=250; 49+113=162; 18+144=162). Zero `fdars-core/src/` edits; Phase-16 sections + archived scikit-fda `AUDIT-REPORT.md` untouched. Committed `2a1f2775`.

## Notes / deviations

- Executed by a codebase-grep-enabled analysis agent (opus); orchestrator verified output against plan must-haves + all ROADMAP SCs and performed phase bookkeeping inline (background-agent process-exit instability this session).
