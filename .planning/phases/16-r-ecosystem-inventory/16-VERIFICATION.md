---
phase: 16-r-ecosystem-inventory
status: passed
verified: 2026-08-14
requirements: [INV-01, INV-02]
plans: ["16-01", "16-02"]
must_haves_verified: 4
must_haves_total: 4
audit_only: true
---

# Phase 16 — R Ecosystem Inventory · Verification

**Verdict: PASSED** — 4/4 ROADMAP success criteria satisfied; both requirements (INV-01, INV-02) delivered. Audit-only: zero `fdars-core/src/` edits.

**Deliverable:** `.planning/research/R-AUDIT-REPORT.md` §Phase 16 (588 lines), distinct from the archived scikit-fda `AUDIT-REPORT.md` (untouched).

## Success Criteria (goal-backward)

| # | Criterion | Verdict | Evidence |
|---|-----------|---------|----------|
| SC1 | §R-Inventory lists every capability capability-first, tagged with source package **and** version, across the core ecosystem + surfaced packages | ✅ | 326 versioned package tags (`` `pkg` (vX.Y.Z) ``); all core packages present (fda 6.3.0, fda.usc 2.2.0, refund 0.1-40, fdapace 0.6.0, roahd 1.4.3, fdaoutlier 0.2.1, ftsa 6.7, MFPCA 1.3-11, funData 1.3-9, fdasrvf 2.4.4, fdatest 2.1.1, fdANOVA 0.1.2, frechet 0.3.0, fdadensity 0.1.4, funHDDC 2.3.1.1, FDboost 1.1-4) + 19 surfaced (funcharts, fdacluster, registr, fdaPDE, face, elasdics, freqdom, …). fit/predict/transform collapsed per capability. |
| SC2 | Capabilities grouped into named areas with a per-area count | ✅ | 9 `### Area` headers, each with an `In-scope count: N   Out-of-scope count: M` line (9/9). |
| SC3 | Every capability carries an in-/out-of-scope tag, with the rule documented once | ✅ | Rule stated once in §Methodology (4 relevance classes + "numeric-underpinning-of-a-plot is in-scope" clarification); every table row carries a Relevance tag. |
| SC4 | Per-area in-scope vs out-of-scope count table yielding the actionable surface | ✅ | §Design-Goal Filter table: 9 areas + TOTAL = **248 in / 27 out (24 plotting + 3 IO) / 275**; "Actionable in-scope capabilities for Phase 17 parity mapping: 248". |

## Integrity checks

- Deliverable file `.planning/research/R-AUDIT-REPORT.md` exists; scikit-fda `AUDIT-REPORT.md` unmodified (separate files confirmed). ✅
- Zero `fdars-core/src/` changes (`git status fdars-core/` clean). ✅
- Stale draft figures (210+/181) excluded; authoritative totals (275/248/27) used. ✅
- Arithmetic correction applied: TOTAL out-of-scope breakdown "24 plotting, 3 IO" (was "25 plotting, 3 IO"). ✅

## Notes

- Survey (RESEARCH.md) produced by a web-enabled `gsd-phase-researcher` (CRAN versions verified 2026-08-14). Consolidation + bookkeeping performed inline by the orchestrator after repeated background-agent losses on process exit; output independently verified against plan must-haves and all ROADMAP success criteria.
- Nyquist VALIDATION.md not produced (no `## Validation Architecture` in RESEARCH — expected for an audit phase; consistent with prior audit-milestone posture).
