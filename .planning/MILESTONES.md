# Milestones

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
