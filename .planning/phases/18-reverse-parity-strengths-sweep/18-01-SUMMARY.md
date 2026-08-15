---
phase: 18-reverse-parity-strengths-sweep
plan: "18-01"
status: complete
requirements: [GAP-03]
deliverable: .planning/research/R-AUDIT-REPORT.md
audit_only: true
completed: 2026-08-15
---

# Phase 18 · Plan 01 — Summary

**Delivered:** `.planning/research/R-AUDIT-REPORT.md` §Phase 18 — Reverse-Parity Strengths Sweep (GAP-03). Audit-only; zero `fdars-core/src/` edits.

## What shipped

- **Per-module coverage table** — every module unit in `fdars-core/src/` (42 units: 21 submodule groups + 21 top-level `.rs`; 6 pure-infra/re-export files excluded) with an explicit strength verdict → demonstrably exhaustive, not cherry-picked.
- **fdars-unique (no R equivalent) — 6:** U-3 **model explainability for functional models** (PDP/SHAP/LIME/ALE/importance/Sobol/counterfactual via `FpcPredictor`; R's DALEX/lime/pdp/iml are not integrated with functional-regression models) and U-5 **streaming/online functional depth** (`streaming_depth/`; all R depth packages are batch-only) are the headliners; plus U-1 Andrews-curve transform, U-2 elastic-model explain, U-4 WIRE workflow container, U-6 simultaneous tolerance bands.
- **fdars-ahead (leads closest R analog) — 6:** A-1 SPM chart breadth vs `funcharts`; A-2 conformal breadth (classification + elastic) vs `conformalInference.fd`; A-4 soft-DTW + barycenter vs `dtw`; A-5 robust L1/Huber scalar-on-function vs `fda.usc`; A-6 2D-surface FOSR vs `refund`; A-7 functional signal toolkit (period/matrix-profile/Hilbert/Lomb-Scargle) vs scalar-TS `tsmp`/`lomb`/`hht`.

## Honesty against R (the critical difference from v0.14.0)

R is far broader than scikit-fda, so the v0.14.0 "30 fdars-only-vs-scikit-fda" list collapses to **12 R-honest strengths**. Re-vet casualties (unique vs scikit-fda but NOT vs R): SPM (`funcharts`), conformal (`conformalInference.fd`), the entire elastic/shape stack (`fdasrvf` is broader — a Phase-17 gap, not a strength; A-3 deliberately skipped), functional SSA (`Rfssa`), and FoF/1D-FOSR (`refund`/`FDboost`). fdars' genuine R-relative moat is **explainability for functional models** and **streaming depth**.

## Completeness note for Phase 19

The sweep surfaced **`Rfssa`** (functional Singular Spectrum Analysis, CRAN) which was NOT in the Phase-16 35-package survey. It was recorded honestly as a re-vet casualty (SSA is present in both fdars and R). Phase 19 should note this as a minor Phase-16 inventory-completeness caveat.

## Verification

All 3 ROADMAP Phase-18 success criteria pass (see `18-VERIFICATION.md`). Committed `6e0ede5c`.
