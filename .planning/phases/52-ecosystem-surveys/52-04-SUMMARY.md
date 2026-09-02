---
phase: 52-ecosystem-surveys
plan: 04
requirements: [PYX-01]
status: complete
completed: 2026-09-02
deliverable: .planning/research/survey-pyx.md
net_new_gaps: 4
---

# Plan 52-04 SUMMARY — Python-beyond-scikit-fda Survey (PYX-01)

## What was produced
`.planning/research/survey-pyx.md` — a capability-first survey of Python FDA/functional-time-series libraries OTHER than scikit-fda (`FDApy@1.0.4`, `tslearn@0.9.0`, `sktime@0.3x`, `pyts@0.13.x`, tsfresh/catch22). scikit-fda is explicitly EXCLUDED (v0.14.0). Standardized columns + present/partial/absent mapping with grep evidence, FDA-relevance discipline applied.

## Net-new gaps found (4, incl. 1 partial slice)
- **Shapelet transform / shapelet-based classification** (tslearn/sktime/pyts) — interpretable local-shape primitives. Absent from `src/`, backlogs, R-AUDIT. Effort M–L.
- **k-Shape clustering (SBD)** (tslearn/sktime) — FFT cross-correlation shape clustering, distinct from fdars' SRVF clustering. Effort M.
- **Global Alignment Kernel (GAK)** (tslearn) — PSD kernel for kernel k-means/SVM on curves; fdars has soft-DTW *divergence* but not a PSD alignment *kernel*. Effort S–M (slice).
- **Multi-dimensional heterogeneous-domain MFPCA** (FDApy) — joint dimension reduction across mixed-domain components (curve + image); fdars `mfpca` is same-type only. Effort M (partial/missing-slice).

Plus flagged likely-out-of-scope: SAX/PAA/bag-of-patterns symbolic representations (TS-ML representations, not functional numeric methods) — recorded for RPT-03 triage.

## De-dup rigor
DTW/soft-DTW/DTW-barycenter (tslearn) and MFPCA/simulation (FDApy) are already present in fdars. scikit-fda entirely excluded. Generic TS-ML feature banks (tsfresh/catch22) out-of-scope; FDA-relevant feature APIs already backlogged (PREP-08).

## Reverse parity
fdars leads in depth/robustness, inference & regression breadth, elastic/SRVF shape analysis, explainability, determinism, and deployment.

## Audit-only fence
Zero `fdars-core/src/` edits. Only the survey markdown file was written.
