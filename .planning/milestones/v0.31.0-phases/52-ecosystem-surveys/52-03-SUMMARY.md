---
phase: 52-ecosystem-surveys
plan: 03
requirements: [TDY-01]
status: complete
completed: 2026-09-02
deliverable: .planning/research/survey-tidyfun.md
net_new_gaps: 2
---

# Plan 52-03 SUMMARY — tidyfun / refund Survey (TDY-01)

## What was produced
`.planning/research/survey-tidyfun.md` — a capability-first survey of the R tidyfun data-representation/workflow slice (`tf@0.3.x`, `tidyfun@0.x`) and refund methods NOT already captured in v0.18.0 (`refund@0.1-38`). Standardized columns + present/partial/absent mapping with grep evidence. The v0.18.0 refund boundary is stated explicitly (excluded methods enumerated).

## Net-new gaps found (2)
- **PEER / longitudinal PEER (`peer`/`lpeer`)** — structured a-priori-penalty scalar-on-function regression. Verified absent from `src/`, both backlogs, and R-AUDIT. Effort M.
- **Wavelet-domain functional regression (`wcr`/`wnet`)** — wavelet compression + lasso/elastic-net for spiky/localized functional predictors. Verified absent (fdars has `rustfft` but no DWT). Effort M–L.

## v0.18.0 refund exclusions (already-captured, out of scope)
`pffr`/`pfr`, `fosr`/`fosr2s`/`bayes_fosr`, `fgam`/GKAM/GSAM, mixed-effects (`denseFLMM`/`multiFAMM`/`fastFMM`), `fbps`/sandwich smoother + `fpca.face`/`fpca.sc`/`fpca.ssvd`/`fpca2s` (R-BACKLOG fbps hits=11), boosting/Bayesian FOSR.

## De-dup rigor
tidyfun's `tf` layer is data-representation & tidyverse ergonomics whose numeric ops fdars already provides; reshaping/plotting fall under the data-IO and visualization scope fences.

## Reverse parity
fdars leads in numeric depth/breadth (elastic, depth, SPM, conformal, explainability), performance, and deployment; it is the numeric substrate under tidyfun's `tfd`/`tfb`.

## Audit-only fence
Zero `fdars-core/src/` edits. Only the survey markdown file was written.
