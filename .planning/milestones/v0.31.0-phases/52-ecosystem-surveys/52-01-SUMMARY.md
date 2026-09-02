---
phase: 52-ecosystem-surveys
plan: 01
requirements: [MAT-01]
status: complete
completed: 2026-09-02
deliverable: .planning/research/survey-matlab.md
net_new_gaps: 1
---

# Plan 52-01 SUMMARY — MATLAB FDA Survey (MAT-01)

## What was produced
`.planning/research/survey-matlab.md` — a capability-first survey of the MATLAB FDA ecosystem: Ramsay's `fda` MATLAB toolbox (`fdaM@6.x`) and PACE (MATLAB) (`PACE@2.17`), each version-pinned as of 2026-09. Every capability row carries an fdars present/partial/absent status with an explicit "searched fdars for:" grep-evidence note (mapped by capability, not API name), and a Net-New Gap List using the six standardized columns.

## Net-new gaps found (1)
- **Optimal experimental design for sparse FDA (PACE `FOptDes`)** — choose measurement locations minimizing FPCA prediction MSE under a sparse-sampling budget. Verified absent from `fdars-core/src/`, `BACKLOG.md`, `R-BACKLOG.md`, and the v0.18.0 `R-AUDIT-REPORT.md`. Effort M (builds on existing `pace_fpca` covariance machinery).

## De-dup rigor
Most of the MATLAB surface is covered by fdars or already tracked in prior backlogs. PACE methods already surveyed in the v0.18.0 fdapace audit (FVPA, stringing, empirical dynamics, FSVD) were excluded as already-considered. Missing basis families and functional-GLM are already in `BACKLOG.md`/`R-BACKLOG.md`.

## Reverse parity
fdars leads MATLAB FDA in elastic/SRVF shape analysis, explainability, conformal prediction, SPM, streaming depth, and deployment (WASM/JS + R).

## Audit-only fence
Zero `fdars-core/src/` edits. Only the survey markdown file was written. Verified: `git status --porcelain fdars-core/src/` empty.
