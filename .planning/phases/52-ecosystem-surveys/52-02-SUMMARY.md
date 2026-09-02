---
phase: 52-ecosystem-surveys
plan: 02
requirements: [JUL-01]
status: complete
completed: 2026-09-02
deliverable: .planning/research/survey-julia.md
net_new_gaps: 2
---

# Plan 52-02 SUMMARY — Julia FDA Survey (JUL-01)

## What was produced
`.planning/research/survey-julia.md` — a capability-first survey of the Julia FDA ecosystem (`ElasticFDA.jl@1.x`, `FDA.jl@0.x`, `MultivariateStats.jl@0.10.x`, `registr`) plus modern/performance-oriented Julia idioms (autodiff-through-FDA, GPU broadcast, generic APIs) captured as candidate gaps per JUL-01. Standardized columns + present/partial/absent mapping with grep evidence.

## Net-new gaps found (2)
- **Autodiff-compatible / differentiable FDA core** — gradients through warping/FPCA/regression via generic number types (Julia ForwardDiff/Zygote idiom); fdars is `f64`-concrete. Effort L (invasive generics refactor; likely a scoped differentiable-elastic-distance subset). Verified absent from `src/`, both backlogs, R-AUDIT.
- **GPU-friendly / batched-broadcast FDA kernels** — flagged as *likely out-of-scope* for a portable CPU/WASM numeric core; recorded honestly for RPT-03 triage.

## De-dup rigor
Julia's FDA *packages* are method-wise covered by or behind fdars (ElasticFDA.jl = SRVF, which fdars exceeds). registr's exponential-family curve registration was already surveyed in v0.18.0 R-AUDIT (already-considered). The net-new value is architectural.

## Reverse parity
fdars vastly exceeds any single Julia FDA package in breadth (classification, depth, SPM, conformal, seasonal, FTS, boosting, explainability) and in deployment.

## Audit-only fence
Zero `fdars-core/src/` edits. Only the survey markdown file was written.
