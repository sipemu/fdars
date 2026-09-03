# Requirements: fdars — v0.36.0 PEER (Structured-Penalty & Longitudinal Scalar-on-Function Regression)

**Defined:** 2026-09-03
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against the reference FDA ecosystems — driven by the ranked `GAP-BACKLOG.md`, top items first.
**Promotes:** GAP-06 (score 2.12, M-effort) from `.planning/research/GAP-BACKLOG.md`. Reference baseline: refund@0.1-38 (`peer`, `lpeer`).

## v0.36.0 Requirements

Implementation milestone — real `fdars-core/src/` code, additive/non-breaking (protects R + WASM bindings + 28 examples), no new crate dependency. Each requirement maps to a roadmap phase (Phase 66+).

### PEER

- [ ] **PER-01**: User can fit structured-penalty scalar-on-function regression via a public `peer(...)` estimator that estimates the coefficient function β(t) using the partially-empirical-eigenvector decomposition (null-space + range-space of the penalty operator), returning a result struct with β(t), intercept, fitted values, and the selected/df diagnostics — distinct from plain FPCR/`pfr`.
- [ ] **PER-02**: User can choose the a-priori penalty structure from three families via a penalty-type parameter/enum: ridge/identity (classical), 2nd-difference/roughness (reusing the existing `penalty_matrix` builder), and a caller-supplied structured/"decree" Q matrix (partitioned-domain a-priori structure — PEER's signature feature).
- [ ] **PER-03**: User can have the smoothing parameter λ chosen automatically by either GCV grid search (reusing the `penalized_solve` + GCV pattern) or REML/mixed-model estimation (reusing `famm`), selectable via config; an explicit λ is honored when supplied.
- [ ] **PER-04**: User can fit longitudinal PEER via a public `lpeer(...)` estimator that extends PEER to repeated per-subject measurements with subject-level random effects, fitted through `famm::fit_scalar_mixed_model` (REML EM), returning a result struct with the (time-varying) coefficient function and variance components.
- [ ] **PER-05**: User can predict on new curves out-of-sample from a fitted PEER/lpeer result (`predict`), access the coefficient function and fitted values, and reach the API from the crate root + prelude; a module doctest demonstrates the end-to-end workflow.

## Future Requirements

Deferred to a later milestone — remaining `GAP-BACKLOG.md` items. Tracked, not in this roadmap.

### Wavelet-domain functional regression (GAP-07)

- **WAV-01**: DWT + regularized (ridge/lasso/elastic-net) scalar-on-function regression in the wavelet domain (`wcr`/`wnet`). Effort L (score 1.73) — requires building a discrete wavelet transform first. Reference refund@0.1-38.

### Autodiff-compatible / differentiable FDA core (GAP-08)

- **DIF-01**: A scoped differentiable subset (e.g. differentiable elastic distance / FPCA) exposing gradients for embedding in larger optimization/ML pipelines. Effort L (score 1.73), invasive generics refactor. Julia idiom (ElasticFDA.jl + Zygote).

## Out of Scope

Explicitly excluded this milestone. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| GAP-07 wavelet regression / GAP-08 differentiable core | Lower-ranked `GAP-BACKLOG.md` items (score 1.73); carry forward to a future milestone. |
| Breaking API changes / removing existing public signatures | Additive/non-breaking convention protects R + WASM bindings and 28 examples. |
| New crate dependency | Carrying the no-new-dependency convention; PEER/lpeer reuse existing `function_on_scalar`, `smooth_basis`, `famm`, `linalg` machinery. |
| Basis-expansion PEER matching refund's exact internal representation | fdars models the coefficient function on the FPC/penalty basis it already uses; documented divergences from refund's `pentype`/basis internals are acceptable (numeric-output parity, not implementation parity). |
| Non-Gaussian / GLM PEER families | refund's core `peer`/`lpeer` are Gaussian-response; exponential-family extension is a possible future item, not this milestone. |

## Traceability

Which phases cover which requirements. Filled during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| PER-01 | TBD | Pending |
| PER-02 | TBD | Pending |
| PER-03 | TBD | Pending |
| PER-04 | TBD | Pending |
| PER-05 | TBD | Pending |

**Coverage:**
- v0.36.0 requirements: 5 total
- Mapped to phases: 0 (roadmap pending)
- Unmapped: 5 ⚠️ (resolved at roadmap creation)

---
*Requirements defined: 2026-09-03*
*Last updated: 2026-09-03 after initial definition (v0.36.0 PEER milestone)*
