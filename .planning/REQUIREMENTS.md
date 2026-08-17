# Requirements: fdars — v0.21.0 Functional Regression Completeness

**Defined:** 2026-08-17
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone ships the two remaining P1 table-stakes functional-regression items from `.planning/research/R-BACKLOG.md`, each reusing existing scalar-on-function design machinery.

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase. IDs match the `R-BACKLOG.md` item IDs (project convention: backlog ID = REQ-ID).

### Concurrent Regression

- [x] **REG-01**: User can fit a **dense functional concurrent (varying-coefficient) regression** — a model relating a functional response to one or more functional predictors evaluated at the *same* shared argument, estimating a time-varying coefficient β(t) — via a new public entry point in `fdars-core/src/concurrent_regression.rs`. β(t) is estimated by pointwise / local-linear least squares over the shared dense grid with a roughness (smoothing) penalty, reusing `smoothing.rs` kernels. The result carries `{ beta_curve, fitted, residuals }`, is `Result`-returning, and is re-exported at the crate root. Existing regression APIs are untouched (additive/non-breaking). **COMPLETED 2026-08-17 — commit 5480ee25**

### Functional GLM

- [ ] **REG-02**: User can fit a **functional GLM for a scalar response** over functional predictors via `functional_glm(data, y, family)` — an IRLS estimator over FPC/basis scores (reusing the `functional_logistic` IRLS loop + `fdata_to_pc_1d`) — with a `GlmFamily { Binomial, Poisson, Gamma, Gaussian }` enum, each carrying its canonical link and variance function. The existing `functional_logistic` public signature is retained unchanged (the `Binomial` family reproduces it); the result type is `Result`-returning and re-exported at the crate root.

## v2 Requirements

Deferred to future milestones. Tracked in `.planning/research/R-BACKLOG.md`, not in this roadmap.

### Regression / ML

- **REG-01 (sparse path)**: kernel-weighted sparse/PACE variant of concurrent regression — deferred; benefits from FPCA-01's PACE infrastructure (not yet built).
- **REG-03**: Elastic multinomial regression + robust-family completions (rank 3, score 3.00, S-effort).
- **REG-04**: Additive / GKAM / GSAM functional regression + variable selection.
- **REG-05**: Flexible mixed-effects regression (denseFLMM, multiFAMM, fastFMM, pffr).
- **REG-06**: Boosting / Bayesian functional regression.

### Other high-leverage backlog

- **FPCA-01**: Unified PACE sparse FPCA + conditional-expectation scores (P1 table-stakes, M).
- **INF-03**: Interval Testing Procedure (ITP) family (deferred from v0.19.0).
- **DEPTH-01 / OUT-01**: Depth-measure long tail + outlier-detector suite.

## Out of Scope

Explicitly excluded this milestone, with reasoning.

| Feature | Reason |
|---------|--------|
| Sparse/PACE concurrent-regression variant | Requires PACE covariance + conditional-expectation infra (FPCA-01), not yet built; dense variant is self-contained and closes the table-stakes gap. Deferred to when FPCA-01 lands. |
| Extra GLM families (inverse-Gaussian, negative-binomial) + configurable links | Binomial/Poisson/Gamma/Gaussian cover the mainstream exponential-family expectations and close the R-parity table-stakes gap; broader families add validation surface for niche use. |
| Changing / deprecating `functional_logistic` | Additive/non-breaking mandate — the existing binary-logistic API stays; `functional_glm` generalizes without replacing it. |
| Plotting / visualization of fitted β(t) or GLM diagnostics | Consistent with the project-wide numeric-library fence (R-audit plotting exclusion). |
| Crate version bump + release (PR + tag) | Ship-time step, decoupled from implementation phases per the milestone-complete convention. |

## Traceability

Each requirement maps to exactly one phase.

| Requirement | Phase | Status |
|-------------|-------|--------|
| REG-01 | Phase 24 | Complete (2026-08-17) |
| REG-02 | Phase 25 | Pending |

**Coverage:**
- v1 requirements: 2 total
- Mapped to phases: 2 ✓
- Unmapped: 0

---
*Requirements defined: 2026-08-17*
*Last updated: 2026-08-17 — REG-01 complete (Phase 24, commit 5480ee25); REG-02 pending (Phase 25)*
