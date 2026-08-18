# Requirements: fdars — v0.22.0 PACE Sparse FPCA & Elastic Multinomial

**Defined:** 2026-08-18
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone ships the final P1 table-stakes item (FPCA-01, unified PACE sparse FPCA) and completes fdars' elastic-regression family (REG-03, elastic multinomial), each by orchestrating/extending existing `fdars-core/src/` code.

## v1 Requirements

Requirements for this milestone. Each maps to exactly one roadmap phase. IDs match the `R-BACKLOG.md` item IDs (project convention: backlog ID = REQ-ID).

### Sparse FPCA

- [ ] **FPCA-01**: User can fit a **unified PACE sparse FPCA** for sparse / irregularly-sampled functional data via a new public entry point in `fdars-core/src/pace_fpca.rs` (re-exported at the crate root). The estimator chains existing pieces into one call: a kernel-smoothed mean, a smoothed covariance surface (`irreg_fdata::cov_irreg`), its eigendecomposition (eigenvalues + eigenfunctions), **conditional-expectation (BLUP / PACE) FPC scores** per curve, and **fitted continuous trajectories with pointwise confidence bands**. It is `Result`-returning, validates its inputs (returning `FdarError` rather than panicking on empty/degenerate/mismatched input), reuses `irreg_fdata` + `spm::partial::conditional_expectation` + `regression` machinery, adds no new crate dependency, and leaves existing FPCA APIs unchanged (additive/non-breaking).

### Elastic Regression

- [ ] **REG-03**: User can fit an **elastic multinomial (multi-class) logistic regression** over SRSF/SRVF space via a new public entry point in `fdars-core/src/elastic_regression/logistic.rs` (re-exported at the crate root), extending the existing binary elastic logistic to K ≥ 2 classes (one-vs-rest or softmax), plus a `predict_elastic_multinomial` companion that returns class probabilities / predicted labels for new curves. It is `Result`-returning, reuses the existing SRVF representation + warping machinery, validates inputs (returning `FdarError`, no panic), adds no new crate dependency, and retains the existing binary `elastic_logistic` public signature unchanged (additive/non-breaking).

## v2 Requirements

Deferred to future milestones. Tracked in `.planning/research/R-BACKLOG.md`, not in this roadmap. After this milestone the P1 table-stakes tier is exhausted; the remaining backlog is all P2/P3 differentiators.

### Highest-ranked remaining differentiators

- **DEPTH-01** (rank 9): depth-measure long tail (HRD/MHRD, HI/MHI, extremal, ERL, L∞, TVD+MSSI).
- **OUT-01** (rank 10): outlier-detector suite (tvdmss, MUOD, sequential-transform, depthgram).
- **INF-03** (rank 11): Interval Testing Procedure (ITP) family (1-/2-sample, FLM coefficient).
- **REG-04 / REG-05 / REG-06**: additive/GKAM/GSAM, flexible mixed-effects, boosting/Bayesian functional regression.
- **FTS-\* / FRE-\* / DENS-\* / CLUS-\* / REP-\* / SPARSE-\* / FPCA-02**: functional time series, Fréchet/object-data regression, density object-data FPCA, model-based clustering, basis-system completions, sparse fast covariance, specialized FPCA variants.

### Enabled by this milestone

- **REG-01 sparse/PACE variant**: the kernel-weighted sparse concurrent-regression path deferred from v0.21.0 — now unblocked by FPCA-01's PACE infrastructure.

## Out of Scope

Explicitly excluded this milestone, with reasoning.

| Feature | Reason |
|---------|--------|
| REG-01 sparse/PACE concurrent-regression variant | Now *enabled* by FPCA-01 but is a distinct capability — deferred to a future milestone to keep v0.22.0 scoped to the two backlog items chosen. |
| Configurable / non-canonical PACE bandwidth-selection subsystem | FPCA-01 reuses existing smoothing bandwidth machinery; a new GCV/CV bandwidth-selection layer for the covariance surface is out of scope (use existing defaults / caller-supplied bandwidth). |
| Elastic multinomial beyond logistic (e.g. multinomial elastic PCR, ordinal) | REG-03 closes the single elastic-logistic multi-class partial; other elastic-family multiclass extensions are not baseline-expected. |
| Plotting / visualization of FPCA trajectories, bands, or class boundaries | Consistent with the project-wide numeric-library fence (R-audit plotting exclusion). |
| Crate version bump + release (PR + tag) | Operator-driven ship-time step, decoupled from implementation phases per the milestone-complete convention. |

## Traceability

Each requirement maps to exactly one phase (filled during roadmap creation).

| Requirement | Phase | Status |
|-------------|-------|--------|
| FPCA-01 | Phase 26 | Pending |
| REG-03 | Phase 27 | Pending |

**Coverage:**
- v1 requirements: 2 total
- Mapped to phases: 2 (pending roadmap)
- Unmapped: 0

---
*Requirements defined: 2026-08-18*
