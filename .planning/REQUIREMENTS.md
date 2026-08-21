# Requirements: fdars — Milestone v0.26.0 FPCA Breadth & Sparse Covariance

**Defined:** 2026-08-21
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone draws the two remaining top-ranked items from the v0.18.0 `R-BACKLOG.md` (score 1.73, M-effort): specialized FPCA variants (FPCA-02) and fast sparse/irregular covariance (SPARSE-01).

**Source:** [`.planning/research/R-BACKLOG.md`](research/R-BACKLOG.md) items FPCA-02 (rank 18) and SPARSE-01 (rank 19).

**Milestone constraints (apply to every requirement):** additive/non-breaking (zero changes to existing public signatures); reuse-first (no new algorithm subsystem); all public functions `Result<T, FdarError>`-returning; inline `#[cfg(test)]` tests with hand-computed/reference checks + error paths; crate-root re-exports; **no new crate dependency**; numeric outputs only (plotting/rendering out of scope); `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean.

## v1 Requirements

Requirements for milestone v0.26.0. Each maps to a roadmap phase.

### FPCA-02 — Specialized FPCA variants

R baseline: `fdapace` (FPCAder, FSVD, GetCrCov, DynCorr, FCCor), `refund` (fpca.sc sandwich, fpca.ssvd). Extends `regression.rs` (or a new `fpca_variants.rs`), reusing the dense FPCA (`fdata_to_pc_1d`) + `covariance.rs`.

- [x] **FPCA-02-01**: User can compute FPCA of curve derivatives (`fpca_der`) — eigenfunction/score decomposition of the differentiated process, returning derivative loadings and scores.
- [x] **FPCA-02-02**: User can compute a functional SVD / cross-FPCA (`fsvd`) between two functional samples — the bivariate singular-value decomposition yielding paired left/right singular functions and singular values.
- [x] **FPCA-02-03**: User can estimate a cross-covariance surface (`cross_covariance`) between two functional samples over their argument grids.
- [x] **FPCA-02-04**: User can compute dynamical / functional correlation (`dynamical_correlation`) between two functional samples as a scalar association measure.
- [x] **FPCA-02-05**: User can run a sandwich-smoother / sparse-SVD (ssvd) FPCA path that estimates loadings/scores via a smoothed-covariance (sandwich) estimator as an alternative to the raw thin-SVD decomposition.

### SPARSE-01 — Sparse/irregular fast covariance + trajectory bands

R baseline: `face` (FACE), `mfaces` (multivariate FACE), `fdapace` (trajectory bands). Extends `irreg_fdata/`, building on `cov_irreg`, and integrates with the shipped PACE `pace_fpca` (FPCA-01).

- [x] **SPARSE-01-01**: User can estimate a sparse-data covariance surface via the FACE fast-sandwich smoother (`face_covariance`) over irregular/sparse functional data.
- [x] **SPARSE-01-02**: User can estimate a multivariate sparse covariance via the `mfaces` extension (`mface_covariance`) for multiple simultaneously-observed sparse functional variables.
- [x] **SPARSE-01-03**: User can obtain fitted continuous trajectories with pointwise confidence bands for sparse curves, integrated with the FACE covariance path (and reusing `pace_fpca` machinery where applicable).

## Future Requirements

Deferred to future milestones (tracked in `R-BACKLOG.md`, not in this roadmap).

### Functional Time Series (1.33 L-effort tier)

- **FTS-01**: Functional time series forecasting (ftsm, FPC-score regression, fplsr, dynamic updating) — depends on the shipped FTS-02.

### Object Data (1.33 L-effort tier)

- **FRE-01**: Fréchet / object-data regression + statistics (global/local Fréchet regression, mean/variance/ANOVA, Wasserstein distance) — shares DENS-01's Wasserstein/quantile machinery.

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| New crate dependency for FACE/SVD/covariance | Milestone constraint — reuse existing nalgebra/faer + `covariance.rs`/`irreg_fdata` machinery; adding a dep triggers a package-legitimacy review |
| Plotting/visualization of FPCA loadings, cross-covariance surfaces, or trajectory bands | Numeric Rust library — renderer stays out of scope (consistent with the v0.14.0/v0.18.0 audit fence); only numeric outputs are delivered |
| Changes to existing public signatures (`fdata_to_pc_1d`, `pace_fpca`, `cov_irreg`, …) | Additive/non-breaking constraint — new functions only; existing paths preserved bit-for-bit |
| General object-space Fréchet machinery (FRE-01/FRE-02) | Separate L-effort backlog items; this milestone is the 1.73-tier FPCA/covariance cluster only |
| FTS forecasting subsystem (FTS-01) | Separate L-effort backlog item; deferred to a later milestone |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| FPCA-02-01 | Phase 37 | Complete |
| FPCA-02-02 | Phase 37 | Complete |
| FPCA-02-03 | Phase 37 | Complete |
| FPCA-02-04 | Phase 37 | Complete |
| FPCA-02-05 | Phase 37 | Complete |
| SPARSE-01-01 | Phase 38 | Complete |
| SPARSE-01-02 | Phase 38 | Complete |
| SPARSE-01-03 | Phase 38 | Complete |

**Coverage:**
- v1 requirements: 8 total
- Mapped to phases: 8 (100% — confirmed by roadmapper)
- Unmapped: 0

**Phase mapping:**
- Phase 37 (FPCA-02): FPCA-02-01, FPCA-02-02, FPCA-02-03, FPCA-02-04, FPCA-02-05
- Phase 38 (SPARSE-01): SPARSE-01-01, SPARSE-01-02, SPARSE-01-03

---
*Requirements defined: 2026-08-21*
*Last updated: 2026-08-21 — traceability confirmed by roadmapper (Phases 37–38, 8/8 mapped)*
