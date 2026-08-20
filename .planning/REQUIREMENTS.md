# Requirements: fdars — v0.23.0 Depth, Outliers & Interval Inference

**Defined:** 2026-08-19
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability and performance gaps against the reference FDA ecosystems — this milestone closes the top three P2 differentiator gaps from the v0.18.0 `R-BACKLOG.md` (score 2.31 each): the depth-measure long tail, the robust outlier-detector suite, and the Interval Testing Procedure family.

Milestone-level conventions (carried from v0.19.0–v0.22.0, apply to every requirement below):

- Real `fdars-core/src/` code — **additive/non-breaking**, `Result<T, FdarError>`-returning, inline `#[cfg(test)]` tests, crate-root re-exports; **zero changes to existing public signatures.**
- **Reuse-first** — extend `depth/` / `outliers.rs` / `inference/` + `basis/`; no new algorithm subsystem, **no new crate dependency.**
- R baselines matched by **capability**, not R's exact signatures. Plotting/rendering is **out of scope** (numeric outputs only).

## v1 Requirements

Requirements for milestone v0.23.0. Each maps to exactly one roadmap phase.

### Depth Measures

- [ ] **DEPTH-01**: Add the missing univariate functional depth measures to `depth/` — half-region depth (HRD) and modified half-region depth (MHRD), hypograph/modified-hypograph indices (HI/MHI) and the un-modified epigraph index (EI), extremal depth, extreme-rank-length depth (ERL), L∞ depth, and total-variation depth with MSSI. Each is a `Result`-returning function over the column-major `FdMatrix` following the existing per-file convention, and each is registered in the T-02 `DepthMethod` dispatcher. Excludes streaming depth (batch measures only). R baseline: `roahd` / `fdaoutlier`.

### Outlier Detection

- [ ] **OUT-01**: Add the outlier-detection algorithms to `outliers.rs` — `tvdmss` (total-variation-depth + MSSI detector, reusing DEPTH-01's TVD+MSSI), `muod` (Massive Unsupervised Outlier Detection), `sequential_transform_outliers` (sequential-transformation detection), and the `depthgram` statistic (numeric outputs; renderer out-of-scope). Reuses the existing MS-plot / outliergram machinery. R baseline: `fdaoutlier` / `roahd`. **Depends on DEPTH-01.**

### Functional Inference

- [x] **INF-03**: Implement the Interval Testing Procedure (ITP) family in new `inference/itp.rs` — one-population and two-population interval-wise tests (B-spline and Fourier bases) with domain-selective adjusted p-values, plus interval-wise FLM coefficient testing. Reuses the INF-01 permutation infrastructure and `basis/` projection. Independent of DEPTH-01/OUT-01. R baseline: `fdatest`.

## v2 Requirements

Deferred to future milestones (from `.planning/research/R-BACKLOG.md`, ordered by score). Not in this roadmap.

### Regression (differentiators)

- **REG-04**: Additive / GKAM / GSAM functional regression + variable selection (score 1.73)
- **REG-05**: Flexible mixed-effects regression (denseFLMM, multiFAMM, fastFMM, pffr) (score 1.73)

### Clustering / Representation / Time Series (differentiators)

- **CLUS-01**: Model-based / density functional clustering (funHDDC, funFEM, DBSCAN, kCFC) (score 1.73)
- **REP-01**: Basis-system completions (monomial, exponential, power, polygonal, multi-domain) (score 1.73)
- **FTS-02**: Functional ACF/PACF + stationarity + long-run covariance (score 1.73)

### Specialized (P3 differentiators & large items)

- **DENS-01**, **FPCA-02**, **SPARSE-01** (score 1.73, P3); **FTS-01**, **FRE-01** (score 1.33, L); **FTS-03**, **FRE-02**, **REG-06**, **REP-02**, **CLUS-02** (score ≤ 1.00, L)

## Out of Scope

Explicitly excluded. Documented to prevent scope creep.

| Feature | Reason |
|---------|--------|
| Plotting / rendering of depth regions, functional boxplots, depthgram, outlier flags, or ITP p-value surfaces | Numeric Rust library — numeric statistics are in scope, the plots are not (consistent with the R-audit plotting exclusion) |
| Streaming / online depth variants of the DEPTH-01 measures | fdars strength U-5 already covers streaming depth; DEPTH-01 is batch measures only |
| `fdaPOIFD` partially-observed depth outlier detectors | Adjacent to OUT-01 but a distinct partially-observed-data capability — deferred |
| Random-projection ANOVA/MANOVA (`fdANOVA`) | Adjacent to INF-03's ITP family but a separate method — deferred |
| New crate dependencies | All three items are reuse-first over existing infrastructure; no new dependency permitted |
| Changes to existing public signatures | Milestone is strictly additive/non-breaking |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DEPTH-01 | Phase 28 | Pending |
| OUT-01 | Phase 29 | Pending |
| INF-03 | Phase 30 | Complete |

**Coverage:**

- v1 requirements: 3 total
- Mapped to phases: 3
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-19*
*Last updated: 2026-08-19 after initial definition*
