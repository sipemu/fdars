# Requirements: fdars — v0.20.0 Table-Stakes Quick Wins

**Defined:** 2026-08-16
**Core Value:** A comprehensive, fast Rust functional-data-analysis library that closes the highest-leverage capability gaps against the reference FDA ecosystems — this milestone ships the two top-ranked (score 5.00) R-parity quick wins from `.planning/research/R-BACKLOG.md`.

Implementation milestone — real `fdars-core/src/` code. Both items are S-effort and **wrap existing infrastructure** (low risk); additive/non-breaking, `Result`-returning, inline `#[cfg(test)]` tests, crate-root re-exports.

## v1 Requirements

### Representation & Smoothing

- [ ] **T-01**: Add a named **constant/intercept basis** constructor to `basis/` (a basis object usable in regression design matrices, not just an ad-hoc constant column), AND add an **AIC criterion** to the automatic smoothing-parameter selector so `smooth_basis` / the roughness-penalty selection can choose by AIC as well as the existing GCV/CV. Reuses the existing basis system + hat-matrix trace already computed for GCV. (R baseline: `fda`/`fda.usc` constant basis + `akaike_information_criterion` smoothing.)

### Exploratory & Depth

- [ ] **T-02**: Add the López-Pintado **depth-fence functional boxplot** — central region (inner 50% by depth) + a 1.5×IQR-of-depths whisker + per-curve outlier flags, all as **numeric outputs** (not a plot) — AND a unified **`functional_depth(data, method: DepthMethod)`** dispatcher over the existing depth functions (`fraiman_muniz_1d`, `band_1d`, `modified_band_1d`, `random_projection_1d`, …). (R baseline: `roahd`/`fdaoutlier`/`fda.usc` functional boxplot fences + general depth dispatcher.)

## v2 Requirements

Deferred to future milestones (from `.planning/research/R-BACKLOG.md`).

### Regression

- **REG-01** (2.89, P1, M) — concurrent / varying-coefficient functional regression.
- **REG-02** (2.31, P1, M) — functional GLM exponential-family families (Poisson, Gamma, …).
- **REG-03** (3.00, P2, S) — elastic multinomial regression + robust-family completions.

### Inference

- **INF-03** (2.31, P2, M) — Interval Testing Procedure (ITP) family, extending the `inference/` module shipped in v0.19.0.

### Larger clusters

- **FPCA-01** (unified PACE sparse FPCA), **FTS-01/02/03** (functional time series), **FRE-01/02** (Fréchet/object-data regression) — ranked in `R-BACKLOG.md`.

## Out of Scope

| Feature | Reason |
|---------|--------|
| Plotting the functional boxplot / depth diagnostics | Numeric-library design goal — T-02 delivers the *numeric* central-region/whisker/outlier outputs; rendering is out of scope (consistent with the R audit's plotting exclusion). |
| Broader smoothing-criteria set (FPE/Shibata/Rice) | T-01 adds AIC only (the mainstream criterion); the rest are a separate lower-ranked backlog item (PREP-01-adjacent). |
| Re-implementing R's exact API surface | Match the capability, not R's function signatures. |

## Traceability

Which phases cover which requirements. Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| T-01 | Phase 22 | Pending |
| T-02 | Phase 23 | Pending |

**Coverage:**
- v1 requirements: 2 total
- Mapped to phases: 2 (T-01 → Phase 22, T-02 → Phase 23)
- Unmapped: 0 ✓

---
*Requirements defined: 2026-08-16*
*Last updated: 2026-08-16 after roadmap creation (v0.20.0 Table-Stakes Quick Wins) — T-01 → Phase 22, T-02 → Phase 23, 100% coverage.*
