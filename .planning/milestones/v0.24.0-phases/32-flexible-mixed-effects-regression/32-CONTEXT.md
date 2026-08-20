# Phase 32: Flexible Mixed-Effects Regression - Context

**Gathered:** 2026-08-20
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous)

<domain>
## Phase Boundary

Extend functional mixed models beyond fixed-effect testing to full random-effects estimation
(REG-05), by extending `fdars-core/src/famm.rs` (today: `fmm`, `fmm_predict`, `fmm_test_fixed`)
and wiring a flexible random-effects path into the already-present `fdars-core/src/fof_regression.rs`:

- **denseFLMM-style dense functional linear mixed model** — fixed + random effects via
  mixed-model equations over FPC scores, returning fixed-effect estimates, random-effect /
  variance-component estimates, and fitted functional curves.
- **multiFAMM** — multivariate functional additive mixed variant.
- **fastFMM** — fast functional mixed-model inference (massively-univariate path).
- **flexible random-effects function-on-function** estimator wired into `fof_regression.rs`
  (extends ONLY the flexible/RE variant; the base FoF capability is already at parity and its
  signatures stay untouched).

Strictly additive/non-breaking: no existing public signature changes; `Result<T, FdarError>`;
inline `#[cfg(test)]` tests; crate-root re-exports. Numeric outputs only.

</domain>

<decisions>
## Implementation Decisions

### Parametrization
- Mixed-model equations formulated over **FPC scores** (reuse `regression::fdata_to_pc_1d` +
  the existing `famm.rs` fixed-effect machinery), not spline/basis coefficients.
- Document this choice vs the R baselines' basis-coefficient formulations in rustdoc.

### Variance Components
- **REML-style / method-of-moments** variance-component estimation over the per-component score
  models — no new crate dependency, reuse existing linear-algebra (`linalg`, nalgebra).
- Return variance components (random-intercept/slope variances + residual) alongside fixed effects.

### Variant Depth
- multiFAMM and fastFMM implemented **faithful-by-capability**, not to exact R signatures.
- fastFMM realized as a **massively-univariate** fit (per-gridpoint / per-component mixed model
  with a fast inference path), documenting the divergence from the R `fastFMM` internals in rustdoc.
- multiFAMM covers the multivariate (stacked-response) additive mixed case reusing the denseFLMM core.

### Correctness Tests
- Synthetic-recovery: generate data from a known mixed model (fixed effect + grouped random
  intercepts/slopes with known variance components); assert recovery of fixed effects and the
  variance-component structure within a documented tolerance, and fitted curves track truth.
- Invalid-input `FdarError` paths: empty data / mismatched grouping-factor length / singular
  design / mismatched dimensions — never panic.

### Claude's Discretion
- Config/result struct field names, default REML iteration counts / tolerances, internal helper
  factoring, and exact test counts are at Claude's discretion within the above.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/famm.rs` — `fmm` (mixed-model fit), `FmmResult`, `fmm_predict`,
  `fmm_test_fixed` (fixed-effect permutation test), `FmmTestResult`. The fixed-effect machinery
  and result-struct shape are the analog for the new estimators.
- `fdars-core/src/fof_regression.rs` — `fof_regression`, `FofResult`, `predict_fof`, `fof_cv`
  (base function-on-function at parity — extend only with the flexible/RE variant).
- `fdars-core/src/regression.rs::fdata_to_pc_1d` — FPC scores for the mixed-model over components.
- `fdars-core/src/linalg.rs` — Cholesky / ridge solves (behind `linalg` feature) for the
  mixed-model equations / variance-component solves.

### Established Patterns
- Column-major `FdMatrix`; `Result<T, FdarError>`; config-struct + result-struct pairing;
  `#[must_use]`, `#[non_exhaustive]`, serde-feature gating; per-thread RNG seeding for any
  permutation/bootstrap reproducibility.

### Integration Points
- New public fns + result structs in `famm.rs`; flexible-RE fn in `fof_regression.rs`;
  `pub use` in the module barrels; crate-root re-exports in `src/lib.rs`.

</code_context>

<specifics>
## Specific Ideas

- Reuse the `famm.rs` `FmmResult`-style result shape (fixed effects + fitted curves) extended
  with random-effect / variance-component fields.
- No new crate dependency (milestone constraint); reuse `linalg`/nalgebra for the mixed-model solves.

</specifics>

<deferred>
## Deferred Ideas

- Plotting/rendering of mixed-model diagnostics (out of scope — numeric outputs only).
- Changing the base function-on-function capability (already at parity — REG-05 extends only the RE variant).
- Bayesian functional mixed models (out of scope).

</deferred>
