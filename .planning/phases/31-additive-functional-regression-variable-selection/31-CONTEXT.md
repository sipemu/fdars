# Phase 31: Additive Functional Regression & Variable Selection - Context

**Gathered:** 2026-08-20
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous)

<domain>
## Phase Boundary

Add nonparametric additive scalar-on-function regression to `fdars-core` in a new
`scalar_on_function/additive.rs`, delivering the REG-04 capability set:

- **FAM** — functional additive model via backfitting over FPC-score components.
- **GKAM** — generalized kernel additive model variant.
- **GSAM** — generalized spectral additive model variant.
- **variable_selection** — group-penalized scalar-on-function selection helper.
- **permutation-test wrapper** — significance testing with seeded reproducibility.
- **history-index estimator** — lagged-predictor-window (functional-history) model.

Reuses `smoothing.rs` kernels (`nadaraya_watson`, `local_linear`, `local_polynomial`)
and `regression.rs::fdata_to_pc_1d`. Strictly additive/non-breaking: no existing public
signature changes, `Result<T, FdarError>`-returning, inline `#[cfg(test)]` tests, crate-root
re-exports. Numeric outputs only — no plotting/rendering.

</domain>

<decisions>
## Implementation Decisions

### Reference Fidelity
- Match R baselines (`fdapace` FAM / `fda.usc` GKAM+GSAM / `refund` fosr.vs+fosr.perm+history-index)
  **by capability**, not by exact R signature.
- Document any divergence from the R reference formulation in rustdoc (established milestone
  convention — e.g. Fast-MUOD, response-permutation FLM divergences were documented this way).
- Pin the exact backfitting / kernel / spectral construction and the group-penalty for
  `variable_selection` during plan-phase research.

### API Shape
- Config structs for the complex estimators (e.g. `FamConfig`, `GkamConfig`/`GsamConfig`,
  `VarSelectConfig`, `HistoryIndexConfig`) following the builder-config convention
  (`GmmClusterConfig`, `ElasticConfig`, etc.), with serde behind the `serde` feature.
- Structured immutable `Result` types (e.g. `FamResult`, `VarSelectResult`) deriving
  `Debug, Clone, PartialEq`, carrying scores/fitted/diagnostics for reproducibility.
- Parameter ordering follows `(data, y, [argvals,] [scalar_covariates,] config)`.

### Estimator Depth
- Full multi-functional-covariate support for FAM backfitting (the additive model is defined
  over multiple additive components — single-covariate would not be a faithful FAM).
- Scalar covariates supported in the same functions (no separate overloads), per convention.

### Correctness Tests
- Synthetic-recovery tests (fit on data generated from a known additive structure, check
  recovery within tolerance).
- Known-property invariants (e.g. additive decomposition sums, permutation-null centering).
- Seeded-permutation reproducibility for the permutation-test wrapper
  (`StdRng::seed_from_u64(seed + k)` pattern; mirror INF-01's 999-perm default).

### Claude's Discretion
- Exact config field names, default bandwidths/component counts, internal helper factoring,
  and the precise number of test cases are at Claude's discretion within the above.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fdars-core/src/smoothing.rs` — `nadaraya_watson`, `local_linear`, `local_polynomial`,
  `optim_bandwidth`, GCV/AIC smoother selectors (kernel machinery for GKAM/backfitting smooths).
- `fdars-core/src/regression.rs::fdata_to_pc_1d` — FPCA scores/loadings for FAM component basis.
- `fdars-core/src/scalar_on_function/nonparametric.rs` — existing kernel functional-regression
  patterns (`fregre_np_*`) as a style/interface analog.
- `fdars-core/src/scalar_on_function/mod.rs` — barrel re-export pattern; new `additive` module
  wires in here + crate-root re-export in `lib.rs`.

### Established Patterns
- Column-major `FdMatrix`; `Result<T, FdarError>` on all public fns; feature-gated rayon
  (`iter_maybe_parallel!`); per-thread RNG seeding for reproducible permutations.
- Config-struct + Result-struct pairing; `#[must_use]` on expensive computations;
  inline `#[cfg(test)] mod tests`.

### Integration Points
- New file `scalar_on_function/additive.rs`; `pub use` in `scalar_on_function/mod.rs`;
  crate-root re-exports in `src/lib.rs`; optional prelude additions.

</code_context>

<specifics>
## Specific Ideas

- Permutation-test wrapper should default to a 999-permutation count with a caller-supplied
  seed, mirroring the INF-01 inference convention.
- Reuse `distance`/kernel/FPCA infrastructure rather than introducing new numeric primitives;
  **no new crate dependency** (milestone constraint).

</specifics>

<deferred>
## Deferred Ideas

- Plotting/rendering of additive fits (out of scope — numeric outputs only).
- Boosting/Bayesian functional additive regression (REG-06, deferred).

</deferred>
