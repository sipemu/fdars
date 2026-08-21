# Phase 34: Functional Serial-Dependence Tooling - Context

**Gathered:** 2026-08-21
**Status:** Ready for planning

<domain>
## Phase Boundary

Deliver FTS-02 — functional serial-dependence diagnostics for a time-ordered series of functional observations, in a new `fdars-core/src/fts/` module. Scope: L2-norm functional ACF (fACF) and partial ACF (fPACF) with strong-white-noise confidence bands, a functional stationarity test, a long-run-covariance kernel-sandwich estimator, and a functional differencing operator. Reuse `helpers` quadrature (`simpsons_weights`, `l2_distance`, `trapz`, `cumulative_trapz`) and `covariance.rs`. Numeric outputs only — no plotting/rendering. Additive/non-breaking, `Result`-returning, crate-root re-exported, inline `#[cfg(test)]` tests. Zero changes to existing public signatures, no new crate dependency. Independent of Phases 35/36. Foundational for the deferred FTS-01/FTS-03 forecasting items. R baseline: `ftsa` / `fdaACF`.

</domain>

<decisions>
## Implementation Decisions

### Module layout & API surface
- New `fts/` directory with `fts/acf.rs` implementation + `fts/mod.rs` barrel (mirrors the `inference/` module structure).
- One result struct per tool: `FacfResult` (lags, acf, pacf, confidence bands), a stationarity-test result, a long-run-covariance result — each deriving `Debug, Clone, PartialEq` with conditional serde, `#[non_exhaustive]` per convention.
- Named public entry points: `functional_acf`, `functional_pacf`, `stationarity_test`, `long_run_covariance`, `functional_difference` (final names at planner's discretion but this surface).
- Crate-root `pub use` re-export for all entry points and result types (project convention).

### fACF/fPACF formulation
- L2-norm functional ACF following the `fdaACF` convention: autocorrelation at lag h as the L2 norm of the lag-h autocovariance operator, normalized by the lag-0 term over the domain (Simpson/quadrature-weighted).
- White-noise confidence bands derived from the strong-white-noise limiting distribution (the `fdaACF` quadratic-form / χ²-mixture band). Document in rustdoc if the limiting distribution is approximated.
- Default lag range `max_lag = min(20, N/4)` when unspecified.
- Partial ACF via Durbin-Levinson-style recursion over the functional ACF sequence.

### Stationarity test & long-run covariance
- Monte-Carlo functional stationarity test (`ftsa` T_stationary style): a test statistic plus a seeded resampling p-value.
- Long-run covariance via a Bartlett kernel-sandwich (HAC) estimator by default, with a bandwidth argument.
- Default bandwidth `⌊N^{1/3}⌋` (standard HAC rule); bandwidth 0 reduces to the lag-0 sample covariance.
- Reproducible randomness via a single shared `StdRng::seed_from_u64(seed)` seed parameter (mirrors the permutation-test convention — NOT per-lag seed+k).

### Differencing, errors & divergence
- Functional first-difference operator (order 1): output curve series has length N−1 and round-trips against a cumulative-sum reconstruction within a documented tolerance.
- Return `FdarError` (never panic) on: empty matrix, fewer curves than requested max lag, argvals/values length mismatch, degenerate/zero-variance columns, invalid (negative) bandwidth.
- Numeric output only — no plotting.
- Document any divergence from the R baseline (esp. white-noise band approximation) in rustdoc, per prior-milestone convention.

### Claude's Discretion
- Exact final function/struct names, internal helper factoring, and the precise χ²-mixture band quantile approximation are at the planner/executor's discretion, guided by the fdaACF/ftsa references and codebase conventions.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `helpers.rs`: `simpsons_weights(argvals)`, `l2_distance(c1,c2,weights)`, `trapz(y,x)`, `cumulative_trapz(y,x)` — quadrature for L2 norms and the differencing round-trip check.
- `covariance.rs`: `CovKernel` enum, `covariance_matrix`, `generate_gaussian_process` (useful for synthetic test data / GP-based white-noise curves).
- `seasonal/` already has a scalar `autocorrelation` helper (period detection) — a naming/behavior reference, but functional ACF is a distinct L2-operator quantity; do not reuse the scalar path directly.
- `inference/` module (7 files, `TestResult`, χ²/F survival functions self-contained) — the model for a new `Result`-returning numeric module with seeded resampling and no new dependency.

### Established Patterns
- Column-major `FdMatrix` (`src/matrix.rs`); rows = observations/curves, columns = evaluation points. A "time-ordered series of curves" is the rows of an `FdMatrix`.
- Feature-gated rayon via `iter_maybe_parallel!` etc.; per-thread RNG seeding `StdRng::seed_from_u64(seed + k)` (but here use a single shared seed per the decision above).
- `Result<T, FdarError>` on all public fns; dimension checks at entry.
- Full clippy gate: `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.

### Integration Points
- New `fts/mod.rs` registered in `src/lib.rs` (`pub mod fts;`) + crate-root `pub use fts::{...}` re-exports.
- Self-contained limiting-distribution quantiles (reuse/extend `inference/dist.rs` χ² survival if a χ²-mixture band is used) — no new crate dependency.

</code_context>

<specifics>
## Specific Ideas

- Test discipline (from success criteria): on i.i.d. (white-noise) synthetic curves the ACF at nonzero lags falls inside the bands; on curves with injected lag-1 dependence the lag-1 fACF exceeds the band; differencing round-trips vs cumulative sum; stationarity test rejects on a trended series and not on a stationary one; long-run covariance reduces to the lag-0 sample covariance at bandwidth 0. All seeded for reproducibility.

</specifics>

<deferred>
## Deferred Ideas

- Full functional time series forecasting (ftsm, FPC-regression, fplsr, updating) — FTS-01, deferred to v2 (this phase builds the serial-dependence foundation it needs).
- Spectral / dynamic FTS methods — FTS-03, deferred.
- Flat-top / Parzen long-run-covariance kernels and configurable higher-order differencing (order d) — kept as future extensions; this phase ships Bartlett + first-difference.

</deferred>
