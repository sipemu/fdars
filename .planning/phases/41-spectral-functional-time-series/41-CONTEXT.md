# Phase 41: Spectral Functional Time Series - Context

**Gathered:** 2026-08-22
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — all 4 grey areas accepted as recommended

<domain>
## Phase Boundary

Deliver frequency-domain analysis for functional time series (FTS-03), additively, in a new
`fdars-core/src/fts/spectral.rs` module plus simulator additions in `simulation.rs`:

1. Spectral density operator estimation at Fourier frequencies (FTS-03-01).
2. Dynamic functional PCA (DPCA): dynamic eigen-filters + dynamic scores (FTS-03-02).
3. Curve reconstruction from DPCA scores via inverse dynamic filtering (FTS-03-03).
4. Functional VAR/VMA simulator from user-supplied operator kernels (FTS-03-04).
5. Functional ARMA (FARMA) simulator combining AR+MA operator terms (FTS-03-05).

Numeric outputs only — no plotting/rendering. Additive/non-breaking: zero changes to existing
public signatures (`fts/acf.rs`, `fts/forecast.rs`, `simulation.rs`). Reuses the existing
`rustfft` dependency and the FTS-01/FTS-02 autocovariance machinery. No new crate dependency.

</domain>

<decisions>
## Implementation Decisions

### Spectral Density Operator (FTS-03-01)
- **Kernel weighting:** Bartlett lag-window weights over lagged autocovariance operators, reusing
  the existing `long_run_covariance` / `autocovariance_matrix` machinery in `fts/acf.rs`. Default
  bandwidth ⌊N^{1/3}⌋ (matches `long_run_covariance` convention), user-overridable.
- **Frequency set:** standard Fourier grid θ_j = 2πj/N for j = 0..N-1; return one m×m operator per
  frequency.
- **rustfft application:** DFT across the lag index h for each (j₁,j₂) entry of the autocovariance
  operator sequence → complex m×m spectral density operator per frequency.
- **Output representation:** per-frequency m×m operator stored as separate real + imaginary flat
  `Vec<f64>` (Hermitian), column-major, consistent with `LongRunCovResult.cov_matrix`.

### Dynamic FPCA — Filters, Scores, Reconstruction (FTS-03-02/03)
- **Dynamic eigen-filters:** eigendecompose the spectral density operator at each Fourier frequency,
  then inverse-FFT the frequency-domain eigenvectors to obtain time-domain filter coefficients over
  a symmetric lag window (`freqdom` `dpca.filters` convention).
- **Filter lag support:** symmetric window h ∈ [−L, L], with a modest default L (user-overridable).
- **Dynamic scores:** time-domain convolution of the curve series with the dynamic filters, yielding
  one score series per retained dynamic component (`freqdom` `dpca.scores`).
- **Reconstruction (FTS-03-03):** inverse dynamic filtering — convolve scores with the adjoint
  (time-reversed) filters and sum over retained components. Integrated-L2 reconstruction error must
  decrease monotonically as more dynamic components are retained (success criterion 3).

### VAR/VMA + FARMA Simulators (FTS-03-04/05)
- **Operator-kernel parameterization:** user supplies m×m operator matrices (one per AR lag, one per
  MA lag) that act on the grid-discretized curve vector via matrix-vector product.
- **Innovations:** Gaussian white-noise curves generated via the existing KL machinery
  (`sim_kl`-style eigenstructure), seeded.
- **Burn-in:** documented default burn-in (e.g. 200) discarded to reach stationarity; user-settable.
- **Seeding:** explicit `seed: u64` → `StdRng::seed_from_u64(seed)`, fully deterministic output
  (crate convention; per-thread `seed + k` if any parallel loop is used).

### API Surface & Result Types
- **Location:** new `fdars-core/src/fts/spectral.rs` for spectral density + DPCA; VAR/VMA + FARMA
  simulators added to `simulation.rs`. Re-export new public items at the crate root and in `fts/mod.rs`.
- **Result structs:** `SpectralDensityResult`, `DpcaResult`, `DpcaReconstruction` (or similarly
  named), each deriving `Debug, Clone, PartialEq`, serde-gated, `#[non_exhaustive]`, following the
  existing `*Result` convention.
- **Function names:** `spectral_density`, `dpca`, `dpca_reconstruct`, `sim_fvarma`, `sim_farma`
  (snake_case with domain hints).
- **Grid handling:** take explicit `argvals` and use Simpson integration weights, consistent with
  the rest of `fts` and the crate's integration-weight convention.

### Claude's Discretion
- Exact default value of the DPCA filter lag support L, the burn-in length, and the precise
  Result-struct field layout are at the planner's discretion within the conventions above.
- Whether to expose a private `autocovariance_matrix`-style helper (reuse vs. thin wrapper) is an
  implementation detail — reuse the `fts/acf.rs` machinery where possible.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `fts/acf.rs`: private `autocovariance_matrix(data, xbar, h, n, m)` and `mean_curve`, plus public
  `long_run_covariance` (Bartlett kernel-sandwich, default bandwidth ⌊N^{1/3}⌋) — directly reusable
  for the lagged-autocovariance-operator input to the spectral density estimator.
- `simulation.rs`: `sim_kl`, `eigenfunctions`/`eigenvalues`, `add_error_*`, KL-expansion machinery
  and RNG seeding pattern (`rand`, `rand_distr::Normal`, `StdRng`) — reusable for simulator innovations.
- `rustfft` already used across the crate (`seasonal/mod.rs`, `metric/fourier.rs`, `basis/auto_select.rs`)
  via `FftPlanner::<f64>::new()` and `rustfft::num_complex::Complex` — the established FFT pattern.
- `FdMatrix` (column-major) row/column helpers; nalgebra SVD/eigendecomposition path via `to_dmatrix()`.

### Established Patterns
- `fts/mod.rs`: `mod acf; mod forecast;` with explicit `pub use`; `*Result` structs defined in `mod.rs`,
  derive `Debug, Clone, PartialEq`, serde-gated, `#[non_exhaustive]`. New `mod spectral;` + `pub use`
  follows the same shape.
- `validate_fts_input(data, argvals) -> (n, m)` entry-point validation; all public fns return
  `Result<_, FdarError>`; deterministic `seed` with 999 default MC reps.

### Integration Points
- New public items re-exported in `fts/mod.rs` and at crate root (`src/lib.rs`) per the flat-API convention.
- Simpson weights via `helpers::simpsons_weights` (integration-weight convention shared across FPCA/fts).

</code_context>

<specifics>
## Specific Ideas

- R baselines: `freqdom` (DPCA, spectral density, VAR/VMA) + `ftsa` (FARMA simulation, dynamic FPCA).
  Match by capability, not exact R signatures; document any divergence from `freqdom`/`ftsa` in rustdoc.
- Success criterion 3 requires a *testable monotonicity property*: reconstruction error must decrease
  as more dynamic components are retained — plan an inline `#[cfg(test)]` check for it.
- Simulators must be deterministic under a fixed seed (success criteria 4 & 5) — inline tests assert
  bit-reproducibility across two calls with the same seed.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Plotting/rendering of spectra/DPCA filters is an
explicit milestone Out-of-Scope item; FRE-02 object-data Fréchet regression is Phase 42.)

</deferred>
