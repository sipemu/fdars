---
phase: 41-spectral-functional-time-series
plan: 01
subsystem: fts
tags: [spectral-density, dynamic-fpca, dpca, rustfft, nalgebra, functional-time-series]

requires:
  - phase: 34-functional-time-series (FTS-01/02, shipped)
    provides: fts/acf.rs autocovariance_matrix, mean_curve, long_run_covariance bandwidth convention
provides:
  - spectral_density — spectral density operator at Fourier frequencies (FTS-03-01)
  - dpca — dynamic eigen-filters + dynamic scores (FTS-03-02)
  - dpca_reconstruct — inverse dynamic filtering with monotone reconstruction error (FTS-03-03)
  - SpectralDensityResult / DpcaResult / DpcaReconstruction result structs + crate-root re-exports
affects: [phase-42, fts, spectral, dynamic-fpca]

actuals:
  tokens: 24000
  tasks: 4
  commits: 1

tech-stack:
  added: []
  patterns:
    - "Bartlett-weighted DFT across the lag index via a single reused rustfft plan (m² entry DFTs)"
    - "Simpson-metric-consistent DPCA: eigendecompose W^{1/2}Re(f̂)W^{1/2}, recover physical filters φ=ψ/√w so estimator/score/reconstruction share one inner product"
    - "Eigenvector sign-alignment (largest-abs entry positive) across frequencies before IFFT"

key-files:
  created:
    - fdars-core/src/fts/spectral.rs
  modified:
    - fdars-core/src/fts/mod.rs
    - fdars-core/src/lib.rs

key-decisions:
  - "Re-only Hermitian eigendecomposition via nalgebra::SymmetricEigen (no faer / no new dep) — documented freqdom divergence"
  - "1/2π pre-factor omitted (matches long_run_covariance) — eigenvalues 2π larger than freqdom; documented"
  - "Dynamic scores trimmed to valid interior t∈[L,N-1-L] (valid_range), not zero-padded; reconstruction error computed over the fully-defined window [2L,N-1-2L] for clean monotonicity"
  - "Default filter lag L = resolved bandwidth = ⌊N^{1/3}⌋"

patterns-established:
  - "Metric-consistent dynamic FPCA: fold Simpson weights into the eigenproblem (W^{1/2}·W^{1/2} scaling, mirrors acf.rs MC-band scaling) so rank-1 series reconstructs exactly with K=1"
  - "Direct-sum DFT as an in-test oracle to validate the FFT path to 1e-9"

requirements-completed: [FTS-03-01, FTS-03-02, FTS-03-03]

coverage:
  - id: D1
    description: "spectral_density returns per-frequency Hermitian m×m operators; white-noise spectrum is flat (verified via direct-sum DFT equality)"
    requirement: FTS-03-01
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/spectral.rs#tracer_white_noise_flat (+ spectral_density_hermitian_symmetry, spectral_density_deterministic, spectral_density_errors_empty_and_argvals)"
        status: pass
  - id: D2
    description: "dpca returns correctly-shaped, finite, sign-aligned dynamic eigen-filters ((2L+1)×m) and dynamic scores ((N-2L)×ncomp) with validated ncomp/filter_lag"
    requirement: FTS-03-02
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/spectral.rs#dpca_shapes_and_finiteness (+ dpca_white_noise_flat_eigenvalues, dpca_parameter_range_errors)"
        status: pass
  - id: D3
    description: "dpca_reconstruct reconstruction error monotone non-increasing in K; rank-1 series reconstructs to <1e-4 with K=1"
    requirement: FTS-03-03
    verification:
      - kind: unit
        ref: "fdars-core/src/fts/spectral.rs#dpca_reconstruct_monotone_error, dpca_reconstruct_rank1_exact (+ dpca_reconstruct_dimension_mismatch)"
        status: pass
---

# Plan 41-01 Summary: Spectral Functional Time Series (spectral density, DPCA, reconstruction)

## Accomplishments

- **`spectral_density`** (FTS-03-01): estimates the spectral density operator as a Bartlett-weighted DFT (via `rustfft`) across the lag index of the reused `fts::acf::autocovariance_matrix` operators, evaluated at the Fourier grid `θ_k = 2πk/N`. Stored as separate real/imag flat column-major m×m matrices per frequency; Hermitian by construction (negative lags via `C_{-h}[j1,j2]=C_h[j2,j1]`). Bandwidth defaults to `⌊N^{1/3}⌋` with a `min(n-1)` underflow guard.
- **`dpca`** (FTS-03-02): eigendecomposes the Simpson-metric-scaled real part `W^{1/2}Re(f̂)W^{1/2}` per frequency (`nalgebra::SymmetricEigen`), sign-aligns eigenvectors across frequencies, inverse-FFTs each eigenvector trajectory into real symmetric time-domain filter taps over `[-L,L]`, and forms dynamic scores by Simpson-weighted convolution over the interior `t∈[L,N-1-L]`.
- **`dpca_reconstruct`** (FTS-03-03): inverse dynamic filtering summing retained components; integrated-L2 `reconstruction_error[K-1]` is monotone non-increasing in K over the fully-defined window, and a rank-1 series reconstructs exactly (<1e-4) with K=1 thanks to the metric-consistent projection.
- Additive-only: three new public functions + three result structs, re-exported from `fts/mod.rs` and the crate root; **no existing signature changed, no new dependency**.

## Verification

- `cargo test -p fdars-core --features linalg,parallel --lib fts::spectral` — 10/10 pass.
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
- `cargo fmt` — clean.

## Divergences from freqdom/ftsa (documented in rustdoc)

- Eigendecomposition on `Re(f̂(θ))` (nalgebra has no complex-Hermitian path without faer) — exact for the leading dynamic subspace of a real-lag-window estimator.
- `1/2π` pre-factor omitted (matches `long_run_covariance`); eigenvalues are `2π` larger than R.
- Scores trimmed to `valid_range` rather than zero-padded.

## Notes for Phase 42 / downstream

- The metric-consistent eigenproblem pattern (fold Simpson weights via `W^{1/2}` scaling) is the reusable trick that makes projection/reconstruction self-consistent — mirrors the existing `acf.rs` MC-band scaling.
