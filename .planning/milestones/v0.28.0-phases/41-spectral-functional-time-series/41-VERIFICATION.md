---
phase: 41-spectral-functional-time-series
verified: 2026-08-22T23:30:00Z
status: passed
score: 5/5 must-haves verified
behavior_unverified: 0
overrides_applied: 0
---

# Phase 41: Spectral Functional Time Series Verification Report

**Phase Goal:** Users can analyze a functional time series in the frequency domain — estimate its spectral density operator, run dynamic FPCA, reconstruct curves from dynamic scores — and simulate functional VAR/VMA and FARMA processes.
**Verified:** 2026-08-22T23:30:00Z
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | User can estimate the spectral density operator at Fourier frequencies (rustfft-transformed long-run covariance over lagged autocovariance operators) → numeric per-frequency operator result. (FTS-03-01) | VERIFIED | `spectral_density` in `fts/spectral.rs` (lines 114–181): Bartlett-weighted DFT over `autocovariance_matrix` calls; 10 tests pass including `tracer_white_noise_flat` (FFT == direct-sum DFT to 1e-9), `spectral_density_hermitian_symmetry`, `spectral_density_deterministic`, `spectral_density_errors_empty_and_argvals`. |
| 2 | User can compute dynamic FPCA from that spectral density → dynamic eigen-filters + dynamic scores. (FTS-03-02) | VERIFIED | `dpca` in `fts/spectral.rs` (lines 262–358): eigendecomposes `Re(f̂)` per frequency, inverse-FFTs into time-domain filter taps, computes Simpson-weighted convolution scores over interior `[L, N-1-L]`. Tests `dpca_shapes_and_finiteness`, `dpca_white_noise_flat_eigenvalues`, `dpca_parameter_range_errors` pass. |
| 3 | User can reconstruct the curve series from DPCA scores via inverse dynamic filtering; reconstruction error decreases as more dynamic components are retained. (FTS-03-03) | VERIFIED | `dpca_reconstruct` in `fts/spectral.rs` (lines 375–466): cumulative-K reconstruction with integrated-L2 error. `dpca_reconstruct_monotone_error` asserts monotone non-increasing within 1e-9 tolerance; `dpca_reconstruct_rank1_exact` asserts K=1 error < 1e-4 for a rank-1 AR(1) series. Both pass. |
| 4 | User can simulate a functional VAR/VMA curve series from user-supplied operator kernels → deterministic (seeded) numeric curve set. (FTS-03-04) | VERIFIED | `sim_fvarma` in `simulation.rs` (lines 663–678) delegates to private `fvarma_core` (lines 535–624). Five tests pass: `fvarma_deterministic` (bit-identical under fixed seed), `fvarma_zero_op_white_noise` (‖C_1‖ < 0.15·‖C_0‖), `fvarma_rank1_dependence` (‖C_1‖ > 0.1·‖C_0‖), `fvarma_dimension_errors`, `fvarma_divergence_guard` (2×identity → ComputationFailed). |
| 5 | User can simulate a functional ARMA (FARMA) curve series combining AR and MA terms → deterministic (seeded) numeric curve set. (FTS-03-05) | VERIFIED | `sim_farma` in `simulation.rs` (lines 691–706): thin wrapper over `fvarma_core`, bit-identical to `sim_fvarma`. Three tests pass: `farma_shape_and_order`, `farma_deterministic`, `farma_equals_fvarma` (identical inputs + seed → equal `curves`). |

**Score:** 5/5 truths verified (0 present, behavior-unverified)

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `fdars-core/src/fts/spectral.rs` | New file with `spectral_density`, `dpca`, `dpca_reconstruct` | VERIFIED | 762 lines; full implementation + 10 inline tests |
| `SpectralDensityResult` in `fts/mod.rs` | Struct with `freqs`, `re`, `im`, `m`, `n_curves`, `bandwidth` | VERIFIED | Lines 49–62 of `fts/mod.rs`; derive `Debug, Clone, PartialEq`, serde-gated, `#[non_exhaustive]` |
| `DpcaResult` in `fts/mod.rs` | Struct with `filters`, `scores`, `eigenvalues`, `n_freqs`, `filter_lag`, `ncomp`, `valid_range` | VERIFIED | Lines 79–97 of `fts/mod.rs` |
| `DpcaReconstruction` in `fts/mod.rs` | Struct with `fitted`, `reconstruction_error`, `valid_range` | VERIFIED | Lines 107–115 of `fts/mod.rs` |
| `sim_fvarma`, `sim_farma` in `simulation.rs` | New public functions with `FvarmaResult`/`FarmaResult` | VERIFIED | Lines 663–706 of `simulation.rs`; 422 lines added total |
| `FvarmaResult`, `FarmaResult` in `simulation.rs` | Structs with `curves`, `ar_order`, `ma_order`, `burn_in` | VERIFIED | Lines 465–492 of `simulation.rs`; derive `Debug, Clone, PartialEq`, serde-gated, `#[non_exhaustive]` |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `spectral.rs` | `super::acf::autocovariance_matrix` | `super::acf::autocovariance_matrix(data, &xbar, h, n, m)` — line 138 | WIRED | Reuses existing pub(crate) function; no reimplementation |
| `spectral.rs` | `crate::helpers::simpsons_weights` | Direct call at lines 286, 407 | WIRED | Simpson weights used consistently in both `dpca` and `dpca_reconstruct` |
| `spectral.rs` | `rustfft::FftPlanner` | `FftPlanner::new()` at lines 142, 303; `plan_fft_forward(n_freq)` / `plan_fft_inverse(n_freq)` | WIRED | Single planner per call; plan reused across all (j1,j2) entries |
| `fts/mod.rs` | `spectral.rs` functions | `mod spectral;` (line 22) + `pub use spectral::{dpca, dpca_reconstruct, spectral_density};` (line 28) | WIRED | All three public functions re-exported from the module |
| `lib.rs` | `fts` module | `pub use fts::{dpca, dpca_reconstruct, spectral_density, DpcaReconstruction, DpcaResult, SpectralDensityResult, ...}` (lines 262–265) | WIRED | Extends existing block; `functional_acf`, `ftsm`, and other pre-existing exports preserved |
| `simulation.rs` | `fvarma_core` (shared recurrence) | `sim_farma` delegates to `fvarma_core` at line 699; `sim_fvarma` at line 671 | WIRED | Bit-identity between FVARMA and FARMA confirmed by `farma_equals_fvarma` test |
| `lib.rs` | `simulation` module | `pub use simulation::{sim_farma, sim_fvarma, EFunType, EValType, FarmaResult, FvarmaResult};` (line 235) | WIRED | Existing `EFunType`/`EValType` preserved; four new symbols added |
| `simulation.rs` | `validate_operator_kernels` | Called from `fvarma_core` line 543; validates `m==0`, `ar_ops[k].len()!=m*m`, `ma_ops[k].len()!=m*m` | WIRED | Dimension guard fires before any index arithmetic |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `spectral_density` | `re[k]`, `im[k]` | Bartlett-weighted DFT of `autocovariance_matrix` results for h=0..=max_h | Yes — computed from input `data` matrix via FFT | FLOWING |
| `dpca` | `filters[c]`, `scores` | IFFT of per-frequency eigenvectors + Simpson-weighted convolution over `data` | Yes — eigenvectors derive from `spectral_density`; scores from `data` directly | FLOWING |
| `dpca_reconstruct` | `fitted`, `reconstruction_error` | Adjoint convolution of `dpca.filters` with `dpca.scores`; error from `data` vs `fitted` | Yes — both derive from live `data` and `dpca` inputs | FLOWING |
| `sim_fvarma`/`sim_farma` | `curves` | `fvarma_core` recurrence: i.i.d. N(0,1) innovations from `StdRng::seed_from_u64(seed)` combined with AR/MA operator kernels | Yes — deterministic from seed, not static | FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| 10 spectral tests all pass | `cargo test -p fdars-core --features linalg,parallel --lib fts::spectral` | `10 passed; 0 failed; finished in 0.12s` | PASS |
| 5 sim_fvarma oracle tests pass | `cargo test -p fdars-core --features linalg,parallel --lib "simulation::tests::fvarma"` | `5 passed; 0 failed; finished in 0.01s` | PASS |
| 3 sim_farma tests pass | `cargo test -p fdars-core --features linalg,parallel --lib "simulation::tests::farma"` | `3 passed; 0 failed; finished in 0.00s` | PASS |
| Clippy clean across all targets | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` | `Finished dev profile` (no warnings/errors) | PASS |

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| FTS-03-01 | 41-01 | Spectral density operator estimator at Fourier frequencies | SATISFIED | `spectral_density` implemented + 4 tests covering flat spectrum, Hermitian symmetry, determinism, error paths |
| FTS-03-02 | 41-01 | Dynamic FPCA — dynamic eigen-filters + scores | SATISFIED | `dpca` implemented + 3 tests covering shapes/finiteness, flat eigenvalues (white noise), parameter range errors |
| FTS-03-03 | 41-01 | Curve reconstruction from DPCA scores with monotone error | SATISFIED | `dpca_reconstruct` implemented + 3 tests covering monotone error, rank-1 exact (K=1 < 1e-4), dimension mismatch |
| FTS-03-04 | 41-02 | Functional VAR/VMA simulator from operator kernels | SATISFIED | `sim_fvarma` + `fvarma_core` implemented + 5 tests covering determinism, white-noise oracle, rank-1 dependence, dim errors, divergence guard |
| FTS-03-05 | 41-02 | Functional ARMA (combined AR+MA) simulator | SATISFIED | `sim_farma` implemented as thin wrapper; 3 tests covering shape/order, determinism, equality with sim_fvarma |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| None | — | — | — | — |

No `TBD`, `FIXME`, or `XXX` markers in phase-modified files. No stub patterns (empty returns, placeholder implementations). All dynamic data flows to rendered output.

### Additive-Only Constraint

Git commit `7a1c0652` (plan 41-01): `+857 / -4 lines` — 4 deleted lines are reorganized re-export block entries in `lib.rs`, not signature changes. Commit `95f59609` (plan 41-02): `+424 / -2 lines` — 2 deleted lines are re-export block reorganization. No existing public function signatures were removed or altered. Confirmed by inspecting both commit stats.

### Human Verification Required

None — all must-haves are verifiable from the codebase and passing tests encode the behavioral oracles directly (white-noise flatness via direct-sum DFT equality to 1e-9, rank-1 exact reconstruction < 1e-4, monotone error, bit-identical determinism, divergence guard). No visual, real-time, or external-service behavior involved.

### Gaps Summary

No gaps. All 5 success criteria are met with passing automated oracle tests encoding the behavioral invariants (not just presence checks). Crate-root re-exports confirmed for all 10 new symbols. Clippy clean. No breaking changes to existing signatures.

---

_Verified: 2026-08-22T23:30:00Z_
_Verifier: Claude (gsd-verifier)_
