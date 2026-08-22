# Phase 41: Spectral Functional Time Series — Pattern Map

**Mapped:** 2026-08-22
**Files analyzed:** 4 (1 new, 3 modified)
**Analogs found:** 4 / 4

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/src/fts/spectral.rs` | service (computation) | transform (FFT + eigendecomposition + convolution) | `fdars-core/src/fts/acf.rs` | exact — same module, same FdMatrix + FdarError + FftPlanner + SymmetricEigen pattern |
| `fdars-core/src/fts/mod.rs` | config / barrel | — | `fdars-core/src/fts/mod.rs` itself | self — add `mod spectral;`, `pub use`, and new Result structs following existing entries |
| `fdars-core/src/simulation.rs` | service (simulation) | batch / generative | `fdars-core/src/simulation.rs` itself (`sim_kl`) | self-extension — append `sim_fvarma`/`sim_farma` following `sim_kl` signature style |
| `fdars-core/src/lib.rs` | config / barrel | — | `fdars-core/src/lib.rs` itself | self — add `pub use fts::{...}` and `pub use simulation::{...}` entries |

---

## Pattern Assignments

### `fdars-core/src/fts/spectral.rs` (service, transform)

**Analog:** `fdars-core/src/fts/acf.rs`

**Imports pattern** (`acf.rs` lines 13–18):
```rust
use super::FacfResult;      // replace with super::{SpectralDensityResult, DpcaResult, DpcaReconstruction}
use crate::error::FdarError;
use crate::helpers::{simpsons_weights, trapz, NUMERICAL_EPS};
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::SeedableRng;
```

Add for spectral.rs (from `metric/fourier.rs` lines 7–8):
```rust
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use nalgebra::DMatrix;
use super::acf::autocovariance_matrix;   // pub(crate) — call directly
```

**Input validation pattern** (`acf.rs` lines 25–42) — copy verbatim, rename function or call from spectral.rs:
```rust
fn validate_fts_input(data: &FdMatrix, argvals: &[f64]) -> Result<(usize, usize), FdarError> {
    let (n, m) = data.shape();
    if n == 0 || m == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "non-empty matrix".to_string(),
            actual: format!("{n} rows, {m} columns"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m} elements (matching data columns)"),
            actual: format!("{} elements", argvals.len()),
        });
    }
    Ok((n, m))
}
```
Note: `forecast.rs` re-implements this verbatim (it is `pub(super)` only in `acf.rs`). Do the same in `spectral.rs` — copy the body, do not promote visibility.

**Bandwidth resolution pattern** (`acf.rs` lines 684–687) — copy exactly:
```rust
let resolved_bandwidth = match bandwidth {
    None => (n as f64).cbrt().floor() as usize,
    Some(b) => b,
};
// Guard: clip to n-1 to prevent usize underflow in autocovariance_matrix loop.
let max_h = resolved_bandwidth.min(n - 1);
```

**`autocovariance_matrix` call pattern** (`acf.rs` lines 708–720) — the column-major flat `Vec<f64>` convention; element `(j1, j2)` at index `j1 + j2 * m`:
```rust
let xbar = mean_curve(data, n, m);
let c0 = autocovariance_matrix(data, &xbar, 0, n, m);
// For h = 1..max_h:
let w_h = 1.0 - (h as f64) / (resolved_bandwidth as f64);
let c_h = autocovariance_matrix(data, &xbar, h, n, m);
// c_h[j1 + j2 * m] is the (j1, j2) entry.
```

**FFT planner pattern** (`metric/fourier.rs` lines 33–36):
```rust
let mut planner = FftPlanner::<f64>::new();
let fft = planner.plan_fft_forward(n_freq);
let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
fft.process(&mut buffer);
```
For inverse FFT: `planner.plan_fft_inverse(n_freq)` — same shape, divide output by `n_freq` to normalize.

**SymmetricEigen pattern** (`acf.rs` lines 337–348):
```rust
// Symmetrize defensively before eigendecomposition.
let mut c0_mat = DMatrix::from_column_slice(m, m, &spec_real);
for j1 in 0..m {
    for j2 in (j1 + 1)..m {
        let avg = 0.5 * (c0_mat[(j1, j2)] + c0_mat[(j2, j1)]);
        c0_mat[(j1, j2)] = avg;
        c0_mat[(j2, j1)] = avg;
    }
}
let eig = nalgebra::SymmetricEigen::new(c0_mat);
let mut eigenvalues: Vec<f64> = eig.eigenvalues.iter().copied().collect();
// Sort descending.
eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
```
Adapt for spectral.rs: after sorting, extract paired (eigenvalue, eigenvector column) before sign-alignment.

**Sign-alignment pattern** (RESEARCH.md Pitfall 2 / Code Examples):
```rust
// Apply after eigendecomposition at each frequency before assembling IFFT buffer.
// Align eigenvectors so the entry of maximum absolute value is positive.
let max_abs = evec.iter().copied().fold(f64::NEG_INFINITY, |a, x| a.max(x.abs()));
let sign = evec.iter().copied()
    .find(|&x| x.abs() == max_abs)
    .map(|x| if x < 0.0 { -1.0 } else { 1.0 })
    .unwrap_or(1.0);
evec.iter_mut().for_each(|x| *x *= sign);
```

**Simpson weights pattern** (`regression.rs` line 325 / `acf.rs` line 15):
```rust
use crate::helpers::simpsons_weights;
let weights = simpsons_weights(argvals);
```

**`#[must_use]` on public entry points** (crate-wide convention, 74+ instances):
```rust
#[must_use = "returns spectral density result; result should be examined"]
pub fn spectral_density(...) -> Result<SpectralDensityResult, FdarError> { ... }
```

**Inline test pattern** (`acf.rs` line 734+):
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
    // ... oracle tests
}
```
Required tests for spectral.rs:
- Oracle 1: white-noise input → flat spectral density (all frequencies yield matrices close to `C_0`).
- Oracle 2: rank-1 AR(1) series → `dpca_reconstruct` with K=1 error < 1e-4.
- Oracle 3: monotone-decreasing reconstruction error as K increases from 1 to `ncomp`.

---

### `fdars-core/src/fts/mod.rs` (barrel modification)

**Analog:** `fdars-core/src/fts/mod.rs` itself.

**Module declaration + pub use pattern** (lines 20–26) — append after `mod forecast;`:
```rust
mod acf;
mod forecast;
mod spectral;   // ADD

pub use acf::{
    functional_acf, functional_difference, functional_pacf, long_run_covariance, stationarity_test,
};
pub use forecast::{fplsr, ftsm, ftsm_forecast, ftsm_forecast_multistep, ftsm_update};
pub use spectral::{dpca, dpca_reconstruct, spectral_density};  // ADD
```

**Result struct pattern** (lines 33–45, 65–77) — derive block, serde gate, `#[non_exhaustive]`, rustdoc on all fields:
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FacfResult {
    /// Lag values (1..=max_lag).
    pub lags: Vec<u32>,
    /// Functional autocorrelation ρ_h at each lag (L2-norm, fdaACF convention).
    pub acf: Vec<f64>,
    ...
}
```

Apply this exact derive block to every new Result struct added in `mod.rs`:

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct SpectralDensityResult {
    /// Fourier frequencies θ_j = 2πj/N for j = 0..N-1, length N.
    pub freqs: Vec<f64>,
    /// Real part of the m×m spectral density operator at each frequency.
    /// `re[k]` is the flat column-major m×m matrix for frequency k; length N.
    pub re: Vec<Vec<f64>>,
    /// Imaginary part of the m×m spectral density operator at each frequency.
    /// `im[k]` is the flat column-major m×m matrix for frequency k; length N.
    pub im: Vec<Vec<f64>>,
    /// Grid dimension m (each operator is m×m).
    pub m: usize,
    /// Number of curves N (= number of Fourier frequencies).
    pub n_curves: usize,
    /// Bartlett bandwidth used.
    pub bandwidth: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DpcaResult {
    /// Dynamic filter taps per component.
    /// `filters[k]` is an FdMatrix of shape (2L+1) × m;
    /// row `l` holds the filter tap at lag `l - L` (i.e. row 0 = lag -L).
    pub filters: Vec<crate::matrix::FdMatrix>,
    /// Dynamic scores, shape (N - 2L) × ncomp; column k = score series for component k.
    pub scores: crate::matrix::FdMatrix,
    /// Per-frequency eigenvalues, shape ncomp × N_freq.
    /// `eigenvalues[k][freq]` = eigenvalue of spectral density at frequency `freq` for component k.
    pub eigenvalues: Vec<Vec<f64>>,
    /// Number of Fourier frequencies N.
    pub n_freqs: usize,
    /// Filter lag support L (filter window is [-L, L]).
    pub filter_lag: usize,
    /// Number of retained dynamic components.
    pub ncomp: usize,
    /// Valid time range (L, N-1-L) — indices into the original series with valid scores.
    pub valid_range: (usize, usize),
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct DpcaReconstruction {
    /// Reconstructed curves for the valid interior range, shape N' × m (N' = N - 2L).
    pub fitted: crate::matrix::FdMatrix,
    /// Cumulative reconstruction error per component (length ncomp).
    /// `reconstruction_error[k]` = integrated L2 error using components 1..=k+1.
    pub reconstruction_error: Vec<f64>,
    /// Valid time range (L, N-1-L) matching `DpcaResult::valid_range`.
    pub valid_range: (usize, usize),
}
```

---

### `fdars-core/src/simulation.rs` (service modification, batch generative)

**Analog:** `fdars-core/src/simulation.rs` itself (`sim_kl`).

**Module-level docstring pattern** (lines 1–27) — for the new functions, add rustdoc describing the recurrence, stationarity responsibility, and burn-in:
```rust
/// Simulate a functional VAR/VMA process.
///
/// Generates `n` curves from the recurrence
/// `X_t = Σ_{k=1}^{p} A_k · X_{t-k}  +  ε_t  +  Σ_{k=1}^{q} B_k · ε_{t-k}`
/// where `A_k`, `B_k` are user-supplied m×m operator kernels (column-major flat `Vec<f64>`)
/// and `ε_t` are i.i.d. N(0, I) innovations.
///
/// The first `burn_in` curves are discarded to approximate stationarity.
/// **The user is responsible for supplying operators with spectral radius < 1;**
/// non-stationary operators cause divergent output (detected via NaN/Inf guard).
///
/// # R baseline divergence
///
/// `freqdom::fts.rar` accepts a user-supplied covariance matrix σ for innovations.
/// This implementation uses identity covariance (i.i.d. N(0,1) per grid point).
```

**RNG seeding pattern** (`simulation.rs` lines 311–313):
```rust
// For sim_fvarma / sim_farma: seed: u64 (not Option<u64>) — mandatory seed per FTS convention.
// Mirror: fts/acf.rs functional_acf uses seed: u64; sim_kl uses Option<u64>.
// New FTS simulators use seed: u64 (no entropy fallback, per RESEARCH.md §acf.rs line 258).
let mut rng = StdRng::seed_from_u64(seed);
let normal = Normal::new(0.0, 1.0).expect("valid distribution parameters");
```

**Normal sampling pattern** (`simulation.rs` lines 316–321):
```rust
let normal = Normal::new(0.0, 1.0).expect("valid distribution parameters");
// ...
let xi: f64 = rng.sample::<f64, _>(normal);
```

**Column-major FdMatrix assembly pattern** (`simulation.rs` lines 330–345):
```rust
// Assemble output after burn-in: rows = curves, cols = grid points (column-major).
let data: Vec<f64> = /* collect n * m values in column-major order */;
FdMatrix::from_column_major(data, n, m).expect("dimension invariant: data.len() == n * m")
```

**Matrix-vector product for AR/MA kernel** (RESEARCH.md Code Examples):
```rust
// ar_ops[k] is flat m×m column-major; apply to history curve (m-vector) x_prev.
// element (j1, j2) of A_k is at ar_ops[k][j1 + j2 * m].
for j1 in 0..m {
    let mut s = 0.0;
    for j2 in 0..m {
        s += ar_ops[k][j1 + j2 * m] * history_x[k][j2];
    }
    x_new[j1] += s;
}
```

**Error handling for dimension check** (`error.rs` convention):
```rust
if ar_ops.iter().any(|op| op.len() != m * m) {
    return Err(FdarError::InvalidDimension {
        parameter: "ar_ops",
        expected: format!("{} elements (m×m = {}×{})", m * m, m, m),
        actual: format!("{} elements in one or more kernels", ar_ops.iter().map(|op| op.len()).min().unwrap_or(0)),
    });
}
```

**NaN/Inf guard after burn-in** (RESEARCH.md Pitfall 5):
```rust
if current_curve.iter().any(|v| !v.is_finite()) {
    return Err(FdarError::ComputationFailed {
        operation: "sim_fvarma burn-in",
        detail: "curve values diverged to NaN/Inf during burn-in; ensure AR operators have spectral radius < 1".to_string(),
    });
}
```

**Result structs for simulation.rs** — place immediately before `sim_fvarma`, follow `EFunType`/`EValType` derive style (lines 30–42) but add `#[non_exhaustive]` and serde gate per `*Result` convention:
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FvarmaResult {
    /// Simulated curve series, shape N × m.
    pub curves: FdMatrix,
    /// AR order p (number of AR operator kernels supplied).
    pub ar_order: usize,
    /// MA order q (number of MA operator kernels supplied).
    pub ma_order: usize,
    /// Burn-in length used.
    pub burn_in: usize,
}

#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FarmaResult {
    /// Simulated FARMA curve series, shape N × m.
    pub curves: FdMatrix,
    /// AR order p.
    pub ar_order: usize,
    /// MA order q.
    pub ma_order: usize,
    /// Burn-in length used.
    pub burn_in: usize,
}
```

**Required inline tests** for simulation.rs:
- Oracle 4: `sim_fvarma` with zero AR operator (m×m zeros) produces ~i.i.d. curves (near-zero lag-1 ACF).
- Oracle 5: Two calls with same seed produce bit-identical `FvarmaResult.curves`. Same for `sim_farma`.
- Oracle 6: rank-1 AR operator with coefficient 0.8 → `autocovariance_matrix` at h=1 has ‖C_1‖ > 0.1‖C_0‖.

---

### `fdars-core/src/lib.rs` (barrel modification)

**Analog:** `fdars-core/src/lib.rs` itself.

**pub mod declaration style** (`lib.rs` lines 93, 106) — both `fts` and `simulation` are already declared as `pub mod`; no new `pub mod` line needed. Only add `pub use` for the new public items.

There is no existing `pub use fts::...` at the crate root (the module is re-exported as `pub mod fts` — users access via `fdars_core::fts::spectral_density`). Check whether the milestone convention is to add flat crate-root re-exports or leave them under `fts::*`. Based on CONTEXT.md ("Re-export new public items at the crate root") and RESEARCH.md architecture diagram, add:

```rust
// In lib.rs, after or alongside existing module declarations:
pub use fts::{
    dpca, dpca_reconstruct, spectral_density,
    SpectralDensityResult, DpcaResult, DpcaReconstruction,
};
pub use simulation::{sim_fvarma, sim_farma, FvarmaResult, FarmaResult};
```

Check whether other `fts` functions (`functional_acf` etc.) are currently re-exported at crate root before adding to confirm placement. If they are not, follow the same pattern as `fts` items that are already re-exported (or add without flat re-export and document per the `prelude` convention).

---

## Shared Patterns

### Error Handling
**Source:** `fdars-core/src/error.rs` (used in `acf.rs` lines 28–39)
**Apply to:** `spectral.rs` (all entry points), `simulation.rs` (new functions)
```rust
// Dimension mismatch:
FdarError::InvalidDimension { parameter: "argvals", expected: "...", actual: "..." }
// Parameter range:
FdarError::InvalidParameter { parameter: "bandwidth", message: "must be >= 1".to_string() }
// Numerical failure:
FdarError::ComputationFailed { operation: "sim_fvarma burn-in", detail: "...".to_string() }
```

### Column-major FdMatrix layout
**Source:** `fdars-core/src/matrix.rs` (documented convention, used throughout)
**Apply to:** all matrix access in `spectral.rs` and `simulation.rs`
- Element `(row, col)` is at index `row + col * nrows`
- `data[(i, j)]` for observation i at grid point j
- Column-major flat Vec construction: `FdMatrix::from_column_major(vec, nrows, ncols)`

### `#[must_use]` on expensive computations
**Source:** crate-wide (74+ instances)
**Apply to:** `spectral_density`, `dpca`, `dpca_reconstruct`, `sim_fvarma`, `sim_farma`
```rust
#[must_use = "returns ... result; result should be examined"]
pub fn spectral_density(...) -> Result<SpectralDensityResult, FdarError>
```

### Integration weights (Simpson)
**Source:** `fdars-core/src/helpers.rs` via `crate::helpers::simpsons_weights`
**Apply to:** `spectral.rs` (reconstruction error computation), score inner products
```rust
let weights = simpsons_weights(argvals);
// L2 inner product of two m-vectors a, b: a.iter().zip(b).zip(weights).map(|((ai,bi),wi)| ai*bi*wi).sum()
```

### Rustfft planner reuse
**Source:** `metric/fourier.rs` lines 33–36; Pitfall 6 (RESEARCH.md)
**Apply to:** `spectral.rs` `spectral_density` and DPCA IFFT step
```rust
// Create planner ONCE outside all loops; share the Arc<dyn Fft<f64>> across iterations.
let mut planner = FftPlanner::<f64>::new();
let fft_fwd = planner.plan_fft_forward(n_freq);
let fft_inv = planner.plan_fft_inverse(n_freq);
// Reuse fft_fwd.process(&mut buf) for each of the m² entries.
```

---

## No Analog Found

All files have close analogs. No entries.

---

## Metadata

**Analog search scope:** `fdars-core/src/fts/`, `fdars-core/src/metric/`, `fdars-core/src/simulation.rs`, `fdars-core/src/regression.rs`, `fdars-core/src/lib.rs`
**Files scanned:** 7 source files read in full or targeted sections
**Pattern extraction date:** 2026-08-22

---

## PATTERN MAPPING COMPLETE

**Phase:** 41 - Spectral Functional Time Series
**Files classified:** 4
**Analogs found:** 4 / 4

### Coverage
- Files with exact analog: 1 (`spectral.rs` → `acf.rs`)
- Files with role-match / self-extension analog: 3 (`mod.rs`, `simulation.rs`, `lib.rs` self-extensions)
- Files with no analog: 0

### Key Patterns Identified
- All entry points copy `validate_fts_input` verbatim from `acf.rs` lines 25–42 (not promoted to `pub(super)`).
- `autocovariance_matrix` is already `pub(crate)` in `acf.rs:73`; `spectral.rs` calls it directly via `super::acf::autocovariance_matrix`.
- Bandwidth resolution and guard (`max_h = resolved_bandwidth.min(n - 1)`) copied from `acf.rs` lines 684–708.
- FFT planner pattern from `metric/fourier.rs` lines 33–36: create once outside loops, reuse `Arc<dyn Fft<f64>>`.
- `SymmetricEigen` pattern from `acf.rs` lines 337–348: `DMatrix::from_column_slice` + defensive symmetrize + `nalgebra::SymmetricEigen::new` + sort descending.
- All Result structs derive `Debug, Clone, PartialEq`, are serde-gated with `#[cfg_attr(feature = "serde", ...)]`, and carry `#[non_exhaustive]` — pattern from `fts/mod.rs` lines 33–35.
- `sim_fvarma`/`sim_farma` use `seed: u64` (not `Option<u64>`) per FTS convention (`fts/acf.rs` line 258), contrasting with `sim_kl`'s `Option<u64>`.
- Column-major matrix-vector product for AR/MA kernel: `a_k[j1 + j2 * m] * x[j2]` accumulated into `x_new[j1]`.

### File Created
`/home/simonm/projects/rust/fdars/.planning/phases/41-spectral-functional-time-series/41-PATTERNS.md`

### Ready for Planning
Pattern mapping complete. Planner can now reference analog patterns in PLAN.md files.
