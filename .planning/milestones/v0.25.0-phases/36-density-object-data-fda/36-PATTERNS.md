# Phase 36: Density Object-Data FDA — Pattern Map

**Mapped:** 2026-08-21
**Files analyzed:** 2 (1 new module, 1 modified lib.rs)
**Analogs found:** 2 / 2

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/src/density_fda.rs` | module (numeric, single-file) | transform + FPCA + batch | `fdars-core/src/pda.rs` (struct/doc/test pattern), `fdars-core/src/fts/mod.rs` (result-struct pattern) | role-match (exact for struct/validation/inline-test pattern) |
| `fdars-core/src/lib.rs` | crate root | config / re-export | existing `src/lib.rs` lines 130–145 | exact |

---

## Pattern Assignments

### `fdars-core/src/density_fda.rs` (new module)

**Primary analog:** `fdars-core/src/pda.rs` (single-file module, result structs, inline tests)
**Secondary analog:** `fdars-core/src/fts/mod.rs` (result struct with serde gate + `#[non_exhaustive]`)
**FPCA analog:** `fdars-core/src/regression.rs` (FpcaResult + fdata_to_pc_1d)

---

#### Module-level doc comment pattern

**Source:** `fdars-core/src/pda.rs` lines 1–44

```rust
//! Linear differential operators and principal differential analysis.
//!
//! This module provides ...
//!
//! # Relationship to the R `fda` package
//!
//! ... (cite R baseline package; document divergences here) ...
//!
//! # Examples
//!
//! ```
//! use fdars_core::density_fda::{lqd_transform, inverse_lqd};
//! // minimal smoke-test example
//! ```
```

**For density_fda.rs:** Open with `//!` block naming the R baseline (`fdadensity` 0.1.4),
the LQD mathematical reference (Petersen & Mueller 2016), and a **Divergences from fdadensity**
subsection documenting linear interpolation vs. spline and the optional weights extension.

---

#### Imports pattern

**Source:** `fdars-core/src/fts/acf.rs` lines 13–18 and `fdars-core/src/pda.rs` lines 45–47

```rust
use crate::error::FdarError;
use crate::helpers::{cumulative_trapz, linear_interp, simpsons_weights, trapz};
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
```

No external crates. No `#[cfg(feature = "linalg")]` needed in density_fda.rs itself — `fdata_to_pc_1d`
handles its own feature gating internally.

---

#### Result struct pattern

**Source:** `fdars-core/src/fts/mod.rs` lines 31–43 and 46–58

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FacfResult {
    /// Lag values (1..=max_lag).
    pub lags: Vec<u32>,
    /// Functional autocorrelation ρ_h at each lag ...
    pub acf: Vec<f64>,
    ...
}
```

**For `LqdFpcaResult`:**

```rust
/// Result of functional PCA on log-quantile-density (LQD) transformed densities.
///
/// All fields (`fpca`, scores, loadings, mean) are in **LQD space** (the uniform
/// quantile grid t ∈ [0, 1]), not in the original density space. To obtain
/// density-space variation modes, apply [`inverse_lqd`] to `fpca.mean ± scale * loading`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct LqdFpcaResult {
    /// FPCA result in LQD space.
    pub fpca: FpcaResult,
    /// Fraction of variance explained by first k components.
    ///
    /// `fve[k]` = cumsum(sv²)[k] / sum(sv²). Monotone non-decreasing;
    /// reaches 1.0 only when `ncomp == min(n_densities, n_quantile_pts)`.
    pub fve: Vec<f64>,
}
```

**Note:** `FpcaResult` is already defined in `regression.rs` lines 22–38 with fields
`singular_values`, `rotation`, `scores`, `mean`, `centered`, `weights` — import and embed, do not redefine.

---

#### Input validation pattern

**Source:** `fdars-core/src/fts/acf.rs` lines 25–42 and `fdars-core/src/regression.rs` lines 293–319

```rust
// Pattern: validate at function entry, return Err immediately, then proceed.
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

**For density_fda.rs validation gates (copy at function entry for each public fn):**

```rust
// Length mismatch
if vals.len() != argvals.len() {
    return Err(FdarError::InvalidDimension {
        parameter: "vals",
        expected: format!("{}", argvals.len()),
        actual: format!("{}", vals.len()),
    });
}
// Negative density
if vals.iter().any(|&v| v < 0.0) {
    return Err(FdarError::InvalidParameter {
        parameter: "vals",
        message: "density values must be non-negative".to_string(),
    });
}
// All-zero density (checked after non-negative, using trapz result)
let integral = trapz(vals, argvals);
if integral < 1e-15 {
    return Err(FdarError::InvalidParameter {
        parameter: "vals",
        message: "density integrates to zero or is all-zero".to_string(),
    });
}
// Non-monotone / duplicate argvals
if argvals.windows(2).any(|w| w[1] <= w[0]) {
    return Err(FdarError::InvalidParameter {
        parameter: "argvals",
        message: "argvals must be strictly increasing".to_string(),
    });
}
// Empty sample (for wasserstein_barycenter)
if density_matrix.shape().0 == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "density_matrix",
        expected: "at least 1 row".to_string(),
        actual: "0 rows".to_string(),
    });
}
```

---

#### Core function signatures

All five public entry points follow the project's `(data, [argvals,] config_params) -> Result<T, FdarError>` pattern.

```rust
/// Normalize a density to integrate to 1 via trapezoidal quadrature.
pub fn normalize_density(vals: &[f64], argvals: &[f64]) -> Result<Vec<f64>, FdarError>

/// Log-quantile-density (LQD) forward transform.
///
/// Maps `density` sampled on `argvals` to ψ on a uniform quantile grid of
/// length `n_quantile_pts` (default: `argvals.len().max(101)`).
pub fn lqd_transform(
    density: &[f64],
    argvals: &[f64],
    n_quantile_pts: Option<usize>,
) -> Result<Vec<f64>, FdarError>

/// Inverse LQD transform: reconstruct a normalized density on `target_argvals`.
pub fn inverse_lqd(
    psi: &[f64],
    t_grid: &[f64],
    target_argvals: &[f64],
) -> Result<Vec<f64>, FdarError>

/// 1D Wasserstein Fréchet mean (quantile-average barycenter) of densities.
pub fn wasserstein_barycenter(
    density_matrix: &FdMatrix,
    argvals: &[f64],
    weights: Option<&[f64]>,
) -> Result<Vec<f64>, FdarError>

/// Functional PCA of densities in LQD space.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn lqd_fpca(
    density_matrix: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    n_quantile_pts: Option<usize>,
) -> Result<LqdFpcaResult, FdarError>
```

**Note:** `#[must_use]` on `lqd_fpca` only (expensive SVD); simpler transforms do not warrant it per project convention (checked: `trapz`, `normalize_density` analogs are not `#[must_use]`).

---

#### FpcaResult reuse + FVE pattern

**Source:** `fdars-core/src/regression.rs` lines 287–321 (`fdata_to_pc_1d` call) and RESEARCH.md §Pattern 4

```rust
// In lqd_fpca — assemble LQD FdMatrix and delegate:
let n_q = n_quantile_pts.unwrap_or(argvals.len().max(101));
let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();

let mut lqd_data = FdMatrix::zeros(n_dens, n_q);
for i in 0..n_dens {
    let row: Vec<f64> = (0..m).map(|j| density_matrix[(i, j)]).collect();
    let psi = lqd_transform(&row, argvals, Some(n_q))?;
    for (j, &val) in psi.iter().enumerate() {
        lqd_data[(i, j)] = val;
    }
}

let fpca = fdata_to_pc_1d(&lqd_data, ncomp, &t_grid)?;

// FVE = cumsum(sv²) / sum(sv²)
let sv_sq: Vec<f64> = fpca.singular_values.iter().map(|&s| s * s).collect();
let total: f64 = sv_sq.iter().sum();
let mut cumsum = 0.0_f64;
let fve: Vec<f64> = sv_sq.iter().map(|&s| { cumsum += s; cumsum / total }).collect();

Ok(LqdFpcaResult { fpca, fve })
```

---

#### Error handling pattern

**Source:** `fdars-core/src/regression.rs` lines 293–319 + `fdars-core/src/error.rs` (FdarError variants)

All errors use `FdarError` variants:
- `FdarError::InvalidDimension { parameter, expected, actual }` — length mismatches, empty matrices
- `FdarError::InvalidParameter { parameter, message }` — negative density, all-zero, non-monotone grid
- `FdarError::ComputationFailed { operation, detail }` — NaN/Inf in ψ after log (log of zero that slipped validation), or `fdata_to_pc_1d` propagation

No `unwrap()` or `expect()` in public-facing code paths. Internal private helpers may use `Option<T>` bridged via `.ok_or_else()`.

---

#### Inline test pattern

**Source:** `fdars-core/src/fts/acf.rs` (inline `#[cfg(test)] mod tests`) and `fdars-core/src/pda.rs`

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::trapz;

    // Helper: truncated Gaussian density on argvals
    fn truncated_gaussian(argvals: &[f64], mu: f64) -> Vec<f64> {
        let raw: Vec<f64> = argvals.iter().map(|&x| (-(x - mu).powi(2) / 2.0).exp()).collect();
        let integral = trapz(&raw, argvals);
        raw.iter().map(|&d| d / integral).collect()
    }

    #[test]
    fn normalize_density_integral_to_one() { ... }

    #[test]
    fn error_negative_density() { ... }

    // ... one #[test] per requirement in RESEARCH.md §Test Map
}
```

All 12 test cases from RESEARCH.md §Phase Requirements → Test Map live inline in `density_fda.rs`.
No separate test file. Run with: `cargo test -p fdars-core --features linalg density_fda`.

---

### `fdars-core/src/lib.rs` (modified — re-exports only)

**Analog:** existing lib.rs lines 130–145 (recent additions for `pda`, `multi_fdata`)

**Source:** `fdars-core/src/lib.rs` lines 130–141

```rust
pub mod multi_fdata;
pub mod pda;
pub mod smooth_basis;

// Re-export multi-domain functional data container
pub use multi_fdata::{FdComponent, MultiFunData};

// Re-export linear differential operator and principal differential analysis
pub use pda::{Lfd, PdaResult, principal_differential_analysis};
```

**For density_fda additions (copy this pattern exactly):**

```rust
pub mod density_fda;

// Re-export density-valued FDA entry points and result types
pub use density_fda::{
    inverse_lqd, lqd_fpca, lqd_transform, normalize_density, wasserstein_barycenter,
    LqdFpcaResult,
};
```

Place `pub mod density_fda;` in alphabetical order among the `pub mod` declarations (between `detrend` and `distance`). Place `pub use density_fda::{ ... }` in the re-export block adjacent to the comment `// Re-export linear differential operator ...` or immediately after the `pda` re-export block.

---

## Shared Patterns

### Error construction
**Source:** `fdars-core/src/regression.rs` lines 294–319
**Apply to:** all five public functions in `density_fda.rs`

```rust
return Err(FdarError::InvalidDimension {
    parameter: "argvals",
    expected: format!("{m} elements"),
    actual: format!("{} elements", argvals.len()),
});
```

### Trapz-based normalization (one-liner)
**Source:** `fdars-core/src/helpers.rs` lines 233–240
**Apply to:** `normalize_density`, `lqd_transform` (post-normalization), `inverse_lqd` (final step)

```rust
let integral = trapz(vals, argvals);
let normed: Vec<f64> = vals.iter().map(|&v| v / integral).collect();
```

### cumulative_trapz for CDF
**Source:** `fdars-core/src/helpers.rs` lines 197–231
**Apply to:** `lqd_transform` (density → CDF), `inverse_lqd` (exp(ψ) → Q), `wasserstein_barycenter` (each row → CDF → Q_i)

Contract: `out[0] = 0.0` always. Do NOT prepend a 0; it is already there.

### linear_interp for quantile inversion
**Source:** `fdars-core/src/helpers.rs` lines 172–191
**Apply to:** `lqd_transform` (CDF → t_grid mapping), `inverse_lqd` (Q → target_argvals), `wasserstein_barycenter` (F_i → Q_i(t))

Contract: clamps at boundaries; no `Result` — safe in map iterators; binary search interior.

### Column-major FdMatrix row extraction
**Source:** `fdars-core/src/fts/acf.rs` line 49 and throughout (project-wide convention)

```rust
// Extract row i from an FdMatrix:
let row: Vec<f64> = (0..m).map(|j| density_matrix[(i, j)]).collect();
// Write row i into an FdMatrix:
for (j, &val) in psi.iter().enumerate() {
    lqd_data[(i, j)] = val;
}
```

Rows = observations/densities; columns = evaluation points. Element (i, j) is at index `i + j*nrows` in the flat Vec.

---

## No Analog Found

None. All patterns are covered by existing codebase analogs.

---

## Metadata

**Analog search scope:** `fdars-core/src/` (all .rs files)
**Key files read:** `regression.rs` (FpcaResult, fdata_to_pc_1d), `helpers.rs` (cumulative_trapz, linear_interp, trapz, simpsons_weights), `fts/mod.rs` (result struct pattern), `fts/acf.rs` (validation + inline test pattern), `pda.rs` (single-file module doc pattern), `lib.rs` (re-export pattern)
**Pattern extraction date:** 2026-08-21
