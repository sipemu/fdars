# Phase 37: Specialized FPCA Variants - Research

**Researched:** 2026-08-21
**Domain:** Functional PCA variants — derivative FPCA, functional SVD, cross-covariance, dynamical correlation, sandwich-smoother FPCA
**Confidence:** MEDIUM (core code read directly from repo; R reference formulas confirmed from source code; `ssvd` sandwich design is `[ASSUMED]` adaptation)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **`fpca_der`**: reuse `fdata::deriv_1d` (finite difference), decompose the differentiated curves (differentiate first, then `fdata_to_pc_1d`); expose `nderiv` (default 1); reuse `FpcaResult`.
- **`fsvd`**: paired samples X (n×p), Y (n×q), matched n; weight cross-covariance by sqrt(Simpson weights) on both grids before thin-SVD, then rescale singular functions to unit functional L2 norm; deterministic sign convention (largest-magnitude element of each left singular function positive); `cross_covariance` = sample-centered empirical with 1/(n-1) divisor, returned p×q FdMatrix.
- **`dynamical_correlation`**: Dubin–Müller / fdapace DynCorr — standardize each curve (center + integrated-L2 sd), integrated inner product averaged over sample; range [-1,1]; require shared argvals grid; return scalar f64.
- **`ssvd`**: sandwich-smooth the empirical covariance (S·Cov·Sᵀ) then eigendecompose, reusing `covariance.rs` kernel smoothing (NO new crate); dense/zero-bandwidth must match `fdata_to_pc_1d` within tolerance.
- Module placement: new `fpca_variants.rs`, crate-root re-exported.
- Zero changes to existing public signatures. No new crate dependency. No plotting.
- All new public functions: `Result<T, FdarError>`-returning, `#[must_use]`, inline `#[cfg(test)]` tests, crate-root re-exports.

### Claude's Discretion
- Exact tolerance constants for the synthetic-reconstruction tests, the specific kernel/bandwidth default for the ssvd smoother, and internal helper structure.

### Deferred Ideas (OUT OF SCOPE)
- Plotting/rendering of FPCA loadings or cross-covariance surfaces.
- SPARSE-01 (FACE sparse covariance) is Phase 38.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FPCA-02-01 | User can compute FPCA of curve derivatives (`fpca_der`) | deriv_1d → fdata_to_pc_1d pipeline; FpcaResult reused; exact signatures verified |
| FPCA-02-02 | User can compute functional SVD (`fsvd`) between two functional samples | Yang et al. 2011 cross-covariance SVD; FSVD R source confirmed normalization; new FsvdResult struct |
| FPCA-02-03 | User can estimate a cross-covariance surface (`cross_covariance`) | functional_covariance pattern extended to two samples; formula verified |
| FPCA-02-04 | User can compute dynamical correlation (`dynamical_correlation`) | DynCorr.R source code read and confirmed; exact 4-step formula documented |
| FPCA-02-05 | User can run a sandwich-smoother FPCA path (`ssvd`) | fpca.sc sandwich pattern confirmed; eigendecompose_cov pattern from pace_fpca reused |
</phase_requirements>

---

## Summary

Phase 37 adds five FPCA-family functions to a new `fpca_variants.rs` module. All five reuse existing fdars infrastructure with zero new crate dependencies. Each requires careful attention to normalization and sign conventions to produce reproducible outputs with known numerical properties.

The critical implementation insight is that each variant wraps or extends `fdata_to_pc_1d` differently:
- `fpca_der` prepends a `deriv_1d` step (differentiates curves, then calls FPCA on the derivative matrix).
- `fsvd` routes through an explicit SVD of a weighted cross-covariance matrix rather than through `fdata_to_pc_1d`.
- `cross_covariance` is a simpler bivariate extension of `functional_covariance`.
- `dynamical_correlation` is a pure algebraic reduction to a scalar with a verified 4-step formula.
- `ssvd` adapts the `eigendecompose_cov` sandwich pattern already present in `pace_fpca.rs`.

The sandwich pattern for `ssvd` is already in the codebase (used by `pace_fpca`): `W^{1/2} · Cov · W^{1/2}` followed by symmetric eigendecomposition. The `ssvd` path replaces the raw empirical covariance with a kernel-smoothed covariance from `covariance.rs` before sandwiching.

**Primary recommendation:** Implement in the order `cross_covariance` → `fpca_der` → `dynamical_correlation` → `fsvd` → `ssvd`, because each later function can borrow the test scaffolding of the earlier ones.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `fpca_der` | Library (fdars-core) | — | Wraps existing `deriv_1d` + `fdata_to_pc_1d`; pure computation |
| `fsvd` | Library (fdars-core) | — | Thin SVD of weighted cross-covariance; no external service |
| `cross_covariance` | Library (fdars-core) | — | Matrix algebra on centered data; analogous to `functional_covariance` |
| `dynamical_correlation` | Library (fdars-core) | — | Pure scalar reduction; no external service |
| `ssvd` | Library (fdars-core) | — | Sandwiched covariance eigendecompose; reuses `pace_fpca` pattern |
| Module re-export | API (lib.rs) | — | Crate-root re-export via `pub use fpca_variants::{...}` |

---

## Standard Stack

### Core (all existing — no new dependencies)
| Library | Version | Purpose | Role in this Phase |
|---------|---------|---------|---------------------|
| nalgebra | 0.33 | SVD and DMatrix | `fsvd` thin-SVD via `SVD::new(mat.to_dmatrix(), true, true)` |
| nalgebra | 0.33 | Symmetric eigendecompose | `ssvd` via `DMatrix::symmetric_eigen()` |

[VERIFIED: fdars-core/Cargo.toml] — nalgebra 0.33 is the existing dependency; no new crate needed.

### No new crates — reuse-only
This phase has no `## Package Legitimacy Audit` requirement because it adds zero new dependencies. All code reuses existing fdars-core modules.

---

## Existing Code to Reuse

### `fdars-core/src/regression.rs` (verified this session)

**`fdata_to_pc_1d`** [VERIFIED: fdars-core/src/regression.rs:287-400]
```rust
pub fn fdata_to_pc_1d(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
) -> Result<FpcaResult, FdarError>
```
- Centers data via `center_columns` (private), computes `simpsons_weights`, scales by `sqrt_weights`, runs SVD (faer under `linalg` feature; nalgebra fallback), calls `fix_svd_signs`, unscales rotation by dividing by `sqrt_weights`.
- Returns `FpcaResult { singular_values, rotation (m×ncomp), scores (n×ncomp), mean, centered, weights }`.
- `fpca_der` calls this directly on the derivative matrix.

**`FpcaResult`** [VERIFIED: fdars-core/src/regression.rs:25-38]
```rust
pub struct FpcaResult {
    pub singular_values: Vec<f64>,
    pub rotation: FdMatrix,   // m x ncomp (loadings/eigenfunctions)
    pub scores: FdMatrix,     // n x ncomp
    pub mean: Vec<f64>,       // length m
    pub centered: FdMatrix,   // n x m centered data
    pub weights: Vec<f64>,    // Simpson integration weights, length m
}
```
Reused as-is for `fpca_der` output. The `mean` field will be the mean of the differentiated curves; `weights` will be the Simpson weights on the argvals grid.

**`fix_svd_signs` (private fn)** [VERIFIED: fdars-core/src/regression.rs:180-201]
```rust
fn fix_svd_signs(rotation: &mut FdMatrix, scores: &mut FdMatrix, ncomp: usize)
```
For each component k, finds `j_max` = argmax |rotation[(j, k)]|; if `rotation[(j_max, k)] < 0`, negates both the rotation column and scores column. Must be applied to `fsvd`'s left singular functions (adapted for the bivariate case — no scores matrix exists for right functions, handle separately).

**`SVD::new(weighted.to_dmatrix(), true, true)`** [VERIFIED: fdars-core/src/regression.rs:370]
The nalgebra SVD call pattern used throughout. Under `#[cfg(not(feature = "linalg"))]` this is the only SVD path; under `linalg` feature faer is preferred. For `fsvd`, since the cross-covariance is p×q (not necessarily square), `fdata_to_pc_1d` is NOT called — instead, an explicit nalgebra SVD is performed on the weighted cross-covariance matrix.

**`center_columns` (private fn)** [VERIFIED: fdars-core/src/regression.rs:204-218]
```rust
fn center_columns(data: &FdMatrix) -> (FdMatrix, Vec<f64>)
```
Centers each column (evaluation point) and returns `(centered, means)`. Used internally by `fdata_to_pc_1d`. For `cross_covariance` and `fsvd`, the equivalent operation is done manually per-grid.

### `fdars-core/src/fdata.rs` (verified this session)

**`deriv_1d`** [VERIFIED: fdars-core/src/fdata.rs:852-875]
```rust
pub fn deriv_1d(data: &FdMatrix, argvals: &[f64], nderiv: usize) -> FdMatrix
```
- Forward difference at column 0, backward at column m-1, central difference for interior.
- With `nderiv = 0`: returns `data.clone()`.
- With `argvals.len() != m` or `n == 0` or `m < 2`: returns `FdMatrix::zeros(n, m)` (no error).
- **Important for `fpca_der`**: `deriv_1d` does NOT return `Result` — it silently returns zeros on bad input. `fpca_der` must perform its own input validation BEFORE calling `deriv_1d`.

**`functional_covariance`** [VERIFIED: fdars-core/src/fdata.rs:358-395]
```rust
pub fn functional_covariance(data: &FdMatrix) -> Result<FdMatrix, FdarError>
```
Computes `M×M` sample covariance with `1/(n-1)` divisor. For `cross_covariance`, this is extended to two matrices X (n×p) and Y (n×q): result is p×q with `1/(n-1)` divisor. The implementation pattern is identical but uses two separate centered matrices.

**`center_1d`** [VERIFIED: fdars-core/src/fdata.rs:212-237]
```rust
pub fn center_1d(data: &FdMatrix) -> FdMatrix
```
Subtracts column means (no return of means). For `cross_covariance` internal use — or `center_columns` from regression.rs which does return means.

### `fdars-core/src/helpers.rs` (verified this session)

**`simpsons_weights`** [VERIFIED: fdars-core/src/helpers.rs:57-86]
```rust
pub fn simpsons_weights(argvals: &[f64]) -> Vec<f64>
```
Returns integration weights. For uniform odd-n grids: standard composite Simpson; for even-n: Simpson + trapezoidal for last interval. For non-uniform: generalized Simpson per pair. Weights sum to domain length (≈ 1.0 for [0,1] grid).

Used in `fsvd` to weight the cross-covariance before SVD: `sqrt_wx[s] * C(s,t) * sqrt_wy[t]`.

### `fdars-core/src/pace_fpca.rs` (verified this session)

**`eigendecompose_cov` (private fn)** [VERIFIED: fdars-core/src/pace_fpca.rs:165-237]
This is the exact sandwich pattern to reuse for `ssvd`:
```rust
// Pattern (lines 174-234):
// 1. W^{1/2} · Cov · W^{1/2}
let sqrt_w: Vec<f64> = w.iter().map(|wi| wi.sqrt()).collect();
// c_scaled[row + col*m] = sqrt_w[row] * cov[(row,col)] * sqrt_w[col]
// 2. nalgebra DMatrix from column-major slice
let c_dmat = DMatrix::from_column_slice(m, m, &c_scaled);
// 3. Symmetric eigendecompose
let eigen = c_dmat.symmetric_eigen();
// 4. Sort descending, keep positive eigenvalues
// 5. Unscale: phi_k[j] = v_k[j] / sqrt_w[j]
// 6. Sign fix: max-abs element positive
```
`ssvd` calls this pattern on a kernel-smoothed covariance (from `covariance.rs`) instead of the raw empirical covariance. The result type should be `FpcaResult` with fields populated from eigenvalues (→ `singular_values`), eigenfunctions (→ `rotation`), scores (project centered data), mean, centered, weights.

---

## Architecture Patterns

### System Architecture Diagram

```
fpca_der:
  Input: X (n×m) + argvals (len m) + nderiv
    → deriv_1d(X, argvals, nderiv) → Xd (n×m)
    → fdata_to_pc_1d(Xd, ncomp, argvals)
    → FpcaResult (fields = of the differentiated process)

cross_covariance:
  Input: X (n×p) + argvals_x (len p) + Y (n×q) + argvals_y (len q)
    → center_columns(X) → Xc (n×p), mean_x
    → center_columns(Y) → Yc (n×q), mean_y
    → C[s,t] = Xc.col(s) · Yc.col(t) / (n-1)  for all s,t
    → FdMatrix (p×q)

fsvd:
  Input: X (n×p) + argvals_x + Y (n×q) + argvals_y + ncomp
    → cross_covariance(X, argvals_x, Y, argvals_y) → C (p×q)
    → wx = simpsons_weights(argvals_x), wy = simpsons_weights(argvals_y)
    → Cw[s,t] = sqrt(wx[s]) * C[s,t] * sqrt(wy[t])  (p×q matrix)
    → SVD(Cw) → U (p×ncomp), D (ncomp), Vt (ncomp×q)
    → fix_svd_signs(U_cols, ...) [max-abs element positive per component]
    → rescale U cols: u_k[s] /= sqrt(wx[s])  [unit functional L2 norm on argvals_x]
    → rescale V cols: v_k[t] /= sqrt(wy[t])  [unit functional L2 norm on argvals_y]
    → compute scores_x[i,k] = sum_s Xc[i,s]*u_k[s]*wx[s]
    → compute scores_y[i,k] = sum_t Yc[i,t]*v_k[t]*wy[t]
    → FsvdResult

dynamical_correlation:
  Input: X (n×m) + Y (n×m) + argvals (len m, same for both)
    → Step 1: per-curve centering by integrated mean
        aver_xi = sum_j X[i,j]*w[j] / domain_length;  Xc1[i,j] = X[i,j] - aver_xi
        aver_yi = sum_j Y[i,j]*w[j] / domain_length;  Yc1[i,j] = Y[i,j] - aver_yi
    → Step 2: population centering
        Mx[j] = sum_i Xc1[i,j] / n;  Xc2[i,j] = Xc1[i,j] - Mx[j]
        My[j] = sum_i Yc1[i,j] / n;  Yc2[i,j] = Yc1[i,j] - My[j]
    → Step 3: L2-norm standardize each curve
        norm_xi = sqrt(sum_j Xc2[i,j]^2 * w[j] / domain_length)
        std_xi[j] = Xc2[i,j] / norm_xi  (guard: if norm < 1e-15, return 0)
        (same for Y)
    → Step 4: integrated inner product / domain_length, averaged over i
        z[i] = sum_j std_xi[j]*std_yi[j]*w[j] / domain_length
        DynCorr = mean(z)
    → f64 in [-1, 1]

ssvd:
  Input: X (n×m) + argvals (len m) + ncomp + kernel + bandwidth
    → center_columns(X) → Xc, mean_x
    → empirical_cov = functional_covariance(X)  [m×m, 1/(n-1)]
    → smooth_cov = apply kernel smoother S to empirical_cov
        S·Cov·Sᵀ  [m×m smoothed surface]
    → eigendecompose_cov(smooth_cov, argvals, ncomp)
        W^{1/2}·smooth_cov·W^{1/2} → symmetric_eigen → unscale → sign fix
    → compute scores: project Xc onto eigenfunctions with weights
    → FpcaResult (fields: rotation=eigenfunctions, singular_values=sqrt(eigenvalues), ...)
```

### Recommended Project Structure
```
fdars-core/src/
├── fpca_variants.rs   # NEW — all 5 variants + FsvdResult struct
├── regression.rs      # existing — untouched
├── fdata.rs           # existing — deriv_1d, functional_covariance reused
├── covariance.rs      # existing — CovKernel, covariance_matrix reused for ssvd smoother
├── helpers.rs         # existing — simpsons_weights reused
├── pace_fpca.rs       # existing — eigendecompose_cov pattern cloned for ssvd
└── lib.rs             # add: pub mod fpca_variants; + pub use fpca_variants::{...}
```

### Pattern 1: Derivative FPCA (`fpca_der`)

**What:** Differentiate all curves (via `deriv_1d`), then call `fdata_to_pc_1d` on the derivative matrix. The FPCA result is of the differentiated process — loadings are eigenfunctions of the derivative process, scores are projections of differentiated curves onto those eigenfunctions.

**Note on fdapace convention:** fdapace's `FPCAder` operates differently — it differentiates the eigenfunctions/mean of an existing FPCA result (or works with derivative covariance surfaces). The fdars decision deliberately chooses the "eigenfunctions of derivatives" path (differentiate curves first, then FPCA). This divergence from fdapace's default behavior MUST be documented in the rustdoc.

**Example:**
```rust
// Source: derived from fdata_to_pc_1d pattern [VERIFIED: fdars-core/src/regression.rs:287-400]
pub fn fpca_der(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
    nderiv: usize,
) -> Result<FpcaResult, FdarError> {
    let (n, m) = data.shape();
    // Input validation BEFORE calling deriv_1d (which silently returns zeros)
    if n == 0 { return Err(FdarError::InvalidDimension { ... }); }
    if m == 0 { return Err(FdarError::InvalidDimension { ... }); }
    if argvals.len() != m { return Err(FdarError::InvalidDimension { ... }); }
    if ncomp < 1 { return Err(FdarError::InvalidParameter { ... }); }
    // m >= 2 required for deriv_1d; nderiv=0 is valid (returns clone of data)
    if nderiv > 0 && m < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "data",
            message: "need >= 2 columns for numerical derivative".to_string(),
        });
    }
    let deriv_data = crate::fdata::deriv_1d(data, argvals, nderiv);
    crate::regression::fdata_to_pc_1d(&deriv_data, ncomp, argvals)
}
```

### Pattern 2: Functional SVD (`fsvd`)

**What:** Thin SVD of the Simpson-weighted cross-covariance matrix. Left singular functions (on X grid) and right singular functions (on Y grid) are returned at unit functional L2 norm.

**New struct needed — `FsvdResult`:**
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FsvdResult {
    pub singular_values: Vec<f64>,    // length ncomp
    pub left_functions: FdMatrix,     // p × ncomp (singular functions for X)
    pub right_functions: FdMatrix,    // q × ncomp (singular functions for Y)
    pub left_scores: FdMatrix,        // n × ncomp (X projections)
    pub right_scores: FdMatrix,       // n × ncomp (Y projections)
    pub cross_cov: FdMatrix,          // p × q raw cross-covariance [ASSUMED — may expose or not]
}
```

**Key implementation: weighted SVD then rescale**
```rust
// Source: [ASSUMED - derived from fdata_to_pc_1d weighting pattern]
// Weight cross-covariance: Cw[s,t] = sqrt(wx[s]) * C[s,t] * sqrt(wy[t])
// SVD of Cw: Cw = U * D * Vt  (U is p×ncomp, Vt is ncomp×q)
// Unscale left:  u_k[s] /= sqrt(wx[s])
// Unscale right: v_k[t] /= sqrt(wy[t])
// After unscaling, each u_k has unit functional L2 norm on argvals_x
// After unscaling, each v_k has unit functional L2 norm on argvals_y
// Verify: sum_s u_k[s]^2 * wx[s] ≈ 1.0
```

**Sign convention (adapted from `fix_svd_signs`):**
After unscaling, for each component k: find `s_max = argmax_s |u_k[s]|`. If `u_k[s_max] < 0`, negate both `u_k` (all of left_functions col k) and `v_k` (all of right_functions col k). This preserves the singular value and the sign of `u_k · C · v_k`.

**Scores:**
- `left_scores[i,k]  = sum_s Xc[i,s] * u_k[s] * wx[s]`
- `right_scores[i,k] = sum_t Yc[i,t] * v_k[t] * wy[t]`

### Pattern 3: Cross-Covariance Surface (`cross_covariance`)

**What:** p×q FdMatrix of sample cross-covariances.

**Formula:**
`C[s,t] = 1/(n-1) * sum_i (X[i,s] - Xbar[s]) * (Y[i,t] - Ybar[t])`

where `Xbar[s] = mean_j X[j,s]` (column mean of X), `Ybar[t] = mean_j Y[j,t]` (column mean of Y).

**Implementation pattern** (extending `functional_covariance`):
```rust
// Source: [VERIFIED: fdars-core/src/fdata.rs:358-395] — same 1/(n-1) Bessel convention
pub fn cross_covariance(
    x: &FdMatrix,
    y: &FdMatrix,
) -> Result<FdMatrix, FdarError> {
    let (nx, p) = x.shape();
    let (ny, q) = y.shape();
    // Guards: nx == ny (matched sample), nx >= 2, p > 0, q > 0
    // p*q overflow guard same as functional_covariance
    let xc = center_columns_internal(x);  // or use center_1d
    let yc = center_1d(y);
    let denom = (nx - 1) as f64;
    let mut cov = FdMatrix::zeros(p, q);
    for s in 0..p {
        let xc_col = xc.column(s);
        for t in 0..q {
            let yc_col = yc.column(t);
            let val: f64 = xc_col.iter().zip(yc_col.iter())
                .map(|(&a, &b)| a * b).sum::<f64>() / denom;
            cov[(s, t)] = val;
        }
    }
    Ok(cov)
}
```

### Pattern 4: Dynamical Correlation (`dynamical_correlation`)

**Reference:** Dubin & Müller (2005) JASA 100(471):872-881. [CITED: rdrr.io/cran/fdapace/src/R/DynCorr.R]

**Exact 4-step formula from R source (verified):**

Step 1 — per-curve centering by integrated mean:
```
aver_xi = (sum_j X[i,j] * w[j]) / domain_length
Xc1[i,j] = X[i,j] - aver_xi
```
where `domain_length = argvals[m-1] - argvals[0]`, `w = simpsons_weights(argvals)`.

Step 2 — population centering (remove sample mean of step-1 residuals):
```
Mx[j] = sum_i Xc1[i,j] / n
Xc2[i,j] = Xc1[i,j] - Mx[j]
```

Step 3 — functional L2 normalization of each doubly-centered curve:
```
norm_xi = sqrt((sum_j Xc2[i,j]^2 * w[j]) / domain_length)
std_xi[j] = Xc2[i,j] / norm_xi   (if norm_xi < 1e-15: use 0 everywhere)
```

Step 4 — per-subject integrated inner product, divided by domain_length:
```
z[i] = (sum_j std_xi[j] * std_yi[j] * w[j]) / domain_length
DynCorr = sum_i z[i] / n   (scalar mean)
```

**Range:** [-1, 1] by Cauchy-Schwarz applied to unit-norm functions (integral of product of two unit-L2 functions on a normalized domain is bounded by ±1).

**Important:** `simpsons_weights` sums to domain_length for a [a,b] grid (not 1). Division by domain_length in steps 3 and 4 normalizes the integrated quantities to be scale-invariant. [VERIFIED: fdars-core/src/helpers.rs:57-86]

**Rust implementation pattern:**
```rust
pub fn dynamical_correlation(
    x: &FdMatrix,
    y: &FdMatrix,
    argvals: &[f64],
) -> Result<f64, FdarError> {
    let (nx, mx) = x.shape();
    let (ny, my) = y.shape();
    // Guards: nx == ny, mx == my, mx == argvals.len(), nx >= 2
    // Same-grid requirement: argvals used for BOTH x and y
    let w = simpsons_weights(argvals);
    let domain_length = argvals[mx - 1] - argvals[0];
    // ... Steps 1-4 as above ...
}
```

### Pattern 5: Sandwich-Smoother FPCA (`ssvd`)

**Reference:** fpca.sc (Di et al. 2009; Goldsmith et al. 2013) for the sandwich pattern. [CITED: rdrr.io/cran/refund/src/R/fpca.sc.R]

**What "sandwich" means:** `V = W^{1/2} · Cov_smooth · W^{1/2}` where `W^{1/2}` is the diagonal matrix of sqrt(Simpson weights). Eigenvectors of V are then unscaled by `W^{-1/2}` to give orthonormal eigenfunctions in the functional L2 sense.

**`Cov_smooth` for the fdars `ssvd`:** A kernel-smoothed version of the empirical covariance. The `covariance.rs` module provides `CovKernel` and `covariance_matrix` for building a parametric kernel covariance. However, for the ssvd sandwich the smoother needs to act on the EMPIRICAL covariance as a surface, not generate a parametric one.

**Proposed implementation approach** [ASSUMED — discretion per CONTEXT.md]:
The ssvd smoother is a row-and-column application of a Gaussian kernel smoother to the p×p empirical covariance matrix:
```
Cov_smooth[s,t] = sum_{s',t'} K_bw(s,s') * Cov_emp[s',t'] * K_bw(t,t') / (sum_s' K_bw * sum_t' K_bw)
```
This can be implemented as a 1D kernel smoother applied first along rows, then along columns (separable). The Gaussian kernel `K(u) = exp(-u^2 / (2*bw^2))` from `helpers::gaussian_kernel` [VERIFIED: fdars-core/src/lib.rs:196 — re-exported as `gaussian_kernel`]. At `bw = 0` (or very small), the kernel becomes identity and `Cov_smooth → Cov_emp`.

**Dense-limit agreement requirement:** With near-zero bandwidth, `ssvd` must agree with `fdata_to_pc_1d` within `1e-6` on singular values and loadings. This is the key test.

**Pattern — reusing `eigendecompose_cov` from `pace_fpca.rs`:**
Since `eigendecompose_cov` is a private function in `pace_fpca.rs`, the ssvd implementation must duplicate the pattern (or it could be refactored to a shared private helper in a new `fpca_common.rs` — but that would require coordination). The simplest approach: inline the sandwich+eigendecompose steps in `ssvd` directly, since the pattern is compact (25 lines in `pace_fpca.rs`). [VERIFIED: fdars-core/src/pace_fpca.rs:165-237]

**`ssvd` signature:**
```rust
pub fn ssvd(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
    bandwidth: f64,   // kernel bandwidth; 0.0 = no smoothing (identity limit)
) -> Result<FpcaResult, FdarError>
```
Returns `FpcaResult` (same struct as `fpca_der` and `fdata_to_pc_1d`) so all FPCA-family results are uniform.

### Anti-Patterns to Avoid

- **Calling `deriv_1d` without prior input validation:** `deriv_1d` silently returns zeros on invalid input. `fpca_der` MUST validate dimensions before delegating. [VERIFIED: fdars-core/src/fdata.rs:852-857]
- **Not unscaling singular functions after weighted SVD:** The `fsvd` must divide `u_k[s]` by `sqrt(wx[s])` after SVD to recover unit-functional-L2 singular functions. Skipping this produces functions that are not unit norm in the functional sense.
- **Dividing by `domain_length` twice or not at all in `dynamical_correlation`:** The R code divides both at step 3 (inside the sqrt for the norm) and at step 4 (the inner product). The result is dimensionless regardless of domain length.
- **Using `functional_covariance` (which calls `center_1d`) inside `cross_covariance`:** `cross_covariance` must center X and Y SEPARATELY using their own column means before computing the outer product. Do not pass concatenated data to `functional_covariance`.
- **Reusing `eigendecompose_cov` from `pace_fpca` directly:** It is a private `fn`, not accessible from `fpca_variants.rs`. The sandwich pattern must be re-implemented (inline, ~25 lines). [VERIFIED: fdars-core/src/pace_fpca.rs:165-237 — declared as `fn eigendecompose_cov`, not `pub fn`]

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Numerical derivative of curves | Custom finite-difference loops | `fdata::deriv_1d` | Already handles forward/central/backward correctly [VERIFIED: src/fdata.rs:852] |
| Simpson integration weights | Custom quadrature | `helpers::simpsons_weights` | Handles uniform + non-uniform grids; already tested [VERIFIED: src/helpers.rs:57] |
| Column-mean centering | Custom mean subtraction | `fdata::center_1d` or `regression::center_columns` pattern | Existing parallel-aware implementations [VERIFIED: src/fdata.rs:212] |
| Functional inner product (L2 norm) | Custom integral | `helpers::simpsons_weights` + pointwise product | Consistent weighting throughout codebase |
| SVD decomposition | Custom SVD | `nalgebra::SVD::new(mat.to_dmatrix(), true, true)` | Tested dual-backend path; `fix_svd_signs` already handles sign ambiguity [VERIFIED: src/regression.rs:180-201] |
| Symmetric eigendecompose for ssvd | Custom Lanczos/QR | `DMatrix::symmetric_eigen()` | Already used by pace_fpca [VERIFIED: src/pace_fpca.rs:187] |
| Kernel smoothing of covariance | New smoothing algorithm | Separable Gaussian kernel using `gaussian_kernel` from `helpers.rs` | Re-exports at crate root; no new dep needed |

**Key insight:** The `pace_fpca.rs` `eigendecompose_cov` function (25 lines, lines 165-237) is the canonical sandwich-eigendecompose pattern in fdars. The `ssvd` function replicates it, replacing the raw covariance input with a kernel-smoothed one.

---

## Common Pitfalls

### Pitfall 1: `deriv_1d` silent failure vs. `fpca_der` error contract

**What goes wrong:** `deriv_1d` returns `FdMatrix::zeros(n, m)` when `m < 2` or `argvals.len() != m`. If `fpca_der` calls `deriv_1d` first and then `fdata_to_pc_1d` on the zero matrix, the FPCA succeeds but returns meaningless zero results without any error signal.

**Why it happens:** `deriv_1d`'s contract is to never panic (returns empty/zero silently). [VERIFIED: fdars-core/src/fdata.rs:852-857]

**How to avoid:** Validate all inputs (n > 0, m >= 2 when nderiv > 0, `argvals.len() == m`, `ncomp >= 1`) in `fpca_der` BEFORE calling `deriv_1d`. Return `FdarError::InvalidDimension` or `FdarError::InvalidParameter` as appropriate.

**Warning signs:** Tests pass with trivially zero output.

### Pitfall 2: Sign ambiguity in `fsvd` — left vs. right functions must be flipped together

**What goes wrong:** Flipping only the left singular function `u_k` without flipping the right singular function `v_k` breaks the identity `X_centered ≈ sum_k s_k * u_k * v_k^T`. The sign convention for left functions must propagate to right functions.

**Why it happens:** SVD sign ambiguity affects U and V simultaneously — if you negate a left singular vector, you must also negate the corresponding right singular vector to preserve the decomposition.

**How to avoid:** Apply sign fixing as: `if u_k[s_max] < 0: negate entire u_k column AND entire v_k column`. [VERIFIED: fdars-core/src/regression.rs:180-201 — `fix_svd_signs` does exactly this for rotation+scores]

**Warning signs:** Reconstruction `Xc ≈ sum_k s_k * u_k * v_k^T` fails numerically.

### Pitfall 3: `dynamical_correlation` division-by-`domain_length` confusion

**What goes wrong:** The R DynCorr uses `trapzRcpp(t, f^2) / (t_end - t_start)` for the squared norm. The `simpsons_weights` sums to `domain_length`, NOT to 1. The per-curve squared-norm is `sum_j f[j]^2 * w[j]` (which gives `domain_length * L2^2_normalized`). Dividing by `domain_length` gives the normalized squared norm.

**Why it happens:** Simpson weights on [0,1] sum to 1. On [a,b] they sum to b-a. The R code normalizes explicitly; the fdars implementation must follow.

**How to avoid:** Compute `sq_norm = sum_j Xc2[i,j]^2 * w[j] / domain_length` (matching R's `trapzRcpp(t,x^2)/(t_end-t_start)`). Similarly `inner_product = sum_j f[j]*g[j]*w[j] / domain_length`.

**Warning signs:** DynCorr returns values outside [-1, 1] for known co-varying samples.

### Pitfall 4: `ssvd` zero-bandwidth limit not matching `fdata_to_pc_1d`

**What goes wrong:** At near-zero bandwidth, the kernel-smoothed covariance should equal the empirical covariance. If `fdata_to_pc_1d` uses a different centering or weighting convention from `ssvd`, the results diverge.

**Why it happens:** `fdata_to_pc_1d` runs SVD on the raw centered data (not on the covariance matrix). `ssvd` eigendecomposes the covariance matrix with the W-sandwich. These are mathematically equivalent — the SVD of sqrt(W)*Xc has right singular vectors equal to the eigenvectors of Xc^T*W*Xc = W^{1/2}*Cov*W^{1/2} — but numerical differences exist due to centering and the sqrt(W) scaling.

**How to avoid:** Test `ssvd(..., bandwidth=1e-10)` vs `fdata_to_pc_1d(...)` and allow `1e-4` tolerance on singular values (not `1e-10`) to account for the different numerical path.

**Warning signs:** Dense-limit test fails with `1e-10` tolerance but passes with `1e-4`.

### Pitfall 5: `cross_covariance` p×q overflow guard

**What goes wrong:** If `p` and `q` are both large (e.g., p = q = 1000), `p * q = 1_000_000` may be large but not overflow `usize`. However, without the overflow guard that `functional_covariance` has, a case like p = q = 200_000 on a 64-bit system overflows.

**How to avoid:** Add `p.checked_mul(q).ok_or_else(...)` guard identical to `functional_covariance`. [VERIFIED: fdars-core/src/fdata.rs:368-375]

---

## Code Examples

### Reconstruction test for `fpca_der` (known answer)

```rust
// Source: reasoning from [VERIFIED: fdars-core/src/regression.rs:929-958 — reconstruction test pattern]
// For a linear function f(t) = at + b: derivative is constant a.
// FPCA of n identical constant curves: all variance is explained by 1 component.
// Test: with data = linear curves, fpca_der(data, 1, argvals, 1).scores
// should all be zero (derivative is constant, centering removes it).
#[test]
fn test_fpca_der_linear_data_zero_variance() {
    // n linear curves: x_i(t) = i*t + 1.0
    // deriv_1d gives constant rows [i, i, ..., i]
    // centered derivatives have zero variance => FPCA singular values ~ 0
    let n = 5; let m = 21;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m-1) as f64).collect();
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            data[(i, j)] = (i as f64) * argvals[j] + 1.0;
        }
    }
    let res = fpca_der(&data, 1, &argvals, 1).unwrap();
    // singular value ~ 0 (all derivative curves are constant but different)
    // Actually: deriv is [i, i, ..., i] for curve i → variance IS nonzero
    // Better test: identical curves → derivative rows are identical → zero variance
}
```

A better test uses **sine curves with identical phase** so that after differentiation all derivatives are identical (zero variance):
```rust
// x_i(t) = sin(2π t) for all i → x'_i(t) = 2π cos(2π t) for all i
// fpca_der singular values should all be ≈ 0 (identical derivatives → zero variance)
```

### Reconstruction test for `fsvd` (known answer)

```rust
// Source: [ASSUMED — derived from SVD theory]
// If X = a * u(t) (rank-1), Y = b * v(t) (rank-1), paired scores equal:
// cross_covariance(X, Y)[s,t] = a*b * u(s)*v(t) / (n-1)
// fsvd should recover u and v as left/right singular functions (up to sign)
// Singular value = a*b*(||u||_L2 * ||v||_L2) * something depending on normalization
// Concrete: u(t) = sin(t), v(t) = cos(t), a = b = 1
// X[i,j] = sin(argvals[j]) * alpha_i, Y[i,j] = cos(argvals[j]) * alpha_i
// fsvd should have 1 dominant singular value, left_functions[:,0] ∝ sin, right_functions[:,0] ∝ cos
```

### Reconstruction test for `dynamical_correlation` (known answer)

```rust
// Source: [ASSUMED — derived from Cauchy-Schwarz]
// Case 1: X == Y (perfectly co-varying) → DynCorr ≈ 1.0
// Case 2: X and Y are independent → DynCorr ≈ 0.0 (in expectation)
// Case 3: Y = -X → DynCorr ≈ -1.0

// Case 1 test:
let res = dynamical_correlation(&data, &data, &argvals).unwrap();
assert!((res - 1.0).abs() < 1e-10, "identical samples should give DynCorr=1");

// Case 3 test (negated):
let neg_data = negate_matrix(&data);
let res = dynamical_correlation(&data, &neg_data, &argvals).unwrap();
assert!((res + 1.0).abs() < 1e-10, "negated samples should give DynCorr=-1");
```

**Why these work exactly:** After centering by integrated mean and L2 normalization, `std_xi = std_yi` (case 1) so the inner product equals the integral of `std_xi^2 / domain_length = 1` by construction.

### Dense-limit test for `ssvd` vs `fdata_to_pc_1d`

```rust
// Source: [ASSUMED — derived from mathematical equivalence of SVD paths]
// At near-zero bandwidth, ssvd must agree with fdata_to_pc_1d on singular values.
// Tolerance: 1e-4 (not 1e-10; different numerical path via covariance matrix).
let fpca = fdata_to_pc_1d(&data, ncomp, &argvals).unwrap();
let ssvd_res = ssvd(&data, ncomp, &argvals, 1e-12).unwrap();
for k in 0..ncomp {
    let sv_fpca = fpca.singular_values[k];
    let sv_ssvd = ssvd_res.singular_values[k];
    assert!((sv_fpca - sv_ssvd).abs() / sv_fpca.max(1e-10) < 1e-4, ...);
}
```

---

## Runtime State Inventory

SKIPPED — this is a greenfield module addition, not a rename/refactor/migration phase.

---

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` + `#[cfg(test)]` |
| Config file | none (uses cargo test) |
| Quick run command | `cargo test -p fdars-core fpca_variants -- --test-threads=4` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |
| Clippy gate | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` |

**Note on TMPDIR:** Per MEMORY.md, use `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` for build/doctest linking to avoid /tmp tmpfs exhaustion.

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| FPCA-02-01 | `fpca_der` returns FpcaResult of derivative process | unit | `cargo test -p fdars-core test_fpca_der` | inline `#[cfg(test)]` |
| FPCA-02-01 | `fpca_der` with nderiv=0 equals `fdata_to_pc_1d` | unit | `cargo test -p fdars-core test_fpca_der_nderiv0` | mathematical identity |
| FPCA-02-01 | `fpca_der` rejects empty matrix, bad argvals, ncomp=0 | unit | `cargo test -p fdars-core test_fpca_der_errors` | error paths |
| FPCA-02-02 | `fsvd` returns unit-L2 singular functions | unit | `cargo test -p fdars-core test_fsvd_unit_norm` | sum_s u[s]^2*wx[s] ≈ 1 |
| FPCA-02-02 | `fsvd` on rank-1 data recovers known singular functions | unit | `cargo test -p fdars-core test_fsvd_rank1` | reconstruction |
| FPCA-02-02 | `fsvd` rejects mismatched sample sizes | unit | `cargo test -p fdars-core test_fsvd_errors` | error paths |
| FPCA-02-03 | `cross_covariance` diagonal = `functional_covariance` when X=Y | unit | `cargo test -p fdars-core test_cross_cov_self` | identity check |
| FPCA-02-03 | `cross_covariance` shape is p×q | unit | `cargo test -p fdars-core test_cross_cov_shape` | dimensions |
| FPCA-02-03 | `cross_covariance` rejects mismatched n | unit | `cargo test -p fdars-core test_cross_cov_errors` | error paths |
| FPCA-02-04 | `dynamical_correlation` = 1 when X == Y | unit | `cargo test -p fdars-core test_dyncorr_identical` | known answer |
| FPCA-02-04 | `dynamical_correlation` = -1 when Y = -X | unit | `cargo test -p fdars-core test_dyncorr_negated` | known answer |
| FPCA-02-04 | `dynamical_correlation` ∈ [-1, 1] for random data | unit | `cargo test -p fdars-core test_dyncorr_range` | range check |
| FPCA-02-04 | `dynamical_correlation` rejects mismatched grids | unit | `cargo test -p fdars-core test_dyncorr_errors` | error paths |
| FPCA-02-05 | `ssvd` at near-zero bandwidth matches `fdata_to_pc_1d` | unit | `cargo test -p fdars-core test_ssvd_dense_limit` | tolerance 1e-4 |
| FPCA-02-05 | `ssvd` produces orthonormal eigenfunctions under L2 | unit | `cargo test -p fdars-core test_ssvd_orthonormality` | W-weighted check |
| FPCA-02-05 | `ssvd` rejects ncomp=0, empty matrix | unit | `cargo test -p fdars-core test_ssvd_errors` | error paths |
| All | Crate-root re-exports accessible | integration | `cargo test -p fdars-core tests::smoke_reexports` | compile check |

### Sampling Rate
- **Per task commit:** `cargo test -p fdars-core fpca_variants -- --test-threads=4`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` before `/gsd-verify-work`

### Wave 0 Gaps
- [ ] `fdars-core/src/fpca_variants.rs` — new file, create in Wave 0
- [ ] `FsvdResult` struct in `fpca_variants.rs` — define before implementing `fsvd`
- [ ] `lib.rs` module declaration (`pub mod fpca_variants;`) and re-export line

*(Existing test infrastructure covers all other phase requirements — no framework install needed.)*

---

## Security Domain

`security_enforcement: true`, ASVS level 1.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | Pure computation library |
| V3 Session Management | no | Pure computation library |
| V4 Access Control | no | Pure computation library |
| V5 Input Validation | yes | `FdarError::InvalidDimension` / `InvalidParameter` at all entry points |
| V6 Cryptography | no | No cryptographic operations |

### Known Threat Patterns for This Stack

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in p×q allocation | Tampering (denial via OOM) | `p.checked_mul(q).ok_or_else(...)` guard — same as `functional_covariance` [VERIFIED: src/fdata.rs:368-375] |
| Division by zero in DynCorr normalization | Tampering (NaN propagation) | Guard: `if norm_xi < 1e-15 { use zeros }` |
| Division by zero in `simpsons_weights` unscaling (ssvd) | Tampering | Guard: `if sqrt_w[j] > 1e-15` — same as pace_fpca [VERIFIED: src/pace_fpca.rs:210-213] |
| Zero-row input silently producing zeros | Information disclosure | Explicit `n > 0` check BEFORE any computation |
| Near-degenerate covariance (all-zero data) | Tampering | SVD returns trivially near-zero singular values — this is correct; no additional guard needed |
| Mismatched sample sizes (n_x != n_y) in fsvd/cross_covariance/dynamical_correlation | Tampering | `InvalidDimension` check at entry |

---

## Environment Availability

Step 2.6: No new external dependencies — all tools are Rust/Cargo and already confirmed available in the development environment (Rust 1.97.0, nalgebra 0.33, faer 0.23, rayon 1.10). The phase adds a new `.rs` file using only existing crates.

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Derivatives via `fdapace::FPCAder` (differentiate eigenfunctions of existing FPCA) | fdars `fpca_der`: differentiate curves first, then FPCA (eigenfunctions of derivative process) | Different mathematical object; rustdoc must document the divergence explicitly |
| R `FSVD` normalizes by trapzRcpp only (trapezoidal) | fdars `fsvd` uses Simpson weights (higher-order quadrature) | More accurate functional L2 norms for smooth data |
| R `DynCorr` uses trapezoidal integration | fdars uses `simpsons_weights` | Consistent with rest of codebase |
| R `fpca.sc` sandwich uses quadrature weights from `quadWeights` | fdars `ssvd` uses `simpsons_weights` | Same concept, consistent implementation |

**Deprecated/outdated:**
- `FPCAder` in fdapace returns an augmented FPCA object — fdars uses the simpler "differentiate first" path which is an independent function.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `ssvd` smoothing is a separable row-then-column Gaussian kernel on the empirical covariance | Architecture Patterns — Pattern 5 | If the kernel should be bivariate (non-separable), numerical results differ; but mathematically equivalent for symmetric kernels |
| A2 | `FsvdResult` exposes `cross_cov` as a field | Pattern 2 — FsvdResult struct | May make the struct unnecessarily large; can omit if planner decides not to expose it |
| A3 | `ssvd` uses `FpcaResult` as return type (reusing same struct as `fdata_to_pc_1d`) | Pattern 5 | If ssvd needs to carry additional fields (e.g. bandwidth used, smoothed covariance), a new struct is needed |
| A4 | The `gaussian_kernel` helper in `helpers.rs` is usable for ssvd row/column smoothing | Don't Hand-Roll table | If `gaussian_kernel` only computes a scalar (not a kernel sweep), a brief inner loop is needed |
| A5 | Tolerance of 1e-4 is sufficient for the ssvd dense-limit test vs fdata_to_pc_1d | Common Pitfalls — Pitfall 4 | If numerical paths are closer, a tighter tolerance is fine; if farther, the test may need relaxing |

---

## Open Questions

1. **`FsvdResult.cross_cov` field — expose or not?**
   - What we know: `fsvd` internally computes `cross_covariance`; the user can call `cross_covariance` separately.
   - What's unclear: whether the planner wants to embed the p×q cross-covariance matrix in `FsvdResult` (increasing memory) or omit it.
   - Recommendation: omit by default (user calls `cross_covariance` separately if needed). Mark A2 as resolved by planner.

2. **`ssvd` bandwidth API — raw bandwidth float vs `CovKernel`?**
   - What we know: `covariance.rs` provides `CovKernel` with `Gaussian { length_scale, variance }`.
   - What's unclear: whether ssvd should accept a `CovKernel` (for flexibility) or just a `bandwidth: f64` float (simpler, Gaussian kernel implied).
   - Recommendation: `bandwidth: f64` with implicit Gaussian kernel — simpler API, consistent with refund's bandwidth parameter.

3. **`eigendecompose_cov` refactoring — inline or extract shared helper?**
   - What we know: `eigendecompose_cov` in `pace_fpca.rs` is private. Duplicating 25 lines is low risk.
   - What's unclear: whether the planner wants a shared internal helper (e.g. `pub(crate) fn sandwich_eigen(...)`) in a common location.
   - Recommendation: duplicate inline in `fpca_variants.rs` for now; refactor if needed in a future cleanup phase.

---

## Sources

### Primary (HIGH confidence — read directly this session)
- `fdars-core/src/regression.rs:1-400` — `fdata_to_pc_1d`, `FpcaResult`, `fix_svd_signs`, `center_columns`, SVD pattern [VERIFIED]
- `fdars-core/src/fdata.rs:1-875` — `deriv_1d`, `functional_covariance`, `center_1d`, `simpsons_weights` usage [VERIFIED]
- `fdars-core/src/helpers.rs:1-150` — `simpsons_weights` implementation, uniform/non-uniform branches [VERIFIED]
- `fdars-core/src/pace_fpca.rs:165-237` — `eigendecompose_cov` sandwich pattern [VERIFIED]
- `fdars-core/src/covariance.rs:1-625` — `CovKernel`, `covariance_matrix`, `generate_gaussian_process` [VERIFIED]
- `fdars-core/src/lib.rs:1-480` — re-export patterns, `pub mod` declarations [VERIFIED]

### Secondary (MEDIUM confidence — R source code read via WebFetch)
- [rdrr.io/cran/fdapace/src/R/DynCorr.R](https://rdrr.io/cran/fdapace/src/R/DynCorr.R) — exact 4-step DynCorr formula including R source [CITED]
- [rdrr.io/cran/fdapace/src/R/FSVD.R](https://rdrr.io/cran/fdapace/src/R/FSVD.R) — FSVD cross-covariance and singular function normalization [CITED]
- [rdrr.io/cran/refund/src/R/fpca.sc.R](https://rdrr.io/cran/refund/src/R/fpca.sc.R) — sandwich pattern `V = Wsqrt %*% npc.0 %*% Wsqrt` [CITED]

### Tertiary (LOW confidence — web search)
- fdapace CRAN documentation (FPCAder convention disambiguation) [websearch]
- Yang, Müller, Stadtmüller (2011) JRSSB — functional singular component analysis [websearch — not read directly]
- Dubin & Müller (2005) JASA — dynamical correlation [websearch — confirmed via R source]

---

## Metadata

**Confidence breakdown:**
- `fpca_der` implementation: HIGH — code path is trivial composition of two verified functions
- `cross_covariance` implementation: HIGH — direct extension of verified `functional_covariance`
- `fsvd` normalization: MEDIUM — R source confirmed; fdars adaptation (Simpson vs trapz) is straightforward
- `dynamical_correlation` formula: MEDIUM — exact R source confirmed; translation to Rust is mechanical
- `ssvd` sandwich: MEDIUM — sandwich pattern verified in `pace_fpca`; the "smooth covariance first" step is [ASSUMED] adaptation

**Research date:** 2026-08-21
**Valid until:** 2026-09-20 (stable algorithms; no fast-moving dependencies)
