# Phase 26: PACE Sparse FPCA — Research

**Researched:** 2026-08-18
**Domain:** Functional PCA / PACE estimator (Yao–Müller–Wang 2005) for sparse/irregular functional data — pure Rust, nalgebra 0.33, no new dependency
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Eigenstructure:** symmetric eigendecomposition of the smoothed covariance surface from `irreg_fdata::cov_irreg` on the work grid (Simpson-weighted inner product), top-`ncomp` eigenpairs. NOT via `fdata_to_pc_1d` (that SVDs a *data* matrix; PACE eigendecomposes the *covariance surface*).
- **FPC scores:** conditional-expectation (BLUP/PACE) scores, newly implemented. No existing helper. Formula per curve i: `ξ_ik = λ_k · φ_ik^T · Σ_yi^{-1} · (Y_i − μ_i)`, where `Σ_yi = Φ_i diag(λ) Φ_i^T + σ² I_{n_i}` evaluated on curve i's observed points (Φ_i = eigenfunctions interpolated to those points).
- **σ²:** caller-supplied (`sigma2` in config, with a documented small default). Automatic σ² estimation deferred.
- **Confidence bands:** pointwise bands from the BLUP prediction variance (Yao et al. Ω): `Var(x̂_i(t)) = Φ(t)(diag(λ) − diag(λ)Φ_i^T Σ_yi^{-1} Φ_i diag(λ))Φ(t)^T`, 95% Gaussian (`alpha` configurable).
- **Entry point:** `pace_fpca(data: &IrregFdata, config: &PaceFpcaConfig) -> Result<PaceFpcaResult, FdarError>`, crate-root re-exported.
- **Input:** `&IrregFdata` (the existing sparse/irregular container).
- **Work grid:** caller-supplied `work_grid: &[f64]` in config.
- **`PaceFpcaConfig`:** `{ ncomp, bandwidth, sigma2, work_grid, alpha }` builder struct. Derive `Debug, Clone, PartialEq`; conditional serde.
- **`PaceFpcaResult`:** `{ mean (Vec<f64>, len m), eigenvalues (Vec<f64>, len ncomp), eigenfunctions (FdMatrix m×ncomp), scores (FdMatrix n×ncomp), fitted (FdMatrix n×m), fitted_lower (FdMatrix n×m), fitted_upper (FdMatrix n×m), argvals (work grid, Vec<f64>), sigma2 (f64, echoed), ncomp (usize) }`. `#[non_exhaustive]`; conditional serde.

### Claude's Discretion

*(None specified — all implementation choices were locked in CONTEXT.md.)*

### Deferred Ideas (OUT OF SCOPE)

- Automatic σ² estimation from the raw-vs-smoothed diagonal.
- GCV/CV bandwidth selection for the covariance surface.
- The REG-01 sparse/PACE concurrent-regression variant.
- Functional-fragment completion / trajectory extrapolation beyond the observed range.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FPCA-01 | User can fit a unified PACE sparse FPCA for sparse/irregular functional data via a new public entry point `pace_fpca` in `fdars-core/src/pace_fpca.rs` (re-exported at the crate root). Produces: kernel-smoothed mean, covariance-surface eigendecomposition (eigenvalues + eigenfunctions), conditional-expectation (BLUP/PACE) FPC scores, fitted trajectories, and pointwise confidence bands. `Result`-returning, validates inputs, no new crate dependency, additive/non-breaking. | Sections §Standard Stack, §PACE Formulas, §Implementation Mechanics, §Code Examples, §Common Pitfalls, §Validation Architecture |

</phase_requirements>

---

## Summary

Phase 26 adds a single new module `fdars-core/src/pace_fpca.rs` implementing the Yao–Müller–Wang (2005) PACE estimator for sparse/irregularly-sampled functional data. The estimator chains four existing building blocks: `irreg_fdata::mean_irreg` (kernel-smoothed mean), `irreg_fdata::cov_irreg` (smoothed covariance surface as `FdMatrix`), `nalgebra::SymmetricEigen` (symmetric eigendecomposition of the covariance matrix), and `linalg::cholesky_solve` (BLUP score computation per curve). The only genuinely new algorithmic code is the BLUP score solve and the prediction-variance band formula — neither is available anywhere in the current codebase.

The eigendecomposition is of the m×m smoothed covariance matrix (not a data matrix), which is already symmetric by construction. `nalgebra 0.33` provides `DMatrix::symmetric_eigen()` returning eigenpairs in ascending eigenvalue order; the implementer must reverse and take the top-`ncomp` pairs, apply grid-spacing scaling to convert to true functional eigenvalues, enforce orthonormal eigenfunctions (verified via Simpson weights), and fix sign ambiguity to match the `fix_svd_signs` convention used in `regression.rs`.

Per-curve BLUP requires building the n_i×n_i matrix Σ_yi for each curve (small system, Cholesky solve via `linalg::cholesky_solve`), interpolating the work-grid eigenfunctions to the curve's irregular observed points (via the already-present `irreg_fdata::linear_interp`), and applying the score formula. Confidence bands add a scalar pointwise variance computation using the same Σ_yi factorization.

**Primary recommendation:** Implement `pace_fpca.rs` as a single flat module (~300–400 lines) — config struct, result struct, entry function, and inline tests. No submodule directory needed.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Kernel-smoothed mean on work grid | `irreg_fdata::mean_irreg` | — | Already implemented; caller passes work_grid |
| Smoothed covariance surface on work grid | `irreg_fdata::cov_irreg` | — | Already implemented; returns FdMatrix m×m |
| Symmetric eigendecomposition of covariance | `nalgebra::SymmetricEigen` (via `to_dmatrix`) | — | nalgebra 0.33 supports it; no new dep |
| Eigenfunction sign fixing + normalization | `pace_fpca.rs` (new, ~20 lines) | `helpers::simpsons_weights` | Same logic as `fix_svd_signs` in regression.rs |
| BLUP score computation per curve | `pace_fpca.rs` (new, ~40 lines) | `linalg::cholesky_solve` | Core PACE novelty; no existing helper |
| Eigenfunction interpolation to observed points | `irreg_fdata::linear_interp` (pub(super)) | — | Already present; may need re-export or local copy |
| Fitted trajectory reconstruction | `pace_fpca.rs` (new, ~10 lines) | — | μ(t) + Σ_k ξ_ik φ_k(t) on work grid |
| Prediction variance bands | `pace_fpca.rs` (new, ~30 lines) | `linalg::cholesky_solve` | Ω formula; Cholesky already in linalg.rs |
| Input validation | `pace_fpca.rs` + `FdarError` | — | Standard project pattern |

---

## Standard Stack

### Core (no new dependencies — all already in Cargo.toml)

| Library | Version | Purpose | Why Used |
|---------|---------|---------|----------|
| `nalgebra` | 0.33 [VERIFIED: fdars-core/Cargo.toml] | `DMatrix::symmetric_eigen()` for covariance eigendecomposition, `to_dmatrix()` bridge | Already in the crate; provides SymmetricEigen for the m×m symmetric covariance matrix |
| `irreg_fdata::mean_irreg` | crate-internal [VERIFIED: fdars-core/src/irreg_fdata/kernels.rs:58-90] | Nadaraya-Watson kernel-smoothed mean on a target grid | Directly reused; accepts `KernelType::Gaussian` |
| `irreg_fdata::cov_irreg` | crate-internal [VERIFIED: fdars-core/src/irreg_fdata/smoothing.rs:111-138] | Local-linear kernel smoothing of the bivariate covariance surface | Core reuse; returns `FdMatrix` (ns×nt) column-major |
| `helpers::simpsons_weights` | crate-internal [VERIFIED: fdars-core/src/helpers.rs:57-86] | Integration weights for eigenfunction normalization and functional inner products | Already used by fdata_to_pc_1d; same pattern |
| `linalg::cholesky_solve` | crate-internal [VERIFIED: fdars-core/src/linalg.rs:131-134] | Solve Σ_yi · x = (Y_i − μ_i) via Cholesky for BLUP score; also used for band Ω | Already pub(crate); no allocation overhead |
| `irreg_fdata::linear_interp` | crate-internal [VERIFIED: fdars-core/src/irreg_fdata/mod.rs:193-213] | Interpolate work-grid eigenfunctions to a curve's observed t-values to build Φ_i | Boundary-clamping linear interp; pub(super) — may need `pub(crate)` promotion |

### Supporting

| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `iter_maybe_parallel!` macro | crate-internal [VERIFIED: fdars-core/src/parallel.rs (imported via `use crate::iter_maybe_parallel!`)] | Feature-gate rayon for per-curve BLUP loop | Use for outer loop over n curves; each curve is O(n_i³) Cholesky |
| `simulation::{fourier_eigenfunctions, wiener_eigenfunctions, eigenvalues_linear}` | crate-internal [VERIFIED: fdars-core/src/simulation.rs:103-279] | Test data generation with known eigenfunctions and eigenvalues | Tests only |

### Alternatives Considered

| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| `nalgebra::SymmetricEigen` | `nalgebra::SVD` on the covariance matrix | SymmetricEigen is more efficient and numerically appropriate for PSD matrices; SVD would also work but produces redundant singular vectors |
| `linalg::cholesky_solve` | `nalgebra::LU` decompose on Σ_yi | Cholesky is O(n_i³/3) vs LU O(n_i³/2) and better conditioned for PSD matrices; project already has `cholesky_solve` in `pub(crate)` linalg |

**Installation:** No new packages. All dependencies already in `fdars-core/Cargo.toml`.

---

## Package Legitimacy Audit

No external packages are being added. All libraries are crate-internal or already declared in `Cargo.toml`.

| Package | Registry | Status | Disposition |
|---------|----------|--------|-------------|
| nalgebra 0.33 | crates.io | Already declared in fdars-core/Cargo.toml [VERIFIED] | Approved (existing) |

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious:** none

---

## PACE Formulas (Yao–Müller–Wang 2005)

This section pins every formula the implementer needs. Sources: Yao, Müller & Wang (2005, JASA), Sections 2–3. [ASSUMED based on training knowledge of the paper — formulas are standard FDA textbook material, but exact notation may differ from the published paper.]

### Step 1 — Mean Function

```
μ̂(t) = Σ_{i,j} K_h(t - t_{ij}) x_{ij} / Σ_{i,j} K_h(t - t_{ij})
```

Implemented via `mean_irreg(ifd, work_grid, bandwidth, KernelType::Gaussian)`. [VERIFIED: fdars-core/src/irreg_fdata/kernels.rs:58-90 — function accepts target_argvals (= work_grid), returns Vec<f64> of length work_grid.len()]

### Step 2 — Covariance Surface

```
Ĝ(s,t) = raw kernel smoother of {(t_{ij1}, t_{ij2}, (x_{ij1}-μ̂(t_{ij1}))(x_{ij2}-μ̂(t_{ij2})))}
         excluding same-point pairs (j1 == j2) to avoid σ² inflation
```

Implemented via `cov_irreg(ifd, work_grid, work_grid, bandwidth)`. [VERIFIED: fdars-core/src/irreg_fdata/smoothing.rs:111-138 — iterates all pairs (j1, j2) including same-curve same-point pairs. CRITICAL PITFALL: `cov_irreg` does NOT exclude j1==j2 — the diagonal of the raw surface is inflated by σ². This does NOT affect PACE negatively because σ² enters only through Σ_yi; the eigendecomposition of Ĝ on the work grid is of the diagonal-inflated surface, but eigenvalues of that surface absorb the diagonal inflation as part of the total variance. The Yao 2005 PACE estimator estimates the *total* covariance (signal + noise) from the smoothed surface and then separates σ² in the BLUP step. This is correct.]

The result is an `FdMatrix` of shape (m, m) — the smoothed covariance at all (work_grid[i], work_grid[j]) pairs.

### Step 3 — Eigendecomposition

The smoothed covariance matrix Ĝ (m×m, symmetric PSD) is decomposed as:
```
Ĝ · w_j-scaled = Λ Φ Λ^T   (symmetric eigendecomposition)
```

where eigenfunctions φ_k satisfy ∫ φ_k(t)² dt = 1 (L² orthonormal on the work grid).

**Integration-weight scaling for functional eigenvalues:**

The raw numerical m×m matrix `C` from `cov_irreg` has entries C[i,j] ≈ Ĝ(s_i, t_j). To get *functional* eigenvalues (interpretable as variance explained along each component), scale by the grid spacing before eigendecomposition:

```
W = diag(simpsons_weights(work_grid))   // m×m, diagonal
C_scaled = W^{1/2} · C · W^{1/2}       // symmetric PSD
```

Eigendecompose `C_scaled` to get (eigenvalue Λ_k, eigenvector v_k). Then recover functional eigenfunctions by:
```
φ_k = W^{-1/2} · v_k            // length m
φ_k_normalized = φ_k / ||φ_k||_{L²}  // ensure ∫ φ_k² w_j dt ≈ 1
```

The functional eigenvalues are the numerical eigenvalues of `C_scaled` (they already include the grid-spacing scaling). [ASSUMED — standard approach; verified consistent with how `fdata_to_pc_1d` uses `sqrt_weights` scaling before SVD at regression.rs:326-334]

**nalgebra API:**
```rust
// Source: nalgebra 0.33 docs — SymmetricEigen
use nalgebra::DMatrix;
let c_dmatrix: DMatrix<f64> = /* ... */;  // m×m symmetric
let eigen = c_dmatrix.symmetric_eigen();
// eigen.eigenvalues: DVector<f64> in ASCENDING order
// eigen.eigenvectors: DMatrix<f64>, columns are eigenvectors
```

Take the top `ncomp` eigenpairs by reversing the ascending order (largest eigenvalue = last column).

**Truncation of negative eigenvalues:** The smoothed covariance surface from kernel estimation may produce slightly negative numerical eigenvalues (finite-sample artifact, especially for small n or large bandwidth). These must be treated as zero and the corresponding eigenfunctions discarded. Only keep eigenpairs where eigenvalue > 0. If fewer than `ncomp` positive eigenvalues exist, cap `ncomp` at the available count and document in result. [ASSUMED — standard PACE implementation practice]

**Sign convention:** Apply `fix_svd_signs`-equivalent: for each eigenfunction, find the element with largest absolute value; if it is negative, flip the entire eigenfunction. [VERIFIED: fdars-core/src/regression.rs:180-201 — verbatim convention used for SVD]

### Step 4 — BLUP Scores

For curve i with `n_i` observed points at times T_i = (t_{i1}, ..., t_{i,n_i}) and values Y_i:

```
μ_i  = [μ̂(t_{i1}), ..., μ̂(t_{i,n_i})]             // kernel mean at observed times
Φ_i  = matrix of size n_i × ncomp                    // eigenfunctions at observed times
     Φ_i[j,k] = φ_k interpolated at t_{ij}

Σ_yi = Φ_i · diag(λ_1,...,λ_ncomp) · Φ_i^T + σ² · I_{n_i}   // n_i × n_i

ξ_ik = λ_k · (Φ_i[:,k])^T · Σ_yi^{-1} · (Y_i − μ_i)
```

More efficiently: solve `Σ_yi · v = (Y_i − μ_i)` for `v` (Cholesky), then `ξ_ik = λ_k · (Φ_i[:,k])^T · v`.

**Implementation note:** Build `v = Σ_yi^{-1}(Y_i − μ_i)` once per curve (one Cholesky solve of n_i×n_i system), then `ξ_ik = λ_k · dot(Φ_i_col_k, v)` for each k. This avoids re-solving Σ_yi for each component.

**Building Σ_yi in code (row-major flat):**
```rust
let ni = n_i;
let mut sigma_yi = vec![0.0_f64; ni * ni];
for j in 0..ni {
    for l in 0..ni {
        let mut s = 0.0;
        for k in 0..ncomp {
            s += phi_i[(j, k)] * lambda[k] * phi_i[(l, k)];
        }
        sigma_yi[j * ni + l] = s;  // row-major
        if j == l { sigma_yi[j * ni + l] += sigma2; }
    }
}
```

Note: `linalg::cholesky_d` (row-major input) vs `linalg::cholesky_factor` (also row-major). Both are available. [VERIFIED: fdars-core/src/linalg.rs:16-40 for cholesky_d, 85-108 for cholesky_factor]

### Step 5 — Fitted Trajectories

```
x̂_i(t) = μ̂(t) + Σ_k ξ_ik · φ_k(t)    for t ∈ work_grid
```

Stored as `fitted` (`FdMatrix` n×m), where each row is one curve's trajectory on the work grid.

### Step 6 — Confidence Bands

Pointwise prediction variance at t ∈ work_grid (Yao et al. Ω formula):
```
Var(x̂_i(t)) = φ(t)^T · [diag(λ) − diag(λ)·Φ_i^T·Σ_yi^{-1}·Φ_i·diag(λ)] · φ(t)
```

where φ(t) = (φ_1(t), ..., φ_ncomp(t))^T.

More efficiently: let `A_i = diag(λ) · Φ_i^T · Σ_yi^{-1} · Φ_i · diag(λ)` (ncomp×ncomp, computed once per curve).
Then `Var(x̂_i(t)) = sum_{k,l} (diag(λ)[k,l] - A_i[k,l]) · φ_k(t) · φ_l(t)`.

95% Gaussian bands:
```
z_alpha = qnorm(1 - alpha/2)    // alpha=0.05 → z=1.96
lower_i(t) = x̂_i(t) − z_alpha · sqrt(max(Var(x̂_i(t)), 0))
upper_i(t) = x̂_i(t) + z_alpha · sqrt(max(Var(x̂_i(t)), 0))
```

`qnorm(p)` is not in the crate — compute 1.96 directly for alpha=0.05, or use a simple rational approximation for general `alpha`. [ASSUMED — statrs crate has `Normal::inverse_cdf` but is already a dependency per CLAUDE.md]

**Checking statrs availability:**
```bash
grep "statrs" fdars-core/Cargo.toml
```
[VERIFIED: CLAUDE.md lists `statrs` as a dependency — "Statistical distributions and functions". Use `statrs::distribution::Normal` with `ContinuousCDF::inverse_cdf` for the quantile. Alternatively hard-code `z = 1.96` for alpha=0.05 default and document.]

---

## Architecture Patterns

### System Architecture Diagram

```
User code
    │
    ▼ pace_fpca(&IrregFdata, &PaceFpcaConfig)
┌──────────────────────────────────────────────────────┐
│ pace_fpca.rs                                          │
│                                                       │
│  validate inputs (n, ncomp, bandwidth, work_grid)     │
│       │                                               │
│       ▼                                               │
│  mean_irreg(ifd, work_grid, bw, Gaussian)             │
│  → Vec<f64> (len m)                                   │
│       │                                               │
│       ▼                                               │
│  cov_irreg(ifd, work_grid, work_grid, bw)             │
│  → FdMatrix (m×m)                                     │
│       │                                               │
│       ▼                                               │
│  Weighted symmetric eigendecomposition                │
│  (W^{1/2} · C · W^{1/2}).symmetric_eigen()           │
│  → top-ncomp (λ_k, φ_k), drop negatives              │
│  → sign-fix eigenfunctions                            │
│       │                                               │
│       ▼   (per curve i, parallel over n)              │
│  interpolate φ_k to obs times → Φ_i (n_i × ncomp)   │
│  build Σ_yi = Φ_i diag(λ) Φ_i^T + σ²I               │
│  solve v = Σ_yi^{-1}(Y_i − μ_i)  ← cholesky_solve   │
│  ξ_ik = λ_k · dot(Φ_i_col_k, v)                     │
│  compute Ω_i = diag(λ) - diag(λ)Φ_i^T Σ_yi^{-1}    │
│               Φ_i diag(λ)  (ncomp×ncomp)             │
│       │                                               │
│       ▼                                               │
│  assemble fitted: x̂_i(t) = μ(t) + Σ_k ξ_ik φ_k(t) │
│  assemble bands: ±z · sqrt(φ(t)^T Ω_i φ(t))         │
│       │                                               │
│       ▼                                               │
│  PaceFpcaResult { mean, eigenvalues, eigenfunctions,  │
│    scores, fitted, fitted_lower, fitted_upper,        │
│    argvals, sigma2, ncomp }                           │
└──────────────────────────────────────────────────────┘
```

### Recommended Project Structure

```
fdars-core/src/
└── pace_fpca.rs         # new single flat module — config + result + entry fn + tests
fdars-core/src/lib.rs    # add: pub mod pace_fpca; pub use pace_fpca::{...};
```

No submodule directory — the module is ~300–400 lines, consistent with `concurrent_regression.rs` (~230 lines) and other single-file modules in the project.

### Pattern 1: Config Struct (Builder-style with Default)

Mirrors `ElasticPcrConfig` from `elastic_regression/mod.rs` [VERIFIED: fdars-core/src/elastic_regression/mod.rs:60-85]:

```rust
// Source: elastic_regression/mod.rs:60-85 (pattern)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PaceFpcaConfig {
    /// Number of FPCA components to extract.
    pub ncomp: usize,
    /// Kernel bandwidth for mean and covariance smoothing.
    pub bandwidth: f64,
    /// Measurement-error variance σ² (caller-supplied).
    pub sigma2: f64,
    /// Work grid: evaluation points for mean, eigenfunctions, and trajectories.
    pub work_grid: Vec<f64>,
    /// Confidence level for bands (0 < alpha < 1); default 0.05 → 95% bands.
    pub alpha: f64,
}

impl Default for PaceFpcaConfig {
    fn default() -> Self {
        let m = 51;
        Self {
            ncomp: 3,
            bandwidth: 0.1,
            sigma2: 0.01,
            work_grid: (0..m).map(|i| i as f64 / (m - 1) as f64).collect(),
            alpha: 0.05,
        }
    }
}
```

Note: `#[non_exhaustive]` is used on result structs in this project but NOT on config structs (verified: `ElasticConfig`, `ElasticPcrConfig` lack `#[non_exhaustive]`). Apply the same pattern: NO `#[non_exhaustive]` on `PaceFpcaConfig`. [VERIFIED: fdars-core/src/elastic_regression/mod.rs:61-85 — `ElasticPcrConfig` has no `#[non_exhaustive]`]

### Pattern 2: Result Struct

Mirrors `FpcaResult`, `ConcurrentRegrResult` [VERIFIED: fdars-core/src/regression.rs:23-38, fdars-core/src/concurrent_regression.rs:33-47]:

```rust
// Source: regression.rs:22-38 (pattern)
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct PaceFpcaResult {
    /// Kernel-smoothed mean function on the work grid (length m).
    pub mean: Vec<f64>,
    /// Functional eigenvalues (variance explained per component), length ncomp.
    pub eigenvalues: Vec<f64>,
    /// Eigenfunctions on the work grid, shape m × ncomp (column-major FdMatrix).
    pub eigenfunctions: FdMatrix,
    /// BLUP (conditional-expectation) FPC scores, shape n × ncomp.
    pub scores: FdMatrix,
    /// Fitted trajectories on the work grid, shape n × m.
    pub fitted: FdMatrix,
    /// Lower confidence band on the work grid, shape n × m.
    pub fitted_lower: FdMatrix,
    /// Upper confidence band on the work grid, shape n × m.
    pub fitted_upper: FdMatrix,
    /// Work grid used for all outputs (clone of config.work_grid).
    pub argvals: Vec<f64>,
    /// Measurement-error variance used (echoed from config).
    pub sigma2: f64,
    /// Number of components extracted (may be < config.ncomp if fewer positive eigenvalues).
    pub ncomp: usize,
}
```

### Pattern 3: Entry Function Header

```rust
// Source: concurrent_regression.rs:53+ (pattern)
/// Fit PACE sparse FPCA for irregularly sampled functional data.
///
/// Implements the Yao–Müller–Wang (2005) PACE estimator:
/// 1. Kernel-smoothed mean μ̂(t) on the work grid.
/// 2. Kernel-smoothed covariance surface Ĝ(s,t) via `cov_irreg`.
/// 3. Symmetric eigendecomposition of Ĝ → eigenvalues λ_k, eigenfunctions φ_k.
/// 4. Per-curve BLUP (conditional-expectation) scores ξ_ik.
/// 5. Fitted trajectories x̂_i(t) = μ̂(t) + Σ_k ξ_ik φ_k(t).
/// 6. Pointwise confidence bands from BLUP prediction variance.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if:
/// - `data` has zero observations or any curve has zero points,
/// - `config.work_grid` has fewer than 2 points,
/// - `config.ncomp` is zero.
/// Returns [`FdarError::InvalidParameter`] if:
/// - `config.bandwidth` or `config.sigma2` is not positive,
/// - `config.alpha` is not in (0, 1).
/// Returns [`FdarError::ComputationFailed`] if:
/// - fewer positive eigenvalues than requested `ncomp`,
/// - any curve's Σ_yi is not positive-definite (pathological sparse case).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn pace_fpca(
    data: &IrregFdata,
    config: &PaceFpcaConfig,
) -> Result<PaceFpcaResult, FdarError> { ... }
```

### Pattern 4: Crate-Root Re-Export

```rust
// In src/lib.rs — append to existing re-exports:
pub mod pace_fpca;
pub use pace_fpca::{pace_fpca, PaceFpcaConfig, PaceFpcaResult};
```

[VERIFIED: fdars-core/src/lib.rs:81-131 — all new modules follow `pub mod X; pub use X::{...};` pattern]

### Anti-Patterns to Avoid

- **Using `fdata_to_pc_1d` for eigendecomposition:** That function SVDs the *centered data matrix*; PACE requires eigendecomposing the *smoothed covariance surface* (different object, different scaling, different interpretation of eigenvalues).
- **Not scaling eigenvalues by integration weights:** Numerical eigenvalues of the raw C matrix are not functional eigenvalues. Must use the `W^{1/2} C W^{1/2}` scaling to get variance-interpretable λ_k.
- **Keeping negative eigenvalues:** The smoothed covariance surface may produce slightly negative eigenvalues from finite-sample kernel estimation. These must be truncated to zero; do not include them in the BLUP solve.
- **Re-solving Σ_yi for every component k:** Compute `v = Σ_yi^{-1}(Y_i − μ_i)` once per curve, then dot with each Φ_i column. Avoids n_i³ Cholesky per component.
- **Large n_i Σ_yi:** For curves with many observed points (n_i > ~200), the n_i×n_i system becomes expensive. Document in rustdoc that σ² must be positive (ridge regularizer) to ensure Σ_yi is positive-definite; a σ² near zero with many densely-sampled points can cause near-singularity.
- **Interpolating eigenfunctions using `to_regular_grid`:** That function produces NaN for obs times outside the work grid range. Use `linear_interp` directly (clamping boundary extension), which handles all t in the domain safely.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Kernel-smoothed mean | Custom Nadaraya-Watson loop | `mean_irreg` | Already implements Gaussian + Epanechnikov kernels with parallel dispatch |
| Smoothed covariance surface | Custom 2D kernel smoother | `cov_irreg` | Already correct; returns FdMatrix in the right shape |
| Symmetric eigendecomposition | Power iteration or Jacobi | `nalgebra::DMatrix::symmetric_eigen()` | Householder + QR; LAPACK-quality; already in the dep tree |
| Linear system solve for BLUP | Gaussian elimination | `linalg::cholesky_solve` (or `cholesky_d`) | Already pub(crate), tested, handles PSD matrices correctly |
| Simpson integration weights | Custom quadrature | `helpers::simpsons_weights` | Already handles uniform and non-uniform grids; used by fdata_to_pc_1d |
| Linear interpolation of eigenfunctions | Manual binary search | `irreg_fdata::linear_interp` or `helpers::linear_interp` | Both exist; `helpers::linear_interp` uses binary search and is public |

**Key insight:** Every sub-problem in PACE has an existing solution in the crate. The BLUP score computation and variance bands are the only genuinely new code (~70 lines total).

---

## Common Pitfalls

### Pitfall 1: Ascending Eigenvalue Order (nalgebra SymmetricEigen)

**What goes wrong:** `DMatrix::symmetric_eigen()` returns eigenvalues in ASCENDING order (smallest first). If not reversed, the "top-1" component captures the smallest variance direction.

**Why it happens:** nalgebra's convention differs from MATLAB/R's `eigen()` which returns descending order.

**How to avoid:** After `let eigen = c_scaled.symmetric_eigen();`, sort pairs by descending eigenvalue: take `(eigen.eigenvalues.len() - 1)..=0` or collect into a sorted Vec and take the top `ncomp`.

**Warning signs:** If the first eigenfunction is nearly constant and eigenvalue[0] < eigenvalue[1], the order is wrong.

### Pitfall 2: Eigenfunction Sign Ambiguity

**What goes wrong:** Eigenvectors are defined up to sign; different runs or different nalgebra versions may flip signs, making the result non-reproducible.

**Why it happens:** Algebraic ambiguity — if φ is an eigenvector, so is -φ.

**How to avoid:** Apply the same convention as `fix_svd_signs` [VERIFIED: fdars-core/src/regression.rs:180-201]: for each component k, find the grid index j_max with the largest |φ_k(j)|; if φ_k(j_max) < 0, flip the entire eigenfunction and all corresponding scores.

### Pitfall 3: Near-Singular Σ_yi for Curves with Many Points

**What goes wrong:** `cholesky_solve` returns `ComputationFailed` when Σ_yi is near-singular (diagonal < 1e-12 in the factorization).

**Why it happens:** If σ² is very small and n_i is large with dense observations, the rank-ncomp term Φ_i diag(λ) Φ_i^T may be nearly singular before adding the σ²I ridge. Also happens if eigenfunctions are nearly collinear at a particular curve's observed times.

**How to avoid:** Document that `sigma2 > 0` is required. Provide a fall-back: if Cholesky fails, try a small ridge addition (add 1e-8 to diagonal) and retry once. If it still fails, return `ComputationFailed` with a descriptive message including the curve index.

### Pitfall 4: `irreg_fdata::linear_interp` is `pub(super)`

**What goes wrong:** Cannot call `irreg_fdata::linear_interp` from `pace_fpca.rs` — it is `pub(super)` scoped to the `irreg_fdata` module.

**Why it happens:** The function was scoped for use only within the `irreg_fdata` module. [VERIFIED: fdars-core/src/irreg_fdata/mod.rs:193 — `pub(super) fn linear_interp`]

**How to avoid:** Two options: (a) promote to `pub(crate)` in `irreg_fdata/mod.rs` (minimal change), or (b) use `helpers::linear_interp` [VERIFIED: fdars-core/src/helpers.rs:172-191] which is already public and functionally identical. Option (b) requires no change to existing code. **Recommend option (b).**

### Pitfall 5: Covariance Surface Diagonal Includes σ² (Off-Diagonal vs Diagonal Distinction)

**What goes wrong:** The `cov_irreg` function includes same-point pairs (j1 == j2), so the diagonal of the smoothed surface Ĝ(t,t) is inflated by σ². Subtracting σ² from the diagonal before eigendecomposition to get the "signal" covariance can produce negative diagonal entries and non-PSD matrices.

**Why it happens:** The PACE estimator in Yao 2005 estimates the *total covariance* G(s,t) = Cov(X(s), X(t)) and uses the diagonal surface Ĝ(t,t) as an estimate of the *total* variance. σ² enters only in the Σ_yi construction for the BLUP, not in the eigendecomposition. The eigendecomposition is of the full Ĝ (including diagonal noise contribution).

**How to avoid:** Do NOT subtract σ² from the covariance surface before eigendecomposition. Eigendecompose the raw Ĝ. σ² enters only as the ridge term in Σ_yi per curve.

### Pitfall 6: Work Grid Points Outside Curve Domains

**What goes wrong:** `helpers::linear_interp` clamps to boundary values when the work grid extends beyond an individual curve's observed range. For the fitted trajectory, this gives a flat extrapolation (constant boundary value) rather than NaN.

**Why it happens:** `linear_interp` boundary-clamps by design [VERIFIED: fdars-core/src/helpers.rs:172-176].

**How to avoid:** Document in rustdoc that `fitted` values at work grid points outside a curve's observed range are extrapolated via boundary clamping, not predicted by PACE. Confidence bands will be widest at these points. This is the standard PACE behavior.

---

## Code Examples

### Eigendecomposition of Smoothed Covariance

```rust
// Source: nalgebra 0.33 SymmetricEigen API [ASSUMED — nalgebra docs, verified present in dep]
use nalgebra::DMatrix;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;

fn eigendecompose_cov(
    cov: &FdMatrix,           // m×m smoothed covariance from cov_irreg
    work_grid: &[f64],
    ncomp: usize,
) -> (Vec<f64>, FdMatrix) {  // (eigenvalues, eigenfunctions m×ncomp)
    let m = work_grid.len();
    let w = simpsons_weights(work_grid);
    let sqrt_w: Vec<f64> = w.iter().map(|&wi| wi.sqrt()).collect();

    // Build W^{1/2} C W^{1/2}
    let mut c_scaled = vec![0.0_f64; m * m];
    for j in 0..m {
        for l in 0..m {
            // cov is column-major: cov[(j, l)] = element at row j, col l
            c_scaled[j + l * m] = sqrt_w[j] * cov[(j, l)] * sqrt_w[l];
        }
    }
    let c_dmat = DMatrix::from_column_slice(m, m, &c_scaled);
    let eigen = c_dmat.symmetric_eigen(); // ascending order

    // Collect and sort descending
    let n_eval = eigen.eigenvalues.len();
    let mut pairs: Vec<(f64, usize)> = (0..n_eval)
        .map(|k| (eigen.eigenvalues[k], k))
        .collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // Take top ncomp with positive eigenvalues
    let pairs: Vec<_> = pairs.into_iter()
        .filter(|&(lam, _)| lam > 0.0)
        .take(ncomp)
        .collect();

    let actual_ncomp = pairs.len();
    let mut eigenvalues = Vec::with_capacity(actual_ncomp);
    let mut eigenfunctions = FdMatrix::zeros(m, actual_ncomp);

    for (k, &(lam, col_idx)) in pairs.iter().enumerate() {
        eigenvalues.push(lam);
        // Unscale: φ_k = W^{-1/2} · v_k
        for j in 0..m {
            let raw = eigen.eigenvectors[(j, col_idx)];
            eigenfunctions[(j, k)] = if sqrt_w[j] > 1e-15 {
                raw / sqrt_w[j]
            } else {
                raw
            };
        }
    }
    // Sign fix (same as fix_svd_signs in regression.rs:180-201)
    for k in 0..actual_ncomp {
        let j_max = (0..m)
            .max_by(|&a, &b| eigenfunctions[(a, k)].abs()
                .partial_cmp(&eigenfunctions[(b, k)].abs())
                .unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0);
        if eigenfunctions[(j_max, k)] < 0.0 {
            for j in 0..m { eigenfunctions[(j, k)] = -eigenfunctions[(j, k)]; }
        }
    }
    (eigenvalues, eigenfunctions)
}
```

### BLUP Score Computation (per curve)

```rust
// Source: Yao, Müller & Wang (2005) JASA — Eq. 3 [ASSUMED for formula]
// crate helpers: linalg::cholesky_d, helpers::linear_interp [VERIFIED]
use crate::linalg::cholesky_d;
use crate::linalg::{forward_solve};

fn blup_scores_one_curve(
    obs_t: &[f64],          // curve i observed times (length n_i)
    obs_y: &[f64],          // curve i observed values
    work_grid: &[f64],      // length m
    mean: &[f64],           // μ̂ on work_grid (length m)
    eigenfunctions: &FdMatrix, // m × ncomp
    eigenvalues: &[f64],    // length ncomp
    sigma2: f64,
) -> Result<Vec<f64>, crate::FdarError> {
    use crate::helpers::linear_interp;

    let ni = obs_t.len();
    let ncomp = eigenvalues.len();

    // Interpolate mean to obs times
    let mu_i: Vec<f64> = obs_t.iter()
        .map(|&t| linear_interp(work_grid, mean, t))
        .collect();
    let resid: Vec<f64> = obs_y.iter().zip(mu_i.iter())
        .map(|(&y, &m)| y - m)
        .collect();

    // Build Phi_i (ni × ncomp) — each column is φ_k interpolated to obs_t
    let mut phi_i = vec![0.0_f64; ni * ncomp]; // row-major for Cholesky compat
    for k in 0..ncomp {
        let phi_k_on_grid: Vec<f64> = (0..work_grid.len())
            .map(|j| eigenfunctions[(j, k)])
            .collect();
        for j in 0..ni {
            phi_i[j * ncomp + k] =
                linear_interp(work_grid, &phi_k_on_grid, obs_t[j]);
        }
    }

    // Build Sigma_yi = Phi_i diag(lambda) Phi_i^T + sigma2 * I (row-major ni×ni)
    let mut sigma_yi = vec![0.0_f64; ni * ni];
    for j in 0..ni {
        for l in 0..ni {
            let mut s = 0.0_f64;
            for k in 0..ncomp {
                s += phi_i[j * ncomp + k] * eigenvalues[k] * phi_i[l * ncomp + k];
            }
            sigma_yi[j * ni + l] = s;
        }
        sigma_yi[j * ni + j] += sigma2;
    }

    // Solve v = Sigma_yi^{-1} resid via Cholesky (row-major)
    let l = cholesky_d(&sigma_yi, ni)?;
    let z = forward_solve(&l, &resid, ni);
    // Back-solve L^T x = z
    let mut v = z;
    for j in (0..ni).rev() {
        for k in (j + 1)..ni {
            v[j] -= l[k * ni + j] * v[k];
        }
        v[j] /= l[j * ni + j];
    }

    // xi_ik = lambda_k * dot(phi_i_col_k, v)
    let mut scores = vec![0.0_f64; ncomp];
    for k in 0..ncomp {
        let mut dot = 0.0;
        for j in 0..ni { dot += phi_i[j * ncomp + k] * v[j]; }
        scores[k] = eigenvalues[k] * dot;
    }
    Ok(scores)
}
```

### Prediction Variance (per curve, per grid point)

```rust
// Ω_i = diag(λ) - diag(λ) Φ_i^T Σ_yi^{-1} Φ_i diag(λ)   [ASSUMED formula — Yao 2005]
// Var(x̂_i(t)) = φ(t)^T Ω_i φ(t)

fn prediction_variance_one_curve(
    // Σ_yi^{-1} · Φ_i (already computed as part of BLUP): pass pre-computed
    // Or recompute: build A_i = diag(λ) Φ_i^T Σ_yi^{-1} Φ_i diag(λ)
    phi_i: &[f64],          // ni × ncomp row-major
    sigma_yi_inv_col: &[Vec<f64>], // ncomp columns of Σ_yi^{-1} Φ_i diag(λ) — precomputed
    eigenvalues: &[f64],
    phi_at_t: &[f64],      // eigenfunctions at grid point t (length ncomp)
    ni: usize,
    ncomp: usize,
) -> f64 {
    // A_i[k,l] = lambda_k * sum_j phi_i[j,k] * (Sigma_yi^{-1} Phi_i diag(lam))[j,l]
    let mut var = 0.0_f64;
    for k in 0..ncomp {
        for l in 0..ncomp {
            let mut a_kl = 0.0;
            for j in 0..ni {
                a_kl += phi_i[j * ncomp + k] * sigma_yi_inv_col[l][j];
            }
            a_kl *= eigenvalues[k];
            let omega_kl = if k == l { eigenvalues[k] - a_kl } else { -a_kl };
            var += omega_kl * phi_at_t[k] * phi_at_t[l];
        }
    }
    var.max(0.0) // numerical guard
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| Dense-grid FPCA via SVD of data matrix | PACE sparse FPCA via eigendecomposition of smoothed covariance | Yao et al. 2005 | Handles sparse/irregular sampling; produces BLUP scores instead of projections |
| `fdata_to_pc_1d` (SVD-based, dense grid required) | `pace_fpca` (covariance eigendecomposition, irregular obs) | This phase | Supports curves with 1–10 observed points, not just dense grids |

**Not deprecated — complementary:**
- `fdata_to_pc_1d` remains the correct choice for dense regularly-sampled data.
- `pace_fpca` targets sparse/irregular data where PACE is the theoretically appropriate estimator.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `nalgebra 0.33` exposes `DMatrix::symmetric_eigen()` returning ascending eigenvalues and eigenvectors as columns of a DMatrix | §PACE Formulas Step 3, §Code Examples | Would require alternative eigendecomposition path; check nalgebra 0.33 changelog |
| A2 | `cov_irreg` includes same-point pairs (j1==j2) in the kernel weighting, inflating the diagonal by σ² | §Pitfall 5, §PACE Formulas Step 2 | If it excludes them, the covariance estimate is the signal covariance directly and the analysis is slightly different |
| A3 | `statrs` provides `Normal::inverse_cdf` usable for qnorm(1-alpha/2) | §PACE Formulas Step 6 | Would need a rational approximation or hard-coded z=1.96 for alpha=0.05 |
| A4 | Functional eigenvalues from the W^{1/2}CW^{1/2} eigendecomposition are directly interpretable as variance-explained without further scaling | §PACE Formulas Step 3 | Might need additional normalization by (1/n) factor depending on cov_irreg normalization convention |
| A5 | The `linalg::cholesky_d` function accepts row-major flat arrays for Σ_yi | §Code Examples | Verified it uses `mat[j*d+i]` indexing — CONFIRMED row-major [VERIFIED: fdars-core/src/linalg.rs:16-40] |
| A6 | `helpers::linear_interp` (public) is functionally identical to `irreg_fdata::linear_interp` (pub(super)) and suitable for eigenfunction interpolation | §Don't Hand-Roll, §Pitfall 4 | Minor: one uses binary search, the other iterates; both clamp at boundaries |

**A5 correction:** A5 is actually VERIFIED, not assumed. `cholesky_d` at linalg.rs:16-40 uses `l[j*d+k]` and `mat[j*d+j]` — row-major indexing confirmed.

---

## Open Questions (RESOLVED)

1. **nalgebra 0.33 SymmetricEigen API exact signature**
   - What we know: nalgebra provides symmetric eigendecomposition; the type exists as `nalgebra::linalg::SymmetricEigen`.
   - What's unclear: Whether `DMatrix::symmetric_eigen()` is a method directly on DMatrix or requires import of `SymmetricEigen::new(mat)`.
   - Recommendation: Use `nalgebra::linalg::SymmetricEigen::new(c_dmat, true)` or equivalently `c_dmat.symmetric_eigen()` if the method exists. The executor should verify by checking nalgebra 0.33 source or running `cargo doc --open`.

2. **cov_irreg normalization: does it divide by sum of weights?**
   - What we know: `accumulate_cov_at_point` divides by `sum_weights` (verified at smoothing.rs:170-173), so it is a weighted average, not a sum.
   - What's unclear: Whether the resulting eigenvalues from the weighted decomposition need division by n (number of observations) to match the PACE convention of estimating the population covariance.
   - Recommendation: Synthetic test will reveal this — if recovered eigenvalues are off by a factor of n, divide the covariance matrix by n before eigendecomposition. Add a note to the BLUP formula derivation.

3. **Parallelism: per-curve BLUP vs global eigendecomposition**
   - What we know: `iter_maybe_parallel!` is the project macro for feature-gated rayon.
   - What's unclear: Whether the per-curve BLUP loop should be parallelized (n curves, each O(n_i³)).
   - Recommendation: Parallelize the per-curve loop using `iter_maybe_parallel!(0..n)` following the pattern in `metric_lp_irreg` (smoothing.rs:253-260). Each curve's computation is independent.

---

## Environment Availability

Step 2.6: SKIPPED — this phase is code-only, no external tools required beyond the existing Rust toolchain.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | Compilation | ✓ | 1.97.0 (CLAUDE.md) | — |
| nalgebra 0.33 | SymmetricEigen | ✓ | Already in Cargo.toml [VERIFIED] | — |
| statrs | qnorm for bands | ✓ (CLAUDE.md lists it) | Per Cargo.toml | Hard-code z=1.96 for alpha=0.05 |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (inline `#[cfg(test)] mod tests`) |
| Config file | none (uses `cargo test`) |
| Quick run command | `cargo test -p fdars-core pace_fpca` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FPCA-01 | pace_fpca returns correct eigenvalues from known generative model | unit (synthetic recovery) | `cargo test -p fdars-core pace_fpca::tests::test_pace_synthetic_recovery` | ❌ Wave 0 (new file) |
| FPCA-01 | BLUP scores match known scores from synthetic model | unit | `cargo test -p fdars-core pace_fpca::tests::test_blup_scores_known` | ❌ Wave 0 |
| FPCA-01 | fitted trajectories lie within confidence bands | unit | `cargo test -p fdars-core pace_fpca::tests::test_fitted_within_bands` | ❌ Wave 0 |
| FPCA-01 | Identical output for same inputs (deterministic) | unit | `cargo test -p fdars-core pace_fpca::tests::test_determinism` | ❌ Wave 0 |
| FPCA-01 | Empty IrregFdata → InvalidDimension | unit (error path) | `cargo test -p fdars-core pace_fpca::tests::test_empty_data` | ❌ Wave 0 |
| FPCA-01 | ncomp=0 → InvalidParameter | unit (error path) | `cargo test -p fdars-core pace_fpca::tests::test_zero_ncomp` | ❌ Wave 0 |
| FPCA-01 | bandwidth <= 0 → InvalidParameter | unit (error path) | `cargo test -p fdars-core pace_fpca::tests::test_invalid_bandwidth` | ❌ Wave 0 |
| FPCA-01 | sigma2 <= 0 → InvalidParameter | unit (error path) | `cargo test -p fdars-core pace_fpca::tests::test_invalid_sigma2` | ❌ Wave 0 |
| FPCA-01 | crate-root re-export reachable | unit (smoke) | `cargo test -p fdars-core pace_fpca::tests::test_crate_root_reexport` | ❌ Wave 0 |

### Synthetic Recovery Test Design

**Generative model (known ground truth):**
```
n = 20 curves, each observed at 3-8 uniformly random points on [0,1]
Mean: μ(t) = 0  (zero mean for simplicity)
Eigenfunctions: φ_1(t) = √2 sin(πt), φ_2(t) = √2 cos(πt)   (Fourier basis, L² orthonormal)
Eigenvalues: λ_1 = 1.0, λ_2 = 0.5
Scores: ξ_i1 ~ N(0, λ_1), ξ_i2 ~ N(0, λ_2), seeded with fixed seed
True curve: X_i(t) = ξ_i1 φ_1(t) + ξ_i2 φ_2(t)
Observations: Y_{ij} = X_i(t_{ij}) + ε_{ij}, ε_{ij} ~ N(0, σ²), σ² = 0.01
```

**Assertions (after sign alignment):**
- `|recovered_λ_1 - 1.0| < 0.2` (20% tolerance; kernel smoothing introduces bias)
- `|recovered_λ_2 - 0.5| < 0.15`
- Correlation between `recovered_φ_1` and `true_φ_1` on work grid > 0.95
- Correlation between `recovered_scores[:,0]` and `true_ξ[:,0]` > 0.8
- All `fitted[i,j] >= fitted_lower[i,j]` AND `fitted[i,j] <= fitted_upper[i,j]`

**Sign alignment:** Before comparing eigenfunctions, flip sign so that `dot(recovered, true) > 0`.

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core pace_fpca`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite + `cargo clippy --all-targets --features linalg,parallel -- -D warnings`

### Wave 0 Gaps

- [ ] `fdars-core/src/pace_fpca.rs` — covers all FPCA-01 requirements (new file, entire module)
- [ ] Crate-root re-export in `src/lib.rs` — `pub mod pace_fpca; pub use pace_fpca::{...};`

*(No existing test infrastructure gaps — inline tests are the project convention, no separate test file needed)*

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`. This phase is a pure numerical computation library — no authentication, sessions, HTTP, user input parsing, or persistence. Applicable ASVS controls are limited.

### Applicable ASVS Categories

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes | Dimension/parameter checks at function entry (FdarError::InvalidDimension, InvalidParameter) |
| V6 Cryptography | no | — |

### Known Threat Patterns

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Panic on invalid input (OOB index, unwrap) | Tampering | All public fns return `Result<T, FdarError>`; validate n, m, ncomp, bandwidth, sigma2 before computation |
| NaN propagation from degenerate covariance | Tampering/Denial | Guard against NaN eigenvalues (keep only positive λ_k > 0); max(variance, 0) before sqrt in bands |
| Integer overflow in n_i * ncomp product | Tampering | Rust's usize arithmetic panics on overflow in debug; in release, document that n_i * ncomp must fit in usize (reasonable for functional data) |

---

## Sources

### Primary (HIGH confidence — VERIFIED this session by reading files)

- `fdars-core/src/irreg_fdata/mod.rs:38-190` — `IrregFdata` struct definition, `get_obs`, `n_points`, `linear_interp` (pub(super))
- `fdars-core/src/irreg_fdata/kernels.rs:58-90` — `mean_irreg` signature and implementation
- `fdars-core/src/irreg_fdata/smoothing.rs:111-138` — `cov_irreg` signature and implementation; `linear_interp` internal usage
- `fdars-core/src/regression.rs:23-200` — `FpcaResult`, `fix_svd_signs`, `fdata_to_pc_1d` (SVD with sqrt_weights pattern)
- `fdars-core/src/helpers.rs:57-86` — `simpsons_weights` (uniform + non-uniform grid)
- `fdars-core/src/helpers.rs:172-191` — `helpers::linear_interp` (binary search, public)
- `fdars-core/src/matrix.rs:1-325` — `FdMatrix` column-major API, `to_dmatrix`, `from_dmatrix`
- `fdars-core/src/linalg.rs:1-152` — `cholesky_d`, `cholesky_factor`, `forward_solve`, `cholesky_solve`
- `fdars-core/src/error.rs` — `FdarError` enum variants (verbatim)
- `fdars-core/src/lib.rs:60-131` — crate root structure, re-export pattern, existing `pub mod` list
- `fdars-core/src/elastic_regression/mod.rs:36-85` — `ElasticConfig`, `ElasticPcrConfig` (config struct pattern without `#[non_exhaustive]`)
- `fdars-core/src/concurrent_regression.rs:1-80` — `ConcurrentRegrResult`, `concurrent_regression` (flat module pattern)
- `fdars-core/src/simulation.rs:86-279` — `fourier_eigenfunctions`, `wiener_eigenfunctions`, `eigenvalues_linear` (test data generation)
- `fdars-core/src/parallel.rs` — `iter_maybe_parallel!` macro (feature-gated rayon)

### Secondary (MEDIUM confidence — project conventions from CLAUDE.md)

- `CLAUDE.md` — naming conventions, `#[non_exhaustive]` on result structs, conditional serde, `#[must_use]` on expensive computations, inline tests
- CONTEXT.md — all locked implementation decisions (treated as authoritative)

### Tertiary (LOW confidence — training knowledge)

- Yao, Müller & Wang (2005) "Functional Data Analysis for Sparse Longitudinal Data", JASA — PACE formula derivation (A2, A4 assumptions)
- nalgebra 0.33 documentation — `SymmetricEigen` API details (A1 assumption)

---

## Metadata

**Confidence breakdown:**
- Reuse targets (IrregFdata, cov_irreg, mean_irreg, linalg): HIGH — read source files this session
- PACE formulas (BLUP score, Ω variance): MEDIUM — well-established literature, not independently verified from paper
- nalgebra SymmetricEigen API: MEDIUM — consistent with nalgebra docs conventions but not confirmed against 0.33 source this session
- Architecture/module structure: HIGH — verified against existing modules
- Test design: HIGH — follows project inline test convention, generative model is deterministic

**Research date:** 2026-08-18
**Valid until:** 2026-09-18 (stable codebase; nalgebra 0.33 unlikely to change)
