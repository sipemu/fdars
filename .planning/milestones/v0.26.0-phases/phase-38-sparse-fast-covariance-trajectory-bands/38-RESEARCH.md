# Phase 38: Sparse Fast Covariance & Trajectory Bands — Research

**Researched:** 2026-08-21
**Domain:** Rust FDA — sparse/irregular FPCA, sandwich covariance estimation, multivariate block covariance
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **FACE covariance (`face_covariance`)**: sandwich-smooth the `cov_irreg` surface (symmetric smoother S on both sides, S·Cov·Sᵀ), reusing the existing kernel-smoothing machinery — the SAME approach as Phase 37's `ssvd`. This is a kernel-sandwich approximation of `refund::face`'s penalized tensor-product spline FACE; document the divergence in rustdoc (capability match, not exact).
- Validity: symmetrize by construction and clip negative eigenvalues to 0 so the returned surface is a valid PSD covariance (documented).
- Signature: `face_covariance(ifd, grid, bandwidth) -> Result<FdMatrix, FdarError>` — mirrors `cov_irreg` but `Result`-returning with input validation (empty sample, non-monotone/mismatched argvals, invalid bandwidth → `FdarError`).
- Correctness: dense-limit test — on densely-sampled synthetic curves it recovers a known covariance surface within a documented tolerance.

- **Multivariate `mface_covariance`**: Input `&[IrregFdata]` — P variables observed on the same n subjects, one argvals grid per variable. Block layout: a (P·G)×(P·G) block covariance where diagonal blocks are the per-variable FACE covariance and off-diagonal blocks are the cross-variable covariance. Cross-block estimation: kernel-smoothed cross-covariance between paired sparse variables (extend the `cov_irreg` accumulation to cross-variable point pairs). Return: a struct carrying the block covariance matrix + per-variable grids + a block accessor (documented block layout). `#[non_exhaustive]`.

- **Fitted trajectories + pointwise bands**: Engine: reuse `pace_fpca`'s BLUP scores → fitted trajectory → pointwise Gaussian band machinery (it already produces `fitted`, `fitted_lower`, `fitted_upper`). Band type: reuse `pace_fpca`'s alpha-controlled pointwise Gaussian bands. Return: reuse `PaceFpcaResult` (or a thin wrapper exposing fitted + lower/upper + grid). `face_covariance` remains available separately for the FACE surface.
- Correctness: dense-curve test — the fitted trajectory tracks the true curve within its pointwise bands within a documented tolerance.

- **Packaging**: new `fdars-core/src/irreg_fdata/face.rs`, re-exported via `irreg_fdata/mod.rs` and the crate root; no new crate dependency.
- **Result types**: standard derives (`Debug, Clone, PartialEq`) + conditional serde + `#[non_exhaustive]`.
- **Divergence docs**: rustdoc documents FACE here as a kernel-sandwich approximation of `refund::face` / `mfaces` (matched by capability, not R's exact penalized-spline internals).

### Claude's Discretion

- Exact sandwich smoother kernel/bandwidth defaults, PSD-clipping eigenvalue threshold, tolerance constants for the dense-limit tests, and the precise wrapper-vs-reuse shape for the trajectory entry point are at Claude's discretion, guided by the `pace_fpca`/`cov_irreg` conventions and the R baselines.

### Deferred Ideas (OUT OF SCOPE)

- Plotting/rendering of covariance surfaces or trajectory bands (out of scope — numeric outputs only).
- Automatic σ² estimation for `pace_fpca` (already deferred upstream; caller supplies σ²).
- FPCA-02 (Phase 37) is a separate, already-completed disjoint area (`fpca_variants.rs`).
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| SPARSE-01-01 | User can estimate a sparse-data covariance surface via the FACE fast-sandwich smoother (`face_covariance`) over irregular/sparse functional data. | Section "FACE Covariance: Sandwich Construction" + "Existing Code Inventory" |
| SPARSE-01-02 | User can estimate a multivariate sparse covariance via the `mfaces` extension (`mface_covariance`) for multiple simultaneously-observed sparse functional variables. | Section "Multivariate FACE: Block Covariance Construction" |
| SPARSE-01-03 | User can obtain fitted continuous trajectories with pointwise confidence bands for sparse curves, integrated with the FACE covariance path (and reusing `pace_fpca` machinery where applicable). | Section "Fitted Trajectories + Pointwise Bands" |
</phase_requirements>

---

## Summary

Phase 38 adds three entry points to `fdars-core/src/irreg_fdata/face.rs`: `face_covariance` (FACE kernel-sandwich covariance surface for sparse/irregular data), `mface_covariance` (multivariate block covariance across P simultaneously-observed sparse variables), and `face_trajectory` (per-curve fitted trajectories + pointwise confidence bands via the existing `pace_fpca` BLUP engine). All build on code already shipped: `cov_irreg` (kernel-smoothed sparse covariance), `gaussian_smooth_cov` / the Phase 37 `ssvd` sandwich pattern, `pace_fpca` (BLUP scores + bands), and the `IrregFdata` CSR container.

The FACE sandwich approximation used here is a **kernel-smoother sandwich**: a row-then-column Gaussian smoothing matrix S applied as S·Cov·Sᵀ, followed by symmetrization and PSD clipping via nalgebra `symmetric_eigen`. This is the same pattern as Phase 37's `ssvd` / `gaussian_smooth_cov` private helper, now applied to a sparse-data covariance instead of a dense empirical covariance. The result is documented as a kernel-sandwich approximation of `refund::face`'s penalized tensor-product spline FACE — matched by capability, not exact R internals.

For fitted trajectories, `face_trajectory` is a thin entry point that calls `pace_fpca` with the caller-supplied FACE covariance (or computes it internally) and returns the `PaceFpcaResult` directly — no new struct needed because `PaceFpcaResult` already carries `fitted`, `fitted_lower`, `fitted_upper`, `argvals`, `scores`, and `eigenvalues` as documented below.

**Primary recommendation:** Build `face.rs` as a flat module with three public functions. Reuse `gaussian_smooth_cov` by making it `pub(crate)` in `fpca_variants.rs` (or inline the pattern in `face.rs`). Wire `face_trajectory` as a thin wrapper over `pace_fpca`. All three satisfy the additive/non-breaking, no-new-crate, `Result`-returning constraints.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Sparse covariance estimation | `irreg_fdata/face.rs` (new) | `irreg_fdata/smoothing.rs` (reuse `cov_irreg`) | Covariance lives in the sparse-fdata layer; smoothing reused from existing kernel machinery |
| Sandwich smoothing | `fpca_variants.rs::gaussian_smooth_cov` (reuse) | inline in `face.rs` if can't pub(crate) | Row-then-column Gaussian smoother already implemented for Phase 37 `ssvd` |
| PSD enforcement | nalgebra `symmetric_eigen` (same as `pace_fpca` / `ssvd`) | — | Eigendecompose → clip negative eigenvalues → reconstruct |
| Block covariance (mfaces) | `irreg_fdata/face.rs` (new) | `irreg_fdata/smoothing.rs` (`accumulate_cov_at_point` pattern) | Extends within-variable FACE to cross-variable kernel accumulation |
| Fitted trajectories + bands | `pace_fpca::pace_fpca` (reuse directly) | `irreg_fdata/face.rs` (thin entry point) | `PaceFpcaResult` already carries all needed fields |

---

## Standard Stack

### Core (all already in Cargo.toml — no new dependency)

| Library | Role in Phase 38 |
|---------|-----------------|
| nalgebra 0.33 | `DMatrix::from_column_slice`, `symmetric_eigen` — PSD clipping |
| `crate::irreg_fdata::cov_irreg` | Base kernel-smoothed raw covariance; FACE sandwiches on top |
| `crate::irreg_fdata::mean_irreg` | Mean estimation inside `face_trajectory` (via `pace_fpca`) |
| `crate::pace_fpca::{pace_fpca, PaceFpcaConfig, PaceFpcaResult}` | BLUP FPCA + fitted trajectories + bands engine |
| `crate::helpers::simpsons_weights` | Simpson integration weights for PSD sandwich eigendecompose |
| `crate::fpca_variants::gaussian_smooth_cov` | Separable Gaussian smoothing — make `pub(crate)` and reuse |

**Installation:** No new packages. All dependencies already present.

---

## Package Legitimacy Audit

No new external packages are added in this phase. The "no new crate dependency" constraint is a locked milestone requirement.

**Packages removed due to [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

---

## Existing Code Inventory

This section is the central reference for all code the planner must cite in tasks.

### `IrregFdata` — `fdars-core/src/irreg_fdata/mod.rs:38-213`

[VERIFIED: fdars-core/src/irreg_fdata/mod.rs:38-48]

```rust
pub struct IrregFdata {
    pub offsets: Vec<usize>,   // length n+1; offsets[i]..offsets[i+1] is curve i
    pub argvals: Vec<f64>,     // all observation points concatenated
    pub values: Vec<f64>,      // all values concatenated
    pub rangeval: [f64; 2],
}
```

Key methods [VERIFIED: fdars-core/src/irreg_fdata/mod.rs:146-189]:
- `ifd.n_obs() -> usize` — number of curves (offsets.len() - 1)
- `ifd.n_points(i) -> usize` — points in curve i
- `ifd.get_obs(i) -> (&[f64], &[f64])` — (argvals_slice, values_slice) for curve i
- `ifd.total_points() -> usize` — total observation points across all curves
- `ifd.obs_counts() -> Vec<usize>` — per-curve point counts
- `ifd.min_obs() -> usize`, `ifd.max_obs() -> usize`

Constructor: `IrregFdata::from_lists(argvals_list, values_list)` [VERIFIED: fdars-core/src/irreg_fdata/mod.rs:59-99]

### `cov_irreg` — `fdars-core/src/irreg_fdata/smoothing.rs:111-138`

[VERIFIED: fdars-core/src/irreg_fdata/smoothing.rs:111-138]

```rust
pub fn cov_irreg(ifd: &IrregFdata, s_grid: &[f64], t_grid: &[f64], bandwidth: f64) -> FdMatrix
```

Returns a `ns × nt` column-major `FdMatrix`. Each entry (si, ti) is computed by `accumulate_cov_at_point` using double-loop over all within-curve point pairs, Nadaraya-Watson weighted by `kernel_gaussian((obs_t[j1]-s)/bw) * kernel_gaussian((obs_t[j2]-t)/bw)`. Does **not** return `Result` — does not validate inputs. FACE calls this and then validates.

Note: `cov_irreg` does NOT subtract σ² from the diagonal — the raw estimate absorbs measurement error (documented in `pace_fpca` module doc). Same for `face_covariance`.

### `accumulate_cov_at_point` — `fdars-core/src/irreg_fdata/smoothing.rs:141-176`

[VERIFIED: fdars-core/src/irreg_fdata/smoothing.rs:141-176]

Private helper. Signature:
```rust
fn accumulate_cov_at_point(
    offsets: &[usize], obs_times: &[f64], centered: &[f64],
    n: usize, s: f64, t: f64, bandwidth: f64,
) -> f64
```

For `mface_covariance` cross-blocks: the cross-variable analogue needs access to `offsets`, `obs_times`, `centered` from **two different** `IrregFdata` objects (variable p and variable q) and loops over pairs (j1 from curve_p, j2 from curve_q) of the **same subject** i. This is a new accumulation loop (not a direct call to `accumulate_cov_at_point`) because that function's double loop is within-curve. The cross-variable loop is an outer loop over subjects, then over obs points in var_p and obs points in var_q, accumulating `w_p * w_q * centered_p[j1] * centered_q[j2]`.

### `mean_irreg` and `KernelType` — `fdars-core/src/irreg_fdata/kernels.rs:58-90`

[VERIFIED: fdars-core/src/irreg_fdata/kernels.rs:58-90]

```rust
pub fn mean_irreg(
    ifd: &IrregFdata, target_argvals: &[f64], bandwidth: f64, kernel_type: KernelType,
) -> Vec<f64>
```

Nadaraya-Watson estimator. `KernelType::Gaussian` is the standard choice (matches `cov_irreg` and `pace_fpca`).

`kernel_gaussian` [VERIFIED: fdars-core/src/irreg_fdata/kernels.rs:31-33]:
```rust
pub(crate) fn kernel_gaussian(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
}
```

### `gaussian_smooth_cov` — `fdars-core/src/fpca_variants.rs:589-629`

[VERIFIED: fdars-core/src/fpca_variants.rs:589-629]

```rust
fn gaussian_smooth_cov(cov: &FdMatrix, argvals: &[f64], bandwidth: f64) -> FdMatrix
```

Currently `fn` (private to `fpca_variants`). Does a **separable row-then-column** Gaussian smoothing: build an m×m normalized kernel weight matrix K, then compute `tmp = K · cov` (row pass), then `out = tmp · Kᵀ` (column pass). This is the S·Cov·Sᵀ sandwich with S = K (row-stochastic normalized kernel matrix).

For `face_covariance` this function must be made `pub(crate)` in `fpca_variants.rs` OR the pattern must be inlined in `face.rs`. Making it `pub(crate)` is preferred (DRY, testable, no signature change to public API). This is a **single-line change** to `fpca_variants.rs` (change `fn` to `pub(crate) fn`).

### Phase 37 `ssvd` sandwich+PSD pattern — `fdars-core/src/fpca_variants.rs:728-762`

[VERIFIED: fdars-core/src/fpca_variants.rs:728-762]

The PSD clipping pattern used in `ssvd` (identical to `pace_fpca::eigendecompose_cov`):

```rust
let w = simpsons_weights(argvals);
let sqrt_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
let mut c_scaled = vec![0.0_f64; m * m];
for col in 0..m {
    for row in 0..m {
        c_scaled[row + col * m] = sqrt_w[row] * smooth_cov[(row, col)] * sqrt_w[col];
    }
}
let eigen = DMatrix::from_column_slice(m, m, &c_scaled).symmetric_eigen();
let mut pairs: Vec<(f64, usize)> = (0..eigen.eigenvalues.len())
    .map(|k| (eigen.eigenvalues[k], k)).collect();
pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
let pairs: Vec<(f64, usize)> = pairs.into_iter().filter(|&(lam, _)| lam > 0.0).take(ncomp).collect();
```

The eigenvalues of the weighted W^{1/2}·Cov·W^{1/2} are all non-negative if Cov is PSD; clipping `lam > 0.0` enforces PSD.

To **reconstruct a PSD covariance surface** from the clipped eigenpairs (for `face_covariance`):
- After clipping, for each retained (lam_k, vec_k): unscale `phi_k[j] = vec_k[j] / sqrt_w[j]`
- Reconstruct: `Cov_psd[i,j] = sum_k lam_k * phi_k[i] * phi_k[j]`
- (Or equivalently via the smoother output symmetrization: `Cov_sym = 0.5*(C + Cᵀ)` then eigendecompose)

For `face_covariance` the returned `FdMatrix` is the **reconstructed PSD surface**, not the eigenvectors.

### `pace_fpca` — `fdars-core/src/pace_fpca.rs:272-622`

[VERIFIED: fdars-core/src/pace_fpca.rs:272-622]

```rust
pub fn pace_fpca(data: &IrregFdata, config: &PaceFpcaConfig) -> Result<PaceFpcaResult, FdarError>
```

`PaceFpcaConfig` fields [VERIFIED: fdars-core/src/pace_fpca.rs:53-70]:
- `ncomp: usize` — number of FPCA components
- `bandwidth: f64` — kernel bandwidth for mean and covariance smoothing (must be strictly positive)
- `sigma2: f64` — caller-supplied measurement-error variance (must be strictly positive)
- `work_grid: Vec<f64>` — evaluation points; must have at least 2 points, sorted
- `alpha: f64` — confidence level for bands; default 0.05 → 95% pointwise bands

`PaceFpcaResult` fields [VERIFIED: fdars-core/src/pace_fpca.rs:99-120]:
- `mean: Vec<f64>` — smoothed mean on work_grid (length m)
- `eigenvalues: Vec<f64>` — functional eigenvalues (length ncomp)
- `eigenfunctions: FdMatrix` — m × ncomp, column-major
- `scores: FdMatrix` — n × ncomp BLUP scores
- `fitted: FdMatrix` — n × m fitted trajectories
- `fitted_lower: FdMatrix` — n × m lower pointwise confidence band
- `fitted_upper: FdMatrix` — n × m upper pointwise confidence band
- `argvals: Vec<f64>` — clone of work_grid
- `sigma2: f64` — echoed from config
- `ncomp: usize` — actual components extracted

`pace_fpca` internally calls `cov_irreg(data, work_grid, work_grid, bandwidth)` then `eigendecompose_cov`. The `face_trajectory` entry point can either: (A) call `pace_fpca` directly with a `PaceFpcaConfig` — caller supplies bandwidth/sigma2/work_grid/ncomp — or (B) be a thin wrapper that also accepts a pre-computed `face_covariance` surface. Option A is simpler; option B exposes the FACE surface explicitly. The locked decision says "reuse `pace_fpca`" — option A is the natural reading.

### Re-export pattern — `fdars-core/src/lib.rs:266`

[VERIFIED: fdars-core/src/lib.rs:266]

```rust
// Re-export PACE sparse FPCA types
pub use pace_fpca::{pace_fpca, PaceFpcaConfig, PaceFpcaResult};
```

The new phase-38 symbols (`face_covariance`, `mface_covariance`, `MfaceCovResult`, `face_trajectory`) follow the same pattern:

```rust
// Re-export FACE sparse covariance types (Phase 38)
pub use irreg_fdata::{face_covariance, mface_covariance, MfaceCovResult, face_trajectory};
```

And `irreg_fdata/mod.rs` gets:
```rust
pub mod face;
pub use face::{face_covariance, mface_covariance, MfaceCovResult, face_trajectory};
```

---

## Architecture Patterns

### System Architecture Diagram

```
IrregFdata (CSR container)
    │
    ├─► cov_irreg(ifd, grid, grid, bw)         ← existing, returns FdMatrix (raw kernel-smoothed Cov)
    │           │
    │           ▼
    │   gaussian_smooth_cov(cov, grid, bw)      ← pub(crate) from fpca_variants (S·Cov·Sᵀ)
    │           │
    │           ▼
    │   symmetrize + eigendecompose (symmetric_eigen)
    │           │                                      clipping lam < 0 → 0
    │           ▼
    │   reconstruct PSD FdMatrix                ← face_covariance return value
    │
    ├─► mface_covariance(&[IrregFdata], grids, bw)
    │     ├─ diagonal blocks: face_covariance per variable
    │     └─ off-diagonal blocks: cross_cov_irreg_pair(ifd_p, grid_p, ifd_q, grid_q, bw)
    │              (new private helper — symmetric accumulation over same subjects)
    │         → assembled into (P·G)×(P·G) block FdMatrix
    │
    └─► face_trajectory(ifd, config)            ← thin wrapper
              └─► pace_fpca(ifd, config)        ← existing, returns PaceFpcaResult
                       (fitted, fitted_lower, fitted_upper, argvals, scores, ...)
```

### Recommended Project Structure

```
fdars-core/src/irreg_fdata/
├── mod.rs          (add: pub mod face; pub use face::{...})
├── kernels.rs      (unchanged)
├── smoothing.rs    (unchanged)
├── tests.rs        (unchanged)
└── face.rs         (NEW — face_covariance, mface_covariance, MfaceCovResult, face_trajectory + inline tests)

fdars-core/src/fpca_variants.rs  (single change: gaussian_smooth_cov → pub(crate) fn)
fdars-core/src/lib.rs            (add: pub use irreg_fdata::{face_covariance, mface_covariance, MfaceCovResult, face_trajectory})
```

---

## FACE Covariance: Sandwich Construction

### Mathematical Definition

The FACE (Fast Covariance Estimation) estimator in `refund::face` uses penalized tensor-product splines. The fdars implementation approximates it with a **kernel-sandwich smoother**:

```
Ĉ_FACE(s,t) = [S · Ĉ_raw · Sᵀ]_{sym,PSD}
```

where:
- `Ĉ_raw` = output of `cov_irreg(ifd, grid, grid, bw)` — the kernel-smoothed empirical sparse covariance (m×m)
- S = row-stochastic normalized Gaussian kernel weight matrix (m×m): `S[a,b] = K_h(|grid[a]-grid[b]|) / sum_b K_h(...)`
- `S·Ĉ_raw·Sᵀ` = `gaussian_smooth_cov(&Ĉ_raw, grid, bw)` (the existing Phase 37 helper)
- `[·]_{sym,PSD}` = symmetrize by construction (sandwich is symmetric if S is symmetric, which it is since K is symmetric), then clip negative eigenvalues: eigendecompose → set lam < 0 to 0 → reconstruct

**Symmetry note:** `gaussian_smooth_cov` already produces a symmetric result because K is symmetric (K[a,b] = K[b,a]) and `S·Cov·Sᵀ = S·Cov·S` (since K is symmetric → S = Sᵀ). So explicit symmetrization `0.5*(C + Cᵀ)` is a defensive check, not strictly required.

**PSD enforcement:** `symmetric_eigen` from nalgebra returns eigenvalues in ascending order. After sorting descending, clip at 0.0. Reconstruct:

```rust
let mut cov_psd = vec![0.0_f64; m * m];
for (lam, col_idx) in &pos_pairs {  // pos_pairs: lam > 0
    let v: Vec<f64> = (0..m).map(|j| eigen.eigenvectors[(j, *col_idx)]).collect();
    // unscale: phi[j] = v[j] / sqrt_w[j]
    let phi: Vec<f64> = v.iter().enumerate().map(|(j, &vj)| {
        if sqrt_w[j] > 1e-15 { vj / sqrt_w[j] } else { vj }
    }).collect();
    // outer product: cov_psd[i,j] += lam * phi[i] * phi[j]
    for i in 0..m {
        for j in 0..m {
            cov_psd[i + j * m] += lam * phi[i] * phi[j];
        }
    }
}
```

**Dense-limit property:** When bandwidth → 0 (no smoothing), `gaussian_smooth_cov` approaches identity (each row of S concentrated on one grid point). In that limit `Ĉ_FACE ≈ Ĉ_raw`. The dense-limit test constructs curves sampled at 51 uniform points on [0,1] (dense, same as work_grid) with known covariance C(s,t) = exp(-|s-t|) (Ornstein-Uhlenbeck kernel), calls `face_covariance` with a small bandwidth, and checks `max|Ĉ_FACE - C_true| < 0.2` (calibrated from the `pace_fpca` eigenvalue bias experience of ~35% for n=20 sparse).

### `face_covariance` Signature

```rust
/// FACE fast-sandwich covariance surface for sparse/irregular functional data.
///
/// Approximates the FACE estimator (Xiao et al. 2016, `refund::face`) via a kernel-sandwich
/// smoother: S·Cov·Sᵀ where S is a separable row-normalised Gaussian kernel matrix applied
/// on both sides of the kernel-smoothed raw covariance. The result is symmetrised by
/// construction and projected to PSD (negative eigenvalues clipped to zero).
///
/// **Divergence from `refund::face`:** The R implementation uses penalized tensor-product
/// splines (P-FACE); this implementation uses a kernel-smoother sandwich (K-FACE),
/// matching the capability (smooth sparse-data covariance surface, PSD result) but not
/// the exact penalised-spline computations. For smooth, moderately sparse data the two
/// agree closely on the leading modes.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if data has zero observations or grid has fewer
/// than 2 points. Returns [`FdarError::InvalidParameter`] if bandwidth is not strictly
/// positive or finite. Returns [`FdarError::ComputationFailed`] if mean smoothing returns
/// non-finite values (bandwidth too narrow for the data range).
#[must_use = "expensive computation whose result should not be discarded"]
pub fn face_covariance(
    ifd: &IrregFdata,
    grid: &[f64],
    bandwidth: f64,
) -> Result<FdMatrix, FdarError>
```

---

## Multivariate FACE: Block Covariance Construction

### Block Layout

For P variables, each with `G_p` grid points, the block matrix is `(G_1 + ... + G_P) × (G_1 + ... + G_P)`.

If all grids have the same size G, the block matrix is `(P·G) × (P·G)`.

Block `(p, q)` occupies rows `[offset_p .. offset_p + G_p]`, cols `[offset_q .. offset_q + G_q]` where `offset_p = sum_{k<p} G_k`.

Diagonal block (p, p): `face_covariance(ifd[p], grid[p], bandwidth)` — G_p × G_p.

Off-diagonal block (p, q), p ≠ q: cross-variable kernel-smoothed covariance. This is NOT `face_covariance` — it is the raw cross-covariance surface smoothed bilaterally. The cross-covariance between variable p (observed at times t^p_ij) and variable q (observed at times t^q_ij) for the SAME n subjects:

```
Ĉ_{pq}(s, t) = [sum_i sum_{j1} sum_{j2} K_h(t^p_{i,j1} - s) K_h(t^q_{i,j2} - t) (x^p_{i,j1} - mu_p(t^p_{i,j1})) (x^q_{i,j2} - mu_q(t^q_{i,j2}))]
              / [sum_i sum_{j1} sum_{j2} K_h(t^p_{i,j1} - s) K_h(t^q_{i,j2} - t)]
```

This is an accumulation over **same subjects** (i), cross-product of centered values from variable p at t^p_{i,j1} and variable q at t^q_{i,j2}.

Note: `Ĉ_{pq}(s,t) ≠ Ĉ_{qp}(s,t)ᵀ` in general (unless both variables have the same grid); the block matrix is forced symmetric by setting `block(q,p) = block(p,q)ᵀ` after computing upper triangular.

The off-diagonal blocks need the means of each variable: compute `mean_p = mean_irreg(ifd[p], ifd[p].argvals, bw, KernelType::Gaussian)` for each variable, then center values.

### `MfaceCovResult` Struct

```rust
/// Result of multivariate FACE block covariance estimation.
///
/// The block covariance matrix has shape (G_total × G_total) where
/// G_total = sum of grid lengths across all P variables. Block (p,q) occupies
/// rows [offset_p..offset_p+G_p], cols [offset_q..offset_q+G_q].
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct MfaceCovResult {
    /// Full (G_total × G_total) block covariance matrix (column-major FdMatrix).
    pub block_cov: FdMatrix,
    /// Grid for each variable (length P).
    pub grids: Vec<Vec<f64>>,
    /// Row/col offset for each variable in the block matrix (length P).
    pub offsets: Vec<usize>,
}

impl MfaceCovResult {
    /// Extract the covariance block for variables p and q as a contiguous FdMatrix.
    /// Returns the G_p × G_q submatrix (column-major).
    pub fn block(&self, p: usize, q: usize) -> FdMatrix { ... }
}
```

### `mface_covariance` Signature

```rust
/// Multivariate FACE block covariance for P simultaneously-observed sparse variables.
///
/// Constructs a (G_total × G_total) block covariance matrix where:
/// - diagonal blocks (p,p): FACE covariance surface for variable p
/// - off-diagonal blocks (p,q): kernel-smoothed cross-covariance between variables p and q
///   (same subjects observed at different sparse times)
///
/// All P variables must have the same number of subjects n.
///
/// **Divergence from `mfaces`:** R's `mfaces` uses penalized tensor-product splines;
/// this uses kernel-sandwich smoothing (same capability match as `face_covariance`).
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if variables has length < 2, if n_obs differ
/// across variables, or if any grid has fewer than 2 points.
/// Returns [`FdarError::InvalidParameter`] if bandwidth is not strictly positive.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn mface_covariance(
    variables: &[IrregFdata],
    grids: &[Vec<f64>],
    bandwidth: f64,
) -> Result<MfaceCovResult, FdarError>
```

### Block Assembly Algorithm

```
1. Validate: variables.len() >= 2, variables.len() == grids.len()
             each grids[p].len() >= 2, all ifd.n_obs() == n
             bandwidth > 0 and finite

2. Compute means for all P variables:
   mean_p[..] = mean_irreg(&variables[p], &variables[p].argvals, bandwidth, KernelType::Gaussian)

3. Compute centered values for all P variables.

4. Diagonal blocks: for p in 0..P: block_pp = face_covariance(&variables[p], &grids[p], bandwidth)?

5. Off-diagonal blocks (upper triangle only): for p in 0..P, q in p+1..P:
   block_pq = cross_cov_irreg_pair(&variables[p], &grids[p], &variables[q], &grids[q], bandwidth)
   block_qp = block_pq.transpose()   (or fill by symmetry)

6. Assemble into G_total × G_total FdMatrix (column-major).
   offsets[p] = sum_{k<p} grids[k].len()

7. Return MfaceCovResult { block_cov, grids, offsets }
```

Private helper `cross_cov_irreg_pair`:
```rust
fn cross_cov_irreg_pair(
    ifd_p: &IrregFdata, centered_p: &[f64],
    grid_s: &[f64],
    ifd_q: &IrregFdata, centered_q: &[f64],
    grid_t: &[f64],
    bandwidth: f64,
) -> FdMatrix
```
Accumulates `K_h(t^p_{i,j1}-s) * K_h(t^q_{i,j2}-t) * c_p[i,j1] * c_q[i,j2]` over all i (subjects), j1 (points in variable p curve i), j2 (points in variable q curve i). Returns `ns × nt` FdMatrix.

---

## Fitted Trajectories + Pointwise Bands

### How `pace_fpca` Already Satisfies SPARSE-01-03

`PaceFpcaResult` [VERIFIED: fdars-core/src/pace_fpca.rs:99-120] already carries:
- `fitted: FdMatrix` — n × m BLUP-reconstructed fitted trajectories: `x̂_i(t) = µ̂(t) + Σ_k ξ_ik φ_k(t)`
- `fitted_lower: FdMatrix` — n × m lower pointwise confidence band: `fitted[i,j] - z * sqrt(max(Var_j, 0))`
- `fitted_upper: FdMatrix` — n × m upper pointwise confidence band
- `argvals: Vec<f64>` — the work grid
- `scores: FdMatrix` — BLUP FPC scores ξ_ik (n × ncomp)
- `eigenvalues: Vec<f64>` — functional eigenvalues λ_k

The Var_j formula (BLUP prediction variance) is [VERIFIED: fdars-core/src/pace_fpca.rs:569-585]:
```
Var(x̂_i(t_j)) = Σ_{k,l} Ω_i[k,l] φ_k(t_j) φ_l(t_j)
Ω_i[k,l] = (k==l ? λ_k : 0) - A_i[k,l]
A_i[k,l] = λ_k · Φ_i[:,k]^T · Σ_yi^{-1} · Φ_i[:,l] · λ_l
```
This is the exact Yao-Müller-Wang (2005) pointwise band formula. `alpha` controls `z = qnorm(1 - alpha/2)`.

### `face_trajectory` Entry Point

```rust
/// Fit sparse functional trajectories with pointwise confidence bands using FACE+PACE.
///
/// Estimates per-curve continuous fitted trajectories and pointwise Gaussian
/// confidence bands by running [`pace_fpca`] on the sparse functional data.
/// The fitted trajectory is the BLUP reconstruction from the FPCA decomposition;
/// bands are computed from the BLUP prediction variance (Yao et al. 2005, eq. 3.2).
///
/// This is a convenience entry point: it calls [`pace_fpca`] directly and returns its
/// [`PaceFpcaResult`], which carries `fitted`, `fitted_lower`, `fitted_upper`,
/// `argvals`, `scores`, and `eigenvalues`. For the FACE covariance surface separately,
/// call [`face_covariance`].
///
/// # Errors
/// All errors from [`pace_fpca`] are propagated: [`FdarError::InvalidDimension`] if
/// data is empty or any curve has < 2 points; [`FdarError::InvalidParameter`] if
/// config parameters are out of range; [`FdarError::ComputationFailed`] for numerical
/// failures.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn face_trajectory(
    data: &IrregFdata,
    config: &PaceFpcaConfig,
) -> Result<PaceFpcaResult, FdarError> {
    pace_fpca(data, config)
}
```

This is literally a one-line delegation. Its value is in the name (`face_trajectory` vs `pace_fpca`) and the rustdoc connecting it to the FACE covariance path, satisfying SPARSE-01-03 without any new machinery.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead |
|---------|-------------|-------------|
| Covariance PSD clipping | Custom eigenvalue truncation | `nalgebra::DMatrix::symmetric_eigen()` → filter `lam > 0` (see `pace_fpca::eigendecompose_cov`) |
| Separable Gaussian smoothing | Row/col kernel loops | `pub(crate) gaussian_smooth_cov` from `fpca_variants.rs` (make it `pub(crate)`) |
| BLUP trajectories + bands | Re-implement PACE | `pace_fpca(data, config)` already returns `fitted`, `fitted_lower`, `fitted_upper` |
| Normal quantile for bands | `qnorm` implementation | `standard_normal_quantile` in `pace_fpca.rs` (already used there) |
| Simpson weights | Integration weight hand-code | `crate::helpers::simpsons_weights(grid)` |
| Nadaraya-Watson mean | Custom mean smoother | `mean_irreg(ifd, target, bw, KernelType::Gaussian)` |

---

## Common Pitfalls

### Pitfall 1: Forgetting That `cov_irreg` Is NOT `Result`-returning

**What goes wrong:** Calling `cov_irreg` with an empty `ifd` or a zero-length grid silently returns a 0×0 `FdMatrix`. The `FdMatrix::from_column_major` call inside `cov_irreg` returns `Err` but uses `.expect()` — so an invalid dimension panics at runtime instead of returning `FdarError`.

**How to avoid:** Validate all inputs in `face_covariance` BEFORE calling `cov_irreg`. Mirror the validation pattern from `pace_fpca`: check `n_obs > 0`, `grid.len() >= 2`, `bandwidth > 0 and finite`, grid is sorted. Return `FdarError` on any failure.

**Warning signs:** Any test that panics rather than returning `Err` from `face_covariance` with invalid inputs.

### Pitfall 2: Off-Diagonal Block Asymmetry in `mface_covariance`

**What goes wrong:** The cross-variable covariance `Ĉ_pq(s,t)` computed for block (p,q) is G_p × G_t. Block (q,p) should be G_q × G_p = `Ĉ_pq(s,t)^T`. If you compute both independently, you get a slightly asymmetric block matrix due to floating-point rounding.

**How to avoid:** Compute the upper-triangle blocks only (p < q). For block (q,p), set it equal to the transpose of block (p,q). Enforce symmetry of the full block matrix by the construction: `block_qp = block_pq^T`.

### Pitfall 3: Column-Major Indexing for the Block Matrix Assembly

**What goes wrong:** When assembling the (P·G)×(P·G) block matrix into a flat `Vec<f64>`, using row-major indexing instead of column-major gives a silently wrong result (elements at wrong positions, wrong shape in FdMatrix operations).

**How to avoid:** For a G_total × G_total matrix, element `(row, col)` is at flat index `row + col * G_total`. When copying block (p, q) into the big matrix: `big[row_in_p + offset_p + (col_in_q + offset_q) * G_total]`. Follow the `cov_irreg` pattern: `cov[si + ti * ns] = ...` [VERIFIED: fdars-core/src/irreg_fdata/smoothing.rs:132].

### Pitfall 4: PSD Reconstruction vs Returning Raw Smoothed Cov

**What goes wrong:** Returning the raw output of `gaussian_smooth_cov` without PSD enforcement. On finite samples with sparse data, the kernel-smoothed covariance can have small negative eigenvalues from estimation noise. Downstream consumers (FPCA eigendecomposition, Cholesky solve in BLUP) fail or give nonsense.

**How to avoid:** Always reconstruct from the clipped eigenpairs. The pattern: `symmetric_eigen` → sort descending → filter `lam > 0` → reconstruct outer sum `sum_k lam_k * phi_k * phi_k^T`. The threshold `lam > 0.0` is exact (not `> -eps`): eigenvalues of a theoretically PSD matrix that go slightly negative are artifacts and should be zeroed. Document the threshold in rustdoc.

### Pitfall 5: `gaussian_smooth_cov` Bandwidth Convention vs `cov_irreg` Bandwidth

**What goes wrong:** Using the same bandwidth for both `cov_irreg` and `gaussian_smooth_cov` (the sandwich smoother) may over-smooth. The `cov_irreg` bandwidth controls how much information from nearby observation times contributes; the sandwich bandwidth controls how much the resulting surface is smoothed across the grid. In the R `refund::face`, the penalty parameter plays the role of the sandwich bandwidth.

**How to avoid:** Use the same `bandwidth` parameter for both (simplest defensible default), document in rustdoc that users can decouple them by calling `cov_irreg` + the sandwich separately if needed. This matches the locked decision ("reusing the existing kernel-smoothing machinery"). A single `bandwidth` parameter is correct for Phase 38.

### Pitfall 6: `mface_covariance` — Mismatched `n_obs` Across Variables

**What goes wrong:** If `variables[p].n_obs() != variables[q].n_obs()`, the cross-variable accumulation loop silently processes `min(n_p, n_q)` subjects and gives a wrong result.

**How to avoid:** Validate `all variables have same n_obs` at entry. Return `FdarError::InvalidDimension` with clear message citing which variable has the mismatched count.

### Pitfall 7: Clippy `--all-targets` Gate

**What goes wrong:** Clippy errors in test code (e.g., unused variable in `#[cfg(test)]` block, or an allow attribute missing) pass a plain `cargo clippy -p fdars-core -- -D warnings` but fail the CI gate `cargo clippy --all-targets --features linalg,parallel -- -D warnings`.

**How to avoid:** After implementing, always run `cargo clippy --all-targets --features linalg,parallel -- -D warnings`. This is documented in MEMORY.md and CONTEXT.md specifics.

---

## Code Examples

### Pattern 1: `face_covariance` Full Implementation Skeleton

```rust
// Source: modeled on fpca_variants.rs:ssvd (lines 667-799) + pace_fpca.rs:eigendecompose_cov
pub fn face_covariance(
    ifd: &IrregFdata,
    grid: &[f64],
    bandwidth: f64,
) -> Result<FdMatrix, FdarError> {
    // --- Validation ---
    let n = ifd.n_obs();
    if n == 0 {
        return Err(FdarError::InvalidDimension { parameter: "ifd", expected: "at least 1 observation".to_string(), actual: "0".to_string() });
    }
    let m = grid.len();
    if m < 2 {
        return Err(FdarError::InvalidDimension { parameter: "grid", expected: "at least 2 points".to_string(), actual: format!("{m}") });
    }
    if !bandwidth.is_finite() || bandwidth <= 0.0 {
        return Err(FdarError::InvalidParameter { parameter: "bandwidth", message: format!("must be finite and strictly positive, got {bandwidth}") });
    }
    // Validate grid is sorted
    for w in grid.windows(2) {
        if w[0] >= w[1] {
            return Err(FdarError::InvalidParameter { parameter: "grid", message: "must be strictly increasing".to_string() });
        }
    }

    // --- Step 1: Raw kernel-smoothed covariance ---
    let raw_cov = cov_irreg(ifd, grid, grid, bandwidth);

    // --- Step 2: Sandwich smooth (S·Cov·Sᵀ) ---
    let smooth_cov = gaussian_smooth_cov(&raw_cov, grid, bandwidth);

    // --- Step 3: PSD projection via symmetric_eigen ---
    let w = simpsons_weights(grid);
    let sqrt_w: Vec<f64> = w.iter().map(|v| v.sqrt()).collect();
    let mut c_scaled = vec![0.0_f64; m * m];
    for col in 0..m {
        for row in 0..m {
            c_scaled[row + col * m] = sqrt_w[row] * smooth_cov[(row, col)] * sqrt_w[col];
        }
    }
    let eigen = DMatrix::from_column_slice(m, m, &c_scaled).symmetric_eigen();
    let mut pairs: Vec<(f64, usize)> = (0..eigen.eigenvalues.len())
        .map(|k| (eigen.eigenvalues[k], k)).collect();
    pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    // --- Step 4: Reconstruct PSD covariance surface ---
    let mut cov_data = vec![0.0_f64; m * m];
    for &(lam, col_idx) in pairs.iter().filter(|&&(lam, _)| lam > 0.0) {
        let phi: Vec<f64> = (0..m).map(|j| {
            let raw = eigen.eigenvectors[(j, col_idx)];
            if sqrt_w[j] > 1e-15 { raw / sqrt_w[j] } else { raw }
        }).collect();
        for i in 0..m {
            for j in 0..m {
                cov_data[i + j * m] += lam * phi[i] * phi[j];
            }
        }
    }
    FdMatrix::from_column_major(cov_data, m, m)
        .map_err(|e| FdarError::ComputationFailed { operation: "face_covariance", detail: e.to_string() })
}
```

### Pattern 2: Cross-Variable Block Accumulation

```rust
// Source: modeled on accumulate_cov_at_point (smoothing.rs:141-176)
fn cross_cov_at_point(
    ifd_p: &IrregFdata, centered_p: &[f64],
    ifd_q: &IrregFdata, centered_q: &[f64],
    n: usize, s: f64, t: f64, bandwidth: f64,
) -> f64 {
    let mut sum_weights = 0.0;
    let mut sum_products = 0.0;
    for i in 0..n {
        let (tp, _) = ifd_p.get_obs(i);
        let start_p = ifd_p.offsets[i];
        let (tq, _) = ifd_q.get_obs(i);
        let start_q = ifd_q.offsets[i];
        for j1 in 0..tp.len() {
            let w1 = kernel_gaussian((tp[j1] - s) / bandwidth);
            for j2 in 0..tq.len() {
                let w2 = kernel_gaussian((tq[j2] - t) / bandwidth);
                let w = w1 * w2;
                sum_weights += w;
                sum_products += w * centered_p[start_p + j1] * centered_q[start_q + j2];
            }
        }
    }
    if sum_weights > 0.0 { sum_products / sum_weights } else { 0.0 }
}
```

### Pattern 3: Block Matrix Assembly (Column-Major)

```rust
// Column-major assembly of diagonal + off-diagonal blocks
let g_total: usize = grids.iter().map(|g| g.len()).sum();
let mut block_data = vec![0.0_f64; g_total * g_total];

for p in 0..num_vars {
    // Diagonal block (p, p)
    let bp = face_covariance(&variables[p], &grids[p], bandwidth)?;
    let gp = grids[p].len();
    let op = offsets[p];
    for col in 0..gp {
        for row in 0..gp {
            block_data[(op + row) + (op + col) * g_total] = bp[(row, col)];
        }
    }
    // Off-diagonal blocks (upper triangle)
    for q in p+1..num_vars {
        let gq = grids[q].len();
        let oq = offsets[q];
        let bpq = cross_cov_surface(&variables[p], &centered[p], &grids[p],
                                     &variables[q], &centered[q], &grids[q], bandwidth);
        for col in 0..gq {
            for row in 0..gp {
                // block(p,q) at (row, col)
                block_data[(op + row) + (oq + col) * g_total] = bpq[(row, col)];
                // block(q,p) = block(p,q)^T
                block_data[(oq + col) + (op + row) * g_total] = bpq[(row, col)];
            }
        }
    }
}
```

### Pattern 4: Dense-Limit Synthetic Test for `face_covariance`

```rust
// Dense-limit test: curves observed at 51 uniform points on [0,1], known covariance C(s,t)=exp(-|s-t|)
fn known_cov(s: f64, t: f64) -> f64 { (-(s - t).abs()).exp() }

let n = 30usize;
let m = 51usize;
let grid: Vec<f64> = (0..m).map(|i| i as f64 / (m-1) as f64).collect();
// Generate curves from the OU process (known covariance)
// ...build IrregFdata with all curves observed at all 51 grid points (dense)...
let fc = face_covariance(&ifd_dense, &grid, 0.05).unwrap();
let mut max_err = 0.0_f64;
for (si, &s) in grid.iter().enumerate() {
    for (ti, &t) in grid.iter().enumerate() {
        max_err = max_err.max((fc[(si, ti)] - known_cov(s, t)).abs());
    }
}
assert!(max_err < 0.3, "dense-limit error {max_err} > 0.3");
```

### Pattern 5: Two-Variable Synthetic Test for `mface_covariance`

```rust
// Known structure: X_i(t) = a_i sin(πt), Y_i(t) = a_i cos(πt)
// => C_XY(s,t) = E[a_i^2] sin(πs) cos(πt) = λ sin(πs) cos(πt)
// Diagonal blocks: C_XX(s,t) = λ sin(πs) sin(πt), C_YY(s,t) = λ cos(πs) cos(πt)
// Build sparse IrregFdata for both variables, call mface_covariance,
// verify off-diagonal block approximates C_XY at grid points within tolerance.
let bxy = result.block(0, 1);  // G×G off-diagonal block
let mut err = 0.0_f64;
for (si, &s) in grid.iter().enumerate() {
    for (ti, &t) in grid.iter().enumerate() {
        let expected = lambda * (PI * s).sin() * (PI * t).cos();
        err = err.max((bxy[(si, ti)] - expected).abs());
    }
}
assert!(err < 0.4, "cross-block error {err} > 0.4");
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in (`#[test]`, `#[cfg(test)]`) |
| Config file | none (inline `#[cfg(test)] mod tests { ... }`) |
| Quick run command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel irreg_fdata::face` |
| Full suite command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| SPARSE-01-01 | `face_covariance` shape/symmetry/PSD | unit | `cargo test ... face::tests::test_face_covariance_shape` |
| SPARSE-01-01 | `face_covariance` dense-limit recovery | unit | `cargo test ... face::tests::test_face_covariance_dense_limit` |
| SPARSE-01-01 | `face_covariance` error paths | unit | `cargo test ... face::tests::test_face_covariance_errors` |
| SPARSE-01-02 | `mface_covariance` block shape/symmetry | unit | `cargo test ... face::tests::test_mface_shape` |
| SPARSE-01-02 | `mface_covariance` known cross-structure | unit | `cargo test ... face::tests::test_mface_known_structure` |
| SPARSE-01-02 | `mface_covariance` error paths | unit | `cargo test ... face::tests::test_mface_errors` |
| SPARSE-01-03 | `face_trajectory` fitted-within-bands | unit | `cargo test ... face::tests::test_face_trajectory_bands` |
| SPARSE-01-03 | `face_trajectory` delegates to `pace_fpca` | unit | `cargo test ... face::tests::test_face_trajectory_delegation` |
| All | crate-root re-export smoke | unit | `cargo test ... face::tests::test_reexports` |

### Sampling Rate

- **Per task commit:** `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel irreg_fdata::face -- --nocapture`
- **Per wave merge:** full suite above
- **Phase gate:** Full suite green before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `fdars-core/src/irreg_fdata/face.rs` — entire new file
- [ ] Single-line change: `gaussian_smooth_cov` in `fpca_variants.rs` → `pub(crate) fn`
- [ ] `irreg_fdata/mod.rs` additions: `pub mod face; pub use face::{...}`
- [ ] `lib.rs` additions: `pub use irreg_fdata::{face_covariance, mface_covariance, MfaceCovResult, face_trajectory}`

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| R `refund::face`: penalized tensor-product spline FACE | fdars: kernel-sandwich FACE (S·Cov·Sᵀ) — same capability class | Simpler to implement without spline basis; no new crate; documented divergence |
| R `mfaces`: multivariate penalized FACE | fdars: kernel-sandwich block covariance | Matches mfaces API shape (block matrix + accessor); same implementation strategy |
| R `fdapace`: PACE BLUP trajectories + bands | fdars `pace_fpca`: already ships BLUP + bands | `face_trajectory` is a thin wrapper; no new BLUP implementation needed |

**Deprecated/outdated:** None — this is entirely additive.

---

## Environment Availability

All dependencies are already present in the Cargo.lock (no new packages).

| Dependency | Required By | Available | Notes |
|------------|-------------|-----------|-------|
| nalgebra 0.33 | PSD clipping via `symmetric_eigen` | Yes | Already in Cargo.toml |
| `cov_irreg` | raw covariance step | Yes | `irreg_fdata/smoothing.rs:111` |
| `gaussian_smooth_cov` | sandwich smoother | Yes (needs `pub(crate)`) | `fpca_variants.rs:589`, currently private |
| `pace_fpca` | fitted trajectories entry point | Yes | `pace_fpca.rs:272` |
| `simpsons_weights` | PSD sandwich weights | Yes | `helpers.rs` |
| `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` | Avoid /tmp exhaustion | Configured per MEMORY.md | Set before all cargo commands |

**Missing dependencies with no fallback:** None.

---

## Security Domain

Security enforcement is enabled. V5 input validation is the only applicable ASVS category for this pure-numeric Rust library phase.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | not applicable — library crate, no auth |
| V3 Session Management | no | not applicable |
| V4 Access Control | no | not applicable |
| V5 Input Validation | yes | `FdarError::InvalidDimension` / `FdarError::InvalidParameter` at function entry |
| V6 Cryptography | no | not applicable |

All three new entry points must validate inputs at the top (before any computation) and return `FdarError` for: empty sample, zero/short grid, non-monotone grid, invalid bandwidth, mismatched variable counts (mface).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | `gaussian_smooth_cov` can be made `pub(crate)` in `fpca_variants.rs` with a single keyword change, with no effect on the public API or existing tests | Standard Stack | Very low — it is already a private `fn` with no public callers; adding `pub(crate)` only widens visibility within the crate |
| A2 | The dense-limit tolerance of 0.3 for `face_covariance` recovery of a known covariance surface is achievable with n=30 dense curves and bandwidth=0.05 | Validation Architecture | Low-medium — calibrated from similar experiments in `pace_fpca` (35% bias for n=20 sparse); dense curves (all 51 grid points observed) should give tighter recovery; executor should calibrate and document |
| A3 | The cross-block tolerance of 0.4 for `mface_covariance` off-diagonal block recovery is achievable with n=20 sparse curves and a rank-1 known cross-structure | Validation Architecture | Low-medium — same bias regime as pace_fpca synthetic test; executor should calibrate |
| A4 | `FdMatrix::from_column_major` returns `Err` (not panics) on dimension mismatch in the PSD reconstruction step | Code Examples (Pattern 1) | Very low — `from_column_major` is defined in `matrix.rs` and returns `Result`; the `.expect()` call in `cov_irreg` is the exception, not the rule |

**If this table is empty:** n/a — four assumptions logged above, all low risk.

---

## Open Questions

1. **Whether to inline `gaussian_smooth_cov` or make it `pub(crate)`**
   - What we know: the function exists at `fpca_variants.rs:589`, currently private to that module
   - What's unclear: whether the project style prefers DRY reuse vs. module isolation
   - Recommendation: make it `pub(crate)` (one-character change, eliminates duplication, keeps the sandwich logic in one place)

2. **Eigenvalue threshold for PSD clipping: `> 0.0` vs `> -eps`**
   - What we know: `pace_fpca` uses `lam > 0.0` (exact); `ssvd` also uses `lam > 0.0`
   - What's unclear: whether very small positive eigenvalues from estimation noise should also be clipped
   - Recommendation: use `> 0.0` to match the existing pattern; document that this clips numerical noise; if empirically needed, use `> 1e-10` (but match the existing pattern first)

3. **`face_trajectory` as a thin wrapper vs richer entry point**
   - What we know: locked decision says "reuse `PaceFpcaResult` or a thin wrapper"; `pace_fpca` already does everything SPARSE-01-03 requires
   - What's unclear: whether to accept a `PaceFpcaConfig` directly or define a narrower `FaceTrajectoryConfig`
   - Recommendation: accept `&PaceFpcaConfig` directly (zero new config struct, zero new structs needed); document that `sigma2` and `alpha` are the primary tuning knobs

---

## Sources

### Primary (HIGH confidence — read this session)

- [VERIFIED: fdars-core/src/irreg_fdata/mod.rs:38-213] — `IrregFdata` struct and all methods
- [VERIFIED: fdars-core/src/irreg_fdata/smoothing.rs:111-176] — `cov_irreg` and `accumulate_cov_at_point`
- [VERIFIED: fdars-core/src/irreg_fdata/kernels.rs:1-90] — `mean_irreg`, `kernel_gaussian`, `KernelType`
- [VERIFIED: fdars-core/src/pace_fpca.rs:1-1327] — full `pace_fpca` implementation: `PaceFpcaConfig`, `PaceFpcaResult`, `eigendecompose_cov`, BLUP, band machinery
- [VERIFIED: fdars-core/src/fpca_variants.rs:589-799] — `gaussian_smooth_cov` (private), `ssvd` full sandwich+PSD pattern
- [VERIFIED: fdars-core/src/lib.rs:64-370] — crate root structure, existing re-exports, module list
- [VERIFIED: fdars-core/src/irreg_fdata/tests.rs:1-100] — existing test patterns for the module

### Secondary (MEDIUM confidence)

- [ASSUMED] `refund::face` (Xiao et al. 2016) uses penalized tensor-product splines for the FACE estimator — the kernel-sandwich approximation is documented as divergent in the locked decision and consistent with the Phase 37 `ssvd` rustdoc divergence pattern
- [ASSUMED] `mfaces` constructs a block covariance with within-variable and cross-variable blocks — consistent with the locked decision description and standard multivariate FDA methodology

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all code read directly from source files this session
- Architecture: HIGH — patterns verified in `pace_fpca.rs` and `fpca_variants.rs` source
- Pitfalls: HIGH — derived from reading the actual implementations (column-major layout, `cov_irreg` non-`Result`, `symmetric_eigen` ascending order)
- Test tolerances: MEDIUM — calibrated by analogy from `pace_fpca` synthetic test comments; executor must calibrate during implementation

**Research date:** 2026-08-21
**Valid until:** 2026-09-21 (stable codebase — no external deps to drift)
