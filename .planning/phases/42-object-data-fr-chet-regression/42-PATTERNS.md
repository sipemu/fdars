# Phase 42: Object-Data Fréchet Regression - Pattern Map

**Mapped:** 2026-08-22
**Files analyzed:** 7 (5 new space files, 2 modified existing files + mod/lib edits)
**Analogs found:** 7 / 7

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|---|---|---|---|---|
| `frechet/spaces/spd.rs` (NEW) | service/backend | transform | `frechet/space.rs` (WassersteinDensitySpace impl) | exact-role |
| `frechet/spaces/correlation.rs` (NEW) | service/backend | transform | `frechet/space.rs` (WassersteinDensitySpace impl) | exact-role |
| `frechet/spaces/spherical.rs` (NEW) | service/backend | transform (iterative) | `frechet/space.rs` (WassersteinDensitySpace impl) | exact-role |
| `frechet/spaces/network.rs` (NEW) | service/backend | transform | `frechet/space.rs` (WassersteinDensitySpace impl) | exact-role |
| `frechet/spaces/point_process.rs` (NEW) | service/backend | transform | `frechet/space.rs` (WassersteinDensitySpace impl) | exact-role |
| `frechet/regression.rs` (MODIFY) | service | request-response | `frechet/regression.rs` (existing) | self |
| `frechet/anova.rs` (MODIFY) | service | request-response | `frechet/anova.rs` (existing) | self |
| `frechet/spaces/mod.rs` (NEW) | config/re-export | — | `frechet/mod.rs` | exact-role |
| `frechet/mod.rs` (MODIFY) | config/re-export | — | `frechet/mod.rs` (existing) | self |
| `src/lib.rs` (MODIFY) | config/re-export | — | `src/lib.rs` (existing) | self |

---

## Pattern Assignments

### `frechet/spaces/spd.rs`, `correlation.rs`, `network.rs`, `point_process.rs`, `spherical.rs` (NEW — MetricSpace backends)

**Analog:** `fdars-core/src/frechet/space.rs` — `WassersteinDensitySpace` impl

**Struct + derive pattern** (space.rs lines 51-56):
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WassersteinDensitySpace {
    /// Shared strictly-increasing evaluation grid for all density objects.
    pub argvals: Vec<f64>,
}
```
Every new space struct follows the same derive block. For structs with an enum field (e.g. `SpdMatrixSpace { d: usize, metric: SpdMetric }`), add `#[non_exhaustive]` only if it is a *result type*; plain space structs do NOT need `#[non_exhaustive]` (only result structs get it, per `mod.rs` pattern).

**Constructor validation pattern** (space.rs lines 65-80):
```rust
pub fn new(argvals: Vec<f64>) -> Result<Self, FdarError> {
    if argvals.len() < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: "at least 2 grid points".to_string(),
            actual: format!("{} points", argvals.len()),
        });
    }
    if argvals.windows(2).any(|w| w[1] <= w[0]) {
        return Err(FdarError::InvalidParameter {
            parameter: "argvals",
            message: "argvals must be strictly increasing".to_string(),
        });
    }
    Ok(Self { argvals })
}
```
For space structs: validate `d >= 1` on construction; no SPD-ness check at construction (only Log-Cholesky path does Cholesky internally).

**MetricSpace trait impl pattern** (space.rs lines 83-121):
```rust
impl MetricSpace for WassersteinDensitySpace {
    type Object = Vec<f64>;

    fn distance(&self, a: &Self::Object, b: &Self::Object) -> Result<f64, FdarError> {
        wasserstein2_distance(a, b, &self.argvals)
    }

    fn weighted_frechet_mean(
        &self,
        objects: &[Self::Object],
        weights: &[f64],
    ) -> Result<Self::Object, FdarError> {
        let m = self.argvals.len();
        if objects.is_empty() {
            return Err(FdarError::InvalidDimension {
                parameter: "objects",
                expected: "at least 1 object".to_string(),
                actual: "0 objects".to_string(),
            });
        }
        // ... dim-check each object, then compute mean
    }
}
```
All five new backends use `type Object = Vec<f64>`. Dimension check per object inside `weighted_frechet_mean` before computation:
```rust
if obj.len() != expected_len {
    return Err(FdarError::InvalidDimension {
        parameter: "objects",
        expected: format!("each object has {expected_len} elements"),
        actual: format!("object {i} has {} elements", obj.len()),
    });
}
```

**SymmetricEigen / matrix-power pattern** (from `fts/acf.rs` lines 337-348, confirmed):
```rust
use nalgebra::DMatrix;

// Build from column-major flat Vec<f64> (MUST use from_column_slice, not from_row_slice)
let mut mat = DMatrix::from_column_slice(d, d, &mat_flat);
// Symmetrize defensively
for j1 in 0..d {
    for j2 in (j1 + 1)..d {
        let avg = 0.5 * (mat[(j1, j2)] + mat[(j2, j1)]);
        mat[(j1, j2)] = avg;
        mat[(j2, j1)] = avg;
    }
}
let eig = nalgebra::SymmetricEigen::new(mat);
// eig.eigenvalues: ascending order (smallest first)
// eig.eigenvectors: columns are eigenvectors
// Reconstruct V · diag(λ^α) · Vᵀ in column-major output:
let evecs = &eig.eigenvectors;
let evals = &eig.eigenvalues;
let mut result = vec![0.0_f64; d * d];
for k in 0..d {
    let lk_alpha = evals[k].max(0.0).powf(alpha);  // clamp negatives before powf
    for i in 0..d {
        for j in 0..d {
            result[i + j * d] += evecs[(i, k)] * lk_alpha * evecs[(j, k)];
        }
    }
}
```
Column-major index: element `(i, j)` is at `i + j*d`. Use `DMatrix::from_column_slice` always. Eigenvalues are ascending — no sort needed for `V diag(λ^α) Vᵀ` reconstruction.

**Cholesky pattern** (from `linalg.rs` lines 85-108, used in `regression.rs` line 122):
```rust
// Import: use crate::linalg::{cholesky_factor, cholesky_forward_back};
let l = cholesky_factor(&mat_flat_row_major, d)?;  // returns ComputationFailed on non-PD
// l is row-major: element (i,j) at l[i*d + j]
// diagonal: l[i*d + i]
// strictly lower: l[i*d + j] for i > j
```
Note: `cholesky_factor` takes a **row-major** flat `&[f64]` and returns a **row-major** lower-triangular `Vec<f64>`. The crate's normal column-major `Vec<f64>` for symmetric SPD matrices happens to be identical to row-major (since A = Aᵀ for symmetric matrices) — but reconstruct `L Lᵀ` in column-major output explicitly.

**`ComputationFailed` error pattern** (from space.rs and anova.rs):
```rust
return Err(FdarError::ComputationFailed {
    operation: "SphericalSpace::weighted_frechet_mean",
    detail: "Karcher mean did not converge in 50 iterations".to_string(),
});
```

**Inline test pattern** (space.rs lines 283-427):
```rust
#[cfg(test)]
mod tests {
    use super::*;  // gets space struct + MetricSpace impl

    // For cross-module types use absolute crate path:
    // use crate::error::FdarError;

    fn helper_data() -> Vec<f64> { ... }

    #[test]
    fn distance_of_identical_is_zero() { ... }

    #[test]
    fn weighted_mean_of_identical_recovers_object() { ... }

    #[test]
    fn rejects_dimension_mismatch() {
        assert!(matches!(
            space.distance(&a, &b_wrong).unwrap_err(),
            FdarError::InvalidDimension { .. }
        ));
    }
}
```

---

### `frechet/regression.rs` (MODIFY — extract helpers + add generic entry points)

**Analog:** Self (`fdars-core/src/frechet/regression.rs`)

**Global weight block to extract into `pub(crate) compute_global_weights`** (regression.rs lines 94-141):
```rust
// Lines 95-121: predictor means + sample covariance + ridge + Cholesky
let mut x_bar = vec![0.0; p];
for j in 0..p {
    let mut s = 0.0;
    for i in 0..n { s += predictors[(i, j)]; }
    x_bar[j] = s / n as f64;
}
let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
let mut sigma = vec![0.0; p * p];
for i in 0..n {
    for a in 0..p {
        let da = predictors[(i, a)] - x_bar[a];
        for b in 0..p {
            sigma[a * p + b] += (predictors[(i, b)] - x_bar[b]) * da;
        }
    }
}
for v in sigma.iter_mut() { *v /= denom; }
for j in 0..p { sigma[j * p + j] += 1e-6; }
let chol = cholesky_factor(&sigma, p)?;

// Lines 125-136: per xout-row signed weight computation
for r in 0..n_out {
    let diff_x: Vec<f64> = (0..p).map(|j| xout[(r, j)] - x_bar[j]).collect();
    let v = cholesky_forward_back(&chol, &diff_x, p);
    let mut weights = vec![0.0; n];
    for i in 0..n {
        let mut dot = 0.0;
        for j in 0..p { dot += (predictors[(i, j)] - x_bar[j]) * v[j]; }
        weights[i] = (1.0 + dot) / n as f64;
    }
    // ... use weights
}
```
Extract these two blocks into:
```rust
pub(crate) fn compute_global_weights(
    predictors: &FdMatrix,
    xout: &FdMatrix,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), FdarError>
// returns (weights_per_row: Vec<Vec<f64>>, x_bar: Vec<f64>)
```

**Local weight block to extract into `pub(crate) compute_local_weights`** (regression.rs lines 188-241):
```rust
// Product Gaussian kernel weights (lines 188-196):
let mut kern = vec![0.0; n];
for i in 0..n {
    let mut k = 1.0;
    for j in 0..p { k *= gaussian_kernel(predictors[(i, j)] - x0[j], bandwidth); }
    kern[i] = k;
}
// Local moments + ridge + solve (lines 199-221):
let mut mu1 = vec![0.0; p];
let mut mu2 = vec![0.0; p * p];
for i in 0..n {
    let ki = kern[i];
    for a in 0..p {
        let da = predictors[(i, a)] - x0[a];
        mu1[a] += ki * da;
        for b in 0..p { mu2[a * p + b] += ki * da * (predictors[(i, b)] - x0[b]); }
    }
}
for v in mu1.iter_mut() { *v /= n as f64; }
for v in mu2.iter_mut() { *v /= n as f64; }
for j in 0..p { mu2[j * p + j] += 1e-6; }
let a_vec = cholesky_solve(&mu2, &mu1, p)?;
// Local-linear signed weights (lines 223-241):
let mut weights = vec![0.0; n];
for i in 0..n {
    let mut corr = 0.0;
    for j in 0..p { corr += (predictors[(i, j)] - x0[j]) * a_vec[j]; }
    weights[i] = kern[i] * (1.0 - corr);
}
let sum_w: f64 = weights.iter().sum();
if sum_w.abs() < NUMERICAL_EPS {
    return Err(FdarError::ComputationFailed {
        operation: "frechet_local_reg",
        detail: "local weights sum to zero (bandwidth too small or no nearby points)".to_string(),
    });
}
for w in weights.iter_mut() { *w /= sum_w; }
```
Extract into:
```rust
pub(crate) fn compute_local_weights(
    predictors: &FdMatrix,
    x0: &[f64],
    bandwidth: f64,
    n: usize,
    p: usize,
) -> Result<Vec<f64>, FdarError>
```

**Refactored density path pattern** (existing `frechet_global_reg` after extraction):
```rust
pub fn frechet_global_reg(...) -> Result<FrechetGlobalRegResult, FdarError> {
    let (n, p, m) = validate_reg_input(predictors, responses, argvals, xout)?;
    let n_q = m.max(101);
    let (weights_per_row, x_bar) = compute_global_weights(predictors, xout)?;
    let n_out = xout.nrows();
    let mut predicted = FdMatrix::zeros(n_out, m);
    for r in 0..n_out {
        let dens = signed_quantile_average(responses, argvals, &weights_per_row[r], n_q)?;
        for j in 0..m { predicted[(r, j)] = dens[j]; }
    }
    Ok(FrechetGlobalRegResult { predicted, xout: xout.clone(), x_bar })
}
```

**New generic entry-point pattern**:
```rust
#[must_use = "expensive regression — store or use the returned prediction"]
pub fn frechet_global_reg_space<S: MetricSpace>(
    space: &S,
    predictors: &FdMatrix,
    responses: &[S::Object],
    xout: &FdMatrix,
) -> Result<Vec<S::Object>, FdarError> {
    // validate n, p alignment (responses.len() == predictors.nrows(), etc.)
    let (weights_per_row, _x_bar) = compute_global_weights(predictors, xout)?;
    let mut out = Vec::with_capacity(xout.nrows());
    for r in 0..xout.nrows() {
        let pred = space.weighted_frechet_mean(responses, &weights_per_row[r])?;
        out.push(pred);
    }
    Ok(out)
}
```
Note: signed weights (possible negatives) are passed directly to `weighted_frechet_mean` for the four linear-combination spaces (SPD Frobenius, Correlation, Network, PointProcess). For `SphericalSpace` the implementation should clip negatives to zero and renormalize before the Karcher call; document this per-space divergence in rustdoc.

**imports needed in regression.rs** (regression.rs lines 11-16):
```rust
use super::{FrechetGlobalRegResult, FrechetLocalRegResult};
use crate::error::FdarError;
use crate::frechet::space::{signed_quantile_average, MetricSpace};  // add MetricSpace
use crate::helpers::{gaussian_kernel, NUMERICAL_EPS};
use crate::linalg::{cholesky_factor, cholesky_forward_back, cholesky_solve};
use crate::matrix::FdMatrix;
```

---

### `frechet/anova.rs` (MODIFY — generify compute_tn + add frechet_anova_space)

**Analog:** Self (`fdars-core/src/frechet/anova.rs`)

**Existing `compute_tn` signature to generify** (anova.rs lines 32-37):
```rust
// BEFORE (concrete):
fn compute_tn(
    space: &WassersteinDensitySpace,
    objects: &[Vec<f64>],
    labels: &[usize],
    k: usize,
) -> Result<(f64, f64, f64, Vec<f64>, f64), FdarError>

// AFTER (generic — rename to compute_tn_generic):
pub(crate) fn compute_tn_generic<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    labels: &[usize],
    k: usize,
) -> Result<(f64, f64, f64, Vec<f64>, f64), FdarError>
```
The body is unchanged: it calls `frechet_mean(space, ...)` and `frechet_variance(space, ...)` which are already generic over `S: MetricSpace` (confirmed: `mean.rs` lines 40-83). `space.distance(o, &mu_g)` also works because `distance` is a trait method. The existing `frechet_anova` simply calls `compute_tn_generic(&space, &objects, ...)` with a `WassersteinDensitySpace` — no API change.

**Full Tₙ body to keep verbatim** (anova.rs lines 40-97):
```rust
let n = objects.len();
let mut groups: Vec<Vec<usize>> = vec![Vec::new(); k];
for (i, &g) in labels.iter().enumerate() { groups[g].push(i); }

let pooled_mean = frechet_mean(space, objects, None)?;
let pooled_var = frechet_variance(space, objects, &pooled_mean, None)?;

let mut group_vars = vec![0.0; k];
let mut sigma2 = vec![0.0; k];
let mut lambda = vec![0.0; k];
for (g, idx) in groups.iter().enumerate() {
    let n_g = idx.len();
    lambda[g] = n_g as f64 / n as f64;
    let subset: Vec<_> = idx.iter().map(|&i| objects[i].clone()).collect();
    let mu_g = frechet_mean(space, &subset, None)?;
    let d2: Vec<f64> = subset.iter()
        .map(|o| space.distance(o, &mu_g).map(|d| d * d))
        .collect::<Result<Vec<f64>, _>>()?;
    let v_g = d2.iter().sum::<f64>() / n_g as f64;
    let s2 = d2.iter().map(|&d| (d - v_g).powi(2)).sum::<f64>() / n_g as f64;
    group_vars[g] = v_g;
    sigma2[g] = s2.max(NUMERICAL_EPS);
}
// ... Fn, Un, Tn computation unchanged (lines 71-97)
```

**Seeded permutation loop pattern** (anova.rs lines 168-179):
```rust
let n_perm = if n_perm == 0 { 999 } else { n_perm };
let mut n_ge = 0usize;
for perm in 0..n_perm {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
    let mut perm_labels = group_labels.to_vec();
    perm_labels.shuffle(&mut rng);
    if let Ok((tn_perm, _, _, _, _)) = compute_tn_generic(space, objects, &perm_labels, k) {
        if tn_perm >= tn_obs { n_ge += 1; }
    }
}
let p_permutation = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

**New generic ANOVA entry point**:
```rust
#[must_use = "returns the Fréchet ANOVA result; examine the p-values"]
pub fn frechet_anova_space<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    group_labels: &[usize],
    n_perm: usize,
    seed: u64,
) -> Result<FrechetAnovaResult, FdarError>
```
Validation mirrors `frechet_anova` (lines 127-155) but without the `argvals`/`FdMatrix` checks (objects are already `S::Object`).

**Imports in anova.rs** (anova.rs lines 9-18):
```rust
use super::mean::{frechet_mean, frechet_variance};
use super::space::{MetricSpace, WassersteinDensitySpace};  // add MetricSpace
use super::FrechetAnovaResult;
use crate::error::FdarError;
use crate::helpers::NUMERICAL_EPS;
use crate::inference::dist::chi_square_sf;
use crate::matrix::FdMatrix;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
```

---

### `frechet/spaces/mod.rs` (NEW — barrel re-export)

**Analog:** `frechet/mod.rs` lines 34-42

```rust
mod correlation;
mod network;
mod point_process;
mod spherical;
mod spd;

pub use correlation::CorrelationMatrixSpace;
pub use network::NetworkSpace;
pub use point_process::PointProcessSpace;
pub use spherical::SphericalSpace;
pub use spd::{SpdMatrixSpace, SpdMetric};
```
No wildcard re-exports — list each item explicitly (project convention).

---

### `frechet/mod.rs` (MODIFY — add `mod spaces` + new pub uses)

**Analog:** Self (frechet/mod.rs lines 34-42)

Current block:
```rust
mod anova;
mod mean;
mod regression;
mod space;

pub use anova::frechet_anova;
pub use mean::{frechet_mean, frechet_variance};
pub use regression::{frechet_global_reg, frechet_local_reg};
pub use space::{wasserstein2_distance, MetricSpace, WassersteinDensitySpace};
```

Add:
```rust
mod spaces;  // NEW

pub use anova::frechet_anova_space;  // NEW
pub use regression::{frechet_global_reg_space, frechet_local_reg_space};  // NEW
pub use spaces::{
    CorrelationMatrixSpace, NetworkSpace, PointProcessSpace,
    SphericalSpace, SpdMatrixSpace, SpdMetric,  // NEW
};
```

Result structs `FrechetGlobalRegResult`, `FrechetLocalRegResult`, `FrechetAnovaResult` are defined in `frechet/mod.rs` (lines 46-102) with this exact pattern:
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FrechetGlobalRegResult {
    pub predicted: FdMatrix,
    pub xout: FdMatrix,
    pub x_bar: Vec<f64>,
}
```
`#[non_exhaustive]` is applied to all *result structs*; space structs do NOT get it.

---

### `src/lib.rs` (MODIFY — extend frechet re-export block)

**Analog:** Self (lib.rs lines 152-156)

Current block:
```rust
pub use frechet::{
    frechet_anova, frechet_global_reg, frechet_local_reg, frechet_mean, frechet_variance,
    wasserstein2_distance, FrechetAnovaResult, FrechetGlobalRegResult, FrechetLocalRegResult,
    MetricSpace, WassersteinDensitySpace,
};
```

Extend (do NOT replace — add to the `pub use frechet::{...}` list):
```rust
pub use frechet::{
    // existing items (unchanged):
    frechet_anova, frechet_global_reg, frechet_local_reg, frechet_mean, frechet_variance,
    wasserstein2_distance, FrechetAnovaResult, FrechetGlobalRegResult, FrechetLocalRegResult,
    MetricSpace, WassersteinDensitySpace,
    // NEW:
    frechet_anova_space, frechet_global_reg_space, frechet_local_reg_space,
    CorrelationMatrixSpace, NetworkSpace, PointProcessSpace,
    SphericalSpace, SpdMatrixSpace, SpdMetric,
};
```

---

## Shared Patterns

### Error Handling
**Source:** `fdars-core/src/error.rs` (used throughout `frechet/`)
**Apply to:** All new space files, modified regression.rs, modified anova.rs

Four variants only — use exactly these:
```rust
FdarError::InvalidDimension { parameter: &'static str, expected: String, actual: String }
FdarError::InvalidParameter { parameter: &'static str, message: String }
FdarError::ComputationFailed { operation: &'static str, detail: String }
FdarError::InvalidEnumValue { enum_name: &'static str, value: String }
```
- Dimension mismatches → `InvalidDimension`
- Out-of-range / non-positive parameters (e.g. `alpha <= 0`, `d == 0`) → `InvalidParameter`
- Non-convergence, non-PD matrix, degenerate weights → `ComputationFailed`

### `#[must_use]` on expensive computations
**Source:** `frechet/regression.rs` lines 83, 167; `frechet/anova.rs` line 118
**Apply to:** `frechet_global_reg_space`, `frechet_local_reg_space`, `frechet_anova_space`

```rust
#[must_use = "expensive regression — store or use the returned prediction"]
pub fn frechet_global_reg_space<S: MetricSpace>(...)
```

### Seeded permutation RNG
**Source:** `frechet/anova.rs` lines 169-171
**Apply to:** `frechet_anova_space` permutation loop

```rust
let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
```
Use `seed.wrapping_add(perm as u64)` — NOT `seed + perm as u64` (prevents overflow panic in debug mode).

### Column-major flat matrix indexing
**Source:** Throughout `frechet/` and `matrix.rs`
**Apply to:** All SPD/Correlation/Network space computations

Element `(i, j)` of a `d×d` column-major flat `Vec<f64>`: index = `i + j * d`.
`DMatrix::from_column_slice(d, d, &flat)` interprets input as column-major — always use this form.
Reconstruction from nalgebra: `result[i + j * d] = mat[(i, j)]`.

### Serde-gating on structs
**Source:** `frechet/mod.rs` lines 51-52, 65-66, 81-82; `frechet/space.rs` lines 52-53
**Apply to:** All new space structs and any new result structs

```rust
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
```

### Inline test module with cross-module imports
**Source:** `frechet/space.rs` lines 283-286
**Apply to:** All new `frechet/spaces/*.rs` files

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::FdarError;  // explicit crate path for cross-module types
    // NOT: use crate::frechet::space::FdarError — use the direct crate::error path
}
```

---

## No Analog Found

None — all files have exact or role-match analogs within the `frechet/` module.

---

## Metadata

**Analog search scope:** `fdars-core/src/frechet/` (all files), `fdars-core/src/fts/acf.rs` (SymmetricEigen pattern), `fdars-core/src/linalg.rs` (Cholesky), `fdars-core/src/lib.rs` (re-export block)
**Files read:** 8
**Pattern extraction date:** 2026-08-22

---

## PATTERN MAPPING COMPLETE

**Phase:** 42 - Object-Data Fréchet Regression
**Files classified:** 10 (5 new space files + spaces/mod.rs + 2 modified + mod.rs + lib.rs)
**Analogs found:** 10 / 10

### Coverage
- Files with exact analog: 4 (mod.rs, lib.rs, regression.rs, anova.rs — self-referential)
- Files with role-match analog: 6 (all 5 space files + spaces/mod.rs — `WassersteinDensitySpace` / `frechet/mod.rs`)
- Files with no analog: 0

### Key Patterns Identified
- All five new MetricSpace backends mirror `WassersteinDensitySpace` exactly: same derive block, same `new()` validation, same `impl MetricSpace` with empty-check + per-object dim-check + `type Object = Vec<f64>`
- SPD matrix power uses `nalgebra::SymmetricEigen::new(DMatrix::from_column_slice(d, d, &flat))` (verified pattern from `fts/acf.rs:337-345`); column-major reconstruction: `result[i + j*d] += evecs[(i,k)] * lk^alpha * evecs[(j,k)]`
- Log-Cholesky uses `crate::linalg::cholesky_factor` (row-major L, element at `l[i*d+j]`); coordinate map: lower entries as-is, diagonal as `log(lii)`; back-map: lower as-is, diagonal as `exp`; reconstruct `M = L̄ L̄ᵀ` in column-major
- Generic solver refactoring is mechanical: extract lines 94-136 of `regression.rs` into `pub(crate) compute_global_weights`, lines 188-241 into `pub(crate) compute_local_weights`; rename `compute_tn` → `compute_tn_generic<S: MetricSpace>` with no body change
- Permutation seeding: `StdRng::seed_from_u64(seed.wrapping_add(perm as u64))` — use `wrapping_add`, not `+`
- `#[non_exhaustive]` only on result structs; space structs derive `Debug, Clone, PartialEq` + serde-gate only
