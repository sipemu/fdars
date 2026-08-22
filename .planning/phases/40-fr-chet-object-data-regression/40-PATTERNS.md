# Phase 40: Fréchet / Object-Data Regression - Pattern Map

**Mapped:** 2026-08-22
**Files analyzed:** 6 (5 new files + 1 modified)
**Analogs found:** 6 / 6

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/src/frechet/mod.rs` | module barrel + result types | — | `fdars-core/src/fts/mod.rs` | exact |
| `fdars-core/src/frechet/space.rs` | trait definition + backend + distance helper | transform | `fdars-core/src/density_fda.rs` (inner loop) + `explain_generic` (trait design) | role-match |
| `fdars-core/src/frechet/mean.rs` | utility functions | CRUD | `fdars-core/src/fts/acf.rs` — generic stat functions calling density_fda | role-match |
| `fdars-core/src/frechet/regression.rs` | service / fitting | request-response | `fdars-core/src/density_fda.rs` + `fdars-core/src/linalg.rs` | role-match |
| `fdars-core/src/frechet/anova.rs` | test / permutation | event-driven | `fdars-core/src/inference/permutation.rs` | exact |
| `fdars-core/src/lib.rs` (modified) | crate root wiring | — | existing lib.rs `density_fda` block (lines 146–148) | exact |

---

## Pattern Assignments

### `fdars-core/src/frechet/mod.rs` (module barrel + result structs)

**Analog:** `fdars-core/src/fts/mod.rs`

**Module doc comment + submodule declaration pattern** (fts/mod.rs lines 1–27):
```rust
//! Fréchet regression and object-data statistics.
//!
//! # R baseline
//!
//! * [`frechet_global_reg`] / [`frechet_local_reg`] — `frechet::GloWassReg` / `LocWassReg`
//!   (Petersen & Müller 2019, *Annals of Statistics* 47(2)).
//! * [`frechet_anova`] — `frechet::DenANOVA` (Dubey & Müller 2019, *Biometrika*).
//!
//! # Conventions
//!
//! Entry points take an explicit deterministic `seed` for permutation tests
//! (`StdRng::seed_from_u64(seed + k)`), default 999 permutations.
//! All public functions return `Result<_, FdarError>` and validate inputs at entry.
//! Result structs derive `Debug, Clone, PartialEq` and are serde-gated.

mod anova;
mod mean;
mod regression;
mod space;

pub use anova::{frechet_anova, FrechetAnovaResult};
pub use mean::{frechet_mean, frechet_variance};
pub use regression::{frechet_global_reg, frechet_local_reg, FrechetGlobalRegResult, FrechetLocalRegResult};
pub use space::{MetricSpace, WassersteinDensitySpace};
```

**Result struct pattern** (fts/mod.rs lines 33–45 — exact template to copy for each new result type):
```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct FacfResult {
    /// Lag values (1..=max_lag).
    pub lags: Vec<u32>,
    /// Functional autocorrelation ρ_h at each lag.
    pub acf: Vec<f64>,
}
```

Apply this template to define in `mod.rs`:
- `FrechetGlobalRegResult` — fields: `predicted: FdMatrix`, `xout: FdMatrix`, `x_bar: Vec<f64>`
- `FrechetLocalRegResult` — fields: `predicted: FdMatrix`, `xout: FdMatrix`, `bandwidth: f64`
- `FrechetAnovaResult` — fields: `statistic`, `p_value_asymptotic`, `p_value_permutation`, `n_perm`, `group_frechet_variances`, `pooled_frechet_variance`, `fn_statistic`, `un_statistic`, `group_labels`

**`#[must_use]` annotation** — apply to all three result-producing public functions (mirrors 74+ existing usages in codebase).

---

### `fdars-core/src/frechet/space.rs` (MetricSpace trait + WassersteinDensitySpace + helpers)

**Analog for trait design:** `fdars-core/src/explain_generic.rs` (FpcPredictor trait — Send + Sync public trait)

The `MetricSpace` trait must be `Send + Sync` (same as `FpcPredictor`) to allow rayon-parallel regression loops:
```rust
// explain_generic.rs — trait design template
pub trait FpcPredictor: Send + Sync {
    fn fpca_mean(&self) -> &[f64];
    fn ncomp(&self) -> usize;
    // ...
}
```

Apply to `MetricSpace`:
```rust
pub trait MetricSpace: Send + Sync {
    type Object;
    fn distance(&self, a: &Self::Object, b: &Self::Object) -> Result<f64, FdarError>;
    fn weighted_frechet_mean(
        &self,
        objects: &[Self::Object],
        weights: &[f64],
    ) -> Result<Self::Object, FdarError>;
}
```

**Imports for space.rs:**
```rust
use crate::density_fda::wasserstein_barycenter;
use crate::error::FdarError;
use crate::helpers::{cumulative_trapz, linear_interp, trapz, NUMERICAL_EPS};
use crate::matrix::FdMatrix;
```

**W₂ distance core — density→quantile inner loop** (density_fda.rs lines 468–496 — the EXACT pattern to replicate as a private `density_to_quantile` helper):
```rust
// density_fda.rs:469 — quantile grid resolution
let n_q = m.max(101);
let t_grid: Vec<f64> = (0..n_q).map(|i| i as f64 / (n_q - 1) as f64).collect();

// density_fda.rs:474–496 — density → normalized CDF → quantile via interpolation
let mut q_bar = vec![0.0_f64; n_q];
for i in 0..n {
    let row: Vec<f64> = (0..m).map(|j| density_matrix[(i, j)]).collect();
    let integral = trapz(&row, argvals);               // helpers::trapz
    let norm_row: Vec<f64> = row.iter().map(|&v| v / integral).collect();
    let cdf_i = cumulative_trapz(&norm_row, argvals);  // helpers::cumulative_trapz
    let wi = w_vec[i];
    for j in 0..n_q {
        q_bar[j] += wi * linear_interp(&cdf_i, argvals, t_grid[j]);  // helpers::linear_interp
    }
}
```

Use this pattern in the private `density_to_quantile(row, argvals, t_grid) -> Vec<f64>` helper and in `w2_distance`.

**Q̄ range guard** (density_fda.rs lines 504–510 — mirror in `signed_quantile_average`):
```rust
// density_fda.rs:504-510
let q_range = q_bar[n_q - 1] - q_bar[0];
if q_range < 1e-15 {
    return Err(FdarError::ComputationFailed {
        operation: "wasserstein_barycenter",
        detail: "quantile average has zero range; degenerate input densities".to_string(),
    });
}
```

**Quantile-to-density back-map** (density_fda.rs lines 500–535 — reference for `signed_quantile_average` Step 4–5):
```rust
// density_fda.rs:511-535 — rescale Q̄ to support, finite-diff density, dedup, interpolate, renormalize
let d_range = ub - lb;
let q_scaled: Vec<f64> = q_bar
    .iter()
    .map(|&v| (v - q_bar[0]) * d_range / q_range + lb)
    .collect();
let dens_raw = quantile_density_from_q(&q_scaled, &t_grid);
let (q_dedup, dens_dedup) = dedup_adjacent(&q_scaled, &dens_raw);
let dens: Vec<f64> = argvals
    .iter()
    .map(|&x| linear_interp(&q_dedup, &dens_dedup, x))
    .collect();
let integral = trapz(&dens, argvals);
Ok(dens.iter().map(|&d| d / integral).collect())
```

Note: `quantile_density_from_q` and `dedup_adjacent` are private helpers in `density_fda.rs`. Either make them `pub(crate)` or re-implement a minimal version in `frechet/space.rs`. The simplest approach: add `pub(crate)` to both helpers in `density_fda.rs` at implementation time.

**`wasserstein_barycenter` call for non-negative weighted mean** (density_fda.rs:407 — the ONLY path for non-negative weights):
```rust
// Call site pattern for WassersteinDensitySpace::weighted_frechet_mean
// (non-negative weights only — NEVER for global/local regression weights)
wasserstein_barycenter(&mat, &self.argvals, Some(weights))
```

**CRITICAL — negative weight guard** (density_fda.rs lines 450–453):
```rust
// density_fda.rs:450-453 — this validation blocks regression use
if w.iter().any(|&wi| wi < 0.0 || !wi.is_finite()) {
    return Err(FdarError::InvalidParameter { ... });
}
```
The `signed_quantile_average` private helper bypasses `wasserstein_barycenter` entirely and computes the quantile average directly with signed weights, then applies `q_bar.sort_by(...)` for monotone projection.

---

### `fdars-core/src/frechet/mean.rs` (frechet_mean, frechet_variance — generic over MetricSpace)

**Analog:** `fdars-core/src/inference/permutation.rs` — validation-first pattern, then delegate to inner functions

**Imports for mean.rs:**
```rust
use crate::error::FdarError;
use crate::frechet::space::MetricSpace;
```

**Validation pattern** (inference/permutation.rs lines 24–59 — mirror for mean.rs):
```rust
// Check empty sample → InvalidDimension
// Check weight length mismatch → InvalidDimension
// Check weight sum near zero → InvalidParameter
```

**Function signature pattern** (consistent with codebase conventions):
```rust
#[must_use = "expensive Fréchet mean computation — store or use the returned object"]
pub fn frechet_mean<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    weights: Option<&[f64]>,
) -> Result<S::Object, FdarError>

pub fn frechet_variance<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    mean: &S::Object,
    weights: Option<&[f64]>,
) -> Result<f64, FdarError>
```

Both functions: validate at entry, resolve uniform weights if `None`, then delegate to trait methods. `frechet_variance` iterates `objects`, calls `space.distance(obj, mean)?`, squares and accumulates weighted sum.

---

### `fdars-core/src/frechet/regression.rs` (frechet_global_reg, frechet_local_reg)

**Analog:** `fdars-core/src/density_fda.rs` (structure) + `fdars-core/src/linalg.rs` (Σ̂⁻¹)

**Imports for regression.rs:**
```rust
use crate::density_fda::wasserstein_barycenter;
use crate::error::FdarError;
use crate::frechet::space::{signed_quantile_average, MetricSpace, WassersteinDensitySpace};
use crate::helpers::{gaussian_kernel, NUMERICAL_EPS};
use crate::linalg::{cholesky_factor, cholesky_forward_back};
use crate::matrix::FdMatrix;
```

**Covariance regularization + Cholesky solve** (elastic_changepoint.rs lines 296–304 + linalg.rs lines 85–134):
```rust
// elastic_changepoint.rs:296-299 — ridge regularization before Cholesky
for j in 0..p {
    cov[(j, j)] += 1e-6;  // ridge to guard near-singular Σ̂
}

// linalg.rs:85-107 — cholesky_factor
pub(crate) fn cholesky_factor(a: &[f64], p: usize) -> Result<Vec<f64>, FdarError> { ... }

// linalg.rs:113-128 — cholesky_forward_back (solves Σ̂⁻¹ b)
pub(crate) fn cholesky_forward_back(l: &[f64], b: &[f64], p: usize) -> Vec<f64> { ... }

// linalg.rs:131-134 — convenience cholesky_solve (= factor + forward_back)
pub(crate) fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Result<Vec<f64>, FdarError> {
    let l = cholesky_factor(a, p)?;
    Ok(cholesky_forward_back(&l, b, p))
}
```

**Global weight formula:**
```rust
// For each prediction point x_pred:
// diff_x = x_pred - x_bar  (length p)
// v = cholesky_solve(&sigma_hat_reg, &diff_x, p)?  (= Σ̂⁻¹(x-x̄))
// For each training obs i:
//   s_i = 1.0 + diff_i.iter().zip(v.iter()).map(|(&a,&b)| a*b).sum::<f64>()
// Then: signed_quantile_average(density_matrix, argvals, &s_vec, n_q)?
```

**Gaussian product kernel for local regression** (helpers.rs line 247):
```rust
// helpers.rs:247
pub fn gaussian_kernel(d: f64, h: f64) -> f64 {
    if h < 1e-15 { return 0.0; }
    (-d * d / (2.0 * h * h)).exp()
}
// Product kernel over p predictors:
// K_i = (0..p).map(|j| gaussian_kernel(x_train[(i,j)] - x_pred[j], h)).product::<f64>()
```

**Bandwidth validation** (mirrors existing parameter validation pattern):
```rust
if h <= 0.0 || !h.is_finite() {
    return Err(FdarError::InvalidParameter {
        parameter: "bandwidth",
        message: "bandwidth h must be positive and finite".to_string(),
    });
}
```

**FdMatrix row extraction pattern** (density_fda.rs line 475 — use this for assembling density rows):
```rust
let row: Vec<f64> = (0..m).map(|j| density_matrix[(i, j)]).collect();
```

---

### `fdars-core/src/frechet/anova.rs` (frechet_anova — Dubey–Müller Tn + permutation)

**Analog:** `fdars-core/src/inference/permutation.rs` (exact role + data flow match)

**Imports for anova.rs:**
```rust
use crate::error::FdarError;
use crate::frechet::mean::{frechet_mean, frechet_variance};
use crate::frechet::space::MetricSpace;
use crate::frechet::FrechetAnovaResult;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use statrs::distribution::{ChiSquared, ContinuousCDF};
```

**Default permutation count** (inference/permutation.rs line 18 — mirror the constant):
```rust
pub const DEFAULT_N_PERM: usize = 999;
```

**Permutation loop pattern** (elastic_explain.rs lines 313–314 — per-thread seeded RNG, NOT a single shared RNG):
```rust
for k in 0..n_perm {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(k as u64));
    let mut perm_labels = group_labels.clone();
    perm_labels.shuffle(&mut rng);
    // recompute Tn with perm_labels
}
let p_value_permutation = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

Note: `inference/permutation.rs` uses a single RNG (line 173: `let mut rng = StdRng::seed_from_u64(seed)`), while `elastic_explain.rs` uses per-iteration seeding. For Fréchet ANOVA, use the **per-iteration pattern** from `elastic_explain.rs:313-314` to match the CONTEXT.md requirement.

**Fisher-Yates shuffle** (inference/permutation.rs lines 120–127 — reusable pattern):
```rust
fn shuffle_labels(v: &mut [usize], rng: &mut StdRng) {
    use rand::Rng;
    let n = v.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=i);
        v.swap(i, j);
    }
}
```
Or use `perm_labels.shuffle(&mut rng)` from `rand::seq::SliceRandom` (both patterns exist in the codebase).

**p-value formula** (inference/permutation.rs line 183):
```rust
let p_value = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
```

**Validation — fewer than 2 groups** (mirrors inference/permutation.rs lines 52–58):
```rust
if n_groups < 2 {
    return Err(FdarError::InvalidParameter {
        parameter: "group_labels",
        message: "Fréchet ANOVA requires at least 2 groups".to_string(),
    });
}
```

**σ̂ₗ² near-zero guard** — use `NUMERICAL_EPS` from `helpers.rs` (value confirmed: `pub const NUMERICAL_EPS: f64 = 1e-12`):
```rust
if sigma_sq_l < NUMERICAL_EPS {
    // clamp denominator to avoid Tn divergence
    sigma_sq_l = NUMERICAL_EPS;
}
```

---

### `fdars-core/src/lib.rs` (modified — add pub mod frechet + crate-root re-exports)

**Analog:** existing `density_fda` block (lib.rs lines 84 + 146–148 — exact template):
```rust
// lib.rs:84 — module declaration (alphabetical order in the pub mod list)
pub mod density_fda;
// Add after `pub mod famm;` or in alphabetical position:
pub mod frechet;

// lib.rs:146-148 — re-export block (add equivalent after density_fda block)
pub use density_fda::{
    inverse_lqd, lqd_fpca, lqd_transform, normalize_density, wasserstein_barycenter, LqdFpcaResult,
};
// Add:
pub use frechet::{
    frechet_anova, frechet_global_reg, frechet_local_reg, frechet_mean, frechet_variance,
    FrechetAnovaResult, FrechetGlobalRegResult, FrechetLocalRegResult,
    MetricSpace, WassersteinDensitySpace,
};
```

---

## Shared Patterns

### Error Handling
**Source:** `fdars-core/src/density_fda.rs` lines 413–435 + `fdars-core/src/inference/permutation.rs` lines 31–58
**Apply to:** All public functions in `frechet/`

```rust
// InvalidDimension — empty input or shape mismatch
return Err(FdarError::InvalidDimension {
    parameter: "density_matrix",
    expected: "at least 1 row".to_string(),
    actual: "0 rows".to_string(),
});
// InvalidParameter — non-monotone argvals, negative density, h <= 0, < 2 groups
return Err(FdarError::InvalidParameter {
    parameter: "argvals",
    message: "argvals must be strictly increasing".to_string(),
});
// ComputationFailed — degenerate quantile range, singular Σ̂
return Err(FdarError::ComputationFailed {
    operation: "frechet_global_reg",
    detail: "quantile average has zero range after signed weight application; \
             prediction point may be too far from training data".to_string(),
});
```

### Strictly-Increasing argvals Validation
**Source:** `fdars-core/src/density_fda.rs` lines 434–439
**Apply to:** All functions taking `argvals: &[f64]`

```rust
if argvals.windows(2).any(|w| w[1] <= w[0]) {
    return Err(FdarError::InvalidParameter {
        parameter: "argvals",
        message: "argvals must be strictly increasing".to_string(),
    });
}
```

### Result Struct Template
**Source:** `fdars-core/src/fts/mod.rs` lines 33–45
**Apply to:** All result types defined in `frechet/mod.rs`

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[non_exhaustive]
pub struct /* Result */ {
    pub /* field */: /* type */,
}
```

### `#[must_use]` on Expensive Computations
**Source:** `fdars-core/src/density_fda.rs` line 562
**Apply to:** `frechet_global_reg`, `frechet_local_reg`, `frechet_anova`, `frechet_mean`

```rust
#[must_use = "expensive computation — store or use the returned result"]
pub fn frechet_global_reg(...) -> Result<FrechetGlobalRegResult, FdarError> {
```

### `#[inline]` on Hot Paths
**Source:** `fdars-core/src/matrix.rs` (row_to_buf, row_dot)
**Apply to:** private helpers `density_to_quantile`, `w2_distance_inner` in `space.rs`

---

## Implementation Notes for Planner

### KEY RISK: Signed Weights Cannot Use `wasserstein_barycenter`

The `wasserstein_barycenter` function validates `weights[i] >= 0` at **density_fda.rs:450**:
```rust
if w.iter().any(|&wi| wi < 0.0 || !wi.is_finite()) {
    return Err(FdarError::InvalidParameter { ... });
}
```

Global Fréchet regression weights `sᵢ(x) = 1 + (Xᵢ−X̄)ᵀΣ̂⁻¹(x−X̄)` **can be negative**. The R reference (`GloWassReg.R`) computes `gx = colMeans(qin * s)` — a signed quantile average — not a call to the barycenter. The Rust implementation must implement a private `signed_quantile_average` helper in `frechet/space.rs` that:

1. Computes Q̄(t) = (1/n) Σᵢ sᵢ(x) Qᵢ(t) directly (signed weighted sum of per-density quantile functions)
2. Applies `q_bar.sort_by(|a,b| a.partial_cmp(b).unwrap_or(Ordering::Equal))` for monotone projection (no-osqp alternative)
3. Guards on `q_range < 1e-15` → `ComputationFailed`
4. Inverts to density using the `wasserstein_barycenter` back-map pattern (lines 511–535 of density_fda.rs)

The `MetricSpace::weighted_frechet_mean` trait method keeps non-negative semantics; regression bypasses it and calls `signed_quantile_average` directly.

### Private Helpers from `density_fda.rs` Needed by `frechet/space.rs`

`quantile_density_from_q` and `dedup_adjacent` are private to `density_fda.rs`. At implementation time, either:
- Add `pub(crate)` to both helpers in `density_fda.rs` (preferred — avoids duplication), OR
- Re-implement a minimal version of the finite-difference density inversion in `frechet/space.rs`

### Assumption A5 (linalg.rs visibility)

`cholesky_factor`, `cholesky_forward_back`, and `cholesky_solve` are declared `pub(crate)` in `linalg.rs` (verified lines 85, 113, 131). Since `frechet/` will be a submodule of `fdars-core`, they are accessible without any visibility change.

### `statrs` for χ²(k-1) p-value in anova.rs

The asymptotic p-value requires `ChiSquared::new(df)` from `statrs`. Verify `statrs` is in `Cargo.toml` (it is listed in CLAUDE.md dependencies). Import: `use statrs::distribution::{ChiSquared, ContinuousCDF};`.

---

## No Analog Found

All files have close analogs. No new external patterns are required.

---

## Metadata

**Analog search scope:** `fdars-core/src/` (fts/, density_fda.rs, linalg.rs, inference/, elastic_explain.rs, elastic_changepoint.rs, explain_generic.rs, helpers.rs, lib.rs)
**Files read:** 9
**Pattern extraction date:** 2026-08-22
