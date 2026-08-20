# Phase 32: Flexible Mixed-Effects Regression - Pattern Map

**Mapped:** 2026-08-20
**Files analyzed:** 6 new symbols / 2 modified files
**Analogs found:** 6 / 6

## File Classification

| New/Modified Symbol | Role | Data Flow | Closest Analog | Match Quality |
|---------------------|------|-----------|----------------|---------------|
| `famm.rs::DenseFlmmResult` | result struct | request-response | `famm.rs::FmmResult` | exact |
| `famm.rs::dense_flmm` | service fn | CRUD | `famm.rs::fmm` | exact |
| `famm.rs::MultiFlmmResult` + `multi_famm` | result + service fn | CRUD | `famm.rs::FmmResult` + `fmm` | exact |
| `famm.rs::fast_fmm` + `FastFmmResult` | result + service fn | batch | `famm.rs::fmm` | exact |
| `fof_regression.rs::FofReResult` + `fof_re_regression` | result + service fn | request-response | `fof_regression.rs::FofResult` + `fof_regression` | exact |
| Config structs (`DenseFlmmConfig`, `MultiFammConfig`, `FastFmmConfig`, `FofReConfig`) | config | - | `pace_fpca.rs::PaceFpcaConfig` | exact |

---

## Pattern Assignments

### `famm.rs` — new result structs (`DenseFlmmResult`, `MultiFammResult`, `FastFmmResult`)

**Analog:** `famm.rs::FmmResult` (lines 25–50)

**Derives + attributes pattern** (lines 25–27):
```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FmmResult {
```

**Rule:** All new result structs MUST use `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]` + `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`. Every public field gets a `///` doc comment.

**Field shape to extend** — `FmmResult` already carries:
- `mean_function: Vec<f64>`, `beta_functions: FdMatrix`, `random_effects: FdMatrix`
- `fitted: FdMatrix`, `residuals: FdMatrix`
- `sigma2_eps: f64`, `sigma2_u: Vec<f64>` (per-component variance)
- `random_variance: Vec<f64>`, `ncomp: usize`, `n_subjects: usize`, `eigenvalues: Vec<f64>`

**New structs extend this shape**, adding fields as needed:

- `DenseFlmmResult`: same as `FmmResult` but add `random_slope_variance: Vec<f64>` (random-slope variances per component, in addition to `sigma2_u` for random intercepts), `n_iter: usize` (REML iterations to convergence), `converged: bool`.
- `MultiFammResult`: wraps `Vec<DenseFlmmResult>` as `components: Vec<DenseFlmmResult>` (one per response dimension), plus stacked `fitted: FdMatrix` and `residuals: FdMatrix`.
- `FastFmmResult`: per-gridpoint fixed-effect estimates `beta_matrix: FdMatrix` (n_gridpoints × p), t-statistics `t_stats: FdMatrix`, p-values `p_values: FdMatrix`, `sigma2_eps: Vec<f64>` (per gridpoint), and `n_grid: usize`.

---

### `famm.rs` — new public functions (`dense_flmm`, `multi_famm`, `fast_fmm`)

**Analog:** `famm.rs::fmm` (lines 87–182) and `famm.rs::fmm_test_fixed` (lines 797–847)

**`#[must_use]` annotation** (line 87):
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<FmmResult, FdarError> {
```

**Entry validation pattern** (lines 94–115):
```rust
let n_total = data.nrows();
let m = data.ncols();
if n_total == 0 || m == 0 {
    return Err(FdarError::InvalidDimension {
        parameter: "data",
        expected: "non-empty matrix".to_string(),
        actual: format!("{n_total} x {m}"),
    });
}
if subject_ids.len() != n_total {
    return Err(FdarError::InvalidDimension {
        parameter: "subject_ids",
        expected: format!("length {n_total}"),
        actual: format!("length {}", subject_ids.len()),
    });
}
if ncomp == 0 {
    return Err(FdarError::InvalidParameter {
        parameter: "ncomp",
        message: "must be >= 1".to_string(),
    });
}
```

**Rule:** Validate at function entry in this exact order: dimensions first (`InvalidDimension`), then parameter ranges (`InvalidParameter`). Never panic. Propagate FPCA errors with `?`.

**FPCA invocation pattern** (lines 121–123):
```rust
let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
let fpca = fdata_to_pc_1d(data, ncomp, &argvals)?;
let k = fpca.scores.ncols(); // actual number of components
```

**Rule:** New fns must build `argvals` the same way when no grid is provided; use `fdata_to_pc_1d` for FPC decomposition.

**Parallel component loop pattern** (lines 228–235):
```rust
let per_comp: Vec<ScalarMixedResult> = iter_maybe_parallel!(0..k)
    .map(|comp| {
        // ...
    })
    .collect();
```

**Rule:** Per-component loops MUST use `iter_maybe_parallel!` (from `crate::parallel`) for rayon-gated parallelism. `dense_flmm` and `fast_fmm` should follow the same pattern.

**Per-thread RNG seeding** (lines 874–875 in `fmm_test_fixed`):
```rust
let mut rng = StdRng::seed_from_u64(seed);
```
For any bootstrap or permutation paths in new fns, seed as `StdRng::seed_from_u64(seed + k as u64)` when inside per-component parallel closures.

**REML update pattern** — already present in private fn `reml_variance_update` (lines 395–431). New `dense_flmm` with random slopes should factor its own `reml_slope_variance_update` helper following the same signature convention `(residuals, ss, weights, sigma2_u, p) -> (f64, f64)`.

**Signature conventions for new fns:**
```rust
pub fn dense_flmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &DenseFlmmConfig,
) -> Result<DenseFlmmResult, FdarError>

pub fn multi_famm(
    data: &[FdMatrix],          // one FdMatrix per response dimension
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &MultiFammConfig,
) -> Result<MultiFammResult, FdarError>

pub fn fast_fmm(
    data: &FdMatrix,
    subject_ids: &[usize],
    covariates: Option<&FdMatrix>,
    config: &FastFmmConfig,
) -> Result<FastFmmResult, FdarError>
```

Deviation from `fmm`: new fns take a `&Config` struct rather than bare `ncomp: usize` scalar. This is consistent with other complex fns (`pace_fpca`, `gmm_cluster_with_config`, etc.).

---

### Config structs (`DenseFlmmConfig`, `MultiFammConfig`, `FastFmmConfig`, `FofReConfig`)

**Analog:** `pace_fpca.rs::PaceFpcaConfig` (lines 49–83)

**Struct pattern** (lines 49–70):
```rust
/// No `#[non_exhaustive]` — config structs are not non_exhaustive so callers can
/// construct them with struct literals.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Debug, Clone, PartialEq)]
pub struct PaceFpcaConfig {
    /// Number of FPCA components to extract.
    pub ncomp: usize,
    // ... all public fields with doc comments
}

impl Default for PaceFpcaConfig {
    fn default() -> Self {
        Self {
            ncomp: 3,
            bandwidth: 0.1,
            // ...
        }
    }
}
```

**Rule:** Config structs:
- Do NOT use `#[non_exhaustive]` (callers must be able to construct with struct literal)
- DO use `#[derive(Debug, Clone, PartialEq)]`
- DO use `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`
- MUST implement `Default` with sensible defaults documented in the `Default` impl

**Suggested field layouts:**
```rust
pub struct DenseFlmmConfig {
    /// Number of FPC components (default: 3)
    pub ncomp: usize,
    /// Maximum REML EM iterations (default: 50)
    pub max_iter: usize,
    /// Convergence tolerance for variance components (default: 1e-10 relative)
    pub tol: f64,
    /// Whether to include random slopes (in addition to random intercepts) (default: false)
    pub random_slopes: bool,
}

pub struct MultiFammConfig {
    /// Number of FPC components per response dimension (default: 3)
    pub ncomp: usize,
    /// Max REML iterations per component model (default: 50)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-10)
    pub tol: f64,
}

pub struct FastFmmConfig {
    /// Inference mode: massively-univariate per-grid-point (default)
    pub ncomp: usize,
    /// Max iterations for each per-point mixed model (default: 30)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-8)
    pub tol: f64,
    /// Whether to produce pointwise t-statistics and p-values (default: true)
    pub compute_inference: bool,
}

pub struct FofReConfig {
    /// Number of predictor FPC components (default: 3)
    pub ncomp_x: usize,
    /// Number of response FPC components (default: 3)
    pub ncomp_y: usize,
    /// Max REML iterations (default: 50)
    pub max_iter: usize,
    /// Convergence tolerance (default: 1e-10)
    pub tol: f64,
}
```

---

### `fof_regression.rs` — new `FofReResult` + `fof_re_regression`

**Analog:** `fof_regression.rs::FofResult` (lines 27–53) and `fof_regression` (lines 113–293)

**Result struct pattern** (lines 27–53):
```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FofResult {
    pub intercept: Vec<f64>,
    pub beta_surface: FdMatrix,
    pub fitted: FdMatrix,
    pub residuals: FdMatrix,
    pub r_squared_t: Vec<f64>,
    pub r_squared: f64,
    pub ncomp_x: usize,
    pub ncomp_y: usize,
    pub fpca_x: FpcaResult,
    pub fpca_y: FpcaResult,
    pub coef_matrix: FdMatrix,
}
```

`FofReResult` extends `FofResult`'s shape with mixed-effects fields:
```rust
pub struct FofReResult {
    // all fields from FofResult, plus:
    pub random_effects: FdMatrix,     // n_subjects × m_y
    pub sigma2_u: Vec<f64>,           // per response FPC component
    pub sigma2_eps: f64,
    pub n_subjects: usize,
}
```

**Rule:** `FofReResult` MUST also carry `fpca_x: FpcaResult` and `fpca_y: FpcaResult` so that `predict_fof_re` can reuse the same projection path as `predict_fof`.

**Public fn entry validation pattern** (lines 124–165):
```rust
if n_x != n_y {
    return Err(FdarError::InvalidDimension {
        parameter: "y_data",
        expected: format!("{n_x} rows (matching x_data)"),
        actual: format!("{n_y} rows"),
    });
}
// ... then argvals length checks, then ncomp > 0 checks
```

**Signature for new fn:**
```rust
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fof_re_regression(
    x_data: &FdMatrix,
    y_data: &FdMatrix,
    subject_ids: &[usize],
    x_argvals: &[f64],
    y_argvals: &[f64],
    config: &FofReConfig,
) -> Result<FofReResult, FdarError>
```

Deviation: `subject_ids: &[usize]` inserted between `y_data` and `x_argvals`; config struct replaces bare `ncomp_x`/`ncomp_y` scalars.

---

## Shared Patterns

### Error Handling
**Source:** `fdars-core/src/error.rs` + usage throughout `famm.rs`
**Apply to:** All new public functions

Variants to use:
- Dimension mismatch: `FdarError::InvalidDimension { parameter: &'static str, expected: String, actual: String }`
- Bad parameter: `FdarError::InvalidParameter { parameter: &'static str, message: String }`
- Numerical failure: `FdarError::ComputationFailed { operation: &'static str, detail: String }`

Never return `Option`; bridge internal `Option` → `Result` via `.ok_or_else(|| FdarError::ComputationFailed { ... })`.

### Linalg Solves
**Source:** `famm.rs` (lines 13–18, 537–541)
**Apply to:** All variance-component and GLS fixed-effect solves

```rust
use crate::linalg::{
    cholesky_factor as linalg_cholesky_factor,
    cholesky_forward_back as linalg_cholesky_forward_back,
};

fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    let l = linalg_cholesky_factor(a, p).ok()?;
    Some(linalg_cholesky_forward_back(&l, b, p))
}
```

No new crate dependency; all mixed-model solves go through this local wrapper.

### Parallelism Gate
**Source:** `famm.rs` (lines 14, 228)
**Apply to:** All per-component loops in `dense_flmm`, `fast_fmm`, `multi_famm`

```rust
use crate::iter_maybe_parallel;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

// In loops:
let results: Vec<_> = iter_maybe_parallel!(0..k).map(|comp| { ... }).collect();
```

### `#[must_use]`
**Apply to:** All public computation functions (not config-only helpers).
Standard string: `"expensive computation whose result should not be discarded"` for fitting functions; `"prediction result should not be discarded"` for predict functions.

### Module wiring
**Source:** `fdars-core/src/lib.rs` (lines 86, 88, 233, 242)

Current exports to extend:
```rust
// line 233 — extend this:
pub use famm::{fmm, fmm_predict, fmm_test_fixed, FmmResult, FmmTestResult};
// add: dense_flmm, DenseFlmmResult, DenseFlmmConfig,
//      multi_famm, MultiFammResult, MultiFammConfig,
//      fast_fmm, FastFmmResult, FastFmmConfig

// line 242 — extend this:
pub use fof_regression::{fof_cv, fof_regression, predict_fof, FofCvResult, FofResult};
// add: fof_re_regression, predict_fof_re, FofReResult, FofReConfig
```

No new `pub mod` lines needed — both modules are already declared.

---

## Inline Test Patterns

**Source:** `famm.rs` lines 919–1394, `fof_regression.rs` lines 519–686

**Test module header:**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;
    use std::f64::consts::PI;
```

**Synthetic data helper convention** (`famm.rs` lines 926–962):
```rust
fn generate_fmm_data(
    n_subjects: usize,
    n_visits: usize,
    m: usize,
) -> (FdMatrix, Vec<usize>, FdMatrix, Vec<f64>) {
    // Returns (data, subject_ids, covariates, t_grid)
    // Uses deterministic arithmetic (no RNG in data generator):
    // col_major[obs + j * n_total] = ...
    let data = FdMatrix::from_column_major(col_major, n_total, m).unwrap();
    let covariates = FdMatrix::from_column_major(cov_data, n_total, 1).unwrap();
    (data, subject_ids, covariates, t)
}
```

**Required test categories (copy from `famm.rs` tests):**
1. Basic dimensions check (`assert_eq!(result.field.len(), m)` etc.)
2. Invariant check (fitted + residuals == data within 1e-8)
3. Positivity of variance fields (`>= 0.0`)
4. Invalid-input error path checks — match on `FdarError` variant and `parameter` field:
   ```rust
   let err = dense_flmm(&data, &ids, None, &config).unwrap_err();
   match err {
       FdarError::InvalidParameter { parameter, .. } => assert_eq!(parameter, "ncomp"),
       other => panic!("Expected InvalidParameter, got {:?}", other),
   }
   ```
5. No-covariate path (pass `None` for `covariates`)
6. Single-visit-per-subject edge case

**`fof_regression.rs` test helper** (lines 527–559):
```rust
fn make_fof_data(n: usize, mx: usize, my: usize, seed: u64)
    -> (FdMatrix, FdMatrix, Vec<f64>, Vec<f64>)
```
New `fof_re_regression` tests should add `subject_ids` construction to this helper rather than creating a new one.

---

## No Analog Found

All new components have strong analogs in the existing codebase. No RESEARCH.md fallback needed.

---

## Metadata

**Analog search scope:** `fdars-core/src/famm.rs`, `fdars-core/src/fof_regression.rs`, `fdars-core/src/lib.rs`, `fdars-core/src/pace_fpca.rs`
**Files scanned:** 4
**Pattern extraction date:** 2026-08-20
