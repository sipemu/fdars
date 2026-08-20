# Phase 31: Additive Functional Regression & Variable Selection — Pattern Map

**Mapped:** 2026-08-20
**Files analyzed:** 3 (1 new, 2 modified)
**Analogs found:** 3 / 3

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/src/scalar_on_function/additive.rs` | service (estimators + config/result types) | CRUD / batch transform | `scalar_on_function/nonparametric.rs` + `scalar_on_function/fregre_lm.rs` | role-match (same module, same I/O contract) |
| `fdars-core/src/scalar_on_function/mod.rs` | config (barrel re-export) | — | `scalar_on_function/mod.rs` itself (additive entry) | exact |
| `fdars-core/src/lib.rs` | config (crate-root re-export) | — | existing `pub use scalar_on_function::{...}` block (lines 251–261) | exact |

---

## Pattern Assignments

### `fdars-core/src/scalar_on_function/additive.rs` (new file)

Everything in this file is new. Map each logical section to its closest in-repo analog.

---

#### 1. File-level imports

**Analog:** `fdars-core/src/scalar_on_function/fregre_lm.rs` lines 1–10

```rust
use super::{
    build_design_matrix, compute_r_squared, validate_fregre_inputs,
    FregreLmResult,                        // pattern only — don't import, define own results
};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
```

**For `additive.rs` specifically:**
```rust
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::regression::{fdata_to_pc_1d, FpcaResult};
use crate::smoothing::{nadaraya_watson, optim_bandwidth, CvCriterion};
use crate::helpers::simpsons_weights;
use super::nonparametric::{compute_pairwise_distances, gaussian_kernel};
```

`compute_pairwise_distances` and `gaussian_kernel` are `pub(super)` in `nonparametric.rs` (lines 10–11 and 15–16), which makes them visible to all sibling files inside `scalar_on_function/`, including the new `additive.rs`. No visibility change is required.

**Deviation:** Do NOT pull `use rand::prelude::*` at module top level — it triggers clippy's unused-import warning on non-permutation code paths. Place it inside the permutation helper function body or inside `#[cfg(test)]` blocks only (see Pitfall 5 in RESEARCH.md).

---

#### 2. Config structs (`FamConfig`, `GkamConfig`, `GsamConfig`, `VarSelectConfig`, `PermTestConfig`, `HistoryIndexConfig`)

**Analog:** `fdars-core/src/gmm/cluster.rs` lines 49–86 (`GmmClusterConfig`)

```rust
// Source: fdars-core/src/gmm/cluster.rs lines 49-70
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GmmClusterConfig {
    /// Number of basis functions for projection (default: 5).
    pub nbasis: usize,
    // ... fields ...
    /// Base random seed (default: 42).
    pub seed: u64,
}

impl Default for GmmClusterConfig {
    fn default() -> Self {
        Self {
            nbasis: 5,
            // ... defaults ...
            seed: 42,
        }
    }
}
```

**Pattern to copy for every config struct:**
1. `#[derive(Debug, Clone, PartialEq)]` — all three derives, always
2. `#[non_exhaustive]` — present on `GmmClusterConfig`; required for forward-compatible public structs
3. `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` — serde feature gate, matches `FpcaResult` (`regression.rs` line 23) and is required per CONTEXT.md decision
4. `impl Default` block with explicit field-by-field initialization (no `..` shorthand in Default impls)
5. Each field has a `/// doc comment (default: X)` naming the default value

**Note:** `GmmClusterConfig` itself does NOT have the serde `cfg_attr` (cluster.rs has no serde), but `FpcaResult` and the RESEARCH.md API surface both require it. Follow `FpcaResult`'s pattern (regression.rs line 23) for the serde gate.

**Enum configs** (`VarSelectPenalty`, `PermTestStatistic`):

```rust
// Source: fdars-core/src/scalar_on_function/mod.rs lines 341-353 (GlmFamily — closest enum)
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum GlmFamily {
    Binomial,
    Poisson,
    Gamma,
    Gaussian,
}
```

Copy this verbatim for `VarSelectPenalty` and `PermTestStatistic`: `Debug, Clone, Copy, PartialEq`, `#[non_exhaustive]`, `#[cfg_attr(feature = "serde", ...)]`.

---

#### 3. Result structs (`FamResult`, `GkamResult`, `GsamResult`, `VarSelectResult`, `PermTestResult`, `HistoryIndexResult`)

**Analog:** `fdars-core/src/scalar_on_function/mod.rs` lines 55–137 (`FregreLmResult`, `FregreRobustResult`)

```rust
// Source: fdars-core/src/scalar_on_function/mod.rs lines 55-91
/// Result of functional linear regression.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FregreLmResult {
    /// Intercept α
    pub intercept: f64,
    /// Functional coefficient β(t), evaluated on the original grid (length m)
    pub beta_t: Vec<f64>,
    // ...
    /// R² statistic
    pub r_squared: f64,
    /// Number of FPC components used
    pub ncomp: usize,
    /// FPCA result (for projecting new data)
    pub fpca: FpcaResult,
}
```

**Pattern to copy:**
1. `#[derive(Debug, Clone, PartialEq)]` — always all three
2. `#[non_exhaustive]` — required on all public result structs
3. `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]` — required per CONTEXT.md decision (for all result structs that do not embed non-serializable types). `FpcaResult` itself is serde-gated (regression.rs line 23), so `FamResult`/`GsamResult` embedding it can use the same gate safely.
4. Every public field has a `/// doc comment` including the length annotation (e.g., `/// length n`).
5. `converged: bool` and `iterations: usize` pattern comes from `FregreRobustResult` (mod.rs lines 130–133) — copy for `GkamResult` and `VarSelectResult`.
6. `fpcas: Vec<FpcaResult>` (for multi-predictor `VarSelectResult`) mirrors `MultiFregreLmResult.fpcas` (mod.rs lines 249–250).

**`PermTestResult` deviation:** RESEARCH.md specifies no serde gate because `null_statistics: Vec<f64>` can be large and serde support is optional. Follow the existing pattern for result types without the gate when the field set is purely numeric (`FregreNpResult` has no serde gate either — mod.rs lines 93–109). Apply the gate only if the planner decides to include it.

---

#### 4. Public estimator functions (`fam`, `fregre_gkam`, `fregre_gsam`, `variable_selection`, `permutation_test_fam`, `history_index`)

**Analog:** `fdars-core/src/scalar_on_function/fregre_lm.rs` lines 62–80

```rust
// Source: fdars-core/src/scalar_on_function/fregre_lm.rs lines 62-80
#[must_use = "expensive computation whose result should not be discarded"]
pub fn fregre_lm(
    data: &FdMatrix,
    y: &[f64],
    scalar_covariates: Option<&FdMatrix>,
    ncomp: usize,
) -> Result<FregreLmResult, FdarError> {
    let (n, m) = data.shape();
    validate_fregre_inputs(n, m, y, scalar_covariates)?;
    // ...
}
```

**Pattern to copy:**
1. `#[must_use = "expensive computation whose result should not be discarded"]` — exact string from existing functions; copy verbatim on all six public functions
2. Signature order: `(data, y, argvals, scalar_covariates, config)` — argvals explicit (not derived internally) per RESEARCH.md API Surface
3. Return type: `Result<XxxResult, FdarError>` — no unwrap, no panic
4. First two lines of body: `let (n, m) = data.shape();` then input validation via `FdarError::InvalidDimension` checks on `n`, `m`, `y.len()`, `argvals.len()`, `scalar_covariates` rows
5. For multi-predictor functions (`fregre_gkam`, `variable_selection`): validate that all `predictors[k].nrows()` equal `y.len()` before any computation

**Validation error pattern** (from mod.rs lines 487–519):
```rust
// Source: fdars-core/src/scalar_on_function/mod.rs lines 500-507
if y.len() != n {
    return Err(FdarError::InvalidDimension {
        parameter: "y",
        expected: format!("{n}"),
        actual: format!("{}", y.len()),
    });
}
```

**`InvalidParameter` pattern** for config bounds (e.g., `window > argvals range`):
```rust
// Source: fdars-core/src/error.rs — InvalidParameter variant
return Err(FdarError::InvalidParameter {
    parameter: "config.window",
    message: format!(
        "window ({:.4}) exceeds argvals range ({:.4})",
        config.window, argvals_range
    ),
});
```

---

#### 5. Permutation test pattern

**Analog:** `fdars-core/src/famm.rs` lines 860–899

```rust
// Source: fdars-core/src/famm.rs lines 872-899
use rand::prelude::*;
let mut rng = StdRng::seed_from_u64(seed);
let mut n_ge = vec![0usize; p];

for _ in 0..n_perm {
    let mut perm_indices: Vec<usize> = (0..n_total).collect();
    perm_indices.shuffle(&mut rng);
    // ... refit model with permuted data ...
    if perm_stat >= observed_stat {
        n_ge += 1;
    }
}

let p_value = (n_ge + 1) as f64 / (n_perm + 1) as f64;
```

**Deviation for `permutation_test_fam`:** The RESEARCH.md specifies permuting `y` (not rows of the predictor matrix), whereas `famm.rs` permutes covariate rows. Use `y_perm.shuffle(&mut rng)` pattern on a cloned `y` Vec. Keep the same `StdRng::seed_from_u64(seed)` seeding — do NOT use `StdRng::seed_from_u64(seed + k)` per-iteration (the single shared RNG already advances deterministically). The `seed + k` pattern from parallel.rs is for thread-local seeding in rayon contexts; the famm.rs pattern with a single `rng.shuffle` is correct here.

Default `n_perm = 999` matches the INF-01 convention specified in CONTEXT.md.

---

#### 6. NW-on-distance kernel loop (GKAM inner loop)

**Analog:** `fdars-core/src/scalar_on_function/nonparametric.rs` lines 49–81 (`nw_loo_predict`)

```rust
// Source: fdars-core/src/scalar_on_function/nonparametric.rs lines 62-80
let mut num = 0.0;
let mut den = 0.0;
for j in 0..n {
    if i == j { continue; }
    let w = gaussian_kernel(func_dists[i * n + j], h_func);
    num += w * y[j];
    den += w;
}
if den > 1e-15 {
    num / den
} else {
    y[i]          // fallback to self when kernel sums to near-zero
}
```

**Pattern:** `if den > 1e-15` guard before division — copy exactly; do not use `f64::EPSILON` (existing code uses `1e-15`). Fallback to `y[i]` (self-value) when bandwidth is too small. Never return `NaN`.

---

#### 7. Inline test module

**Analog:** `fdars-core/src/scalar_on_function/nonparametric.rs` (bottom of file) and `famm.rs` lines 1045–1091

```rust
// Source: fdars-core/src/famm.rs lines 1045-1055
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn test_fmm_test_fixed_detects_effect() {
        // ... synthetic setup ...
        let result = fmm_test_fixed(&data, &subject_ids, &covariates, 3, 99, 42).unwrap();
        assert!(
            result.p_values[0] < 0.1,
            "expected p < 0.1, got {}",
            result.p_values[0]
        );
    }
}
```

**Pattern:**
1. `#[cfg(test)] mod tests { use super::*; use crate::test_helpers::uniform_grid; }` — standard header
2. Each `#[test]` function is a stand-alone function (no shared state)
3. Assertion messages include the actual value: `assert!(cond, "expected ..., got {}", val)`
4. Use `unwrap()` freely inside `#[cfg(test)]` — not a public API
5. RNG-seeded permutation tests use `seed: 42, n_perm: 99` (fast in CI)
6. Synthetic data from `uniform_grid(n)` — available from `crate::test_helpers`
7. Test names follow `snake_case` noun-verb pattern: `fam_synthetic_recovery`, `gkam_invalid_inputs`

---

### `fdars-core/src/scalar_on_function/mod.rs` (modification)

**Analog:** Existing `mod nonparametric; pub use nonparametric::{...};` block (lines 30 + 45–47)

```rust
// Source: fdars-core/src/scalar_on_function/mod.rs lines 30, 45-47
mod nonparametric;
pub use nonparametric::{
    fregre_np_from_distances, fregre_np_mixed, predict_fregre_np, predict_fregre_np_from_distances,
};
```

**Pattern to copy:** Add immediately after the last existing `mod` declaration:
```rust
mod additive;
pub use additive::{
    fam, fregre_gkam, fregre_gsam, history_index, permutation_test_fam, variable_selection,
    FamConfig, FamResult,
    GkamConfig, GkamResult,
    GsamConfig, GsamResult,
    HistoryIndexConfig, HistoryIndexResult,
    PermTestConfig, PermTestResult, PermTestStatistic,
    VarSelectConfig, VarSelectPenalty, VarSelectResult,
};
```

Alphabetic ordering within the `pub use` list is preferred (matches the existing style in the file).

---

### `fdars-core/src/lib.rs` (modification)

**Analog:** Existing `pub use scalar_on_function::{...}` block (lines 251–261)

```rust
// Source: fdars-core/src/lib.rs lines 251-261
pub use scalar_on_function::{
    bootstrap_ci_fregre_lm, bootstrap_ci_functional_logistic, fregre_basis_cv, fregre_cv,
    fregre_huber, fregre_l1, fregre_lm, fregre_lm_multi, fregre_lm_multi_cv, fregre_np_cv,
    fregre_np_from_distances, fregre_np_mixed, fregre_pls, functional_glm, functional_logistic,
    model_selection_ncomp, predict_fregre_lm, predict_fregre_lm_multi, predict_fregre_np,
    predict_fregre_np_from_distances, predict_fregre_pls, predict_fregre_robust,
    predict_functional_glm, predict_functional_logistic, BootstrapCiResult, FregreBasisCvResult,
    FregreCvResult, FregreLmResult, FregreNpCvResult, FregreNpResult, FregreRobustResult,
    FunctionalGlmResult, FunctionalLogisticResult, GlmFamily, ModelSelectionResult, MultiCvResult,
    MultiFregreLmResult, PlsRegressionResult, SelectionCriterion,
};
```

**Pattern:** Extend the existing `pub use scalar_on_function::{...}` block by appending the new symbols to its list. Do not create a second `pub use scalar_on_function::{...}` block — Rust would accept it but the style convention is a single block per source module.

---

## Shared Patterns

### Error handling
**Source:** `fdars-core/src/scalar_on_function/mod.rs` lines 487–519 (`validate_fregre_inputs`)
**Apply to:** All six public functions in `additive.rs`
- `FdarError::InvalidDimension` for shape mismatches (n, m, y.len(), argvals.len())
- `FdarError::InvalidParameter` for out-of-range config values (e.g., `window > argvals range`, `ncomp > min(n,m)`, `bandwidth <= 0` when explicitly provided)
- `FdarError::ComputationFailed` for numerical failures (FPCA SVD divergence, Cholesky singular matrix in `variable_selection` OLS step)
- Validate at function entry; no silent truncation

### `#[must_use]`
**Source:** `fdars-core/src/scalar_on_function/fregre_lm.rs` line 62
**Apply to:** All six public functions
Exact string: `#[must_use = "expensive computation whose result should not be discarded"]`

### Serde feature gate
**Source:** `fdars-core/src/regression.rs` line 23
**Apply to:** All config and result structs in `additive.rs`
```rust
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
```
Place after `#[derive(...)]`, before `#[non_exhaustive]` on result structs; after `#[non_exhaustive]` on enums (match `GlmFamily` ordering in mod.rs lines 341–343).

### NW denominator guard
**Source:** `fdars-core/src/scalar_on_function/nonparametric.rs` line 76
**Apply to:** Every Nadaraya-Watson evaluation site in `additive.rs`
```rust
if den > 1e-15 { num / den } else { fallback }
```

### R² computation
**Source:** `fdars-core/src/scalar_on_function/mod.rs` lines 605–622 (`compute_r_squared`)
**Apply to:** All result structs carrying `r_squared`
Call `super::compute_r_squared(y, &residuals, p_total)` (returns `(r_squared, r_squared_adj)`). For additive results that only need `r_squared` (not `r_squared_adj`), take `.0` of the tuple.

### Permutation seeding
**Source:** `fdars-core/src/famm.rs` lines 874–895
**Apply to:** `permutation_test_fam`
```rust
use rand::prelude::*;
let mut rng = StdRng::seed_from_u64(perm_config.seed);
// then rng.shuffle per iteration — NOT seed + k (that's for rayon threads)
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| Group-penalized coordinate descent in `variable_selection` | algorithm (inner) | batch transform | No group-lasso / group-MCP / group-SCAD solver exists in fdars-core. Closest: `cholesky_solve` for the OLS sub-step (linalg.rs). The group-threshold update loop is new logic. Follow RESEARCH.md Pattern 4 exactly; no in-repo structural analog. |
| History-index lag extraction with linear interpolation | algorithm (inner) | transform | No column-index interpolation exists in fdars-core matrix layer. Implement: `let j = argvals.partition_point(|&v| v < t_minus_lag).min(m-1)` for nearest-lower-bound, with `min(m-1)` clamp. Document interpolation method in rustdoc. |

---

## Metadata

**Analog search scope:** `fdars-core/src/scalar_on_function/`, `fdars-core/src/famm.rs`, `fdars-core/src/gmm/cluster.rs`, `fdars-core/src/regression.rs`, `fdars-core/src/lib.rs`
**Files scanned:** 7
**Pattern extraction date:** 2026-08-20
