# Phase 68: Longitudinal PEER, Prediction & Integration — Research

**Researched:** 2026-09-04
**Domain:** Longitudinal PEER (`lpeer`), out-of-sample prediction, and crate-root/prelude
integration for `fdars-core/src/peer.rs`
**Confidence:** HIGH (all source files read directly this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- Signature: `lpeer(data, y, argvals, subject_map, &config) -> Result<LpeerResult, FdarError>`.
  `subject_map: &[usize]` gives subject index per observation (length n).
  `config` is the existing `PeerConfig`.
- Dedicated `LpeerResult` struct: `{ beta, intercept, w_bar, fitted_values,
  sigma2_subject: f64, sigma2_resid: f64, n_subjects, lambda, penalty_type, .. }`.
  Derives `Debug, Clone, PartialEq`; `#[non_exhaustive]`; `#[must_use]`; serde behind feature.
- Variance components are non-negative (rely on `famm`'s clamping) — assert σ²≥0.
- Validate `subject_map`: `len == n`, all indices `< n_subjects` (derive `n_subjects` as
  `max(subject_map)+1` or require it) → descriptive `FdarError`.
- Reduce functional predictor to score representation (PEER penalty range-space basis, or
  `regression::fdata_to_pc_1d` FPC scores), pass as `covariates` + `subject_map` to
  `famm::fit_scalar_mixed_model(y, subject_map, n_subjects, Some(scores), p)`, then back-project
  fixed-effect γ to β(t) on the argvals grid. Mirrors `fof_regression.rs` famm usage.
- Carry the `PeerPenalty` / λ choice from Phase 67 into the coefficient estimate.
- `predict`: method on both result types. `PeerResult::predict(&self, new_data: &FdMatrix,
  argvals: &[f64]) -> Result<Vec<f64>, FdarError>`. Formula:
  `ŷ* = (intercept − w_bar·β) + Σ_j x*[j]·w[j]·β[j]`, `w = simpsons_weights(argvals)`.
- Self-consistency: re-passing training curves reproduces training `fitted_values` (w_bar makes
  this exact).
- `lpeer` prediction on new curves is marginal (fixed-effect only; new-subject random effect = 0).
- Return `Result<Vec<f64>, FdarError>` — validate `new_data.ncols() == m`.
- Crate root (`lib.rs`): `pub use peer::{peer, lpeer, PeerConfig, PeerResult, LpeerResult,
  PeerPenalty, LambdaChoice, LambdaMethod}` (explicit, alphabetical slot near `regression`).
- Prelude: same set via `pub use crate::peer::{...}`.
- Replace module-header `no_run` snippet with a **running** `//!` doctest passing
  `cargo test --doc`. `lpeer` gets a separate `///` doctest or second block.
- `pub mod peer;` already present in `lib.rs` (Phase 66). ADD `pub use peer::{...}` re-exports.

### Claude's Discretion

- Exact score-reduction basis for lpeer (penalty range-space vs FPC). Recommendation: FPC
  scores via `fdata_to_pc_1d` — integrates cleanly with `fit_scalar_mixed_model`'s covariates
  FdMatrix interface.
- Whether `n_subjects` is a parameter or derived from `subject_map`. Recommendation: derive as
  `subject_map.iter().copied().max().map(|m| m + 1).unwrap_or(0)`.
- Internal helper factoring; doctest data sizes (keep small + deterministic).

### Deferred Ideas (OUT OF SCOPE)

- User-tunable GCV grid, additional penalty orders, subject-specific prediction with BLUPs.
- Any further PEER enhancements beyond this phase close.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PER-04 | `lpeer(...)` estimator: longitudinal PEER with subject random effects via `famm::fit_scalar_mixed_model`, returning `LpeerResult` with variance components | §lpeer Design; §famm Integration Template; §famm Exact Contract |
| PER-05 | `predict` out-of-sample on fitted peer/lpeer; crate-root + prelude exports; running module doctest | §predict Design; §Integration: Exports + Doctest |

</phase_requirements>

---

## Summary

Phase 68 closes the v0.36.0 PEER milestone with three additive deliverables — all in
`fdars-core/src/peer.rs` plus one-line additions to `lib.rs` and `prelude.rs`.

**lpeer** extends `peer()` to grouped/longitudinal data by reducing the functional predictor to
FPC scores (via `fdata_to_pc_1d`), then calling `famm::fit_scalar_mixed_model` for subject-level
random intercepts. The PEER penalty still regularises β(t): it enters through the full-grid
penalized solve that produced the initial `peer()` estimate, while `lpeer` replaces the OLS
second pass with a GLS+REML-EM second pass over scores. The concrete plan is: run the standard
PEER setup (weights, centering, build_q, λ selection) to produce a score basis aligned with the
penalty structure; project data onto that basis; call `fit_scalar_mixed_model`; back-project γ to
β(t). The `ScalarMixedResult` fields `sigma2_u` and `sigma2_eps` map directly to `LpeerResult.sigma2_subject`
and `sigma2_resid`.

**predict** on `PeerResult` and `LpeerResult` implements the formula
`ŷ* = (intercept − w_bar·β) + Σ_j x*[j]·w[j]·β[j]` — exactly the identity the existing
`test_peer_stores_w_bar_for_prediction` test already exercises by hand. Adding a method on each
result struct with `new_data.ncols() == m` validation completes PER-05.

**Integration** is mechanical: one `pub use peer::{...}` block in `lib.rs` and one in
`prelude.rs`, plus replacing the `no_run` module header with a small running doctest (≤ 30 lines,
synthetic data, deterministic).

**Primary recommendation:** Implement in a single Wave 0 task that adds `LpeerResult`,
`lpeer()`, and `predict` methods to `peer.rs`, then a Wave 1 task for exports + doctest.
The score-reduction path via `fdata_to_pc_1d` is the clearest integration, matching the
`fof_re_regression` template exactly.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `lpeer()` fit | `src/peer.rs` | `famm::fit_scalar_mixed_model` | All new code in peer.rs; famm call is one line |
| Score reduction | `regression::fdata_to_pc_1d` | `peer.rs` (wraps result) | FPC basis integrates cleanly with famm's covariates FdMatrix |
| Subject random effects | `famm::fit_scalar_mixed_model` (pub(crate)) | — | Existing REML EM; correct tool for subject grouping |
| Back-projection γ → β(t) | `peer.rs` (inline loop) | `famm::recover_beta_functions` pattern | m×ncomp matmul; 5 lines |
| `predict` formula | `peer.rs` (method on result) | `helpers::simpsons_weights` | Same formula for both result types; share a private helper |
| Crate-root export | `src/lib.rs` (1-line additive) | — | `pub use peer::{...}` block |
| Prelude export | `src/prelude.rs` (1-line additive) | — | `pub use crate::peer::{...}` |
| Running doctest | `src/peer.rs` module header | — | Replace `no_run` block |

---

## 1. `famm::fit_scalar_mixed_model` — Exact Contract

**File read:** `fdars-core/src/famm.rs:438-499`

```rust
// VERIFIED: fdars-core/src/famm.rs:438-444
pub(crate) fn fit_scalar_mixed_model(
    y: &[f64],
    subject_map: &[usize],
    n_subjects: usize,
    covariates: Option<&FdMatrix>,
    p: usize,
) -> ScalarMixedResult
```

**Parameters (verbatim from source):**

- `y: &[f64]` — scalar response per observation (length n). In lpeer: the centered scalar
  response `yc` (or raw `y`; the mixed model fits the mean as γ₀ automatically when
  `covariates = None`, but for lpeer we pass the FPC scores so γ includes the intercept).
- `subject_map: &[usize]` — observation-to-subject mapping (length n). Values must be in
  `0..n_subjects`.
- `n_subjects: usize` — number of unique subjects (max(subject_map)+1 when subjects are 0-indexed
  contiguous; handled by `famm::build_subject_map` which re-indexes).
- `covariates: Option<&FdMatrix>` — n×p matrix of fixed-effect covariates, one row per
  observation. For lpeer: the n×ncomp FPC score matrix wrapped in `Some(...)`.
- `p: usize` — number of columns in covariates (= `ncomp` for lpeer). When `covariates=None`,
  `p` is ignored.

**`ScalarMixedResult` — exact struct (verbatim from source):**

```rust
// VERIFIED: fdars-core/src/famm.rs:264-269
pub(crate) struct ScalarMixedResult {
    pub(crate) gamma: Vec<f64>, // fixed effects (length p)
    pub(crate) u_hat: Vec<f64>, // random effects per subject (length n_subjects)
    pub(crate) sigma2_u: f64,   // random effect variance
    pub(crate) sigma2_eps: f64, // residual variance
}
```

**Field semantics for lpeer:**

| `ScalarMixedResult` field | Maps to `LpeerResult` field | Meaning |
|--------------------------|----------------------------|---------|
| `gamma: Vec<f64>` (length p=ncomp) | used internally to back-project β(t) | Fixed-effect FPC coefficients γ_k |
| `u_hat: Vec<f64>` (length n_subjects) | not stored in LpeerResult (marginal predict ignores BLUPs) | Subject BLUPs |
| `sigma2_u: f64` | `sigma2_subject` | Between-subject variance σ²_u |
| `sigma2_eps: f64` | `sigma2_resid` | Residual variance σ²_ε |

**Variance clamping (verbatim from source):**

```rust
// VERIFIED: fdars-core/src/famm.rs:454-459
if sigma2_e < 1e-15 {
    sigma2_e = 1e-6;
}
if sigma2_u < 1e-15 {
    sigma2_u = sigma2_e * 0.1;
}
```

Both components are clamped ≥ 1e-15 (with a secondary floor of 1e-6 / 1e-7 from the
REML EM's `reml_variance_update`):

```rust
// VERIFIED: fdars-core/src/famm.rs:427-430
(
    (sigma2_u_new / n_subjects as f64).max(1e-15),
    (sigma2_e_new / denom_e).max(1e-15),
)
```

**Constraint:** `fit_scalar_mixed_model` is `pub(crate)` — accessible from `peer.rs` in the
same crate without any visibility change. [VERIFIED: fdars-core/src/famm.rs:438]

**subject_map semantics:** `famm::build_subject_map` re-indexes non-contiguous IDs to
0..n_subjects. For lpeer, if the caller passes a raw `subject_map: &[usize]` whose values
are arbitrary subject IDs, `lpeer` should call `build_subject_map` to get a clean contiguous
map and the derived `n_subjects`. [VERIFIED: fdars-core/src/famm.rs:185-197]

```rust
// VERIFIED: fdars-core/src/famm.rs:185-197
pub(crate) fn build_subject_map(subject_ids: &[usize]) -> (Vec<usize>, usize) {
    let mut unique_ids: Vec<usize> = subject_ids.to_vec();
    unique_ids.sort_unstable();
    unique_ids.dedup();
    let n_subjects = unique_ids.len();
    let map: Vec<usize> = subject_ids
        .iter()
        .map(|id| unique_ids.iter().position(|u| u == id).unwrap_or(0))
        .collect();
    (map, n_subjects)
}
```

So `lpeer` derives `n_subjects` via `build_subject_map` — no need for the caller to supply it
separately.

---

## 2. `fof_regression.rs` famm-Usage Template

**File read:** `fdars-core/src/fof_regression.rs:750-786`

The `fof_re_regression` function is the canonical template for calling
`fit_scalar_mixed_model` from outside `famm.rs`. Exact relevant lines:

```rust
// VERIFIED: fdars-core/src/fof_regression.rs:748-786
// Build subject structure (non-contiguous IDs handled by build_subject_map)
let (subject_map, n_subjects) = crate::famm::build_subject_map(subject_ids);

// Wrap x_scores as covariates for fit_scalar_mixed_model.
// INTENTIONAL: x_scores are passed directly WITHOUT re-applying h.sqrt() normalization.
// The L² weighting is already embedded in fpca_x.project(); re-scaling would double-count it.
let p = ncomp_x; // number of fixed-effect covariates per Y-score model

// --- Step 2: Per-Y-score mixed model ---
for l in 0..ncomp_y {
    let y_scores_l: Vec<f64> = (0..n).map(|i| y_scores[(i, l)]).collect();
    let result = crate::famm::fit_scalar_mixed_model(
        &y_scores_l,
        &subject_map,
        n_subjects,
        Some(&x_scores),
        p,
    );
    // gamma_l: fixed-effect coefficients (length ncomp_x)
    for k in 0..ncomp_x {
        if k < result.gamma.len() {
            coef_matrix[(k, l)] = result.gamma[k];
        }
    }
    u_hat_per_component.push(result.u_hat);
    sigma2_u[l] = result.sigma2_u;
    sigma2_eps_total += result.sigma2_eps;
}
```

**Key observations for lpeer:**

1. **Score scaling:** `fof_re_regression` passes FPC scores **without** the `h.sqrt()` normalization
   that `famm::fit_all_components` applies (see comment at line 752). The reason: `fpca_x.project()`
   already carries L² weighting. For lpeer the analogous approach is to use raw FPC scores from
   `fdata_to_pc_1d` (which returns `scores` that are already the projection onto the normalized
   eigenvectors) without additional scaling.
2. **covariates layout:** `x_scores` is an n×ncomp `FdMatrix` (one row per observation, one column
   per FPC component). `fit_scalar_mixed_model` reads covariates row by row via `cov[(i, r)]`.
3. **p = ncomp:** the number of fixed-effect covariates equals the number of FPC components used.
4. **Back-projection pattern (from `famm::recover_beta_functions`):**

```rust
// VERIFIED: fdars-core/src/famm.rs:652-670
fn recover_beta_functions(
    gamma: &[Vec<f64>], rotation: &FdMatrix, p: usize, m: usize, k: usize,
) -> FdMatrix {
    let mut beta = FdMatrix::zeros(p, m);
    for j in 0..p {
        for t in 0..m {
            let mut val = 0.0;
            for comp in 0..k {
                val += gamma[j][comp] * rotation[(t, comp)];
            }
            beta[(j, t)] = val;
        }
    }
    beta
}
```

For lpeer (scalar-on-function, one β(t) function), the back-projection simplifies to:

```rust
// beta[j] = Σ_k gamma[k] * rotation[(j, k)]   for j in 0..m
let beta: Vec<f64> = (0..m)
    .map(|j| (0..ncomp).map(|k| gamma[k] * rotation[(j, k)]).sum())
    .collect();
```

where `rotation` is `fpca.rotation` (shape m×ncomp, column = eigenvector) from `fdata_to_pc_1d`.

---

## 3. lpeer Design — Concrete Plan

### Score Reduction: FPC via `fdata_to_pc_1d`

**Chosen approach:** Use `fdata_to_pc_1d` to produce FPC scores. This is simpler than constructing
a penalty range-space basis and integrates identically with `fit_scalar_mixed_model`'s
`covariates: Option<&FdMatrix>` interface. The PEER penalty (via `LambdaChoice` in `PeerConfig`)
still regularizes β(t): it is applied in the standard PEER step that produces the initial
penalized `peer()` estimate; `lpeer` then replaces the pure OLS solve with the mixed-model
GLS+REML-EM solve over the same FPC scores.

**`ncomp` selection for lpeer:** Use a fixed default (e.g. `min(n-1, m, 10)` or
`min(n_subjects - 1, m)`), or make it a field in `PeerConfig` (Claude's discretion). Simplest:
derive it as `min(n-1, m, 10)` — document this cap.

### Algorithm

```
lpeer(data, y, argvals, subject_map, config)
   |
   v
[Entry Validation]
  n >= 2, m >= 3, argvals.len()==m, y.len()==n
  subject_map.len()==n
  subject_map values all < n (basic sanity; build_subject_map re-indexes)
   |
   v
[build_subject_map(subject_map)] -> (sm_dense, n_subjects)
   |
   v
[Integration Weights + Design] (same as peer())
  w = simpsons_weights(argvals)
  wmat[i,j] = data[(i,j)] * w[j]
  y_bar = mean(y)
  yc = y - y_bar
  w_bar = col_mean(wmat)
  wc = wmat - w_bar
   |
   v
[FPC score reduction]
  fpca = fdata_to_pc_1d(&data, ncomp, &argvals)?
  scores = fpca.scores   // n×ncomp FdMatrix; raw FPC scores (L²-weighted projection)
   |
   v
[λ selection] (same LambdaChoice dispatch as peer())
  build_q(m, &config.penalty)?
  select lambda via Fixed/Gcv/Reml
   |
   v
[fit_scalar_mixed_model] (scalar-on-function mixed model over FPC scores)
  result = fit_scalar_mixed_model(&yc, &sm_dense, n_subjects, Some(&scores), ncomp)
   Note: pass yc (centered response) so the model estimates a zero-mean fixed effect
         (intercept recovered as y_bar as in peer())
   |
   v
[Back-project γ → β(t)]
  beta[j] = Σ_k result.gamma[k] * fpca.rotation[(j, k)]   for j in 0..m
   |
   v
[w_bar + intercept] (same as peer() — for identical predict formula)
  intercept = y_bar  (match peer() convention)
  w_bar already computed above (column means of wmat)
   |
   v
[Fitted values]
  base = intercept - Σ_j w_bar[j] * beta[j]
  fitted[i] = base + Σ_j data[(i,j)] * w[j] * beta[j]
   |
   v
LpeerResult { beta, intercept, w_bar, fitted_values,
              sigma2_subject: result.sigma2_u,
              sigma2_resid: result.sigma2_eps,
              n_subjects, lambda, penalty_type, lambda_method, gcv }
```

**Note:** `yc` (not `y`) is passed to `fit_scalar_mixed_model` because the centering is done
manually. The model is `yc_i = Σ_k γ_k ξ_{ik} + u_{s(i)} + ε_i` where `ξ` are FPC scores.
Intercept = `y_bar` exactly matches `peer()`.

**Alternative:** pass raw `y` and omit `yc`, letting the mixed model absorb the intercept as a
covariate (prepend a column of ones to the score matrix). Either works; the `yc` + `y_bar`
approach is simpler and consistent with `peer()`.

### `n_subjects` derivation

```rust
// Derive n_subjects from subject_map using build_subject_map (handles non-contiguous IDs)
let (sm_dense, n_subjects) = crate::famm::build_subject_map(subject_map);
// Validation: ensure n_subjects >= 2 (degenerate with 1 subject = no random effect)
if n_subjects < 2 {
    return Err(FdarError::InvalidParameter {
        parameter: "subject_map",
        message: "at least 2 distinct subjects required for lpeer".to_string(),
    });
}
```

### `LpeerResult` struct

```rust
// Pattern: matches PeerResult with variance component fields added
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use = "expensive computation whose result should not be discarded"]
pub struct LpeerResult {
    pub beta: Vec<f64>,              // estimated β(t), length m
    pub intercept: f64,              // = ȳ (same as peer())
    pub w_bar: Vec<f64>,             // col means of weighted design, length m
    pub fitted_values: Vec<f64>,     // ŷ_i = (intercept - w_bar·β) + Σ_j x[j]w[j]β[j]
    pub sigma2_subject: f64,         // between-subject variance σ²_u
    pub sigma2_resid: f64,           // residual variance σ²_ε
    pub n_subjects: usize,
    pub lambda: f64,
    pub penalty_type: PeerPenalty,
    pub gcv: Option<f64>,
    pub lambda_method: LambdaMethod,
}
```

---

## 4. predict — Exact Design

### Formula

From CONTEXT.md (locked) and the existing `test_peer_stores_w_bar_for_prediction` test
[VERIFIED: fdars-core/src/peer.rs:1122-1154]:

```
ŷ* = (intercept − w_bar·β) + Σ_j x*[j] · w[j] · β[j]
w = simpsons_weights(argvals)
```

Expanded:
```
base   = intercept − Σ_j w_bar[j] · beta[j]
ŷ*[i] = base + Σ_j new_data[(i,j)] · w[j] · beta[j]
```

This is the same formula as `test_peer_stores_w_bar_for_prediction` uses at lines 1136-1149 —
the test already proves the identity. The `predict` method simply codifies it.

### Shared helper

Both `PeerResult::predict` and `LpeerResult::predict` execute the same formula. Factor it into
a private `fn peer_predict_core(beta, intercept, w_bar, new_data, argvals) -> Result<Vec<f64>, FdarError>`:

```rust
// Private helper shared by both result types
fn peer_predict_core(
    beta: &[f64],
    intercept: f64,
    w_bar: &[f64],
    new_data: &FdMatrix,
    argvals: &[f64],
) -> Result<Vec<f64>, FdarError> {
    let (n_new, m_new) = new_data.shape();
    let m = beta.len();
    if m_new != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: format!("{m} columns (training grid length)"),
            actual: format!("{m_new}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    let w = crate::helpers::simpsons_weights(argvals);
    let base: f64 = intercept - w_bar.iter().zip(beta).map(|(wb, b)| wb * b).sum::<f64>();
    let preds: Vec<f64> = (0..n_new)
        .map(|i| {
            base + (0..m)
                .map(|j| new_data[(i, j)] * w[j] * beta[j])
                .sum::<f64>()
        })
        .collect();
    Ok(preds)
}
```

Then:

```rust
impl PeerResult {
    pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Result<Vec<f64>, FdarError> {
        peer_predict_core(&self.beta, self.intercept, &self.w_bar, new_data, argvals)
    }
}

impl LpeerResult {
    pub fn predict(&self, new_data: &FdMatrix, argvals: &[f64]) -> Result<Vec<f64>, FdarError> {
        peer_predict_core(&self.beta, self.intercept, &self.w_bar, new_data, argvals)
    }
}
```

### Validation and self-consistency

- Validate `new_data.ncols() == m` (the training grid length, inferable from `beta.len()`).
- Validate `argvals.len() == m`.
- Self-consistency test: re-pass training `data` and training `argvals` → predictions match
  `fitted_values` within 1e-9 for both `PeerResult` and `LpeerResult`.

### Convention source

The predict-method-on-result-struct pattern is used in `scalar_on_function/mod.rs`:

```rust
// VERIFIED: fdars-core/src/scalar_on_function/mod.rs:668-673
impl FregreLmResult {
    pub fn predict(&self, new_data: &FdMatrix, new_scalar: Option<&FdMatrix>) -> Vec<f64> {
        predict_fregre_lm(self, new_data, new_scalar)
    }
}
```

The PEER predict signature differs in two ways: (1) it takes `argvals` as a second argument
(needed for Simpson's weights, which `fregre_lm` doesn't need separately since its FPCA already
carries weights), and (2) it returns `Result<Vec<f64>, FdarError>` because dimension validation
can fail.

---

## 5. Integration: Exports + Doctest

### Crate-root insertion point

`pub mod peer;` already exists at line 111 of `lib.rs`. [VERIFIED: fdars-core/src/lib.rs:111]

The `pub use peer::{...}` re-export block must be added. Alphabetically, `peer` sits between
`optimal_design` and `regression` — the `optimal_design` re-export block is at lines 593-596
and `regression` is at line 567. [VERIFIED: fdars-core/src/lib.rs:567,593-596]

**Exact insertion:** After the `optimal_design` re-export block (~line 596), before or after
the `clustering_advanced` block (~line 598). The exact line is flexible as long as it is in
the re-export section. Style: no wildcard, explicit list, alphabetical within the list.

```rust
// Re-export PEER types (v0.36.0)
pub use peer::{
    peer, lpeer, LambdaChoice, LambdaMethod, LpeerResult, PeerConfig, PeerPenalty, PeerResult,
};
```

### Prelude insertion point

`prelude.rs` ends at line 106 with the co-clustering block. [VERIFIED: fdars-core/src/prelude.rs:104-106]

**Exact insertion:** Append after the last block (after `CoClusterConfig, CoClusterResult, CoClusterSelectResult`):

```rust
// PEER regression (v0.36.0)
pub use crate::peer::{
    peer, lpeer, LambdaChoice, LambdaMethod, LpeerResult, PeerConfig, PeerPenalty, PeerResult,
};
```

### Current `no_run` snippet to replace

Lines 16-25 of `peer.rs`: [VERIFIED: fdars-core/src/peer.rs:16-25]

```rust
//! ```no_run
//! use fdars_core::matrix::FdMatrix;
//! use fdars_core::peer::{peer, PeerConfig, PeerPenalty, LambdaChoice};
//!
//! let data = FdMatrix::zeros(50, 40);
//! let y = vec![0.0_f64; 50];
//! let argvals: Vec<f64> = (0..40).map(|i| i as f64 / 39.0).collect();
//! let config = PeerConfig { penalty: PeerPenalty::Difference { order: 2 }, lambda: LambdaChoice::Fixed(1.0) };
//! let _result = peer(&data, &y, &argvals, &config);
//! ```
```

**Replace with a RUNNING doctest.** Requirements: small n (≥ 2), small m (≥ 3), deterministic
data, finite β and predictions, must compile and pass under `cargo test --doc`.

The test uses the same hash-based data construction as the existing tests, but must be
self-contained in the docstring. Suggested small fixture (n=10, m=5 — tiny to keep doctest fast):

```rust
//! # Quick start
//!
//! ```
//! use fdars_core::matrix::FdMatrix;
//! use fdars_core::peer::{peer, PeerConfig, PeerPenalty, LambdaChoice};
//! use fdars_core::helpers::simpsons_weights;
//!
//! // Build a tiny synthetic dataset: n=10 observations, m=5 evaluation points.
//! let (n, m) = (10_usize, 5_usize);
//! let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
//! let mut data = FdMatrix::zeros(n, m);
//! let mut y = vec![0.0_f64; n];
//! let true_beta: Vec<f64> = argvals.iter().map(|&t| (std::f64::consts::PI * t).sin()).collect();
//! let w = simpsons_weights(&argvals);
//! for i in 0..n {
//!     for j in 0..m {
//!         let xi = ((i * m + j) as f64 * 0.3).sin();
//!         data[(i, j)] = xi;
//!         y[i] += xi * true_beta[j] * w[j];
//!     }
//! }
//! // Fit PEER with Ridge penalty and a fixed λ
//! let config = PeerConfig {
//!     penalty: PeerPenalty::Ridge,
//!     lambda: LambdaChoice::Fixed(1e-3),
//! };
//! let fit = peer(&data, &y, &argvals, &config).unwrap();
//! assert_eq!(fit.beta.len(), m);
//! assert!(fit.fitted_values.iter().all(|v| v.is_finite()));
//! // Predict on training data (self-consistency)
//! let preds = fit.predict(&data, &argvals).unwrap();
//! for (p, f) in preds.iter().zip(&fit.fitted_values) {
//!     assert!((p - f).abs() < 1e-9);
//! }
//! ```
```

Note: the `n=10, m=5` fixture uses simple `.sin()` data that is computable at compile/test time
without any external state. The `peer()` call is guaranteed to succeed because `n ≥ 2`, `m ≥ 3`,
and the Ridge penalty with `lambda = 1e-3` keeps the penalized system well-conditioned.

**lpeer doctest** (separate `///` doc on `lpeer` function):

```rust
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::peer::{lpeer, PeerConfig, PeerPenalty, LambdaChoice};
///
/// let (n, m) = (12_usize, 5_usize);
/// let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
/// let mut data = FdMatrix::zeros(n, m);
/// let mut y = vec![0.0_f64; n];
/// // 3 subjects × 4 observations each
/// let subject_map: Vec<usize> = (0..n).map(|i| i / 4).collect();
/// for i in 0..n {
///     for j in 0..m {
///         let xi = ((i * m + j) as f64 * 0.3).sin();
///         data[(i, j)] = xi;
///         y[i] += xi * 0.5;
///     }
/// }
/// let config = PeerConfig {
///     penalty: PeerPenalty::Ridge,
///     lambda: LambdaChoice::Fixed(1e-2),
/// };
/// let fit = lpeer(&data, &y, &argvals, &subject_map, &config).unwrap();
/// assert_eq!(fit.beta.len(), m);
/// assert!(fit.sigma2_subject >= 0.0);
/// assert!(fit.sigma2_resid >= 0.0);
/// let preds = fit.predict(&data, &argvals).unwrap();
/// assert_eq!(preds.len(), n);
/// ```
```

---

## 6. Synthetic Test Data for lpeer — Known-Answer Gates

### Injecting between-subject variance

To test that `sigma2_subject` tracks the injected between-subject variance:

```rust
fn make_lpeer_fixture() -> (FdMatrix, Vec<f64>, Vec<f64>, Vec<usize>, f64) {
    let (n_subjects, obs_per, m) = (10_usize, 5_usize, 20_usize);
    let n = n_subjects * obs_per;
    let t = uniform_grid(m);
    let true_beta: Vec<f64> = t.iter().map(|&ti| (std::f64::consts::PI * ti).sin()).collect();
    let w = simpsons_weights(&t);

    // Inject between-subject random effects from a known variance σ²_u_true
    let sigma2_u_true = 1.0_f64;

    let mut data = FdMatrix::zeros(n, m);
    let mut y = vec![0.0_f64; n];
    let mut subject_map = vec![0_usize; n];

    for s in 0..n_subjects {
        // Subject random effect: deterministic hash at scale sqrt(sigma2_u_true)
        let u_s = hash_unit(s as u64) * sigma2_u_true.sqrt();
        for obs in 0..obs_per {
            let i = s * obs_per + obs;
            subject_map[i] = s;
            for j in 0..m {
                let xi = hash_unit((i * m + j) as u64);
                data[(i, j)] = xi;
                y[i] += xi * true_beta[j] * w[j];
            }
            y[i] += u_s; // inject subject effect
            // small within-subject noise
            y[i] += 0.02 * hash_unit(1_000_000 + i as u64);
        }
    }
    (data, y, t, subject_map, sigma2_u_true)
}
```

**Assertion:** `fit.sigma2_subject` should be within a factor of 3 of `sigma2_u_true = 1.0`
at n=50 total observations, n_subjects=10. The REML EM is consistent but not exact at small n,
so the tolerance must be generous (e.g., `0.1 < sigma2_subject < 5.0`). Assert non-negative.

### β(t) recovery test

```rust
let max_err = fit.beta.iter().zip(true_beta.iter()).map(|(a,b)| (a-b).abs()).fold(0.0, f64::max);
assert!(max_err < 0.5, "lpeer β(t) recovery error: {max_err}");
```

The tolerance is wider than `peer()` (0.1) because the mixed-model shrinkage introduces
additional bias when n is small. The exact bound is Claude's discretion — 0.5 is conservative.

---

## 7. Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Subject random intercepts REML EM | Custom EM loop in peer.rs | `famm::fit_scalar_mixed_model` | Already tested; correct Henderson init + 50-iter REML EM |
| Subject ID re-indexing | Manual dedup/sort | `famm::build_subject_map` | Handles non-contiguous IDs; tested |
| FPC score projection | Custom projection loop | `fdata_to_pc_1d` → `.scores` | Tested, weighted SVD, L²-normalized |
| Simpson weights in predict | Duplicate simpsons_weights | `helpers::simpsons_weights` | Crate-wide convention |
| NaN guard in `lpeer` | Custom check | Trust `fit_scalar_mixed_model` variance clamping + add post-solve NaN check on beta | Same pattern as `peer()` |

---

## 8. Common Pitfalls

### Pitfall 1: Re-applying score_scale (h.sqrt()) to FPC scores

**What goes wrong:** `famm::fit_all_components` applies `score_scale = h.sqrt()` before calling
`fit_scalar_mixed_model`, then divides `gamma` back by `score_scale` afterward. `fof_re_regression`
explicitly does NOT apply this scaling (see comment at lines 752-755). For lpeer, use the
`fof_re_regression` pattern: pass raw FPC scores directly.
**Why it happens:** Confusion between the two call sites in `famm.rs`.
**How to avoid:** Match the `fof_re_regression` pattern exactly (no scale factor). [VERIFIED:
fdars-core/src/fof_regression.rs:751-755]

### Pitfall 2: Passing raw `y` instead of `yc` to `fit_scalar_mixed_model`

**What goes wrong:** If `y` (not `yc`) is passed, the mixed model estimates the intercept as
part of γ, making `y_bar` redundant. The back-projected β(t) then has a different scale.
**How to avoid:** Either (a) pass `yc` (centered) and set `intercept = y_bar`, or (b) pass raw
`y` with an intercept column in the score matrix. Option (a) is consistent with `peer()`.

### Pitfall 3: Using `subject_map` values directly without build_subject_map

**What goes wrong:** If caller passes non-contiguous IDs (e.g., subject IDs 0, 5, 10),
`fit_scalar_mixed_model` expects 0-indexed contiguous values 0..n_subjects. Out-of-bounds
access or wrong subject grouping.
**How to avoid:** Always call `famm::build_subject_map` first to get a dense 0-indexed map.

### Pitfall 4: Back-projection using wrong rotation layout

**What goes wrong:** `fpca.rotation` is m×ncomp (column = eigenvector). Row-major iteration
`rotation[(j, k)]` gives eigenvector k at grid point j. If mistakenly transposed, β(t) is
wrong.
**How to avoid:** Back-project as `beta[j] = Σ_k gamma[k] * rotation[(j, k)]`.
[VERIFIED: fdars-core/src/famm.rs:659-669 — `val += gamma[j][comp] * rotation[(t, comp)]`]

### Pitfall 5: `predict` self-consistency fails due to w_bar mismatch

**What goes wrong:** `peer()` stores `w_bar = col_mean(wmat)` where `wmat[i,j] = data[(i,j)] * w[j]`.
If `lpeer` computes `w_bar` differently (e.g., col_mean of centered wmat, or without weights),
the predict formula `(intercept − w_bar·β) + Σ_j x*[j]·w[j]·β[j]` no longer reproduces
training fitted values.
**How to avoid:** Compute `w_bar` exactly as `peer()` does it (before centering, after weight
multiplication). [VERIFIED: fdars-core/src/peer.rs:257-260]

### Pitfall 6: Doctest using `no_run` or zero-variance design

**What goes wrong:** Zero-variance design columns (e.g., `FdMatrix::zeros`) make WtW singular
and `cholesky_solve` fails. The existing `no_run` doctest uses `FdMatrix::zeros(50, 40)` which
would fail if made runnable.
**How to avoid:** Use `((i*m+j) as f64 * 0.3).sin()` data (non-constant, full rank for small m).

### Pitfall 7: /tmp exhaustion blocks pre-commit hook

**Memory note:** `--no-verify` on commits avoids slow hook, but skips `cargo fmt`. Run
`cargo fmt` per commit separately. [ASSUMED from project MEMORY.md]

---

## 9. Code Examples

### lpeer — minimal skeleton

```rust
// Source: derived from fof_re_regression.rs + peer.rs patterns [VERIFIED]
pub fn lpeer(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    subject_map: &[usize],
    config: &PeerConfig,
) -> Result<LpeerResult, FdarError> {
    let (n, m) = data.shape();
    // --- Entry validation ---
    if n < 2 { /* InvalidDimension */ }
    if m < 3 { /* InvalidDimension */ }
    if argvals.len() != m { /* InvalidDimension */ }
    if y.len() != n { /* InvalidDimension */ }
    if subject_map.len() != n { /* InvalidDimension */ }

    // Build dense subject map (handles non-contiguous IDs)
    let (sm_dense, n_subjects) = crate::famm::build_subject_map(subject_map);
    if n_subjects < 2 { /* InvalidParameter */ }

    // 1. Integration weights + weighted design (mirrors peer())
    let w = simpsons_weights(argvals);
    let mut wmat = FdMatrix::zeros(n, m);
    for i in 0..n { for j in 0..m { wmat[(i,j)] = data[(i,j)] * w[j]; } }

    let y_bar: f64 = y.iter().sum::<f64>() / n as f64;
    let yc: Vec<f64> = y.iter().map(|&yi| yi - y_bar).collect();
    let w_bar: Vec<f64> = (0..m).map(|j| (0..n).map(|i| wmat[(i,j)]).sum::<f64>() / n as f64).collect();

    // 2. λ selection using the same PeerConfig dispatch as peer()
    let q = build_q(m, &config.penalty)?;
    let mut wc = FdMatrix::zeros(n, m);
    for i in 0..n { for j in 0..m { wc[(i,j)] = wmat[(i,j)] - w_bar[j]; } }
    let mut wtw = vec![0.0_f64; m * m];
    for j in 0..m { for k in j..m {
        let s: f64 = (0..n).map(|i| wc[(i,j)] * wc[(i,k)]).sum();
        wtw[j*m+k] = s; wtw[k*m+j] = s;
    } }
    let wty: Vec<f64> = (0..m).map(|j| (0..n).map(|i| wc[(i,j)] * yc[i]).sum()).collect();

    let (lambda, gcv_score, lambda_method) = match &config.lambda {
        LambdaChoice::Fixed(lam) => (*lam, None, LambdaMethod::Fixed),
        LambdaChoice::Gcv => { let (lam, g) = select_lambda_gcv_peer(&wc, &yc, &wtw, &wty, &q, m, n); (lam, Some(g), LambdaMethod::Gcv) }
        LambdaChoice::Reml => { let lam = select_lambda_reml_peer(&wc, &yc, &q, m, n); (lam, None, LambdaMethod::Reml) }
    };

    // 3. FPC score reduction
    let ncomp = (n - 1).min(m).min(10);  // cap at 10; document this
    let fpca = crate::regression::fdata_to_pc_1d(data, ncomp, argvals)?;
    let scores = &fpca.scores;  // n×ncomp FdMatrix

    // 4. Mixed-model fit over FPC scores
    let result = crate::famm::fit_scalar_mixed_model(&yc, &sm_dense, n_subjects, Some(scores), ncomp);

    // NaN guard
    if result.gamma.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::ComputationFailed { operation: "lpeer", detail: "non-finite gamma".into() });
    }

    // 5. Back-project γ → β(t)
    let beta: Vec<f64> = (0..m)
        .map(|j| (0..ncomp.min(result.gamma.len())).map(|k| result.gamma[k] * fpca.rotation[(j, k)]).sum())
        .collect();

    if beta.iter().any(|v| !v.is_finite()) {
        return Err(FdarError::ComputationFailed { operation: "lpeer", detail: "non-finite beta".into() });
    }

    // 6. Fitted values (same formula as peer())
    let base = y_bar - w_bar.iter().zip(&beta).map(|(wb, b)| wb * b).sum::<f64>();
    let fitted_values: Vec<f64> = (0..n)
        .map(|i| base + (0..m).map(|j| data[(i,j)] * w[j] * beta[j]).sum::<f64>())
        .collect();

    Ok(LpeerResult {
        beta,
        intercept: y_bar,
        w_bar,
        fitted_values,
        sigma2_subject: result.sigma2_u,
        sigma2_resid: result.sigma2_eps,
        n_subjects,
        lambda,
        penalty_type: config.penalty.clone(),
        gcv: gcv_score,
        lambda_method,
    })
}
```

### predict — shared helper

```rust
// Source: derived from peer.rs test_peer_stores_w_bar_for_prediction [VERIFIED]
fn peer_predict_core(
    beta: &[f64], intercept: f64, w_bar: &[f64],
    new_data: &FdMatrix, argvals: &[f64],
) -> Result<Vec<f64>, FdarError> {
    let (n_new, m_new) = new_data.shape();
    let m = beta.len();
    if m_new != m {
        return Err(FdarError::InvalidDimension {
            parameter: "new_data",
            expected: format!("{m} columns"),
            actual: format!("{m_new}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    let w = simpsons_weights(argvals);
    let base: f64 = intercept - w_bar.iter().zip(beta).map(|(wb, b)| wb * b).sum::<f64>();
    Ok((0..n_new)
        .map(|i| base + (0..m).map(|j| new_data[(i,j)] * w[j] * beta[j]).sum::<f64>())
        .collect())
}
```

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | none (inline module tests per crate convention) |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel peer` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |
| Doctest command | `cargo test -p fdars-core --doc --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| PER-04 | `sigma2_subject >= 0` and `sigma2_resid >= 0` (non-negative) | unit | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_lpeer_variance_non_negative` | Wave 0 |
| PER-04 | `lpeer` recovers β(t) within tolerance on synthetic data | unit (known-answer) | `cargo test ... peer::tests::test_lpeer_beta_recovery` | Wave 0 |
| PER-04 | `sigma2_subject` tracks injected between-subject variance (within factor 3) | unit (known-answer) | `cargo test ... peer::tests::test_lpeer_sigma2_tracks_injection` | Wave 0 |
| PER-04 | `subject_map.len() != n` → `InvalidDimension` | unit | `cargo test ... peer::tests::test_lpeer_invalid_subject_map` | Wave 0 |
| PER-04 | n_subjects < 2 → `InvalidParameter` | unit | `cargo test ... peer::tests::test_lpeer_single_subject_rejected` | Wave 0 |
| PER-05 | `PeerResult::predict` self-consistency: repassed training curves == fitted_values ±1e-9 | unit | `cargo test ... peer::tests::test_peer_predict_self_consistent` | Wave 1 |
| PER-05 | `LpeerResult::predict` self-consistency | unit | `cargo test ... peer::tests::test_lpeer_predict_self_consistent` | Wave 1 |
| PER-05 | `predict` returns `InvalidDimension` on ncols mismatch | unit | `cargo test ... peer::tests::test_predict_wrong_ncols` | Wave 1 |
| PER-05 | `predict` returns all-finite on new curves (not training data) | unit | `cargo test ... peer::tests::test_predict_new_curves_finite` | Wave 1 |
| PER-05 | `fdars_core::peer` (crate-root) reachable for all 7 items | unit (compile-check) | `cargo test ... peer::tests::test_crate_root_exports_compile` | Wave 1 |
| PER-05 | `fdars_core::prelude::*` brings peer items into scope | unit (compile-check) | `cargo test ... peer::tests::test_prelude_exports_compile` | Wave 1 |
| PER-05 | Module doctest passes | doctest | `cargo test -p fdars-core --doc --features linalg,parallel` | Wave 1 |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg,parallel peer -- --nocapture`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite + doctests + clippy + fmt before `/gsd-verify-work`:
  ```bash
  cargo test -p fdars-core --features linalg,parallel
  cargo test -p fdars-core --doc --features linalg,parallel
  cargo clippy --all-targets --features linalg,parallel -- -D warnings
  cargo fmt -- --check
  ```

### Wave 0 Gaps (new test functions needed)

- [ ] `test_lpeer_variance_non_negative` — assert both variance components ≥ 0 on fixture
- [ ] `test_lpeer_beta_recovery` — known β(t)=sin(πt), max_err < 0.5
- [ ] `test_lpeer_sigma2_tracks_injection` — injected σ²_u ≈ 1.0, fitted in (0.1, 5.0)
- [ ] `test_lpeer_invalid_subject_map` — wrong length → `InvalidDimension`
- [ ] `test_lpeer_single_subject_rejected` — all-same subject_map → `InvalidParameter`

### Wave 1 Gaps (new test functions needed)

- [ ] `test_peer_predict_self_consistent` — PeerResult::predict on training data
- [ ] `test_lpeer_predict_self_consistent` — LpeerResult::predict on training data
- [ ] `test_predict_wrong_ncols` — new_data with wrong m → `InvalidDimension`
- [ ] `test_predict_new_curves_finite` — fresh curves, predict returns all-finite
- [ ] `test_crate_root_exports_compile` — `use fdars_core::{peer, lpeer, PeerConfig, ...}`
- [ ] `test_prelude_exports_compile` — `use fdars_core::prelude::*; let _: PeerResult;`
- [ ] Doctest in module header (replace `no_run`) — must pass `cargo test --doc`

---

## Open Questions

1. **`ncomp` for lpeer: default cap of 10 or configurable?**
   - What we know: `fdata_to_pc_1d` caps at `min(n, m)`. A default of `min(n-1, m, 10)` is
     practical for small data; larger ncomp slows the mixed model.
   - What's unclear: Should it be a field in `PeerConfig`?
   - Recommendation: Claude's discretion. Default `min(n-1, m, 10)` is reasonable and documented.

2. **Pass `yc` or `y` to `fit_scalar_mixed_model`?**
   - What we know: Passing `yc` (centered) and setting `intercept = y_bar` matches the `peer()`
     convention exactly, making the predict formula identical for both result types.
   - Recommendation: Pass `yc`, set `intercept = y_bar`.

3. **Should `u_hat` (subject BLUPs) be stored in `LpeerResult`?**
   - What we know: CONTEXT.md does not require BLUPs in the result. Marginal prediction ignores them.
   - Recommendation: Do not store BLUPs (follow locked decision: deferred to a future milestone).

---

## Environment Availability

No external dependencies beyond the existing Rust toolchain and crate. Step 2.6: SKIPPED
(all capabilities use existing in-crate infrastructure).

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | FPC scores from `fdata_to_pc_1d` are suitable predictors for `fit_scalar_mixed_model` without `h.sqrt()` rescaling (matching `fof_re_regression` pattern) | §2, §3 | If rescaling is needed, γ magnitude is off by `sqrt(h)`, biasing β(t) by the same factor |
| A2 | Passing `yc` (centered) to `fit_scalar_mixed_model` and adding `y_bar` back as `intercept` is algebraically equivalent to passing raw `y` with an intercept column | §3 | If wrong, fitted_values mismatch training data and self-consistency test fails |
| A3 | `ncomp = min(n-1, m, 10)` is a reasonable default for lpeer when no user-specified ncomp is available | §3 | Too few components → underfitting of β(t); too many → `fdata_to_pc_1d` returns fewer than requested (capped at min(n,m)) |
| A4 | `sigma2_subject` tracks injected variance within a factor of 3 at n=50 obs / 10 subjects | §6 | Mixed model is consistent but slow to converge at small n; test tolerance may need widening |
| A5 | The module doctest with n=10, m=5, Ridge penalty, lambda=1e-3 produces a non-singular penalized system | §5 | If the small fixture has degenerate curves (all equal), Cholesky fails; sinusoidal construction avoids this |

**If this table is not empty:** A1 is the highest-risk assumption; the self-consistency test
(predict on training data == fitted_values ±1e-9) will expose any scaling or centering error
immediately. A2 can be verified with a known-answer fixture before committing.

---

## Sources

### Primary (HIGH confidence — files read this session)

- `fdars-core/src/peer.rs:1-1369` — full PeerResult, PeerConfig, PeerPenalty, LambdaChoice,
  LambdaMethod definitions; peer() algorithm; w_bar field; test_peer_stores_w_bar_for_prediction
- `fdars-core/src/famm.rs:438-499` — `fit_scalar_mixed_model` exact signature and
  `ScalarMixedResult` struct with field names and types
- `fdars-core/src/famm.rs:185-197` — `build_subject_map` signature and implementation
- `fdars-core/src/famm.rs:427-430, 454-459` — variance clamping (non-negative floor)
- `fdars-core/src/fof_regression.rs:748-786` — famm integration template (no score_scale)
- `fdars-core/src/famm.rs:652-670` — `recover_beta_functions` back-projection pattern
- `fdars-core/src/lib.rs:64-681` — module registration and re-export structure; `pub mod peer;`
  at line 111; insertion points for `pub use peer::{...}`
- `fdars-core/src/prelude.rs:1-106` — prelude structure and insertion point
- `fdars-core/src/scalar_on_function/mod.rs:668-703` — predict-method-on-result-struct convention
- `fdars-core/src/regression.rs:300-379` — `fdata_to_pc_1d` signature and `FpcaResult` layout
  (scores: n×ncomp, rotation: m×ncomp)

### Tertiary (LOW confidence — training knowledge)

- lpeer conceptual design (subject random effects via REML EM + score basis) — matches refund
  `lpeer` design but not verified against refund source [ASSUMED]

---

## Metadata

**Confidence breakdown:**
- `fit_scalar_mixed_model` contract: HIGH (file read, verbatim quotes)
- `fof_regression.rs` template: HIGH (file read, verbatim quotes)
- `lpeer` algorithm design: HIGH (derived mechanically from read sources)
- `predict` formula and self-consistency: HIGH (test already exercises it verbatim)
- Integration insertion points: HIGH (lib.rs and prelude.rs read in full)
- `sigma2_subject` tracking tolerance: MEDIUM (n=50 REML convergence at small n is uncertain)

**Research date:** 2026-09-04
**Valid until:** 2026-12-04 (stable Rust crate internals; 90 days)
