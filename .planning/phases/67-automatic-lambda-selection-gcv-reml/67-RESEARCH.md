# Phase 67: Automatic λ Selection — GCV + REML — Research

**Researched:** 2026-09-04
**Domain:** Automatic smoothing-parameter selection for PEER penalized regression (GCV grid search + REML EM via eigendecomposition of Q)
**Confidence:** HIGH (all source files read directly this session; REML math tagged [ASSUMED] where not cross-verified against a primary source)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- `PeerConfig.lambda: f64` → `lambda: LambdaChoice` enum with variants `Fixed(f64)`, `Gcv`, `Reml`. Serde behind `serde` feature per crate convention. `Default` = `Gcv`.
- `Fixed(λ)` is used verbatim: no search runs, λ appears unchanged in `PeerResult` diagnostics.
- GCV: fixed internal log-spaced grid ~[1e-6, 1e4], ~40 points (not user-tunable this phase). Score: `GCV = n·RSS/(n−tr(H))²`. Selection = argmin GCV. Deterministic (no RNG); ties broken by smaller grid index (documented).
- REML: self-contained EM via eigendecomposition of Q (null space → fixed/unpenalized, range space → random effect `b~N(0,σ²_u I)`). λ = σ²_e/σ²_u. Do NOT call `famm::fit_scalar_mixed_model` directly.
- `PeerResult` gains: `gcv: Option<f64>` (GCV score at selected λ; `None` when GCV did not run) and `lambda_method: LambdaMethod` (marker: Fixed/Gcv/Reml).
- All changes in `fdars-core/src/peer.rs`. No new files, no new crate dependencies.
- Additive/non-breaking. Tests that construct `PeerConfig { lambda: <f64>, ... }` must be mechanically updated to `lambda: LambdaChoice::Fixed(<f64>)`.

### Claude's Discretion

- Exact grid point count and REML EM iteration cap / convergence tolerance.
- Whether `lambda_method` is a new small enum or reuses `LambdaChoice`'s discriminant.
- Internal factoring (helper fns) within `peer.rs`.

### Deferred Ideas (OUT OF SCOPE)

- Longitudinal `lpeer`, out-of-sample `predict`, crate-root/prelude exports, end-to-end module doctest — Phase 68.
- User-configurable GCV grid bounds/count.

</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PER-03 | Automatic λ selection via GCV or REML, config-selectable; explicit λ honored verbatim | GCV recipe §GCV Path; REML math + EM plan §REML Path; API evolution §API Evolution; test plan §Validation Architecture |

</phase_requirements>

---

## Summary

Phase 67 adds automatic smoothing-parameter selection on top of the Phase 66 PEER penalized fit `(W_c'W_c + λQ)β = W_c'y_c`. Two selectors are added: GCV grid search and a self-contained REML EM, selectable via a new `LambdaChoice` enum that replaces the plain `f64` in `PeerConfig`. The phase is entirely contained in `fdars-core/src/peer.rs` with no new dependencies.

**GCV** is a thin wrapper around the existing pattern in `function_on_scalar.rs`: iterate over a fixed log-spaced grid, call the penalized Cholesky solve + `compute_peer_trace_hat`, compute `n·RSS/(n−tr(H))²`, return the argmin. Every value needed is already computed by Phase 66 helpers.

**REML** is the structurally novel piece. The PEER-as-mixed-model equivalence maps the range space of Q to a random effect `b~N(0,σ²_u I)` and the null space to fixed (unpenalized) components. The eigendecomposition of Q (via nalgebra `symmetric_eigen`, already used in `fpca_variants.rs`) partitions the m-dimensional coefficient space. In the transformed basis the model is a standard one-way random-intercept mixed model, but with a single "group" (all m range-space dimensions share one σ²_u), so `famm::fit_scalar_mixed_model` (which is subject-grouped) must NOT be called directly. A thin 8–12 iteration EM loop is written locally in `peer.rs`, borrowing the variance-update math from `famm::reml_variance_update`'s logic but without the subject-grouping structure.

**Primary recommendation:** Implement GCV first (it is a 40-point grid loop over existing helpers — ~30 lines). Implement REML as a self-contained eigendecomposition + EM in `peer.rs` (no new files). Add `LambdaChoice` and `LambdaMethod` enums, update `PeerResult`, mechanically update tests. Total new code is modest; the hard part is the REML math, which is specified in full below.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| GCV grid search | `peer.rs` (new `select_lambda_gcv_peer` fn) | `compute_peer_trace_hat` (Phase 66, reuse) | GCV uses the same WtW + Q + Cholesky path already in peer.rs |
| REML EM loop | `peer.rs` (new `select_lambda_reml_peer` fn) | nalgebra `symmetric_eigen` (via fpca_variants.rs pattern) | Self-contained; no file or module added |
| Variance update math | `peer.rs` inline | `famm::reml_variance_update` pattern (reuse math, not function) | subject-grouped structure in famm is incompatible; port the update equations |
| LambdaChoice / LambdaMethod enums | `peer.rs` public types | — | Config and result marker live next to PeerConfig/PeerResult |
| Test mechanical update | `peer.rs` #[cfg(test)] | — | All Phase 66 tests that set `lambda: f64` migrate to `LambdaChoice::Fixed` |

---

## Standard Stack

### Core (no change)

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nalgebra | 0.33 [VERIFIED: fdars-core/Cargo.toml] | `DMatrix::symmetric_eigen()` for REML eigendecomposition of Q | Crate-wide LA primitive; already used in fpca_variants.rs |
| (no new dep) | — | — | Locked constraint |

### Supporting (in-crate reuse)

| Module / Function | Location | Line Range | Purpose |
|-------------------|----------|-----------|---------|
| `compute_peer_trace_hat` | `peer.rs` | 324–340 | Computes tr(H) = tr(A^{-1} WtW); reused in GCV inner loop |
| `cholesky_solve` | `linalg.rs` | 131–134 | Penalized solve for each grid-point candidate λ |
| `cholesky_factor` + `cholesky_forward_back` | `linalg.rs` | 85–128 | Used internally by `compute_peer_trace_hat` |
| `DMatrix::from_fn(...).symmetric_eigen()` | nalgebra (via `fpca_variants.rs:473–484` pattern) | — | Eigendecompose Q for REML null/range split |

**Installation:** No `Cargo.toml` change.

---

## Package Legitimacy Audit

No new packages. Not applicable.

---

## API Evolution

### Before (Phase 66)

```rust
// VERIFIED: fdars-core/src/peer.rs:70-85
pub struct PeerConfig {
    pub penalty: PeerPenalty,
    pub lambda: f64,     // <-- plain f64
}

impl Default for PeerConfig {
    fn default() -> Self {
        Self { penalty: PeerPenalty::default(), lambda: 1.0 }
    }
}

// VERIFIED: fdars-core/src/peer.rs:87-119
pub struct PeerResult {
    pub beta: Vec<f64>,
    pub intercept: f64,
    pub w_bar: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub effective_df: f64,
    pub lambda: f64,          // <-- selected/used λ
    pub penalty_type: PeerPenalty,
    // -- no gcv, no lambda_method --
}
```

### After (Phase 67)

```rust
/// How to choose the smoothing parameter λ.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LambdaChoice {
    /// Use this value verbatim; no search is performed.
    Fixed(f64),
    /// Select λ by minimising GCV on a fixed log-spaced internal grid.
    Gcv,
    /// Estimate λ via REML/mixed-model (eigendecomposition of Q, self-contained EM).
    Reml,
}

impl Default for LambdaChoice {
    fn default() -> Self { LambdaChoice::Gcv }
}

/// Which selection path actually ran.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum LambdaMethod { Fixed, Gcv, Reml }

pub struct PeerConfig {
    pub penalty: PeerPenalty,
    pub lambda: LambdaChoice,   // <-- replaces f64
}

impl Default for PeerConfig {
    fn default() -> Self {
        Self { penalty: PeerPenalty::default(), lambda: LambdaChoice::default() }
    }
}

#[non_exhaustive]
pub struct PeerResult {
    pub beta: Vec<f64>,
    pub intercept: f64,
    pub w_bar: Vec<f64>,
    pub fitted_values: Vec<f64>,
    pub effective_df: f64,
    pub lambda: f64,            // always the value actually used
    pub penalty_type: PeerPenalty,
    pub gcv: Option<f64>,       // GCV score at selected λ; None if GCV did not run
    pub lambda_method: LambdaMethod,  // which path ran
}
```

### Tests that must be mechanically updated

Every test in the existing `#[cfg(test)] mod tests` block (VERIFIED: peer.rs:347–773) that constructs `PeerConfig { ..., lambda: <f64> }` must change `lambda: <f64>` to `lambda: LambdaChoice::Fixed(<f64>)`. The concrete occurrences are:

| Test name | Line (approx) | Old | New |
|-----------|---------------|-----|-----|
| `test_peer_difference_beta_recovery` | ~402 | `lambda: 1e-4` | `lambda: LambdaChoice::Fixed(1e-4)` |
| `test_peer_result_shape` | ~430 | `lambda: 1e-4` | `lambda: LambdaChoice::Fixed(1e-4)` |
| `test_peer_ridge_fits` | ~474 | `lambda: 1e-4` | `lambda: LambdaChoice::Fixed(1e-4)` |
| `test_peer_decree_fits` | ~494 | `lambda: 1e-4` | `lambda: LambdaChoice::Fixed(1e-4)` |
| `test_peer_difference_order_rejected` | ~515 | `lambda: 1e-4` | `lambda: LambdaChoice::Fixed(1e-4)` |
| `test_peer_decree_distinct_from_roughness` (×2) | ~533, ~563 | `lambda: 1.0` | `lambda: LambdaChoice::Fixed(1.0)` |
| `test_peer_no_nan_all_families` (×3) | ~640, ~654, ~672 | `lambda: 1.0` | `lambda: LambdaChoice::Fixed(1.0)` |
| `test_peer_stores_w_bar_for_prediction` | ~705 | `lambda: 1e-4` | `lambda: LambdaChoice::Fixed(1e-4)` |
| `PeerConfig::default()` in three tests | ~615, ~739, ~757 | (default was `lambda: 1.0`) | default now `LambdaChoice::Gcv` — no code change needed in those tests, but be aware the default behavior changes |

Also: the Phase 66 doc-comment `quick start` example in the module header (peer.rs:22–24) constructs `PeerConfig { penalty: ..., lambda: 1.0 }` — update to `lambda: LambdaChoice::Fixed(1.0)`. [VERIFIED: fdars-core/src/peer.rs:22–24]

After the mechanical update, also add `gcv: None, lambda_method: LambdaMethod::Fixed` to any `PeerResult { ... }` literals in tests if they construct one directly (check whether any test builds PeerResult by value — likely none do since PeerResult is returned from `peer()`).

---

## GCV Path — Concrete Recipe

### Existing GCV convention in the crate (template to match)

`function_on_scalar.rs` `compute_fosr_gcv` uses:
```rust
// VERIFIED: fdars-core/src/function_on_scalar.rs:171-178
fn compute_fosr_gcv(residuals: &FdMatrix, trace_h: f64) -> f64 {
    let (n, m) = residuals.shape();
    let denom = (1.0 - trace_h / n as f64).max(1e-10);
    let ss_res: f64 = (0..n)
        .flat_map(|i| (0..m).map(move |t| residuals[(i, t)].powi(2)))
        .sum();
    ss_res / (n as f64 * m as f64 * denom * denom)
}
```
This is the FOSR form `GCV = RSS / (nm · (1 − tr(H)/n)²)`. For PEER the response is scalar (m_rhs=1), so:
```
GCV_peer = RSS / (n · (1 − tr(H)/n)²)
         = n · RSS / (n − tr(H))²
```
Both forms are algebraically equivalent (multiply numerator and denominator by n). Use the CONTEXT-mandated form `n · RSS / (n − tr(H))²` directly. [VERIFIED formulation matches CONTEXT.md §GCV grid search]

`function_on_scalar.rs` `select_lambda_gcv` uses a 9-point coarse grid:
```rust
// VERIFIED: fdars-core/src/function_on_scalar.rs:371-397
let lambdas = [0.0, 1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0];
```
Phase 67 uses a finer 40-point log-spaced grid over [1e-6, 1e4] (not user-tunable, per CONTEXT.md). Match the `select_lambda_gcv` pattern but with the finer grid.

### GCV helper implementation plan

```rust
/// Build a 40-point log-spaced grid on [1e-6, 1e4].
fn gcv_lambda_grid() -> Vec<f64> {
    // 40 points: 10^x for x in linspace(-6, 4, 40)
    (0..40)
        .map(|i| 10.0_f64.powf(-6.0 + 10.0 * i as f64 / 39.0))
        .collect()
}

/// Select λ via GCV for the PEER model.
///
/// Grid: 40 log-spaced points on [1e-6, 1e4].
/// Score: GCV(λ) = n · RSS(λ) / (n − tr(H(λ)))².
/// Returns (best_lambda, gcv_at_best).
/// Guard: if tr(H) >= n for every candidate, returns the smallest grid λ.
fn select_lambda_gcv_peer(
    wc: &FdMatrix,    // n×m centered weighted design (already computed)
    yc: &[f64],       // n-vector centered response
    wtw: &[f64],      // m×m WtW (row-major)
    wty: &[f64],      // m-vector Wty
    q: &[f64],        // m×m penalty (row-major)
    m: usize,
    n: usize,
) -> (f64, f64) {
    let grid = gcv_lambda_grid();
    let mut best_lam = grid[0];
    let mut best_gcv = f64::INFINITY;

    for &lam in &grid {
        // Build A = WtW + lam*Q
        let mut a = vec![0.0_f64; m * m];
        for i in 0..m * m { a[i] = wtw[i] + lam * q[i]; }
        // Solve for beta
        let Ok(beta) = cholesky_solve(&a, wty, m) else { continue; };
        // RSS = ||y_c - W_c beta||^2
        let rss: f64 = (0..n)
            .map(|i| {
                let yhat: f64 = (0..m).map(|j| wc[(i, j)] * beta[j]).sum();
                (yc[i] - yhat).powi(2)
            })
            .sum();
        // tr(H) reuses compute_peer_trace_hat
        let trh = compute_peer_trace_hat(wtw, q, lam, m, n);
        // Guard: skip if denominator <= 0
        let denom = n as f64 - trh;
        if denom <= 0.0 { continue; }
        let gcv = n as f64 * rss / (denom * denom);
        // Ties broken by smaller grid index (first occurrence wins)
        if gcv < best_gcv {
            best_gcv = gcv;
            best_lam = lam;
        }
    }
    (best_lam, best_gcv)
}
```

Note: `wc` is needed only for computing RSS (the beta solve uses `wty` directly). To avoid recomputing `wc` in the GCV loop, pass it through from the main `peer()` function where it is already built. Alternatively, since `beta` is computed from `wty` anyway, and `fitted[i] = sum_j wc[i,j]*beta[j]`, the RSS loop above is the most direct form.

**tr(H) guard:** `compute_peer_trace_hat` (VERIFIED: peer.rs:324–340) already clamps to `n as f64` via `.min(n as f64)`, so `n − tr(H)` can be near-zero but not negative. Add the `denom <= 0.0` skip above for safety.

---

## REML Path — Math and Rust Plan

### Background: PEER as a mixed model

The PEER model is:
```
y_c = W_c β + ε,  β ~ smoothness prior through Q
```
The PEER-as-mixed-model identity: eigendecompose Q = V Λ V' (V: m×m orthogonal, Λ = diag(λ_1,...,λ_m), λ_i ≥ 0 sorted ascending). Partition:
- Null space of Q: columns of V where |λ_i| < tol (eigenvalue ≈ 0) → unpenalized fixed effect α.
- Range space of Q: columns of V where λ_i ≥ tol → random effect b ~ N(0, σ²_u I_{r}) where r = dim(range space).

Let r = number of range-space columns, s = m − r (null-space columns).

Reparameterize: β = V_null · α + V_range · b, where α ∈ ℝˢ (null-space coefficients, unpenalized) and b ∈ ℝʳ (range-space coefficients, random, b ~ N(0, σ²_u I_r)).

Design in new basis:
```
Z_null  = W_c · V_null    (n×s)   — fixed-effect design
Z_range = W_c · V_range   (n×r)   — random-effect design
```
Model: `y_c = Z_null α + Z_range b + ε,  b ~ N(0, σ²_u I_r),  ε ~ N(0, σ²_e I_n)`.

This is a standard linear mixed model with one random-effect "group" (all b components are i.i.d. with variance σ²_u). λ = σ²_e / σ²_u falls out directly from the variance ratio.

**This is NOT subject-grouped** — `famm::fit_scalar_mixed_model` uses a per-subject random intercept (each subject has its own scalar u_i), accumulated via `SubjectStructure` (VERIFIED: famm.rs:272–288) and `shrinkage_weights` (VERIFIED: famm.rs:291–303). The PEER smoothing random effect has a DIFFERENT structure: it is a shared random vector b (not per-subject scalars), so `fit_scalar_mixed_model` cannot be called directly. [VERIFIED: famm.rs:438–499 — `fit_scalar_mixed_model` takes `subject_map: &[usize]`, `n_subjects: usize`; not applicable to PEER's range-space random effect]

### EM algorithm for σ²_u and σ²_e

Given α (fixed effect, length s) and current (σ²_u, σ²_e), the EM updates are [ASSUMED — standard smoothing-REML EM; cross-referenced with famm.rs reml_variance_update logic]:

**E-step** (conditional expectation of b):
```
Σ_b|y = (I_r/σ²_u + Z_range'Z_range/σ²_e)^{-1}          (r×r)
b_hat  = Σ_b|y · Z_range' (y_c - Z_null α) / σ²_e       (length r)
```

**M-step** (variance updates):
```
σ²_u_new = (b_hat'b_hat + tr(Σ_b|y)) / r
σ²_e_new = (||y_c - Z_null α - Z_range b_hat||^2 + tr(Z_range Σ_b|y Z_range')) / n
```

**Fixed effect update** (iterate with variance updates):
```
α_new = (Z_null' Σ^{-1} Z_null)^{-1} Z_null' Σ^{-1} y_c
where Σ = σ²_e I_n + σ²_u Z_range Z_range'
```
For the GLS update of α, exploit that Σ^{-1} = (1/σ²_e)(I - σ²_u/(σ²_e + σ²_u·||z||^2) Z_range Z_range' locally) — the Woodbury identity. In practice, for small r (range-space dimension), forming Z_null'Σ^{-1}Z_null explicitly is tractable.

**Simplified init + convergence:**
- Initialize: σ²_e = var(y_c), σ²_u = σ²_e * 0.1, α = OLS solution for α.
- Convergence: |σ²_u_new − σ²_u| + |σ²_e_new − σ²_e| < 1e-8 * (σ²_u + σ²_e), OR iteration cap of 100.
- Guard σ²_u → 0: clamp to 1e-12 (λ → ∞ means over-smoothing; return λ = σ²_e/σ²_u which stays finite).
- Guard σ²_u → ∞: no upper clamp needed; large σ²_u gives λ = σ²_e/σ²_u → 0 (no smoothing), which is valid.
- Guard all-zero null space (r = m): the full β is penalized; α = scalar intercept or nothing — handle by falling back to GCV if r = m and s = 0 (degenerate case).

### Eigendecomposition of Q

Precedent: `fpca_variants.rs:473–484` uses:
```rust
// VERIFIED: fdars-core/src/fpca_variants.rs:473-484
use nalgebra::DMatrix;
let eigen = DMatrix::from_fn(q, q, |a, b| { ... }).symmetric_eigen();
let pairs: Vec<(f64, usize)> = (0..eigen.eigenvalues.len())
    .map(|idx| (eigen.eigenvalues[idx], idx)).collect();
pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
```
`symmetric_eigen()` returns eigenvalues in arbitrary order (the sort is applied manually in fpca_variants.rs). [VERIFIED: fpca_variants.rs:485–488 shows explicit sort by descending value]

For PEER REML, sort ascending (ascending λ_i, null first):

```rust
use nalgebra::DMatrix;

// Q is m×m row-major flat Vec<f64>; for symmetric Q, row-major == column-major
let q_mat = DMatrix::from_row_slice(m, m, &q);
let eigen = q_mat.symmetric_eigen();
// Sort ascending: smallest eigenvalues = null space
let mut idx_sorted: Vec<usize> = (0..m).collect();
idx_sorted.sort_by(|&a, &b| {
    eigen.eigenvalues[a]
        .partial_cmp(&eigen.eigenvalues[b])
        .unwrap_or(std::cmp::Ordering::Equal)
});

// Eigenvalue tolerance for null/range split
let max_ev = idx_sorted.iter()
    .map(|&i| eigen.eigenvalues[i].abs())
    .fold(0.0_f64, f64::max);
let tol = 1e-8 * max_ev.max(1.0);   // protect against all-zero Q

let null_idx: Vec<usize> = idx_sorted.iter()
    .copied()
    .filter(|&i| eigen.eigenvalues[i].abs() < tol)
    .collect();
let range_idx: Vec<usize> = idx_sorted.iter()
    .copied()
    .filter(|&i| eigen.eigenvalues[i].abs() >= tol)
    .collect();
let s = null_idx.len();   // null-space dim
let r = range_idx.len();  // range-space dim
```

Form the projected designs:
```rust
// V_null is m×s; columns = null eigenvectors
// V_range is m×r; columns = range eigenvectors
// Z_null = W_c · V_null  (n×s, row-major)
// Z_range = W_c · V_range (n×r, row-major)
let mut z_null  = vec![0.0_f64; n * s];
let mut z_range = vec![0.0_f64; n * r];
for i in 0..n {
    for (col, &ev_idx) in null_idx.iter().enumerate() {
        let mut val = 0.0;
        for j in 0..m {
            val += wc[(i, j)] * eigen.eigenvectors[(j, ev_idx)];
        }
        z_null[i * s + col] = val;
    }
    for (col, &ev_idx) in range_idx.iter().enumerate() {
        let mut val = 0.0;
        for j in 0..m {
            val += wc[(i, j)] * eigen.eigenvectors[(j, ev_idx)];
        }
        z_range[i * r + col] = val;
    }
}
```

### EM loop: concrete implementation structure

```rust
fn select_lambda_reml_peer(
    wc: &FdMatrix,    // n×m
    yc: &[f64],       // length n
    q: &[f64],        // m×m row-major
    m: usize,
    n: usize,
) -> f64 {
    // 1. Eigendecompose Q
    // ... (as above) ...

    // 2. Edge case: no range space → Q is zero matrix → no smoothing needed
    if r == 0 { return 1e-4; }  // fallback to a small fixed λ

    // 3. Build Z_null (n×s) and Z_range (n×r)
    // ... (as above) ...

    // 4. Initialize
    let y_var = {
        let mean = yc.iter().sum::<f64>() / n as f64;
        yc.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1).max(1) as f64
    };
    let mut sigma2_e = y_var.max(1e-12);
    let mut sigma2_u = (sigma2_e * 0.1).max(1e-12);

    // OLS init of alpha from z_null (if s > 0)
    let mut alpha = vec![0.0_f64; s]; // ... solve (Z_null' Z_null) alpha = Z_null' yc ...

    for _iter in 0..100 {
        let su_old = sigma2_u;
        let se_old = sigma2_e;

        // E-step: compute Sigma_b = (I_r/sigma2_u + ZtZ_range/sigma2_e)^{-1}
        // ZtZ_range is r×r
        // ... form ZtZ_range + (sigma2_e/sigma2_u)*I_r, Cholesky-invert ...
        // b_hat = Sigma_b * Z_range' * r_alpha / sigma2_e
        // r_alpha = yc - Z_null*alpha

        // M-step: update sigma2_u, sigma2_e
        // sigma2_u_new = (b_hat'b_hat + tr(Sigma_b)) / r
        // sigma2_e_new = (||r_alpha - Z_range*b_hat||^2 + tr(Z_range * Sigma_b * Z_range')) / n

        // GLS update of alpha (if s > 0)
        // ... see below ...

        // Clamp to avoid degenerate variance
        sigma2_u = sigma2_u.max(1e-12);
        sigma2_e = sigma2_e.max(1e-12);

        let delta = (sigma2_u - su_old).abs() + (sigma2_e - se_old).abs();
        if delta < 1e-8 * (su_old + se_old) { break; }
    }

    // lambda = sigma2_e / sigma2_u
    (sigma2_e / sigma2_u).max(1e-15)
}
```

### E-step: Σ_b and b_hat via Cholesky

The E-step requires inverting the r×r matrix `M = (1/σ²_u)I_r + ZtZ_range/σ²_e`. For r ≤ m (typically 30–50 for a 40-point grid), this is tractable via the standard Cholesky helpers:

```rust
// ZtZ_range = Z_range' Z_range (r×r, row-major)
let mut ztZ = vec![0.0_f64; r * r];
for a in 0..r {
    for b in a..r {
        let s: f64 = (0..n).map(|i| z_range[i*r+a] * z_range[i*r+b]).sum();
        ztZ[a*r+b] = s;
        ztZ[b*r+a] = s;
    }
}
// M = ZtZ_range/sigma2_e + (1/sigma2_u)*I_r
let mut big_m = vec![0.0_f64; r * r];
for i in 0..r*r { big_m[i] = ztZ[i] / sigma2_e; }
for i in 0..r { big_m[i*r+i] += 1.0 / sigma2_u; }

// Sigma_b = M^{-1} via Cholesky (r×r solve for each column of I_r)
// b_hat = Sigma_b * (Z_range' r_alpha) / sigma2_e
```

`tr(Sigma_b)` = sum of diagonal of `M^{-1}` = sum over j of `M^{-1}[:,j][j]` (one Cholesky solve per column, accumulate diagonal). For r ≤ 40 this is 40 solves each of size 40 — negligible.

`tr(Z_range Σ_b Z_range')` = `tr(Σ_b Z_range' Z_range)` = `tr(Σ_b ZtZ_range)` = sum_j of `(Σ_b ZtZ_range)_{jj}`. Computed as sum_a sum_j `Σ_b[j,a] * ZtZ_range[a,j]`.

### GLS update of α

If s > 0 (null space is non-empty), iterate α alongside variance components:
```
α = (Z_null' Σ^{-1} Z_null)^{-1} Z_null' Σ^{-1} y_c
Σ = σ²_e I + σ²_u Z_range Z_range'
```
Use the Woodbury identity: `Σ^{-1} = (1/σ²_e)(I − Z_range (σ²_e/σ²_u I + Z_range'Z_range)^{-1} Z_range')`. This avoids forming n×n Σ. For n ≤ a few thousand this is fast enough. [ASSUMED — standard Woodbury application; not verified against a specific source]

If s = 0 (all dimensions are penalized, e.g., Q = I_m for Ridge), there are no unpenalized fixed effects; α is empty and the GLS step is skipped.

---

## Reuse Assessment: `famm.rs` helpers

Reading `famm.rs` directly (VERIFIED: famm.rs:290–615):

| famm helper | Signature | Subject-grouped? | Reusable for PEER REML? |
|-------------|-----------|------------------|-------------------------|
| `shrinkage_weights` (line 291) | `(ss: &SubjectStructure, sigma2_u, sigma2_e) -> Vec<f64>` | YES — iterates `ss.counts` (per-subject) | NO — PEER has no subjects |
| `gls_update_gamma` (line 308) | `(cov, p, ss, weights, y, sigma2_e) -> Option<Vec<f64>>` | YES — accumulates per-subject sums | NO — PEER α-update uses Woodbury, not this |
| `reml_variance_update` (line 395) | `(residuals, ss, weights, sigma2_u, p) -> (f64, f64)` | YES — iterates `ss.obs[s]` | NO — but the M-step math (E[b²] + conditional_var) is the template |
| `fit_scalar_mixed_model` (line 438) | `(y, subject_map, n_subjects, covariates, p)` | YES — explicitly takes `subject_map` | NO — per CONTEXT decision |
| `estimate_variance_components` (line 567) | `(residuals, subject_map, n_subjects, n)` | YES — ANOVA moments estimator for subjects | NO — but the init idea (method-of-moments from residual variance) is portable |
| `SubjectStructure` (line 272) | struct | YES | NO |

**What to port (math pattern, not the function):**
- From `reml_variance_update`: the E-step identity `u_hat_s = w_s * mean_r_s` and `cond_var_s = sigma2_u * (1 - w_s)` maps to the PEER REML `b_hat = Σ_b Z_range' r_alpha / σ²_e` and `tr(Σ_b)`. The structure is analogous; the update equations are ported as math, not as a function call.
- From `estimate_variance_components`: the OLS residual variance init `sigma2_e = ss_within / df_within` maps to PEER's `y_var = var(y_c)` init.

**Confirmed:** `fit_scalar_mixed_model` must NOT be called. The REML EM in `peer.rs` is self-contained. [VERIFIED: famm.rs:438 — function takes `subject_map: &[usize]`, requires subject structure inapplicable to PEER's range-space random effect]

---

## Architecture Patterns

### System Architecture Diagram

```
peer(data, y, argvals, config)
         |
         v
[Input Validation]   (unchanged from Phase 66)
         |
         v
[Integration Weights + Design + Centering]   (unchanged)
  w = simpsons_weights(argvals)
  W[i,j] = data[i,j]*w[j]; W_c = center(W); y_c = center(y)
         |
         v
[Penalty Q]          (unchanged)
  build_q(m, &config.penalty)
         |
         v
[Compute WtW, wty]   (unchanged)
         |
         v
[Lambda Selection]
  match config.lambda:
  ┌── Fixed(lam) → lam, gcv=None, method=Fixed
  ├── Gcv → select_lambda_gcv_peer(wc, yc, wtw, wty, q, m, n)
  │           → (lam, gcv_score), method=Gcv
  └── Reml → select_lambda_reml_peer(wc, yc, q, m, n)
              → lam (=σ²_e/σ²_u), gcv=None, method=Reml
         |
         v
[Penalized Solve: A = WtW + lam*Q; beta = cholesky_solve(A, wty)]
         |
         v
[Effective df + Fitted Values]  (unchanged)
         |
         v
PeerResult { beta, intercept, w_bar, fitted_values, effective_df,
             lambda: lam, penalty_type, gcv, lambda_method }
```

### Recommended Project Structure

```
fdars-core/src/
└── peer.rs          # All Phase 67 additions (extends Phase 66)
    ├── LambdaChoice  # New public enum (Fixed, Gcv, Reml)
    ├── LambdaMethod  # New public enum (Fixed, Gcv, Reml)
    ├── PeerConfig    # Updated: lambda: LambdaChoice
    ├── PeerResult    # Updated: + gcv: Option<f64>, lambda_method: LambdaMethod
    ├── peer()        # Updated: dispatches to selectors; passes selected lam to solve
    ├── gcv_lambda_grid()          # Private: 40-pt log-spaced [1e-6, 1e4]
    ├── select_lambda_gcv_peer()   # Private: GCV grid search
    ├── select_lambda_reml_peer()  # Private: REML EM via eigendecomposition
    ├── build_q()                  # Unchanged
    ├── compute_peer_trace_hat()   # Unchanged (reused by GCV)
    └── #[cfg(test)] mod tests     # Existing + new GCV/REML tests
```

### Anti-Patterns to Avoid

- **Calling `famm::fit_scalar_mixed_model` from PEER REML:** It expects `subject_map: &[usize]` (per-subject grouping) — inapplicable to the range-space random effect. The CONTEXT decision is explicit: do NOT call it. [VERIFIED: famm.rs:438]
- **REML EM without null-space guard:** When Q has a large null space (e.g., Q = D'D, which has a 2-dim null space for a 2nd-difference operator), s=2 and the unpenalized α must be estimated by GLS or OLS. Failing to account for α inflates the residuals and biases σ²_e.
- **tr(H) ≥ n in GCV denominator:** Guard `if denom <= 0.0 { continue; }` before computing the GCV score. `compute_peer_trace_hat` clamps at n (VERIFIED: peer.rs:339), but floating-point near-equality means `n − tr(H)` can be a tiny positive or negative number.
- **Forgetting wc in GCV RSS:** The GCV RSS is computed against the centered design `W_c`, not the raw data. Passing raw data[(i,j)] into the RSS loop gives wrong RSS.
- **Using `from_column_slice` instead of `from_row_slice` for symmetric Q:** Q is built row-major in `build_q` (VERIFIED: peer.rs:290–317). nalgebra `DMatrix::from_row_slice(m, m, &q)` correctly interprets a row-major flat slice. For symmetric Q the transpose is equivalent, but using the right constructor avoids confusion.
- **Non-determinism from uninitialized variance:** Both selectors must produce the same result on two calls with identical input. REML init uses `y_var = var(y_c)` (deterministic); GCV iterates a fixed sorted grid (deterministic). No RNG anywhere.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Eigendecomposition of Q | Power iteration, hand-rolled QR | `nalgebra::DMatrix::from_row_slice(m,m,q).symmetric_eigen()` | Already used in fpca_variants.rs:473–484; handles symmetric PSD matrices |
| r×r Cholesky solve (E-step Σ_b) | Gaussian elimination from scratch | `linalg::cholesky_factor` + `cholesky_forward_back` | Already `pub(crate)`; handles the r×r size |
| GCV trace-of-hat | Explicit H = WA^{-1}W' then diagonal sum | `compute_peer_trace_hat` (Phase 66) | O(m²) column-solve trick already implemented |
| GCV grid | Manual 9-point coarse grid | 40-pt log-spaced `gcv_lambda_grid()` | Better coverage; still fully deterministic |

---

## Common Pitfalls

### Pitfall 1: REML null-space collapse (Q = identity / Ridge)

**What goes wrong:** For `PeerPenalty::Ridge`, Q = I_m, so ALL eigenvalues = 1.0 (above tolerance) — the null space is empty (s = 0), the range space is all m dimensions (r = m). The EM must handle s = 0 (no α to estimate, no GLS step). If the GLS step runs on empty Z_null, it attempts a 0×0 solve.
**Why it happens:** No check for s = 0 before the GLS update.
**How to avoid:** Wrap the GLS α-update in `if s > 0 { ... }`. When s = 0, α = [] and the residual for the M-step is just `y_c − Z_range b_hat`.
**Warning signs:** Panic or dimension mismatch in the GLS update for Ridge penalty.

### Pitfall 2: REML range-space collapse (Q = zero matrix)

**What goes wrong:** If Q is all-zeros (degenerate Decree), all eigenvalues = 0, the range space is empty (r = 0). The random-effect model is undefined; σ²_u is meaningless.
**Why it happens:** Caller passes a zero Decree Q.
**How to avoid:** Guard at the top of `select_lambda_reml_peer`: if `r == 0`, return a small fallback λ (e.g., 1e-4) and record `lambda_method: LambdaMethod::Reml` (or warn in debug mode). Document this in the docstring.
**Warning signs:** Division-by-zero in σ²_u update when r = 0.

### Pitfall 3: GCV favors λ = 0 (under-smoothing) on noisy data

**What goes wrong:** For very noisy data (low SNR), the GCV surface may be flat or monotonically decreasing on the grid, favoring the smallest λ. The smallest grid value is 1e-6, which gives near-OLS fit.
**Why it happens:** GCV without a lower-bound signal criterion can pick underfitting-regime λ.
**How to avoid:** No special fix — this is mathematically correct behavior (GCV will pick the best among the grid candidates). Known-answer tests use high-SNR data where GCV is expected to pick a sensible λ. Document the behavior.
**Warning signs:** In tests, assert `gcv_lambda > 1e-10` (not at the absolute grid minimum) on the SNR fixture.

### Pitfall 4: REML σ²_u → 0 before convergence

**What goes wrong:** If the data has very low between-"group" variance (in the range-space sense), σ²_u collapses to 0 before the EM converges, causing λ = σ²_e/σ²_u → ∞ and extreme over-smoothing.
**Why it happens:** Unclamped M-step update.
**How to avoid:** After each M-step, clamp: `sigma2_u = sigma2_u.max(1e-12)`. The minimum clamp corresponds to λ ≤ σ²_e/1e-12 — in practice, if σ²_e ≈ 1, λ ≤ 1e12, which is still finite and the β → 0 plateau is detectable by β recovery tests.
**Warning signs:** `sigma2_u` at or below 1e-12 after first iteration; λ extremely large.

### Pitfall 5: nalgebra eigenvector ordering

**What goes wrong:** `symmetric_eigen()` does not guarantee ascending-eigenvalue order. If eigenvalues are not sorted before partitioning null/range, the null-space index set is wrong.
**Why it happens:** Forgetting the explicit sort (see fpca_variants.rs:485–488 where descending sort IS applied explicitly).
**How to avoid:** Always sort `idx_sorted` by ascending eigenvalue before computing `tol` and partitioning.
**Warning signs:** Incorrect REML λ; β(t) not recovered; REML and GCV β disagree far beyond tolerance.

### Pitfall 6: serde feature — LambdaChoice with Decree variant

**What goes wrong:** `LambdaChoice::Fixed(f64)` is serde-trivial. `LambdaChoice` itself is fine. But if `PeerPenalty::Decree(Vec<f64>, usize)` already works under `serde` (it does — Vec<f64> is serializable), then adding `LambdaChoice` with serde is straightforward.
**Note:** The pre-existing serde build break (`ShapeletTransformClassifier` / `ClassifFit`) is unrelated. PEER serde derives are on plain types (f64, Vec<f64>, enum variants) — do NOT attempt to build or gate on `--features serde` to verify this phase. [VERIFIED: MEMORY.md serde-feature-build-broken-shapelet-classiffit.md]

---

## Code Examples

### GCV grid construction

```rust
// Source: derived from function_on_scalar.rs select_lambda_gcv pattern [VERIFIED: lines 371-397]
fn gcv_lambda_grid() -> Vec<f64> {
    // 40 points log-spaced on [1e-6, 1e4]
    (0..40)
        .map(|i| 10.0_f64.powf(-6.0 + 10.0 * i as f64 / 39.0))
        .collect()
}
```

### GCV score formula (PEER-specific)

```rust
// GCV = n * RSS / (n - tr(H))^2
// This matches CONTEXT.md §GCV grid search exactly.
// Note: FOSR uses GCV/(nm) which is equivalent per-observation; PEER has scalar y so n_rhs=1.
let rss: f64 = (0..n)
    .map(|i| {
        let yhat: f64 = (0..m).map(|j| wc[(i, j)] * beta[j]).sum();
        (yc[i] - yhat).powi(2)
    })
    .sum();
let trh = compute_peer_trace_hat(wtw, q, lam, m, n);
let denom = (n as f64 - trh).max(1e-10);
let gcv = n as f64 * rss / (denom * denom);
```

### nalgebra symmetric_eigen (PEER REML eigendecompose Q)

```rust
// Source: pattern from fpca_variants.rs:473-484 [VERIFIED]
use nalgebra::DMatrix;

// q is m×m row-major flat Vec<f64> (symmetric, so row-major == column-major)
let q_mat = DMatrix::from_row_slice(m, m, &q);
let eigen = q_mat.symmetric_eigen();

// Sort ascending by eigenvalue to identify null space
let mut idx_sorted: Vec<usize> = (0..m).collect();
idx_sorted.sort_by(|&a, &b| {
    eigen.eigenvalues[a]
        .partial_cmp(&eigen.eigenvalues[b])
        .unwrap_or(std::cmp::Ordering::Equal)
});

let max_ev = idx_sorted.iter()
    .map(|&i| eigen.eigenvalues[i].abs())
    .fold(0.0_f64, f64::max);
let tol = 1e-8 * max_ev.max(1.0);

let null_idx: Vec<usize> = idx_sorted.iter().copied()
    .filter(|&i| eigen.eigenvalues[i].abs() < tol).collect();
let range_idx: Vec<usize> = idx_sorted.iter().copied()
    .filter(|&i| eigen.eigenvalues[i].abs() >= tol).collect();
```

### Updated peer() dispatch sketch

```rust
// In peer() after WtW / wty are computed:
let (lambda, gcv_score, lambda_method) = match &config.lambda {
    LambdaChoice::Fixed(lam) => (*lam, None, LambdaMethod::Fixed),
    LambdaChoice::Gcv => {
        let (lam, gcv_val) = select_lambda_gcv_peer(&wc, &yc, &wtw, &wty, &q, m, n);
        (lam, Some(gcv_val), LambdaMethod::Gcv)
    }
    LambdaChoice::Reml => {
        let lam = select_lambda_reml_peer(&wc, &yc, &q, m, n);
        (lam, None, LambdaMethod::Reml)
    }
};
// ... proceed with cholesky_solve using `lambda` ...
Ok(PeerResult {
    // ... existing fields ...
    lambda,
    gcv: gcv_score,
    lambda_method,
    // ...
})
```

---

## Determinism and Known-Answer Tests

### Determinism properties

- **GCV:** Deterministic by construction. Grid is a pure function of constants (no RNG). For identical (WtW, wty, Q), two calls produce identical `lam`. Ties broken by first occurrence in grid order (smallest lam wins on tie — smaller index in the 40-pt array). Document this in the docstring.
- **REML:** Deterministic by construction. Init is `y_var = var(y_c)` (pure function of `yc`). EM iterations are pure floating-point arithmetic. For identical (wc, yc, Q), two calls produce identical `lam`.

### Synthetic SNR fixture for GCV+REML tests

Reuse `make_fixture()` from Phase 66 tests (n=200, m=40, true β(t)=sin(πt), deterministic hash-unit design). [VERIFIED: peer.rs:371–391]

This fixture has high SNR (noise amplitude = 0.005 × |true response|). Both GCV and REML should:
1. Pick a non-degenerate λ (in range [1e-5, 1e2] is reasonable for n=200, m=40).
2. Recover β(t) with max absolute error < 0.1 (same gate as Phase 66 tests).

Additional fixture for REML convergence validation: add noise at 0.1 level so SNR is moderate and REML variance components are non-trivial (σ²_u, σ²_e both > 1e-10 at convergence).

### Key test assertions

| Test | Assertion |
|------|-----------|
| GCV determinism | Call `peer` twice with `LambdaChoice::Gcv`; assert `result1.lambda == result2.lambda` (bit-exact) |
| REML determinism | Call `peer` twice with `LambdaChoice::Reml`; assert `result1.lambda == result2.lambda` |
| GCV picks non-degenerate λ | `assert!(result.lambda >= 1e-10 && result.lambda <= 1e6)` on SNR fixture |
| REML picks positive λ | `assert!(result.lambda > 0.0 && result.lambda.is_finite())` on SNR fixture |
| REML/GCV β agreement | On SNR fixture, max |β_gcv(t) − β_reml(t)| < 0.2 (documented tolerance; loose because selectors may choose different λ) |
| Both recover known β | max |β_method(t) − true_β(t)| < 0.15 on SNR fixture for both methods |
| Fixed λ verbatim | `result.lambda == 1e-4` when `LambdaChoice::Fixed(1e-4)` (already tested by Phase 66 tests after mechanical update) |
| GCV score recorded | `assert!(result.gcv.is_some())` when Gcv; `assert!(result.gcv.is_none())` when Fixed or Reml |
| lambda_method marker | `assert_eq!(result.lambda_method, LambdaMethod::Gcv)` etc. |
| Variance components non-negative | Assert `sigma2_u > 0` and `sigma2_e > 0` at EM exit (indirectly: λ > 0 and finite, which holds iff both are > 0) |

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Fixed λ (Phase 66) | Automatic via GCV or REML (Phase 67) | Matches refund's default REML; GCV chosen as Phase 67 default for reproducibility |
| `fosr` 9-pt coarse GCV grid | 40-pt fine log-spaced grid [1e-6,1e4] | Better λ coverage; still O(40 × m²) = negligible for m ≤ 200 |
| `famm` REML EM for subject-grouped effects | Self-contained REML EM for range-space smoothing effect | Additive; famm is unchanged |

**refund comparison:** refund's `peer()` defaults to REML via mgcv's `gam()` with `method="REML"` [ASSUMED — training knowledge; mgcv details not verified this session]. The CONTEXT.md decision is to use GCV as the Phase 67 default for reproducibility, with REML as a first-class alternative. This is documented as a deliberate divergence.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | REML EM update equations for σ²_u and σ²_e follow the standard smoothing-spline mixed model (Wahba 1985, Ruppert et al. 2003) with the forms given in §REML Path | REML Path §EM algorithm | If the EM for the range-space random effect has additional correction terms (e.g., restricted likelihood correction for the fixed-effect null space), the selected λ may be biased. Test: REML/GCV β agreement within 0.2 tolerance catches gross errors. |
| A2 | nalgebra `symmetric_eigen()` returns correct eigenvalues for PSD matrices of size ≤ 40×40 without numerical issues | REML Path §Eigendecomposition | Low risk for small m; nalgebra is production-quality. |
| A3 | refund's `peer()` defaults to REML/mgcv; the GCV-vs-REML tolerance 0.2 on the test fixture is achievable | State of the Art, Test plan | If refund REML behavior is substantially different, the tolerance may need loosening. Adjust tolerance after implementation if needed. |
| A4 | The GLS α-update via Woodbury is numerically stable for small s (null-space dimension ≤ 5 for typical penalty families) | REML Path §GLS update | If s is large and Z_null is near-rank-deficient, the GLS solve can be ill-conditioned. Regularize with a small ridge (1e-10 on diagonal) as in `gls_update_gamma` (famm.rs:343). |
| A5 | The 40-pt log-spaced grid [1e-6, 1e4] covers the optimal λ on the test fixture | GCV Path §grid | If the true optimal λ lies outside this range, GCV returns the boundary value. The β recovery test catches this: if β error > 0.15, the grid boundary was reached. |

---

## Open Questions

1. **REML convergence speed on 40-point m**
   - What we know: The EM is on an r×r matrix (r ≤ m = 40). Each iteration is cheap (Cholesky on r×r + a few matrix products). 50–100 iterations is the famm.rs cap (VERIFIED: famm.rs:463 `for _iter in 0..50`).
   - What's unclear: Whether 50 iterations is sufficient for the PEER REML EM (which has the additional α-update step not present in the per-component scalar mixed model).
   - Recommendation: Use 100 iterations as the cap (Claude's discretion). Test convergence on the SNR fixture by checking whether 50 vs 100 iterations gives the same λ to 4 decimal places.

2. **Whether to expose `gcv_at_each_grid_point` in PeerResult**
   - What we know: CONTEXT.md only requires `gcv: Option<f64>` (GCV score at the selected λ).
   - What's unclear: Whether the planner will want the full GCV curve for diagnostics in Phase 68.
   - Recommendation: Keep only `gcv: Option<f64>` for Phase 67 per CONTEXT decision; Phase 68 can add a `gcv_curve` field if needed.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | none — inline module tests per crate convention |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel peer` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command |
|--------|----------|-----------|-------------------|
| PER-03 | GCV picks argmin deterministically (two runs → identical λ) | unit (determinism) | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_gcv_deterministic` |
| PER-03 | GCV picks non-degenerate λ on SNR fixture | unit (known-answer) | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_gcv_recovers_beta` |
| PER-03 | REML picks positive finite λ on SNR fixture | unit (structural) | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_reml_lambda_positive` |
| PER-03 | REML picks same λ on two runs (determinism) | unit (determinism) | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_reml_deterministic` |
| PER-03 | REML and GCV β(t) agree within documented tolerance on SNR fixture | unit (known-answer) | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_reml_gcv_beta_agreement` |
| PER-03 | Both REML and GCV recover known β(t) within 0.15 max abs error | unit (known-answer) | included in above tests |
| PER-03 | Fixed(λ) used verbatim; no search; recorded in result | unit (API contract) | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_fixed_verbatim` |
| PER-03 | `gcv` field is `Some` when GCV ran, `None` when Fixed or REML | unit (API contract) | included in gcv and reml tests |
| PER-03 | `lambda_method` marker correct for each path | unit (API contract) | included in all three selector tests |
| PER-03 | Variance components non-negative (σ²_u > 0, σ²_e > 0) indirectly via λ > 0 | unit (structural) | included in `test_peer_reml_lambda_positive` |
| (regression) | All Phase 66 tests still pass after mechanical PeerConfig update | regression | `cargo test -p fdars-core --features linalg,parallel peer` |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg,parallel peer -- --nocapture`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite + clippy + fmt before `/gsd-verify-work`

### Wave 0 Gaps

All changes are additions to the existing `fdars-core/src/peer.rs`:

- [ ] Add `LambdaChoice` enum, `LambdaMethod` enum
- [ ] Update `PeerConfig.lambda: f64` → `lambda: LambdaChoice`
- [ ] Update `PeerResult`: add `gcv: Option<f64>`, `lambda_method: LambdaMethod`
- [ ] Add `gcv_lambda_grid()` private fn
- [ ] Add `select_lambda_gcv_peer()` private fn
- [ ] Add `select_lambda_reml_peer()` private fn (with eigendecompose + EM)
- [ ] Update `peer()` dispatch to call selectors
- [ ] Mechanically update all Phase 66 tests: `lambda: <f64>` → `lambda: LambdaChoice::Fixed(<f64>)`
- [ ] Add new tests for GCV determinism, REML determinism, β recovery, β agreement, Fixed verbatim

---

## Verify Commands

```bash
# Run only peer module tests (fast, ~seconds)
cargo test -p fdars-core --features linalg,parallel peer

# Full suite (required before /gsd-verify-work)
cargo test -p fdars-core --features linalg,parallel

# Clippy (CI gate — must use --all-targets to catch test code warnings)
cargo clippy --all-targets --features linalg,parallel -- -D warnings

# Format check
cargo fmt -- --check

# Commit without pre-commit hook (slow hook / /tmp pressure)
git commit --no-verify -m "..."
```

Source: MEMORY.md [`ci-clippy-all-targets-gate.md`, `noverify-commits-leave-fmt-drift.md`]. [VERIFIED: project MEMORY.md]

---

## Security Domain

`security_enforcement` absent — treated as enabled. PEER is pure numerical computation with no I/O, no network, no authentication. ASVS categories:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (unchanged from Phase 66) | `FdarError::InvalidDimension` / `InvalidParameter` at entry |
| V6 Cryptography | no | — |

No new threat patterns introduced by λ selection (grid search and EM are pure arithmetic on already-validated inputs).

---

## Environment Availability

No external dependencies beyond the Rust toolchain and nalgebra (already in Cargo.toml). REML path adds a nalgebra `DMatrix` construction — nalgebra is already in scope from fpca_variants.rs. [VERIFIED: Cargo.toml via CLAUDE.md + fpca_variants.rs:25 `use nalgebra::DMatrix;`]

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| nalgebra | REML eigendecompose Q | ✓ | 0.33 [VERIFIED: Cargo.toml] | — |
| linalg helpers (pub(crate)) | GCV inner solve + tr(H) | ✓ | in-crate | — |

---

## Sources

### Primary (HIGH confidence — files read this session)

- `fdars-core/src/peer.rs:1–773` — Full Phase 66 implementation: PeerConfig, PeerResult, PeerPenalty, peer(), build_q(), compute_peer_trace_hat(), all inline tests [VERIFIED]
- `fdars-core/src/function_on_scalar.rs:99-116` — `penalty_matrix` [VERIFIED]
- `fdars-core/src/function_on_scalar.rs:121-149` — `penalized_solve` [VERIFIED]
- `fdars-core/src/function_on_scalar.rs:171-178` — `compute_fosr_gcv` [VERIFIED]
- `fdars-core/src/function_on_scalar.rs:371-397` — `select_lambda_gcv` [VERIFIED]
- `fdars-core/src/function_on_scalar.rs:401-419` — `compute_trace_hat` [VERIFIED]
- `fdars-core/src/famm.rs:272-615` — SubjectStructure, shrinkage_weights, gls_update_gamma, reml_variance_update, fit_scalar_mixed_model, estimate_variance_components [VERIFIED]
- `fdars-core/src/fpca_variants.rs:473-488` — nalgebra symmetric_eigen usage + sort pattern [VERIFIED]
- `fdars-core/src/linalg.rs:85-151` — cholesky_factor, cholesky_forward_back, cholesky_solve, compute_xtx [VERIFIED]

### Tertiary (LOW confidence — training knowledge, not verified this session)

- Smoothing-REML EM update equations (Wahba 1985, Ruppert/Wand/Carroll 2003 form for spline-as-mixed-model): E-step conditional expectation, M-step variance updates [ASSUMED: A1]
- refund `peer()` defaults to REML via mgcv [ASSUMED: A3]
- Woodbury identity for GLS α-update via Σ^{-1} [ASSUMED: A4]

---

## Metadata

**Confidence breakdown:**
- API evolution (PeerConfig, PeerResult, enum shapes): HIGH — derived from verified peer.rs source
- GCV recipe: HIGH — derived from verified function_on_scalar.rs + peer.rs helpers
- REML math (EM equations): LOW/ASSUMED — standard smoothing mixed model; not verified against a primary source this session
- famm reuse assessment: HIGH — famm.rs read directly; confirmed subject-grouped structure incompatible with PEER
- Test plan: HIGH — based on verified crate patterns

**Research date:** 2026-09-04
**Valid until:** 2026-11-04 (stable Rust crate internals; 60 days)
