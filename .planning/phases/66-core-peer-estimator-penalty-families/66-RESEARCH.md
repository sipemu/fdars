# Phase 66: Core PEER Estimator & Penalty Families — Research

**Researched:** 2026-09-04
**Domain:** PEER (Partially Empirical Eigenvectors for Regression) — structured-penalty scalar-on-function regression
**Confidence:** HIGH (source files read directly this session; PEER algorithm from training knowledge, [ASSUMED] where not cross-verified)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- New code lives in a single top-level file `src/peer.rs` (may split into `peer/` submodule later).
- Public estimator signature: `peer(data, y, argvals, &config)` — functional data first, response second, optional evaluation points, then config.
- Configuration in `PeerConfig` builder struct (matches `ElasticConfig`/`GmmClusterConfig` pattern), with serde behind the `serde` feature per crate convention.
- `PeerResult` carries: β(t) (`beta`), `intercept`, `fitted_values`, `effective_df`, `lambda` used, and the selected `penalty_type`. Derives `Debug, Clone, PartialEq`; `#[non_exhaustive]`.
- Penalty families as `PeerPenalty` enum: `Ridge` (identity), `Difference { order }` (2nd-difference), `Decree(Q)` (caller-supplied).
- Roughness penalty reuses existing `penalty_matrix` builder from `function_on_scalar.rs` — not hand-rolled.
- "Decree" Q is a caller-supplied raw matrix (`Vec<f64>` + dimensions).
- Default penalty is `Difference { order: 2 }`, matching refund's default.
- β(t) represented pointwise on the `argvals` grid (Q is p×p on the grid).
- Null-space/range-space split via eigendecomposition/SVD of Q via nalgebra.
- Linear solve reuses penalized normal-equations + Cholesky pattern (`penalized_solve` in `function_on_scalar.rs`). Claude's discretion whether to make that helper crate-visible or replicate the pattern locally.
- Intercept by centering y and the design; recovered as ȳ (matches `fregre_lm`).
- Single `lambda` field in `PeerConfig` with default `1.0`. Auto-selection deferred.
- Wrong-dimension Q (or invalid penalty input) returns `FdarError::InvalidDimension { parameter, expected, actual }` — never panics.
- Effective df via trace of the smoother/hat matrix.
- Integration uses Simpson's weights (`helpers::simpsons_weights`).
- Register `pub mod peer;` in `src/lib.rs`. Crate-root + prelude re-exports deferred to Phase 68.

### Claude's Discretion

- Whether to expose/replicate `penalized_solve`.
- Exact internal helper factoring within `peer.rs`.
- Precise default numeric tolerances for the null/range decomposition.

### Deferred Ideas (OUT OF SCOPE)

- Automatic λ selection (GCV/REML) — Phase 67.
- Longitudinal `lpeer`, out-of-sample `predict`, crate-root/prelude exports, and the end-to-end module doctest — Phase 68.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| PER-01 | Public `peer(...)` estimator returning β(t), intercept, fitted values, df/selection diagnostics via the partially-empirical-eigenvector decomposition | Full algorithm derivation in §PEER Method; reuse map in §Reuse Map |
| PER-02 | Three penalty families via `PeerPenalty` enum: Ridge, Difference, Decree | §Penalty Families; exact builder signatures confirmed in §Reuse Map |
</phase_requirements>

---

## Summary

PEER (Partially Empirical Eigenvectors for Regression), due to Randolph, Reiss et al. (2012), is a scalar-on-function regression method that distinguishes itself from plain FPCR by partitioning the function space into the null space and range space of a structured penalty operator Q. The null space is estimated without shrinkage; the range space is penalized by λ. This lets the caller inject a-priori signal structure (the "decree" Q) rather than letting all regularization be data-driven.

The implementation slots neatly into the fdars-core infrastructure. The key linear algebra is: form the integrated design W (n×p, where W[i,j] = ∫ X_i(t) ψ_j(t) dt using Simpson's weights and the argvals grid), center y and W, build the penalized normal equations (W'W + λQ)β = W'y_c, and solve via Cholesky. The penalty Q is constructed from three families selectable via `PeerPenalty`: identity (Ridge), second-difference D'D (Difference, reusing `penalty_matrix` from `function_on_scalar.rs`), or a caller-supplied matrix (Decree). For the PEER decomposition proper, Q is eigendecomposed via nalgebra's `.symmetric_eigen()` (already used in `fpca_variants.rs`), the null space (eigenvalues below tolerance) is left unpenalized, and the range space is penalized.

**Primary recommendation:** Implement in a single `src/peer.rs` following the `fosr` pattern from `function_on_scalar.rs`: build W, center y, call `penalized_solve`-style Cholesky solve with the appropriate Q, recover β(t) on the argvals grid, compute trace-of-hat-matrix as effective df, assemble `PeerResult`. The null/range decomposition is an optional diagnostic layer on top — the core fit is just penalized least squares with Q as the penalty matrix.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| PEER estimator fit | Algorithm module (`src/peer.rs`) | Linalg helpers (`linalg.rs`) | All computation is pure Rust; no I/O |
| Penalty matrix construction | `function_on_scalar.rs::penalty_matrix` (reused) | `peer.rs` (Ridge: identity; Decree: caller) | Reuse-first per project constraint |
| Cholesky penalized solve | `linalg.rs` (`cholesky_factor`, `cholesky_forward_back`) | Pattern from `function_on_scalar.rs::penalized_solve` | Already `pub(crate)` — accessible from `peer.rs` in the same crate |
| Null/range eigendecomposition | nalgebra `DMatrix::symmetric_eigen()` | — | Already used in `fpca_variants.rs` |
| Integration weights | `helpers::simpsons_weights` | — | Crate-wide convention |
| Module registration | `src/lib.rs` | — | Add `pub mod peer;` |

---

## Standard Stack

### Core

| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| nalgebra | 0.33 [VERIFIED: fdars-core/Cargo.toml] | Matrix ops, SVD, `symmetric_eigen` | Crate-wide linear algebra primitive |
| (no new dep) | — | — | Constraint: no new crate dependency |

No external packages are added. All linear algebra uses existing crate infrastructure.

### Supporting (in-crate reuse)

| Module / Function | Location | Purpose | When to Use |
|-------------------|----------|---------|-------------|
| `penalty_matrix(m)` | `function_on_scalar.rs:99-116` | Build p×p D'D second-difference penalty | `PeerPenalty::Difference` |
| `cholesky_factor(a, p)` | `linalg.rs:85-108` | Cholesky factorization | penalized solve |
| `cholesky_forward_back(l, b, p)` | `linalg.rs:113-128` | Forward/backward substitution | penalized solve |
| `cholesky_solve(a, b, p)` | `linalg.rs:131-134` | Convenience Cholesky solve | optional |
| `simpsons_weights(argvals)` | `helpers.rs:76-105` | Simpson's 1/3 integration weights | functional inner product ∫X_i(t)ψ_j(t)dt |
| `FdMatrix::zeros(nrows, ncols)` | `matrix.rs:84-90` | Zero-init matrix | all intermediate matrices |
| `FdMatrix::from_column_major(...)` | `matrix.rs:50-63` | Construct from flat vec | result assembly |
| `DMatrix::from_fn(...)` .symmetric_eigen() | nalgebra (in-crate) | Eigendecompose Q | null/range split |

**Installation:** No changes to `Cargo.toml`. No new dependencies.

---

## Package Legitimacy Audit

No new packages are introduced. This section is not applicable.

---

## PEER Method — Full Algorithm

### 1. Setup and centering

Given:
- `data`: n×m `FdMatrix` of functional predictors X_i(t), rows = observations, columns = evaluation points [VERIFIED: fdars-core/src/matrix.rs:9-44]
- `y`: scalar response vector (length n)
- `argvals`: evaluation grid (length m = p in PEER notation)
- `lambda`: smoothing parameter
- `penalty_type`: `PeerPenalty` enum

Step 1 — compute Simpson's weights w = simpsons_weights(argvals) [VERIFIED: fdars-core/src/helpers.rs:76-105].

Step 2 — form the integrated design matrix W (n×m): W[i,j] = X_i(argvals[j]) · w[j]. This integrates each curve against point-mass basis functions at the grid (the "data matrix" in PEER). In refund's formulation this is the raw functional data scaled by quadrature weights. [ASSUMED — PEER paper formulation; refund source not read]

Step 3 — center:
```
y_bar = mean(y)
y_c   = y - y_bar           (length n)
W_bar = column mean of W    (length m)
W_c   = W - 1·W_bar'       (n×m)
```
Intercept recovered as y_bar at the end (matches `fregre_lm` pattern [VERIFIED: fdars-core/src/scalar_on_function/fregre_lm.rs:79-120]).

### 2. Penalty matrix Q (p×p, where p = m = argvals length)

Three cases via `PeerPenalty`:

**Ridge:** Q = I_p (identity matrix, p×p). Uniform penalization.

**Difference { order }:** Q = D'D where D is the `order`-th difference operator. Built by `penalty_matrix(m)` [VERIFIED: fdars-core/src/function_on_scalar.rs:99-116]:
```rust
// Verbatim from function_on_scalar.rs:99-116:
pub(crate) fn penalty_matrix(m: usize) -> Vec<f64> {
    if m < 3 {
        return vec![0.0; m * m];
    }
    // D is (m-2)×m second-difference operator
    // D'D is m×m symmetric banded matrix
    let mut dtd = vec![0.0; m * m];
    for i in 0..m - 2 {
        // D[i,:] = [0..0, 1, -2, 1, 0..0] at positions i, i+1, i+2
        let coeffs = [(i, 1.0), (i + 1, -2.0), (i + 2, 1.0)];
        for &(r, cr) in &coeffs {
            for &(c, cc) in &coeffs {
                dtd[r * m + c] += cr * cc;
            }
        }
    }
    dtd
}
```
Note: `penalty_matrix` is `pub(crate)` and accessible from `peer.rs` in the same crate. It hardcodes order=2 (second differences). For a `Difference { order }` variant with order≠2, a general-order builder must be written or order constrained to 2. The CONTEXT.md default is order 2 and the refund baseline uses order 2; supporting other orders is Claude's discretion.

**Decree(Q_raw):** Caller supplies Q as a `Vec<f64>` (length m×m, row-major or column-major — must be documented) plus dimensions. Must be validated: length == m*m, else `FdarError::InvalidDimension`. Q should be symmetric PSD for numerical stability; optionally warn/check.

### 3. PEER penalized normal equations

The core estimator solves:
```
(W_c'W_c + λ·Q) β = W_c' y_c
```
This is exactly the shape handled by `penalized_solve` [VERIFIED: fdars-core/src/function_on_scalar.rs:121-149]:
```rust
// Verbatim from function_on_scalar.rs:121-149:
fn penalized_solve(
    xtx: &[f64],
    xty: &FdMatrix,
    penalty: &[f64],
    lambda: f64,
) -> Result<FdMatrix, FdarError> {
    let p = xty.nrows();
    let m = xty.ncols();
    // Build (X'X + λP)
    let mut a = vec![0.0; p * p];
    for i in 0..p * p {
        a[i] = xtx[i] + lambda * penalty[i];
    }
    // Cholesky factor
    let l = cholesky_factor(&a, p)?;
    // Solve for each grid point
    let mut beta = FdMatrix::zeros(p, m);
    for t in 0..m {
        let b: Vec<f64> = (0..p).map(|j| xty[(j, t)]).collect();
        let x = cholesky_forward_back(&l, &b, p);
        for j in 0..p {
            beta[(j, t)] = x[j];
        }
    }
    Ok(beta)
}
```
`penalized_solve` is `fn` (private, no `pub`). Claude must either:
- (Option A) Widen to `pub(crate)` with a one-line additive change in `function_on_scalar.rs` — zero API break, preferred.
- (Option B) Replicate the pattern locally in `peer.rs` (4 lines of math, trivial duplication).

The PEER solve is a special case where `xty` is a column vector (n_rhs = 1) because we solve for β(t) as a single vector, not a matrix over many grid points as in FOSR. The formula W_c'y_c gives an m-vector (right-hand side), and W_c'W_c gives an m×m matrix. So the call reduces to solving one m×m system once — simpler than FOSR's per-grid-point loop.

Concretely, in `peer.rs`:
```rust
// Compute WtW = W_c' W_c (m×m, row-major)
let mut wtw = vec![0.0; m * m];
for j in 0..m {
    for k in j..m {
        let s: f64 = (0..n).map(|i| wc[(i,j)] * wc[(i,k)]).sum();
        wtw[j*m + k] = s;
        wtw[k*m + j] = s;
    }
}

// Compute wty = W_c' y_c (length m)
let mut wty = vec![0.0; m];
for j in 0..m {
    wty[j] = (0..n).map(|i| wc[(i,j)] * yc[i]).sum();
}

// Build A = WtW + lambda * Q (m×m)
let mut a = vec![0.0; m * m];
for i in 0..m*m {
    a[i] = wtw[i] + lambda * q[i];
}

// Solve A beta = wty via Cholesky
let beta = cholesky_solve(&a, &wty, m)?;
```
`cholesky_solve` is `pub(crate)` [VERIFIED: fdars-core/src/linalg.rs:131-134].

### 4. PEER null/range-space split (the distinguishing decomposition)

Eigendecompose Q:
```
Q = V Λ V'
```
where Λ = diag(λ_1, ..., λ_p) sorted ascending. Eigenvalues near zero (|λ_i| < tol) correspond to the null space of Q (unpenalized directions); the rest are the range space.

In Rust via nalgebra (already used in `fpca_variants.rs:484` [VERIFIED: fdars-core/src/fpca_variants.rs:484]):
```rust
use nalgebra::DMatrix;
let q_mat = DMatrix::from_row_slice(m, m, &q_vec); // or from_column_slice
let eigen = q_mat.symmetric_eigen();
// eigen.eigenvalues: DVector<f64> (ascending by nalgebra convention)
// eigen.eigenvectors: DMatrix<f64> columns = eigenvectors

let tol = 1e-8 * eigen.eigenvalues.iter().copied().fold(0.0_f64, f64::max);
let null_idx: Vec<usize> = (0..m).filter(|&i| eigen.eigenvalues[i].abs() < tol).collect();
let range_idx: Vec<usize> = (0..m).filter(|&i| eigen.eigenvalues[i].abs() >= tol).collect();
```

The null-space components are estimated by OLS (unpenalized); the range-space components are penalized. This is the PEER "partially empirical eigenvector" representation [ASSUMED — from PEER paper; refund source not read].

For Phase 66 (fixed λ, no auto-selection), the practical effect: the above eigendecomposition is used to compute `effective_df` and to enable the `Decree` penalty variant to represent a structured partition. The actual β(t) estimate is still obtained by the direct Cholesky solve above — the decomposition just shapes the penalty Q that enters it.

### 5. Effective degrees of freedom

```
df_eff = tr(H)
where H = W_c (W_c'W_c + λQ)^{-1} W_c'
         = tr((W_c'W_c + λQ)^{-1} W_c'W_c)
```
Reuse `compute_trace_hat` pattern from `function_on_scalar.rs:401-419` [VERIFIED: fdars-core/src/function_on_scalar.rs:401-419]:
```rust
fn compute_trace_hat(xtx: &[f64], penalty: &[f64], lambda: f64, p: usize, n: usize) -> f64 {
    let mut a = vec![0.0; p * p];
    for i in 0..p * p {
        a[i] = xtx[i] + lambda * penalty[i];
    }
    let Ok(l) = cholesky_factor(&a, p) else {
        return p as f64; // fallback
    };
    let mut trace = 0.0;
    for j in 0..p {
        let col: Vec<f64> = (0..p).map(|i| xtx[i * p + j]).collect();
        let z = cholesky_forward_back(&l, &col, p);
        trace += z[j];
    }
    trace.min(n as f64)
}
```
For `peer.rs`, this becomes `compute_trace_hat(&wtw, &q, lambda, m, n)`.

### 6. Fitted values and intercept recovery

```rust
let y_bar: f64 = y.iter().sum::<f64>() / n as f64;

// fitted[i] = y_bar + sum_j wc[(i,j)] * beta[j]
let mut fitted = vec![0.0_f64; n];
for i in 0..n {
    let mut sum = y_bar;
    for j in 0..m {
        sum += wc[(i,j)] * beta[j];
    }
    fitted[i] = sum;
}
```

---

## Reuse Map — Exact Signatures and Line Numbers

### `function_on_scalar.rs::penalty_matrix`

```rust
// VERIFIED: fdars-core/src/function_on_scalar.rs:99-116
pub(crate) fn penalty_matrix(m: usize) -> Vec<f64>
```
Builds p×p D'D second-difference operator as flat row-major `Vec<f64>`. Returns `vec![0.0; m*m]` if `m < 3`. Callable from `peer.rs` without modification.

### `function_on_scalar.rs::penalized_solve`

```rust
// VERIFIED: fdars-core/src/function_on_scalar.rs:121-149
fn penalized_solve(
    xtx: &[f64],
    xty: &FdMatrix,
    penalty: &[f64],
    lambda: f64,
) -> Result<FdMatrix, FdarError>
```
Currently `fn` (private — module-only). Takes row-major `xtx` (p×p), `xty` as `FdMatrix` (p×m, where m is number of RHS columns), row-major `penalty` (p×p), scalar `lambda`. Returns β as `FdMatrix` (p×m). To reuse in `peer.rs`, either widen to `pub(crate)` (recommended) or inline the pattern.

### `function_on_scalar.rs::compute_trace_hat`

```rust
// VERIFIED: fdars-core/src/function_on_scalar.rs:401-419
fn compute_trace_hat(xtx: &[f64], penalty: &[f64], lambda: f64, p: usize, n: usize) -> f64
```
Private. Computes tr(H) = tr((X'X + λP)^{-1} X'X). Returns `p as f64` as fallback on Cholesky failure. Widen to `pub(crate)` or replicate inline in `peer.rs`.

### `linalg.rs::cholesky_factor`

```rust
// VERIFIED: fdars-core/src/linalg.rs:85-108
pub(crate) fn cholesky_factor(a: &[f64], p: usize) -> Result<Vec<f64>, FdarError>
```
Row-major p×p symmetric PD input. Returns L (row-major, p×p). Errors: `FdarError::ComputationFailed` if diagonal ≤ 1e-12.

### `linalg.rs::cholesky_forward_back`

```rust
// VERIFIED: fdars-core/src/linalg.rs:113-128
pub(crate) fn cholesky_forward_back(l: &[f64], b: &[f64], p: usize) -> Vec<f64>
```
Solves Lz=b then L'x=z. Returns x (length p).

### `linalg.rs::cholesky_solve`

```rust
// VERIFIED: fdars-core/src/linalg.rs:131-134
pub(crate) fn cholesky_solve(a: &[f64], b: &[f64], p: usize) -> Result<Vec<f64>, FdarError>
```
One-shot: factor then solve. Returns x (length p).

### `linalg.rs::compute_xtx`

```rust
// VERIFIED: fdars-core/src/linalg.rs:137-152
pub(crate) fn compute_xtx(x: &FdMatrix) -> Vec<f64>
```
Returns X'X as flat row-major p×p. `peer.rs` can use this if W is stored as an `FdMatrix`.

### `helpers.rs::simpsons_weights`

```rust
// VERIFIED: fdars-core/src/helpers.rs:76-105
pub fn simpsons_weights(argvals: &[f64]) -> Vec<f64>
```
Returns integration weights; handles uniform and non-uniform grids, n=2 (trapezoidal), n≥3 (Simpson's). Public — no visibility change needed.

### `regression.rs::fdata_to_pc_1d`

```rust
// VERIFIED: fdars-core/src/regression.rs:331-335
pub fn fdata_to_pc_1d(
    data: &FdMatrix,
    ncomp: usize,
    argvals: &[f64],
) -> Result<FpcaResult, FdarError>
```
Not needed for Phase 66 (PEER works directly on the raw design matrix). Available if a basis projection step is ever desired.

### `error.rs::FdarError`

```rust
// VERIFIED: fdars-core/src/error.rs:6-25
pub enum FdarError {
    InvalidDimension { parameter: &'static str, expected: String, actual: String },
    InvalidParameter { parameter: &'static str, message: String },
    ComputationFailed { operation: &'static str, detail: String },
    InvalidEnumValue { enum_name: &'static str, value: i32 },
}
```
Use `InvalidDimension` for wrong-dim Q, `ComputationFailed` for Cholesky failure in the solve.

### `matrix.rs::FdMatrix`

Key construction and access methods [VERIFIED: fdars-core/src/matrix.rs]:
- `FdMatrix::zeros(nrows, ncols)` — line 84
- `FdMatrix::from_column_major(data: Vec<f64>, nrows, ncols) -> Result<Self, FdarError>` — line 50
- `mat[(i, j)]` — `IndexMut` / `Index` (row i, col j in column-major)
- `mat.column(col) -> &[f64]` — zero-copy contiguous column slice — line 127
- `mat.row_to_buf(row, buf)` — zero-alloc row gather — line 157
- `mat.shape() -> (nrows, ncols)` — line 106
- `mat.nrows()`, `mat.ncols()` — lines 94, 100

Column-major storage: element (row, col) is at flat index `row + col * nrows`. [VERIFIED: fdars-core/src/matrix.rs:19-44]

### `src/lib.rs` — additive registration pattern

```rust
// VERIFIED: fdars-core/src/lib.rs:80-123 (representative excerpt)
pub mod kernel_kmeans;   // line 105
pub mod kshape;          // line 106
pub mod optimal_design;  // line 109
```
`pub mod peer;` is added in the same alphabetical region. Crate-root `pub use` lines are deferred to Phase 68. [VERIFIED: fdars-core/src/lib.rs:105-109]

### `FregreLmResult` — template for `PeerResult` shape

```rust
// VERIFIED: fdars-core/src/scalar_on_function/mod.rs:65-98
pub struct FregreLmResult {
    pub intercept: f64,
    pub beta_t: Vec<f64>,         // length m
    pub beta_se: Vec<f64>,        // length m
    pub gamma: Vec<f64>,
    pub fitted_values: Vec<f64>,  // length n
    pub residuals: Vec<f64>,      // length n
    pub r_squared: f64,
    pub r_squared_adj: f64,
    pub std_errors: Vec<f64>,
    pub ncomp: usize,
    pub fpca: FpcaResult,
    pub coefficients: Vec<f64>,
    pub residual_se: f64,
    pub gcv: f64,
    pub aic: f64,
    pub bic: f64,
}
```
`PeerResult` is analogous but simpler (no FPC layer, no ncomp). Carry: `beta` (Vec<f64>, length m), `intercept` (f64), `fitted_values` (Vec<f64>, length n), `effective_df` (f64), `lambda` (f64), `penalty_type` (PeerPenalty).

---

## Architecture Patterns

### System Architecture Diagram

```
peer(data, y, argvals, config)
         |
         v
[Input Validation]
  n > 0, m > 0, argvals.len()==m,
  Q dim check (Decree variant)
         |
         v
[Integration Weights]
  simpsons_weights(argvals) -> w (length m)
         |
         v
[Design Matrix W]
  W[i,j] = data[(i,j)] * w[j]  (n x m)
         |
         v
[Centering]
  y_bar = mean(y)
  y_c = y - y_bar
  w_bar = col_mean(W)
  W_c = W - w_bar             (n x m)
         |
         v
[Penalty Matrix Q] ─── PeerPenalty::Ridge: I_m
(p x p = m x m)   ─── PeerPenalty::Difference: penalty_matrix(m)
                   ─── PeerPenalty::Decree(raw): validate + use as-is
         |
         v
[Penalized Normal Equations]
  WtW = W_c' W_c       (m x m, row-major)
  wty = W_c' y_c       (length m)
  A   = WtW + lambda*Q (m x m)
  beta = cholesky_solve(A, wty, m)
         |
         v
[Effective df]
  effective_df = trace(W_c * A^{-1} * W_c')
               = trace(A^{-1} * WtW)  [compute_trace_hat pattern]
         |
         v
[Fitted values]
  fitted[i] = y_bar + W_c[i,:] . beta
         |
         v
PeerResult { beta, intercept: y_bar, fitted_values, effective_df, lambda, penalty_type }
```

### Recommended Project Structure

```
fdars-core/src/
├── peer.rs              # New file: all Phase 66 code
│   ├── PeerPenalty      # Enum (Ridge, Difference { order }, Decree(Vec<f64>, usize))
│   ├── PeerConfig       # Builder struct (lambda, penalty_type, ...)
│   ├── PeerResult       # Result struct (#[non_exhaustive])
│   ├── peer()           # Public entry point
│   ├── build_w()        # Private: integrate design (helper)
│   ├── build_q()        # Private: dispatch penalty construction
│   ├── compute_peer_trace_hat() # Private: effective df
│   └── #[cfg(test)] mod tests
└── lib.rs               # Add: pub mod peer;
```

### Pattern: PeerConfig builder (matching GmmClusterConfig)

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PeerConfig {
    pub penalty: PeerPenalty,
    pub lambda: f64,
}

impl Default for PeerConfig {
    fn default() -> Self {
        Self {
            penalty: PeerPenalty::Difference { order: 2 },
            lambda: 1.0,
        }
    }
}
```

### Pattern: PeerPenalty enum

```rust
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum PeerPenalty {
    Ridge,
    Difference { order: usize },
    Decree(Vec<f64>, usize), // (flat Q, dimension p)  [caller supplies p×p matrix]
}
```

Note: `Decree(Vec<f64>, usize)` stores the flat Q (row-major or column-major — must match how `cholesky_solve` expects it; crate convention is row-major for solve matrices) and the dimension p. `serde` support on `Decree` requires `Vec<f64>` which is trivially serializable.

### Pattern: PeerResult struct

```rust
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[must_use = "expensive computation whose result should not be discarded"]
pub struct PeerResult {
    /// Estimated coefficient function β(t), length m (on the argvals grid)
    pub beta: Vec<f64>,
    /// Intercept (= y_bar, the centered response mean)
    pub intercept: f64,
    /// Fitted values ŷ_i, length n
    pub fitted_values: Vec<f64>,
    /// Effective degrees of freedom tr(H)
    pub effective_df: f64,
    /// Smoothing parameter λ used
    pub lambda: f64,
    /// Penalty type used
    pub penalty_type: PeerPenalty,
}
```

### Anti-Patterns to Avoid

- **Applying Q as a basis penalty (wrong model):** PEER applies Q directly to the gridwise coefficient vector β (length m = p). Do NOT treat Q as a basis expansion penalty (that is the FOSR model). The design integral W and the coefficient β live on the same grid as argvals. [ASSUMED — from refund baseline formulation]
- **Forgetting to weight W by integration weights:** W[i,j] = X_i(t_j) · w_j (not raw X values). Without the quadrature weighting, the integral approximation is wrong. [ASSUMED]
- **NaN from Cholesky on a rank-deficient Q:** When Q is Decree and caller supplies a rank-deficient matrix, A = WtW + λQ may be singular. Guard with `cholesky_factor` → `FdarError::ComputationFailed`. [VERIFIED pattern in function_on_scalar.rs]
- **Row-major vs column-major confusion for Q:** `cholesky_solve`/`cholesky_factor` in `linalg.rs` expect row-major matrices [VERIFIED: linalg.rs:85-134]. If Decree Q comes from nalgebra or is built column-major, transpose before passing.
- **penalty_matrix returns a fixed order-2 builder:** Do not use `penalty_matrix(m)` for `Difference { order: k }` where k≠2. Either restrict Difference to order=2 (safe for Phase 66 given CONTEXT.md default) or write a general builder.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Cholesky penalized solve | Custom LU or iterative solver | `cholesky_factor` + `cholesky_forward_back` from `linalg.rs` | Already handles edge cases, returns descriptive FdarError |
| Simpson's integration weights | Hand-rolled trapezoid | `helpers::simpsons_weights` | Handles uniform/non-uniform, n=2 edge case |
| 2nd-difference penalty matrix | Write D'D from scratch | `function_on_scalar::penalty_matrix(m)` | Tested, banded construction |
| Eigendecomposition of Q | Power iteration | nalgebra `DMatrix::symmetric_eigen()` | Already used in fpca_variants.rs, handles symmetric matrices |
| Hat-matrix trace | Explicit H = WA^{-1}W' then sum diagonal | `compute_trace_hat` pattern (column-solve trick) | O(p²) trace via p Cholesky solves, not O(np²) for explicit H |

**Key insight:** All linear algebra primitives exist and are `pub(crate)`. The only new code is the PEER-specific plumbing: penalty dispatch, design weighting, and result assembly.

---

## Common Pitfalls

### Pitfall 1: W not weighted by integration weights

**What goes wrong:** β(t) is estimated without functional inner product, giving a biased estimate that depends on the grid spacing.
**Why it happens:** Forgetting that ∫X_i(t)ψ(t)dt ≈ Σ_j X_i(t_j)w_j, so W[i,j] = data[(i,j)] * w[j].
**How to avoid:** Apply `simpsons_weights` immediately after construction, before centering.
**Warning signs:** β(t) magnitude changes with number of grid points at fixed data.

### Pitfall 2: Decree Q dimension mismatch

**What goes wrong:** Caller supplies a p_Q × p_Q matrix with p_Q ≠ m (number of argvals), causing dimension mismatch in the Cholesky solve.
**Why it happens:** Caller forgets Q must be m×m (grid-space dimension).
**How to avoid:** Validate `p_Q == m` before solve; return `FdarError::InvalidDimension { parameter: "penalty_type", expected: format!("{m}x{m}"), actual: format!("{p_Q}x{p_Q}") }`.
**Warning signs:** Panic or incorrect output without error.

### Pitfall 3: Decree Q not symmetric or not PSD

**What goes wrong:** Cholesky fails midway (non-positive diagonal) returning `FdarError::ComputationFailed`.
**Why it happens:** Caller passes a non-symmetric or indefinite matrix.
**How to avoid:** Optionally symmetrize Q = (Q + Q')/2 before solve, or document requirement and let Cholesky catch it.
**Warning signs:** `ComputationFailed` on valid-looking data.

### Pitfall 4: NaN β(t) when lambda=0 and W_c'W_c is rank-deficient

**What goes wrong:** λ=0 and n < m means WtW is rank-deficient; Cholesky fails.
**Why it happens:** Under-determined system.
**How to avoid:** For `lambda = 0.0`, add a small ridge: `a[j*m+j] += 1e-10` before solve; or document that `lambda > 0` is required and validate.
**Warning signs:** `ComputationFailed` or NaN in β.

### Pitfall 5: Q storage convention mismatch

**What goes wrong:** Decree Q is built or passed column-major, but `cholesky_factor` expects row-major.
**Why it happens:** nalgebra stores column-major; fdars linalg helpers use row-major.
**How to avoid:** For symmetric Q, row-major and column-major are equivalent (Q[i,j] == Q[j,i]), so no explicit transpose is needed IF Q is symmetric. Document this constraint.
**Warning signs:** Incorrect β(t) that is not caught by tests (numerical error, not crash).

### Pitfall 6: `penalty_matrix(m)` is row-major with index `r*m+c`

The second-difference penalty matrix from `penalty_matrix` uses row-major index `dtd[r * m + c]` [VERIFIED: fdars-core/src/function_on_scalar.rs:113]. This is consistent with what `cholesky_factor` expects.

---

## Code Examples

### Minimal PEER fit skeleton

```rust
// Source: derived from function_on_scalar.rs patterns [VERIFIED]
use crate::helpers::simpsons_weights;
use crate::linalg::{cholesky_factor, cholesky_forward_back, cholesky_solve};
use crate::matrix::FdMatrix;
use crate::error::FdarError;
use crate::function_on_scalar::penalty_matrix;  // pub(crate) — needs widening or replication

pub fn peer(
    data: &FdMatrix,
    y: &[f64],
    argvals: &[f64],
    config: &PeerConfig,
) -> Result<PeerResult, FdarError> {
    let (n, m) = data.shape();
    // ... validate ...

    // 1. Integration weights
    let w = simpsons_weights(argvals);

    // 2. Weighted design W[i,j] = data[(i,j)] * w[j]
    let mut wmat = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            wmat[(i, j)] = data[(i, j)] * w[j];
        }
    }

    // 3. Center
    let y_bar: f64 = y.iter().sum::<f64>() / n as f64;
    let yc: Vec<f64> = y.iter().map(|&yi| yi - y_bar).collect();
    let w_bar: Vec<f64> = (0..m).map(|j| (0..n).map(|i| wmat[(i,j)]).sum::<f64>() / n as f64).collect();
    let mut wc = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            wc[(i,j)] = wmat[(i,j)] - w_bar[j];
        }
    }

    // 4. Penalty Q (m×m, row-major)
    let q: Vec<f64> = build_q(m, &config.penalty)?;

    // 5. WtW and wty
    let mut wtw = vec![0.0_f64; m * m];
    for j in 0..m {
        for k in j..m {
            let s: f64 = (0..n).map(|i| wc[(i,j)] * wc[(i,k)]).sum();
            wtw[j*m + k] = s;
            wtw[k*m + j] = s;
        }
    }
    let wty: Vec<f64> = (0..m).map(|j| (0..n).map(|i| wc[(i,j)] * yc[i]).sum()).collect();

    // 6. A = WtW + lambda*Q, solve
    let lambda = config.lambda;
    let mut a = vec![0.0_f64; m * m];
    for i in 0..m*m { a[i] = wtw[i] + lambda * q[i]; }
    let beta = cholesky_solve(&a, &wty, m)?;

    // 7. Effective df
    let effective_df = compute_peer_trace_hat(&wtw, &q, lambda, m, n);

    // 8. Fitted values
    let fitted_values: Vec<f64> = (0..n).map(|i| {
        y_bar + (0..m).map(|j| wc[(i,j)] * beta[j]).sum::<f64>()
    }).collect();

    Ok(PeerResult {
        beta,
        intercept: y_bar,
        fitted_values,
        effective_df,
        lambda,
        penalty_type: config.penalty.clone(),
    })
}
```

### build_q: Penalty dispatch

```rust
fn build_q(m: usize, penalty: &PeerPenalty) -> Result<Vec<f64>, FdarError> {
    match penalty {
        PeerPenalty::Ridge => {
            let mut q = vec![0.0_f64; m * m];
            for i in 0..m { q[i*m + i] = 1.0; }
            Ok(q)
        }
        PeerPenalty::Difference { order: 2 } => {
            Ok(penalty_matrix(m))  // reuse pub(crate) helper
        }
        PeerPenalty::Difference { order } => {
            // General-order difference penalty — or restrict to order=2 for Phase 66
            Err(FdarError::InvalidParameter {
                parameter: "penalty_type",
                message: format!("Difference order {} not supported; use order 2", order),
            })
        }
        PeerPenalty::Decree(q_raw, p_q) => {
            if *p_q != m || q_raw.len() != m * m {
                return Err(FdarError::InvalidDimension {
                    parameter: "penalty_type",
                    expected: format!("{m}x{m} matrix ({})", m*m),
                    actual: format!("{p_q}x{p_q} matrix ({})", q_raw.len()),
                });
            }
            Ok(q_raw.clone())
        }
    }
}
```

### Known-answer synthetic test

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn test_peer_ridge_beta_recovery() {
        // Synthetic: y_i = integral of X_i(t) * sin(pi*t) dt + epsilon
        // True beta(t) = sin(pi*t)
        let (n, m) = (50, 40);
        let t = uniform_grid(m);
        let true_beta: Vec<f64> = t.iter().map(|&ti| (std::f64::consts::PI * ti).sin()).collect();

        // Generate X_i(t) as random curves + noise
        let mut data = FdMatrix::zeros(n, m);
        let mut y = vec![0.0_f64; n];
        let w = simpsons_weights(&t);
        for i in 0..n {
            for j in 0..m {
                let xi = ((i as f64 * 0.3 + j as f64 * 0.1) * 0.5).sin();
                data[(i, j)] = xi;
                y[i] += xi * true_beta[j] * w[j];  // integral approximation
            }
            y[i] += 0.01 * (i as f64 * 0.7 % 1.0 - 0.5); // small noise
        }

        let config = PeerConfig { penalty: PeerPenalty::Ridge, lambda: 1e-4 };
        let result = peer(&data, &y, &t, &config).unwrap();

        // beta recovery: max absolute error < 0.1
        let max_err = result.beta.iter().zip(true_beta.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 0.1, "Ridge beta recovery error: {}", max_err);
        assert!(result.fitted_values.iter().all(|v| v.is_finite()));
    }
}
```

---

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| FPCR (plain FPC scores regression) | PEER (structured penalty with null/range split) | Randolph et al. 2012 [ASSUMED] | Allows a-priori domain knowledge in regularization |
| Manual penalty matrix | Reuse `penalty_matrix(m)` from `function_on_scalar.rs` | v0.35.0+ (already exists) | Zero code duplication |
| `linalg::cholesky_solve` as private | Exists as `pub(crate)` already | Current codebase | Accessible from `peer.rs` without change |

**Not deprecated:** The FPCR approach (`fregre_lm`) remains; PEER is an additive alternative.

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (#[test], #[cfg(test)]) |
| Config file | none (inline module tests per crate convention) |
| Quick run command | `cargo test -p fdars-core --features linalg,parallel peer` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| PER-01 | β(t) recovery on synthetic data within tolerance | unit (known-answer) | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_ridge_beta_recovery` | Wave 0 |
| PER-01 | `effective_df` is finite and positive | unit | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_df_positive` | Wave 0 |
| PER-01 | `fitted_values` length = n, no NaN | unit | included in beta recovery test | Wave 0 |
| PER-01 | Wrong-dim Q → `FdarError::InvalidDimension`, no panic | unit | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_decree_wrong_dim` | Wave 0 |
| PER-02 | Ridge family fits without error | unit | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_ridge_fits` | Wave 0 |
| PER-02 | Difference family fits without error | unit | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_difference_fits` | Wave 0 |
| PER-02 | Decree family fits without error | unit | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_decree_fits` | Wave 0 |
| PER-02 | Decree β(t) ≠ Difference β(t) (partition-awareness) | unit | `cargo test -p fdars-core --features linalg,parallel peer::tests::test_peer_decree_distinct_from_roughness` | Wave 0 |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core --features linalg,parallel peer -- --nocapture`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel`
- **Phase gate:** Full suite green before `/gsd-verify-work`. Also: `cargo clippy --all-targets --features linalg,parallel -- -D warnings` and `cargo fmt -- --check`.

### Wave 0 Gaps

- [ ] `fdars-core/src/peer.rs` — entire new file; covers all above tests
- [ ] `fdars-core/src/lib.rs` — add `pub mod peer;`
- [ ] Optional: widen `penalty_matrix` and `penalized_solve`/`compute_trace_hat` to `pub(crate)` in `function_on_scalar.rs`

*(No existing test infrastructure covers Phase 66 — all tests are new.)*

---

## Verify Commands

```bash
# Run only peer module tests (fast)
cargo test -p fdars-core --features linalg,parallel peer

# Full suite (required before /gsd-verify-work)
cargo test -p fdars-core --features linalg,parallel

# Clippy (CI gate — must use --all-targets)
cargo clippy --all-targets --features linalg,parallel -- -D warnings

# Format check
cargo fmt -- --check

# Commit without pre-commit hook (when hook is slow / /tmp is full)
git commit --no-verify -m "..."
```
Source: MEMORY.md (`ci-clippy-all-targets-gate.md`, `noverify-commits-leave-fmt-drift.md`). [VERIFIED: project MEMORY.md]

---

## Security Domain

`security_enforcement` is not configured; treated as enabled. PEER is a pure numerical computation module with no I/O, no network, no file access, no user authentication. ASVS categories:

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V2 Authentication | no | — |
| V3 Session Management | no | — |
| V4 Access Control | no | — |
| V5 Input Validation | yes (dim checks) | `FdarError::InvalidDimension` at entry; no silent truncation |
| V6 Cryptography | no | — |

No threat patterns beyond bounds-checking on matrix dimensions (already enforced by the error-handling convention) and NaN propagation from ill-conditioned inputs (guarded by Cholesky failure path).

---

## Open Questions

1. **`penalty_matrix` visibility — widen or replicate?**
   - What we know: `penalty_matrix` is `pub(crate)` at the `function_on_scalar` module level and callable from `peer.rs` in the same crate.
   - Wait: re-reading — `penalty_matrix` is `pub(crate)` [VERIFIED: fdars-core/src/function_on_scalar.rs:99 — `pub(crate) fn penalty_matrix`]. It IS accessible from `peer.rs` without any change. No decision needed.

2. **Should `Difference { order }` support order ≠ 2?**
   - What we know: `penalty_matrix(m)` only builds order-2. The CONTEXT.md default is order 2.
   - What's unclear: Whether Phase 66 should support arbitrary order.
   - Recommendation: Restrict to order=2 in Phase 66 via a validation error for other orders; add a general-order builder in Phase 67 or later if needed. Keeps scope tight.

3. **Normalization of Decree Q storage convention (row-major vs column-major)?**
   - What we know: `cholesky_solve` expects row-major; `penalty_matrix` builds row-major; Q symmetric means row-major == column-major.
   - What's unclear: Should the doc say row-major explicitly to guide callers?
   - Recommendation: For symmetric Q (the only valid case for a penalty matrix), both conventions are equivalent. Document that Q must be symmetric. If caller passes non-symmetric Q, the asymmetry is silently ignored by the symmetric Cholesky.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | PEER design matrix is W[i,j] = X_i(t_j) · w_j (raw data weighted by integration weights, not a basis expansion) | PEER Method §1 | If PEER uses a basis expansion, W dimensions differ and the penalized system is basis-space, not grid-space |
| A2 | refund's `peer()` with `pentype="DECREE"` uses Q directly in the penalized normal equations without additional transformation | PEER Method §3 | If refund applies Q in a transformed basis, fdars output differs |
| A3 | nalgebra `.symmetric_eigen()` returns eigenvalues in ascending order | PEER Method §4 | Sort ascending explicitly to be safe; no functional risk |
| A4 | The null/range split tolerance `1e-8 * max_eigenvalue` is appropriate | PEER Method §4 | Too large → over-penalized; too small → numerical null-space errors. Claude's discretion on the value |
| A5 | Difference order > 2 is not needed in Phase 66 | Open Questions §2 | Low risk; revisit in Phase 67 if needed |

**If this table is not empty:** A1 and A2 are the highest-risk assumptions. The β(t) recovery test on known synthetic data will expose them — if recovery fails, suspect the design matrix formulation first.

---

## Sources

### Primary (HIGH confidence — files read this session)

- `fdars-core/src/function_on_scalar.rs:99-116` — `penalty_matrix` (D'D second-difference builder)
- `fdars-core/src/function_on_scalar.rs:121-149` — `penalized_solve` (penalized normal equations + Cholesky)
- `fdars-core/src/function_on_scalar.rs:401-419` — `compute_trace_hat` (hat-matrix trace via Cholesky)
- `fdars-core/src/linalg.rs:85-134` — `cholesky_factor`, `cholesky_forward_back`, `cholesky_solve`
- `fdars-core/src/linalg.rs:137-152` — `compute_xtx`
- `fdars-core/src/helpers.rs:76-105` — `simpsons_weights`
- `fdars-core/src/matrix.rs:1-215` — `FdMatrix` methods
- `fdars-core/src/error.rs:6-25` — `FdarError` variants
- `fdars-core/src/scalar_on_function/mod.rs:65-98` — `FregreLmResult` structure
- `fdars-core/src/lib.rs:64-156` — module registration pattern
- `fdars-core/src/fpca_variants.rs:484` — nalgebra `.symmetric_eigen()` usage
- `fdars-core/src/regression.rs:331-335` — `fdata_to_pc_1d` signature

### Tertiary (LOW confidence — training knowledge, not verified)

- PEER algorithm formulation (Randolph et al. 2012): null-space/range-space split, partially-empirical eigenvectors [ASSUMED]
- refund@0.1-38 `peer()` internals: exact design matrix W construction and `pentype` dispatch [ASSUMED]

---

## Metadata

**Confidence breakdown:**
- Standard stack (no new deps): HIGH — confirmed via codebase read
- Architecture (PEER algorithm + reuse map): HIGH for fdars internals; LOW (ASSUMED) for PEER paper formulation
- Pitfalls: HIGH (based on actual code paths read)
- Validation architecture: HIGH (test commands verified against MEMORY.md)

**Research date:** 2026-09-04
**Valid until:** 2026-11-04 (stable Rust crate internals; 60 days)
