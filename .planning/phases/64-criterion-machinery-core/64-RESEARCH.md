# Phase 64: Criterion Machinery Core - Research

**Researched:** 2026-09-02
**Domain:** Optimal Experimental Design for Sparse FDA — BLUP-MSE trajectory criterion + FPC-score posterior covariance criterion
**Confidence:** HIGH

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **Criterion selection via nested `DesignCriterion` enum**: `DesignCriterion::Trajectory` and `DesignCriterion::Score(OptimalityKind)`. One public dispatch point; mirrors the `pace_fpca` config-enum style. Not two separate functions, not a flat 3-variant enum.
- **`design_criterion` signature**: `design_criterion(model: &PaceFpcaResult, selected: &[usize], criterion: DesignCriterion) -> Result<f64, FdarError>`. `selected` are **indices into `model.argvals`** (exact index arithmetic, no interpolation). Read-only borrow of the model.
- **`OptimalityKind` variants**: `A` (trace of posterior covariance) and `D` (log-det of posterior covariance) only. No `E`/`G` now.
- **Empty-set `selected == &[]` returns the prior baseline**: `MSE(∅) ≈ Σ_k λ_k`, `A(∅) = Σ_k λ_k`, `D(∅) = Σ_k log λ_k`. Not rejected as `InvalidParameter`.
- **Shared `build_sigma_design`**: assembles `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p` where p = |selected|, row-major, mirroring `pace_fpca.rs:461–474`. Shape is `|S|×|S|`, NOT K×K.
- **Trajectory criterion (FOD-01)**: integrated Simpson-weighted BLUP-MSE. MUST use `helpers::simpsons_weights(&model.argvals)`. Known-answer: `MSE(∅) ≈ Σ_k λ_k`.
- **Score criterion (FOD-02)**: K×K posterior `Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ`. A-opt = trace(Cov), D-opt = log det(Cov) (NEGATIVE). Known-answer: `Cov(ξ|∅) = diag(λ)`.
- **Optimality-sign / monotonicity gate**: `criterion(S∪{t}) ≤ criterion(S) + 1e-12` for all three criteria.
- **Validation**: out-of-range index → `FdarError::InvalidParameter`; `ncomp == 0` / `sigma2 <= 0` guards.
- **Ridge-retry `1e-8`** on near-singular Σ_d; never panic.
- **New file `src/optimal_design.rs`** — top-level peer of `kshape.rs`/`kernel_kmeans.rs`.
- **Additive lib.rs re-export**: enums + `design_criterion` only. Phase 65 adds full surface.
- **No new crate dependency**; MSRV 1.81 preserved; `linalg` feature NOT required.
- **`linalg::cholesky_solve`** is always available (NOT behind the `linalg` feature).

### Claude's Discretion

- Exact internal helper factoring (~3 private helpers), variable naming, and test-module layout follow existing `optimal_design`-peer conventions (`kshape.rs`, `kernel_kmeans.rs`).

### Deferred Ideas (OUT OF SCOPE)

- Greedy `optimal_design` wrapper, `OptDesConfig`/`OptDesResult`, full re-export surface, benchmark → Phase 65.
- FOD-BREADTH (SR-criterion, exhaustive/branch-and-bound, CV-ridge selection, rank-1 Cholesky update, off-grid interpolated candidates) → future milestone.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FOD-01 | Trajectory-reconstruction design criterion — integrated conditional BLUP-MSE of x̂(t), Simpson-weighted, known-answer `MSE(∅) ≈ Σ_k λ_k` | Math section, Σ_d assembly, simpsons_weights pattern, empty-set derivation |
| FOD-02 | FPC-score-prediction design criterion — A-/D-optimality of posterior score covariance `Cov(ξ|Y_S) = Λ − Λ Φ_Sᵀ Σ_d⁻¹ Φ_S Λ`, known-answer `Cov(ξ|∅) = diag(λ)` | Score criterion math, cholesky_solve pattern, log-det from Cholesky factor, empty-set derivation |
| FOD-03 | Single public evaluator `design_criterion` with `DesignCriterion`/`OptimalityKind` enums, independently useful for hand-chosen designs, serde-gated derives, additive lib.rs re-export | API surface section, enum design, lib.rs re-export pattern |
</phase_requirements>

---

## Summary

Phase 64 delivers the pure numerical core of the FOptDes module: the `design_criterion` evaluator and its two criterion branches, the shared `build_sigma_design` private helper, and the `DesignCriterion`/`OptimalityKind` public enum pair. Everything lives in a new `src/optimal_design.rs` (top-level peer of `kshape.rs`/`kernel_kmeans.rs`).

The mathematics derives directly from Ji & Müller (2017) and the Yao–Müller–Wang (2005) PACE formulation already implemented in `pace_fpca.rs`. Both the Σ_d assembly and the A_mat/Ω_i posterior-covariance pattern are present verbatim in the existing codebase at lines 461–474 and 547–558 of `pace_fpca.rs`, making this primarily a refactor-to-expose exercise rather than novel math. The trajectory criterion integrates the BLUP prediction variance over the work grid using Simpson weights; the score criterion evaluates the posterior score covariance's trace or log-determinant.

The critical implementation detail is that **`Σ_d` is p×p where p = |selected|**, NOT K×K. The two branches share `build_sigma_design` but differ only in post-solve usage: the trajectory branch integrates a grid-wide quadratic form, while the score branch computes a K×K posterior covariance matrix. All known-answer tests (empty-set priors, monotonicity) are exact algebraic identities derivable from the formulas — they require no external data.

**Primary recommendation:** Copy the Σ_yi assembly pattern from `pace_fpca.rs:461–474` directly into `build_sigma_design`, substituting grid-subset indices for per-curve observation indices. Copy the A_mat/Ω_i pattern from `pace_fpca.rs:547–558` for the score criterion. Gate both on the existing `cholesky_solve` (no new dependency). Wire the `log_det_from_cholesky` from `linalg.rs` for D-optimality.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| `build_sigma_design` helper | `optimal_design.rs` (private) | — | p×p Gram sub-matrix; pattern mirrors `pace_fpca.rs:461–474` |
| Trajectory criterion (BLUP-MSE) | `optimal_design.rs` | `helpers::simpsons_weights` | Grid quadrature; must not use uniform weights |
| Score criterion (posterior Cov) | `optimal_design.rs` | `linalg::cholesky_solve`, `linalg::log_det_from_cholesky` | K×K matrix solve; log-det for D-opt |
| Public API dispatch | `optimal_design::design_criterion` | `lib.rs` re-export | Single enum-dispatched entry point |
| Validation | `design_criterion` entry | `FdarError` variants | Index range, sigma2 > 0, ncomp > 0 |

---

## Standard Stack

### Core (no new dependencies — reuse-first)

| Component | Location | Purpose | Status |
|-----------|----------|---------|--------|
| `linalg::cholesky_solve` | `src/linalg.rs:131–134` | Solve p×p system Ax = b | Always available, NOT behind `linalg` feature |
| `linalg::cholesky_factor` | `src/linalg.rs:85–108` | Factor A = LL' for log-det | Always available |
| `linalg::log_det_from_cholesky` | `src/linalg.rs:63–69` | 2·Σ ln(L_ii) for D-optimality | Always available |
| `linalg::cholesky_forward_back` | `src/linalg.rs:113–128` | Forward+back substitution for multi-RHS | Always available |
| `helpers::simpsons_weights` | `src/helpers.rs:76–105` | Quadrature weights for trajectory integral | Pass `&model.argvals` |
| `pace_fpca::PaceFpcaResult` | `src/pace_fpca.rs:99–120` | Read-only model: eigenvalues, eigenfunctions, argvals, sigma2, ncomp | Borrow only |
| `matrix::FdMatrix` | `src/matrix.rs` | Column-major functional data — `(row, col)` at `row + col * nrows` | Indexing for eigenfunctions |
| `error::FdarError` | `src/error.rs` | `InvalidParameter`, `InvalidDimension`, `ComputationFailed` | All public fns return Result |

**Installation:** No `Cargo.toml` change required.

---

## Architecture Patterns

### System Architecture Diagram

```
Caller
  │
  ├─ design_criterion(model, selected_indices, DesignCriterion::Trajectory)
  │      │
  │      ├─ validate: indices in-range, sigma2>0, ncomp>0
  │      ├─ build_sigma_design(model, selected)  →  p×p Σ_d [row-major]
  │      │      Φ_d diag(λ) Φ_dᵀ + σ²I_p
  │      │      (mirror of pace_fpca.rs:461–474, Σ_yi but at design indices)
  │      │      ridge-retry 1e-8 if Cholesky fails
  │      ├─ for each grid point j:
  │      │      φ_d(t_j) = eigenfunctions[(j, :)] at selected rows  [p-vec]
  │      │      prior_var_j = Σ_k λ_k φ_k(t_j)²                    [scalar]
  │      │      reduction_j = φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j)            [scalar via cholesky_solve]
  │      │      integrand_j = prior_var_j − reduction_j
  │      └─ return Σ_j w_j · integrand_j   (Simpson-weighted)
  │
  └─ design_criterion(model, selected_indices, DesignCriterion::Score(OptimalityKind::A|D))
         │
         ├─ validate: same guards
         ├─ build_sigma_design(model, selected)  →  p×p Σ_d
         ├─ empty-set fast path: return Σλ_k (A) or Σ log λ_k (D)
         ├─ for each component k (1..ncomp):
         │      φ_d[:,k] = eigenfunctions[(j, k)] for j in selected  [p-vec]
         │      λ_k · cholesky_solve(Σ_d, φ_d[:,k])  →  sol_k        [p-vec]
         │      for l (1..ncomp): A_mat[k,l] = λ_k · dot(φ_d[:,k], sol_l) · λ_l
         ├─ Cov[k,l] = (k==l ? λ_k : 0) − A_mat[k,l]  (K×K posterior Cov)
         ├─ A-opt: trace(Cov) = Σ_k Cov[k,k]
         └─ D-opt: log det(Cov) via cholesky_factor(Cov) + log_det_from_cholesky
```

### Recommended Project Structure

```
fdars-core/src/
├── optimal_design.rs      # NEW — entire Phase 64 output
│   ├─ DesignCriterion enum (pub, Debug/Clone/PartialEq, serde-gated)
│   ├─ OptimalityKind enum (pub, Debug/Clone/PartialEq, serde-gated)
│   ├─ build_sigma_design() — private helper
│   ├─ trajectory_criterion() — private helper
│   ├─ score_criterion() — private helper
│   ├─ design_criterion() — pub #[must_use]
│   └─ #[cfg(test)] mod tests { ... }
├── lib.rs                  # additive: pub mod optimal_design; re-export enums + fn
├── pace_fpca.rs            # UNCHANGED — read-only model source
└── linalg.rs               # UNCHANGED — cholesky_solve, log_det_from_cholesky
```

### Pattern 1: `build_sigma_design` — mirroring pace_fpca.rs:461–474

**What:** Assembles p×p `Σ_d = Φ_d diag(λ) Φ_dᵀ + σ²I_p` where `p = selected.len()`, using design-point rows from `model.eigenfunctions` (column-major: element `(j, k)` at `j + k*m`).

**When to use:** Called by both criterion branches before any Cholesky solve.

```rust
// Source: pace_fpca.rs:461–474 (adapted — grid indices instead of per-curve obs)
fn build_sigma_design(
    model: &PaceFpcaResult,
    selected: &[usize],
) -> Result<Vec<f64>, FdarError> {
    let p = selected.len();
    let ncomp = model.ncomp;
    // Φ_d: p × ncomp, row-major: phi_d[row * ncomp + k]
    // eigenfunctions is m × ncomp column-major: element (j, k) = eigenfunctions[(j, k)]
    let mut sigma_d = vec![0.0_f64; p * p];
    for row in 0..p {
        let j_row = selected[row];
        for col in 0..p {
            let j_col = selected[col];
            let mut s = 0.0_f64;
            for k in 0..ncomp {
                // eigenfunctions[(j, k)] = col-major: j + k * m
                s += model.eigenfunctions[(j_row, k)]
                    * model.eigenvalues[k]
                    * model.eigenfunctions[(j_col, k)];
            }
            sigma_d[row * p + col] = s;
        }
        sigma_d[row * p + row] += model.sigma2;  // σ²I_p diagonal
    }
    Ok(sigma_d)
}
// Ridge-retry (pace_fpca.rs:480–490 pattern):
// match cholesky_solve(&sigma_d, &rhs, p) {
//     Err(_) => { add 1e-8 to diagonals, retry; Err on second fail }
//     Ok(v) => use v
// }
```

**Provenance:** `[VERIFIED: fdars-core/src/pace_fpca.rs:461–474]` — verbatim Σ_yi assembly at those lines:
```rust
// Build Σ_yi (n_i × n_i, row-major): Φ_i diag(λ) Φ_i^T + σ²I
let mut sigma_yi = vec![0.0_f64; n_i * n_i];
for row in 0..n_i {
    for col in 0..n_i {
        let mut s = 0.0_f64;
        for k in 0..actual_ncomp {
            s += phi_i[row * actual_ncomp + k]
                * eigenvalues[k]
                * phi_i[col * actual_ncomp + k];
        }
        sigma_yi[row * n_i + col] = s;
    }
    sigma_yi[row * n_i + row] += sigma2;
}
```

### Pattern 2: `trajectory_criterion` — Simpson-weighted BLUP-MSE

**What:** Computes `Σ_j w_j · (Σ_k λ_k φ_k(t_j)² − φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j))` over the full work grid.

**Key detail:** The quadratic form `φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j)` is computed by solving `Σ_d x = φ_d(t_j)` once per grid point, then computing `dot(φ_d(t_j), x)`. Do NOT use uniform weights — always call `simpsons_weights(&model.argvals)`.

```rust
// Source: pace_fpca.rs pattern + helpers::simpsons_weights
fn trajectory_criterion(
    model: &PaceFpcaResult,
    sigma_d: &[f64],      // p×p row-major
    selected: &[usize],   // indices into model.argvals
    p: usize,
) -> Result<f64, FdarError> {
    let m = model.argvals.len();
    let ncomp = model.ncomp;
    let weights = simpsons_weights(&model.argvals);  // length m
    let mut mse = 0.0_f64;
    for j in 0..m {
        // Prior variance at grid point j: Σ_k λ_k φ_k(t_j)²
        let prior_var: f64 = (0..ncomp)
            .map(|k| model.eigenvalues[k] * model.eigenfunctions[(j, k)].powi(2))
            .sum();
        if p == 0 {
            mse += weights[j] * prior_var;
            continue;
        }
        // φ_d(t_j) = eigenfunctions[(j, k)] for k ∈ selected rows
        // Wait — φ_d(t_j) is a p-vector: the k-th component for each component,
        // evaluated at the design points' rows. But here we need the FULL grid
        // eigenfunctions evaluated at t_j, restricted to which rows appear in selected.
        // Actually: phi_d_at_j[row] = eigenfunctions[(selected[row], :)] — NO.
        // The quadratic form uses: phi_d_at_j is a p-vector where
        //   phi_d_at_j[i] = Σ_k eigenfunctions[(j, k)] * eigenfunctions[(selected[i], k)] * λ_k?
        // NO. Re-read the formula carefully — see Math section below.
        let reduction = trajectory_reduction_at(model, sigma_d, selected, p, j)?;
        mse += weights[j] * (prior_var - reduction);
    }
    Ok(mse)
}
```

(See Math section for the exact quadratic form derivation.)

### Pattern 3: Score criterion — A_mat/Ω_i adapted for prospective design

**What:** Computes K×K posterior `Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ` using the `pace_fpca.rs:547–558` pattern.

**Provenance:** `[VERIFIED: fdars-core/src/pace_fpca.rs:547–558]` — verbatim A_mat assembly:
```rust
// Step 2: A_i[k,l] = diag(λ)[k] · Φ_i[:,k]^T · Σ_yi^{-1} · Φ_i[:,l] · diag(λ)[l]
//   = Σ_j phi_i[j,k] · sigma_inv_phi_lam[j,l]
let mut a_mat = vec![0.0_f64; actual_ncomp * actual_ncomp];
for k in 0..actual_ncomp {
    for l in 0..actual_ncomp {
        let mut s = 0.0_f64;
        for j in 0..n_i {
            s += phi_i[j * actual_ncomp + k] * sigma_inv_phi_lam[j * actual_ncomp + l];
        }
        a_mat[k * actual_ncomp + l] = eigenvalues[k] * s;
    }
}
// Step 3: Ω_i[k,l] = (k==l ? λ_k : 0) - A_i[k,l]
```

In Phase 64, `phi_i` is replaced by `Φ_d` (design-point rows of the eigenfunctions, shape p×K), and `sigma_yi` is replaced by `Σ_d` (p×p). The A_mat / Ω_i pattern is identical.

### Anti-Patterns to Avoid

- **Wrong shape for Σ_d**: Σ_d must be `|S|×|S|` (p×p), NOT K×K. The K×K dimension is for the posterior covariance `Cov(ξ|Y_S)`.
- **Forgetting σ²I**: Omitting `sigma_d[row * p + row] += model.sigma2` makes Σ_d singular for perfectly orthonormal eigenfunctions. This is the most common silent-singularity bug.
- **Uniform integration weights**: Using `1.0/m` instead of `simpsons_weights(&model.argvals)` produces a grid-scale-dependent result that changes with grid density. Always call `simpsons_weights`.
- **Sign flip on D-optimality**: `log det(Cov)` is NEGATIVE because posterior eigenvalues ≤ prior λ_k (information reduces uncertainty). The criterion is minimized (more negative = better). Never negate it or take the absolute value.
- **Using `cholesky_d` instead of `cholesky_solve`**: The public consolidation function is `cholesky_solve` in `linalg.rs:131–134`; `cholesky_d` is a lower-level helper. Use `cholesky_solve` directly.
- **`linalg` feature gate**: `cholesky_solve` is `pub(crate)` and always compiled — it is NOT behind the `linalg` feature. Do not add `#[cfg(feature = "linalg")]`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| p×p Cholesky solve | Custom Gaussian elimination | `linalg::cholesky_solve(a, b, p)` | Validated, handles n_i×n_i in pace_fpca already |
| Log-determinant | Manual diagonal log product | `linalg::log_det_from_cholesky(l, d)` | Already present, correct formula: `2·Σ ln(L_ii)` |
| Simpson quadrature weights | Custom weight loop | `helpers::simpsons_weights(&model.argvals)` | Non-uniform grid handling, edge-case n=2 coverage |
| Σ_d matrix assembly | Per-phase custom variant | Pattern from `pace_fpca.rs:461–474` | Same algebra; share the pattern to avoid drift |
| FdMatrix column-major indexing | Manual `j + k*m` arithmetic | `model.eigenfunctions[(j, k)]` | Operator overload handles the index arithmetic |

---

## The Math (Core Formulas for Implementation)

### Notation

- `m` = work grid length (`model.argvals.len()`)
- `K` = `model.ncomp` (number of FPC components)
- `p` = `selected.len()` (design set size)
- `λ_k` = `model.eigenvalues[k]`
- `φ_k(t_j)` = `model.eigenfunctions[(j, k)]` (column-major FdMatrix)
- `Φ_d` = p×K matrix where `Φ_d[i, k]` = `model.eigenfunctions[(selected[i], k)]`
- `Σ_d` = p×p matrix: `Φ_d diag(λ) Φ_dᵀ + σ²I_p`
- `Λ` = K×K diagonal matrix with `Λ[k,k]` = `λ_k`

### Σ_d assembly (shared helper)

```
Σ_d[i, j] = Σ_k λ_k · φ_k(argvals[selected[i]]) · φ_k(argvals[selected[j]]) + σ² · δ_{ij}
           = Σ_k λ_k · eigenfunctions[(selected[i], k)] · eigenfunctions[(selected[j], k)] + σ² · δ_{ij}
```

This is the covariance of `Y_S` under the PACE model: `Γ(S, S) + σ²I_p` where `Γ` is the functional covariance surface `Σ_k λ_k φ_k φ_kᵀ` evaluated at the design points.

**Empty-set fast path (p=0):** No matrix to build; both criteria use prior values.

### Trajectory criterion (FOD-01)

The full BLUP prediction variance at grid point `t_j` is:
```
Var[x̂(t_j) | Y_S] = Σ_k λ_k φ_k(t_j)² − φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j)
```

where `φ_d(t_j)` is the p-vector of cross-covariances between `x(t_j)` and `Y_S`:
```
φ_d(t_j)[i] = Σ_k λ_k · φ_k(t_j) · φ_k(argvals[selected[i]])
             = Σ_k λ_k · eigenfunctions[(j, k)] · eigenfunctions[(selected[i], k)]
```

This is a p-vector, NOT a K-vector. To compute the quadratic form:
1. Build `rhs_j[i] = Σ_k λ_k · eigenfunctions[(j,k)] · eigenfunctions[(selected[i],k)]`
2. Solve `Σ_d · v = rhs_j` via `cholesky_solve`
3. Reduction = `dot(rhs_j, v)`

The integrated criterion:
```
MSE(S) = Σ_j w_j · (Σ_k λ_k φ_k(t_j)² − φ_d(t_j)ᵀ Σ_d⁻¹ φ_d(t_j))
```

**Known-answer (p=0):** `φ_d(t_j)` is a 0-vector, reduction = 0, so:
```
MSE(∅) = Σ_j w_j · Σ_k λ_k φ_k(t_j)²
```
Since eigenfunctions are orthonormal w.r.t. Simpson weights:
```
Σ_j w_j φ_k(t_j)² = 1  (by orthonormality)
```
Therefore `MSE(∅) = Σ_k λ_k`. This is **exactly** `sum(model.eigenvalues)`. [ASSUMED — standard FPCA orthonormality; verified by running the empty-set test]

**Implementation note:** The rhs_j vector can be computed as a matrix-vector product: `rhs_j = Φ_d · (λ ⊙ φ_full(t_j))` where `φ_full(t_j) = [φ_1(t_j), ..., φ_K(t_j)]` and `⊙` is elementwise. Computing the p×K product `Φ_d` and caching it once before the grid loop avoids O(m·p·K) re-reads.

### Score criterion (FOD-02)

**Posterior score covariance** (K×K):
```
Cov(ξ|Y_S) = Λ − Λ Φ_dᵀ Σ_d⁻¹ Φ_d Λ
```

Following the A_mat/Ω_i pattern from `pace_fpca.rs:547–558`:
1. For each component k, solve `Σ_d · x_k = Φ_d[:,k]` (p-vector: the k-th eigenfunction at design points)
2. Scale: `sol_k = λ_k · x_k`
3. Build A_mat[k,l] = `λ_k · dot(Φ_d[:,k], sol_l)` = `λ_k · λ_l · Φ_d[:,k]ᵀ Σ_d⁻¹ Φ_d[:,l]`
4. `Cov[k,l] = (k==l ? λ_k : 0) − A_mat[k,l]`

**Known-answer (p=0):** No design points → no information → posterior = prior:
```
Cov(ξ|∅) = Λ  →  A(∅) = trace(Λ) = Σ_k λ_k,  D(∅) = log det(Λ) = Σ_k log(λ_k)
```

**A-optimality:** `trace(Cov) = Σ_k Cov[k,k]` — simple diagonal sum.

**D-optimality:** `log det(Cov)` — computed via `cholesky_factor(Cov, K)` then `log_det_from_cholesky(L, K)`. The result is NEGATIVE (all posterior eigenvalues < prior λ_k). Minimizing log-det = maximizing information = reducing posterior entropy.

**D-opt empty-set fast path:** Avoid Cholesky of K×K Λ (diagonal). Use `Σ_k log(λ_k)` directly. Guard against λ_k ≤ 0 (should not occur for a valid PaceFpcaResult but defensively return `ComputationFailed`).

---

## Known-Answer Test Architecture

### Test 1: Empty-set prior recovery (all three criteria)

```rust
// Synthetic model: 2 orthonormal eigenfunctions on uniform [0,1] grid (m=51)
// λ = [2.0, 1.0], σ² = 0.5
// MSE(∅) must equal 2.0 + 1.0 = 3.0
// A(∅) must equal 3.0
// D(∅) must equal ln(2.0) + ln(1.0) = ln(2) ≈ 0.6931...
assert!((design_criterion(&model, &[], DesignCriterion::Trajectory)? - 3.0).abs() < 1e-10);
assert!((design_criterion(&model, &[], DesignCriterion::Score(OptimalityKind::A))? - 3.0).abs() < 1e-10);
assert!((design_criterion(&model, &[], DesignCriterion::Score(OptimalityKind::D))? - 2.0_f64.ln()).abs() < 1e-10);
```

**Why this works:** On a uniform grid with proper Simpson weights, the orthonormality of eigenfunctions guarantees `Σ_j w_j φ_k(t_j)² = 1`. Build the synthetic model from known orthonormal functions (e.g., Fourier basis cosines) so eigenvalue recovery is exact.

### Test 2: Single-point design reduces MSE below prior

```rust
// Adding any grid point must reduce the trajectory criterion
let mse_empty = design_criterion(&model, &[], DesignCriterion::Trajectory)?;
let mse_one   = design_criterion(&model, &[25], DesignCriterion::Trajectory)?;
assert!(mse_one <= mse_empty + 1e-12);  // monotone non-increasing
```

### Test 3: Monotonicity gate (optimality-sign)

```rust
// criterion(S ∪ {t}) ≤ criterion(S) + 1e-12 for all three criteria
for criterion in [DesignCriterion::Trajectory,
                  DesignCriterion::Score(OptimalityKind::A),
                  DesignCriterion::Score(OptimalityKind::D)] {
    let s0 = design_criterion(&model, &[10], criterion)?;
    let s1 = design_criterion(&model, &[10, 30], criterion)?;
    assert!(s1 <= s0 + 1e-12, "monotonicity violated for {:?}", criterion);
}
```

### Test 4: Score prior exact recovery

```rust
// Cov(ξ|∅) = diag(λ) → trace = Σλ_k
let a_empty = design_criterion(&model, &[], DesignCriterion::Score(OptimalityKind::A))?;
assert!((a_empty - model.eigenvalues.iter().sum::<f64>()).abs() < 1e-10);
// D-opt: Σ log λ_k
let d_empty = design_criterion(&model, &[], DesignCriterion::Score(OptimalityKind::D))?;
let expected_d: f64 = model.eigenvalues.iter().map(|&lam| lam.ln()).sum();
assert!((d_empty - expected_d).abs() < 1e-10);
```

### Test 5: Grid-invariance of MSE(∅)

```rust
// Same synthetic model, different grid densities (m=21, m=51, m=101)
// All must give MSE(∅) ≈ Σλ_k (same value)
// This verifies simpsons_weights is used (not uniform 1/m)
```

### Test 6: Ridge-retry robustness

```rust
// Model with sigma2 = 1e-12 (near-singular regime)
// Should succeed after ridge-retry (not panic)
let result = design_criterion(&near_singular_model, &[10, 20, 30], DesignCriterion::Trajectory);
assert!(result.is_ok());
```

### Test 7: Validation guards

```rust
// Out-of-range index
assert!(design_criterion(&model, &[m], criterion).is_err()); // m == argvals.len()
// sigma2 <= 0 (model constructed artificially)
assert!(design_criterion(&model_sigma2_zero, &[0], criterion).is_err());
// ncomp == 0
assert!(design_criterion(&model_ncomp_zero, &[0], criterion).is_err());
```

---

## Validation Architecture

> `nyquist_validation: true` in `.planning/config.json`.

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in test harness (`#[test]`, `#[cfg(test)]`) |
| Config file | None — uses Cargo's `cargo test` runner |
| Quick run command | `cargo test -p fdars-core --features linalg optimal_design 2>&1` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel 2>&1` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Notes |
|--------|----------|-----------|-------------------|-------|
| FOD-01 | `MSE(∅) = Σ_k λ_k` (grid-invariant) | unit | `cargo test -p fdars-core optimal_design::tests::test_trajectory_empty_set` | Known-answer, no external data |
| FOD-01 | MSE strictly decreases on adding point | unit | `cargo test -p fdars-core optimal_design::tests::test_trajectory_reduces_on_point` | Monotonicity lower bound |
| FOD-01 | Trajectory monotone non-increasing | unit | `cargo test -p fdars-core optimal_design::tests::test_monotonicity_trajectory` | Optimality-sign gate |
| FOD-01 | Grid-invariance: MSE(∅) unchanged for m=21/51/101 | unit | `cargo test -p fdars-core optimal_design::tests::test_trajectory_grid_invariance` | Simpson vs uniform check |
| FOD-02 | `A(∅) = Σ_k λ_k` | unit | `cargo test -p fdars-core optimal_design::tests::test_score_a_empty_set` | Known-answer |
| FOD-02 | `D(∅) = Σ_k log λ_k` (negative) | unit | `cargo test -p fdars-core optimal_design::tests::test_score_d_empty_set` | Sign check |
| FOD-02 | `Cov(ξ|∅) = diag(λ)` exact recovery | unit | `cargo test -p fdars-core optimal_design::tests::test_score_prior_recovery` | A and D together |
| FOD-02 | A-opt monotone non-increasing | unit | `cargo test -p fdars-core optimal_design::tests::test_monotonicity_a_opt` | Optimality-sign gate |
| FOD-02 | D-opt monotone non-increasing | unit | `cargo test -p fdars-core optimal_design::tests::test_monotonicity_d_opt` | Optimality-sign gate |
| FOD-03 | Enum dispatch routes to correct branch | unit | `cargo test -p fdars-core optimal_design::tests::test_enum_dispatch` | All three DesignCriterion variants |
| FOD-03 | Out-of-range index → InvalidParameter | unit | `cargo test -p fdars-core optimal_design::tests::test_validation_index_range` | |
| FOD-03 | sigma2 <= 0 → InvalidParameter | unit | `cargo test -p fdars-core optimal_design::tests::test_validation_sigma2` | |
| FOD-03 | ncomp == 0 → InvalidParameter | unit | `cargo test -p fdars-core optimal_design::tests::test_validation_ncomp` | |
| FOD-03 | Near-singular Σ_d → ridge-retry succeeds | unit | `cargo test -p fdars-core optimal_design::tests::test_ridge_retry` | |
| FOD-03 | lib.rs additive re-export compiles | build | `cargo build -p fdars-core 2>&1` | No existing signatures broken |
| FOD-03 | serde-gated derives compile with feature | build | `cargo build -p fdars-core --features serde 2>&1` | |
| FOD-01/02/03 | Full test suite unbroken | integration | `cargo test -p fdars-core --features linalg,parallel 2>&1` | All 1654+ existing tests still pass |
| FOD-01/02/03 | Clippy clean | lint | `cargo clippy -p fdars-core --all-targets --features linalg,parallel -- -D warnings 2>&1` | CI gate |
| FOD-01/02/03 | fmt clean | lint | `cargo fmt -p fdars-core --check 2>&1` | CI gate |

### Sampling Rate

- **Per task commit:** `cargo test -p fdars-core optimal_design 2>&1`
- **Per wave merge:** `cargo test -p fdars-core --features linalg,parallel 2>&1`
- **Phase gate (before `/gsd-verify-work`):** Full suite green + clippy `--all-targets --features linalg,parallel` + fmt check

### Wave 0 Gaps

- [ ] `fdars-core/src/optimal_design.rs` — new file; covers FOD-01/02/03 + all test functions listed above
- [ ] `lib.rs` additive lines: `pub mod optimal_design;` + re-export of `DesignCriterion`, `OptimalityKind`, `design_criterion`

*(No existing test infrastructure covers optimal_design — all tests are new in Wave 0)*

---

## Common Pitfalls

### Pitfall 1: Wrong Σ_d dimension (K×K instead of |S|×|S|)

**What goes wrong:** Building a K×K matrix `Σ_d` keyed by eigenfunction index instead of a p×p matrix keyed by design-point index.
**Why it happens:** Confusing the two different "design" dimensions: K (number of FPC components) and p (number of design points). The model has both.
**How to avoid:** Always name the dimension `p = selected.len()` and allocate `vec![0.0; p * p]`. The loop is `for row in 0..p { for col in 0..p { ... for k in 0..ncomp { } } }`.
**Warning signs:** `MSE(∅)` returns `K·Σλ_k` instead of `Σλ_k` (the extra factor of K reveals the wrong dimension).

### Pitfall 2: Forgetting σ²I in Σ_d

**What goes wrong:** Σ_d becomes singular (or near-singular) for perfectly spaced grid points where some eigenfunctions are collinear at the design locations.
**Why it happens:** The σ²I ridge term makes `Σ_d` positive-definite by construction. It is easy to copy the outer product `Φ_d diag(λ) Φ_dᵀ` and forget the diagonal addition.
**How to avoid:** Explicitly add after the double loop: `sigma_d[row * p + row] += model.sigma2;`
**Warning signs:** `cholesky_solve` fails immediately (non-positive diagonal at index 0) even for well-conditioned models with σ² = 0.01.

### Pitfall 3: Uniform integration weights in trajectory criterion

**What goes wrong:** `MSE(∅)` returns `(1/m) · Σ_k λ_k · Σ_j φ_k(t_j)²` which is NOT grid-invariant — it changes with `m`.
**Why it happens:** Natural loop `for j in 0..m { mse += integrand_j / m as f64; }`.
**How to avoid:** `let weights = simpsons_weights(&model.argvals);` before the loop. Access `weights[j]` inside.
**Warning signs:** Grid-invariance test (Test 5) fails: MSE(∅) for m=21 ≠ MSE(∅) for m=101.

### Pitfall 4: Sign error on D-optimality

**What goes wrong:** Returning `−log det(Cov)` (positive) instead of `log det(Cov)` (negative), causing the Phase 65 greedy loop to MAXIMIZE uncertainty instead of minimizing it.
**Why it happens:** "Optimality" is associated with maximization in some conventions; the design literature minimizes the criterion.
**How to avoid:** Return `log_det_from_cholesky(&l, K)` directly — it is already negative. Assert in tests: `D(∅) = Σ log λ_k < 0` for all `λ_k < 1`.
**Warning signs:** D(∅) is positive; monotonicity test fails because adding a point increases the returned value.

### Pitfall 5: cholesky_solve on the wrong matrix in the score criterion

**What goes wrong:** Solving `K×K` system instead of `p×p` system — computing `Λ Φ_dᵀ (Φ_d Λ Φ_dᵀ)⁻¹ Φ_d Λ` instead of `Λ Φ_dᵀ (Φ_d diag(λ) Φ_dᵀ + σ²I_p)⁻¹ Φ_d Λ`.
**Why it happens:** Temptation to solve a K×K system (fewer dimensions when K < p), but the correct Woodbury factoring requires the p×p `Σ_d` system.
**How to avoid:** Always call `cholesky_solve(&sigma_d, phi_d_col_k, p)` where `sigma_d` has shape p×p and `phi_d_col_k` has length p.
**Warning signs:** A_mat values blow up (no σ²I regularization); test_score_prior_recovery fails.

### Pitfall 6: Re-solving Σ_d per-grid-point in trajectory criterion (performance)

**What goes wrong:** Not a correctness bug, but O(m·p³) instead of O(m·p²) — calling `cholesky_solve` inside the grid loop recomputes the Cholesky factorization m times.
**How to avoid:** Factor `Σ_d` once via `cholesky_factor`, then call `cholesky_forward_back` inside the loop. Or cache the solution for the whole-grid matrix form.
**Warning signs:** Slow tests; profiling shows Cholesky inside the grid loop.

### Pitfall 7: Using `linalg::cholesky_d` instead of `linalg::cholesky_solve`

**What goes wrong:** `cholesky_d` is a separate lower-level Cholesky variant (uses `1e-0` threshold for "non-positive diagonal", not `1e-12`). It is `pub(crate)` too, but intended for the Mahalanobis helper.
**How to avoid:** Use `cholesky_solve` (the consolidated helper that chains `cholesky_factor` + `cholesky_forward_back`) or factor explicitly with `cholesky_factor` then `cholesky_forward_back`.

---

## Code Examples

### Minimal working `design_criterion` structure (verified patterns from codebase)

```rust
// Source: pace_fpca.rs:461–474, linalg.rs:131–134, helpers.rs:76–105
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::linalg::{cholesky_factor, cholesky_forward_back, cholesky_solve, log_det_from_cholesky};
use crate::pace_fpca::PaceFpcaResult;

/// Which design criterion to compute.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DesignCriterion {
    /// Integrated BLUP trajectory-reconstruction MSE (FOD-01).
    Trajectory,
    /// FPC-score posterior covariance summary (FOD-02).
    Score(OptimalityKind),
}

/// Optimality kind for the Score criterion.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum OptimalityKind {
    /// A-optimality: trace of posterior score covariance.
    A,
    /// D-optimality: log-determinant of posterior score covariance (negative).
    D,
}

/// Score a design point index set against a fitted PACE FPCA model.
///
/// `selected` holds indices into `model.argvals` (0-based). Empty `selected`
/// returns the prior baseline: `Σ_k λ_k` for Trajectory and A, `Σ_k log λ_k` for D.
#[must_use]
pub fn design_criterion(
    model: &PaceFpcaResult,
    selected: &[usize],
    criterion: DesignCriterion,
) -> Result<f64, FdarError> {
    // --- Validation ---
    let m = model.argvals.len();
    if model.ncomp == 0 { return Err(FdarError::InvalidParameter { ... }); }
    if model.sigma2 <= 0.0 { return Err(FdarError::InvalidParameter { ... }); }
    for &idx in selected {
        if idx >= m { return Err(FdarError::InvalidParameter { ... }); }
    }
    // --- Dispatch ---
    match criterion {
        DesignCriterion::Trajectory => trajectory_criterion_impl(model, selected),
        DesignCriterion::Score(kind) => score_criterion_impl(model, selected, kind),
    }
}
```

### Ridge-retry pattern (from pace_fpca.rs:480–490)

```rust
// Source: pace_fpca.rs:480–490
fn cholesky_solve_with_retry(mat: &mut Vec<f64>, rhs: &[f64], p: usize, op: &'static str)
    -> Result<Vec<f64>, FdarError>
{
    match cholesky_solve(mat, rhs, p) {
        Ok(v) => Ok(v),
        Err(_) => {
            for i in 0..p { mat[i * p + i] += 1e-8; }
            cholesky_solve(mat, rhs, p).map_err(|_| FdarError::ComputationFailed {
                operation: op,
                detail: "Cholesky failed after 1e-8 ridge; sigma2 may be too small".into(),
            })
        }
    }
}
```

### lib.rs additive re-export (Phase 64 scope only)

```rust
// In src/lib.rs — additive, no existing lines removed
pub mod optimal_design;
pub use optimal_design::{design_criterion, DesignCriterion, OptimalityKind};
```

---

## Runtime State Inventory

This is a greenfield module addition — no rename/refactor involved.

**Nothing found in any category** — verified by inspection: `optimal_design.rs` does not exist yet; no stored data, no live service config, no OS state, no secrets, no build artifacts referencing this module.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable toolchain | Build | ✓ | 1.97.0 (> MSRV 1.81) | — |
| `cargo test` | Test runner | ✓ | built-in | — |
| `cargo clippy` | Lint gate | ✓ | built-in | — |
| `cargo fmt` | Fmt gate | ✓ | built-in | — |
| `linalg::cholesky_solve` | Σ_d solve | ✓ | always compiled | — |
| `helpers::simpsons_weights` | Quadrature | ✓ | always compiled | — |
| `pace_fpca::PaceFpcaResult` | Model source | ✓ | shipped in crate | — |

**Missing dependencies with no fallback:** None — all required components are in-crate.

**disk/tmp pressure:** MEMORY.md flags `/tmp` exhaustion with doctests and `target/` filling `/home`. This phase adds one new file and no new examples or benchmarks (benchmark deferred to Phase 65). Risk is LOW. If `cargo test` fails with "No space left", run `rm -rf target/debug/{incremental,examples}` before retrying.

---

## Security Domain

> `security_enforcement: true` in config.json; `security_asvs_level: 1`.

### Applicable ASVS Categories

| ASVS Category | Applies | Control |
|---------------|---------|---------|
| V2 Authentication | No | Pure library function, no auth surface |
| V3 Session Management | No | Stateless computation |
| V4 Access Control | No | No resource gating |
| V5 Input Validation | Yes | Index range, sigma2 > 0, ncomp > 0 — all gated at `design_criterion` entry |
| V6 Cryptography | No | No cryptographic operations |

### Known Threat Patterns

| Pattern | STRIDE | Mitigation |
|---------|--------|------------|
| Integer overflow in index arithmetic | Tampering | All indices are `usize`; bounds-checked at entry |
| NaN propagation from degenerate eigenvalues | Tampering | sigma2 > 0 guard prevents near-singular prior; ridge-retry prevents Cholesky NaN |
| Panic on empty slice in simpsons_weights | DoS | `simpsons_weights` returns `vec![1.0; n]` for n < 2 (`[VERIFIED: src/helpers.rs:79]`); entry guard requires `m >= 2` via argvals length check |

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Separate design functions per criterion | Single `design_criterion` dispatcher with nested enum | Cleaner Phase 65 integration; independently usable |
| fdapace `FOptDes` MATLAB implementation | Rust reuse of existing `pace_fpca.rs` numerical patterns | No new math; just isolation of the design-point sub-problem |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Eigenfunctions in `PaceFpcaResult` are orthonormal w.r.t. `simpsons_weights(&model.argvals)`, so `MSE(∅) = Σλ_k` exactly | Math / Known-answer Tests | Empty-set test would fail; need tolerance adjustment |
| A2 | `log_det_from_cholesky` in `linalg.rs` is correct for the K×K posterior covariance Cholesky (using `cholesky_factor`, not `cholesky_d`) | Code Examples | D-opt value would be wrong; test would catch it |
| A3 | `cholesky_solve` singular threshold (`1e-12`) is sufficient for well-conditioned PACE models with σ² ≥ 0.001; the ridge-retry (`1e-8`) covers the σ² → 0 edge case | Pitfalls | Cholesky fails on legitimate inputs; covered by ridge-retry test |

**A1 note:** The PACE eigenfunctions from `pace_fpca.rs` are computed from the Simpson-weighted symmetric eigendecomposition `W^{½} Ĝ W^{½}`, so they are orthonormal in the weighted L² sense by construction. [CITED: pace_fpca.rs module doc, lines 7–8: "symmetric eigendecomposition of W^{½} Ĝ W^{½} (Simpson-weighted) to obtain functional eigenvalues λ_k and orthonormal eigenfunctions φ_k"] The empty-set test uses a synthetic model with eigenfunctions defined to be exactly orthonormal under the grid's Simpson weights — this avoids dependency on the PACE estimation pipeline for the unit test.

---

## Open Questions

1. **Should Φ_d be pre-built once or extracted per-call inside `build_sigma_design`?**
   - What we know: Both criterion branches need the p×K matrix `Φ_d[i, k] = eigenfunctions[(selected[i], k)]`.
   - What's unclear: Whether to build it once in `design_criterion` and pass to helpers, or re-extract inside each helper.
   - Recommendation: Build once in `design_criterion` (or in `build_sigma_design` return it alongside Σ_d as a tuple) to avoid the O(p·K) re-read per criterion branch. Private helper can return `(sigma_d, phi_d)` or take `phi_d` as a pre-computed input.

2. **Empty-set fast-path for D-optimality: use `Σ_k log λ_k` directly or factor K×K Λ?**
   - What we know: K×K diagonal Λ Cholesky is trivial but unnecessary.
   - What's unclear: Whether the factoring overhead matters for typical K ≤ 5.
   - Recommendation: Use `model.eigenvalues.iter().map(|&lam| lam.ln()).sum()` directly — simpler, avoids Cholesky of Λ, and the empty-set guard already handles `p == 0` before reaching any Cholesky call.

---

## Sources

### Primary (HIGH confidence)

- `[VERIFIED: fdars-core/src/pace_fpca.rs:99–120]` — PaceFpcaResult field names and types (eigenvalues Vec<f64> ncomp, eigenfunctions FdMatrix m×ncomp column-major, argvals Vec<f64> m, sigma2 f64, ncomp usize)
- `[VERIFIED: fdars-core/src/pace_fpca.rs:461–474]` — Σ_yi assembly verbatim (row-major, Φ diag(λ) Φᵀ + σ²I pattern)
- `[VERIFIED: fdars-core/src/pace_fpca.rs:480–490]` — Ridge-retry pattern verbatim
- `[VERIFIED: fdars-core/src/pace_fpca.rs:547–558]` — A_mat/Ω_i posterior-covariance assembly verbatim
- `[VERIFIED: fdars-core/src/linalg.rs:131–134]` — `cholesky_solve(a, b, p)` signature and availability (not feature-gated)
- `[VERIFIED: fdars-core/src/linalg.rs:63–69]` — `log_det_from_cholesky(l, d)` formula: `2·Σ ln(L_ii)`
- `[VERIFIED: fdars-core/src/linalg.rs:85–108]` — `cholesky_factor(a, p)` signature and 1e-12 threshold
- `[VERIFIED: fdars-core/src/helpers.rs:76–105]` — `simpsons_weights(&argvals)` signature and fallback for n<2
- `[VERIFIED: fdars-core/src/error.rs:1–51]` — FdarError variants: InvalidDimension, InvalidParameter, ComputationFailed, InvalidEnumValue
- `[VERIFIED: fdars-core/src/matrix.rs:40–44]` — FdMatrix column-major layout: element (row, col) at `row + col * nrows`
- `[VERIFIED: fdars-core/src/lib.rs:105–139]` — existing `pub mod` declarations; `pace_fpca`, `kshape`, `kernel_kmeans` are direct peers at this level
- `[VERIFIED: fdars-core/src/kshape.rs:67–70]` — KShapeConfig uses `#[non_exhaustive]` (config convention)
- `[VERIFIED: fdars-core/src/kshape.rs:112–114]` — KShapeResult uses `#[non_exhaustive]` + `Debug, Clone, PartialEq` + serde-gated
- `[VERIFIED: .planning/config.json:25]` — `nyquist_validation: true`
- `[VERIFIED: .planning/config.json:47]` — `security_enforcement: true`

### Secondary (MEDIUM confidence)

- `[CITED: pace_fpca.rs module doc, lines 1–32]` — PACE pipeline description, orthonormality claim for eigenfunctions under Simpson weights, σ² design note
- `[CITED: 64-CONTEXT.md decisions section]` — Locked API contracts and known-answer values (verified against STATE.md)

### Tertiary (LOW confidence, marked ASSUMED where used)

- `[ASSUMED]` — Eigenfunctions from a synthetic orthonormal model will satisfy `Σ_j w_j φ_k(t_j)² ≈ 1` exactly when built from closed-form functions — requires test verification.

---

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — all components read from source files this session
- Math formulas: HIGH — transcribed verbatim from pace_fpca.rs with cross-check against CONTEXT.md
- Architecture: HIGH — direct peer-module pattern (kshape.rs, kernel_kmeans.rs) read this session
- Pitfalls: HIGH — derived from code reading (actual linalg signatures, actual helper behavior)
- Known-answer test values: HIGH (empty-set formulas algebraically exact) + A1 assumption

**Research date:** 2026-09-02
**Valid until:** 2026-10-02 (stable codebase; no external dependencies to drift)
