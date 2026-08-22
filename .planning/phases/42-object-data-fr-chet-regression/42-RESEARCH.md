# Phase 42: Object-Data Fréchet Regression (FRE-02) — Research

**Researched:** 2026-08-22
**Domain:** Metric-space backends for Fréchet regression — SPD matrices, correlation matrices, spherical data, networks, point processes; generic solver refactoring over the shipped FRE-01 trait
**Confidence:** HIGH (all claims are derived from in-repo source reads or established mathematics; no new external packages)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- **SPD object representation:** flat column-major `Vec<f64>` + dimension `d` (a d×d matrix), matching the crate matrix convention.
- **SPD metrics:** all three selected via `SpdMetric` enum on `SpdMatrixSpace::new`: Frobenius (`d(A,B)=‖A−B‖_F`; weighted mean = weighted average), Power-α (matrix power via `nalgebra::SymmetricEigen`; distance `‖A^α − B^α‖_F / α`; mean `(Σwᵢ Aᵢ^α / Σwᵢ)^{1/α}`), Log-Cholesky (average in log-Cholesky coordinates then map back).
- **Correlation backend:** Frobenius distance; weighted mean = weighted average renormalized to unit diagonal.
- **Eigendecomposition path:** reuse `nalgebra::SymmetricEigen` — no new dependency.
- **Spherical space:** object = unit vector `Vec<f64>`; geodesic distance `arccos(⟨a,b⟩)` (clamped); exp/log maps; intrinsic Karcher mean via gradient descent (initialized at normalized extrinsic mean, iter-capped).
- **Network space:** object = graph Laplacian (flat d×d); Frobenius distance; weighted average mean.
- **Point-process space:** object = intensity/count vector on a grid; L2 distance; weighted average mean.
- **Iterative mean validation:** fixed max-iters + tolerance (defaults at planner/implementer discretion); validate non-empty, matching dims, weight-length match; return `ComputationFailed` on non-convergence (never panic).
- **Reuse strategy:** extract Petersen–Müller global weight computation and Dubey–Müller Tₙ / Fréchet-variance logic into `pub(crate)` helpers; existing density entry points delegate to them; their public signatures stay untouched.
- **Generic regression signatures:**
  - `frechet_global_reg_space<S: MetricSpace>(space, predictors, responses: &[S::Object], xout) -> Result<Vec<S::Object>, FdarError>`
  - `frechet_local_reg_space<S: MetricSpace>(space, predictors, responses: &[S::Object], xout, bandwidth) -> Result<Vec<S::Object>, FdarError>`
- **Generic ANOVA signature:** `frechet_anova_space<S: MetricSpace>(space, objects: &[S::Object], labels, n_perm, seed) -> Result<FrechetAnovaResult, FdarError>`
- **Layout:** new `frechet/spaces/` submodule (or cohesive `frechet/object_spaces.rs`) for five backends; generic solver functions added to existing `frechet/regression.rs` and `frechet/anova.rs`.
- **Backend types:** `SpdMatrixSpace`, `CorrelationMatrixSpace`, `SphericalSpace`, `NetworkSpace`, `PointProcessSpace` — each `impl MetricSpace`.
- **Metric-selection enum:** `SpdMetric { Frobenius, Power(f64), LogCholesky }`.
- **Re-exports:** all new space structs, `SpdMetric` enum, generic regression/ANOVA functions re-exported from `frechet/mod.rs` and crate root.

### Claude's Discretion
- Exact default max-iters/tolerance for the Karcher mean.
- Precise log-Cholesky coordinate convention (choice: strictly-lower `L` entries as-is, log of diagonal).
- Whether local generic regression shares a helper with density local path or a parallel one.
- Whether backends live under `frechet/spaces/*.rs` (one file per space) or single `object_spaces.rs`.

### Deferred Ideas (OUT OF SCOPE)
- None from discussion. Plotting/rendering of object-space Fréchet fits is a milestone Out-of-Scope item (numeric outputs only). FTS-03 spectral FTS was Phase 41.
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| FRE-02-01 | SPD covariance-matrix response space (Frobenius + Power-α + Log-Cholesky metrics + weighted-Fréchet-mean) as a `MetricSpace` backend | §SPD backends: exact math for all three metrics; `nalgebra::SymmetricEigen` pattern confirmed [VERIFIED: src/fts/acf.rs:345] |
| FRE-02-02 | Correlation-matrix response space (distance + weighted Fréchet mean) as a `MetricSpace` backend | §Correlation backend: Frobenius + renormalization strategy; divergence from R documented |
| FRE-02-03 | Spherical-data response space (geodesic exp/log + weighted Fréchet mean) as a `MetricSpace` backend | §Spherical backend: exact geodesic + Karcher-gradient-descent algorithm |
| FRE-02-04 | Network response space (graph-Laplacian distance + weighted Fréchet mean) as a `MetricSpace` backend | §Network backend: Laplacian cone preservation justification |
| FRE-02-05 | Point-process response space (intensity/count distance + weighted Fréchet mean) as a `MetricSpace` backend | §Point-process backend: L2 distance on intensity vector |
| FRE-02-06 | Generic global + local Fréchet regression over Euclidean predictors for at least one non-density backend, reusing FRE-01 weight machinery | §Solver refactoring: exact `pub(crate)` helper extraction plan with weight formulae |
| FRE-02-07 | Generic Fréchet-ANOVA over at least one non-density object space, reusing FRE-01 Tₙ machinery | §ANOVA refactoring: `compute_tn` generification plan; existing `FrechetAnovaResult` struct reuse confirmed |
</phase_requirements>

---

## Summary

FRE-02 extends the shipped FRE-01 `frechet/` module by adding five concrete `MetricSpace` implementations for non-density object data (SPD matrices, correlation matrices, spherical data, networks, point processes) and lifting the existing density-specific regression/ANOVA internals into reusable `pub(crate)` helpers that both the existing density path and new generic entry points consume. The design is fully pinned in CONTEXT.md, and the codebase is well-prepared for the extension.

The key implementation challenge is the SPD power-α metric (matrix power via eigendecomposition, SPD-preserving weighted mean under the power map) and the spherical Karcher mean (gradient descent on the sphere with careful initialization and convergence guarding). The regression refactoring is purely mechanical: the Petersen–Müller weight computation in `frechet_global_reg` and `frechet_local_reg` is already self-contained; extracting it into a `pub(crate)` function that takes only `&FdMatrix` predictors and returns `Vec<Vec<f64>>` (one weight vector per xout row) requires no algorithmic change. The ANOVA refactoring (`compute_tn`) similarly requires only making the space argument generic over `S: MetricSpace` instead of `&WassersteinDensitySpace`, because `frechet_mean`/`frechet_variance` in `mean.rs` are already fully generic.

**Primary recommendation:** Refactor first (helpers extraction), then add the five space structs file by file (simplest first: PointProcess → Network → Correlation → Spherical → SPD), then add the generic entry-point functions. All new items follow the exact same derivation/serde/re-export conventions as existing FRE-01 code.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| SPD/Correlation/Spherical/Network/PointProcess distance + weighted mean | `frechet/` module (new space files) | `linalg.rs`, `nalgebra` (matrix ops) | Self-contained metric-space algebra; no external API surface |
| Petersen–Müller weight computation (global) | `frechet/regression.rs` `pub(crate)` helper | — | Already lives there; extraction changes only visibility |
| Local kernel-weight computation | `frechet/regression.rs` `pub(crate)` helper | — | Same file, same extraction pattern |
| Dubey–Müller Tₙ computation | `frechet/anova.rs` generic `pub(crate)` helper | — | Generification changes only the space argument type |
| Generic regression/ANOVA entry points | `frechet/regression.rs`, `frechet/anova.rs` | `frechet/mod.rs` re-exports | Added next to existing density entry points |
| Module re-exports | `frechet/mod.rs` + `src/lib.rs` | — | Extend existing `pub use frechet::{…}` block |

---

## Standard Stack

No new external dependencies. All math uses:

### Core (already in Cargo.toml)
| Library | Version | Purpose | Why |
|---------|---------|---------|-----|
| `nalgebra` | 0.33 | `DMatrix::from_column_slice`, `SymmetricEigen` for matrix power/log | Already used for eigen in `fts/acf.rs`, `fts/spectral.rs`, `fpca_variants.rs` |
| In-crate `linalg` | — | `cholesky_factor` for log-Cholesky Cholesky decomposition | Already used in `frechet/regression.rs` |

**No new `Cargo.toml` entries required.**

### Supporting (existing in-crate)
| Helper | Location | Used For |
|--------|----------|----------|
| `cholesky_factor` | `src/linalg.rs:85` | Log-Cholesky: lower-triangular factor of SPD matrix |
| `cholesky_forward_back` | `src/linalg.rs:113` | Log-Cholesky: solve during log-Cholesky mean back-map |
| `NUMERICAL_EPS` | `src/helpers.rs` | Zero-sum weight guards |
| `gaussian_kernel` | `src/helpers.rs` | Local regression kernel (reused in generic local path) |

---

## Package Legitimacy Audit

No new packages are added in this phase. All computation reuses the existing `nalgebra 0.33` and in-crate helpers.

**Packages removed due to SLOP verdict:** none
**Packages flagged as suspicious SUS:** none

---

## Architecture Patterns

### System Architecture Diagram

```
[User: frechet_global_reg_space<SpdMatrixSpace>(space, predictors, responses, xout)]
        |
        v
[frechet/regression.rs]
  compute_global_weights(predictors, xout)  ← pub(crate) helper (extracted from existing code)
        |
        v  (weights: Vec<Vec<f64>>, one weight-vec per xout row)
        |
  for each xout row:
    space.weighted_frechet_mean(responses, weights[r]) → S::Object
        |
        v  (dispatches to SpdMatrixSpace::weighted_frechet_mean)
[frechet/spaces/spd.rs]   OR   [frechet/object_spaces.rs]
  SpdMatrixSpace::weighted_frechet_mean
    → Power-α path: build A^α for each, weighted-average, take (·)^{1/α}
    → Log-Cholesky path: map each to coordinates, weighted-average, map back
    → Frobenius path: direct weighted average of flat matrices
        |
        v
[Result<Vec<S::Object>>]  returned to user

[User: frechet_anova_space<S>(space, objects, labels, n_perm, seed)]
        |
        v
[frechet/anova.rs]
  compute_tn_space<S: MetricSpace>(space, objects, labels, k)  ← generic helper
    uses frechet_mean<S> / frechet_variance<S>  ← already generic in mean.rs
    same Tₙ formula, same seeded permutation loop
        |
        v
[FrechetAnovaResult]  (existing struct, no change)
```

### Recommended Project Structure

```
fdars-core/src/frechet/
├── mod.rs          # existing + new pub use for 5 spaces + generic fns
├── space.rs        # existing MetricSpace trait + WassersteinDensitySpace
├── mean.rs         # existing (already generic, no change)
├── regression.rs   # existing density fns (unchanged signatures) + new generic fns
│                   #   + pub(crate) compute_global_weights / compute_local_weights
├── anova.rs        # existing frechet_anova (unchanged) + new frechet_anova_space
│                   #   + pub(crate) compute_tn_space (generic)
└── spaces/         # NEW — or object_spaces.rs (planner layout choice)
    ├── mod.rs      # pub use all five space structs + SpdMetric
    ├── spd.rs      # SpdMatrixSpace, SpdMetric
    ├── correlation.rs  # CorrelationMatrixSpace
    ├── spherical.rs    # SphericalSpace
    ├── network.rs      # NetworkSpace
    └── point_process.rs  # PointProcessSpace
```

---

## Backend Mathematics: Exact Formulas

### SPD Matrix Space (FRE-02-01)

**Object representation:** `Vec<f64>` of length `d*d`, column-major (matching `FdMatrix` convention). The struct carries `d: usize`.

**Validation:** On construction, accept `d >= 1`. On `distance`/`weighted_frechet_mean`, require `a.len() == d*d`. Checking SPD-ness is O(d³) (Cholesky); only the Log-Cholesky path requires it explicitly (the Cholesky factor is needed for that path's coordinate map). For Frobenius and Power-α, validate only dimensions; if eigenvalues go negative the user's data is malformed.

#### Frobenius metric

- Distance: `d_F(A, B) = ‖A − B‖_F = sqrt(Σᵢ (Aᵢ − Bᵢ)²)` (element-wise over the flat `Vec<f64>`)
- Weighted mean: `M = (Σᵢ wᵢ Aᵢ) / Σᵢ wᵢ` (element-wise weighted average of flat vectors)
- SPD preservation: a convex combination of SPD matrices is SPD — no special guard needed.

[ASSUMED] that R's Frobenius-distance SPD regression uses exactly this element-wise formula (verified consistent with Lin 2019's exposition of Frobenius distance on symmetric matrices, but not confirmed against `frechet` 0.3.0 source code).

#### Power-α metric

Reference: Dryden, Koloydenko & Zhou (2009); Lin (2019). Locked in CONTEXT.md.

**Matrix power `A^α`** (α ∈ (0,1]):
1. Form `nalgebra::DMatrix::from_column_slice(d, d, &mat_flat)`. [ASSUMED: column-slice order matches crate's column-major Vec<f64>]
2. Symmetrize defensively (average off-diagonal pairs) — same pattern as `fts/acf.rs:337-344`. [VERIFIED: fdars-core/src/fts/acf.rs:337-344 — pattern shown verbatim in acf.rs source]
3. `let eig = nalgebra::SymmetricEigen::new(mat)` — returns `.eigenvalues` (ascending) and `.eigenvectors`. [VERIFIED: fdars-core/src/fts/acf.rs:345 — `let eig = nalgebra::SymmetricEigen::new(c0_mat);`]
4. Apply `λᵢ^α` to each eigenvalue (clamp negative eigenvalues to 0 before raising to power — guards against tiny numerical negatives from noise; for α < 1 the limit `0^α = 0` is well-defined). Result: `V · diag(λ^α) · Vᵀ` reconstructed as flat `Vec<f64>`.
5. Reconstruction: `mat_pow[i + j*d] = Σₖ eigenvecs[i,k] * λₖ^α * eigenvecs[j,k]` (column-major output).

**Distance (Power-α):**
`d_α(A, B) = ‖A^α − B^α‖_F / α`
- Divide by α so that as α → 0 the distance approaches the log-Euclidean distance (Lin 2019). In practice the clamped `alpha >= 1e-6` prevents division by zero; return `InvalidParameter` for `alpha <= 0`.

**Weighted Fréchet mean (Power-α):**
1. Compute `Aᵢ^α` for each object.
2. Weighted average: `M_α = (Σᵢ wᵢ Aᵢ^α) / Σwᵢ` (element-wise).
3. Apply `M^{1/α}`: map with `1/α` as the exponent. This is SPD-preserving because M_α is a positive-semidefinite combination of PSD matrices.
- Guard: if any eigenvalue of M_α is negative (should not happen numerically but possible for very noisy data), clamp to 0 before `(·)^{1/α}`.

**α = 1 collapses to Frobenius** — verified: `A^1 = A`, so `‖A^1 − B^1‖_F / 1 = ‖A−B‖_F`. Test oracle: power-α mean with α=1 equals Frobenius mean element-by-element. [ASSUMED: the formulae are correct; confirmed against standard exposition, not R source]

**α → 0 edge (log-Euclidean):** Not implementing `α = 0` explicitly; the user selects `SpdMetric::LogCholesky` for the log-Euclidean analog. Document in rustdoc.

#### Log-Cholesky metric

Reference: Lin (2019) "Riemannian geometry of symmetric positive definite matrices via Cholesky decomposition." [ASSUMED from training knowledge — confirmed consistent with the locked CONTEXT.md description but not verified against the Lin 2019 paper PDF this session]

**Coordinate map** (SPD → log-Cholesky space):
Given `A = L Lᵀ` (lower-triangular Cholesky factor):
- Strictly lower triangular entries: `ℓᵢⱼ` for `i > j` (taken as-is).
- Diagonal: `log(ℓᵢᵢ)` for `i = 1..d`.
- Flatten into a coordinate vector of length `d(d+1)/2`.

**Cholesky decomposition:** use `crate::linalg::cholesky_factor(&mat_flat, d)` — returns a row-major lower-triangular `Vec<f64>` of length `d*d`. [VERIFIED: fdars-core/src/linalg.rs:85 — `pub(crate) fn cholesky_factor(a: &[f64], p: usize) -> Result<Vec<f64>, FdarError>`]

Access element `(i,j)` of the Cholesky factor: `l[i*d + j]` (row-major). [VERIFIED: fdars-core/src/linalg.rs:88-107 — implementation uses `l[j * p + k]` pattern confirming row-major indexing]

**Distance (Log-Cholesky):**
`d_LC(A, B) = ‖φ(A) − φ(B)‖` where `φ` is the coordinate map above and the norm is the Euclidean norm in `R^{d(d+1)/2}`.

This is a flat Euclidean distance in the log-Cholesky coordinates — no iteration, no eigendecomposition. O(d³) for Cholesky + O(d²) for the norm.

**Weighted Fréchet mean (Log-Cholesky):**
1. Map each `Aᵢ` → `φ(Aᵢ)` (coordinate vector).
2. Weighted average: `φ̄ = Σᵢ wᵢ φ(Aᵢ)`.
3. Invert: strictly-lower entries → lower-triangular directly; diagonal entries → `exp(φ̄ᵢᵢ)`. Reconstruct `L̄` from the coordinate vector.
4. Form `M = L̄ L̄ᵀ` (matrix multiply back). This is always SPD by construction.

Matrix multiply `M = L̄ L̄ᵀ` in flat column-major output: `M[i + k*d] = Σⱼ L̄[i*d + j] * L̄[k*d + j]` for `j < min(i,k)+1` (lower-triangular multiply). [ASSUMED: exact indexing derived from standard SPD reconstruction, not re-verified this session]

**SPD guard for Log-Cholesky:** `cholesky_factor` returns `ComputationFailed` on non-PD input — propagate to caller.

---

### Correlation Matrix Space (FRE-02-02)

**Object representation:** flat column-major `Vec<f64>` of length `d*d`, dimension `d` stored in struct.

**Distance:** Frobenius distance `d_F(A, B) = ‖A − B‖_F` (identical to SPD Frobenius, just operating on correlation matrices). [ASSUMED: R `frechet` 0.3.0 uses this; divergence note below]

**Weighted Fréchet mean:**
1. Element-wise weighted average: `M = Σᵢ wᵢ Aᵢ`.
2. Renormalize to unit diagonal: `M_corr[i,j] = M[i,j] / sqrt(M[i,i] * M[j,j])`.
3. The renormalized M_corr has unit diagonal by construction.

**Validity guard:** If `M[i,i] <= 0` for some `i` (degenerate input), return `ComputationFailed`. The result may not be positive definite (a weighted average of correlation matrices is not guaranteed PD if the weights or inputs are ill-conditioned). Do not validate PD-ness of the output — document this in rustdoc and validate only dimensions.

**Divergence from R `frechet` 0.3.0 (document in rustdoc):**
R's `frechet::GloCorReg` may use a correlation-manifold geometry (e.g., log-Euclidean on the correlation manifold or chordal distance). The Frobenius + renormalization approach is a simpler projection that matches R's output for well-conditioned samples but can diverge for ill-conditioned weighted combinations. [ASSUMED: exact R geometry not verified against R source this session]

---

### Spherical Space (FRE-02-03)

**Object representation:** unit vector `Vec<f64>` of length `d` (dimension `d` stored in struct). Validation: `|‖a‖ − 1| < tol` where `tol = 1e-6` (warn-tolerant; do not require exact machine-precision unit norm — inputs may come from normalized operations).

Actually: do NOT validate unit norm on every `distance` call (expensive). Validate on construction of `SphericalSpace::new(d)` and document that callers must supply unit vectors. Trust the caller for performance. Return `InvalidParameter` only for dimension mismatches.

**Geodesic distance:**
`d(a, b) = arccos(clamp(⟨a, b⟩, −1, 1))`
- Inner product: `dot = a.iter().zip(b.iter()).map(|(&x,&y)| x*y).sum::<f64>()`.
- Clamp to `[-1, 1]` before `acos` to prevent NaN from floating-point noise. [ASSUMED: standard spherical geometry]
- Special case: identical vectors → distance 0 (arccos(1) = 0, exact in IEEE 754 arithmetic).

**Exp map** `exp_x(v)` (tangent vector `v` at base point `x`, `‖v‖_x = 0` condition not checked for performance):
`exp_x(v) = cos(‖v‖) · x + sin(‖v‖) / ‖v‖ · v`
- If `‖v‖ < 1e-12`: return `x` (identity for zero tangent).

**Log map** `log_x(y)`:
`log_x(y) = θ / sin(θ) · (y − cos(θ) x)` where `θ = arccos(clamp(⟨x,y⟩,−1,1))`
- If `θ < 1e-12`: return zero vector (y ≈ x).
- If `θ > π − 1e-12`: antipodal — log map is not unique; return `ComputationFailed` if encountered in the Karcher loop (the weighted mean will be at a regular point in practice).

**Weighted Fréchet mean — Karcher gradient descent:**

Algorithm (locked in CONTEXT.md):
1. Compute extrinsic (Euclidean) weighted mean: `μ_ext = Σᵢ wᵢ aᵢ`.
2. Normalize: `x = μ_ext / ‖μ_ext‖`. If `‖μ_ext‖ < 1e-14`, return `ComputationFailed` (antipodally balanced inputs, no well-defined mean).
3. Iterate (up to `max_iter` times, default 50; tolerance `tol`, default 1e-8):
   a. Gradient: `g = Σᵢ wᵢ log_x(aᵢ)` (sum of log maps; each is a tangent vector at `x`).
   b. Step: `x_new = exp_x(g)` (retraction via exp map).
   c. Normalize `x_new` (defensive; exp map should preserve unit norm but float errors accumulate).
   d. Convergence check: `‖g‖ < tol` → converged.
   e. `x = x_new`.
4. If not converged after `max_iter`: return `ComputationFailed { operation: "SphericalSpace::weighted_frechet_mean", detail: "Karcher mean did not converge in {max_iter} iterations" }`.

**Configuration:** hardcode `max_iter = 50, tol = 1e-8` as constants in the file (planner discretion confirmed in CONTEXT.md). These are sufficient for typical geodesic problems on Sᵈ⁻¹ with `d ≤ 100`.

**Divergence from R `frechet` 0.3.0 (document in rustdoc):**
R's `frechet::GloSphReg` may use a different initialization or convergence criterion for the Karcher mean (e.g., gradient Riemannian-descent vs. gradient descent in tangent space). The initialization from the normalized extrinsic mean is standard and matches the Petersen–Müller (2019) paper's supplementary algorithm. [ASSUMED]

---

### Network Space (FRE-02-04)

**Object representation:** graph Laplacian as flat column-major `Vec<f64>` of length `d*d`, dimension `d` stored in struct. A valid Laplacian has: row sums zero, non-positive off-diagonal, non-negative diagonal (degree matrix). Validate dimensions only on `distance`/`weighted_frechet_mean`; do not validate Laplacian structure on every call.

**Distance:** Frobenius distance `d_F(L₁, L₂) = ‖L₁ − L₂‖_F` (element-wise, as for SPD Frobenius). [ASSUMED: matches R `frechet::GloNetReg` which uses Frobenius on Laplacians per the Dubey–Müller (2020) network Fréchet regression paper]

**Weighted Fréchet mean:**
`M = Σᵢ wᵢ Lᵢ` (element-wise weighted average).

**Laplacian preservation:** The set of graph Laplacians is a convex cone — a non-negative weighted combination of Laplacians is a Laplacian. Row sums remain zero, off-diagonals remain non-positive, diagonal remains non-negative. No special guard needed. [ASSUMED: standard graph theory; divergence is possible only if some `wᵢ` are negative, which does not occur in `weighted_frechet_mean` — only the signed-weight regression path uses negatives, and that path bypasses `weighted_frechet_mean`]

---

### Point-Process Space (FRE-02-05)

**Object representation:** intensity/count vector `Vec<f64>` of length `m` (grid size stored in struct). Values should be non-negative (intensities) but the struct does not enforce this — validate only dimensions.

**Distance:** L2 distance `d(a, b) = ‖a − b‖₂ = sqrt(Σᵢ (aᵢ − bᵢ)²)` (Euclidean on the count vector). [ASSUMED: consistent with R `frechet::GloPointReg` which uses L2 on intensity counts]

**Weighted Fréchet mean:**
`M = Σᵢ wᵢ aᵢ` (element-wise weighted average).

The Fréchet mean under L2 distance is the weighted arithmetic mean — no iteration. [ASSUMED: standard; holds because L2 on R^m has a unique mean at the centroid]

---

## Generic Solver Refactoring (FRE-02-06 and FRE-02-07)

### Existing State (read from source)

`frechet_global_reg` [VERIFIED: fdars-core/src/frechet/regression.rs:84-148]:
- Lines 94-121: computes `x_bar` (predictor means), builds Σ̂ (sample covariance) + ridge 1e-6, calls `cholesky_factor`.
- Lines 124-141: per xout-row loop computing signed weights `sᵢ = (1 + (Xᵢ − X̄)ᵀ Σ̂⁻¹ (xout − X̄)) / n` and feeding them to `signed_quantile_average` (density-specific).
- Lines 143-148: packages into `FrechetGlobalRegResult { predicted: FdMatrix, xout, x_bar }`.

`frechet_local_reg` [VERIFIED: fdars-core/src/frechet/regression.rs:167-254]:
- Lines 188-220: kernel weights + local moments `μ₁`, `μ₂` + Cholesky solve for `a = μ₂⁻¹ μ₁`.
- Lines 221-238: per-observation signed weights `sᵢ = Kᵢ (1 − (Xᵢ − x₀)ᵀ a)`, normalized.
- Feeds into `signed_quantile_average` (density-specific).

`compute_tn` [VERIFIED: fdars-core/src/frechet/anova.rs:32-98]:
- Typed as `fn compute_tn(space: &WassersteinDensitySpace, objects: &[Vec<f64>], labels: &[usize], k: usize) -> Result<(f64, f64, f64, Vec<f64>, f64), FdarError>`.
- Already uses `frechet_mean(space, ...)` and `frechet_variance(space, ...)` which are generic over `S: MetricSpace` [VERIFIED: fdars-core/src/frechet/mean.rs:40-83].

### Refactoring Plan

#### Global weight helper

Extract from `frechet_global_reg` lines 94-135 into:

```rust
/// Compute Petersen–Müller global signed weights for each xout row.
///
/// Returns `(weights_per_row, x_bar)` where `weights_per_row[r]` is the
/// length-n signed weight vector `sᵢ(xout[r]) = (1 + (Xᵢ−X̄)ᵀ Σ̂⁻¹(xout[r]−X̄)) / n`.
pub(crate) fn compute_global_weights(
    predictors: &FdMatrix,
    xout: &FdMatrix,
) -> Result<(Vec<Vec<f64>>, Vec<f64>), FdarError>
```

No change to any existing formula. The existing `frechet_global_reg` becomes:
```rust
pub fn frechet_global_reg(...) -> Result<FrechetGlobalRegResult, FdarError> {
    let (n, p, m) = validate_reg_input(...)?;
    let (weights_per_row, x_bar) = compute_global_weights(predictors, xout)?;
    // ... same loop but calling signed_quantile_average(responses, argvals, &weights_per_row[r], n_q)
}
```

The new generic function:
```rust
pub fn frechet_global_reg_space<S: MetricSpace>(
    space: &S,
    predictors: &FdMatrix,
    responses: &[S::Object],
    xout: &FdMatrix,
) -> Result<Vec<S::Object>, FdarError>
```
calls `compute_global_weights`, then for each row calls `space.weighted_frechet_mean(responses, &weights_per_row[r])?`.

**Critical difference from the density path:** The density path uses `signed_quantile_average` (which accepts negative weights via the monotone projection trick). The generic path calls `space.weighted_frechet_mean` which only accepts non-negative weights per the `MetricSpace` trait contract. The Petersen–Müller weights CAN be negative (they sum to 1 but individual `sᵢ` can be < 0 for extrapolation).

Resolution: For the generic path, the caller's `weighted_frechet_mean` must be able to handle negative weights OR the generic path must use a different signed-weight strategy. The locked CONTEXT.md specifies `weighted_frechet_mean` accepts **non-negative** weights. Therefore:

**For SPD/Correlation/Network/PointProcess (Euclidean-structure spaces), the Fréchet mean under a signed-weight scheme reduces to a signed element-wise combination** — just a weighted sum. These spaces can implement `weighted_frechet_mean` to handle arbitrary real weights since their mean is just `Σwᵢ·obj_i` (no probability-space constraint). The trait doc says "non-negative" but the existing density constraint only arises because `wasserstein_barycenter` rejects negatives. For linear-combination spaces, negatives are fine — just document this per-space.

**For `SphericalSpace`,** negative weights in the Karcher mean are problematic (the gradient-descent mean cannot absorb negative weights cleanly). The generic regression over spherical space is therefore best served by normalizing weights to be non-negative before calling `weighted_frechet_mean`. The planner should decide whether to: (a) warn and clip negatives to zero (biased but never fails), or (b) return `ComputationFailed` for extreme xout points that induce negatives in the spherical path. **Recommendation:** clip negatives to zero and renormalize for the spherical path, document as a divergence from the global Fréchet regression theory. For the other four spaces, pass signed weights directly since the mean is a simple weighted sum.

[ASSUMED: this handling matches common practice; the R `frechet` package does not expose the internal weight handling for non-density spaces]

**Practical note for FRE-02-06:** The requirement is "at least one non-density backend." Even if the spherical path requires weight clipping, the SPD/Correlation/Network/PointProcess paths can take signed weights natively (their mean is a linear combination). The planner should deliver the SPD Frobenius path as the primary demo of FRE-02-06, with the same signed-weight behavior as the density path.

#### Local weight helper

Extract from `frechet_local_reg` lines 188-238 into:

```rust
pub(crate) fn compute_local_weights(
    predictors: &FdMatrix,
    xout_row: &[f64],
    bandwidth: f64,
    n: usize,
    p: usize,
) -> Result<Vec<f64>, FdarError>
```

Returns the normalized signed weight vector for a single xout row. The existing `frechet_local_reg` calls this in its loop; `frechet_local_reg_space` also calls this per-row.

#### Generic ANOVA helper

Change `compute_tn` from concrete `space: &WassersteinDensitySpace` to:

```rust
pub(crate) fn compute_tn_space<S: MetricSpace<Object = Vec<f64>>>(
    space: &S,
    objects: &[Vec<f64>],
    labels: &[usize],
    k: usize,
) -> Result<(f64, f64, f64, Vec<f64>, f64), FdarError>
```

Wait — the Object type is `Vec<f64>` only for the density space. For SPD matrices the Object is also `Vec<f64>` (flat matrix). For spherical it is `Vec<f64>` (unit vector). For all five new spaces the `Object` type is `Vec<f64>`. So `compute_tn_space<S: MetricSpace<Object = Vec<f64>>>` works for ALL six spaces (density + 5 new).

Alternatively, use the fully generic `S: MetricSpace` with `S::Object` (no restriction) — this is cleaner:

```rust
pub(crate) fn compute_tn_generic<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    labels: &[usize],
    k: usize,
) -> Result<(f64, f64, f64, Vec<f64>, f64), FdarError>
```

The existing `compute_tn` wrapping `WassersteinDensitySpace` can be replaced entirely by `compute_tn_generic` (it already calls `frechet_mean` and `frechet_variance` which are generic — [VERIFIED: fdars-core/src/frechet/mean.rs:40-83]). The existing `frechet_anova` simply calls `compute_tn_generic` with `&space` — no API change.

**The existing `frechet_anova` tests pass unchanged** because the density path goes through the same generic helper. [VERIFIED: all the anova test helper functions use `FdMatrix`/`Vec<f64>` objects that are already the concrete `Object` type]

The new `frechet_anova_space<S: MetricSpace>` is a thin wrapper identical to `frechet_anova` but parametrized on `S` and accepting `objects: &[S::Object]` instead of an `FdMatrix`/`argvals`.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Eigendecomposition of d×d symmetric matrix | Custom Jacobi iteration | `nalgebra::SymmetricEigen::new(DMatrix<f64>)` | Already used in `fts/acf.rs:345`, `fts/spectral.rs:211`, `fpca_variants.rs:488`; battle-tested |
| Cholesky factorization | Custom | `crate::linalg::cholesky_factor` | Already used in `frechet/regression.rs:122`; returns `ComputationFailed` on non-PD |
| Gaussian kernel | Custom | `crate::helpers::gaussian_kernel` | Already used in `frechet/regression.rs:192` |
| Fréchet mean / variance | Inline code | `crate::frechet::mean::{frechet_mean, frechet_variance}` | Already fully generic over `S: MetricSpace` |
| Seeded permutation | `rand::thread_rng()` | `StdRng::seed_from_u64(seed + perm_k)` | Required convention; already in `frechet/anova.rs:169` |
| Isotonic projection for density signed weights | Custom QP | `signed_quantile_average` (existing `pub(crate)` helper) | Zero new dep; density path already uses it |

---

## Common Pitfalls

### Pitfall 1: nalgebra `DMatrix::from_column_slice` vs row-major flat Vec

**What goes wrong:** The crate stores matrices as column-major `Vec<f64>` (element `(i,j)` at index `i + j*d`). `DMatrix::from_column_slice(d, d, &flat)` interprets the input as column-major — this matches. If the developer accidentally uses `DMatrix::from_row_slice` instead, the matrix is silently transposed for symmetric matrices (same result for symmetric SPD, but not for reconstruction back to flat).

**How to avoid:** Always use `DMatrix::from_column_slice(d, d, &mat_flat)`. When reconstructing back from nalgebra to flat: `result[i + j*d] = recon[(i,j)]` (column-major index). Or use `eig.eigenvectors.column(k)` indexing directly.

[VERIFIED: fdars-core/src/fts/acf.rs:337 — `let mut c0_mat = DMatrix::from_column_slice(m, m, &c0_scaled);` confirms this pattern]

### Pitfall 2: `SymmetricEigen` eigenvalue/eigenvector ordering

**What goes wrong:** `nalgebra::SymmetricEigen` returns eigenvalues in **ascending** order (smallest first), not descending. If the developer assumes descending order and applies `λ^α` without sorting, the power-α reconstruction is correct (order doesn't matter for `V diag(λ^α) Vᵀ`) but sort-based logic breaks.

**How to avoid:** For the power-α mean, ordering does not matter (all eigenvalues processed). No sort required. Explicitly document this in the function. [VERIFIED: fdars-core/src/fts/spectral.rs:219 — `pairs.sort_by(|a, b| b.0.partial_cmp(&a.0)...)` shows the code explicitly sorts after calling `SymmetricEigen`]

### Pitfall 3: Log-Cholesky — reconstruction after coordinate averaging

**What goes wrong:** After averaging the log-Cholesky coordinates, the diagonal entries are `log-space averages` and must be exponentiated. A developer might forget to `exp()` the diagonal entries, producing a non-positive-definite reconstructed matrix.

**How to avoid:** Separate the reconstruction into two explicit steps: (1) strictly-lower entries copied directly, (2) diagonal entries exponentiated. Test with a hand-computed 2×2 case.

[ASSUMED: based on Lin (2019) coordinate definition; implementation detail not in existing codebase to verify against]

### Pitfall 4: Karcher mean — division by zero on antipodal spherical inputs

**What goes wrong:** The log map `log_x(y)` divides by `sin(θ)`. For `θ ≈ π` (antipodal points), `sin(π) = 0` causes division by zero or NaN.

**How to avoid:** Guard `if θ > π − 1e-8 { return Err(ComputationFailed {...}) }` inside the log map helper. Document that the Karcher mean on `Sᵈ⁻¹` is undefined when the sample is antipodally balanced.

### Pitfall 5: Generic regression — negative weights passed to `weighted_frechet_mean`

**What goes wrong:** The Petersen–Müller weights `sᵢ = (1 + dot) / n` can be negative for extrapolation. Passing them to `SphericalSpace::weighted_frechet_mean` (which uses gradient descent initialized at the normalized extrinsic mean) propagates negative weights into the `log_x` gradient computation, producing a gradient that points away from the true mean.

**How to avoid:** For `SphericalSpace`, clip negative weights to 0 and renormalize before calling `weighted_frechet_mean`. For linear-combination spaces (SPD Frobenius, Network, PointProcess, Correlation), allow signed weights since the mean is a weighted sum. Document per-space.

### Pitfall 6: Correlation matrix renormalization — zero diagonal

**What goes wrong:** The renormalization step `M_corr[i,j] = M[i,j] / sqrt(M[i,i] * M[j,j])` fails if any `M[i,i] <= 0`.

**How to avoid:** Guard before dividing: check all diagonal entries `M[i,i] > 0` else return `ComputationFailed`.

### Pitfall 7: `#[cfg(test)]` imports shadow crate root re-exports

**What goes wrong:** Within `frechet/spaces/spd.rs`, tests that `use super::*` get the space types but not `FdarError` — which is at the crate root. The pattern in `frechet/space.rs:293` imports `use super::*` inside `mod tests { ... }` and separately imports helpers.

**How to avoid:** Follow the exact pattern in `frechet/space.rs` tests: `use super::*;` for module items; use absolute crate paths `crate::error::FdarError` or `use crate::error::FdarError;` for cross-module types inside tests.

[VERIFIED: fdars-core/src/frechet/space.rs:283-427 — test module pattern confirmed]

---

## Code Examples

### SymmetricEigen pattern for matrix power (from verified codebase)

```rust
// Source: fdars-core/src/fts/acf.rs:337-348 (pattern)
use nalgebra::DMatrix;

fn spd_power(mat_flat: &[f64], d: usize, alpha: f64) -> Vec<f64> {
    let mut mat = DMatrix::from_column_slice(d, d, mat_flat);
    // Symmetrize defensively (matches acf.rs pattern)
    for j1 in 0..d {
        for j2 in (j1 + 1)..d {
            let avg = 0.5 * (mat[(j1, j2)] + mat[(j2, j1)]);
            mat[(j1, j2)] = avg;
            mat[(j2, j1)] = avg;
        }
    }
    let eig = nalgebra::SymmetricEigen::new(mat);
    // Reconstruct V · diag(λ^α) · Vᵀ in column-major
    let evecs = &eig.eigenvectors;
    let evals = &eig.eigenvalues;
    let mut result = vec![0.0_f64; d * d];
    for k in 0..d {
        let lk_alpha = evals[k].max(0.0).powf(alpha);
        for i in 0..d {
            for j in 0..d {
                result[i + j * d] += evecs[(i, k)] * lk_alpha * evecs[(j, k)];
            }
        }
    }
    result
}
```

[VERIFIED pattern: fdars-core/src/fts/acf.rs:337,345 for DMatrix::from_column_slice + SymmetricEigen::new; reconstruction formula is [ASSUMED] standard]

### Karcher mean on sphere (algorithmic pattern)

```rust
// Pattern for SphericalSpace::weighted_frechet_mean
fn karcher_mean(objects: &[Vec<f64>], weights: &[f64], d: usize,
                max_iter: usize, tol: f64) -> Result<Vec<f64>, FdarError> {
    // 1. Extrinsic init
    let mut x = vec![0.0_f64; d];
    for (obj, &w) in objects.iter().zip(weights.iter()) {
        for k in 0..d { x[k] += w * obj[k]; }
    }
    let norm = x.iter().map(|v| v*v).sum::<f64>().sqrt();
    if norm < 1e-14 {
        return Err(FdarError::ComputationFailed { ... });
    }
    for k in 0..d { x[k] /= norm; }

    for _iter in 0..max_iter {
        // 2. Gradient = Σwᵢ log_x(aᵢ)
        let mut grad = vec![0.0_f64; d];
        for (obj, &w) in objects.iter().zip(weights.iter()) {
            let log = log_map(&x, obj, d)?;
            for k in 0..d { grad[k] += w * log[k]; }
        }
        // 3. Convergence check
        let gnorm = grad.iter().map(|v| v*v).sum::<f64>().sqrt();
        if gnorm < tol { break; }
        // 4. Retract
        x = exp_map(&x, &grad, d);
        // 5. Re-normalize (defensive)
        let n2 = x.iter().map(|v| v*v).sum::<f64>().sqrt();
        for k in 0..d { x[k] /= n2.max(1e-300); }
    }
    Ok(x)
}
```

[ASSUMED: standard Riemannian gradient descent on Sᵈ⁻¹; consistent with Petersen–Müller supplementary]

### Generic ANOVA after refactor

```rust
// frechet_anova_space after generic compute_tn_generic extraction
pub fn frechet_anova_space<S: MetricSpace>(
    space: &S,
    objects: &[S::Object],
    labels: &[usize],
    n_perm: usize,
    seed: u64,
) -> Result<FrechetAnovaResult, FdarError> {
    // same validation as frechet_anova minus the argvals/FdMatrix checks
    let k = labels.iter().copied().max().map_or(0, |mx| mx + 1);
    // ... distinct group checks ...
    let n_perm = if n_perm == 0 { 999 } else { n_perm };
    let (tn_obs, fn_stat, un_stat, group_vars, pooled_var) =
        compute_tn_generic(space, objects, labels, k)?;
    let p_asymptotic = chi_square_sf(tn_obs, k - 1);
    let mut n_ge = 0usize;
    for perm in 0..n_perm {
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(perm as u64));
        let mut perm_labels = labels.to_vec();
        perm_labels.shuffle(&mut rng);
        if let Ok((tn_perm, _, _, _, _)) = compute_tn_generic(space, objects, &perm_labels, k) {
            if tn_perm >= tn_obs { n_ge += 1; }
        }
    }
    let p_permutation = (n_ge as f64 + 1.0) / (n_perm as f64 + 1.0);
    Ok(FrechetAnovaResult { statistic: tn_obs, p_value_asymptotic: p_asymptotic,
        p_value_permutation: p_permutation, n_perm, group_frechet_variances: group_vars,
        pooled_frechet_variance: pooled_var, fn_statistic: fn_stat, un_statistic: un_stat,
        group_labels: labels.to_vec() })
}
```

[VERIFIED pattern: fdars-core/src/frechet/anova.rs:161-191 — seeding pattern `StdRng::seed_from_u64(seed.wrapping_add(perm as u64))`]

---

## Hand-Computable Test Oracles

These are concrete test oracles the executor should implement as `#[test]` functions.

### SPD / Correlation

| Test | Oracle | Tolerance |
|------|--------|-----------|
| Frobenius mean of identical 2×2 SPD matrices = that matrix | `mean == input` element-by-element | < 1e-10 |
| Power-α with α=1: `d_α(A,B) = ‖A−B‖_F / 1 = ‖A−B‖_F` | Same as Frobenius distance | < 1e-10 |
| Log-Cholesky mean of two identical SPD matrices = that matrix | `mean == input` element-by-element | < 1e-10 |
| Log-Cholesky mean of identity and 4I in 2×2: geometric mean = 2I | (See computation below) | < 1e-8 |
| Frobenius distance of A from itself = 0 | `d = 0` | < 1e-12 |
| Correlation mean of identical corr matrices = that matrix | `mean == input` | < 1e-10 |

**Log-Cholesky 2×2 verification:**
`I = [[1,0],[0,1]]` → `L_I = [[1,0],[0,1]]` → coord = `[log(1), 0, log(1)] = [0, 0, 0]`.
`4I = [[4,0],[0,4]]` → `L = [[2,0],[0,2]]` → coord = `[log(2), 0, log(2)]`.
Mean coord = `[log(2)/2, 0, log(2)/2]` → `L_mean = [[exp(log(2)/2), 0], [0, exp(log(2)/2)]] = [[√2,0],[0,√2]]`.
Mean matrix = `L_mean @ L_meanᵀ = [[2,0],[0,2]] = 2I`. So oracle: mean of `I` and `4I` (equal weights) = `2I`. [ASSUMED: computation is standard, checked by hand]

### Spherical

| Test | Oracle | Tolerance |
|------|--------|-----------|
| Geodesic distance of `[1,0]` and `[-1,0]` (antipodal on S¹) = π | `d = π` | < 1e-12 |
| Geodesic distance of identical vectors = 0 | `d = 0` | < 1e-12 |
| Karcher mean of two nearby unit vectors on S¹: `[1,0]` and `[cos(0.1), sin(0.1)]` | Great-circle midpoint = `[cos(0.05), sin(0.05)]` normalized | < 1e-6 |
| Karcher mean of three identical unit vectors = that vector | `mean == input` | < 1e-8 |

### Network

| Test | Oracle | Tolerance |
|------|--------|-----------|
| Frobenius distance of two identical 3-node Laplacians = 0 | `d = 0` | < 1e-12 |
| Weighted mean of identical Laplacians = that Laplacian | Element-by-element | < 1e-10 |
| Laplacian mean of two valid Laplacians: row-sums of result = 0 | `sum(row_i) ≈ 0 ∀i` | < 1e-10 |

### Point-Process

| Test | Oracle | Tolerance |
|------|--------|-----------|
| L2 distance of two identical vectors = 0 | `d = 0` | < 1e-12 |
| L2 distance of `[1,0,0]` and `[0,1,0]` = √2 | `d = √2` | < 1e-12 |
| Weighted mean of identical intensity vectors = that vector | Element-by-element | < 1e-10 |

### Generic Regression

| Test | Oracle | Notes |
|------|--------|-------|
| Constant-response SPD regression: all responses = A, predict at any xout → A | `‖predicted − A‖_F < tol` | |
| Constant-response PointProcess regression: all responses = v → predicted = v | Element-by-element | |

**Constant-response oracle (works for all linear-combination spaces):**
When all responses are the same object `A`, the Petersen–Müller weights `Σᵢ sᵢ = 1` (they sum to 1) and `Σᵢ sᵢ · A = A`. So the predicted response is exactly `A` for any xout, independent of the weights. This is a clean, dimension-independent oracle. [ASSUMED: follows from the Petersen–Müller weight normalization `Σᵢ sᵢ = 1`, which holds because `Σᵢ (Xᵢ − X̄) = 0` so `Σᵢ (1 + dot_i)/n = n/n = 1`]

### Generic ANOVA

| Test | Oracle | Notes |
|------|--------|-------|
| Homogeneous 2-group SPD sample (all matrices identical): `p_perm > 0.05` | Non-significant | Must be seed-reproducible |
| Seed reproducibility: same `seed` → same `p_value_permutation` | Bit-exact equality | |
| Well-separated 2-group SPD sample (group A: identity matrices, group B: 4×identity): `p_perm < 0.05` | Significant | |

---

## Module Re-Export Plan

### `frechet/mod.rs` additions

```rust
mod spaces;  // NEW

pub use spaces::{
    SpdMetric, SpdMatrixSpace,
    CorrelationMatrixSpace,
    SphericalSpace,
    NetworkSpace,
    PointProcessSpace,
};
pub use regression::{frechet_global_reg_space, frechet_local_reg_space};
pub use anova::frechet_anova_space;
```

[VERIFIED pattern: fdars-core/src/frechet/mod.rs:34-42 — existing `pub use anova::frechet_anova; pub use mean::{...}; pub use regression::{...}; pub use space::{...}`]

### `src/lib.rs` additions

Extend the existing block [VERIFIED: fdars-core/src/lib.rs:152-156]:
```rust
pub use frechet::{
    frechet_anova, frechet_global_reg, frechet_local_reg, frechet_mean, frechet_variance,
    wasserstein2_distance, FrechetAnovaResult, FrechetGlobalRegResult, FrechetLocalRegResult,
    MetricSpace, WassersteinDensitySpace,
    // NEW:
    frechet_anova_space, frechet_global_reg_space, frechet_local_reg_space,
    SpdMetric, SpdMatrixSpace, CorrelationMatrixSpace, SphericalSpace,
    NetworkSpace, PointProcessSpace,
};
```

---

## State of the Art

| Old Approach | Current Approach | Notes |
|--------------|------------------|-------|
| Density-only Fréchet regression (R `frechet` ≤ 0.2) | Object-data backends (`frechet` 0.3.0) | This phase implements the Rust equivalent |
| Global-only Petersen–Müller weights | Global + local (kernel) | Both already in FRE-01 |
| Log-Euclidean SPD metric (Arsigny 2007) | Log-Cholesky metric (Lin 2019) | Simpler computation; same class of invariant metrics |
| Affine-invariant SPD metric (`power α=0` limit) | Power-α + Log-Cholesky | True affine-invariant metric requires matrix square root iteration; not needed here |

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | R `frechet` 0.3.0 uses Frobenius distance for correlation matrix space | Correlation backend | Low — only affects rustdoc divergence note, not correctness |
| A2 | R `frechet` 0.3.0 uses Frobenius distance on Laplacians for network space | Network backend | Low — same |
| A3 | R `frechet` 0.3.0 uses L2 distance on intensity vectors for point-process space | Point-process backend | Low — same |
| A4 | Lin (2019) log-Cholesky coordinate convention: strictly-lower entries as-is, diagonal as log | Log-Cholesky backend | Medium — if convention reversed (log-lower, diagonal as-is), mean back-map produces wrong diagonal; test oracle catches it |
| A5 | Petersen–Müller global weights sum to 1 (constant-response regression oracle) | Generic regression test | Low — derivable from `Σᵢ (Xᵢ − X̄) = 0`; well-established |
| A6 | `nalgebra::SymmetricEigen` reconstructed via `V diag(λ^α) Vᵀ` uses `.eigenvectors` as column matrix | SPD power-α | Low — all existing uses in codebase follow this pattern |
| A7 | Clipping negative Power-α eigenvalues to 0 before `powf(alpha)` is numerically safe | SPD power-α | Low — edge case only arises from floating-point noise on PSD inputs; well-established guard |
| A8 | Spherical Karcher mean convergence in ≤ 50 iterations with tol=1e-8 for typical d ≤ 100 and uniform-ish distributions | SphericalSpace | Medium — for near-antipodal inputs or high-d spheres, 50 iters may be insufficient; the `ComputationFailed` fallback makes this safe |
| A9 | Signed (negative) Petersen–Müller weights can be passed to SPD Frobenius / Network / PointProcess / Correlation `weighted_frechet_mean` (which are just linear combinations) | Generic regression | Low — linear combination is well-defined for any real weights; only the Karcher mean on Sᵈ⁻¹ has issues |

---

## Validation Architecture

`nyquist_validation: true` per `.planning/config.json`. [VERIFIED: /home/simonm/projects/rust/fdars/.planning/config.json — `"nyquist_validation": true`]

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` via `cargo test` |
| Config file | none — inline `#[cfg(test)]` blocks in each source file |
| Quick run command | `cargo test -p fdars-core --features linalg frechet -- --test-threads=1` |
| Full suite command | `cargo test -p fdars-core --features linalg,parallel --all-targets -- --test-threads=4` |
| Clippy gate | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` |

Note: use `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` prefix for all cargo commands to avoid /tmp exhaustion. [VERIFIED: fdars-core memory pointer]

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| FRE-02-01 | SPD Frobenius distance + mean | unit | `cargo test -p fdars-core --features linalg spd_frobenius` | No — Wave 0 |
| FRE-02-01 | SPD Power-α distance + mean (α=1 = Frobenius) | unit | `cargo test -p fdars-core --features linalg spd_power_alpha` | No — Wave 0 |
| FRE-02-01 | Log-Cholesky distance + mean (2×2 oracle) | unit | `cargo test -p fdars-core --features linalg log_cholesky` | No — Wave 0 |
| FRE-02-02 | Correlation distance + renormalized mean | unit | `cargo test -p fdars-core --features linalg correlation_space` | No — Wave 0 |
| FRE-02-03 | Geodesic distance: antipodal = π, identical = 0 | unit | `cargo test -p fdars-core --features linalg spherical_distance` | No — Wave 0 |
| FRE-02-03 | Karcher mean: great-circle midpoint oracle | unit | `cargo test -p fdars-core --features linalg karcher_mean` | No — Wave 0 |
| FRE-02-04 | Network Frobenius distance + mean (Laplacian structure preserved) | unit | `cargo test -p fdars-core --features linalg network_space` | No — Wave 0 |
| FRE-02-05 | PointProcess L2 distance + mean | unit | `cargo test -p fdars-core --features linalg point_process_space` | No — Wave 0 |
| FRE-02-06 | Generic global reg: constant-response SPD → predicts constant | integration | `cargo test -p fdars-core --features linalg frechet_global_reg_space` | No — Wave 0 |
| FRE-02-06 | Existing `frechet_global_reg` density tests still pass post-refactor | regression | `cargo test -p fdars-core --features linalg global_tracks_known_relationship` | Yes |
| FRE-02-07 | Generic ANOVA: homogeneous SPD sample non-significant + seed-reproducible | integration | `cargo test -p fdars-core --features linalg frechet_anova_space` | No — Wave 0 |
| FRE-02-07 | Existing `frechet_anova` density tests still pass post-refactor | regression | `cargo test -p fdars-core --features linalg anova_flags_shifted_groups` | Yes |

### Sampling Rate
- **Per task commit:** `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg frechet 2>&1 | tail -5`
- **Per wave merge:** `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --all-targets 2>&1 | tail -10`
- **Phase gate:** Full suite green + `cargo clippy --all-targets --features linalg,parallel -- -D warnings` clean before `/gsd-verify-work`

### Wave 0 Gaps

- [ ] `frechet/spaces/mod.rs` (or `frechet/object_spaces.rs`) — module skeleton
- [ ] `frechet/spaces/spd.rs` — `SpdMatrixSpace`, `SpdMetric` struct + `impl MetricSpace` stubs
- [ ] `frechet/spaces/correlation.rs` — `CorrelationMatrixSpace` stub
- [ ] `frechet/spaces/spherical.rs` — `SphericalSpace` stub
- [ ] `frechet/spaces/network.rs` — `NetworkSpace` stub
- [ ] `frechet/spaces/point_process.rs` — `PointProcessSpace` stub

*(Tests live inline in each file per crate convention [VERIFIED: fdars-core/src/frechet/space.rs:283])*

---

## Security Domain

**`security_enforcement: true`** per `.planning/config.json`. This phase is a pure in-process numerical computation library — no I/O, no serialization of untrusted input, no network calls, no auth/session/crypto.

### Applicable ASVS Categories (ASVS Level 1)

| ASVS Category | Applies | Control |
|---------------|---------|---------|
| V2 Authentication | No | Pure library, no auth |
| V3 Session Management | No | No state/sessions |
| V4 Access Control | No | No access control |
| V5 Input Validation | Yes | All public fns validate dimensions + parameter ranges; return `FdarError` not panic |
| V6 Cryptography | No | No encryption |
| V9 Data Protection | No | No persistence, no PII |

### Threat Patterns

| Pattern | STRIDE | Mitigation |
|---------|--------|------------|
| Integer overflow in index arithmetic `i + j*d` for large `d` | Tampering | Rust's debug-mode overflow panics + safe `Vec` indexing; `d` validated to be `>= 1` (no upper bound guard needed — a d=1000 matrix = 8 MB, within normal RAM) |
| NaN propagation from malformed SPD input | Tampering | Eigenvalue clamp to 0 before `powf`; Cholesky `ComputationFailed` on non-PD; arccos clamp |
| Division by zero in log map (`sin(θ) = 0`) | Tampering | θ guard before division; `ComputationFailed` return |
| Panic on empty input | Denial of Service | `if objects.is_empty() { return Err(InvalidDimension {...}) }` at function entry |

---

## Environment Availability

All computation is in-process Rust. No external tools, databases, or services required.

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust stable | Build | ✓ | 1.97.0 (runtime), MSRV 1.81.0 | — |
| `nalgebra` 0.33 | SPD matrix power via `SymmetricEigen` | ✓ | Already in Cargo.lock | — |
| `cargo test` | Unit tests | ✓ | Ships with Rust | — |
| `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` | doctest linking avoids /tmp exhaustion | ✓ | Directory must exist | `mkdir -p ~/.cache/fdars-bench-tmp` |

**Missing dependencies with no fallback:** none
**Missing dependencies with fallback:** none

---

## Open Questions

1. **Log-Cholesky convention — strictly-lower vs. lower-triangular**
   - What we know: Lin (2019) defines the coordinate as: strictly-lower triangular entries of `L` unchanged, diagonal entries `log(Lᵢᵢ)`.
   - What's unclear: Whether the full lower-triangular (including diagonal) should be packed together or the strictly-lower and diagonal kept in separate passes.
   - Recommendation: Pack as a `Vec<f64>` of length `d*(d+1)/2`: first strictly-lower entries (row-major within the lower triangle), then diagonal entries as log. Unpack symmetrically. The 2×2 test oracle validates this.

2. **Negative Petersen–Müller weights in spherical generic regression**
   - What we know: For SPD/Network/PointProcess the mean is a linear combination so negatives are fine.
   - What's unclear: Whether the planner wants a general rule for all spaces (clip to 0 for Karcher-based spaces) or a per-space signed-weight variant of `weighted_frechet_mean`.
   - Recommendation: For FRE-02-06 (at least one backend demo), use the SPD Frobenius path which takes signed weights natively. The spherical generic regression is a natural stretch goal but not required by the requirement text.

3. **`compute_tn_generic` — replace or duplicate `compute_tn`**
   - What we know: The existing `compute_tn` is a private function with `WassersteinDensitySpace` concrete type.
   - What's unclear: Whether to replace it entirely with the generic version (simpler maintenance) or keep both (safer regression path for density tests).
   - Recommendation: Replace entirely. The generic version with `S: MetricSpace` subsumes the concrete version. Existing `frechet_anova` density tests serve as the regression test.

---

## Project Constraints (from CLAUDE.md)

| Constraint | Requirement |
|------------|-------------|
| Additive/non-breaking | Zero changes to existing public signatures (`frechet_global_reg`, `frechet_local_reg`, `frechet_anova`, all result structs) |
| No new crate dependency | Only `nalgebra` 0.33 (already in Cargo.toml) + in-crate `linalg` helpers |
| All public functions return `Result<T, FdarError>` | No panics on user-facing validation; `ComputationFailed` on non-convergence |
| Inline `#[cfg(test)]` tests | No separate integration test files for this phase (follow `frechet/space.rs` pattern) |
| Full clippy gate | `cargo clippy --all-targets --features linalg,parallel -- -D warnings` |
| Crate-root re-exports | Extend existing `pub use frechet::{...}` block in `src/lib.rs` |
| Numeric outputs only | No plotting, no visualization |
| Document divergences in rustdoc | Frobenius vs affine-invariant correlation geometry; Karcher mean init; signed-weight behavior |
| MSRV 1.81.0 | `nalgebra::SymmetricEigen` is stable since nalgebra 0.30; no MSRV concern |
| `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` | Prefix all cargo build/test/doctest commands |
| `cargo fmt` per commit | Avoid fmt drift from `--no-verify` commits |

---

## Sources

### Primary (HIGH confidence — source read this session)
- `fdars-core/src/frechet/space.rs` (entire file, lines 1-427) — `MetricSpace` trait verbatim; `WassersteinDensitySpace` impl; `signed_quantile_average`; test patterns
- `fdars-core/src/frechet/regression.rs` (entire file, lines 1-387) — `frechet_global_reg`/`frechet_local_reg` weight computation logic; extraction plan
- `fdars-core/src/frechet/anova.rs` (entire file, lines 1-294) — `compute_tn`; `frechet_anova`; seeding pattern; `FrechetAnovaResult` struct
- `fdars-core/src/frechet/mean.rs` (entire file, lines 1-176) — `frechet_mean`/`frechet_variance` genericity confirmed
- `fdars-core/src/frechet/mod.rs` (entire file, lines 1-103) — result struct definitions; `pub use` conventions
- `fdars-core/src/linalg.rs` (entire file, lines 1-152) — `cholesky_factor` exact signature + row-major convention
- `fdars-core/src/fts/acf.rs` (lines 330-376) — `DMatrix::from_column_slice` + `SymmetricEigen::new` pattern
- `fdars-core/src/fts/spectral.rs` (lines 200-238) — eigenvalue sort convention + eigenvector column access
- `fdars-core/src/lib.rs` (lines 150-160) — frechet re-export block to extend
- `.planning/config.json` — nyquist_validation=true, security_enforcement=true

### Secondary (MEDIUM confidence — consistent with training knowledge and codebase evidence)
- Petersen & Müller (2019) "Fréchet regression for random objects with Euclidean predictors" — Annals of Statistics 47(2) — global/local weight formulae
- Dubey & Müller (2019) "Fréchet analysis of variance for random objects" — Biometrika 106(4) — Tₙ statistic
- Lin (2019) "Riemannian geometry of symmetric positive definite matrices via Cholesky decomposition" — log-Cholesky coordinates

### Tertiary (LOW confidence — training knowledge, not verified in session)
- R `frechet` 0.3.0 source — specific metric choices per response space (Frobenius for correlation/network, L2 for point process)
- Convergence behavior of Karcher mean in ≤ 50 iterations for typical FDA settings

---

## Metadata

**Confidence breakdown:**
- Existing code structure to refactor: HIGH — read verbatim from source
- SPD Frobenius + Power-α math: HIGH — standard, consistent with codebase patterns
- Log-Cholesky math: MEDIUM — standard but coordinate convention is [ASSUMED]; test oracle catches wrong convention
- Spherical Karcher mean algorithm: MEDIUM — algorithm is standard; convergence guarantee is [ASSUMED]
- Network / PointProcess / Correlation math: MEDIUM — simple linear-combination means, high confidence; R baseline metric choices are [ASSUMED]
- Refactoring plan (weight helpers + compute_tn_generic): HIGH — derivable directly from the source code read

**Research date:** 2026-08-22
**Valid until:** 2027-02-22 (stable math; no expiry risk from library churn — no new packages)
