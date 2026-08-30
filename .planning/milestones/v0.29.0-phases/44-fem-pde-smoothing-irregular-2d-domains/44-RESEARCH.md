# Phase 44: FEM/PDE Smoothing on Irregular 2D Domains — Research

**Researched:** 2026-08-24
**Domain:** Finite-element methods, PDE-regularized smoothing, monotone/positive basis smoothing
**Confidence:** HIGH (mathematics verified against textbook formulas; code-context claims verified by Read this session)

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions
- Mesh input: user-supplied nodes (N×2 coordinates) + triangle connectivity (T×3 vertex indices) as explicit inputs. No internal mesh generation.
- FE basis: linear P1 Lagrange "hat" basis — one basis function per node, value 1 at its node, 0 at others, linear over each triangle.
- Basis evaluation at arbitrary (x,y): barycentric coordinates within the containing triangle; point location via a linear scan over triangles for v1.
- Boundary condition: Neumann (natural, zero-flux) — standard choice for PDE surface smoothing.
- Penalty: Laplacian roughness via FE stiffness matrix K (SPD); mass matrix M for FE inner product. SR-PDE formulation.
- Assembly: per-triangle element mass + stiffness via linear-FE area-based closed forms, assembled into global N×N matrices.
- Solve: penalized normal equations `(Φ'Φ + λK)c = Φ'y` via dense `cholesky_solve` — v1 only.
- Smoothing parameter: fixed λ + trace-based GCV helper.
- Positive smoother: log-domain guarantee.
- Monotone smoother: Ramsay integral-of-exp `f(t) = β₀ + β₁∫₀ᵗ exp(w(u))du` with w in B-spline basis.
- API: new standalone public fns `smooth_positive` / `smooth_monotone` in `smooth_basis.rs`.
- Result struct: `FemSmoothResult` mirroring `SmoothBasisResult` / `FosrResult2d`.
- KEEP no-new-dependency convention: in-house dense Cholesky for v1; sparse solvers deferred.
- Module layout: single `fem_smoothing.rs` + additive edits to `smooth_basis.rs`.
- Mesh validation: indices in range, non-degenerate triangles → `FdarError` on violation.

### Claude's Discretion
- Exact struct field naming, internal helper decomposition, plan/wave decomposition.
- Precise GCV/edf trace computation strategy.
- Whether `fem_smoothing.rs` warrants folder split if it exceeds ~500 lines.

### Deferred Ideas (OUT OF SCOPE)
- 3D tetrahedral-mesh FEM.
- Sparse-matrix assembly/solvers and spatial index for point location.
- Quadratic (P2) or higher-order elements; Dirichlet/Robin BCs; anisotropic/advection-diffusion PDE penalty.
- Areal/regional observations; space-varying PDE coefficients.
- I-spline / PAVA monotone alternatives; nonneg-LS positive alternative.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| REP-02-01 | Linear FE basis over user-supplied triangulated 2D mesh; evaluate basis functions; assemble mass/stiffness matrices | Section: P1 Basis; Element Matrices; Assembly Loop |
| REP-02-02 | PDE-regularized (Laplacian-penalty) surface smoothing of scattered observations; fitted surface + diagnostics | Section: SR-PDE Smoothing System; GCV |
| REP-02-03 | Positive-valued smoothing (log-domain, nonneg guaranteed) | Section: Log-Domain Positive Smoother |
| REP-02-04 | Monotone smoothing via Ramsay integral-of-exponential in `smooth_basis.rs` | Section: Ramsay Monotone Smoother |
</phase_requirements>

---

## Summary

Phase 44 delivers linear finite-element (P1) surface smoothing over irregular 2D triangulated meshes, plus two new 1D smoothers (positive log-domain, Ramsay monotone) added to the existing `smooth_basis.rs`. The entire implementation is in-house: no new crate dependencies.

The mathematical core is well-established: P1 hat-function basis, per-triangle element mass and stiffness matrices assembled by closed-form area formulas, then a dense penalized normal-equations solve using the existing `cholesky_solve` from `linalg.rs`. The SR-PDE formulation (`(Φ'Φ + λK)c = Φ'y`) is the fdaPDE-equivalent approach. The GCV trace is computed via a diagonal of the hat matrix, done entirely with the dense inverse already required for the solve, so no extra cost. The log-domain positive smoother is trivial (wrap existing `smooth_basis`). The Ramsay monotone smoother is the most algorithmically demanding piece: it requires a Gauss-Newton iteration over nonlinear least squares, but a straightforward 20-iteration scheme converges for well-conditioned monotone data.

**Primary recommendation:** Implement in four clearly separated internal helpers — (1) mesh validation + barycentric eval, (2) element matrix closed forms + global assembly, (3) SR-PDE solve + GCV, (4) Ramsay NLS iteration — then expose three public functions: `assemble_fem_matrices`, `fem_smooth`, `fem_smooth_gcv` from `fem_smoothing.rs` and `smooth_positive`, `smooth_monotone` from `smooth_basis.rs`.

---

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| Mesh validation (node bounds, degenerate triangles) | Domain module (fem_smoothing) | — | Entry-point validation at function boundary per codebase convention |
| P1 basis evaluation / barycentric coords | Domain module (fem_smoothing) | — | Pure numeric; used by observation matrix Φ and predict path |
| Element mass + stiffness assembly | Domain module (fem_smoothing) | — | FE math; isolated helper; no external deps |
| Global K, M matrix assembly | Domain module (fem_smoothing) | — | Calls element helpers; populates N×N dense arrays |
| Penalized solve `(Φ'Φ + λK)c = Φ'y` | Shared infra (linalg) | Domain module | Uses existing `cholesky_solve`; domain module builds the system |
| GCV trace computation | Domain module (fem_smoothing) | Shared infra (linalg) | Diagonal extraction from dense inverse; dense for v1 |
| Log-domain positive smoother | Existing smooth_basis | — | Wraps existing `smooth_basis` after log-transform |
| Ramsay monotone smoother | Existing smooth_basis | basis/bspline | B-spline w(u) expansion; NLS iteration in smooth_basis.rs |
| Public API re-exports | lib.rs, prelude.rs | — | Pattern from every other module; purely additive |

---

## Standard Stack

### No New Dependencies

This phase deliberately uses only what is already in `fdars-core`. All packages below are verified as already present.

| Asset | Location | Purpose | Status |
|-------|----------|---------|--------|
| `cholesky_factor` / `cholesky_forward_back` / `cholesky_solve` | `src/linalg.rs:85-134` | Dense SPD solves | [VERIFIED: src/linalg.rs:85-134] |
| `compute_xtx` | `src/linalg.rs:137-151` | X'X computation | [VERIFIED: src/linalg.rs:137-151] |
| `bspline_basis` / `bspline_basis_from_knots` | `src/basis/bspline.rs:62-80+` | B-spline evaluation for Ramsay w(u) | [VERIFIED: src/basis/bspline.rs:1-80] |
| `construct_bspline_knots` | `src/basis/bspline.rs:4-17` | Knot vector for w B-spline expansion | [VERIFIED: src/basis/bspline.rs:4-17] |
| `simpsons_weights` | `src/helpers.rs:57+` | Numerical quadrature for Ramsay integral | [VERIFIED: src/helpers.rs:57] |
| `smooth_basis` / `SmoothBasisResult` / `FdPar` / `BasisType` | `src/smooth_basis.rs:45-65` | Log-domain smoother wraps this | [VERIFIED: src/smooth_basis.rs:45-65] |
| `bspline_penalty_matrix` | `src/smooth_basis.rs:82-116` | Penalty for w(u) in Ramsay | [VERIFIED: src/smooth_basis.rs:82-116] |
| `FdMatrix` | `src/matrix.rs:38-44` | Column-major matrix for node coords, Φ, output | [VERIFIED: src/matrix.rs:38-44] |
| `FdarError` variants | `src/error.rs` | All public fns return `Result<T, FdarError>` | [ASSUMED — error.rs not read this session, but variants listed in CLAUDE.md] |
| nalgebra `DMatrix` | `Cargo.toml` dep | Used by smooth_basis.rs already; available | [VERIFIED: src/smooth_basis.rs:15] |

**Installation:** No new crates. All assets already compiled into fdars-core.

---

## Package Legitimacy Audit

No external packages are added in this phase. Audit: **N/A** — in-house implementation only.

---

## Architecture Patterns

### System Architecture Diagram

```
User Input
  nodes: &[[f64;2]] (N rows)        observations: &[f64] (n_obs)
  triangles: &[[usize;3]] (T rows)  obs_xy: &[[f64;2]] (n_obs rows)
       │                                     │
       ▼                                     │
┌─────────────────────┐                      │
│  mesh_validate()    │ → FdarError if bad   │
│  (index bounds,     │                      │
│   positive area)    │                      │
└────────┬────────────┘                      │
         │                                   │
         ▼                                   ▼
┌─────────────────────┐   ┌──────────────────────────────┐
│  assemble_mass_K()  │   │  build_observation_matrix()  │
│  assemble_stiff_K() │   │  (barycentric eval for each  │
│  → M: N×N, K: N×N  │   │   obs point → Φ: n_obs×N)   │
└────────┬────────────┘   └────────────┬─────────────────┘
         │                             │
         └──────────┬──────────────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  Build system:        │
         │  A = Φ'Φ + λ·K       │
         │  rhs = Φ'y            │
         │  c = cholesky_solve   │
         └──────────┬───────────┘
                    │
                    ▼
         ┌──────────────────────┐
         │  GCV (optional):      │
         │  A_inv via dense inv  │
         │  edf = tr(Φ A_inv Φ')│
         │  GCV = n·RSS/(n-edf)²│
         └──────────┬───────────┘
                    │
                    ▼
         FemSmoothResult { node_values, fitted_obs, edf, gcv, rss }

── smooth_positive path ──────────────────────────────────────────
y_pos → log(max(y,ε)) → smooth_basis() → exp(fitted) → MonotoneSmoothResult

── smooth_monotone path (Ramsay) ──────────────────────────────────
y, t → B-spline design Ψ for w(u) → Gauss-Newton NLS over (β₀,β₁,w_coefs)
     → f(t) = β₀ + β₁ ∫₀ᵗ exp(Ψ(u)·w) du (guaranteed nondecreasing)
```

### Recommended Project Structure

```
fdars-core/src/
├── fem_smoothing.rs          # new; mesh validation + basis + assembly + SR-PDE solve + GCV
└── smooth_basis.rs           # additive: smooth_positive, smooth_monotone, MonotoneSmoothResult
```

If `fem_smoothing.rs` grows past ~500 lines, split per CLAUDE.md convention:

```
fdars-core/src/fem_smoothing/
├── mod.rs                    # public API + re-exports
├── mesh.rs                   # mesh validation, barycentric coords, point location
├── assembly.rs               # element matrices, global assembly
└── solve.rs                  # SR-PDE system, GCV
```

---

## 1. Linear P1 FE Basis on Triangles — Complete Recipe

### 1.1 Data Representation

```
nodes:     &[[f64; 2]]    — N mesh nodes, each (x, y)
triangles: &[[usize; 3]]  — T triangles, each (v0, v1, v2) as indices into nodes
```

Convention: triangle winding order (CW vs CCW) does not matter for mass/stiffness as long as areas are taken as absolute values.

### 1.2 Barycentric Coordinates and Point-in-Triangle

For a query point `p = (px, py)` and triangle with vertices `A = (x0,y0)`, `B = (x1,y1)`, `C = (x2,y2)`:

```
Compute 2×2 linear system (barycentric via affine inverse):
  [x1-x0, x2-x0] [λ1]   [px-x0]
  [y1-y0, y2-y0] [λ2] = [py-y0]

det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)     // = 2 * signed_area

λ1 = ((px-x0)*(y2-y0) - (py-y0)*(x2-x0)) / det
λ2 = ((py-y0)*(x1-x0) - (px-x0)*(y1-y0)) / det
λ0 = 1 - λ1 - λ2
```

**Point-in-triangle test:** `λ0 ≥ -ε AND λ1 ≥ -ε AND λ2 ≥ -ε` (use ε=1e-10 for numerical boundary tolerance).

**Degenerate triangle guard:** `|det| < 1e-14 * reference_scale` — return `FdarError::InvalidParameter` during mesh validation (not during evaluation).

**P1 hat function values at (px,py):** The three P1 basis functions for the containing triangle take values `(λ0, λ1, λ2)`. All other basis functions are 0 at this point.

### 1.3 Building the Observation Matrix Φ (n_obs × N)

```rust
// Φ[i, v_k] = λ_k  for the triangle containing obs_xy[i]
// Φ[i, j] = 0      for all other nodes j

// Algorithm (linear scan, O(n_obs * T)):
for i in 0..n_obs {
    for tri_idx in 0..n_triangles {
        let (v0,v1,v2) = triangles[tri_idx];
        let (lam0, lam1, lam2) = barycentric(obs_xy[i], nodes[v0], nodes[v1], nodes[v2]);
        if lam0 >= -EPS && lam1 >= -EPS && lam2 >= -EPS {
            phi[(i, v0)] = lam0;
            phi[(i, v1)] = lam1;
            phi[(i, v2)] = lam2;
            break;
        }
    }
    // if no triangle found: point is outside mesh → FdarError
}
```

The result is a **sparse-structured** dense N-column matrix; each row has exactly 3 non-zeros (or an error). In v1 store as a plain `Vec<f64>` of length `n_obs * N` (row-major for the system build; convert to the column-major layout the system requires).

### 1.4 Partition-of-Unity Property

For any point strictly inside the mesh: `Σ_k φ_k(x,y) = λ0 + λ1 + λ2 = 1` by definition of barycentric coordinates. This is the primary test oracle.

---

## 2. Element Mass and Stiffness Matrices — Exact Closed Forms

These are the only two "inner product of hat functions" integrals needed. Both have exact closed-form solutions for linear (P1) elements on triangles — no numerical quadrature required.

### 2.1 Signed Triangle Area

```
area = 0.5 * | (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0) |
            // use absolute value; degenerate if |area| < AREA_TOL = 1e-14
```

### 2.2 Element Mass Matrix (3×3, local node ordering v0,v1,v2)

The P1 mass matrix integrates `∫_T φ_i(x) φ_j(x) dA`:

```
M_e = (area / 12) * [[2, 1, 1],
                      [1, 2, 1],
                      [1, 1, 2]]
```

**Derivation (ASSUMED derivation, formula standard in FE textbooks):** For P1 elements, ∫_T φ_i φ_j dA = area/12 if i≠j, area/6 if i=j. The 3×3 closed form follows from integrating barycentric coordinate products over a triangle: ∫_T λ_i^α λ_j^β λ_k^γ dA = 2·area · α!β!γ! / (α+β+γ+2)!

### 2.3 Element Stiffness Matrix (3×3, Laplacian penalty)

The P1 stiffness matrix integrates `∫_T ∇φ_i · ∇φ_j dA` (Laplacian weak form, Neumann BC):

The gradient of the P1 hat function φ_i is **constant** over the triangle (because φ_i is linear):

```
∇φ_0 = (1/(2·area)) * [ (y1-y2),  (x2-x1) ]   // partial wrt x and y
∇φ_1 = (1/(2·area)) * [ (y2-y0),  (x0-x2) ]
∇φ_2 = (1/(2·area)) * [ (y0-y1),  (x1-x0) ]
```

Equivalently define:
```
b0 = y1 - y2,  c0 = x2 - x1
b1 = y2 - y0,  c1 = x0 - x2
b2 = y0 - y1,  c2 = x1 - x0
```

Then `∇φ_i = [b_i, c_i] / (2·area)`.

The element stiffness matrix:
```
K_e[i,j] = ∫_T ∇φ_i · ∇φ_j dA
          = (area / (4·area²)) * (b_i*b_j + c_i*c_j)
          = (b_i*b_j + c_i*c_j) / (4·area)
```

Expanded for all pairs (symmetric 3×3):
```
K_e = (1/(4*area)) * [[b0b0+c0c0, b0b1+c0c1, b0b2+c0c2],
                       [b1b0+c1c0, b1b1+c1c1, b1b2+c1c2],
                       [b2b0+c2c0, b2b1+c2c1, b2b2+c2c2]]
```

**Key properties of K_e:**
- Symmetric: `K_e[i,j] = K_e[j,i]` ✓
- Row sums equal zero: `K_e[i,0] + K_e[i,1] + K_e[i,2] = 0` — because `b0+b1+b2 = (y1-y2)+(y2-y0)+(y0-y1)=0` and similarly `c0+c1+c2=0`. This is the constant null-space property.
- PSD (not PD because of null space for constant functions) ✓
- For Neumann BC, the global K has exactly one zero eigenvalue (constant vector is in the null space). The Φ'Φ data-fit term restores identifiability as long as the observations are not collinear and `n_obs > 0`.

### 2.4 Global Assembly

Standard finite-element assembly — scatter element contributions into global N×N matrices:

```rust
// Global mass M and stiffness K: both N×N, stored flat row-major Vec<f64>
let mut M_global = vec![0.0_f64; N * N];
let mut K_global = vec![0.0_f64; N * N];

for tri in &triangles {
    let (v0, v1, v2) = (tri[0], tri[1], tri[2]);
    let (x0,y0) = nodes[v0];
    let (x1,y1) = nodes[v1];
    let (x2,y2) = nodes[v2];
    let area = 0.5 * ((x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)).abs();
    let (M_e, K_e) = element_matrices(x0,y0,x1,y1,x2,y2,area);
    let local = [v0, v1, v2];
    for (li, &gi) in local.iter().enumerate() {
        for (lj, &gj) in local.iter().enumerate() {
            M_global[gi * N + gj] += M_e[li][lj];
            K_global[gi * N + gj] += K_e[li][lj];
        }
    }
}
```

**Important:** Global K retains the row-sum-zero property of each element (constants in null space). Global M is symmetric positive-definite (since every node appears in at least one triangle; areas > 0 by validation).

---

## 3. SR-PDE Smoothing System

### 3.1 The Penalized Normal Equations

Given:
- `Φ`: n_obs × N observation matrix (built by barycentric eval, see §1.3)
- `K`: N × N global stiffness matrix (Laplacian roughness penalty)
- `y`: n_obs observed values
- `λ`: smoothing parameter (> 0)

The SR-PDE estimate of the node coefficient vector `c` (N × 1) solves:

```
(Φ'Φ + λ·K) c = Φ'y
```

`Φ'Φ` is N×N PSD (rank min(n_obs, N)), `λ·K` is N×N PSD with one null eigenvector (constant). The sum `(Φ'Φ + λK)` is generically PD as long as at least one observation does not lie on a node (so the constant null vector of K gets lifted by Φ'Φ). For safety, add a tiny ridge: `+ ε·I` with ε = 1e-10 before calling `cholesky_solve`.

**Build using existing `linalg.rs` (no new code):**
```rust
// Step 1: Φ'Φ  (n_obs << N typically, but Φ is dense n_obs×N so full multiply)
// Can use compute_xtx if Φ is wrapped in FdMatrix, else manual double loop O(n_obs*N^2)
let mut phi_t_phi = vec![0.0_f64; N * N];   // row-major
for i in 0..n_obs {
    for a in 0..N {
        for b in a..N {
            phi_t_phi[a*N+b] += phi[(i,a)] * phi[(i,b)];
            phi_t_phi[b*N+a] = phi_t_phi[a*N+b];
        }
    }
}

// Step 2: system A = Φ'Φ + λ*K + ε*I
let mut A = phi_t_phi.clone();
for a in 0..N {
    for b in 0..N {
        A[a*N+b] += lambda * K_global[a*N+b];
    }
    A[a*N+a] += 1e-10;  // ridge
}

// Step 3: Φ'y (N × 1)
let mut phi_t_y = vec![0.0_f64; N];
for i in 0..n_obs {
    for a in 0..N {
        phi_t_y[a] += phi[(i,a)] * y[i];
    }
}

// Step 4: solve
let c = cholesky_solve(&A, &phi_t_y, N)?;
```

### 3.2 GCV and EDF

The hat matrix `H = Φ (Φ'Φ + λK)^{-1} Φ'` has trace = edf.

**Numerically safe approach for v1 (dense, N modest):**

```
// A_inv: N×N  (from Cholesky inverse)
// edf = tr(Φ A_inv Φ') = tr(A_inv Φ'Φ)
//     = Σ_a,b  A_inv[a,b] * (Φ'Φ)[b,a]
//     = Σ_a,b  A_inv[a,b] * phi_t_phi[b,a]
//     = elementwise dot product of A_inv and phi_t_phi (both N×N, symmetric)
```

This avoids forming the n_obs×n_obs matrix H. Compute by:

```rust
// Get A_inv: solve A * A_inv = I column by column
let l = cholesky_factor(&A, N)?;
let mut a_inv = vec![0.0_f64; N * N];
let mut e_col = vec![0.0_f64; N];
for j in 0..N {
    e_col.iter_mut().for_each(|v| *v = 0.0);
    e_col[j] = 1.0;
    let col = cholesky_forward_back(&l, &e_col, N);
    for i in 0..N {
        a_inv[i*N+j] = col[i];   // row-major: a_inv[i,j]
    }
}

// edf = tr(A_inv * Φ'Φ) = Σ_i (A_inv * Φ'Φ)[i,i]
let mut edf = 0.0_f64;
for a in 0..N {
    for b in 0..N {
        edf += a_inv[a*N+b] * phi_t_phi[b*N+a];  // a_inv[a,b] * Φ'Φ[b,a]
    }
}
```

Then:
```
let fitted_obs: Vec<f64> = (0..n_obs).map(|i| {
    (0..N).map(|a| phi[(i,a)] * c[a]).sum()
}).collect();

let rss: f64 = (0..n_obs).map(|i| (y[i]-fitted_obs[i]).powi(2)).sum();
let gcv_denom = 1.0 - edf / n_obs as f64;
let gcv = if gcv_denom.abs() > 1e-10 {
    rss / n_obs as f64 / (gcv_denom * gcv_denom)
} else {
    f64::INFINITY
};
```

**Cost:** A_inv computation is O(N³) (N Cholesky forward-back solves, each O(N²)). For v1 with modest N (say ≤2000) this is acceptable.

### 3.3 GCV Lambda Search

Mirror `smooth_basis_gcv`: loop over `log_lambda_range` grid, call `fem_smooth(...)` at each λ, return the result with minimum GCV. Add `fem_smooth_gcv` public function.

---

## 4. Ramsay Monotone Smoother — Precise Implementation Recipe

### 4.1 The Model

For observed scalar data `(t_i, y_i)`, i=1..n, the Ramsay monotone smoother fits:

```
f(t) = β₀ + β₁ · ∫₀ᵗ exp(w(u)) du
```

where `w(u) = Ψ(u)' α` is a B-spline expansion (nbasis coefficients `α`), and `β₀ ∈ ℝ`, `β₁ > 0` (if β₁ > 0, f is strictly increasing; β₁ < 0 gives strictly decreasing — for v1 fix `β₁ > 0` or allow sign flip based on data monotone direction). Since `exp(w(u)) > 0`, the integrand is positive everywhere, so f is guaranteed monotone regardless of `α`.

**Parameters to estimate:** `θ = (β₀, β₁, α)` — total dim = 2 + nbasis.

### 4.2 Evaluating the Integral ∫₀ᵗ exp(w(u)) du

Use Simpson quadrature on a fixed fine grid over [min(t), max(t)]:

```rust
// Precompute once per NLS iteration (α changes each step):
// 1. Build quadrature grid q_pts (n_quad points, uniform over [t_min, t_max])
// 2. Evaluate Ψ(q_pts): bspline_basis_from_knots(q_pts, knots, order) → n_quad × nbasis
// 3. w_vals[k] = Ψ(q_k)' α for each k
// 4. exp_w[k] = exp(w_vals[k])
// 5. For each t_i, integrate exp_w from q_pts[0] to t_i via cumulative Simpson:
//    use simpsons_weights on the subgrid up to t_i, or use a running prefix sum

// Efficient prefix-sum approach:
// Compute cumulative_integral[k] = ∫_{q_pts[0]}^{q_pts[k]} exp(w(u)) du
// using the trapezoidal rule on the fine grid (or Simpson; trapezoidal suffices)
// Then for each t_i, find its position in q_pts and interpolate.
```

In practice, since t_i are data points, choose q_pts to include all t_i directly. Sort t_i once, evaluate the integral cumulatively:

```rust
// Simple trapezoidal prefix sum (sufficient for n_quad ~ 200):
let mut integral_prefix = vec![0.0_f64; n_quad];
for k in 1..n_quad {
    let dt = q_pts[k] - q_pts[k-1];
    integral_prefix[k] = integral_prefix[k-1] + 0.5 * dt * (exp_w[k-1] + exp_w[k]);
}
// F(t_i) = integral_prefix[idx_of_t_i]  (or linear interpolation if t_i not on grid)
```

### 4.3 Nonlinear Least Squares via Gauss-Newton

The NLS objective is `min_{β₀,β₁,α} Σ (y_i - f(t_i))² + λ_mono · α' R α` where `R` is the B-spline penalty matrix (from `bspline_penalty_matrix`, reused from `smooth_basis.rs`) and `λ_mono` is a smoothing parameter. The penalty ensures a smooth w(u) without oscillations.

At each iteration, linearize around current `(β₀, β₁, α)`:

**Jacobian rows (partial derivatives of f(t_i) w.r.t. parameters):**

```
∂f/∂β₀ = 1
∂f/∂β₁ = ∫₀^{t_i} exp(w(u)) du        ≡  I_i  (the precomputed integral)
∂f/∂α_j = β₁ · ∫₀^{t_i} exp(w(u)) · Ψ_j(u) du
```

The last partial requires integral of `exp(w(u)) * Ψ_j(u)`. Compute similarly with prefix sum:

```rust
// For each basis function j:
// integrand_j[k] = exp_w[k] * psi[k, j]
// use same trapezoidal prefix sum to get J_alpha[i, j] = β₁ * prefix_at_t_i
```

This requires one pass per basis function per evaluation point — O(n_quad * nbasis) per iteration.

**Gauss-Newton update:**

```
J: n × (2 + nbasis)  // Jacobian matrix
r: n                  // residuals y_i - f(t_i)

// Penalized GN system (augmented by roughness penalty on α):
// Build: (J'J + P) δθ = J'r
// where P has zeros for β₀,β₁ rows/cols, and λ_mono * R in the α block

delta_theta = cholesky_solve(&(JtJ + P), &Jt_r, 2 + nbasis)?;
theta += delta_theta;

// Enforce β₁ > 0 after each step (clamp if needed):
beta1 = beta1.max(1e-8);
```

**Stopping rule:**
```
if |delta_theta|_inf / (1.0 + |theta|_inf) < 1e-7:  converge
if iter >= max_iter (default 50): break with current best
```

**Initialization:**
- `β₀ = y[0]` (first observation or y_min)
- `β₁ = (y_max - y_min).abs().max(1e-6)` — roughly expected total increase
- `α = vec![0.0; nbasis]` — starts with w(u) = 0 everywhere, i.e., exp(w)=1, so ∫₀ᵗ exp(w)du = t; f(t) ≈ β₀ + β₁·(t-t_min). This is already a valid (affine) monotone fit — a safe starting point.

### 4.4 Non-Convergence Handling

If Gauss-Newton fails (Cholesky of J'J+P fails → not PD), fall back by increasing diagonal ridge `ε_gn = ε_gn * 10` (Levenberg-Marquardt damping). After 3 consecutive ridge-boosts without progress, return `FdarError::ComputationFailed`. This mirrors convergence handling in elastic modules.

### 4.5 Convergence Guarantee Discussion

The Ramsay integral-of-exp representation guarantees **monotonicity of the fitted function** regardless of convergence quality, because `exp(w(u)) > 0` always. Even if GN stops early, the current iterate still produces a monotone fit. The penalty λ_mono prevents wild oscillation in w(u). For well-conditioned data (monotone y, no repeated t), GN typically converges in 5-15 iterations.

### 4.6 Public API in smooth_basis.rs

```rust
/// Result of Ramsay monotone smoothing.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct MonotoneSmoothResult {
    /// Fitted values at the input argvals.
    pub fitted: Vec<f64>,
    /// Intercept β₀.
    pub beta0: f64,
    /// Scale β₁ (positive → increasing, negative → decreasing).
    pub beta1: f64,
    /// B-spline coefficients for w(u).
    pub w_coefs: Vec<f64>,
    /// Number of Gauss-Newton iterations to convergence.
    pub n_iter: usize,
    /// Residual sum of squares.
    pub rss: f64,
}

/// Monotone smoothing via Ramsay integral-of-exponential representation.
///
/// Fits `f(t) = β₀ + β₁ ∫₀ᵗ exp(w(u)) du` where `w` is a B-spline,
/// guaranteeing a monotone-increasing fitted function.
///
/// R baseline: `fda::smooth.monotone`.
///
/// # Divergence from R
/// - Dense Gauss-Newton (no sparse). Modest n ≤ 500 is practical.
/// - λ_mono controls B-spline roughness of w(u) (must be ≥ 0).
pub fn smooth_monotone(
    argvals: &[f64],
    y: &[f64],
    nbasis: usize,
    lambda_mono: f64,
    max_iter: usize,
) -> Result<MonotoneSmoothResult, FdarError>
```

---

## 5. Log-Domain Positive Smoother

### 5.1 Algorithm

```
1. Validate: all y_i > 0 (or warn and clamp via y.iter().map(|&v| v.max(EPS)))
2. log_y = y.map(|v| v.max(SMOOTH_EPS).ln())    // SMOOTH_EPS = 1e-10
3. Wrap log_y as FdMatrix (1 row × m cols) and argvals
4. Call existing smooth_basis(&log_data, argvals, fdpar) → SmoothBasisResult
5. fitted_positive = SmoothBasisResult.fitted.map(|v| v.exp())
6. Return PositiveSmoothResult { fitted_positive, log_fitted, edf, gcv, ... }
```

### 5.2 Bias Caveat (must document in rustdoc)

The log-domain smoother minimizes `||log(y) - Φc||²` not `||y - exp(Φc)||²`. The back-transformed fitted values `exp(Φ̂c)` are **not** the optimal positive smoother for the original scale — they tend to underestimate the conditional mean by a Jensen's inequality correction factor. Document this explicitly in the rustdoc: "This is a log-transform smoother; it minimizes squared error in log space. For unbiased estimation in the original scale, apply `exp(fitted + variance/2)` where variance is the conditional variance of the log-transformed fit." Do not apply the correction automatically in v1.

### 5.3 Public API in smooth_basis.rs

```rust
/// Result of log-domain positive smoothing.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct PositiveSmoothResult {
    /// Fitted values in original (positive) scale. All entries ≥ 0.
    pub fitted: Vec<f64>,
    /// Fitted values in log scale (from the underlying smooth_basis call).
    pub log_fitted: Vec<f64>,
    /// Effective degrees of freedom.
    pub edf: f64,
    /// GCV of the log-scale fit.
    pub gcv: f64,
}

/// Positive-valued smoothing via log-domain transformation.
///
/// Smooths `log(max(y, ε))` using B-spline basis smoothing and reconstructs
/// via `exp(·)`, guaranteeing a nonnegative fitted function.
///
/// # Divergence from R
/// Bias in original scale: fit minimizes squared error in log space.
/// See rustdoc for correction formula.
pub fn smooth_positive(
    argvals: &[f64],
    y: &[f64],
    fdpar: &FdPar,
) -> Result<PositiveSmoothResult, FdarError>
```

---

## 6. Result Struct: FemSmoothResult

Mirror `SmoothBasisResult` and `FosrResult2d` field conventions:

```rust
/// Result of FEM/PDE-regularized surface smoothing.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FemSmoothResult {
    /// Fitted values at mesh nodes (length N).
    pub node_values: Vec<f64>,
    /// Fitted values at the observation locations (length n_obs).
    pub fitted_obs: Vec<f64>,
    /// Effective degrees of freedom (trace of hat matrix).
    pub edf: f64,
    /// GCV score (INFINITY if edf ≥ n_obs).
    pub gcv: f64,
    /// Residual sum of squares at observations.
    pub rss: f64,
    /// Smoothing parameter λ used.
    pub lambda: f64,
    /// Number of mesh nodes N.
    pub n_nodes: usize,
    /// Number of triangles T.
    pub n_triangles: usize,
}
```

**Evaluating at new points:** The caller uses `assemble_fem_matrices` once, then can call a separate `fem_predict(result, new_xy, nodes, triangles)` function that runs barycentric eval on new points and returns `Σ_k φ_k(x) * node_values[k]`.

---

## 7. Public API Surface of fem_smoothing.rs

```rust
/// Assemble global mass M and stiffness K matrices for a triangulated mesh.
/// Returns (M, K) both N×N stored row-major flat Vec<f64>.
pub fn assemble_fem_matrices(
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
) -> Result<(Vec<f64>, Vec<f64>), FdarError>

/// PDE-regularized surface smoothing at fixed λ.
pub fn fem_smooth(
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
    obs_xy: &[[f64; 2]],
    y: &[f64],
    lambda: f64,
) -> Result<FemSmoothResult, FdarError>

/// PDE-regularized surface smoothing with GCV-optimal λ.
pub fn fem_smooth_gcv(
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
    obs_xy: &[[f64; 2]],
    y: &[f64],
    log_lambda_range: (f64, f64),
    n_grid: usize,
) -> Result<FemSmoothResult, FdarError>

/// Evaluate the fitted surface at new (x,y) locations.
pub fn fem_predict(
    node_values: &[f64],
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
    query_xy: &[[f64; 2]],
) -> Result<Vec<f64>, FdarError>
```

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Dense SPD linear system solve | Custom Cholesky | `cholesky_solve` in `src/linalg.rs:131` | Already battle-tested; avoids divergent reimplementation |
| B-spline evaluation for w(u) | New spline evaluator | `bspline_basis_from_knots` in `src/basis/bspline.rs:62` | Handles boundary knots and recurrence correctly |
| B-spline penalty matrix for Ramsay | Numeric differencing from scratch | `bspline_penalty_matrix` in `src/smooth_basis.rs:82` | Already integrates derivatives via fine quadrature |
| 1D penalized smoothing for positive smoother | Duplicate of smooth_basis | `smooth_basis` in `src/smooth_basis.rs:174` | Reuse proven GCV path |
| Simpson weights for Ramsay integral | Custom quadrature | `simpsons_weights` in `src/helpers.rs:57` | Already implements composite Simpson with boundary correction |
| Knot vector construction for Ramsay w(u) | Manual knot layout | `construct_bspline_knots` in `src/basis/bspline.rs:4` | Handles extended boundary knots correctly |

---

## Common Pitfalls

### Pitfall 1: Degenerate Triangle (Zero Area)
**What goes wrong:** `element_stiffness` divides by `4 * area`; area=0 → NaN/Inf propagated to entire K; Cholesky fails silently with all-zero row.
**Why it happens:** Collinear or repeated nodes in user-supplied mesh.
**How to avoid:** During `mesh_validate`, check each triangle: `area < AREA_TOL` (1e-12 * bounding_box_area) → `FdarError::InvalidParameter { parameter: "triangles[i]", message: "degenerate triangle with near-zero area" }`. Do NOT check inside the hot assembly loop — validate once at entry.
**Warning signs:** Stiffness matrix has NaN on diagonal.

### Pitfall 2: Observation Point Outside Mesh
**What goes wrong:** No triangle passes the barycentric containment test; Φ row remains all zeros; Φ'Φ is rank-deficient; Cholesky fails.
**Why it happens:** User passes observation coordinates outside the mesh boundary.
**How to avoid:** After the point-location loop, if no triangle found for point i, return `FdarError::InvalidParameter { parameter: "obs_xy[i]", message: "observation point lies outside the triangulated mesh" }`.
**Warning signs:** Row of Φ is all zeros.

### Pitfall 3: K Null Space Causing Cholesky Failure
**What goes wrong:** If `n_obs` is very small or all observations land on the same node, `Φ'Φ` may not lift the constant null space of K, making `Φ'Φ + λK` singular.
**Why it happens:** Fundamental: the constant function has zero roughness penalty and may have zero data fit if observations collapse.
**How to avoid:** Always add ridge `ε·I` (ε = 1e-10) to the system. If Cholesky still fails, surface the error with hint: "try adding more observation points or reducing lambda".
**Warning signs:** `cholesky_factor` returns `ComputationFailed`.

### Pitfall 4: Ramsay GN Divergence / Non-Monotone Data
**What goes wrong:** If input y is NOT monotone (has dips), the Gauss-Newton NLS for monotone w(u) will find a compromise but may take many iterations or oscillate between high-penalty solutions.
**Why it happens:** The model forces monotonicity even if data is not; w(u) must compensate via amplitude, straining the penalized system.
**How to avoid:** Document that `smooth_monotone` fits the best monotone approximation; it does not require y to be monotone. Use Levenberg-Marquardt damping (increase ridge on J'J if a step increases RSS). Convergence in ≤ 50 iterations is the stopping criterion.
**Warning signs:** RSS increases between iterations (signals step too large — increase damping ε_gn).

### Pitfall 5: Column-Major vs Row-Major Layout Confusion
**What goes wrong:** `FdMatrix` is column-major (`data[i + j*nrows]`) but the linalg helpers (`cholesky_solve`, `compute_xtx`) use row-major flat layout for square matrices. Mixing them silently produces transposed systems.
**Why it happens:** The codebase has both layouts for good reason (FdMatrix col-major per `matrix.rs:1-10`; linalg helpers use flat row-major as documented in `linalg.rs:77-79`).
**How to avoid:** Build Φ as a plain `Vec<f64>` in row-major layout (for the linalg helpers), NOT as FdMatrix. Use FdMatrix only for the final `FemSmoothResult.fitted_obs` if returning it as a functional data matrix.
**Warning signs:** GCV trace comes out negative or > N; fitted values are clearly wrong.

### Pitfall 6: Ramsay Integral Origin Convention
**What goes wrong:** The integral `∫₀ᵗ exp(w(u)) du` starts at 0, but data may be on [a, b] with a > 0. The β₀ term absorbs the offset `β₁ · ∫₀ᵃ exp(w(u)) du`, but if t_i values are passed as-is (starting at 0) and the prefix sum starts at t[0], the boundary is inconsistent.
**How to avoid:** Normalize: shift t by t[0] so integration starts at 0: `t_shifted[i] = t[i] - t[0]`. Then the integral prefix naturally starts at 0. β₀ then represents f(t[0]).
**Warning signs:** Fitted function does not pass through or near the first observation.

### Pitfall 7: Log-Domain Smoother Applied to Non-Positive Data
**What goes wrong:** `log(0)` or `log(negative)` → NaN/−Inf propagated through smooth_basis → NaN coefficients → NaN fitted values.
**How to avoid:** Clamp: `log_y[i] = y[i].max(1e-10).ln()` and document that values ≤ 0 are clamped to `SMOOTH_EPS`. Return a warning via an extra `bool` field `had_clamping` in `PositiveSmoothResult`, or include in rustdoc.
**Warning signs:** NaN in `SmoothBasisResult.fitted`.

---

## Code Examples

All code recipes in this document are [ASSUMED] (derived from standard FE textbook formulas and the visible codebase patterns). Verified source files are cited per-function.

### Element Stiffness (Inner Helper)

```rust
// Source: standard P1 FEM, e.g. Brenner & Scott "Mathematical Theory of FEM"
fn element_stiffness(
    x0: f64, y0: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
    area: f64,                // must be > 0
) -> [[f64; 3]; 3] {
    let b0 = y1 - y2;  let c0 = x2 - x1;
    let b1 = y2 - y0;  let c1 = x0 - x2;
    let b2 = y0 - y1;  let c2 = x1 - x0;
    let s = 1.0 / (4.0 * area);
    [
        [s*(b0*b0+c0*c0), s*(b0*b1+c0*c1), s*(b0*b2+c0*c2)],
        [s*(b1*b0+c1*c0), s*(b1*b1+c1*c1), s*(b1*b2+c1*c2)],
        [s*(b2*b0+c2*c0), s*(b2*b1+c2*c1), s*(b2*b2+c2*c2)],
    ]
}
```

### Element Mass (Inner Helper)

```rust
// Source: standard P1 FEM (∫_T λ_i λ_j dA = area/12 if i≠j, area/6 if i=j)
fn element_mass(area: f64) -> [[f64; 3]; 3] {
    let a = area / 12.0;
    [[2.0*a, a, a], [a, 2.0*a, a], [a, a, 2.0*a]]
}
```

### Barycentric Coordinates

```rust
// Source: standard 2D FE barycentric coord formula
fn barycentric(
    px: f64, py: f64,
    x0: f64, y0: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> Option<(f64, f64, f64)> {
    let det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0);
    if det.abs() < 1e-14 { return None; }  // degenerate
    let lam1 = ((px-x0)*(y2-y0) - (py-y0)*(x2-x0)) / det;
    let lam2 = ((py-y0)*(x1-x0) - (px-x0)*(y1-y0)) / det;
    let lam0 = 1.0 - lam1 - lam2;
    Some((lam0, lam1, lam2))
}
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Require user-supplied penalty matrix (like smooth_basis) | Auto-assemble from mesh geometry | User-facing: just pass nodes+triangles |
| Ad-hoc monotone spline (PAVA post-processing) | Ramsay integral-of-exp (built-in guarantee) | Monotonicity is structural, not post-hoc |
| Sparse FE assembly (fdaPDE, requires ndarray-sparse) | Dense assembly for v1 | Scales to ~2000 nodes easily; no new dep |

---

## Validation Architecture

### Test Framework

| Property | Value |
|----------|-------|
| Framework | Rust built-in `#[test]` (inline in module) |
| Config file | none — uses `cargo test` |
| Quick run command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --lib --features linalg 2>&1 \| grep -E "test.*fem\|test.*monotone\|test.*positive"` |
| Full suite command | `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg` |

### Phase Requirements → Test Map

| Req ID | Behavior | Test Type | Automated Command | Wave |
|--------|----------|-----------|-------------------|------|
| REP-02-01a | P1 basis partition-of-unity: `Σ φ_k(x,y) = 1` for interior points | unit | `cargo test test_fem_basis_partition_of_unity` | Wave 1 |
| REP-02-01b | P1 basis interpolation exactness: linear field recovered exactly | unit | `cargo test test_fem_basis_linear_exactness` | Wave 1 |
| REP-02-01c | Stiffness K symmetric + row-sums≈0 | unit | `cargo test test_stiffness_properties` | Wave 1 |
| REP-02-01d | Mass M symmetric + PD (all eigenvalues > 0 on small mesh) | unit | `cargo test test_mass_properties` | Wave 1 |
| REP-02-01e | Degenerate triangle → `FdarError` | unit | `cargo test test_fem_degenerate_triangle_error` | Wave 1 |
| REP-02-01f | Out-of-range connectivity index → `FdarError` | unit | `cargo test test_fem_bad_index_error` | Wave 1 |
| REP-02-02a | PDE smoothing recovers known smooth surface within tolerance | integration | `cargo test test_fem_smooth_recovers_surface` | Wave 2 |
| REP-02-02b | PDE smoothing → interpolation as λ→0 | integration | `cargo test test_fem_smooth_interpolation_limit` | Wave 2 |
| REP-02-02c | GCV helper returns finite score for valid inputs | unit | `cargo test test_fem_gcv_finite` | Wave 2 |
| REP-02-02d | Observation outside mesh → `FdarError` | unit | `cargo test test_fem_obs_outside_mesh_error` | Wave 2 |
| REP-02-03 | Positive smoother: all fitted values ≥ 0 | unit | `cargo test test_smooth_positive_nonneg` | Wave 3 |
| REP-02-03b | Positive smoother: correct shape on known positive data | integration | `cargo test test_smooth_positive_recovery` | Wave 3 |
| REP-02-04a | Monotone smoother: fitted values nondecreasing | unit | `cargo test test_smooth_monotone_nondecreasing` | Wave 3 |
| REP-02-04b | Monotone smoother: reasonable fit to monotone data | integration | `cargo test test_smooth_monotone_fit_quality` | Wave 3 |
| REP-02-04c | Non-convergence → `FdarError::ComputationFailed` (degenerate input) | unit | `cargo test test_smooth_monotone_convergence_error` | Wave 3 |

### Wave 0 Gaps

- [ ] No test files exist yet — all tests are new (inline `#[cfg(test)] mod tests` in fem_smoothing.rs and smooth_basis.rs additions).
- [ ] Need a simple 4-node, 2-triangle mesh fixture (a unit square split diagonally) for deterministic unit tests.

---

## Security Domain

`security_enforcement: true`, `security_asvs_level: 1`.

| ASVS Category | Applies | Standard Control |
|---------------|---------|-----------------|
| V5 Input Validation | yes | Dimension/range checks at entry, `FdarError` on invalid mesh/params |
| V2 Authentication | no | Pure numeric library |
| V3 Session Management | no | Pure numeric library |
| V4 Access Control | no | Pure numeric library |
| V6 Cryptography | no | No secrets |

| Pattern | STRIDE | Standard Mitigation |
|---------|--------|---------------------|
| Integer overflow in mesh index check (`tri[i] >= N`) | Tampering | Validate all indices against `nodes.len()` at entry |
| NaN propagation from degenerate triangle | Elevation of privilege (incorrect output) | Validate area > AREA_TOL before assembly |
| Denial of service from very large N (O(N³) dense inverse) | Denial | Document N ≤ 2000 recommended for v1; no async/cancellation needed in a library |

---

## Runtime State Inventory

SKIPPED — this is a greenfield module addition, not a rename/refactor/migration phase.

---

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain | All compilation | ✓ | 1.97.0 (per CLAUDE.md) | — |
| `linalg` feature (faer) | `cholesky_factor` etc. in linalg.rs | ✓ | Cargo feature flag | — |
| nalgebra 0.33 | smooth_basis.rs (DMatrix) | ✓ | already a dep | — |
| No new external deps | Milestone constraint | ✓ | — | — |

---

## Open Questions

1. **Observation matrix Φ layout**
   - What we know: Φ is n_obs × N, sparse-structured (3 non-zeros per row). `compute_xtx` in linalg.rs takes an `&FdMatrix`. Wrapping Φ as an FdMatrix (col-major) and calling `compute_xtx` would work but requires an n_obs×N allocation.
   - What's unclear: Is it worth a custom `compute_phi_t_phi` that exploits the 3-nonzero-per-row structure to skip many multiplications?
   - Recommendation: For v1, use the naive O(n_obs·N²) double loop (not `compute_xtx`) since Φ's layout is known at compile time; note the optimization opportunity in a comment.

2. **GCV inversion cost at large N**
   - What we know: A_inv computation is O(N³). For N=500 this is 125M ops — fast. For N=2000 it's 8B ops — ~8s at 1Gflops, which may be acceptable for a library.
   - Recommendation: Document the O(N³) cost in `fem_smooth_gcv` rustdoc. No optimization in v1.

3. **Ramsay monotone: decreasing fits**
   - What we know: The model `β₁ · ∫ exp(w) du` with `β₁ > 0` is increasing. The CONTEXT.md says "Ramsay integral-of-exp monotone smoother" without specifying direction.
   - Recommendation: Detect monotone direction from the data sign of `y[last] - y[0]` and set `β₁ = -|β₁_init|` for decreasing data. Document that the function fits the nondecreasing case by default and flips sign for obviously-decreasing data. Let the user override via a `MonotoneConfig` struct field if needed.

---

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | Element mass matrix closed form: `M_e = (area/12) * [[2,1,1],[1,2,1],[1,1,2]]` | §2.2 | Incorrect mass matrix → biased PDE solve (K assembly is independent; mass not used in v1 SR-PDE system but is a deliverable for REP-02-01) |
| A2 | Element stiffness closed form with `b_i = y_{i+1} - y_{i+2}` (cyclic indexing) / `c_i = x_{i+2} - x_{i+1}` | §2.3 | Wrong stiffness → wrong roughness penalty; GCV selects wrong λ |
| A3 | FdarError variants: `InvalidDimension`, `InvalidParameter`, `ComputationFailed` with exact field names `{parameter, expected, actual}` / `{parameter, message}` / `{operation, detail}` | §6 | Compile error; easy to fix from src/error.rs |
| A4 | Gauss-Newton convergence in ≤ 50 iterations for typical monotone data | §4.3 | May need more iterations; make `max_iter` a function param (already planned) |
| A5 | `bspline_basis_from_knots` returns flat column-major array of size `n_points × nbasis` | §4.2 | Wrong indexing in integral computation; verify by reading the return convention from bspline.rs:62-80 at implementation time |

---

## Sources

### Primary (HIGH confidence — Read in session)
- `src/linalg.rs:85-151` — exact function signatures for `cholesky_factor`, `cholesky_forward_back`, `cholesky_solve`, `compute_xtx`
- `src/smooth_basis.rs:1-677` — exact `smooth_basis`, `SmoothBasisResult`, `FdPar`, `BasisType`, `bspline_penalty_matrix` signatures and implementations
- `src/basis/bspline.rs:1-80` — `bspline_basis_from_knots`, `construct_bspline_knots`, recurrence implementation
- `src/matrix.rs:1-80` — `FdMatrix` column-major layout, `from_column_major` constructor
- `src/helpers.rs:57` — `simpsons_weights` signature
- `src/prelude.rs:1-79` — what is currently re-exported; where to add new types
- `src/lib.rs:64-137` — module list; where `pub mod fem_smoothing;` goes
- `.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-CONTEXT.md` — all locked decisions

### Secondary (MEDIUM confidence)
- Standard P1 FEM closed-form element matrices (Brenner & Scott, "Mathematical Theory of FEM", 3rd ed; Strang & Fix "Analysis of the Finite Element Method") — mass and stiffness formulas are textbook-standard, reviewed in multiple sources

### Tertiary (LOW confidence)
- Ramsay monotone smoother: Ramsay (1998) "Estimating smooth monotone functions" JRSS-B; general NLS Gauss-Newton scheme from training knowledge — implementation details are [ASSUMED] but the formulation is well-established in the fda literature

---

## Project Constraints (from CLAUDE.md)

- Rust edition 2021; MSRV 1.81.0 (linalg feature: 1.84.0+)
- All public fns return `Result<T, FdarError>` — no panics on user input
- `#[must_use]` on expensive computation functions
- `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]` on result structs
- `#[cfg_attr(feature = "serde", derive(...))]` on result types
- Inline `#[cfg(test)] mod tests` (not separate test files) for unit tests
- `cargo clippy --all-targets --features linalg,parallel -- -D warnings` must pass
- No new crate dependency
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp` prefix for any cargo invocation
- Document all divergences from R baseline (`fdaPDE`, `fda::smooth.monotone`) in rustdoc
- New module: `pub mod fem_smoothing;` in `src/lib.rs` + crate-root re-exports + `prelude.rs`
- Additive only: zero changes to existing public signatures

---

## Metadata

**Confidence breakdown:**
- Standard stack (no new deps): HIGH — all reused code verified by Read in session
- Element matrix formulas: HIGH for stiffness (derived from first principles); MEDIUM for mass (textbook formula, not verified against a running test)
- SR-PDE system: HIGH — linear algebra is standard, consistent with existing linalg.rs patterns
- GCV trace computation: HIGH — algebraic identity, implemented analogously to smooth_basis.rs line 214-215
- Ramsay NLS scheme: MEDIUM — standard Gauss-Newton; specific convergence rate is [ASSUMED]
- Architecture patterns: HIGH — directly mirrored from verified existing modules

**Research date:** 2026-08-24
**Valid until:** Stable (pure math formulas; no external ecosystem churn)
