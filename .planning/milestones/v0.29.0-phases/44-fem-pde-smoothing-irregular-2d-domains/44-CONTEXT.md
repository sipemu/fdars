# Phase 44: FEM/PDE Smoothing on Irregular 2D Domains - Context

**Gathered:** 2026-08-24
**Status:** Ready for planning
**Mode:** Smart discuss (autonomous) — grey areas proposed in batch, all accepted as recommended

<domain>
## Phase Boundary

Deliver REP-02: a new module `fdars-core/src/fem_smoothing.rs` implementing (1) a linear (P1)
finite-element basis over a user-supplied triangulated 2D mesh (nodes + triangle
connectivity), with basis-function evaluation and mass/stiffness assembly; (2) PDE-regularized
(Laplacian-penalty) surface smoothing of scattered observations over an irregular 2D domain,
returning a fitted surface plus diagnostics. Plus **additive** positive (log-domain,
nonnegative-guaranteed) and Ramsay integral-of-exp monotone smoothers added to
`fdars-core/src/smooth_basis.rs`.

Scope fences: **v1 is 2D triangulated meshes only** (3D tetrahedral FEM out of scope). Does
**not** overlap the existing regular-grid 2D FOSR strength (`function_on_scalar_2d.rs` /
tensor-product penalty) — this is irregular-mesh FEM. Numeric outputs only (no mesh/surface
plotting). Additive/non-breaking: `Result`-returning public fns, inline `#[cfg(test)]` tests,
crate-root + prelude re-exports, zero changes to existing public signatures. R baseline:
`fdaPDE` 1.1-24 (matched by capability; document divergences in rustdoc).

</domain>

<decisions>
## Implementation Decisions

### Mesh & Linear FE Basis
- Mesh input: user-supplied nodes (N×2 coordinates) + triangle connectivity (T×3 vertex indices) as explicit inputs (matches fdaPDE). No internal mesh generation.
- FE basis: linear P1 Lagrange "hat" basis — one basis function per node, value 1 at its node, 0 at others, linear over each triangle.
- Basis evaluation at arbitrary (x,y): barycentric coordinates within the containing triangle; point location via a linear scan over triangles for v1 (spatial index deferred).
- Boundary condition: Neumann (natural, zero-flux) — the standard choice for PDE surface smoothing.

### PDE-Regularized Smoothing
- Penalty: Laplacian roughness approximated via the FE **stiffness** matrix K (SPD); the **mass** matrix M provides the FE inner product. Follows the fdaPDE SR-PDE (spatial regression with PDE penalization) formulation.
- Assembly: per-triangle element mass + stiffness via linear-FE area-based closed forms, assembled into global N×N matrices (N = number of mesh nodes).
- Solve: penalized normal equations `(Φ'Φ + λK) c = Φ'y` (Φ = observation-to-node evaluation matrix; c = fitted node values) via dense `cholesky_solve` — v1 mesh sizes are modest; sparse solvers deferred.
- Smoothing parameter: fixed λ argument, plus a trace-based GCV helper for automatic selection (consistent with the `smooth_basis` GCV convention).

### Monotone & Positive Smoothers (additive to smooth_basis.rs)
- Positive smoother: log-domain — smooth in log space so the reconstructed fit `exp(·)` is guaranteed nonnegative.
- Monotone smoother: Ramsay integral-of-exp — `f(t) = β₀ + β₁ ∫₀ᵗ exp(w(u)) du` with w expanded in a B-spline basis; the positive integrand guarantees a monotone-increasing fit.
- API shape: new standalone public fns (e.g. `smooth_positive` / `smooth_monotone`) added to `smooth_basis.rs` — additive; existing `smooth_basis*` signatures untouched.
- Domain: 1D functional data (consistent with `smooth_basis`).

### API/Result & Dependency Decision
- Result struct: `FemSmoothResult` — fitted node values, fitted-at-observations, edf/GCV diagnostic, and a mesh reference (field convention mirrors `SmoothBasisResult` / `FosrResult2d`).
- **Dependency decision: KEEP the no-new-dependency convention.** Implement the linear-FE mass/stiffness assembly + dense Cholesky solve entirely in-house (elementary linear algebra; feasible for v1 mesh sizes). This is the one phase permitted to revisit the constraint, but the decision is to stay in-house; sparse-matrix scaling is deferred to a future milestone (flag noted, not exercised).
- Module layout: single new `fem_smoothing.rs` (mesh + basis + assembly + smoothing) + additive edits to `smooth_basis.rs`.
- Mesh validation: connectivity indices in range, non-degenerate triangles (strictly positive area) → `FdarError` on violation.

### Claude's Discretion
- Exact struct field naming, internal helper decomposition, plan/wave decomposition, and the precise GCV/edf trace computation are at the planner/implementer's discretion within the accepted decisions and existing conventions.
- Whether the FEM subsystem warrants a folder split (if `fem_smoothing.rs` exceeds ~500 lines) is at the planner's discretion per the existing factoring convention.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `cholesky_factor` / `cholesky_forward_back` / `cholesky_solve` / `compute_xtx` (`src/linalg.rs`, `linalg` feature) — dense SPD solves for the penalized normal equations and GCV trace.
- `smooth_basis.rs`: `bspline_penalty_matrix`, `smooth_basis`, `smooth_basis_gcv`, `SmoothBasisResult`, `FdPar`, `BasisType` — the monotone/positive smoothers are added here; B-spline basis machinery reused for the Ramsay integral-of-exp w(u) expansion.
- `function_on_scalar_2d.rs`: `Grid2d`, `FosrResult2d`, tensor-product 2D penalty (`src/function_on_scalar_2d.rs:195`) — the **regular-grid** A-6 strength REP-02 must NOT overlap; kept disjoint (irregular mesh vs regular grid).
- `FdMatrix` (column-major) for node coordinates, connectivity (or a `Vec<[usize; 3]>`), assembled matrices, and fitted surfaces.
- `bspline_basis` / `bspline_basis_from_knots` (`src/basis/bspline.rs`) — evaluate B-spline at arbitrary points for the monotone smoother's integrand.

### Established Patterns
- All public fns return `Result<T, FdarError>`; dimension/param validation at entry (no panics).
- `#[must_use]` + `#[derive(Debug, Clone, PartialEq)]` + `#[non_exhaustive]` on result structs; conditional serde via `cfg_attr`.
- New top-level module wired via `pub mod fem_smoothing;` in `src/lib.rs` + crate-root re-exports + key types added to `src/prelude.rs`.
- Inline `#[cfg(test)] mod tests` with `crate::test_helpers::uniform_grid` and shape/recovery/error-path assertions.

### Integration Points
- `src/lib.rs` module list + crate-root `pub use fem_smoothing::{...}` re-export block; `src/prelude.rs` for the key result/mesh types. Additive edits only to `smooth_basis.rs` for the two new smoothers.

</code_context>

<specifics>
## Specific Ideas

- R baseline capability parity: `fdaPDE` SR-PDE (Laplacian-penalized spatial regression over a triangulated mesh) and Ramsay's monotone smoothing (`fda::smooth.monotone`). Match by capability, not exact R signatures; document divergences (e.g. dense vs sparse assembly, Neumann-only BC, no anisotropic/advection PDE terms in v1) in rustdoc.
- Test oracles: (1) FE basis partition-of-unity (Σ_k φ_k(x)=1 inside the domain) and interpolation exactness for a linear field; (2) mass/stiffness matrix symmetry + PSD, stiffness row-sums ≈ 0 (constant in null space); (3) PDE smoothing recovers a smooth known surface on a mesh within tolerance and reduces to interpolation as λ→0; (4) positive smoother fit ≥ 0 everywhere; (5) monotone smoother fit is nondecreasing; (6) error paths (bad connectivity, degenerate triangle, dimension mismatch).

</specifics>

<deferred>
## Deferred Ideas

- 3D tetrahedral-mesh FEM (v1 is 2D triangles only).
- Sparse-matrix assembly/solvers and a spatial index for point location (dense + linear scan for v1; the one place a new dependency was permitted — decision is to stay in-house and defer sparse scaling).
- Quadratic (P2) or higher-order elements; Dirichlet/Robin boundary conditions; anisotropic/advection-diffusion PDE penalty terms (Neumann + isotropic Laplacian only in v1).
- Areal/regional observations and space-varying PDE coefficients (fdaPDE advanced features).
- I-spline / PAVA monotone alternatives and nonnegative-least-squares positive alternative (log-domain + Ramsay integral-of-exp are the v1 choices).

</deferred>
