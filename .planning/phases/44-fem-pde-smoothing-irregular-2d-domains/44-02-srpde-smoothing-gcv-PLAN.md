---
phase: 44-fem-pde-smoothing-irregular-2d-domains
plan: 02
type: execute
wave: 2
depends_on: ["44-01"]
files_modified:
  - fdars-core/src/fem_smoothing.rs
  - fdars-core/src/lib.rs
autonomous: true
requirements: [REP-02-02]
estimate:
  tokens: 66000
  raw_tokens: 36000
  tasks: 3
  confidence: med
must_haves:
  truths:
    - "fem_smooth solves (Φ'Φ + λK)c = Φ'y and recovers a known smooth surface on a mesh within tolerance"
    - "As λ→0 fem_smooth approaches interpolation of the observations (small residuals at obs points)"
    - "fem_smooth_gcv returns a finite GCV score and a lambda from the search grid; fem_predict evaluates the fitted surface at new points"
    - "Observation point outside the mesh returns FdarError (no panic)"
  artifacts:
    - "fem_smooth, fem_smooth_gcv, fem_predict public fns added to fdars-core/src/fem_smoothing.rs"
    - "extended crate-root re-export block in fdars-core/src/lib.rs including fem_smooth, fem_smooth_gcv, fem_predict"
  key_links:
    - "Φ built as row-major flat Vec<f64> (n_obs×N), Φ'Φ + λK + εI assembled row-major, solved with pub(crate) cholesky_solve — NOT FdMatrix"
    - "edf = tr(A_inv · Φ'Φ) via elementwise dot of two symmetric N×N matrices (no dense n_obs×n_obs hat matrix)"
---

<objective>
Deliver REP-02-02: PDE-regularized (Laplacian-penalty) surface smoothing of scattered observations over the irregular 2D FE mesh built in Plan 01. Solve the SR-PDE penalized normal equations `(Φ'Φ + λK)c = Φ'y` via dense in-house Cholesky, compute trace-based GCV/edf, add a GCV λ-search helper and a `fem_predict` evaluator. Builds directly on Plan 01's `assemble_fem_matrices` + `fem_basis_eval`.

Purpose: Turns the FE basis + stiffness matrix into a usable surface smoother with automatic smoothing-parameter selection and out-of-sample prediction — the core user-facing capability of REP-02.
Output: `fem_smooth`, `fem_smooth_gcv`, `fem_predict` public fns in `fem_smoothing.rs`, extended re-exports, inline tests (surface recovery, interpolation limit, finite GCV, outside-mesh error).
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-CONTEXT.md
@.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-RESEARCH.md

@fdars-core/src/linalg.rs
@fdars-core/src/fem_smoothing.rs
</context>

<artifacts_this_phase_produces>
New public symbols introduced by THIS plan (wave 2), added to `fem_smoothing.rs`:
- `fem_smooth(nodes, triangles, obs_xy, y, lambda) -> Result<FemSmoothResult, FdarError>` — SR-PDE surface smoothing at fixed λ
- `fem_smooth_gcv(nodes, triangles, obs_xy, y, log_lambda_range, n_grid) -> Result<FemSmoothResult, FdarError>` — GCV-optimal λ
- `fem_predict(node_values, nodes, triangles, query_xy) -> Result<Vec<f64>, FdarError>` — evaluate fitted surface at new points
- Extended crate-root re-export: `pub use fem_smoothing::{assemble_fem_matrices, fem_basis_eval, fem_predict, fem_smooth, fem_smooth_gcv, FemSmoothResult};`

Reuses (Plan 01): `assemble_fem_matrices`, `fem_basis_eval`, `FemSmoothResult`, `mesh_validate`, `barycentric`/`locate_point` inner helpers.
Reuses (linalg.rs, pub(crate)): `cholesky_factor`, `cholesky_forward_back`, `cholesky_solve`.
</artifacts_this_phase_produces>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end SR-PDE solve on the unit-square fixture — build Φ, solve (Φ'Φ+λK)c=Φ'y, return FemSmoothResult</name>
  <files>fdars-core/src/fem_smoothing.rs, fdars-core/src/lib.rs</files>
  <read_first>
    - fdars-core/src/linalg.rs:85-134 — `cholesky_factor(a,p)`, `cholesky_forward_back(l,b,p)`, `cholesky_solve(a,b,p)` (all pub(crate), ROW-MAJOR flat square matrices, index i*p+j).
    - fdars-core/src/fem_smoothing.rs — Plan 01: `assemble_fem_matrices` (returns row-major M,K), `locate_point`/`barycentric` inner helpers, `FemSmoothResult` fields, the unit-square test fixture.
    - RESEARCH §3.1 (penalized normal equations, ε=1e-10 ridge) and §Pitfall 5 (row-major vs column-major — Φ must be row-major flat, NOT FdMatrix).
  </read_first>
  <action>
Implement `pub fn fem_smooth(nodes: &[[f64;2]], triangles: &[[usize;3]], obs_xy: &[[f64;2]], y: &[f64], lambda: f64) -> Result<FemSmoothResult, FdarError>` with `#[must_use]`.

Validate at entry: `obs_xy.len() == y.len()` else `InvalidDimension`; `lambda >= 0.0` else `InvalidParameter { parameter: "lambda", .. }`; `y` non-empty. (mesh is validated inside assemble/eval.)

Build the observation matrix Φ (n_obs × N) as a ROW-MAJOR flat `Vec<f64>` of length `n_obs*N` (per RESEARCH §1.3 / Pitfall 5, NOT an FdMatrix): reuse Plan 01's `fem_basis_eval(nodes, triangles, obs_xy)?` to get, per observation, the containing triangle's 3 `(node_index, hat_value)` pairs; set `phi[i*N + node_index] = hat_value` for the 3 nonzeros; a point outside the mesh already surfaces as FdarError from `fem_basis_eval`.

Assemble K via `assemble_fem_matrices(nodes, triangles)?` (take the `.1` stiffness; M is not needed in the v1 SR-PDE system but remains a Plan-01 deliverable). Build the system row-major (RESEARCH §3.1):
- `phi_t_phi[a*N+b] = Σ_i phi[i*N+a]*phi[i*N+b]` (symmetric N×N).
- `A = phi_t_phi + lambda*K` elementwise, then add ridge `A[a*N+a] += 1e-10` to lift K's constant null space.
- `phi_t_y[a] = Σ_i phi[i*N+a]*y[i]` (length N).
Solve `let c = crate::linalg::cholesky_solve(&A, &phi_t_y, N)?;` — c is the fitted node-value vector.

Compute `fitted_obs[i] = Σ_a phi[i*N+a]*c[a]`, `rss = Σ_i (y[i]-fitted_obs[i])^2`. For the tracer, set `edf = f64::NAN` and `gcv = f64::NAN` as placeholders (real edf/GCV computed in Task 2 — do NOT compute the dense inverse here yet). Return `FemSmoothResult { node_values: c, fitted_obs, edf, gcv, rss, lambda, n_nodes: nodes.len(), n_triangles: triangles.len() }`.

Extend the lib.rs crate-root re-export block to `pub use fem_smoothing::{assemble_fem_matrices, fem_basis_eval, fem_smooth, FemSmoothResult};` (the remaining two fns join in Task 2/3).

Tracer test `test_fem_smooth_solves_and_reduces_residual`: on a refined mesh (see below), sample a smooth field, fit at a small-to-moderate λ, and assert (a) `node_values.len() == nodes.len()`, (b) `rss` is finite and small relative to the response variance (e.g. `rss / n_obs < 0.1 * var(y)`), (c) `fitted_obs.len() == obs_xy.len()`. Add a `#[cfg(test)]` helper `refined_square_mesh()` that builds a small regular triangulated grid (e.g. a 4×4 node grid = 16 nodes, 18 triangles, each grid cell split into 2 triangles) with a helper to place observation points at cell interiors. Keep it deterministic (no RNG).
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing::tests::test_fem_smooth_solves_and_reduces_residual 2>&1 | tail -20</automated>
  </verify>
  <done>fem_smooth builds Φ row-major, solves the SR-PDE system via cholesky_solve, and returns finite node_values + fitted_obs with small residual on a refined mesh; fem_smooth re-exported.</done>
</task>

<task type="auto">
  <name>Task 2: Trace-based GCV/edf + surface-recovery and interpolation-limit oracles</name>
  <files>fdars-core/src/fem_smoothing.rs</files>
  <read_first>
    - fdars-core/src/linalg.rs:85-128 — `cholesky_factor` then `cholesky_forward_back` for column-by-column inverse.
    - fem_smoothing.rs Task 1 — fem_smooth builds A (row-major), phi_t_phi, fitted_obs, rss.
    - RESEARCH §3.2 — edf = tr(A_inv·Φ'Φ) = elementwise dot of A_inv and phi_t_phi (both symmetric N×N); GCV = (rss/n)/(1 - edf/n)^2, INFINITY if denom ~0.
  </read_first>
  <action>
Refactor `fem_smooth` (or add an inner helper it calls) to compute real `edf` and `gcv` instead of the NaN placeholders (RESEARCH §3.2):
- Factor `let l = crate::linalg::cholesky_factor(&A, N)?;`.
- Build `A_inv` (N×N row-major) column-by-column: for `j in 0..N`, solve `cholesky_forward_back(&l, &e_j, N)` where `e_j` is the j-th unit vector, and write the result into column j: `a_inv[i*N+j] = col[i]`.
- `edf = Σ_{a,b} a_inv[a*N+b] * phi_t_phi[b*N+a]` (elementwise dot exploiting symmetry; equals tr(A_inv·Φ'Φ)).
- `gcv_denom = 1.0 - edf / n_obs`; `gcv = if gcv_denom.abs() > 1e-10 { (rss/n_obs) / (gcv_denom*gcv_denom) } else { f64::INFINITY }`.
Store real `edf` and `gcv` in the returned `FemSmoothResult`. Add a rustdoc note on `fem_smooth` that the dense A_inv is O(N³) and v1 recommends N ≲ 2000 (RESEARCH §Open Questions 2 / Security DoS row).

Add tests:
- `test_fem_smooth_recovers_surface`: on `refined_square_mesh()`, sample a smooth ground-truth surface `g(x,y)=sin(π x)*sin(π y)` (or a low-order polynomial) at the observation points; fit with a moderate λ; assert the fitted surface at the observation points is close to `g` (mean-abs error below a tolerance, e.g. 0.15) — recovers a known smooth surface (RESEARCH §Validation REP-02-02a).
- `test_fem_smooth_interpolation_limit`: fit the same data at a very small λ (e.g. 1e-8); assert `rss` is smaller than at a large λ (e.g. λ=10.0) and the small-λ residual at observations is near zero (→ interpolation as λ→0, RESEARCH §Validation REP-02-02b). Compare the two fits' rss.
- `test_fem_gcv_finite`: assert `fem_smooth(...).unwrap().gcv` is finite (not NaN/Inf) and `edf > 0.0` and `edf <= n_obs` for a valid fit (RESEARCH §Validation REP-02-02c).
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing::tests::test_fem_smooth_recovers_surface fem_smoothing::tests::test_fem_smooth_interpolation_limit fem_smoothing::tests::test_fem_gcv_finite 2>&1 | tail -20</automated>
  </verify>
  <done>fem_smooth returns finite edf in (0, n_obs] and finite GCV; recovers a known smooth surface within tolerance; smaller λ yields smaller residual (interpolation limit).</done>
</task>

<task type="auto">
  <name>Task 3: fem_smooth_gcv λ-search + fem_predict evaluator + outside-mesh error path</name>
  <files>fdars-core/src/fem_smoothing.rs, fdars-core/src/lib.rs</files>
  <read_first>
    - fdars-core/src/smooth_basis.rs:254-299 — `smooth_basis_gcv` log-lambda grid pattern (loop over `10^log_lam`, keep min-GCV result) to mirror.
    - fem_smoothing.rs Tasks 1-2 — fem_smooth, fem_basis_eval, FemSmoothResult.
    - RESEARCH §3.3 (GCV λ search), §6 (fem_predict = Σ_k φ_k(x)*node_values[k]).
  </read_first>
  <action>
Implement `pub fn fem_smooth_gcv(nodes, triangles, obs_xy, y, log_lambda_range: (f64,f64), n_grid: usize) -> Result<FemSmoothResult, FdarError>` with `#[must_use]`, mirroring `smooth_basis_gcv` (RESEARCH §3.3): validate `n_grid >= 2` else `InvalidParameter`; loop `i in 0..n_grid`, `log_lam = lo + (hi-lo)*i/(n_grid-1)`, `lam = 10f64.powf(log_lam)`, call `fem_smooth(nodes, triangles, obs_xy, y, lam)`, keep the result with minimum finite `gcv`. If every grid point errored, return the last error; if all GCVs were non-finite, return `FdarError::ComputationFailed { operation: "fem_smooth_gcv", detail }` advising to widen the λ range or add observations.

Implement `pub fn fem_predict(node_values: &[f64], nodes: &[[f64;2]], triangles: &[[usize;3]], query_xy: &[[f64;2]]) -> Result<Vec<f64>, FdarError>` with `#[must_use]` (RESEARCH §6): validate `node_values.len() == nodes.len()` else `InvalidDimension`; reuse `fem_basis_eval(nodes, triangles, query_xy)?`; for each query point compute `Σ hat_value * node_values[node_index]` over its 3 nonzeros; return the vector. Points outside the mesh already surface as FdarError via fem_basis_eval.

Finalize the lib.rs crate-root re-export block to the complete set: `pub use fem_smoothing::{assemble_fem_matrices, fem_basis_eval, fem_predict, fem_smooth, fem_smooth_gcv, FemSmoothResult};`.

Add tests:
- `test_fem_smooth_gcv_selects_finite`: run `fem_smooth_gcv` on `refined_square_mesh()` data over `(-6.0, 2.0)`, `n_grid=9`; assert Ok, `gcv` finite, and the chosen `lambda` lies within `[10^-6, 10^2]`.
- `test_fem_predict_matches_nodes`: set `node_values` from a linear field at the mesh nodes; call `fem_predict` at interior query points; assert predictions equal the linear field within 1e-9 (P1 reproduces linear fields exactly).
- `test_fem_smooth_obs_outside_mesh_error`: call `fem_smooth` with one obs point outside the mesh (e.g. `[5.0,5.0]`); assert `Err(FdarError::InvalidParameter { .. })` (surfaced by fem_basis_eval, parameter "query_xy") — no panic.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing::tests 2>&1 | tail -25</automated>
  </verify>
  <done>fem_smooth_gcv returns a finite-GCV fit at a grid λ; fem_predict reproduces linear fields exactly at new points; obs-outside-mesh returns FdarError; all three fns re-exported; full fem_smoothing::tests module green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → fem_smooth / fem_smooth_gcv / fem_predict | Untrusted numeric observations (obs_xy, y) and λ cross into the dense solve |

Attack surface: none — pure in-process numeric library. Concerns are numerical: singular (Φ'Φ+λK), points outside the mesh, non-finite GCV. All handled as FdarError / ridge guard.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-44-04 | Elevation (incorrect output) | singular (Φ'Φ+λK) when K's constant null space is unlifted | medium | mitigate | add ε=1e-10 ridge before cholesky_solve; cholesky_factor returns ComputationFailed on residual singularity |
| T-44-05 | Tampering | observation point outside mesh → all-zero Φ row → rank deficiency | medium | mitigate | fem_basis_eval returns InvalidParameter for any point not inside a triangle before the solve |
| T-44-06 | Denial | O(N³) dense A_inv for GCV on large meshes | low | accept | v1 recommends N ≲ 2000; documented in rustdoc; sparse solvers deferred (CONTEXT deferred idea) |
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing` — all module tests green.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean.
</verification>

<success_criteria>
- `fem_smooth` solves the SR-PDE system, recovers a known smooth surface within tolerance, and approaches interpolation as λ→0.
- `fem_smooth` returns finite edf ∈ (0, n_obs] and finite GCV.
- `fem_smooth_gcv` returns a finite-GCV fit at a grid λ; `fem_predict` reproduces linear fields exactly.
- Observation outside the mesh returns FdarError with no panic.
- No API-coverage doc needed: pure in-crate numeric FEM.
</success_criteria>

<output>
Create `.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-02-SUMMARY.md` when done.
</output>
