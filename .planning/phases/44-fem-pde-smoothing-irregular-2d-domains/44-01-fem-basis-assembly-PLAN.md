---
phase: 44-fem-pde-smoothing-irregular-2d-domains
plan: 01
type: execute
wave: 1
depends_on: []
files_modified:
  - fdars-core/src/fem_smoothing.rs
  - fdars-core/src/lib.rs
  - fdars-core/src/prelude.rs
autonomous: true
requirements: [REP-02-01]
estimate:
  tokens: 62000
  raw_tokens: 34000
  tasks: 3
  confidence: med
must_haves:
  truths:
    - "assemble_fem_matrices returns symmetric M and K; K row-sums are ~0 (constant null space); M is symmetric PD on a small mesh"
    - "P1 basis at a query point inside a triangle returns barycentric weights summing to 1 (partition of unity); a linear field is interpolated exactly"
    - "Degenerate (near-zero-area) triangle, out-of-range connectivity index, and dimension mismatches return FdarError (no panic)"
  artifacts:
    - "fdars-core/src/fem_smoothing.rs (new module: mesh validation, barycentric eval, element + global M/K assembly, FemSmoothResult type)"
    - "pub mod fem_smoothing + crate-root re-export block in fdars-core/src/lib.rs"
    - "fem_smoothing key types re-exported in fdars-core/src/prelude.rs"
  key_links:
    - "row-major flat Vec<f64> layout for M and K (matches linalg.rs cholesky helpers) — NOT column-major FdMatrix"
    - "mesh_validate runs once at entry before any assembly loop; area guard prevents divide-by-zero NaN in element stiffness"
---

<objective>
Deliver the foundation of REP-02-01: a new `fdars-core/src/fem_smoothing.rs` module implementing a linear (P1) finite-element basis over a user-supplied triangulated 2D mesh — mesh validation, barycentric point evaluation, per-triangle element mass/stiffness closed forms, global N×N assembly — plus the shared `FemSmoothResult` result type and full crate-root/prelude registration. This is the wave-1 base every wave-2 FEM plan builds on.

Purpose: Establishes the mesh + basis + assembly primitives and the shared result/type surface so wave-2 (SR-PDE smoothing) can build the penalized system without re-touching these definitions.
Output: `fem_smoothing.rs` with public `assemble_fem_matrices`, `fem_basis_eval` (barycentric evaluation entry), `FemSmoothResult` struct, internal mesh/element helpers, inline `#[cfg(test)]` tests; module wired into `lib.rs` and `prelude.rs`.
</objective>

<execution_context>
@$HOME/.claude/gsd-core/workflows/execute-plan.md
@$HOME/.claude/gsd-core/templates/summary.md
</execution_context>

<context>
@.planning/PROJECT.md
@.planning/ROADMAP.md
@.planning/STATE.md
@.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-CONTEXT.md
@.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-RESEARCH.md

@fdars-core/src/error.rs
@fdars-core/src/linalg.rs
</context>

<artifacts_this_phase_produces>
New public symbols introduced by THIS plan (wave 1):
- `fem_smoothing::FemSmoothResult` (struct: node_values, fitted_obs, edf, gcv, rss, lambda, n_nodes, n_triangles)
- `fem_smoothing::assemble_fem_matrices(nodes, triangles) -> Result<(Vec<f64>, Vec<f64>), FdarError>` — global (M, K), both N×N row-major flat
- `fem_smoothing::fem_basis_eval(nodes, triangles, query_xy) -> Result<Vec<(usize, [(usize, f64); 3])>, FdarError>` — for each query point: containing-triangle index + its 3 (node_index, hat_value) pairs (partition-of-unity witness / prediction primitive)
- Crate-root re-export: `pub use fem_smoothing::{assemble_fem_matrices, fem_basis_eval, FemSmoothResult};`
- prelude re-export: `pub use crate::fem_smoothing::FemSmoothResult;`

Wave-2 plans (02) will ADD `fem_smooth`, `fem_smooth_gcv`, `fem_predict` to the same re-export block.
</artifacts_this_phase_produces>

<tasks>

<task type="tracer">
  <name>Task 1: End-to-end mesh → validate → assemble (M, K) on the unit-square fixture — one path, verified</name>
  <files>fdars-core/src/fem_smoothing.rs, fdars-core/src/lib.rs, fdars-core/src/prelude.rs</files>
  <read_first>
    - fdars-core/src/error.rs — FdarError variants. NOTE: `InvalidParameter.parameter` is `&'static str`, so pass a static like "triangles" and put the offending index in `message` (never a dynamic `format!` string in `parameter`).
    - fdars-core/src/linalg.rs:85-134 — `cholesky_factor`/`cholesky_forward_back`/`cholesky_solve` are `pub(crate)` and operate on ROW-MAJOR flat square matrices (element (i,j) at index i*p+j). This dictates the M/K layout produced here.
    - fdars-core/src/lib.rs:64-136 (module declarations, alphabetical block around `pub mod famm; pub mod fdata;`) and :294 / :379-384 (existing re-export block style) — register `pub mod fem_smoothing;` in alphabetical order (between `pub mod famm;` and `pub mod fdata;`) and add a crate-root `pub use fem_smoothing::{...};` block near the other re-exports.
    - fdars-core/src/prelude.rs:42-44 — add `pub use crate::fem_smoothing::FemSmoothResult;` alongside the existing basis-type re-exports.
  </read_first>
  <action>
Create `fdars-core/src/fem_smoothing.rs` with a module doc comment (`//!`) describing linear P1 FEM surface smoothing over irregular 2D triangulated meshes, v1 scope (2D triangles only, Neumann natural BC, dense in-house assembly, no new dependency — per the CONTEXT locked decisions), and R baseline `fdaPDE` 1.1-24 matched by capability. Note the deliberate divergences (dense vs sparse assembly, Neumann-only BC) in the module doc.

Define the shared result struct exactly once here so wave-2 does not redefine it:
`FemSmoothResult` with `#[must_use]`, `#[derive(Debug, Clone, PartialEq)]`, `#[non_exhaustive]`, `#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]`. Fields: `node_values: Vec<f64>`, `fitted_obs: Vec<f64>`, `edf: f64`, `gcv: f64`, `rss: f64`, `lambda: f64`, `n_nodes: usize`, `n_triangles: usize`. Add rustdoc on every field.

Implement mesh validation `fn mesh_validate(nodes: &[[f64; 2]], triangles: &[[usize; 3]]) -> Result<(), FdarError>`: (a) `nodes` non-empty and `triangles` non-empty else `InvalidDimension`; (b) every vertex index `< nodes.len()` else `InvalidParameter { parameter: "triangles", message }` naming the triangle index and the bad vertex index; (c) each triangle signed area magnitude `0.5*|(x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)|` must exceed an area tolerance — compute `bbox_area` from node coordinate spans and use `AREA_TOL = 1e-12 * bbox_area.max(1.0)`; degenerate triangles return `InvalidParameter { parameter: "triangles", message }` naming the index. Run validation ONCE at entry; the assembly loop must not re-check (per RESEARCH Pitfall 1).

Implement the element closed forms as inner helpers (constant-gradient P1, per RESEARCH §2):
- `fn element_mass(area: f64) -> [[f64; 3]; 3]`: `a = area/12.0`; matrix `[[2a,a,a],[a,2a,a],[a,a,2a]]`.
- `fn element_stiffness(x0,y0,x1,y1,x2,y2,area) -> [[f64;3];3]`: set `b0=y1-y2, c0=x2-x1; b1=y2-y0, c1=x0-x2; b2=y0-y1, c2=x1-x0; s=1.0/(4.0*area)`; entry `[i][j] = s*(b_i*b_j + c_i*c_j)`.

Implement `pub fn assemble_fem_matrices(nodes: &[[f64;2]], triangles: &[[usize;3]]) -> Result<(Vec<f64>, Vec<f64>), FdarError>`: call `mesh_validate`; let `N = nodes.len()`; allocate `m_global` and `k_global` as `vec![0.0; N*N]` ROW-MAJOR (index `gi*N + gj`); for each triangle compute area from absolute signed area, get `element_mass`/`element_stiffness`, and scatter the 3×3 local contributions into `m_global`/`k_global` at `[local[li]*N + local[lj]]` (per RESEARCH §2.4). Return `(m_global, k_global)`. Mark `#[must_use]`.

Register the module: add `pub mod fem_smoothing;` to `lib.rs` in alphabetical position (between `famm` and `fdata`), add a crate-root re-export block `pub use fem_smoothing::{assemble_fem_matrices, FemSmoothResult};` (fem_basis_eval added in Task 2), and add `pub use crate::fem_smoothing::FemSmoothResult;` to `prelude.rs`.

Add the unit-square test fixture as a `#[cfg(test)]` helper: nodes `[[0,0],[1,0],[1,1],[0,1]]`, triangles `[[0,1,2],[0,2,3]]` (unit square split diagonally, 2 triangles, 4 nodes). Write the tracer test `test_assemble_unit_square_symmetry_and_nullspace`: assemble (M, K); assert both are 4×4 (len 16); assert symmetry (`m[i*4+j]==m[j*4+i]`, same for k, within 1e-12); assert each K row-sum is ~0 within 1e-9 (constant null space, RESEARCH §2.3); assert M is symmetric positive-definite by factoring it with `crate::linalg::cholesky_factor(&m, 4)` returning Ok.
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing::tests::test_assemble_unit_square_symmetry_and_nullspace 2>&1 | tail -20</automated>
  </verify>
  <done>Unit-square mesh assembles to 4×4 symmetric M and K; K row-sums ~0; M passes Cholesky (SPD); module compiles and is registered in lib.rs + prelude.rs.</done>
</task>

<task type="auto">
  <name>Task 2: Barycentric point evaluation + point location + partition-of-unity / linear-exactness oracles</name>
  <files>fdars-core/src/fem_smoothing.rs, fdars-core/src/lib.rs</files>
  <read_first>
    - fem_smoothing.rs (this plan, Task 1) — mesh_validate, FemSmoothResult, the test fixture.
    - RESEARCH §1.2-1.4 — barycentric formula, point-in-triangle test (`lam >= -EPS`, EPS=1e-10), partition-of-unity oracle.
  </read_first>
  <action>
Add an inner helper `fn barycentric(px,py,x0,y0,x1,y1,x2,y2) -> Option<(f64,f64,f64)>` per RESEARCH §1.2: `det=(x1-x0)*(y2-y0)-(x2-x0)*(y1-y0)`; if `det.abs() < 1e-14` return `None`; `lam1=((px-x0)*(y2-y0)-(py-y0)*(x2-x0))/det`; `lam2=((py-y0)*(x1-x0)-(px-x0)*(y1-y0))/det`; `lam0=1.0-lam1-lam2`; return `Some((lam0,lam1,lam2))`.

Add an inner helper `fn locate_point(nodes, triangles, px, py) -> Option<(usize, (f64,f64,f64))>`: linear scan over triangles (v1 point location, RESEARCH §1.2); for each, compute barycentric coords and accept the first triangle where `lam0 >= -EPS && lam1 >= -EPS && lam2 >= -EPS` with `EPS = 1e-10`; return the triangle index and `(lam0,lam1,lam2)`; return `None` if no triangle contains the point.

Add `pub fn fem_basis_eval(nodes: &[[f64;2]], triangles: &[[usize;3]], query_xy: &[[f64;2]]) -> Result<Vec<(usize, [(usize, f64); 3])>, FdarError>`: call `mesh_validate`; for each query point, call `locate_point`; on hit return `(tri_idx, [(v0, lam0), (v1, lam1), (v2, lam2)])`; on miss return `InvalidParameter { parameter: "query_xy", message }` naming the point index and stating it lies outside the mesh (RESEARCH Pitfall 2). Mark `#[must_use]`. Add rustdoc explaining the return shape is the containing-triangle index plus the three nonzero P1 hat (node, value) pairs, and that hat values sum to 1 for interior points.

Extend the crate-root re-export block in lib.rs to include `fem_basis_eval` (final block: `pub use fem_smoothing::{assemble_fem_matrices, fem_basis_eval, FemSmoothResult};`).

Add tests:
- `test_fem_basis_partition_of_unity`: on the unit-square fixture, eval at an interior point e.g. `[0.25, 0.25]`; assert the three returned hat values sum to 1.0 within 1e-12.
- `test_fem_basis_linear_exactness`: define node values from a linear field `g(x,y)=2.0 + 3.0*x - 1.5*y` at each fixture node; for an interior query point, reconstruct `sum(hat_value * g_at_node)` and assert it equals `g(query)` within 1e-10 (linear interpolation exactness, RESEARCH §1.4).
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing::tests::test_fem_basis 2>&1 | tail -20</automated>
  </verify>
  <done>fem_basis_eval returns containing-triangle + 3 hat weights; partition-of-unity (Σ=1) and linear-field exactness tests pass; fem_basis_eval re-exported at crate root.</done>
</task>

<task type="auto">
  <name>Task 3: Mesh + assembly error paths (degenerate triangle, bad index, dimension mismatch, point outside)</name>
  <files>fdars-core/src/fem_smoothing.rs</files>
  <read_first>
    - fem_smoothing.rs (this plan, Tasks 1-2) — mesh_validate, assemble_fem_matrices, fem_basis_eval.
    - RESEARCH §Common Pitfalls 1 & 2; error.rs FdarError variants.
  </read_first>
  <action>
Add error-path tests exercising the guards (no panics — all return FdarError):
- `test_fem_degenerate_triangle_error`: mesh with a collinear/zero-area triangle (e.g. nodes `[[0,0],[1,0],[2,0]]`, triangle `[0,1,2]`); assert `assemble_fem_matrices` returns `Err(FdarError::InvalidParameter { .. })`.
- `test_fem_bad_index_error`: triangle references a vertex index `>= nodes.len()` (e.g. 4 nodes, triangle `[0,1,4]`); assert `assemble_fem_matrices` returns `Err(FdarError::InvalidParameter { .. })`.
- `test_fem_empty_mesh_error`: empty nodes or empty triangles → assert `Err(FdarError::InvalidDimension { .. })`.
- `test_fem_obs_outside_mesh_error`: on the valid unit-square fixture, call `fem_basis_eval` with a query point clearly outside, e.g. `[5.0, 5.0]`; assert `Err(FdarError::InvalidParameter { .. })` (parameter "query_xy").

Each test matches on the specific FdarError variant. Assert via `matches!(result, Err(FdarError::InvalidParameter { .. }))` etc. Do NOT assert on the `message` text (avoid coupling tests to prose).
  </action>
  <verify>
    <automated>TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing::tests 2>&1 | tail -25</automated>
  </verify>
  <done>All four error-path tests pass; degenerate triangle, out-of-range index, empty mesh, and outside-point inputs each return the expected FdarError variant with no panic. Full `fem_smoothing::tests` module green.</done>
</task>

</tasks>

<threat_model>
## Trust Boundaries

| Boundary | Description |
|----------|-------------|
| caller → assemble_fem_matrices / fem_basis_eval | Untrusted numeric mesh input (node coords, connectivity indices) crosses into FE assembly |

Attack surface: none in the security sense — pure in-process numeric library, no I/O, network, untrusted deserialization, or auth. Only numerical-correctness concerns, handled as FdarError guards.

## STRIDE Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation Plan |
|-----------|----------|-----------|----------|-------------|-----------------|
| T-44-01 | Tampering | connectivity index (`triangles[i][k] >= N`) | medium | mitigate | mesh_validate checks every vertex index against nodes.len() at entry → InvalidParameter |
| T-44-02 | Elevation (incorrect output) | degenerate zero-area triangle → NaN in element_stiffness (÷4·area) | medium | mitigate | mesh_validate rejects area < AREA_TOL before assembly; guard is at entry, not in the hot loop |
| T-44-03 | Denial | point-location linear scan O(n_obs·T) on large meshes | low | accept | v1 dense/linear-scan scope documented; spatial index deferred (CONTEXT deferred idea) |
</threat_model>

<verification>
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing` — all module tests green.
- `TMPDIR=/home/simonm/.cache/fdars-bench-tmp cargo clippy --all-targets --features linalg,parallel -- -D warnings` — clean (CI lints test/bench code; a plain `-p ... -D warnings` false-greens per MEMORY.md).
</verification>

<success_criteria>
- `fem_smoothing.rs` exists, compiles, and is registered in `lib.rs` (`pub mod` + crate-root re-export) and `prelude.rs`.
- `assemble_fem_matrices` produces symmetric M and K; K row-sums ~0; M is SPD (Cholesky ok) on the unit-square fixture.
- `fem_basis_eval` satisfies partition-of-unity and linear-field exactness.
- Degenerate triangle, out-of-range index, empty mesh, and outside-point inputs return the correct FdarError variant with no panic.
- No API-coverage doc needed: no external API integration — pure in-crate numeric FEM.
</success_criteria>

<output>
Create `.planning/phases/44-fem-pde-smoothing-irregular-2d-domains/44-01-SUMMARY.md` when done.
</output>
