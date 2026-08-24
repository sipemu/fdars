---
phase: 44-fem-pde-smoothing-irregular-2d-domains
plan: "01"
subsystem: fem_smoothing
tags: [fem, mesh, p1-basis, mass-stiffness-assembly, REP-02-01]
requirements: [REP-02-01]
status: complete

dependency_graph:
  requires: []
  provides:
    - "assemble_fem_matrices(nodes, triangles) -> Result<(Vec<f64>, Vec<f64>)> — global mass M + stiffness K, row-major N×N"
    - "fem_basis_eval(nodes, triangles, query_xy) -> Result<Vec<(usize, [(usize,f64);3])>> — per point: (triangle idx, 3×(node idx, hat value))"
    - "FemSmoothResult (node_values, fitted_obs, edf, gcv, rss, lambda, n_nodes, n_triangles)"
    - "mesh_validate / locate_point / barycentric inner helpers"
  affects:
    - fdars-core/src/fem_smoothing.rs
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

tech_stack:
  added: []
  patterns:
    - "Mesh as nodes: &[[f64;2]] + triangles: &[[usize;3]]"
    - "Linear P1 hat basis; barycentric coordinates + point-in-triangle location (linear scan)"
    - "Element stiffness K_e[i,j]=(b_i·b_j + c_i·c_j)/(4·area); element mass (area/12)·[[2,1,1],[1,2,1],[1,1,2]]"
    - "Global M/K assembled as ROW-MAJOR flat Vec<f64> (N×N) for the pub(crate) cholesky_* helpers"

key_files:
  created:
    - fdars-core/src/fem_smoothing.rs
  modified:
    - fdars-core/src/lib.rs
    - fdars-core/src/prelude.rs

decisions:
  - "P1 Lagrange hat basis; Neumann natural BC; area-based element mass/stiffness closed forms (no quadrature) per RESEARCH §2."
  - "Global matrices row-major flat Vec (not FdMatrix) so they feed cholesky_solve directly in wave-2."
  - "Mesh validation: connectivity indices in range + strictly-positive triangle area → FdarError; points outside mesh → FdarError from fem_basis_eval/locate_point."

verification:
  module_tests: "7/7 pass — cargo test -p fdars-core --features linalg,parallel --lib fem_smoothing"
  tests:
    - test_fem_basis_partition_of_unity (Σ hat values = 1 at interior points)
    - test_fem_basis_linear_exactness (P1 reproduces a linear field exactly)
    - test_assemble_unit_square_symmetry_and_nullspace (M,K symmetric; K row-sums≈0 / constant null space)
    - test_fem_bad_index_error (out-of-range connectivity → FdarError)
    - test_fem_obs_outside_mesh_error (point outside mesh → FdarError)
    - test_fem_empty_mesh_error
    - test_fem_degenerate_triangle_error (zero-area triangle → FdarError)

notes:
  - "Executor subagent completed all code (Task 1 + a #[must_use] dedup fix, module registered, 7 tests green) but died on 'Connection closed mid-response' before writing this SUMMARY; SUMMARY authored inline by the orchestrator against the committed, green code."
  - "Full crate-wide clippy + fmt + test gate runs at phase end (out-of-band)."

commits:
  - "f6cd2562 feat(44-01): Task 1 — mesh validation + element matrices + global M/K assembly + tracer test"
  - "21f110ab fix(44-01): remove double #[must_use] from assemble_fem_matrices + fem_basis_eval"
---

# Plan 44-01 — FEM basis + mass/stiffness assembly (REP-02-01)

Created `fdars-core/src/fem_smoothing.rs`: mesh representation + validation, linear P1 hat basis
with barycentric evaluation and point location, per-triangle element mass/stiffness closed forms
assembled into global row-major N×N matrices, and the shared `FemSmoothResult` type. Registered
in `src/lib.rs` (`pub mod` + `pub use fem_smoothing::{assemble_fem_matrices, fem_basis_eval,
FemSmoothResult}`) and `src/prelude.rs`.

**Requirement REP-02-01** (linear FE basis over a triangulated 2D mesh with basis evaluation +
mass/stiffness assembly) is satisfied. This plan is the foundation for wave-2 (`fem_smooth`,
`fem_smooth_gcv`, `fem_predict`).

## Verification

7/7 module tests pass (partition-of-unity, linear-field exactness, matrix symmetry + stiffness
null space, and four error paths). Crate-wide gate runs at phase end.
