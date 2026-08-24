//! Linear P1 finite-element surface smoothing over irregular 2D triangulated meshes.
//!
//! This module implements the **SR-PDE** (spatial regression with PDE penalisation) formulation
//! for smoothing scattered observations over an irregular 2D domain specified by a user-supplied
//! triangulated mesh (nodes + triangle connectivity).
//!
//! # Scope (v1)
//!
//! - **2D triangles only** — 3D tetrahedral FEM is out of scope.
//! - **Linear P1 Lagrange "hat" basis** — one basis function per node.
//! - **Neumann (natural, zero-flux) boundary conditions** — the standard choice for PDE surface
//!   smoothing; Dirichlet/Robin BCs are deferred.
//! - **Dense in-house assembly** — no new crate dependencies; sparse solvers are deferred.
//! - **Isotropic Laplacian roughness penalty** — anisotropic/advection-diffusion PDEs are
//!   deferred.
//!
//! # R Baseline
//!
//! Capability is matched against `fdaPDE 1.1-24`. Deliberate divergences:
//! - Dense assembly vs `fdaPDE`'s sparse-matrix assembly — identical output for modest N.
//! - No Dirichlet BC support in v1.
//! - No space-varying PDE coefficients.
//! - Point location via linear scan (O(T) per query) vs `fdaPDE`'s CGAL spatial index.
//!
//! # Public API (this wave)
//!
//! - [`assemble_fem_matrices`] — assemble global mass M and stiffness K (both N×N row-major).
//! - [`fem_basis_eval`] — evaluate P1 hat functions (barycentric coords) at query points.
//! - [`FemSmoothResult`] — result type shared with wave-2 smoothing functions.
//!
//! Wave-2 plans add `fem_smooth`, `fem_smooth_gcv`, and `fem_predict` to the same module.

use crate::error::FdarError;

// ──────────────────────────────────────────────────────────────────────────────
// Public result type
// ──────────────────────────────────────────────────────────────────────────────

/// Result of FEM/PDE-regularized surface smoothing.
///
/// Returned by `fem_smooth` and `fem_smooth_gcv` (wave-2). Defined here in the foundation
/// plan so wave-2 implementations can reference it without re-definition.
///
/// All matrices stored as row-major flat `Vec<f64>` internally; fitted values are plain
/// `Vec<f64>` of length `n_nodes` and `n_obs` respectively.
#[must_use]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FemSmoothResult {
    /// Fitted surface values at the mesh nodes (length `n_nodes`).
    ///
    /// The coefficient vector `c` solving `(Φ'Φ + λ·K) c = Φ'y`.
    pub node_values: Vec<f64>,
    /// Fitted values at the observation locations `obs_xy` (length `n_obs`).
    ///
    /// Computed as `fitted_obs[i] = Σ_k φ_k(obs_xy[i]) * c[k]`.
    pub fitted_obs: Vec<f64>,
    /// Effective degrees of freedom — trace of the hat matrix `Φ(Φ'Φ + λK)⁻¹Φ'`.
    pub edf: f64,
    /// Generalised cross-validation score.
    ///
    /// `GCV = n · RSS / (n − edf)²`. Set to `f64::INFINITY` if `edf ≥ n_obs`.
    pub gcv: f64,
    /// Residual sum of squares at the observation locations.
    pub rss: f64,
    /// Smoothing parameter λ used for this result.
    pub lambda: f64,
    /// Number of mesh nodes N.
    pub n_nodes: usize,
    /// Number of triangles T.
    pub n_triangles: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// Internal constants
// ──────────────────────────────────────────────────────────────────────────────

/// Numerical epsilon for the point-in-triangle test (barycentric tolerance).
const BARY_EPS: f64 = 1e-10;

/// Epsilon used in `barycentric` to guard against degenerate triangles at eval time.
const BARY_DET_EPS: f64 = 1e-14;

// ──────────────────────────────────────────────────────────────────────────────
// Mesh validation
// ──────────────────────────────────────────────────────────────────────────────

/// Validate the mesh: non-empty, all indices in range, no degenerate (zero-area) triangles.
///
/// Called once at entry by every public function before any computation.
fn mesh_validate(nodes: &[[f64; 2]], triangles: &[[usize; 3]]) -> Result<(), FdarError> {
    if nodes.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "nodes",
            expected: "at least one node".to_string(),
            actual: "0 nodes".to_string(),
        });
    }
    if triangles.is_empty() {
        return Err(FdarError::InvalidDimension {
            parameter: "triangles",
            expected: "at least one triangle".to_string(),
            actual: "0 triangles".to_string(),
        });
    }

    let n = nodes.len();

    // Compute bounding-box area for the degenerate-triangle tolerance.
    let x_min = nodes.iter().map(|p| p[0]).fold(f64::INFINITY, f64::min);
    let x_max = nodes.iter().map(|p| p[0]).fold(f64::NEG_INFINITY, f64::max);
    let y_min = nodes.iter().map(|p| p[1]).fold(f64::INFINITY, f64::min);
    let y_max = nodes.iter().map(|p| p[1]).fold(f64::NEG_INFINITY, f64::max);
    let bbox_area = (x_max - x_min) * (y_max - y_min);
    let area_tol = 1e-12 * bbox_area.max(1.0);

    for (tri_idx, tri) in triangles.iter().enumerate() {
        // Check vertex indices are in range.
        for &vi in tri.iter() {
            if vi >= n {
                return Err(FdarError::InvalidParameter {
                    parameter: "triangles",
                    message: format!(
                        "triangle {tri_idx} references vertex index {vi} which is out of range \
                         (mesh has {n} nodes)"
                    ),
                });
            }
        }

        // Check for degenerate triangle (area ≈ 0).
        let [v0, v1, v2] = *tri;
        let (x0, y0) = (nodes[v0][0], nodes[v0][1]);
        let (x1, y1) = (nodes[v1][0], nodes[v1][1]);
        let (x2, y2) = (nodes[v2][0], nodes[v2][1]);
        let signed_area_2 = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
        let area = 0.5 * signed_area_2.abs();
        if area < area_tol {
            return Err(FdarError::InvalidParameter {
                parameter: "triangles",
                message: format!(
                    "triangle {tri_idx} is degenerate (area ≈ {area:.2e} < tolerance \
                     {area_tol:.2e}); check for collinear or coincident nodes"
                ),
            });
        }
    }
    Ok(())
}

// ──────────────────────────────────────────────────────────────────────────────
// Element matrix closed forms (P1 linear FEM)
// ──────────────────────────────────────────────────────────────────────────────

/// Element mass matrix for a P1 triangle (3×3 local, local node ordering v0,v1,v2).
///
/// `M_e = (area / 12) * [[2,1,1],[1,2,1],[1,1,2]]`
///
/// Derivation: `∫_T λ_i λ_j dA = area/6` if `i=j`, `area/12` if `i≠j`.
#[inline]
fn element_mass(area: f64) -> [[f64; 3]; 3] {
    let a = area / 12.0;
    [
        [2.0 * a, a, a],
        [a, 2.0 * a, a],
        [a, a, 2.0 * a],
    ]
}

/// Element stiffness matrix for a P1 triangle (Laplacian weak form, 3×3 local).
///
/// `K_e[i,j] = (b_i·b_j + c_i·c_j) / (4·area)`
///
/// where `b_i`, `c_i` are the gradient coefficients of the P1 hat functions:
/// ```text
/// b0 = y1 − y2,  c0 = x2 − x1
/// b1 = y2 − y0,  c1 = x0 − x2
/// b2 = y0 − y1,  c2 = x1 − x0
/// ```
///
/// # Panics
///
/// Caller must ensure `area > 0` (guaranteed by `mesh_validate`).
#[inline]
fn element_stiffness(
    x0: f64, y0: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
    area: f64,
) -> [[f64; 3]; 3] {
    let b0 = y1 - y2;
    let c0 = x2 - x1;
    let b1 = y2 - y0;
    let c1 = x0 - x2;
    let b2 = y0 - y1;
    let c2 = x1 - x0;
    let s = 1.0 / (4.0 * area);
    [
        [s * (b0 * b0 + c0 * c0), s * (b0 * b1 + c0 * c1), s * (b0 * b2 + c0 * c2)],
        [s * (b1 * b0 + c1 * c0), s * (b1 * b1 + c1 * c1), s * (b1 * b2 + c1 * c2)],
        [s * (b2 * b0 + c2 * c0), s * (b2 * b1 + c2 * c1), s * (b2 * b2 + c2 * c2)],
    ]
}

// ──────────────────────────────────────────────────────────────────────────────
// Global assembly
// ──────────────────────────────────────────────────────────────────────────────

/// Assemble the global N×N mass matrix **M** and stiffness matrix **K** for a triangulated mesh.
///
/// Both matrices are returned as flat `Vec<f64>` in **row-major** order (element `(i, j)` at
/// index `i * N + j`). This matches the layout expected by `crate::linalg::cholesky_solve` and
/// related helpers.
///
/// # Arguments
///
/// * `nodes` — mesh nodes, each `[x, y]` (N nodes).
/// * `triangles` — triangle connectivity, each `[v0, v1, v2]` as indices into `nodes` (T
///   triangles). Triangle winding order (CW vs CCW) does not affect the result; areas are taken
///   as absolute values.
///
/// # Returns
///
/// `(M, K)` — global mass and stiffness matrices, both `Vec<f64>` of length `N * N`.
///
/// **Properties:**
/// - M is symmetric positive-definite (every node appears in at least one triangle with
///   positive area after validation).
/// - K is symmetric; each row sums to ≈ 0 (constant vector is in the null space — this is the
///   Laplacian null-space property). K is PSD (not PD) with exactly one zero eigenvalue.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] for empty `nodes` or `triangles`, and
/// [`FdarError::InvalidParameter`] for out-of-range vertex indices or degenerate
/// (zero-area) triangles.
///
/// # Example
///
/// ```rust
/// use fdars_core::fem_smoothing::assemble_fem_matrices;
/// // Unit square split into 2 triangles (4 nodes):
/// let nodes = [[0.0f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let triangles = [[0usize, 1, 2], [0, 2, 3]];
/// let (m, k) = assemble_fem_matrices(&nodes, &triangles).unwrap();
/// assert_eq!(m.len(), 16); // 4×4
/// assert_eq!(k.len(), 16);
/// ```
#[must_use]
pub fn assemble_fem_matrices(
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
) -> Result<(Vec<f64>, Vec<f64>), FdarError> {
    mesh_validate(nodes, triangles)?;

    let n = nodes.len();
    let mut m_global = vec![0.0_f64; n * n];
    let mut k_global = vec![0.0_f64; n * n];

    for tri in triangles {
        let [v0, v1, v2] = *tri;
        let (x0, y0) = (nodes[v0][0], nodes[v0][1]);
        let (x1, y1) = (nodes[v1][0], nodes[v1][1]);
        let (x2, y2) = (nodes[v2][0], nodes[v2][1]);
        // Absolute area (mesh_validate already ensured > 0).
        let area = 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
        let m_e = element_mass(area);
        let k_e = element_stiffness(x0, y0, x1, y1, x2, y2, area);
        let local = [v0, v1, v2];
        for (li, &gi) in local.iter().enumerate() {
            for (lj, &gj) in local.iter().enumerate() {
                m_global[gi * n + gj] += m_e[li][lj];
                k_global[gi * n + gj] += k_e[li][lj];
            }
        }
    }

    Ok((m_global, k_global))
}

// ──────────────────────────────────────────────────────────────────────────────
// Barycentric coordinates and point location
// ──────────────────────────────────────────────────────────────────────────────

/// Compute barycentric coordinates of `(px, py)` with respect to a triangle
/// `(x0,y0)–(x1,y1)–(x2,y2)`.
///
/// Returns `None` for degenerate triangles (`|det| < BARY_DET_EPS`).
///
/// # Formula
///
/// ```text
/// det = (x1-x0)*(y2-y0) - (x2-x0)*(y1-y0)   // = 2 * signed area
/// λ1  = ((px-x0)*(y2-y0) - (py-y0)*(x2-x0)) / det
/// λ2  = ((py-y0)*(x1-x0) - (px-x0)*(y1-y0)) / det
/// λ0  = 1 - λ1 - λ2
/// ```
#[inline]
fn barycentric(
    px: f64, py: f64,
    x0: f64, y0: f64,
    x1: f64, y1: f64,
    x2: f64, y2: f64,
) -> Option<(f64, f64, f64)> {
    let det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    if det.abs() < BARY_DET_EPS {
        return None;
    }
    let lam1 = ((px - x0) * (y2 - y0) - (py - y0) * (x2 - x0)) / det;
    let lam2 = ((py - y0) * (x1 - x0) - (px - x0) * (y1 - y0)) / det;
    let lam0 = 1.0 - lam1 - lam2;
    Some((lam0, lam1, lam2))
}

/// Locate a query point `(px, py)` in the triangulation via linear scan.
///
/// Returns the **first** triangle whose barycentric coordinates all satisfy `≥ −BARY_EPS`,
/// together with the three barycentric weights `(λ0, λ1, λ2)`.
///
/// Returns `None` if no triangle contains the point (i.e., the point is outside the mesh).
///
/// Complexity: O(T) per query (v1; spatial index deferred per CONTEXT.md).
fn locate_point(
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
    px: f64,
    py: f64,
) -> Option<(usize, (f64, f64, f64))> {
    for (tri_idx, tri) in triangles.iter().enumerate() {
        let [v0, v1, v2] = *tri;
        let (x0, y0) = (nodes[v0][0], nodes[v0][1]);
        let (x1, y1) = (nodes[v1][0], nodes[v1][1]);
        let (x2, y2) = (nodes[v2][0], nodes[v2][1]);
        if let Some((lam0, lam1, lam2)) = barycentric(px, py, x0, y0, x1, y1, x2, y2) {
            if lam0 >= -BARY_EPS && lam1 >= -BARY_EPS && lam2 >= -BARY_EPS {
                return Some((tri_idx, (lam0, lam1, lam2)));
            }
        }
    }
    None
}

// ──────────────────────────────────────────────────────────────────────────────
// Public basis evaluation
// ──────────────────────────────────────────────────────────────────────────────

/// Evaluate the P1 hat functions at a set of query points.
///
/// For each query point, locates the containing triangle via barycentric coordinates and returns
/// the three non-zero hat-function (node, value) pairs. Points outside the mesh return an error.
///
/// # Arguments
///
/// * `nodes` — mesh nodes (N × 2 coordinates).
/// * `triangles` — triangle connectivity (T × 3 vertex indices).
/// * `query_xy` — query points, each `[x, y]`.
///
/// # Returns
///
/// A `Vec` of length `query_xy.len()`, where each entry is:
/// `(containing_triangle_index, [(node_index, hat_value); 3])`.
///
/// The three hat values sum to 1.0 for any interior point (partition of unity). The hat values
/// are the barycentric coordinates `(λ0, λ1, λ2)` of the query point within the containing
/// triangle, corresponding to nodes `(v0, v1, v2)` of that triangle.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] for an empty mesh, and
/// [`FdarError::InvalidParameter`] for:
/// - out-of-range vertex indices or degenerate triangles (via `mesh_validate`).
/// - any query point that lies outside the triangulated domain (parameter `"query_xy"`).
///
/// # Example
///
/// ```rust
/// use fdars_core::fem_smoothing::fem_basis_eval;
/// let nodes = [[0.0f64, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
/// let triangles = [[0usize, 1, 2], [0, 2, 3]];
/// let result = fem_basis_eval(&nodes, &triangles, &[[0.25, 0.25]]).unwrap();
/// let (_tri_idx, weights) = result[0];
/// let sum: f64 = weights.iter().map(|(_, w)| w).sum();
/// assert!((sum - 1.0).abs() < 1e-12, "hat values must sum to 1");
/// ```
#[must_use]
pub fn fem_basis_eval(
    nodes: &[[f64; 2]],
    triangles: &[[usize; 3]],
    query_xy: &[[f64; 2]],
) -> Result<Vec<(usize, [(usize, f64); 3])>, FdarError> {
    mesh_validate(nodes, triangles)?;

    let mut result = Vec::with_capacity(query_xy.len());

    for (qi, &[px, py]) in query_xy.iter().enumerate() {
        match locate_point(nodes, triangles, px, py) {
            Some((tri_idx, (lam0, lam1, lam2))) => {
                let [v0, v1, v2] = triangles[tri_idx];
                result.push((tri_idx, [(v0, lam0), (v1, lam1), (v2, lam2)]));
            }
            None => {
                return Err(FdarError::InvalidParameter {
                    parameter: "query_xy",
                    message: format!(
                        "query point {qi} ([{px}, {py}]) lies outside the triangulated mesh"
                    ),
                });
            }
        }
    }

    Ok(result)
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Unit-square fixture ────────────────────────────────────────────────────
    //
    //   3 ──── 2
    //   |    / |
    //   |   /  |
    //   |  /   |
    //   | /    |
    //   0 ──── 1
    //
    // nodes:     [[0,0],[1,0],[1,1],[0,1]]
    // triangles: [[0,1,2],[0,2,3]]  (unit square split diagonally)
    // Each triangle has area = 0.5.

    fn unit_square_mesh() -> ([[f64; 2]; 4], [[usize; 3]; 2]) {
        let nodes = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let triangles = [[0, 1, 2], [0, 2, 3]];
        (nodes, triangles)
    }

    // ── Task 1 tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_assemble_unit_square_symmetry_and_nullspace() {
        let (nodes, triangles) = unit_square_mesh();
        let (m, k) = assemble_fem_matrices(&nodes, &triangles).unwrap();

        // Both matrices must be 4×4 (16 entries).
        assert_eq!(m.len(), 16, "M must be 4×4");
        assert_eq!(k.len(), 16, "K must be 4×4");

        let n = 4_usize;

        // Symmetry check for M and K.
        for i in 0..n {
            for j in 0..n {
                let m_ij = m[i * n + j];
                let m_ji = m[j * n + i];
                assert!(
                    (m_ij - m_ji).abs() < 1e-12,
                    "M not symmetric at ({i},{j}): {m_ij} vs {m_ji}"
                );
                let k_ij = k[i * n + j];
                let k_ji = k[j * n + i];
                assert!(
                    (k_ij - k_ji).abs() < 1e-12,
                    "K not symmetric at ({i},{j}): {k_ij} vs {k_ji}"
                );
            }
        }

        // K row-sums ≈ 0 (constant null space property of the Laplacian stiffness).
        for i in 0..n {
            let row_sum: f64 = (0..n).map(|j| k[i * n + j]).sum();
            assert!(
                row_sum.abs() < 1e-9,
                "K row {i} sum = {row_sum} (expected ≈ 0)"
            );
        }

        // M is SPD — verify by Cholesky factorisation (must return Ok).
        crate::linalg::cholesky_factor(&m, n)
            .expect("M must be positive-definite (Cholesky should succeed)");
    }

    // ── Task 2 tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_fem_basis_partition_of_unity() {
        let (nodes, triangles) = unit_square_mesh();
        // Interior point in triangle 0 ([0,1,2]).
        let query = [[0.25_f64, 0.25]];
        let result = fem_basis_eval(&nodes, &triangles, &query).unwrap();
        assert_eq!(result.len(), 1);
        let (_tri_idx, weights) = result[0];
        let sum: f64 = weights.iter().map(|(_, w)| w).sum();
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "partition of unity violated: sum = {sum}"
        );
    }

    #[test]
    fn test_fem_basis_linear_exactness() {
        let (nodes, triangles) = unit_square_mesh();

        // Linear field g(x,y) = 2.0 + 3.0*x - 1.5*y
        let g = |x: f64, y: f64| 2.0 + 3.0 * x - 1.5 * y;
        let node_values: Vec<f64> = nodes.iter().map(|&[x, y]| g(x, y)).collect();

        let px = 0.3_f64;
        let py = 0.25_f64;
        let query = [[px, py]];
        let result = fem_basis_eval(&nodes, &triangles, &query).unwrap();
        let (_tri_idx, weights) = result[0];

        // Reconstruct via P1 interpolation: sum hat_value * g_at_node.
        let interpolated: f64 = weights
            .iter()
            .map(|(node_idx, hat_val)| hat_val * node_values[*node_idx])
            .sum();
        let exact = g(px, py);

        assert!(
            (interpolated - exact).abs() < 1e-10,
            "linear exactness violated: interpolated={interpolated}, exact={exact}"
        );
    }

    // ── Task 3 tests — error paths ─────────────────────────────────────────────

    #[test]
    fn test_fem_degenerate_triangle_error() {
        // Collinear nodes: all on the x-axis → area = 0.
        let nodes = [[0.0_f64, 0.0], [1.0, 0.0], [2.0, 0.0]];
        let triangles = [[0_usize, 1, 2]];
        let result = assemble_fem_matrices(&nodes, &triangles);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { .. })),
            "degenerate triangle must return InvalidParameter, got: {result:?}"
        );
    }

    #[test]
    fn test_fem_bad_index_error() {
        // 4 nodes but triangle references index 4 (out of range).
        let (nodes, _) = unit_square_mesh();
        let triangles = [[0_usize, 1, 4]]; // index 4 >= len(nodes)=4
        let result = assemble_fem_matrices(&nodes, &triangles);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { .. })),
            "out-of-range index must return InvalidParameter, got: {result:?}"
        );
    }

    #[test]
    fn test_fem_empty_mesh_error() {
        // Empty nodes.
        let result_empty_nodes = assemble_fem_matrices(
            &[] as &[[f64; 2]],
            &[[0_usize, 1, 2]],
        );
        assert!(
            matches!(result_empty_nodes, Err(FdarError::InvalidDimension { .. })),
            "empty nodes must return InvalidDimension, got: {result_empty_nodes:?}"
        );

        // Empty triangles.
        let (nodes, _) = unit_square_mesh();
        let result_empty_tris = assemble_fem_matrices(&nodes, &[] as &[[usize; 3]]);
        assert!(
            matches!(result_empty_tris, Err(FdarError::InvalidDimension { .. })),
            "empty triangles must return InvalidDimension, got: {result_empty_tris:?}"
        );
    }

    #[test]
    fn test_fem_obs_outside_mesh_error() {
        let (nodes, triangles) = unit_square_mesh();
        // Point clearly outside the [0,1]×[0,1] unit square.
        let query = [[5.0_f64, 5.0]];
        let result = fem_basis_eval(&nodes, &triangles, &query);
        assert!(
            matches!(result, Err(FdarError::InvalidParameter { parameter: "query_xy", .. })),
            "outside-mesh point must return InvalidParameter(query_xy), got: {result:?}"
        );
    }
}
