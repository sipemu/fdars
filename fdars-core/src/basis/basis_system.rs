//! `BasisSystem` — a basis evaluation matrix bundled with its roughness penalty.
//!
//! This struct is the return type for the new basis factories introduced in Phase 35
//! (`monomial_basis`, `exponential_basis`, `power_basis`, `polygonal_basis`). It bundles:
//!
//! - a column-major evaluation matrix over a supplied set of evaluation points, and
//! - the corresponding roughness-penalty (Gram) matrix for the chosen derivative order.
//!
//! ## Column-major layout
//!
//! `eval_matrix` is stored in **column-major order**: element (t_i, j) — the value of the
//! j-th basis function at the i-th evaluation point — is at index `i + j * n_eval`.
//!
//! This matches the `bspline_basis` / `fourier_basis` convention used throughout
//! `fdars-core`.
//!
//! ## Penalty matrix
//!
//! `penalty_matrix` is the `nbasis × nbasis` Gram matrix of the `lfd_order`-th derivatives
//! of the basis functions:
//!
//! ```text
//! R[j, k] = ∫ D^lfd_order B_j(t) · D^lfd_order B_k(t) dt
//! ```
//!
//! It is stored column-major: element (j, k) is at index `j + k * nbasis`.
//! The matrix is symmetric and positive-semi-definite.

/// A basis evaluation matrix bundled with its roughness-penalty (Gram) matrix.
///
/// Created by the basis factories (`monomial_basis`, etc.) — never constructed directly
/// by callers. The `#[non_exhaustive]` attribute allows adding new fields (e.g., `domain`,
/// `basis_type`) in future versions without breaking existing code.
///
/// ## Field layout
///
/// - `eval_matrix`: column-major, shape `(n_eval × nbasis)`.
///   Element (t_i, j) is at index `i + j * n_eval`.
/// - `penalty_matrix`: column-major, shape `(nbasis × nbasis)`.
///   Element (j, k) is at index `j + k * nbasis`.
///
/// ## Example
///
/// ```
/// use fdars_core::monomial_basis;
///
/// let t = vec![0.0, 0.5, 1.0];
/// let bs = monomial_basis(&t, 3).unwrap();
/// assert_eq!(bs.nbasis, 3);
/// assert_eq!(bs.n_eval, 3);
/// assert_eq!(bs.eval_matrix.len(), 9);
/// assert_eq!(bs.penalty_matrix.len(), 9);
/// ```
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BasisSystem {
    /// Column-major evaluation matrix, shape `(n_eval × nbasis)`.
    /// Element (t_i, j) is at index `i + j * n_eval`.
    pub eval_matrix: Vec<f64>,
    /// Column-major roughness-penalty (Gram) matrix, shape `(nbasis × nbasis)`.
    /// `R[j,k] = ∫ D^lfd_order B_j(t) · D^lfd_order B_k(t) dt`.
    /// Symmetric and positive-semi-definite. Element (j, k) at index `j + k * nbasis`.
    pub penalty_matrix: Vec<f64>,
    /// Number of basis functions (`nbasis`).
    pub nbasis: usize,
    /// Number of evaluation points, i.e. `argvals.len()`.
    pub n_eval: usize,
    /// Derivative order used to compute `penalty_matrix` (roughness order).
    pub lfd_order: usize,
}
