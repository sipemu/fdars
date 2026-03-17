//! Horizontal Functional Principal Nested Spheres (FPNS).
//!
//! Performs principal nested spheres analysis on warping functions
//! after elastic alignment. This provides a hierarchical decomposition
//! of phase variability by sequentially projecting onto great subspheres.

use crate::elastic_fpca::{sphere_karcher_mean, warps_to_normalized_psi};
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::warping::{exp_map_sphere, inner_product_l2, inv_exp_map_sphere, l2_norm_l2};

use super::KarcherMeanResult;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Result of horizontal Functional Principal Nested Spheres (FPNS) analysis.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct FpnsResult {
    /// Principal direction components (ncomp x m_psi).
    pub components: FdMatrix,
    /// Scores for each observation and component (n x ncomp).
    pub scores: FdMatrix,
    /// Explained variance for each component.
    pub explained_variance: Vec<f64>,
    /// Karcher mean on the subsphere at each level (ncomp vectors of length m_psi).
    pub subsphere_means: Vec<Vec<f64>>,
}

// ─── Power Iteration ────────────────────────────────────────────────────────

/// Find the top right singular vector of an (n x p) matrix via power iteration.
///
/// Computes the top eigenvector of V^T V without forming the full matrix.
/// Returns the unit vector (length p).
fn top_singular_vector(mat: &FdMatrix, n: usize, p: usize) -> Vec<f64> {
    // Initialize with the first row (or a constant if degenerate)
    let mut u: Vec<f64> = if n > 0 { mat.row(0) } else { vec![1.0; p] };

    // Normalize initial vector
    let norm = u.iter().map(|&v| v * v).sum::<f64>().sqrt().max(1e-15);
    for v in &mut u {
        *v /= norm;
    }

    // Power iteration: u <- V^T (V u) / ||...||
    for _ in 0..200 {
        // Compute w = V u (length n)
        let mut w = vec![0.0; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..p {
                s += mat[(i, j)] * u[j];
            }
            w[i] = s;
        }

        // Compute u_new = V^T w (length p)
        let mut u_new = vec![0.0; p];
        for j in 0..p {
            let mut s = 0.0;
            for i in 0..n {
                s += mat[(i, j)] * w[i];
            }
            u_new[j] = s;
        }

        // Normalize
        let new_norm = u_new.iter().map(|&v| v * v).sum::<f64>().sqrt();
        if new_norm < 1e-15 {
            break;
        }
        for v in &mut u_new {
            *v /= new_norm;
        }

        // Check convergence: |1 - |u . u_new|| < tol
        let dot: f64 = u.iter().zip(u_new.iter()).map(|(&a, &b)| a * b).sum();
        u = u_new;
        if (1.0 - dot.abs()) < 1e-10 {
            break;
        }
    }

    u
}

// ─── Horizontal FPNS ────────────────────────────────────────────────────────

/// Perform horizontal Functional Principal Nested Spheres (FPNS) analysis.
///
/// Decomposes phase variability into nested principal directions on the
/// Hilbert sphere. Each component captures the dominant mode of variation
/// on the current subsphere, after which data is projected to a lower-dimensional
/// subsphere for subsequent components.
///
/// # Arguments
/// * `karcher` — Pre-computed Karcher mean result (with gammas)
/// * `argvals` — Evaluation points (length m)
/// * `ncomp` — Number of principal nested sphere components to extract
///
/// # Errors
/// Returns `FdarError` if inputs are invalid.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn horiz_fpns(
    karcher: &KarcherMeanResult,
    argvals: &[f64],
    ncomp: usize,
) -> Result<FpnsResult, FdarError> {
    let (n, m) = karcher.gammas.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "gammas/argvals/ncomp",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={n}, m={m}, ncomp={ncomp}, argvals.len()={}",
                argvals.len()
            ),
        });
    }
    let ncomp = ncomp.min(n - 1).min(m);

    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Step 1: Convert warps to psi on Hilbert sphere
    let psis = warps_to_normalized_psi(&karcher.gammas, argvals);

    // Step 2: Compute Karcher mean on sphere
    let mut mu = sphere_karcher_mean(&psis, &time, 50);

    // Working copy of psis that will be projected at each level
    let mut current_psis = psis;

    let mut components = FdMatrix::zeros(ncomp, m);
    let mut scores = FdMatrix::zeros(n, ncomp);
    let mut explained_variance = Vec::with_capacity(ncomp);
    let mut subsphere_means = Vec::with_capacity(ncomp);

    for k in 0..ncomp {
        // Step 3: Compute shooting vectors from current mean
        let mut shooting = FdMatrix::zeros(n, m);
        for i in 0..n {
            let v = inv_exp_map_sphere(&mu, &current_psis[i], &time);
            for j in 0..m {
                shooting[(i, j)] = v[j];
            }
        }

        // Step 4a: Find principal direction via power iteration on shooting vectors
        let e_k = top_singular_vector(&shooting, n, m);

        // Store component
        for j in 0..m {
            components[(k, j)] = e_k[j];
        }

        // Step 4d: Compute scores: score_ik = <v_i, e_k>_L2
        let mut score_vec = vec![0.0; n];
        for i in 0..n {
            let v_i: Vec<f64> = (0..m).map(|j| shooting[(i, j)]).collect();
            let s = inner_product_l2(&v_i, &e_k, &time);
            scores[(i, k)] = s;
            score_vec[i] = s;
        }

        // Step 4e: Explained variance
        let var_k = score_vec.iter().map(|s| s * s).sum::<f64>() / (n - 1) as f64;
        explained_variance.push(var_k);

        // Store subsphere mean
        subsphere_means.push(mu.clone());

        // Step 4f-j: Project onto subsphere (remove component along e_k)
        // Only needed if there are more components to extract
        if k + 1 < ncomp {
            // For each psi_i, compute the perpendicular shooting vector
            // then map back to sphere for next iteration
            let mut new_psis = Vec::with_capacity(n);
            for i in 0..n {
                let v_i: Vec<f64> = (0..m).map(|j| shooting[(i, j)]).collect();
                // Remove component along e_k: v_perp = v_i - <v_i, e_k> * e_k
                let s = score_vec[i];
                let v_perp: Vec<f64> = v_i
                    .iter()
                    .zip(e_k.iter())
                    .map(|(&v, &e)| v - s * e)
                    .collect();
                // Map back to sphere from current mean
                let psi_new = exp_map_sphere(&mu, &v_perp, &time);
                new_psis.push(psi_new);
            }

            // Compute new Karcher mean on the subsphere
            mu = sphere_karcher_mean(&new_psis, &time, 50);
            // Normalize mu to unit sphere
            let mu_norm = l2_norm_l2(&mu, &time);
            if mu_norm > 1e-10 {
                for v in &mut mu {
                    *v /= mu_norm;
                }
            }

            current_psis = new_psis;
        }
    }

    Ok(FpnsResult {
        components,
        scores,
        explained_variance,
        subsphere_means,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::karcher_mean;
    use crate::matrix::FdMatrix;
    use std::f64::consts::PI;

    fn generate_test_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let shift = 0.1 * (i as f64 - n as f64 / 2.0);
            let scale = 1.0 + 0.2 * (i as f64 / n as f64);
            for j in 0..m {
                data[(i, j)] = scale * (2.0 * PI * (t[j] + shift)).sin();
            }
        }
        (data, t)
    }

    #[test]
    fn fpns_basic_dimensions() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let ncomp = 3;
        let result = horiz_fpns(&km, &t, ncomp).expect("horiz_fpns should succeed");

        assert_eq!(result.scores.shape(), (15, ncomp));
        assert_eq!(result.components.shape(), (ncomp, 51));
        assert_eq!(result.explained_variance.len(), ncomp);
        assert_eq!(result.subsphere_means.len(), ncomp);
        for mean in &result.subsphere_means {
            assert_eq!(mean.len(), 51);
        }
    }

    #[test]
    fn fpns_variance_decreasing() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let result = horiz_fpns(&km, &t, 3).expect("horiz_fpns should succeed");

        // Explained variances should be non-negative
        for ev in &result.explained_variance {
            assert!(
                *ev >= -1e-10,
                "Explained variance should be non-negative: {ev}"
            );
        }

        // Variance should be approximately decreasing
        // (Note: FPNS doesn't strictly guarantee this like PCA, but for well-behaved
        // data the first component should capture the most variance.)
        // We use a loose check here.
        if result.explained_variance.len() >= 2 {
            assert!(
                result.explained_variance[0] >= result.explained_variance[1] * 0.5,
                "First component should capture substantial variance: {} vs {}",
                result.explained_variance[0],
                result.explained_variance[1]
            );
        }
    }

    #[test]
    fn fpns_subsphere_means_on_sphere() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let result = horiz_fpns(&km, &t, 3).expect("horiz_fpns should succeed");

        let time: Vec<f64> = (0..51).map(|i| i as f64 / 50.0).collect();

        for (k, mean) in result.subsphere_means.iter().enumerate() {
            let norm = l2_norm_l2(mean, &time);
            assert!(
                (norm - 1.0).abs() < 0.1,
                "Subsphere mean {k} should have approximately unit L2 norm, got {norm}"
            );
        }
    }

    #[test]
    fn fpns_ncomp_one_smoke() {
        let (data, t) = generate_test_data(15, 51);
        let km = karcher_mean(&data, &t, 10, 1e-4, 0.0);
        let result = horiz_fpns(&km, &t, 1).expect("horiz_fpns should succeed with ncomp=1");

        assert_eq!(result.scores.shape(), (15, 1));
        assert_eq!(result.components.shape(), (1, 51));
        assert_eq!(result.explained_variance.len(), 1);
        assert_eq!(result.subsphere_means.len(), 1);
        assert!(result.explained_variance[0] >= 0.0);
    }
}
