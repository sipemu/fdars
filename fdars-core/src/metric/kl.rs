//! Kullback-Leibler divergence for functional data.
//!
//! Each curve is treated as a density: it is normalized so that it integrates to 1
//! (trapezoidal rule), a small epsilon is added to avoid log(0), and then
//! re-normalized.  The symmetric KL divergence between two densities p and q is
//! computed as (KL(p||q) + KL(q||p)) / 2.

use crate::matrix::FdMatrix;

use super::{cross_distance_matrix, self_distance_matrix};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Trapezoidal integration weights for a 1-D grid.
fn trapezoidal_weights(argvals: &[f64]) -> Vec<f64> {
    let m = argvals.len();
    if m < 2 {
        return vec![0.0; m];
    }
    let mut w = vec![0.0; m];
    for k in 0..m - 1 {
        let h = argvals[k + 1] - argvals[k];
        w[k] += 0.5 * h;
        w[k + 1] += 0.5 * h;
    }
    w
}

/// Normalize a curve so that it integrates to 1 (trapezoidal rule), add epsilon,
/// and re-normalize.  Returns the density values at each grid point.
fn normalize_density(values: &[f64], weights: &[f64], epsilon: f64) -> Vec<f64> {
    let m = values.len();
    // Make values non-negative (shift if necessary) then integrate.
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let shift = if min_val < 0.0 { -min_val } else { 0.0 };

    let mut density: Vec<f64> = values.iter().map(|&v| v + shift).collect();

    // First integral (trapezoidal)
    let integral: f64 = density
        .iter()
        .zip(weights.iter())
        .map(|(&d, &w)| d * w)
        .sum();
    if integral > 0.0 {
        for d in &mut density {
            *d /= integral;
        }
    } else {
        // Flat curve — use uniform density
        let total_w: f64 = weights.iter().sum();
        let uniform = if total_w > 0.0 {
            1.0 / total_w
        } else {
            1.0 / m as f64
        };
        density.fill(uniform);
    }

    // Add epsilon and re-normalize
    for d in &mut density {
        *d += epsilon;
    }
    let integral2: f64 = density
        .iter()
        .zip(weights.iter())
        .map(|(&d, &w)| d * w)
        .sum();
    if integral2 > 0.0 {
        for d in &mut density {
            *d /= integral2;
        }
    }

    density
}

/// KL(p || q) = integral p(t) * log(p(t)/q(t)) dt  (trapezoidal rule).
fn kl_divergence(p: &[f64], q: &[f64], weights: &[f64]) -> f64 {
    p.iter()
        .zip(q.iter())
        .zip(weights.iter())
        .map(|((&pi, &qi), &w)| {
            if pi > 0.0 && qi > 0.0 {
                pi * (pi / qi).ln() * w
            } else {
                0.0
            }
        })
        .sum()
}

/// Symmetric KL divergence = (KL(p||q) + KL(q||p)) / 2.
fn symmetric_kl(p: &[f64], q: &[f64], weights: &[f64]) -> f64 {
    (kl_divergence(p, q, weights) + kl_divergence(q, p, weights)) * 0.5
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Compute a symmetric n x n KL divergence matrix for a set of functional curves.
///
/// Each row of `data` is treated as a density: it is normalized to integrate to 1
/// over `argvals` using the trapezoidal rule.  A small `epsilon` is added before
/// normalization to prevent log(0).
///
/// # Arguments
/// * `data`    - Functional data matrix (n rows x m columns)
/// * `argvals` - Evaluation grid of length m
/// * `epsilon` - Small constant added to avoid log(0), e.g. 1e-10
///
/// # Returns
/// Symmetric n x n distance matrix where entry (i,j) is the symmetric KL
/// divergence between curves i and j.
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::kl_self_1d;
///
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let data = FdMatrix::from_column_major(
///     (0..30).map(|i| ((i as f64 * 0.1).sin()).abs() + 0.01).collect(),
///     3, 10,
/// ).unwrap();
/// let dist = kl_self_1d(&data, &argvals, 1e-10);
/// assert_eq!(dist.shape(), (3, 3));
/// assert!(dist[(0, 0)].abs() < 1e-10);
/// assert!((dist[(0, 1)] - dist[(1, 0)]).abs() < 1e-10);
/// ```
#[must_use]
pub fn kl_self_1d(data: &FdMatrix, argvals: &[f64], epsilon: f64) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();

    if n == 0 || m < 2 || argvals.len() != m {
        return FdMatrix::zeros(0, 0);
    }

    let weights = trapezoidal_weights(argvals);

    // Pre-compute normalized densities for all curves.
    let densities: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let row = data.row(i);
            normalize_density(&row, &weights, epsilon)
        })
        .collect();

    self_distance_matrix(n, |i, j| {
        symmetric_kl(&densities[i], &densities[j], &weights)
    })
}

/// Compute an n1 x n2 KL divergence matrix between two sets of functional curves.
///
/// Each row of `data1` and `data2` is treated as a density (see [`kl_self_1d`]).
///
/// # Arguments
/// * `data1`   - First dataset matrix  (n1 rows x m columns)
/// * `data2`   - Second dataset matrix (n2 rows x m columns)
/// * `argvals` - Evaluation grid of length m
/// * `epsilon` - Small constant added to avoid log(0), e.g. 1e-10
///
/// # Returns
/// Distance matrix (n1 rows x n2 columns).
///
/// # Examples
///
/// ```
/// use fdars_core::matrix::FdMatrix;
/// use fdars_core::metric::kl_cross_1d;
///
/// let argvals: Vec<f64> = (0..10).map(|i| i as f64 / 9.0).collect();
/// let data1 = FdMatrix::from_column_major(
///     (0..30).map(|i| ((i as f64 * 0.1).sin()).abs() + 0.01).collect(), 3, 10,
/// ).unwrap();
/// let data2 = FdMatrix::from_column_major(
///     (0..20).map(|i| ((i as f64 * 0.2).cos()).abs() + 0.01).collect(), 2, 10,
/// ).unwrap();
/// let dist = kl_cross_1d(&data1, &data2, &argvals, 1e-10);
/// assert_eq!(dist.shape(), (3, 2));
/// assert!(dist[(0, 0)] >= 0.0);
/// ```
#[must_use]
pub fn kl_cross_1d(data1: &FdMatrix, data2: &FdMatrix, argvals: &[f64], epsilon: f64) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m = data1.ncols();

    if n1 == 0 || n2 == 0 || m < 2 || argvals.len() != m || data2.ncols() != m {
        return FdMatrix::zeros(0, 0);
    }

    let weights = trapezoidal_weights(argvals);

    let densities1: Vec<Vec<f64>> = (0..n1)
        .map(|i| {
            let row = data1.row(i);
            normalize_density(&row, &weights, epsilon)
        })
        .collect();
    let densities2: Vec<Vec<f64>> = (0..n2)
        .map(|i| {
            let row = data2.row(i);
            normalize_density(&row, &weights, epsilon)
        })
        .collect();

    cross_distance_matrix(n1, n2, |i, j| {
        symmetric_kl(&densities1[i], &densities2[j], &weights)
    })
}
