//! Amplitude + phase decomposed functional depth in the elastic metric.
//!
//! Combines amplitude and phase distance matrices to produce depth scores
//! that reflect how central a curve is within a dataset, after factoring
//! out phase variation.

use super::pairwise::{amplitude_self_distance_matrix, phase_self_distance_matrix};
use crate::error::FdarError;
use crate::matrix::FdMatrix;

/// Result of elastic depth computation.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ElasticDepthResult {
    /// Amplitude depth for each curve (n values in \[0, 1\]).
    pub amplitude_depth: Vec<f64>,
    /// Phase depth for each curve (n values in \[0, 1\]).
    pub phase_depth: Vec<f64>,
    /// Combined depth: geometric mean of amplitude and phase depths (n values).
    pub combined_depth: Vec<f64>,
    /// Amplitude distance matrix (n x n).
    pub amplitude_distances: FdMatrix,
    /// Phase distance matrix (n x n).
    pub phase_distances: FdMatrix,
}

/// Compute amplitude + phase decomposed elastic depth.
///
/// For each curve, the depth is defined as an inverse-average-distance
/// formulation (lens depth): `depth_i = 1 / (1 + mean_dist_to_others)`.
/// The combined depth is the geometric mean of amplitude and phase depths.
///
/// # Arguments
/// * `data` — Functional data matrix (n x m)
/// * `argvals` — Evaluation points (length m)
/// * `lambda` — Roughness penalty for elastic alignment (0.0 = no penalty)
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`
/// or `n < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn elastic_depth(
    data: &FdMatrix,
    argvals: &[f64],
    lambda: f64,
) -> Result<ElasticDepthResult, FdarError> {
    let (n, m) = data.shape();

    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 2 rows".to_string(),
            actual: format!("{n} rows"),
        });
    }

    // Step 1 & 2: Compute distance matrices.
    let amplitude_distances = amplitude_self_distance_matrix(data, argvals, lambda);
    let phase_distances = phase_self_distance_matrix(data, argvals, lambda);

    // Step 3: Convert distances to depths.
    let amplitude_depth = distance_matrix_to_depth(&amplitude_distances, n);
    let phase_depth = distance_matrix_to_depth(&phase_distances, n);

    // Step 4: Combined depth = geometric mean.
    let combined_depth: Vec<f64> = amplitude_depth
        .iter()
        .zip(phase_depth.iter())
        .map(|(&a, &p)| (a * p).sqrt())
        .collect();

    Ok(ElasticDepthResult {
        amplitude_depth,
        phase_depth,
        combined_depth,
        amplitude_distances,
        phase_distances,
    })
}

/// Convert a symmetric distance matrix to depth values using inverse average distance.
fn distance_matrix_to_depth(dist: &FdMatrix, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let sum: f64 = (0..n).filter(|&j| j != i).map(|j| dist[(i, j)]).sum();
            let mean_dist = sum / (n - 1) as f64;
            1.0 / (1.0 + mean_dist)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    fn make_sine_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let mut data_vec = vec![0.0; n * m];
        for i in 0..n {
            let phase = 0.05 * i as f64;
            for j in 0..m {
                data_vec[i + j * n] = ((t[j] + phase) * 4.0).sin();
            }
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        (data, t)
    }

    #[test]
    fn elastic_depth_dimensions() {
        let (data, t) = make_sine_data(4, 20);
        let result = elastic_depth(&data, &t, 0.0).unwrap();
        assert_eq!(result.amplitude_depth.len(), 4);
        assert_eq!(result.phase_depth.len(), 4);
        assert_eq!(result.combined_depth.len(), 4);
        assert_eq!(result.amplitude_distances.shape(), (4, 4));
        assert_eq!(result.phase_distances.shape(), (4, 4));
    }

    #[test]
    fn elastic_depth_nonneg() {
        let (data, t) = make_sine_data(4, 20);
        let result = elastic_depth(&data, &t, 0.0).unwrap();
        for &d in &result.amplitude_depth {
            assert!((0.0..=1.0).contains(&d), "amplitude depth {d} out of [0,1]");
        }
        for &d in &result.phase_depth {
            assert!((0.0..=1.0).contains(&d), "phase depth {d} out of [0,1]");
        }
        for &d in &result.combined_depth {
            assert!((0.0..=1.0).contains(&d), "combined depth {d} out of [0,1]");
        }
    }

    #[test]
    fn elastic_depth_identical_curves_highest() {
        // Build data: 3 identical curves + 1 different curve.
        let m = 20;
        let t = uniform_grid(m);
        let n = 4;
        let mut data_vec = vec![0.0; n * m];
        for j in 0..m {
            let v = (t[j] * 4.0).sin();
            // Curves 0, 1, 2 are identical.
            data_vec[j * n] = v;
            data_vec[1 + j * n] = v;
            data_vec[2 + j * n] = v;
            // Curve 3 is very different.
            data_vec[3 + j * n] = (t[j] * 12.0).cos() * 3.0;
        }
        let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
        let result = elastic_depth(&data, &t, 0.0).unwrap();

        // Identical curves should have higher depth than the outlier.
        let min_identical = result.combined_depth[0]
            .min(result.combined_depth[1])
            .min(result.combined_depth[2]);
        assert!(
            min_identical > result.combined_depth[3],
            "identical curves depth ({min_identical}) should exceed outlier depth ({})",
            result.combined_depth[3]
        );
    }
}
