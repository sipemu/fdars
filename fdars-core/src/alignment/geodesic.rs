//! Geodesic interpolation between curves in the elastic metric.

use super::dp_alignment_core;
use super::nd::{elastic_align_pair_nd, srsf_transform_nd};
use super::srsf::{reparameterize_curve, srsf_inverse, srsf_single};
use crate::error::FdarError;
use crate::helpers::{l2_distance, simpsons_weights};
use crate::matrix::{FdCurveSet, FdMatrix};
use crate::warping::{exp_map_sphere, gam_to_psi, inv_exp_map_sphere, normalize_warp, psi_to_gam};

/// A geodesic path between two 1-D curves in the elastic metric.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GeodesicPath {
    /// Interpolated curves (n_points x m).
    pub curves: FdMatrix,
    /// Interpolated warps (n_points x m).
    pub warps: FdMatrix,
    /// Elastic distance from f1 at each interpolation point.
    pub distances: Vec<f64>,
    /// Parameter values t in \[0, 1\].
    pub parameter_values: Vec<f64>,
}

/// A geodesic path between two N-D curves in the elastic metric.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct GeodesicPathNd {
    /// Interpolated curves per dimension: d `FdMatrix`es, each (n_points x m).
    pub curves: Vec<FdMatrix>,
    /// Interpolated warps (n_points x m).
    pub warps: FdMatrix,
    /// Elastic distance from f1 at each interpolation point.
    pub distances: Vec<f64>,
    /// Parameter values t in \[0, 1\].
    pub parameter_values: Vec<f64>,
}

/// Compute the geodesic path between two 1-D curves in the elastic metric.
///
/// The path is parameterized by `n_points` values in \[0, 1\]. At t=0 the path
/// is at `f1`; at t=1 it is at the aligned version of `f2`. Interpolation
/// proceeds separately in amplitude (linear in SRSF space) and phase
/// (geodesic on the Hilbert sphere of warping functions).
///
/// # Arguments
/// * `f1`       - First curve (length m).
/// * `f2`       - Second curve (length m).
/// * `argvals`  - Evaluation grid (length m).
/// * `n_points` - Number of interpolation points (>= 2).
/// * `lambda`   - Alignment penalty (0 = no penalty).
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if lengths mismatch or are < 2.
/// Returns `FdarError::InvalidParameter` if `n_points < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_geodesic(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    n_points: usize,
    lambda: f64,
) -> Result<GeodesicPath, FdarError> {
    let m = f1.len();

    // ── Validation ──────────────────────────────────────────────────────
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "f1",
            expected: "length >= 2".to_string(),
            actual: format!("length {m}"),
        });
    }
    if f2.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "f2",
            expected: format!("length {m}"),
            actual: format!("length {}", f2.len()),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m}"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if n_points < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_points",
            message: format!("must be >= 2, got {n_points}"),
        });
    }

    // ── Align f2 to f1 ─────────────────────────────────────────────────
    let q1 = srsf_single(f1, argvals);
    let q2 = srsf_single(f2, argvals);
    let gamma = dp_alignment_core(&q1, &q2, argvals, lambda);
    let f2_aligned = reparameterize_curve(f2, argvals, &gamma);
    let q2a = srsf_single(&f2_aligned, argvals);

    // ── Phase geodesic setup ────────────────────────────────────────────
    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    let time_01: Vec<f64> = (0..m).map(|j| (j as f64) / (m - 1) as f64).collect();
    let binsize = 1.0 / (m - 1) as f64;

    let gamma_01: Vec<f64> = gamma.iter().map(|&g| (g - t0) / domain).collect();
    let psi = gam_to_psi(&gamma_01, binsize);
    let psi_id = gam_to_psi(&time_01, binsize);
    let v = inv_exp_map_sphere(&psi_id, &psi, &time_01);

    // ── Integration weights for distance computation ────────────────────
    let weights = simpsons_weights(argvals);

    // ── Interpolation ───────────────────────────────────────────────────
    let parameter_values: Vec<f64> = (0..n_points)
        .map(|k| k as f64 / (n_points - 1) as f64)
        .collect();

    let mut curves = FdMatrix::zeros(n_points, m);
    let mut warps = FdMatrix::zeros(n_points, m);
    let mut distances = Vec::with_capacity(n_points);

    for (k, &t_k) in parameter_values.iter().enumerate() {
        // Phase: geodesic on the Hilbert sphere
        let scaled_v: Vec<f64> = v.iter().map(|&vi| t_k * vi).collect();
        let psi_k = exp_map_sphere(&psi_id, &scaled_v, &time_01);
        let mut gamma_k_01 = psi_to_gam(&psi_k, &time_01);
        // Rescale from [0,1] to original domain
        for j in 0..m {
            gamma_k_01[j] = t0 + gamma_k_01[j] * domain;
        }
        normalize_warp(&mut gamma_k_01, argvals);

        // Amplitude: linear interpolation in SRSF space
        let q_k: Vec<f64> = (0..m).map(|j| (1.0 - t_k) * q1[j] + t_k * q2a[j]).collect();

        // Reconstruct curve from SRSF
        let f0_k = (1.0 - t_k) * f1[0] + t_k * f2_aligned[0];
        let f_k = srsf_inverse(&q_k, argvals, f0_k);

        // L2 distance from q1 to q_k
        let dist = l2_distance(&q1, &q_k, &weights);

        for j in 0..m {
            curves[(k, j)] = f_k[j];
            warps[(k, j)] = gamma_k_01[j];
        }
        distances.push(dist);
    }

    Ok(GeodesicPath {
        curves,
        warps,
        distances,
        parameter_values,
    })
}

/// Compute the geodesic path between two N-D curves in the elastic metric.
///
/// Similar to [`curve_geodesic`], but for multidimensional (R^d) curves.
/// The warping function is shared across all dimensions.
///
/// # Arguments
/// * `f1`       - First curve set (d dimensions, 1 curve each).
/// * `f2`       - Second curve set (d dimensions, 1 curve each).
/// * `argvals`  - Evaluation grid (length m).
/// * `n_points` - Number of interpolation points (>= 2).
/// * `lambda`   - Alignment penalty (0 = no penalty).
///
/// # Errors
/// Returns `FdarError::InvalidDimension` if curve sets have inconsistent dimensions.
/// Returns `FdarError::InvalidParameter` if `n_points < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn curve_geodesic_nd(
    f1: &FdCurveSet,
    f2: &FdCurveSet,
    argvals: &[f64],
    n_points: usize,
    lambda: f64,
) -> Result<GeodesicPathNd, FdarError> {
    let d = f1.ndim();
    let m = f1.npoints();

    // ── Validation ──────────────────────────────────────────────────────
    if d == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "f1",
            expected: "ndim >= 1".to_string(),
            actual: "ndim 0".to_string(),
        });
    }
    if f2.ndim() != d {
        return Err(FdarError::InvalidDimension {
            parameter: "f2",
            expected: format!("ndim {d}"),
            actual: format!("ndim {}", f2.ndim()),
        });
    }
    if f2.npoints() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "f2",
            expected: format!("{m} points"),
            actual: format!("{} points", f2.npoints()),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "f1",
            expected: "npoints >= 2".to_string(),
            actual: format!("npoints {m}"),
        });
    }
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m}"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if n_points < 2 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_points",
            message: format!("must be >= 2, got {n_points}"),
        });
    }

    // ── Align f2 to f1 ─────────────────────────────────────────────────
    let result = elastic_align_pair_nd(f1, f2, argvals, lambda);
    let gamma = &result.gamma;

    // SRSFs of f1 and aligned f2 per dimension
    let q1_set = srsf_transform_nd(f1, argvals);
    let f2_aligned_set = {
        let dims: Vec<FdMatrix> = result
            .f_aligned
            .iter()
            .map(|fa| FdMatrix::from_slice(fa, 1, m).expect("dimension invariant"))
            .collect();
        FdCurveSet { dims }
    };
    let q2a_set = srsf_transform_nd(&f2_aligned_set, argvals);

    let q1: Vec<Vec<f64>> = q1_set.dims.iter().map(|dm| dm.row(0)).collect();
    let q2a: Vec<Vec<f64>> = q2a_set.dims.iter().map(|dm| dm.row(0)).collect();

    // ── Phase geodesic setup (shared across dimensions) ─────────────────
    let t0 = argvals[0];
    let domain = argvals[m - 1] - t0;
    let time_01: Vec<f64> = (0..m).map(|j| (j as f64) / (m - 1) as f64).collect();
    let binsize = 1.0 / (m - 1) as f64;

    let gamma_01: Vec<f64> = gamma.iter().map(|&g| (g - t0) / domain).collect();
    let psi = gam_to_psi(&gamma_01, binsize);
    let psi_id = gam_to_psi(&time_01, binsize);
    let v = inv_exp_map_sphere(&psi_id, &psi, &time_01);

    // ── Integration weights ─────────────────────────────────────────────
    let weights = simpsons_weights(argvals);

    // ── Interpolation ───────────────────────────────────────────────────
    let parameter_values: Vec<f64> = (0..n_points)
        .map(|k| k as f64 / (n_points - 1) as f64)
        .collect();

    let mut dim_curves: Vec<FdMatrix> = (0..d).map(|_| FdMatrix::zeros(n_points, m)).collect();
    let mut warps_mat = FdMatrix::zeros(n_points, m);
    let mut distances = Vec::with_capacity(n_points);

    for (k, &t_k) in parameter_values.iter().enumerate() {
        // Phase: geodesic on the Hilbert sphere
        let scaled_v: Vec<f64> = v.iter().map(|&vi| t_k * vi).collect();
        let psi_k = exp_map_sphere(&psi_id, &scaled_v, &time_01);
        let mut gamma_k_01 = psi_to_gam(&psi_k, &time_01);
        for j in 0..m {
            gamma_k_01[j] = t0 + gamma_k_01[j] * domain;
        }
        normalize_warp(&mut gamma_k_01, argvals);

        for j in 0..m {
            warps_mat[(k, j)] = gamma_k_01[j];
        }

        // Per-dimension amplitude interpolation + reconstruction
        let mut dist_sq = 0.0;
        for dd in 0..d {
            let q_k: Vec<f64> = (0..m)
                .map(|j| (1.0 - t_k) * q1[dd][j] + t_k * q2a[dd][j])
                .collect();

            let f0_k = (1.0 - t_k) * f1.dims[dd][(0, 0)] + t_k * result.f_aligned[dd][0];
            let f_k = srsf_inverse(&q_k, argvals, f0_k);

            let d_k = l2_distance(&q1[dd], &q_k, &weights);
            dist_sq += d_k * d_k;

            for j in 0..m {
                dim_curves[dd][(k, j)] = f_k[j];
            }
        }
        distances.push(dist_sq.sqrt());
    }

    Ok(GeodesicPathNd {
        curves: dim_curves,
        warps: warps_mat,
        distances,
        parameter_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::uniform_grid;

    #[test]
    fn geodesic_endpoints_match() {
        let m = 51;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).cos())
            .collect();

        let path = curve_geodesic(&f1, &f2, &t, 5, 0.0).unwrap();

        // At t=0 the path curve should approximate f1
        let first_curve = path.curves.row(0);
        let max_diff_start: f64 = first_curve
            .iter()
            .zip(f1.iter())
            .map(|(&a, &b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff_start < 0.5,
            "At t=0 curve should approximate f1, max diff = {max_diff_start}"
        );

        // The last curve is at t=1, which should approximate f2_aligned
        // (not necessarily f2 itself, but should be a valid curve)
        let last_curve = path.curves.row(path.parameter_values.len() - 1);
        assert_eq!(last_curve.len(), m);
    }

    #[test]
    fn geodesic_distances_nonneg() {
        let m = 41;
        let t = uniform_grid(m);
        let f1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f2: Vec<f64> = t
            .iter()
            .map(|&ti| 0.5 * (4.0 * std::f64::consts::PI * ti).sin())
            .collect();

        let path = curve_geodesic(&f1, &f2, &t, 6, 0.0).unwrap();
        for (k, &d) in path.distances.iter().enumerate() {
            assert!(d >= 0.0, "Distance at k={k} should be >= 0, got {d}");
        }
    }

    #[test]
    fn geodesic_identical_curves() {
        let m = 41;
        let t = uniform_grid(m);
        let f: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();

        let path = curve_geodesic(&f, &f, &t, 4, 0.0).unwrap();

        // All interpolated curves should be close to f
        for k in 0..path.parameter_values.len() {
            let curve_k = path.curves.row(k);
            let max_diff: f64 = curve_k
                .iter()
                .zip(f.iter())
                .map(|(&a, &b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            assert!(
                max_diff < 0.5,
                "Identical curve geodesic: curve at k={k} deviates by {max_diff}"
            );
        }

        // Distances should be near zero
        for (k, &d) in path.distances.iter().enumerate() {
            assert!(
                d < 1.0,
                "Identical curve geodesic: distance at k={k} = {d}, expected near 0"
            );
        }
    }

    #[test]
    fn geodesic_nd_dimensions() {
        let m = 31;
        let t = uniform_grid(m);
        let d = 2;
        let n_points = 4;

        let f1x: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let f1y: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).cos())
            .collect();
        let f2x: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
        let f2y: Vec<f64> = t.to_vec();

        let f1 = FdCurveSet::from_dims(vec![
            FdMatrix::from_slice(&f1x, 1, m).unwrap(),
            FdMatrix::from_slice(&f1y, 1, m).unwrap(),
        ])
        .unwrap();
        let f2 = FdCurveSet::from_dims(vec![
            FdMatrix::from_slice(&f2x, 1, m).unwrap(),
            FdMatrix::from_slice(&f2y, 1, m).unwrap(),
        ])
        .unwrap();

        let path = curve_geodesic_nd(&f1, &f2, &t, n_points, 0.0).unwrap();

        assert_eq!(path.curves.len(), d, "Should have d dimension matrices");
        for (dd, dim_mat) in path.curves.iter().enumerate() {
            assert_eq!(
                dim_mat.shape(),
                (n_points, m),
                "Dimension {dd} matrix shape mismatch"
            );
        }
        assert_eq!(path.warps.shape(), (n_points, m));
        assert_eq!(path.distances.len(), n_points);
        assert_eq!(path.parameter_values.len(), n_points);
    }
}
