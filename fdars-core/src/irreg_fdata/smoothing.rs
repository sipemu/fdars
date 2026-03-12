//! Integration, norms, covariance estimation, distances, and grid conversion
//! for irregular functional data.

use crate::matrix::FdMatrix;
use crate::{iter_maybe_parallel, slice_maybe_parallel};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::kernels::{kernel_gaussian, mean_irreg, KernelType};
use super::{linear_interp, IrregFdata};

// =============================================================================
// Integration
// =============================================================================

/// Compute integral of each curve using trapezoidal rule.
///
/// For curve i with observation points t_1, ..., t_m and values x_1, ..., x_m:
/// integral f_i(t)dt approx sum_{j=1}^{m-1} (t_{j+1} - t_j) * (x_{j+1} + x_j) / 2
///
/// # Returns
/// Vector of integrals, one per curve
pub fn integrate_irreg(ifd: &IrregFdata) -> Vec<f64> {
    let n = ifd.n_obs();

    iter_maybe_parallel!(0..n)
        .map(|i| {
            let (t, x) = ifd.get_obs(i);

            if t.len() < 2 {
                return 0.0;
            }

            let mut integral = 0.0;
            for j in 1..t.len() {
                let h = t[j] - t[j - 1];
                integral += 0.5 * h * (x[j] + x[j - 1]);
            }
            integral
        })
        .collect()
}

// =============================================================================
// Norms
// =============================================================================

/// Compute Lp norm for each curve in irregular functional data.
///
/// ||f_i||_p = (integral |f_i(t)|^p dt)^{1/p}
///
/// Uses trapezoidal rule for integration.
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `p` - Norm order (p >= 1)
///
/// # Returns
/// Vector of norms, one per curve
pub fn norm_lp_irreg(ifd: &IrregFdata, p: f64) -> Vec<f64> {
    let n = ifd.n_obs();

    iter_maybe_parallel!(0..n)
        .map(|i| {
            let (t, x) = ifd.get_obs(i);

            if t.len() < 2 {
                return 0.0;
            }

            let mut integral = 0.0;
            if (p - 1.0).abs() < 1e-14 {
                for j in 1..t.len() {
                    let h = t[j] - t[j - 1];
                    integral += 0.5 * h * (x[j - 1].abs() + x[j].abs());
                }
                integral
            } else if (p - 2.0).abs() < 1e-14 {
                for j in 1..t.len() {
                    let h = t[j] - t[j - 1];
                    integral += 0.5 * h * (x[j - 1] * x[j - 1] + x[j] * x[j]);
                }
                integral.sqrt()
            } else {
                for j in 1..t.len() {
                    let h = t[j] - t[j - 1];
                    let val_left = x[j - 1].abs().powf(p);
                    let val_right = x[j].abs().powf(p);
                    integral += 0.5 * h * (val_left + val_right);
                }
                integral.powf(1.0 / p)
            }
        })
        .collect()
}

// =============================================================================
// Covariance Estimation
// =============================================================================

/// Estimate covariance at a grid of points using local linear smoothing.
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `s_grid` - First grid points for covariance
/// * `t_grid` - Second grid points for covariance
/// * `bandwidth` - Kernel bandwidth
///
/// # Returns
/// Covariance matrix estimate at (s_grid, t_grid) points as `FdMatrix`
pub fn cov_irreg(ifd: &IrregFdata, s_grid: &[f64], t_grid: &[f64], bandwidth: f64) -> FdMatrix {
    let n = ifd.n_obs();
    let ns = s_grid.len();
    let nt = t_grid.len();

    // First estimate mean at all observation points
    let mean_at_obs = mean_irreg(ifd, &ifd.argvals, bandwidth, KernelType::Gaussian);

    // Centered values
    let centered: Vec<f64> = ifd
        .values
        .iter()
        .zip(mean_at_obs.iter())
        .map(|(&v, &m)| v - m)
        .collect();

    // Estimate covariance at each (s, t) pair
    let mut cov = vec![0.0; ns * nt];

    for (si, &s) in s_grid.iter().enumerate() {
        for (ti, &t) in t_grid.iter().enumerate() {
            cov[si + ti * ns] =
                accumulate_cov_at_point(&ifd.offsets, &ifd.argvals, &centered, n, s, t, bandwidth);
        }
    }

    FdMatrix::from_column_major(cov, ns, nt).expect("dimension invariant: data.len() == n * m")
}

/// Compute kernel-weighted covariance estimate at a single (s, t) grid point.
fn accumulate_cov_at_point(
    offsets: &[usize],
    obs_times: &[f64],
    centered: &[f64],
    n: usize,
    s: f64,
    t: f64,
    bandwidth: f64,
) -> f64 {
    let mut sum_weights = 0.0;
    let mut sum_products = 0.0;

    for i in 0..n {
        let start = offsets[i];
        let end = offsets[i + 1];
        let obs_t = &obs_times[start..end];
        let obs_c = &centered[start..end];

        for j1 in 0..obs_t.len() {
            for j2 in 0..obs_t.len() {
                let w1 = kernel_gaussian((obs_t[j1] - s) / bandwidth);
                let w2 = kernel_gaussian((obs_t[j2] - t) / bandwidth);
                let w = w1 * w2;

                sum_weights += w;
                sum_products += w * obs_c[j1] * obs_c[j2];
            }
        }
    }

    if sum_weights > 0.0 {
        sum_products / sum_weights
    } else {
        0.0
    }
}

// =============================================================================
// Distance Metrics
// =============================================================================

/// Compute Lp distance between two irregular curves.
///
/// Uses the union of observation points and linear interpolation.
fn lp_distance_pair(t1: &[f64], x1: &[f64], t2: &[f64], x2: &[f64], p: f64) -> f64 {
    // Create union of time points
    let mut all_t: Vec<f64> = t1.iter().chain(t2.iter()).copied().collect();
    all_t.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all_t.dedup();

    // Filter to common range
    let (t_min, t_max) = match (t1.first(), t1.last(), t2.first(), t2.last()) {
        (Some(&a), Some(&b), Some(&c), Some(&d)) => (a.max(c), b.min(d)),
        _ => return f64::NAN,
    };

    let common_t: Vec<f64> = all_t
        .into_iter()
        .filter(|&t| t >= t_min && t <= t_max)
        .collect();

    if common_t.len() < 2 {
        return f64::NAN;
    }

    // Interpolate both curves at common points
    let y1: Vec<f64> = common_t.iter().map(|&t| linear_interp(t1, x1, t)).collect();
    let y2: Vec<f64> = common_t.iter().map(|&t| linear_interp(t2, x2, t)).collect();

    // Compute integral of |y1 - y2|^p
    let mut integral = 0.0;
    if (p - 1.0).abs() < 1e-14 {
        for j in 1..common_t.len() {
            let h = common_t[j] - common_t[j - 1];
            integral += 0.5 * h * ((y1[j - 1] - y2[j - 1]).abs() + (y1[j] - y2[j]).abs());
        }
        integral
    } else if (p - 2.0).abs() < 1e-14 {
        for j in 1..common_t.len() {
            let h = common_t[j] - common_t[j - 1];
            let d_left = y1[j - 1] - y2[j - 1];
            let d_right = y1[j] - y2[j];
            integral += 0.5 * h * (d_left * d_left + d_right * d_right);
        }
        integral.sqrt()
    } else {
        for j in 1..common_t.len() {
            let h = common_t[j] - common_t[j - 1];
            let val_left = (y1[j - 1] - y2[j - 1]).abs().powf(p);
            let val_right = (y1[j] - y2[j]).abs().powf(p);
            integral += 0.5 * h * (val_left + val_right);
        }
        integral.powf(1.0 / p)
    }
}

/// Compute pairwise Lp distances for irregular functional data.
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `p` - Norm order
///
/// # Returns
/// Distance matrix (n x n) as `FdMatrix`
pub fn metric_lp_irreg(ifd: &IrregFdata, p: f64) -> FdMatrix {
    let n = ifd.n_obs();
    let mut dist = vec![0.0; n * n];

    // Compute upper triangle in parallel
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .collect();

    let distances: Vec<f64> = slice_maybe_parallel!(pairs)
        .map(|&(i, j)| {
            let (t_i, x_i) = ifd.get_obs(i);
            let (t_j, x_j) = ifd.get_obs(j);
            lp_distance_pair(t_i, x_i, t_j, x_j, p)
        })
        .collect();

    // Fill symmetric matrix
    for (k, &(i, j)) in pairs.iter().enumerate() {
        dist[i + j * n] = distances[k];
        dist[j + i * n] = distances[k];
    }

    FdMatrix::from_column_major(dist, n, n).expect("dimension invariant: data.len() == n * m")
}

// =============================================================================
// Conversion to Regular Grid
// =============================================================================

/// Convert irregular data to regular grid via linear interpolation.
///
/// Missing values (outside observation range) are marked as NaN.
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `target_grid` - Regular grid to interpolate to
///
/// # Returns
/// Data matrix (n x len(target_grid)) as `FdMatrix`
pub fn to_regular_grid(ifd: &IrregFdata, target_grid: &[f64]) -> FdMatrix {
    let n = ifd.n_obs();
    let m = target_grid.len();

    let result = iter_maybe_parallel!(0..n)
        .flat_map(|i| {
            let (obs_t, obs_x) = ifd.get_obs(i);

            target_grid
                .iter()
                .map(|&t| {
                    if obs_t.is_empty() || t < obs_t[0] || t > obs_t[obs_t.len() - 1] {
                        f64::NAN
                    } else {
                        linear_interp(obs_t, obs_x, t)
                    }
                })
                .collect::<Vec<f64>>()
        })
        .collect::<Vec<f64>>()
        // Transpose to column-major
        .chunks(m)
        .enumerate()
        .fold(vec![0.0; n * m], |mut acc, (i, row)| {
            for (j, &val) in row.iter().enumerate() {
                acc[i + j * n] = val;
            }
            acc
        });

    FdMatrix::from_column_major(result, n, m).expect("dimension invariant: data.len() == n * m")
}
