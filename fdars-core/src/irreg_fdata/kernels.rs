//! Kernel functions and mean estimation for irregular functional data.

use crate::slice_maybe_parallel;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

use super::IrregFdata;

/// Kernel function type for smoothing operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelType {
    /// Epanechnikov kernel: K(u) = 0.75(1 - u^2) for |u| <= 1
    Epanechnikov,
    /// Gaussian kernel: K(u) = exp(-u^2/2) / sqrt(2*pi)
    Gaussian,
}

/// Epanechnikov kernel function.
#[inline]
pub(crate) fn kernel_epanechnikov(u: f64) -> f64 {
    if u.abs() <= 1.0 {
        0.75 * (1.0 - u * u)
    } else {
        0.0
    }
}

/// Gaussian kernel function.
#[inline]
pub(crate) fn kernel_gaussian(u: f64) -> f64 {
    (-0.5 * u * u).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

impl KernelType {
    #[inline]
    pub(crate) fn as_fn(self) -> fn(f64) -> f64 {
        match self {
            KernelType::Epanechnikov => kernel_epanechnikov,
            KernelType::Gaussian => kernel_gaussian,
        }
    }
}

/// Estimate mean function at specified target points using kernel smoothing.
///
/// Uses local weighted averaging (Nadaraya-Watson estimator) at each target point:
/// mu_hat(t) = sum_{i,j} K_h(t - t_{ij}) x_{ij} / sum_{i,j} K_h(t - t_{ij})
///
/// # Arguments
/// * `ifd` - Irregular functional data
/// * `target_argvals` - Points at which to estimate the mean
/// * `bandwidth` - Kernel bandwidth
/// * `kernel_type` - Kernel function to use
///
/// # Returns
/// Estimated mean function values at target points
pub fn mean_irreg(
    ifd: &IrregFdata,
    target_argvals: &[f64],
    bandwidth: f64,
    kernel_type: KernelType,
) -> Vec<f64> {
    let n = ifd.n_obs();
    let kernel = kernel_type.as_fn();

    slice_maybe_parallel!(target_argvals)
        .map(|&t| {
            let mut sum_weights = 0.0;
            let mut sum_values = 0.0;

            for i in 0..n {
                let (obs_t, obs_x) = ifd.get_obs(i);

                for (&ti, &xi) in obs_t.iter().zip(obs_x.iter()) {
                    let u = (ti - t) / bandwidth;
                    let w = kernel(u);
                    sum_weights += w;
                    sum_values += w * xi;
                }
            }

            if sum_weights > 0.0 {
                sum_values / sum_weights
            } else {
                f64::NAN
            }
        })
        .collect()
}
