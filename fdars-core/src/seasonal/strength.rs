use crate::basis::fourier_basis_with_period;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use std::f64::consts::PI;

use super::{
    compute_mean_curve, cwt_morlet_fft, interior_bounds, periodogram, sum_harmonic_power,
    StrengthMethod,
};

/// Measure seasonal strength using variance decomposition.
///
/// Computes SS = Var(seasonal_component) / Var(total) where the seasonal
/// component is extracted using Fourier basis.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
/// * `n_harmonics` - Number of Fourier harmonics to use
///
/// # Returns
/// Seasonal strength in [0, 1]
pub fn seasonal_strength_variance(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    n_harmonics: usize,
) -> f64 {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return f64::NAN;
    }

    // Compute mean curve
    let mean_curve = compute_mean_curve(data);

    // Total variance
    let global_mean: f64 = mean_curve.iter().sum::<f64>() / m as f64;
    let total_var: f64 = mean_curve
        .iter()
        .map(|&x| (x - global_mean).powi(2))
        .sum::<f64>()
        / m as f64;

    if total_var < 1e-15 {
        return 0.0;
    }

    // Fit Fourier basis to extract seasonal component
    let nbasis = 1 + 2 * n_harmonics;
    let basis = fourier_basis_with_period(argvals, nbasis, period);

    // Project data onto basis (simple least squares for mean curve)
    let mut seasonal = vec![0.0; m];
    for k in 1..nbasis {
        // Skip DC component
        let b_sum: f64 = (0..m).map(|j| basis[j + k * m].powi(2)).sum();
        if b_sum > 1e-15 {
            let coef: f64 = (0..m)
                .map(|j| mean_curve[j] * basis[j + k * m])
                .sum::<f64>()
                / b_sum;
            for j in 0..m {
                seasonal[j] += coef * basis[j + k * m];
            }
        }
    }

    // Seasonal variance
    let seasonal_mean: f64 = seasonal.iter().sum::<f64>() / m as f64;
    let seasonal_var: f64 = seasonal
        .iter()
        .map(|&x| (x - seasonal_mean).powi(2))
        .sum::<f64>()
        / m as f64;

    (seasonal_var / total_var).min(1.0)
}

/// Measure seasonal strength using spectral method.
///
/// Computes SS = power at seasonal frequencies / total power.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
pub fn seasonal_strength_spectral(data: &FdMatrix, argvals: &[f64], period: f64) -> f64 {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return f64::NAN;
    }

    // Compute mean curve
    let mean_curve = compute_mean_curve(data);

    let (frequencies, power) = periodogram(&mean_curve, argvals);

    if frequencies.len() < 2 {
        return f64::NAN;
    }

    let fundamental_freq = 1.0 / period;
    let (seasonal_power, total_power) =
        sum_harmonic_power(&frequencies, &power, fundamental_freq, 0.1);

    if total_power < 1e-15 {
        return 0.0;
    }

    (seasonal_power / total_power).min(1.0)
}

/// Compute seasonal strength using Morlet wavelet power at the target period.
///
/// This method uses the Continuous Wavelet Transform (CWT) with a Morlet wavelet
/// to measure power at the specified seasonal period. Unlike spectral methods,
/// wavelets provide time-localized frequency information.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period in argvals units
///
/// # Returns
/// Seasonal strength as ratio of wavelet power to total variance (0 to 1)
///
/// # Notes
/// - Uses Morlet wavelet with omega0 = 6 (standard choice)
/// - Scale is computed as: scale = period * omega0 / (2*pi)
/// - Strength is computed over the interior 80% of the signal to avoid edge effects
pub fn seasonal_strength_wavelet(data: &FdMatrix, argvals: &[f64], period: f64) -> f64 {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return f64::NAN;
    }

    // Compute mean curve
    let mean_curve = compute_mean_curve(data);

    // Remove DC component
    let dc: f64 = mean_curve.iter().sum::<f64>() / m as f64;
    let detrended: Vec<f64> = mean_curve.iter().map(|&x| x - dc).collect();

    // Compute total variance
    let total_variance: f64 = detrended.iter().map(|&x| x * x).sum::<f64>() / m as f64;

    if total_variance < 1e-15 {
        return 0.0;
    }

    // Compute wavelet transform at the seasonal scale
    let omega0 = 6.0;
    let scale = period * omega0 / (2.0 * PI);
    let wavelet_coeffs = cwt_morlet_fft(&detrended, argvals, scale, omega0);

    if wavelet_coeffs.is_empty() {
        return f64::NAN;
    }

    // Compute wavelet power, skipping edges (10% on each side)
    let Some((interior_start, interior_end)) = interior_bounds(m) else {
        return f64::NAN;
    };

    let wavelet_power: f64 = wavelet_coeffs[interior_start..interior_end]
        .iter()
        .map(nalgebra::Complex::norm_sqr)
        .sum::<f64>()
        / (interior_end - interior_start) as f64;

    // Return ratio of wavelet power to total variance
    // Normalize so that a pure sine at the target period gives ~1.0
    (wavelet_power / total_variance).sqrt().min(1.0)
}

/// Compute time-varying seasonal strength using sliding windows.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
/// * `window_size` - Window width (recommended: 2 * period)
/// * `method` - Method for computing strength (Variance or Spectral)
///
/// # Returns
/// Seasonal strength at each time point
pub fn seasonal_strength_windowed(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    window_size: f64,
    method: StrengthMethod,
) -> Vec<f64> {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 || window_size <= 0.0 {
        return Vec::new();
    }

    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let half_window_points = ((window_size / 2.0) / dt).round() as usize;

    // Compute mean curve
    let mean_curve = compute_mean_curve(data);

    iter_maybe_parallel!(0..m)
        .map(|center| {
            let start = center.saturating_sub(half_window_points);
            let end = (center + half_window_points + 1).min(m);
            let window_m = end - start;

            if window_m < 4 {
                return f64::NAN;
            }

            let window_data: Vec<f64> = mean_curve[start..end].to_vec();
            let window_argvals: Vec<f64> = argvals[start..end].to_vec();

            // Create single-sample FdMatrix for the strength functions
            let single_mat = FdMatrix::from_column_major(window_data, 1, window_m)
                .expect("dimension invariant: data.len() == n * m");

            match method {
                StrengthMethod::Variance => {
                    seasonal_strength_variance(&single_mat, &window_argvals, period, 3)
                }
                StrengthMethod::Spectral => {
                    seasonal_strength_spectral(&single_mat, &window_argvals, period)
                }
            }
        })
        .collect()
}
