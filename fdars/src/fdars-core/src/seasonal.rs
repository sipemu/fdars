//! Seasonal time series analysis for functional data.
//!
//! This module provides functions for analyzing seasonal patterns in functional data:
//! - Period estimation (FFT, autocorrelation, regression-based)
//! - Peak detection with prominence calculation
//! - Seasonal strength measurement (variance and spectral methods)
//! - Seasonality change detection (onset/cessation)
//! - Instantaneous period estimation for drifting seasonality

use crate::basis::fourier_basis_with_period;
use crate::fdata::deriv_1d;
use num_complex::Complex;
use rayon::prelude::*;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Result of period estimation.
#[derive(Debug, Clone)]
pub struct PeriodEstimate {
    /// Estimated period
    pub period: f64,
    /// Dominant frequency (1/period)
    pub frequency: f64,
    /// Power at the dominant frequency
    pub power: f64,
    /// Confidence measure (ratio of peak power to mean power)
    pub confidence: f64,
}

/// A detected peak in functional data.
#[derive(Debug, Clone)]
pub struct Peak {
    /// Time at which the peak occurs
    pub time: f64,
    /// Value at the peak
    pub value: f64,
    /// Prominence of the peak (height relative to surrounding valleys)
    pub prominence: f64,
}

/// Result of peak detection.
#[derive(Debug, Clone)]
pub struct PeakDetectionResult {
    /// Peaks for each sample: peaks[sample_idx] contains peaks for that sample
    pub peaks: Vec<Vec<Peak>>,
    /// Inter-peak distances for each sample
    pub inter_peak_distances: Vec<Vec<f64>>,
    /// Mean period estimated from inter-peak distances across all samples
    pub mean_period: f64,
}

/// Method for computing seasonal strength.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StrengthMethod {
    /// Variance decomposition: Var(seasonal) / Var(total)
    Variance,
    /// Spectral: power at seasonal frequencies / total power
    Spectral,
}

/// A detected change point in seasonality.
#[derive(Debug, Clone)]
pub struct ChangePoint {
    /// Time at which the change occurs
    pub time: f64,
    /// Type of change
    pub change_type: ChangeType,
    /// Seasonal strength before the change
    pub strength_before: f64,
    /// Seasonal strength after the change
    pub strength_after: f64,
}

/// Type of seasonality change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChangeType {
    /// Series becomes seasonal
    Onset,
    /// Series stops being seasonal
    Cessation,
}

/// Result of seasonality change detection.
#[derive(Debug, Clone)]
pub struct ChangeDetectionResult {
    /// Detected change points
    pub change_points: Vec<ChangePoint>,
    /// Time-varying seasonal strength curve used for detection
    pub strength_curve: Vec<f64>,
}

/// Result of instantaneous period estimation.
#[derive(Debug, Clone)]
pub struct InstantaneousPeriod {
    /// Instantaneous period at each time point
    pub period: Vec<f64>,
    /// Instantaneous frequency at each time point
    pub frequency: Vec<f64>,
    /// Instantaneous amplitude (envelope) at each time point
    pub amplitude: Vec<f64>,
}

// ============================================================================
// Internal helper functions
// ============================================================================

/// Compute periodogram from data using FFT.
/// Returns (frequencies, power) where frequencies are in cycles per unit time.
fn periodogram(data: &[f64], argvals: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n < 2 || argvals.len() != n {
        return (Vec::new(), Vec::new());
    }

    let dt = (argvals[n - 1] - argvals[0]) / (n - 1) as f64;
    let fs = 1.0 / dt; // Sampling frequency

    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(n);

    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);

    // Compute power spectrum (one-sided)
    let n_freq = n / 2 + 1;
    let mut power = Vec::with_capacity(n_freq);
    let mut frequencies = Vec::with_capacity(n_freq);

    for k in 0..n_freq {
        let freq = k as f64 * fs / n as f64;
        frequencies.push(freq);

        let p = buffer[k].norm_sqr() / (n as f64 * n as f64);
        // Double power for non-DC and non-Nyquist frequencies (one-sided spectrum)
        let p = if k > 0 && k < n / 2 { 2.0 * p } else { p };
        power.push(p);
    }

    (frequencies, power)
}

/// Compute autocorrelation function up to max_lag.
fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-15 {
        return vec![1.0; max_lag.min(n) + 1];
    }

    let max_lag = max_lag.min(n - 1);
    let mut acf = Vec::with_capacity(max_lag + 1);

    for lag in 0..=max_lag {
        let mut sum = 0.0;
        for i in 0..(n - lag) {
            sum += (data[i] - mean) * (data[i + lag] - mean);
        }
        acf.push(sum / (n as f64 * var));
    }

    acf
}

/// Find peaks in a 1D signal, returning indices.
fn find_peaks_1d(signal: &[f64], min_distance: usize) -> Vec<usize> {
    let n = signal.len();
    if n < 3 {
        return Vec::new();
    }

    let mut peaks = Vec::new();

    for i in 1..(n - 1) {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            // Check minimum distance from previous peak
            if peaks.is_empty() || i - peaks[peaks.len() - 1] >= min_distance {
                peaks.push(i);
            } else if signal[i] > signal[peaks[peaks.len() - 1]] {
                // Replace previous peak if this one is higher
                peaks.pop();
                peaks.push(i);
            }
        }
    }

    peaks
}

/// Compute prominence for a peak (height above surrounding valleys).
fn compute_prominence(signal: &[f64], peak_idx: usize) -> f64 {
    let n = signal.len();
    let peak_val = signal[peak_idx];

    // Find lowest point between peak and boundaries/higher peaks
    let mut left_min = peak_val;
    for i in (0..peak_idx).rev() {
        if signal[i] >= peak_val {
            break;
        }
        left_min = left_min.min(signal[i]);
    }

    let mut right_min = peak_val;
    for i in (peak_idx + 1)..n {
        if signal[i] >= peak_val {
            break;
        }
        right_min = right_min.min(signal[i]);
    }

    peak_val - left_min.max(right_min)
}

/// Hilbert transform using FFT to compute analytic signal.
fn hilbert_transform(signal: &[f64]) -> Vec<Complex<f64>> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    // Forward FFT
    let mut buffer: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_forward.process(&mut buffer);

    // Create analytic signal in frequency domain
    // H[0] = 1, H[1..n/2] = 2, H[n/2] = 1 (if n even), H[n/2+1..] = 0
    let half = n / 2;
    for k in 1..half {
        buffer[k] *= 2.0;
    }
    for k in (half + 1)..n {
        buffer[k] = Complex::new(0.0, 0.0);
    }

    // Inverse FFT
    fft_inverse.process(&mut buffer);

    // Normalize
    for c in buffer.iter_mut() {
        *c /= n as f64;
    }

    buffer
}

/// Unwrap phase to remove 2π discontinuities.
fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
    if phase.is_empty() {
        return Vec::new();
    }

    let mut unwrapped = vec![phase[0]];
    let mut cumulative_correction = 0.0;

    for i in 1..phase.len() {
        let diff = phase[i] - phase[i - 1];

        // Check for wraparound
        if diff > PI {
            cumulative_correction -= 2.0 * PI;
        } else if diff < -PI {
            cumulative_correction += 2.0 * PI;
        }

        unwrapped.push(phase[i] + cumulative_correction);
    }

    unwrapped
}

// ============================================================================
// Period Estimation
// ============================================================================

/// Estimate period using FFT periodogram.
///
/// Finds the dominant frequency in the periodogram (excluding DC) and
/// returns the corresponding period.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points (time values)
///
/// # Returns
/// Period estimate with confidence measure
pub fn estimate_period_fft(data: &[f64], n: usize, m: usize, argvals: &[f64]) -> PeriodEstimate {
    if n == 0 || m < 4 || argvals.len() != m {
        return PeriodEstimate {
            period: f64::NAN,
            frequency: f64::NAN,
            power: 0.0,
            confidence: 0.0,
        };
    }

    // Compute mean curve first
    let mean_curve: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    let (frequencies, power) = periodogram(&mean_curve, argvals);

    if frequencies.len() < 2 {
        return PeriodEstimate {
            period: f64::NAN,
            frequency: f64::NAN,
            power: 0.0,
            confidence: 0.0,
        };
    }

    // Find peak in power spectrum (skip DC component at index 0)
    let mut max_power = 0.0;
    let mut max_idx = 1;
    for (i, &p) in power.iter().enumerate().skip(1) {
        if p > max_power {
            max_power = p;
            max_idx = i;
        }
    }

    let dominant_freq = frequencies[max_idx];
    let period = if dominant_freq > 1e-15 {
        1.0 / dominant_freq
    } else {
        f64::INFINITY
    };

    // Confidence: ratio of peak power to mean power (excluding DC)
    let mean_power: f64 = power.iter().skip(1).sum::<f64>() / (power.len() - 1) as f64;
    let confidence = if mean_power > 1e-15 {
        max_power / mean_power
    } else {
        0.0
    };

    PeriodEstimate {
        period,
        frequency: dominant_freq,
        power: max_power,
        confidence,
    }
}

/// Estimate period using autocorrelation function.
///
/// Finds the first significant peak in the ACF after lag 0.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `max_lag` - Maximum lag to consider (in number of points)
pub fn estimate_period_acf(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    max_lag: usize,
) -> PeriodEstimate {
    if n == 0 || m < 4 || argvals.len() != m {
        return PeriodEstimate {
            period: f64::NAN,
            frequency: f64::NAN,
            power: 0.0,
            confidence: 0.0,
        };
    }

    // Compute mean curve
    let mean_curve: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    let acf = autocorrelation(&mean_curve, max_lag);

    // Find first peak after lag 0 (skip first few lags to avoid finding lag 0)
    let min_lag = 2;
    let peaks = find_peaks_1d(&acf[min_lag..], 1);

    if peaks.is_empty() {
        return PeriodEstimate {
            period: f64::NAN,
            frequency: f64::NAN,
            power: 0.0,
            confidence: 0.0,
        };
    }

    let peak_lag = peaks[0] + min_lag;
    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let period = peak_lag as f64 * dt;
    let frequency = if period > 1e-15 { 1.0 / period } else { 0.0 };

    PeriodEstimate {
        period,
        frequency,
        power: acf[peak_lag],
        confidence: acf[peak_lag].abs(),
    }
}

/// Estimate period via Fourier regression grid search.
///
/// Tests multiple candidate periods and selects the one that minimizes
/// the reconstruction error (similar to GCV).
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `period_min` - Minimum period to test
/// * `period_max` - Maximum period to test
/// * `n_candidates` - Number of candidate periods to test
/// * `n_harmonics` - Number of Fourier harmonics to use
pub fn estimate_period_regression(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period_min: f64,
    period_max: f64,
    n_candidates: usize,
    n_harmonics: usize,
) -> PeriodEstimate {
    if n == 0 || m < 4 || argvals.len() != m || period_min >= period_max || n_candidates < 2 {
        return PeriodEstimate {
            period: f64::NAN,
            frequency: f64::NAN,
            power: 0.0,
            confidence: 0.0,
        };
    }

    // Compute mean curve
    let mean_curve: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    let nbasis = 1 + 2 * n_harmonics;

    // Grid search over candidate periods
    let candidates: Vec<f64> = (0..n_candidates)
        .map(|i| period_min + (period_max - period_min) * i as f64 / (n_candidates - 1) as f64)
        .collect();

    let results: Vec<(f64, f64)> = candidates
        .par_iter()
        .map(|&period| {
            let basis = fourier_basis_with_period(argvals, nbasis, period);

            // Simple least squares fit
            let mut rss = 0.0;
            for j in 0..m {
                let mut fitted = 0.0;
                // Simple: use mean of basis function times data as rough fit
                for k in 0..nbasis {
                    let b_val = basis[j + k * m];
                    let coef: f64 = (0..m).map(|l| mean_curve[l] * basis[l + k * m]).sum::<f64>()
                        / (0..m).map(|l| basis[l + k * m].powi(2)).sum::<f64>().max(1e-15);
                    fitted += coef * b_val;
                }
                let resid = mean_curve[j] - fitted;
                rss += resid * resid;
            }

            (period, rss)
        })
        .collect();

    // Find period with minimum RSS
    let (best_period, min_rss) = results
        .iter()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .cloned()
        .unwrap_or((f64::NAN, f64::INFINITY));

    // Confidence based on how much better the best is vs average
    let mean_rss: f64 = results.iter().map(|(_, r)| r).sum::<f64>() / results.len() as f64;
    let confidence = if min_rss > 1e-15 {
        (mean_rss / min_rss).min(10.0)
    } else {
        10.0
    };

    PeriodEstimate {
        period: best_period,
        frequency: if best_period > 1e-15 {
            1.0 / best_period
        } else {
            0.0
        },
        power: 1.0 - min_rss / mean_rss,
        confidence,
    }
}

// ============================================================================
// Peak Detection
// ============================================================================

/// Detect peaks in functional data.
///
/// Uses derivative zero-crossings to find local maxima, with optional
/// smoothing and filtering by minimum distance and prominence.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `min_distance` - Minimum time between peaks (None = no constraint)
/// * `min_prominence` - Minimum prominence (0-1 scale, None = no filter)
/// * `smooth_first` - Whether to smooth data before peak detection
/// * `smooth_lambda` - Smoothing parameter (used if smooth_first = true)
pub fn detect_peaks(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    min_distance: Option<f64>,
    min_prominence: Option<f64>,
    smooth_first: bool,
    smooth_lambda: f64,
) -> PeakDetectionResult {
    if n == 0 || m < 3 || argvals.len() != m {
        return PeakDetectionResult {
            peaks: Vec::new(),
            inter_peak_distances: Vec::new(),
            mean_period: f64::NAN,
        };
    }

    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let min_dist_points = min_distance.map(|d| (d / dt).round() as usize).unwrap_or(1);

    // Optionally smooth the data
    let work_data = if smooth_first && smooth_lambda > 0.0 {
        // Use P-spline smoothing
        if let Some(result) =
            crate::basis::pspline_fit_1d(data, n, m, argvals, 20, smooth_lambda, 2)
        {
            result.fitted
        } else {
            data.to_vec()
        }
    } else {
        data.to_vec()
    };

    // Compute first derivative
    let deriv1 = deriv_1d(&work_data, n, m, argvals, 1);

    // Compute data range for prominence normalization
    let data_range: f64 = {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;
        for &v in work_data.iter() {
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }
        (max_val - min_val).max(1e-15)
    };

    // Find peaks for each sample
    let results: Vec<(Vec<Peak>, Vec<f64>)> = (0..n)
        .into_par_iter()
        .map(|i| {
            // Extract curve and derivative
            let curve: Vec<f64> = (0..m).map(|j| work_data[i + j * n]).collect();
            let d1: Vec<f64> = (0..m).map(|j| deriv1[i + j * n]).collect();

            // Find zero crossings where derivative goes from positive to negative
            let mut peak_indices = Vec::new();
            for j in 1..m {
                if d1[j - 1] > 0.0 && d1[j] <= 0.0 {
                    // Interpolate to find more precise location
                    let idx = if (d1[j - 1] - d1[j]).abs() > 1e-15 {
                        j - 1
                    } else {
                        j
                    };

                    // Check minimum distance
                    if peak_indices.is_empty()
                        || idx - peak_indices[peak_indices.len() - 1] >= min_dist_points
                    {
                        peak_indices.push(idx);
                    }
                }
            }

            // Build peaks with prominence
            let mut peaks: Vec<Peak> = peak_indices
                .iter()
                .map(|&idx| {
                    let prominence = compute_prominence(&curve, idx) / data_range;
                    Peak {
                        time: argvals[idx],
                        value: curve[idx],
                        prominence,
                    }
                })
                .collect();

            // Filter by prominence
            if let Some(min_prom) = min_prominence {
                peaks.retain(|p| p.prominence >= min_prom);
            }

            // Compute inter-peak distances
            let distances: Vec<f64> = peaks
                .windows(2)
                .map(|w| w[1].time - w[0].time)
                .collect();

            (peaks, distances)
        })
        .collect();

    let peaks: Vec<Vec<Peak>> = results.iter().map(|(p, _)| p.clone()).collect();
    let inter_peak_distances: Vec<Vec<f64>> = results.iter().map(|(_, d)| d.clone()).collect();

    // Compute mean period from all inter-peak distances
    let all_distances: Vec<f64> = inter_peak_distances.iter().flatten().cloned().collect();
    let mean_period = if all_distances.is_empty() {
        f64::NAN
    } else {
        all_distances.iter().sum::<f64>() / all_distances.len() as f64
    };

    PeakDetectionResult {
        peaks,
        inter_peak_distances,
        mean_period,
    }
}

// ============================================================================
// Seasonal Strength
// ============================================================================

/// Measure seasonal strength using variance decomposition.
///
/// Computes SS = Var(seasonal_component) / Var(total) where the seasonal
/// component is extracted using Fourier basis.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
/// * `n_harmonics` - Number of Fourier harmonics to use
///
/// # Returns
/// Seasonal strength in [0, 1]
pub fn seasonal_strength_variance(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
    n_harmonics: usize,
) -> f64 {
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return f64::NAN;
    }

    // Compute mean curve
    let mean_curve: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    // Total variance
    let global_mean: f64 = mean_curve.iter().sum::<f64>() / m as f64;
    let total_var: f64 = mean_curve.iter().map(|&x| (x - global_mean).powi(2)).sum::<f64>() / m as f64;

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
            let coef: f64 = (0..m).map(|j| mean_curve[j] * basis[j + k * m]).sum::<f64>() / b_sum;
            for j in 0..m {
                seasonal[j] += coef * basis[j + k * m];
            }
        }
    }

    // Seasonal variance
    let seasonal_mean: f64 = seasonal.iter().sum::<f64>() / m as f64;
    let seasonal_var: f64 =
        seasonal.iter().map(|&x| (x - seasonal_mean).powi(2)).sum::<f64>() / m as f64;

    (seasonal_var / total_var).min(1.0)
}

/// Measure seasonal strength using spectral method.
///
/// Computes SS = power at seasonal frequencies / total power.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
pub fn seasonal_strength_spectral(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
) -> f64 {
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return f64::NAN;
    }

    // Compute mean curve
    let mean_curve: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    let (frequencies, power) = periodogram(&mean_curve, argvals);

    if frequencies.len() < 2 {
        return f64::NAN;
    }

    let fundamental_freq = 1.0 / period;

    // Sum power at seasonal frequencies (fundamental and harmonics)
    let _freq_tol = fundamental_freq * 0.1; // 10% tolerance (for future use)
    let mut seasonal_power = 0.0;
    let mut total_power = 0.0;

    for (i, (&freq, &p)) in frequencies.iter().zip(power.iter()).enumerate() {
        if i == 0 {
            continue;
        } // Skip DC

        total_power += p;

        // Check if frequency is near a harmonic of fundamental
        let ratio = freq / fundamental_freq;
        let nearest_harmonic = ratio.round();
        if (ratio - nearest_harmonic).abs() < 0.1 && nearest_harmonic >= 1.0 {
            seasonal_power += p;
        }
    }

    if total_power < 1e-15 {
        return 0.0;
    }

    (seasonal_power / total_power).min(1.0)
}

/// Compute time-varying seasonal strength using sliding windows.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
/// * `window_size` - Window width (recommended: 2 * period)
/// * `method` - Method for computing strength (Variance or Spectral)
///
/// # Returns
/// Seasonal strength at each time point
pub fn seasonal_strength_windowed(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
    window_size: f64,
    method: StrengthMethod,
) -> Vec<f64> {
    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 || window_size <= 0.0 {
        return Vec::new();
    }

    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let half_window_points = ((window_size / 2.0) / dt).round() as usize;

    // Compute mean curve
    let mean_curve: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    (0..m)
        .into_par_iter()
        .map(|center| {
            let start = center.saturating_sub(half_window_points);
            let end = (center + half_window_points + 1).min(m);
            let window_m = end - start;

            if window_m < 4 {
                return f64::NAN;
            }

            let window_data: Vec<f64> = mean_curve[start..end].to_vec();
            let window_argvals: Vec<f64> = argvals[start..end].to_vec();

            // Create single-sample data for the strength functions
            let single_data = window_data.clone();

            match method {
                StrengthMethod::Variance => {
                    seasonal_strength_variance(&single_data, 1, window_m, &window_argvals, period, 3)
                }
                StrengthMethod::Spectral => {
                    seasonal_strength_spectral(&single_data, 1, window_m, &window_argvals, period)
                }
            }
        })
        .collect()
}

// ============================================================================
// Seasonality Change Detection
// ============================================================================

/// Detect changes in seasonality.
///
/// Monitors time-varying seasonal strength and detects threshold crossings
/// that indicate onset or cessation of seasonality.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
/// * `threshold` - SS threshold for seasonal/non-seasonal (e.g., 0.3)
/// * `window_size` - Window size for local strength estimation
/// * `min_duration` - Minimum duration to confirm a change
pub fn detect_seasonality_changes(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
    threshold: f64,
    window_size: f64,
    min_duration: f64,
) -> ChangeDetectionResult {
    if n == 0 || m < 4 || argvals.len() != m {
        return ChangeDetectionResult {
            change_points: Vec::new(),
            strength_curve: Vec::new(),
        };
    }

    // Compute time-varying seasonal strength
    let strength_curve =
        seasonal_strength_windowed(data, n, m, argvals, period, window_size, StrengthMethod::Variance);

    if strength_curve.is_empty() {
        return ChangeDetectionResult {
            change_points: Vec::new(),
            strength_curve: Vec::new(),
        };
    }

    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let min_dur_points = (min_duration / dt).round() as usize;

    // Detect threshold crossings
    let mut change_points = Vec::new();
    let mut in_seasonal = strength_curve[0] > threshold;
    let mut last_change_idx: Option<usize> = None;

    for (i, &ss) in strength_curve.iter().enumerate().skip(1) {
        if ss.is_nan() {
            continue;
        }

        let now_seasonal = ss > threshold;

        if now_seasonal != in_seasonal {
            // Potential change point
            if let Some(last_idx) = last_change_idx {
                if i - last_idx < min_dur_points {
                    continue; // Too soon after last change
                }
            }

            let change_type = if now_seasonal {
                ChangeType::Onset
            } else {
                ChangeType::Cessation
            };

            let strength_before = if i > 0 && !strength_curve[i - 1].is_nan() {
                strength_curve[i - 1]
            } else {
                ss
            };

            change_points.push(ChangePoint {
                time: argvals[i],
                change_type,
                strength_before,
                strength_after: ss,
            });

            in_seasonal = now_seasonal;
            last_change_idx = Some(i);
        }
    }

    ChangeDetectionResult {
        change_points,
        strength_curve,
    }
}

// ============================================================================
// Instantaneous Period
// ============================================================================

/// Estimate instantaneous period using Hilbert transform.
///
/// For series with drifting/changing period, this computes the period
/// at each time point using the analytic signal.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
pub fn instantaneous_period(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
) -> InstantaneousPeriod {
    if n == 0 || m < 4 || argvals.len() != m {
        return InstantaneousPeriod {
            period: Vec::new(),
            frequency: Vec::new(),
            amplitude: Vec::new(),
        };
    }

    // Compute mean curve
    let mean_curve: Vec<f64> = (0..m)
        .map(|j| {
            let mut sum = 0.0;
            for i in 0..n {
                sum += data[i + j * n];
            }
            sum / n as f64
        })
        .collect();

    // Remove DC component (detrend by subtracting mean)
    let dc: f64 = mean_curve.iter().sum::<f64>() / m as f64;
    let detrended: Vec<f64> = mean_curve.iter().map(|&x| x - dc).collect();

    // Compute analytic signal via Hilbert transform
    let analytic = hilbert_transform(&detrended);

    // Extract instantaneous amplitude and phase
    let amplitude: Vec<f64> = analytic.iter().map(|c| c.norm()).collect();

    let phase: Vec<f64> = analytic.iter().map(|c| c.im.atan2(c.re)).collect();

    // Unwrap phase
    let unwrapped_phase = unwrap_phase(&phase);

    // Compute instantaneous frequency (derivative of phase)
    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let mut inst_freq = vec![0.0; m];

    // Central differences for interior, forward/backward at boundaries
    if m > 1 {
        inst_freq[0] = (unwrapped_phase[1] - unwrapped_phase[0]) / dt / (2.0 * PI);
    }
    for j in 1..(m - 1) {
        inst_freq[j] = (unwrapped_phase[j + 1] - unwrapped_phase[j - 1]) / (2.0 * dt) / (2.0 * PI);
    }
    if m > 1 {
        inst_freq[m - 1] = (unwrapped_phase[m - 1] - unwrapped_phase[m - 2]) / dt / (2.0 * PI);
    }

    // Period = 1/frequency (handle near-zero frequencies)
    let period: Vec<f64> = inst_freq
        .iter()
        .map(|&f| {
            if f.abs() > 1e-10 {
                (1.0 / f).abs()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    InstantaneousPeriod {
        period,
        frequency: inst_freq,
        amplitude,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn generate_sine(n: usize, m: usize, period: f64, argvals: &[f64]) -> Vec<f64> {
        let mut data = vec![0.0; n * m];
        for i in 0..n {
            for j in 0..m {
                data[i + j * n] = (2.0 * PI * argvals[j] / period).sin();
            }
        }
        data
    }

    #[test]
    fn test_period_estimation_fft() {
        let m = 200;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
        let period = 2.0;
        let data = generate_sine(1, m, period, &argvals);

        let estimate = estimate_period_fft(&data, 1, m, &argvals);
        assert!((estimate.period - period).abs() < 0.2);
        assert!(estimate.confidence > 1.0);
    }

    #[test]
    fn test_peak_detection() {
        let m = 100;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
        let period = 2.0;
        let data = generate_sine(1, m, period, &argvals);

        let result = detect_peaks(&data, 1, m, &argvals, Some(1.5), None, false, 0.0);

        // Should find approximately 5 peaks (10 / 2)
        assert!(!result.peaks[0].is_empty());
        assert!((result.mean_period - period).abs() < 0.3);
    }

    #[test]
    fn test_seasonal_strength() {
        let m = 200;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
        let period = 2.0;
        let data = generate_sine(1, m, period, &argvals);

        let strength = seasonal_strength_variance(&data, 1, m, &argvals, period, 3);
        // Pure sine should have high seasonal strength
        assert!(strength > 0.8);

        let strength_spectral = seasonal_strength_spectral(&data, 1, m, &argvals, period);
        assert!(strength_spectral > 0.5);
    }

    #[test]
    fn test_instantaneous_period() {
        let m = 200;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
        let period = 2.0;
        let data = generate_sine(1, m, period, &argvals);

        let result = instantaneous_period(&data, 1, m, &argvals);

        // Check that instantaneous period is close to true period (away from boundaries)
        let mid_period = result.period[m / 2];
        assert!(
            (mid_period - period).abs() < 0.5,
            "Expected period ~{}, got {}",
            period,
            mid_period
        );
    }
}
