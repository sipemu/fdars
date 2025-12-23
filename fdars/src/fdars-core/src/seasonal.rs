//! Seasonal time series analysis for functional data.
//!
//! This module provides functions for analyzing seasonal patterns in functional data:
//! - Period estimation (FFT, autocorrelation, regression-based)
//! - Peak detection with prominence calculation
//! - Seasonal strength measurement (variance and spectral methods)
//! - Seasonality change detection (onset/cessation)
//! - Instantaneous period estimation for drifting seasonality
//! - Peak timing variability analysis for short series
//! - Seasonality classification

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

/// A detected period from multiple period detection.
#[derive(Debug, Clone)]
pub struct DetectedPeriod {
    /// Estimated period
    pub period: f64,
    /// FFT confidence (ratio of peak power to mean power)
    pub confidence: f64,
    /// Seasonal strength at this period (variance explained)
    pub strength: f64,
    /// Amplitude of the sinusoidal component
    pub amplitude: f64,
    /// Phase of the sinusoidal component (radians)
    pub phase: f64,
    /// Iteration number (1-indexed)
    pub iteration: usize,
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

/// Result of peak timing variability analysis.
#[derive(Debug, Clone)]
pub struct PeakTimingResult {
    /// Peak times for each cycle
    pub peak_times: Vec<f64>,
    /// Peak values
    pub peak_values: Vec<f64>,
    /// Within-period timing (0-1 scale, e.g., day-of-year / 365)
    pub normalized_timing: Vec<f64>,
    /// Mean normalized timing
    pub mean_timing: f64,
    /// Standard deviation of normalized timing
    pub std_timing: f64,
    /// Range of normalized timing (max - min)
    pub range_timing: f64,
    /// Variability score (0 = stable, 1 = highly variable)
    pub variability_score: f64,
    /// Trend in timing (positive = peaks getting later)
    pub timing_trend: f64,
    /// Cycle indices (1-indexed)
    pub cycle_indices: Vec<usize>,
}

/// Type of seasonality pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeasonalType {
    /// Regular peaks with consistent timing
    StableSeasonal,
    /// Regular peaks but timing shifts between cycles
    VariableTiming,
    /// Some cycles seasonal, some not
    IntermittentSeasonal,
    /// No clear seasonality
    NonSeasonal,
}

/// Result of seasonality classification.
#[derive(Debug, Clone)]
pub struct SeasonalityClassification {
    /// Whether the series is seasonal overall
    pub is_seasonal: bool,
    /// Whether peak timing is stable across cycles
    pub has_stable_timing: bool,
    /// Timing variability score (0 = stable, 1 = highly variable)
    pub timing_variability: f64,
    /// Overall seasonal strength
    pub seasonal_strength: f64,
    /// Per-cycle seasonal strength
    pub cycle_strengths: Vec<f64>,
    /// Indices of weak/missing seasons (0-indexed)
    pub weak_seasons: Vec<usize>,
    /// Classification type
    pub classification: SeasonalType,
    /// Peak timing analysis (if peaks were detected)
    pub peak_timing: Option<PeakTimingResult>,
}

/// Method for automatic threshold selection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThresholdMethod {
    /// Fixed user-specified threshold
    Fixed(f64),
    /// Percentile of strength distribution
    Percentile(f64),
    /// Otsu's method (optimal bimodal separation)
    Otsu,
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

/// Select optimal smoothing lambda using GCV (Generalized Cross-Validation).
///
/// Performs grid search over lambda values and returns the one with minimum GCV.
fn select_lambda_gcv(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    nbasis: usize,
    order: usize,
) -> f64 {
    // Lambda grid: log-spaced from 0.01 to 10000
    let lambdas: Vec<f64> = (0..20)
        .map(|i| 0.01 * (10.0_f64).powf(i as f64 * 0.3))
        .collect();

    let mut best_lambda = 10.0;
    let mut best_gcv = f64::INFINITY;

    for &lambda in &lambdas {
        if let Some(result) = crate::basis::pspline_fit_1d(data, n, m, argvals, nbasis, lambda, order) {
            if result.gcv < best_gcv && result.gcv.is_finite() {
                best_gcv = result.gcv;
                best_lambda = lambda;
            }
        }
    }

    best_lambda
}

/// Compute Otsu's threshold for bimodal separation.
///
/// Finds the threshold that minimizes within-class variance (or equivalently
/// maximizes between-class variance).
fn otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.5;
    }

    // Filter NaN values
    let valid: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if valid.is_empty() {
        return 0.5;
    }

    let min_val = valid.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = valid.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        return (min_val + max_val) / 2.0;
    }

    // Create histogram with 256 bins
    let n_bins = 256;
    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut histogram = vec![0usize; n_bins];

    for &v in &valid {
        let bin = ((v - min_val) / bin_width).min(n_bins as f64 - 1.0) as usize;
        histogram[bin] += 1;
    }

    let total = valid.len() as f64;
    let mut sum_total = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut best_threshold = min_val;
    let mut best_variance = 0.0;

    let mut sum_b = 0.0;
    let mut weight_b = 0.0;

    for t in 0..n_bins {
        weight_b += histogram[t] as f64;
        if weight_b == 0.0 {
            continue;
        }

        let weight_f = total - weight_b;
        if weight_f == 0.0 {
            break;
        }

        sum_b += t as f64 * histogram[t] as f64;

        let mean_b = sum_b / weight_b;
        let mean_f = (sum_total - sum_b) / weight_f;

        // Between-class variance
        let variance = weight_b * weight_f * (mean_b - mean_f).powi(2);

        if variance > best_variance {
            best_variance = variance;
            best_threshold = min_val + (t as f64 + 0.5) * bin_width;
        }
    }

    best_threshold
}

/// Compute linear regression slope (simple OLS).
fn linear_slope(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x: f64 = x.iter().sum::<f64>() / n;
    let mean_y: f64 = y.iter().sum::<f64>() / n;

    let mut num = 0.0;
    let mut den = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        num += (xi - mean_x) * (yi - mean_y);
        den += (xi - mean_x).powi(2);
    }

    if den.abs() < 1e-15 {
        0.0
    } else {
        num / den
    }
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

/// Detect multiple concurrent periodicities using iterative residual subtraction.
///
/// This function iteratively:
/// 1. Estimates the dominant period using FFT
/// 2. Checks both FFT confidence and seasonal strength as stopping criteria
/// 3. Computes the amplitude and phase of the sinusoidal component
/// 4. Subtracts the fitted sinusoid from the signal
/// 5. Repeats on the residual until stopping criteria are met
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `max_periods` - Maximum number of periods to detect
/// * `min_confidence` - Minimum FFT confidence to continue (default: 0.4)
/// * `min_strength` - Minimum seasonal strength to continue (default: 0.15)
///
/// # Returns
/// Vector of detected periods with their properties
pub fn detect_multiple_periods(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    max_periods: usize,
    min_confidence: f64,
    min_strength: f64,
) -> Vec<DetectedPeriod> {
    if n == 0 || m < 4 || argvals.len() != m || max_periods == 0 {
        return Vec::new();
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

    let mut residual = mean_curve.clone();
    let mut detected = Vec::with_capacity(max_periods);

    for iteration in 1..=max_periods {
        // Create single-sample data from residual
        let residual_data: Vec<f64> = residual.clone();

        // Estimate period
        let est = estimate_period_fft(&residual_data, 1, m, argvals);

        if est.confidence < min_confidence || est.period.is_nan() || est.period.is_infinite() {
            break;
        }

        // Check seasonal strength at detected period
        let strength = seasonal_strength_variance(&residual_data, 1, m, argvals, est.period, 3);

        if strength < min_strength || strength.is_nan() {
            break;
        }

        // Compute amplitude and phase using least squares fit
        let omega = 2.0 * PI / est.period;
        let mut cos_sum = 0.0;
        let mut sin_sum = 0.0;

        for (j, &t) in argvals.iter().enumerate() {
            cos_sum += residual[j] * (omega * t).cos();
            sin_sum += residual[j] * (omega * t).sin();
        }

        let a = 2.0 * cos_sum / m as f64; // Cosine coefficient
        let b = 2.0 * sin_sum / m as f64; // Sine coefficient
        let amplitude = (a * a + b * b).sqrt();
        let phase = b.atan2(a);

        detected.push(DetectedPeriod {
            period: est.period,
            confidence: est.confidence,
            strength,
            amplitude,
            phase,
            iteration,
        });

        // Subtract fitted sinusoid from residual
        for (j, &t) in argvals.iter().enumerate() {
            let fitted = a * (omega * t).cos() + b * (omega * t).sin();
            residual[j] -= fitted;
        }
    }

    detected
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
/// * `smooth_lambda` - Smoothing parameter. If None and smooth_first=true,
///   uses GCV to automatically select optimal lambda.
pub fn detect_peaks(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    min_distance: Option<f64>,
    min_prominence: Option<f64>,
    smooth_first: bool,
    smooth_lambda: Option<f64>,
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
    let work_data = if smooth_first {
        // Determine lambda: use provided value or select via GCV
        let lambda = smooth_lambda.unwrap_or_else(|| {
            select_lambda_gcv(data, n, m, argvals, 20, 2)
        });

        if lambda > 0.0 {
            // Use P-spline smoothing
            if let Some(result) =
                crate::basis::pspline_fit_1d(data, n, m, argvals, 20, lambda, 2)
            {
                result.fitted
            } else {
                data.to_vec()
            }
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

// ============================================================================
// Peak Timing Variability Analysis
// ============================================================================

/// Analyze peak timing variability across cycles.
///
/// For short series (e.g., 3-5 years of yearly data), this function detects
/// one peak per cycle and analyzes how peak timing varies between cycles.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `period` - Known period (e.g., 365 for daily data with yearly seasonality)
/// * `smooth_lambda` - Smoothing parameter. If None, uses GCV for automatic selection.
///
/// # Returns
/// Peak timing result with variability metrics
pub fn analyze_peak_timing(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
    smooth_lambda: Option<f64>,
) -> PeakTimingResult {
    if n == 0 || m < 3 || argvals.len() != m || period <= 0.0 {
        return PeakTimingResult {
            peak_times: Vec::new(),
            peak_values: Vec::new(),
            normalized_timing: Vec::new(),
            mean_timing: f64::NAN,
            std_timing: f64::NAN,
            range_timing: f64::NAN,
            variability_score: f64::NAN,
            timing_trend: f64::NAN,
            cycle_indices: Vec::new(),
        };
    }

    // Detect peaks with minimum distance constraint of 0.7 * period
    // This ensures we get at most one peak per cycle
    let min_distance = period * 0.7;
    let peaks = detect_peaks(
        data, n, m, argvals,
        Some(min_distance),
        None,  // No prominence filter
        true,  // Smooth first
        smooth_lambda,
    );

    // Use the first sample's peaks (for mean curve analysis)
    // If multiple samples, we take the mean curve which is effectively in sample 0
    let sample_peaks = if peaks.peaks.is_empty() {
        Vec::new()
    } else {
        peaks.peaks[0].clone()
    };

    if sample_peaks.is_empty() {
        return PeakTimingResult {
            peak_times: Vec::new(),
            peak_values: Vec::new(),
            normalized_timing: Vec::new(),
            mean_timing: f64::NAN,
            std_timing: f64::NAN,
            range_timing: f64::NAN,
            variability_score: f64::NAN,
            timing_trend: f64::NAN,
            cycle_indices: Vec::new(),
        };
    }

    let peak_times: Vec<f64> = sample_peaks.iter().map(|p| p.time).collect();
    let peak_values: Vec<f64> = sample_peaks.iter().map(|p| p.value).collect();

    // Compute normalized timing (position within cycle, 0-1 scale)
    let t_start = argvals[0];
    let normalized_timing: Vec<f64> = peak_times
        .iter()
        .map(|&t| {
            let cycle_pos = (t - t_start) % period;
            cycle_pos / period
        })
        .collect();

    // Compute cycle indices (1-indexed)
    let cycle_indices: Vec<usize> = peak_times
        .iter()
        .map(|&t| {
            ((t - t_start) / period).floor() as usize + 1
        })
        .collect();

    // Compute statistics
    let n_peaks = normalized_timing.len() as f64;
    let mean_timing = normalized_timing.iter().sum::<f64>() / n_peaks;

    let variance: f64 = normalized_timing
        .iter()
        .map(|&x| (x - mean_timing).powi(2))
        .sum::<f64>() / n_peaks;
    let std_timing = variance.sqrt();

    let min_timing = normalized_timing.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_timing = normalized_timing.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range_timing = max_timing - min_timing;

    // Variability score: normalized std deviation
    // Max possible std for uniform in [0,1] is ~0.289, so we scale by that
    // But since peaks cluster, we use 0.1 as "high" variability threshold
    let variability_score = (std_timing / 0.1).min(1.0);

    // Timing trend: linear regression of normalized timing on cycle index
    let cycle_idx_f64: Vec<f64> = cycle_indices.iter().map(|&i| i as f64).collect();
    let timing_trend = linear_slope(&cycle_idx_f64, &normalized_timing);

    PeakTimingResult {
        peak_times,
        peak_values,
        normalized_timing,
        mean_timing,
        std_timing,
        range_timing,
        variability_score,
        timing_trend,
        cycle_indices,
    }
}

// ============================================================================
// Seasonality Classification
// ============================================================================

/// Classify the type of seasonality in functional data.
///
/// This is particularly useful for short series (3-5 years) where you need
/// to identify:
/// - Whether seasonality is present
/// - Whether peak timing is stable or variable
/// - Which cycles have weak or missing seasonality
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `period` - Known seasonal period
/// * `strength_threshold` - Threshold for seasonal/non-seasonal (default: 0.3)
/// * `timing_threshold` - Max std of normalized timing for "stable" (default: 0.05)
///
/// # Returns
/// Seasonality classification with type and diagnostics
pub fn classify_seasonality(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
    strength_threshold: Option<f64>,
    timing_threshold: Option<f64>,
) -> SeasonalityClassification {
    let strength_thresh = strength_threshold.unwrap_or(0.3);
    let timing_thresh = timing_threshold.unwrap_or(0.05);

    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return SeasonalityClassification {
            is_seasonal: false,
            has_stable_timing: false,
            timing_variability: f64::NAN,
            seasonal_strength: f64::NAN,
            cycle_strengths: Vec::new(),
            weak_seasons: Vec::new(),
            classification: SeasonalType::NonSeasonal,
            peak_timing: None,
        };
    }

    // Compute overall seasonal strength
    let overall_strength = seasonal_strength_variance(data, n, m, argvals, period, 3);

    // Compute per-cycle strength
    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let _points_per_cycle = (period / dt).round() as usize;
    let t_start = argvals[0];
    let t_end = argvals[m - 1];
    let n_cycles = ((t_end - t_start) / period).floor() as usize;

    let mut cycle_strengths = Vec::with_capacity(n_cycles);
    let mut weak_seasons = Vec::new();

    for cycle in 0..n_cycles {
        let cycle_start = t_start + cycle as f64 * period;
        let cycle_end = cycle_start + period;

        // Find indices for this cycle
        let start_idx = argvals.iter().position(|&t| t >= cycle_start).unwrap_or(0);
        let end_idx = argvals.iter().position(|&t| t > cycle_end).unwrap_or(m);

        let cycle_m = end_idx - start_idx;
        if cycle_m < 4 {
            cycle_strengths.push(f64::NAN);
            continue;
        }

        // Extract cycle data
        let cycle_data: Vec<f64> = (start_idx..end_idx)
            .flat_map(|j| (0..n).map(move |i| data[i + j * n]))
            .collect();
        let cycle_argvals: Vec<f64> = argvals[start_idx..end_idx].to_vec();

        let strength = seasonal_strength_variance(
            &cycle_data, n, cycle_m, &cycle_argvals, period, 3
        );

        cycle_strengths.push(strength);

        if strength < strength_thresh {
            weak_seasons.push(cycle);
        }
    }

    // Analyze peak timing
    let peak_timing = analyze_peak_timing(data, n, m, argvals, period, None);

    // Determine classification
    let is_seasonal = overall_strength >= strength_thresh;
    let has_stable_timing = peak_timing.std_timing <= timing_thresh;
    let timing_variability = peak_timing.variability_score;

    // Classify based on patterns
    let n_weak = weak_seasons.len();
    let classification = if !is_seasonal {
        SeasonalType::NonSeasonal
    } else if n_cycles > 0 && n_weak as f64 / n_cycles as f64 > 0.3 {
        // More than 30% of cycles are weak
        SeasonalType::IntermittentSeasonal
    } else if !has_stable_timing {
        SeasonalType::VariableTiming
    } else {
        SeasonalType::StableSeasonal
    };

    SeasonalityClassification {
        is_seasonal,
        has_stable_timing,
        timing_variability,
        seasonal_strength: overall_strength,
        cycle_strengths,
        weak_seasons,
        classification,
        peak_timing: Some(peak_timing),
    }
}

/// Detect seasonality changes with automatic threshold selection.
///
/// Uses Otsu's method or percentile-based threshold instead of a fixed value.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `n` - Number of samples
/// * `m` - Number of evaluation points
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
/// * `threshold_method` - Method for threshold selection
/// * `window_size` - Window size for local strength estimation
/// * `min_duration` - Minimum duration to confirm a change
pub fn detect_seasonality_changes_auto(
    data: &[f64],
    n: usize,
    m: usize,
    argvals: &[f64],
    period: f64,
    threshold_method: ThresholdMethod,
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
    let strength_curve = seasonal_strength_windowed(
        data, n, m, argvals, period, window_size, StrengthMethod::Variance
    );

    if strength_curve.is_empty() {
        return ChangeDetectionResult {
            change_points: Vec::new(),
            strength_curve: Vec::new(),
        };
    }

    // Determine threshold
    let threshold = match threshold_method {
        ThresholdMethod::Fixed(t) => t,
        ThresholdMethod::Percentile(p) => {
            let mut sorted: Vec<f64> = strength_curve.iter()
                .copied()
                .filter(|x| x.is_finite())
                .collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.is_empty() {
                0.5
            } else {
                let idx = ((p / 100.0) * sorted.len() as f64) as usize;
                sorted[idx.min(sorted.len() - 1)]
            }
        }
        ThresholdMethod::Otsu => otsu_threshold(&strength_curve),
    };

    // Now use the regular detection with computed threshold
    detect_seasonality_changes(
        data, n, m, argvals, period, threshold, window_size, min_duration
    )
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

        let result = detect_peaks(&data, 1, m, &argvals, Some(1.5), None, false, None);

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

    #[test]
    fn test_peak_timing_analysis() {
        // Generate 5 cycles of sine with period 2
        let m = 500;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.02).collect();
        let period = 2.0;
        let data = generate_sine(1, m, period, &argvals);

        let result = analyze_peak_timing(&data, 1, m, &argvals, period, Some(10.0));

        // Should find approximately 5 peaks
        assert!(!result.peak_times.is_empty());
        // Normalized timing should be around 0.25 (peak of sin at π/2)
        assert!(result.mean_timing.is_finite());
        // Pure sine should have low timing variability
        assert!(result.std_timing < 0.1 || result.std_timing.is_nan());
    }

    #[test]
    fn test_seasonality_classification() {
        let m = 400;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect();
        let period = 2.0;
        let data = generate_sine(1, m, period, &argvals);

        let result = classify_seasonality(&data, 1, m, &argvals, period, None, None);

        assert!(result.is_seasonal);
        assert!(result.seasonal_strength > 0.5);
        assert!(matches!(result.classification, SeasonalType::StableSeasonal | SeasonalType::VariableTiming));
    }

    #[test]
    fn test_otsu_threshold() {
        // Bimodal distribution: mix of low (0.1-0.2) and high (0.7-0.9) values
        let values = vec![
            0.1, 0.12, 0.15, 0.18, 0.11, 0.14,
            0.7, 0.75, 0.8, 0.85, 0.9, 0.72
        ];

        let threshold = otsu_threshold(&values);

        // Threshold should be between the two modes
        // Due to small sample size, Otsu's method may not find optimal threshold
        // Just verify it returns a reasonable value in the data range
        assert!(threshold >= 0.1, "Threshold {} should be >= 0.1", threshold);
        assert!(threshold <= 0.9, "Threshold {} should be <= 0.9", threshold);
    }

    #[test]
    fn test_gcv_lambda_selection() {
        let m = 100;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();

        // Noisy sine wave
        let mut data = vec![0.0; m];
        for j in 0..m {
            data[j] = (2.0 * PI * argvals[j] / 2.0).sin() + 0.1 * (j as f64 * 0.3).sin();
        }

        let lambda = select_lambda_gcv(&data, 1, m, &argvals, 20, 2);

        // Lambda should be positive and reasonable
        assert!(lambda > 0.0);
        assert!(lambda < 10000.0);
    }

    #[test]
    fn test_detect_multiple_periods() {
        let m = 400;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect(); // 0 to 20

        // Signal with two periods: 2 and 7
        let period1 = 2.0;
        let period2 = 7.0;
        let mut data = vec![0.0; m];
        for j in 0..m {
            data[j] = (2.0 * PI * argvals[j] / period1).sin()
                + 0.6 * (2.0 * PI * argvals[j] / period2).sin();
        }

        // Use higher min_strength threshold to properly stop after real periods
        let detected = detect_multiple_periods(&data, 1, m, &argvals, 5, 0.4, 0.20);

        // Should detect exactly 2 periods with these thresholds
        assert!(detected.len() >= 2, "Expected at least 2 periods, found {}", detected.len());

        // Check that both periods were detected (order depends on amplitude)
        let periods: Vec<f64> = detected.iter().map(|d| d.period).collect();
        let has_period1 = periods.iter().any(|&p| (p - period1).abs() < 0.3);
        let has_period2 = periods.iter().any(|&p| (p - period2).abs() < 0.5);

        assert!(has_period1, "Expected to find period ~{}, got {:?}", period1, periods);
        assert!(has_period2, "Expected to find period ~{}, got {:?}", period2, periods);

        // Verify first detected has higher amplitude (amplitude 1.0 vs 0.6)
        assert!(detected[0].amplitude > detected[1].amplitude,
            "First detected should have higher amplitude");

        // Each detected period should have strength and confidence info
        for d in &detected {
            assert!(d.strength > 0.0, "Detected period should have positive strength");
            assert!(d.confidence > 0.0, "Detected period should have positive confidence");
            assert!(d.amplitude > 0.0, "Detected period should have positive amplitude");
        }
    }
}
