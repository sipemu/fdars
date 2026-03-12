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

mod autoperiod;
mod change;
mod hilbert;
mod lomb_scargle;
mod matrix_profile;
mod peak;
mod period;
mod sazed;
mod ssa;
mod strength;

#[cfg(test)]
mod tests;

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use num_complex::Complex;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use rustfft::FftPlanner;
use std::f64::consts::PI;

// Re-export all public items so `seasonal::X` still works
pub use autoperiod::{
    autoperiod, autoperiod_fdata, cfd_autoperiod, cfd_autoperiod_fdata, AutoperiodCandidate,
    AutoperiodResult, CfdAutoperiodResult,
};
pub use change::{
    analyze_peak_timing, classify_seasonality, detect_amplitude_modulation,
    detect_amplitude_modulation_wavelet, detect_seasonality_changes,
    detect_seasonality_changes_auto, AmplitudeModulationResult, ModulationType, SeasonalType,
    SeasonalityClassification, ThresholdMethod, WaveletAmplitudeResult,
};
pub use hilbert::{hilbert_transform, instantaneous_period};
pub use lomb_scargle::{lomb_scargle, lomb_scargle_fdata, LombScargleResult};
pub use matrix_profile::{
    matrix_profile, matrix_profile_fdata, matrix_profile_seasonality, MatrixProfileResult,
};
pub use peak::detect_peaks;
pub use period::{
    detect_multiple_periods, estimate_period_acf, estimate_period_fft, estimate_period_regression,
};
pub use sazed::{sazed, sazed_fdata, SazedComponents, SazedResult};
pub use ssa::{ssa, ssa_fdata, ssa_seasonality, SsaResult};
pub use strength::{
    seasonal_strength_spectral, seasonal_strength_variance, seasonal_strength_wavelet,
    seasonal_strength_windowed,
};

// Re-export types that are used by multiple submodules and by external consumers
pub use self::types::*;

mod types {
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
        /// Peaks for each sample: `peaks[sample_idx]` contains peaks for that sample
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
}

// ============================================================================
// Internal helper functions (pub(super) so submodules can use them)
// ============================================================================

/// Compute mean curve from functional data matrix.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
/// * `parallel` - Use parallel iteration (default: true)
///
/// # Returns
/// Mean curve of length m
#[inline]
pub(super) fn compute_mean_curve_impl(data: &FdMatrix, parallel: bool) -> Vec<f64> {
    let (n, m) = data.shape();
    if parallel && m >= 100 {
        // Use parallel iteration for larger datasets
        iter_maybe_parallel!(0..m)
            .map(|j| {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += data[(i, j)];
                }
                sum / n as f64
            })
            .collect()
    } else {
        // Sequential for small datasets or when disabled
        (0..m)
            .map(|j| {
                let mut sum = 0.0;
                for i in 0..n {
                    sum += data[(i, j)];
                }
                sum / n as f64
            })
            .collect()
    }
}

/// Compute mean curve (parallel by default for m >= 100).
#[inline]
pub(super) fn compute_mean_curve(data: &FdMatrix) -> Vec<f64> {
    compute_mean_curve_impl(data, true)
}

/// Compute interior bounds for edge-skipping (10% on each side).
///
/// Used to avoid edge effects in wavelet and other analyses.
///
/// # Arguments
/// * `m` - Total number of points
///
/// # Returns
/// `(interior_start, interior_end)` indices, or `None` if range is invalid
#[inline]
pub(super) fn interior_bounds(m: usize) -> Option<(usize, usize)> {
    let edge_skip = (m as f64 * 0.1) as usize;
    let interior_start = edge_skip.min(m / 4);
    let interior_end = m.saturating_sub(edge_skip).max(m * 3 / 4);

    if interior_end <= interior_start {
        None
    } else {
        Some((interior_start, interior_end))
    }
}

/// Validate interior bounds with minimum span requirement.
pub(super) fn valid_interior_bounds(m: usize, min_span: usize) -> Option<(usize, usize)> {
    interior_bounds(m).filter(|&(s, e)| e > s + min_span)
}

/// Compute periodogram from data using FFT.
/// Returns (frequencies, power) where frequencies are in cycles per unit time.
pub(super) fn periodogram(data: &[f64], argvals: &[f64]) -> (Vec<f64>, Vec<f64>) {
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

/// Naive O(n*max_lag) autocorrelation for small inputs.
pub(super) fn autocorrelation_naive(data: &[f64], max_lag: usize, mean: f64, var: f64) -> Vec<f64> {
    let n = data.len();
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

/// Compute autocorrelation function up to max_lag.
///
/// Uses FFT-based Wiener-Khinchin theorem for n > 64, falling back to
/// direct computation for small inputs where FFT overhead isn't worthwhile.
pub(super) fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    // ACF convention: divide by n (matches R's acf(); divisor cancels in normalized ratio)
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    if var < 1e-15 {
        return vec![1.0; max_lag.min(n) + 1];
    }

    let max_lag = max_lag.min(n - 1);

    if n <= 64 {
        return autocorrelation_naive(data, max_lag, mean, var);
    }

    // FFT-based ACF via Wiener-Khinchin theorem
    let fft_len = (2 * n).next_power_of_two();

    let mut planner = FftPlanner::<f64>::new();
    let fft_forward = planner.plan_fft_forward(fft_len);
    let fft_inverse = planner.plan_fft_inverse(fft_len);

    // Zero-padded, mean-centered signal
    let mut buffer: Vec<Complex<f64>> = Vec::with_capacity(fft_len);
    for &x in data {
        buffer.push(Complex::new(x - mean, 0.0));
    }
    buffer.resize(fft_len, Complex::new(0.0, 0.0));

    // Forward FFT
    fft_forward.process(&mut buffer);

    // Power spectral density: |F(k)|^2
    for c in buffer.iter_mut() {
        *c = Complex::new(c.norm_sqr(), 0.0);
    }

    // Inverse FFT
    fft_inverse.process(&mut buffer);

    // Normalize and extract lags 0..=max_lag
    let norm = fft_len as f64 * n as f64 * var;
    (0..=max_lag).map(|lag| buffer[lag].re / norm).collect()
}

/// Try to add a peak, respecting minimum distance. Replaces previous peak if closer but higher.
pub(super) fn try_add_peak(
    peaks: &mut Vec<usize>,
    candidate: usize,
    signal: &[f64],
    min_distance: usize,
) {
    if let Some(&last) = peaks.last() {
        if candidate - last >= min_distance {
            peaks.push(candidate);
        } else if signal[candidate] > signal[last] {
            // Safe: peaks is non-empty since last() succeeded
            *peaks.last_mut().unwrap_or(&mut 0) = candidate;
        }
    } else {
        peaks.push(candidate);
    }
}

/// Find peaks in a 1D signal, returning indices.
pub(crate) fn find_peaks_1d(signal: &[f64], min_distance: usize) -> Vec<usize> {
    let n = signal.len();
    if n < 3 {
        return Vec::new();
    }

    let mut peaks = Vec::new();

    for i in 1..(n - 1) {
        if signal[i] > signal[i - 1] && signal[i] > signal[i + 1] {
            try_add_peak(&mut peaks, i, signal, min_distance);
        }
    }

    peaks
}

/// Compute prominence for a peak (height above surrounding valleys).
pub(crate) fn compute_prominence(signal: &[f64], peak_idx: usize) -> f64 {
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

/// Unwrap phase to remove 2pi discontinuities.
pub(super) fn unwrap_phase(phase: &[f64]) -> Vec<f64> {
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

/// Morlet wavelet function.
///
/// The Morlet wavelet is a complex exponential modulated by a Gaussian:
/// psi(t) = exp(i * omega0 * t) * exp(-t^2 / 2)
///
/// where omega0 is the central frequency (typically 6 for good time-frequency trade-off).
pub(super) fn morlet_wavelet(t: f64, omega0: f64) -> Complex<f64> {
    let gaussian = (-t * t / 2.0).exp();
    let oscillation = Complex::new((omega0 * t).cos(), (omega0 * t).sin());
    oscillation * gaussian
}

/// Continuous Wavelet Transform at a single scale using Morlet wavelet.
///
/// Computes the wavelet coefficients at the specified scale (period) for all time points.
/// Uses convolution in the time domain.
///
/// # Arguments
/// * `signal` - Input signal
/// * `argvals` - Time points
/// * `scale` - Scale parameter (related to period)
/// * `omega0` - Central frequency of Morlet wavelet (default: 6.0)
///
/// # Returns
/// Complex wavelet coefficients at each time point
#[allow(dead_code)]
pub(super) fn cwt_morlet(
    signal: &[f64],
    argvals: &[f64],
    scale: f64,
    omega0: f64,
) -> Vec<Complex<f64>> {
    let n = signal.len();
    if n == 0 || scale <= 0.0 {
        return Vec::new();
    }

    let dt = (argvals[n - 1] - argvals[0]) / (n - 1) as f64;

    // Compute wavelet coefficients via convolution
    // W(a, b) = (1/sqrt(a)) * sum x[k] * psi*((t[k] - b) / a) * dt
    let norm = 1.0 / scale.sqrt();

    (0..n)
        .map(|b| {
            let mut sum = Complex::new(0.0, 0.0);
            for k in 0..n {
                let t_normalized = (argvals[k] - argvals[b]) / scale;
                // Only compute within reasonable range (Gaussian decays quickly)
                if t_normalized.abs() < 6.0 {
                    let wavelet = morlet_wavelet(t_normalized, omega0);
                    sum += signal[k] * wavelet.conj();
                }
            }
            sum * norm * dt
        })
        .collect()
}

/// Continuous Wavelet Transform at a single scale using FFT (faster for large signals).
///
/// Uses the convolution theorem: CWT = IFFT(FFT(signal) * FFT(wavelet)*)
pub(super) fn cwt_morlet_fft(
    signal: &[f64],
    argvals: &[f64],
    scale: f64,
    omega0: f64,
) -> Vec<Complex<f64>> {
    let n = signal.len();
    if n == 0 || scale <= 0.0 {
        return Vec::new();
    }

    let dt = (argvals[n - 1] - argvals[0]) / (n - 1) as f64;
    let norm = 1.0 / scale.sqrt();

    // Compute wavelet in time domain centered at t=0
    let wavelet_time: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            // Center the wavelet
            let t = if k <= n / 2 {
                k as f64 * dt / scale
            } else {
                (k as f64 - n as f64) * dt / scale
            };
            morlet_wavelet(t, omega0) * norm
        })
        .collect();

    let mut planner = FftPlanner::<f64>::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    // FFT of signal
    let mut signal_fft: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_forward.process(&mut signal_fft);

    // FFT of wavelet
    let mut wavelet_fft = wavelet_time;
    fft_forward.process(&mut wavelet_fft);

    // Multiply in frequency domain (use conjugate of wavelet FFT for correlation)
    let mut result: Vec<Complex<f64>> = signal_fft
        .iter()
        .zip(wavelet_fft.iter())
        .map(|(s, w)| *s * w.conj())
        .collect();

    // Inverse FFT
    fft_inverse.process(&mut result);

    // Normalize and scale by dt
    for c in result.iter_mut() {
        *c *= dt / n as f64;
    }

    result
}

/// Compute Otsu's threshold for bimodal separation.
///
/// Finds the threshold that minimizes within-class variance (or equivalently
/// maximizes between-class variance).
pub(super) fn otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.5;
    }

    // Filter NaN values
    let valid: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if valid.is_empty() {
        return 0.5;
    }

    let vmin = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let vmax = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    if (vmax - vmin).abs() < 1e-10 {
        return (vmin + vmax) / 2.0;
    }

    let n_bins = 256;
    let (histogram, hist_min, bin_width) = build_histogram(&valid, n_bins);
    let (best_bin, _) = find_optimal_threshold_bin(&histogram, valid.len() as f64);

    hist_min + (best_bin as f64 + 0.5) * bin_width
}

/// Compute linear regression slope (simple OLS).
pub(super) fn linear_slope(x: &[f64], y: &[f64]) -> f64 {
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

/// Statistics from amplitude envelope analysis (shared by Hilbert and wavelet methods).
pub(super) struct AmplitudeEnvelopeStats {
    pub(super) modulation_score: f64,
    pub(super) amplitude_trend: f64,
    pub(super) has_modulation: bool,
    pub(super) modulation_type: change::ModulationType,
    pub(super) _mean_amp: f64,
    pub(super) min_amp: f64,
    pub(super) max_amp: f64,
}

/// Analyze an amplitude envelope slice to compute modulation statistics.
pub(super) fn analyze_amplitude_envelope(
    interior_envelope: &[f64],
    interior_times: &[f64],
    modulation_threshold: f64,
) -> AmplitudeEnvelopeStats {
    let n_interior = interior_envelope.len() as f64;

    let mean_amp = interior_envelope.iter().sum::<f64>() / n_interior;
    let min_amp = interior_envelope
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_amp = interior_envelope
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    let variance = interior_envelope
        .iter()
        .map(|&a| (a - mean_amp).powi(2))
        .sum::<f64>()
        / n_interior;
    let std_amp = variance.sqrt();
    let modulation_score = if mean_amp > 1e-10 {
        std_amp / mean_amp
    } else {
        0.0
    };

    let t_mean = interior_times.iter().sum::<f64>() / n_interior;
    let mut cov_ta = 0.0;
    let mut var_t = 0.0;
    for (&t, &a) in interior_times.iter().zip(interior_envelope.iter()) {
        cov_ta += (t - t_mean) * (a - mean_amp);
        var_t += (t - t_mean).powi(2);
    }
    let slope = if var_t > 1e-10 { cov_ta / var_t } else { 0.0 };

    let time_span = interior_times.last().unwrap_or(&1.0) - interior_times.first().unwrap_or(&0.0);
    let amplitude_trend = if mean_amp > 1e-10 && time_span > 1e-10 {
        (slope * time_span / mean_amp).clamp(-1.0, 1.0)
    } else {
        0.0
    };

    let has_modulation = modulation_score > modulation_threshold;
    let modulation_type = if !has_modulation {
        change::ModulationType::Stable
    } else if amplitude_trend > 0.3 {
        change::ModulationType::Emerging
    } else if amplitude_trend < -0.3 {
        change::ModulationType::Fading
    } else {
        change::ModulationType::Oscillating
    };

    AmplitudeEnvelopeStats {
        modulation_score,
        amplitude_trend,
        has_modulation,
        modulation_type,
        _mean_amp: mean_amp,
        min_amp,
        max_amp,
    }
}

/// Fit a sinusoid at the given period, subtract it from residual, and return (a, b, amplitude, phase).
pub(super) fn fit_and_subtract_sinusoid(
    residual: &mut [f64],
    argvals: &[f64],
    period: f64,
) -> (f64, f64, f64, f64) {
    let m = residual.len();
    let omega = 2.0 * PI / period;
    let mut cos_sum = 0.0;
    let mut sin_sum = 0.0;

    for (j, &t) in argvals.iter().enumerate() {
        cos_sum += residual[j] * (omega * t).cos();
        sin_sum += residual[j] * (omega * t).sin();
    }

    let a = 2.0 * cos_sum / m as f64;
    let b = 2.0 * sin_sum / m as f64;
    let amplitude = (a * a + b * b).sqrt();
    let phase = b.atan2(a);

    for (j, &t) in argvals.iter().enumerate() {
        residual[j] -= a * (omega * t).cos() + b * (omega * t).sin();
    }

    (a, b, amplitude, phase)
}

/// Validate a single SAZED component: returns Some(period) if it passes range and confidence checks.
pub(super) fn validate_sazed_component(
    period: f64,
    confidence: f64,
    min_period: f64,
    max_period: f64,
    threshold: f64,
) -> Option<f64> {
    if period.is_finite() && period > min_period && period < max_period && confidence > threshold {
        Some(period)
    } else {
        None
    }
}

/// Count how many periods agree with a reference within tolerance, returning (count, sum).
pub(super) fn count_agreeing_periods(
    periods: &[f64],
    reference: f64,
    tolerance: f64,
) -> (usize, f64) {
    let mut count = 0;
    let mut sum = 0.0;
    for &p in periods {
        let rel_diff = (reference - p).abs() / reference.max(p);
        if rel_diff <= tolerance {
            count += 1;
            sum += p;
        }
    }
    (count, sum)
}

/// Find the end of the initial ACF descent (first negative or first uptick).
pub(super) fn find_acf_descent_end(acf: &[f64]) -> usize {
    for i in 1..acf.len() {
        if acf[i] < 0.0 {
            return i;
        }
        if i > 1 && acf[i] > acf[i - 1] {
            return i - 1;
        }
    }
    1
}

/// Find the first ACF peak after initial descent. Returns Some((lag, acf_value)).
pub(super) fn find_first_acf_peak(acf: &[f64]) -> Option<(usize, f64)> {
    if acf.len() < 4 {
        return None;
    }

    let min_search_start = find_acf_descent_end(acf);
    let peaks = find_peaks_1d(&acf[min_search_start..], 1);
    if peaks.is_empty() {
        return None;
    }

    let peak_lag = peaks[0] + min_search_start;
    Some((peak_lag, acf[peak_lag].max(0.0)))
}

/// Compute per-cycle seasonal strengths and identify weak seasons.
pub(super) fn compute_cycle_strengths(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    strength_thresh: f64,
) -> (Vec<f64>, Vec<usize>) {
    let (n, m) = data.shape();
    let t_start = argvals[0];
    let t_end = argvals[m - 1];
    let n_cycles = ((t_end - t_start) / period).floor() as usize;

    let mut cycle_strengths = Vec::with_capacity(n_cycles);
    let mut weak_seasons = Vec::new();

    for cycle in 0..n_cycles {
        let cycle_start = t_start + cycle as f64 * period;
        let cycle_end = cycle_start + period;

        let start_idx = argvals.iter().position(|&t| t >= cycle_start).unwrap_or(0);
        let end_idx = argvals.iter().position(|&t| t > cycle_end).unwrap_or(m);

        let cycle_m = end_idx - start_idx;
        if cycle_m < 4 {
            cycle_strengths.push(f64::NAN);
            continue;
        }

        let cycle_data: Vec<f64> = (start_idx..end_idx)
            .flat_map(|j| (0..n).map(move |i| data[(i, j)]))
            .collect();
        let cycle_mat = FdMatrix::from_column_major(cycle_data, n, cycle_m).unwrap();
        let cycle_argvals: Vec<f64> = argvals[start_idx..end_idx].to_vec();

        let strength_val =
            strength::seasonal_strength_variance(&cycle_mat, &cycle_argvals, period, 3);

        cycle_strengths.push(strength_val);
        if strength_val < strength_thresh {
            weak_seasons.push(cycle);
        }
    }

    (cycle_strengths, weak_seasons)
}

/// Build a histogram from valid values. Returns (histogram, min_val, bin_width).
pub(super) fn build_histogram(valid: &[f64], n_bins: usize) -> (Vec<usize>, f64, f64) {
    let min_val = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut histogram = vec![0usize; n_bins];
    for &v in valid {
        let bin = ((v - min_val) / bin_width).min(n_bins as f64 - 1.0) as usize;
        histogram[bin] += 1;
    }
    (histogram, min_val, bin_width)
}

/// Find the optimal threshold bin using Otsu's between-class variance. Returns (best_bin, best_variance).
pub(super) fn find_optimal_threshold_bin(histogram: &[usize], total: f64) -> (usize, f64) {
    let n_bins = histogram.len();
    let mut sum_total = 0.0;
    for (i, &count) in histogram.iter().enumerate() {
        sum_total += i as f64 * count as f64;
    }

    let mut best_bin = 0;
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
        let variance = weight_b * weight_f * (mean_b - mean_f).powi(2);
        if variance > best_variance {
            best_variance = variance;
            best_bin = t;
        }
    }

    (best_bin, best_variance)
}

/// Sum power at harmonics of a fundamental frequency within tolerance.
pub(super) fn sum_harmonic_power(
    frequencies: &[f64],
    power: &[f64],
    fundamental_freq: f64,
    tolerance: f64,
) -> (f64, f64) {
    let mut seasonal_power = 0.0;
    let mut total_power = 0.0;

    for (i, (&freq, &p)) in frequencies.iter().zip(power.iter()).enumerate() {
        if i == 0 {
            continue;
        }
        total_power += p;
        let ratio = freq / fundamental_freq;
        let nearest_harmonic = ratio.round();
        if (ratio - nearest_harmonic).abs() < tolerance && nearest_harmonic >= 1.0 {
            seasonal_power += p;
        }
    }

    (seasonal_power, total_power)
}

/// Return the new seasonal state if `ss` represents a valid threshold crossing,
/// or `None` if the index should be skipped (NaN, no change, or too close to the
/// previous change point).
pub(super) fn crossing_direction(
    ss: f64,
    threshold: f64,
    in_seasonal: bool,
    i: usize,
    last_change_idx: Option<usize>,
    min_dur_points: usize,
) -> Option<bool> {
    if ss.is_nan() {
        return None;
    }
    let now_seasonal = ss > threshold;
    if now_seasonal == in_seasonal {
        return None;
    }
    if last_change_idx.is_some_and(|last_idx| i - last_idx < min_dur_points) {
        return None;
    }
    Some(now_seasonal)
}

/// Build a `ChangePoint` for a threshold crossing at index `i`.
pub(super) fn build_change_point(
    i: usize,
    ss: f64,
    now_seasonal: bool,
    strength_curve: &[f64],
    argvals: &[f64],
) -> ChangePoint {
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
    ChangePoint {
        time: argvals[i],
        change_type,
        strength_before,
        strength_after: ss,
    }
}

/// Detect threshold crossings in a strength curve, returning change points.
pub(super) fn detect_threshold_crossings(
    strength_curve: &[f64],
    argvals: &[f64],
    threshold: f64,
    min_dur_points: usize,
) -> Vec<ChangePoint> {
    let mut change_points = Vec::new();
    let mut in_seasonal = strength_curve[0] > threshold;
    let mut last_change_idx: Option<usize> = None;

    for (i, &ss) in strength_curve.iter().enumerate().skip(1) {
        let Some(now_seasonal) = crossing_direction(
            ss,
            threshold,
            in_seasonal,
            i,
            last_change_idx,
            min_dur_points,
        ) else {
            continue;
        };

        change_points.push(build_change_point(
            i,
            ss,
            now_seasonal,
            strength_curve,
            argvals,
        ));

        in_seasonal = now_seasonal;
        last_change_idx = Some(i);
    }

    change_points
}

/// Find peaks in power spectrum above noise floor
pub(super) fn find_spectral_peaks(power: &[f64]) -> Vec<usize> {
    if power.len() < 3 {
        return Vec::new();
    }

    // Estimate noise floor as median power
    let mut sorted_power: Vec<f64> = power.to_vec();
    sorted_power.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_floor = sorted_power[sorted_power.len() / 2];
    let threshold = noise_floor * 2.0; // Peaks must be at least 2x median

    // Find all local maxima above threshold
    let mut peaks: Vec<(usize, f64)> = Vec::new();
    for i in 1..(power.len() - 1) {
        if power[i] > power[i - 1] && power[i] > power[i + 1] && power[i] > threshold {
            peaks.push((i, power[i]));
        }
    }

    // Sort by power (descending)
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    peaks.into_iter().map(|(idx, _)| idx).collect()
}

/// Find consensus period from multiple estimates using tolerance-based voting
pub(super) fn find_consensus_period(periods: &[f64], tolerance: f64) -> (f64, usize) {
    if periods.is_empty() {
        return (f64::NAN, 0);
    }
    if periods.len() == 1 {
        return (periods[0], 1);
    }

    let mut best_period = periods[0];
    let mut best_count = 0;
    let mut best_sum = 0.0;

    for &p1 in periods {
        let (count, sum) = count_agreeing_periods(periods, p1, tolerance);

        if count > best_count
            || (count == best_count && sum / count as f64 > best_sum / best_count.max(1) as f64)
        {
            best_count = count;
            best_period = sum / count as f64;
            best_sum = sum;
        }
    }

    (best_period, best_count)
}

/// Validate a candidate period using ACF
pub(super) fn validate_period_acf(acf: &[f64], period: f64, dt: f64) -> f64 {
    let lag = (period / dt).round() as usize;

    if lag == 0 || lag >= acf.len() {
        return 0.0;
    }

    // Score based on ACF value at the period lag
    // Positive ACF values indicate valid periodicity
    let acf_at_lag = acf[lag];

    // Also check harmonics (period/2, period*2) for consistency
    let half_lag = lag / 2;
    let double_lag = lag * 2;

    let mut score = acf_at_lag.max(0.0);

    // For a true period, ACF at half-period should be low/negative
    // and ACF at double-period should also be high
    if half_lag > 0 && half_lag < acf.len() {
        let half_acf = acf[half_lag];
        // Penalize if half-period has high ACF (suggests half-period is real)
        if half_acf > acf_at_lag * 0.7 {
            score *= 0.5;
        }
    }

    if double_lag < acf.len() {
        let double_acf = acf[double_lag];
        // Bonus if double-period also shows periodicity
        if double_acf > 0.3 {
            score *= 1.2;
        }
    }

    score.min(1.0)
}

/// Refine period estimate using gradient ascent on ACF
pub(super) fn refine_period_gradient(
    acf: &[f64],
    initial_period: f64,
    dt: f64,
    steps: usize,
) -> f64 {
    let mut period = initial_period;
    let step_size = dt * 0.5; // Search step size

    for _ in 0..steps {
        let current_score = validate_period_acf(acf, period, dt);
        let left_score = validate_period_acf(acf, period - step_size, dt);
        let right_score = validate_period_acf(acf, period + step_size, dt);

        if left_score > current_score && left_score > right_score {
            period -= step_size;
        } else if right_score > current_score {
            period += step_size;
        }
        // If current is best, we've converged
    }

    period.max(dt) // Ensure period is at least one time step
}

/// Cluster periods using a simple density-based approach
pub(super) fn cluster_periods(
    candidates: &[(f64, f64)],
    tolerance: f64,
    min_size: usize,
) -> Vec<(f64, f64)> {
    if candidates.is_empty() {
        return Vec::new();
    }

    // Sort candidates by period
    let mut sorted: Vec<(f64, f64)> = candidates.to_vec();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut clusters: Vec<(f64, f64)> = Vec::new(); // (center, total_power)
    let mut current_cluster: Vec<(f64, f64)> = vec![sorted[0]];

    for &(period, power) in sorted.iter().skip(1) {
        let cluster_center =
            current_cluster.iter().map(|(p, _)| p).sum::<f64>() / current_cluster.len() as f64;

        let rel_diff = (period - cluster_center).abs() / cluster_center.max(period);

        if rel_diff <= tolerance {
            // Add to current cluster
            current_cluster.push((period, power));
        } else {
            // Finish current cluster and start new one
            if current_cluster.len() >= min_size {
                let center = current_cluster.iter().map(|(p, pw)| p * pw).sum::<f64>()
                    / current_cluster
                        .iter()
                        .map(|(_, pw)| pw)
                        .sum::<f64>()
                        .max(1e-15);
                let total_power: f64 = current_cluster.iter().map(|(_, pw)| pw).sum();
                clusters.push((center, total_power));
            }
            current_cluster = vec![(period, power)];
        }
    }

    // Don't forget the last cluster
    if current_cluster.len() >= min_size {
        let center = current_cluster.iter().map(|(p, pw)| p * pw).sum::<f64>()
            / current_cluster
                .iter()
                .map(|(_, pw)| pw)
                .sum::<f64>()
                .max(1e-15);
        let total_power: f64 = current_cluster.iter().map(|(_, pw)| pw).sum();
        clusters.push((center, total_power));
    }

    clusters
}
