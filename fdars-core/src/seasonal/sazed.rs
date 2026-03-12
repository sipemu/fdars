use crate::matrix::FdMatrix;

use super::{
    autocorrelation, compute_mean_curve, find_consensus_period, find_first_acf_peak, find_peaks_1d,
    periodogram, validate_sazed_component,
};

/// Result of SAZED ensemble period detection.
#[derive(Debug, Clone)]
pub struct SazedResult {
    /// Primary detected period (consensus from ensemble)
    pub period: f64,
    /// Confidence score (0-1, based on component agreement)
    pub confidence: f64,
    /// Periods detected by each component (may be NaN if not detected)
    pub component_periods: SazedComponents,
    /// Number of components that agreed on the final period
    pub agreeing_components: usize,
}

/// Individual period estimates from each SAZED component.
#[derive(Debug, Clone)]
pub struct SazedComponents {
    /// Period from spectral (FFT) detection
    pub spectral: f64,
    /// Period from ACF peak detection
    pub acf_peak: f64,
    /// Period from weighted ACF average
    pub acf_average: f64,
    /// Period from ACF zero-crossing analysis
    pub zero_crossing: f64,
    /// Period from spectral differencing
    pub spectral_diff: f64,
}

/// SAZED: Spectral-ACF Zero-crossing Ensemble Detection
///
/// A parameter-free ensemble method for robust period detection.
/// Combines 5 detection components:
/// 1. Spectral (FFT) - peaks in periodogram
/// 2. ACF peak - first significant peak in autocorrelation
/// 3. ACF average - weighted mean of ACF peaks
/// 4. Zero-crossing - period from ACF zero crossings
/// 5. Spectral differencing - FFT on first-differenced signal
///
/// Each component provides both a period estimate and a confidence score.
/// Only components with sufficient confidence participate in voting.
/// The final period is chosen by majority voting with tolerance.
///
/// # Arguments
/// * `data` - Input signal (1D time series or mean curve from fdata)
/// * `argvals` - Time points corresponding to data
/// * `tolerance` - Relative tolerance for considering periods equal (default: 0.05 = 5%)
///
/// # Returns
/// * `SazedResult` containing the consensus period and component details
pub fn sazed(data: &[f64], argvals: &[f64], tolerance: Option<f64>) -> SazedResult {
    let n = data.len();
    let tol = tolerance.unwrap_or(0.05); // Tighter default tolerance

    if n < 8 || argvals.len() != n {
        return SazedResult {
            period: f64::NAN,
            confidence: 0.0,
            component_periods: SazedComponents {
                spectral: f64::NAN,
                acf_peak: f64::NAN,
                acf_average: f64::NAN,
                zero_crossing: f64::NAN,
                spectral_diff: f64::NAN,
            },
            agreeing_components: 0,
        };
    }

    let dt = (argvals[n - 1] - argvals[0]) / (n - 1) as f64;
    let max_lag = (n / 2).max(4);
    let signal_range = argvals[n - 1] - argvals[0];

    // Minimum detectable period (at least 3 cycles)
    let min_period = signal_range / (n as f64 / 3.0);
    // Maximum detectable period (at most 2 complete cycles)
    let max_period = signal_range / 2.0;

    // Component 1: Spectral (FFT) detection with confidence
    let (spectral_period, spectral_conf) = sazed_spectral_with_confidence(data, argvals);

    // Component 2: ACF peak detection with confidence
    let (acf_peak_period, acf_peak_conf) = sazed_acf_peak_with_confidence(data, dt, max_lag);

    // Component 3: ACF weighted average (uses ACF peak confidence)
    let acf_average_period = sazed_acf_average(data, dt, max_lag);

    // Component 4: Zero-crossing analysis with confidence
    let (zero_crossing_period, zero_crossing_conf) =
        sazed_zero_crossing_with_confidence(data, dt, max_lag);

    // Component 5: Spectral on differenced signal with confidence
    let (spectral_diff_period, spectral_diff_conf) =
        sazed_spectral_diff_with_confidence(data, argvals);

    let components = SazedComponents {
        spectral: spectral_period,
        acf_peak: acf_peak_period,
        acf_average: acf_average_period,
        zero_crossing: zero_crossing_period,
        spectral_diff: spectral_diff_period,
    };

    // Confidence thresholds for each component (tuned to minimize FPR on noise)
    // For Gaussian noise: spectral peaks rarely exceed 6x median, ACF ~1/sqrt(n)
    let spectral_thresh = 8.0; // Power ratio must be > 8x median (noise rarely exceeds 6x)
    let acf_thresh = 0.3; // ACF correlation must be > 0.3 (noise ~0.1 for n=100)
    let zero_crossing_thresh = 0.9; // Zero-crossing consistency > 90%
    let spectral_diff_thresh = 6.0; // Diff spectral power ratio > 6x

    // Minimum number of confident components required to report a period
    let min_confident_components = 2;

    // Collect valid periods (only from components with sufficient confidence)
    let confident_periods: Vec<f64> = [
        validate_sazed_component(
            spectral_period,
            spectral_conf,
            min_period,
            max_period,
            spectral_thresh,
        ),
        validate_sazed_component(
            acf_peak_period,
            acf_peak_conf,
            min_period,
            max_period,
            acf_thresh,
        ),
        validate_sazed_component(
            acf_average_period,
            acf_peak_conf,
            min_period,
            max_period,
            acf_thresh,
        ),
        validate_sazed_component(
            zero_crossing_period,
            zero_crossing_conf,
            min_period,
            max_period,
            zero_crossing_thresh,
        ),
        validate_sazed_component(
            spectral_diff_period,
            spectral_diff_conf,
            min_period,
            max_period,
            spectral_diff_thresh,
        ),
    ]
    .into_iter()
    .flatten()
    .collect();

    // Require minimum number of confident components before reporting a period
    if confident_periods.len() < min_confident_components {
        return SazedResult {
            period: f64::NAN,
            confidence: 0.0,
            component_periods: components,
            agreeing_components: confident_periods.len(),
        };
    }

    // Ensemble voting: find the mode with tolerance
    let (consensus_period, agreeing_count) = find_consensus_period(&confident_periods, tol);
    let confidence = agreeing_count as f64 / 5.0;

    SazedResult {
        period: consensus_period,
        confidence,
        component_periods: components,
        agreeing_components: agreeing_count,
    }
}

/// SAZED for functional data (matrix format)
///
/// Computes mean curve first, then applies SAZED.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `tolerance` - Relative tolerance for period matching
pub fn sazed_fdata(data: &FdMatrix, argvals: &[f64], tolerance: Option<f64>) -> SazedResult {
    let (n, m) = data.shape();
    if n == 0 || m < 8 || argvals.len() != m {
        return SazedResult {
            period: f64::NAN,
            confidence: 0.0,
            component_periods: SazedComponents {
                spectral: f64::NAN,
                acf_peak: f64::NAN,
                acf_average: f64::NAN,
                zero_crossing: f64::NAN,
                spectral_diff: f64::NAN,
            },
            agreeing_components: 0,
        };
    }

    let mean_curve = compute_mean_curve(data);
    sazed(&mean_curve, argvals, tolerance)
}

/// Spectral component with confidence: returns (period, power_ratio)
fn sazed_spectral_with_confidence(data: &[f64], argvals: &[f64]) -> (f64, f64) {
    let (frequencies, power) = periodogram(data, argvals);

    if frequencies.len() < 3 {
        return (f64::NAN, 0.0);
    }

    // Find peaks in power spectrum (skip DC)
    let power_no_dc: Vec<f64> = power.iter().skip(1).copied().collect();

    if power_no_dc.is_empty() {
        return (f64::NAN, 0.0);
    }

    // Calculate noise floor as median
    let mut sorted_power = power_no_dc.clone();
    sorted_power.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let noise_floor = sorted_power[sorted_power.len() / 2].max(1e-15);

    // Find global maximum
    let mut max_idx = 0;
    let mut max_val = 0.0;
    for (i, &p) in power_no_dc.iter().enumerate() {
        if p > max_val {
            max_val = p;
            max_idx = i;
        }
    }

    let power_ratio = max_val / noise_floor;
    let freq = frequencies[max_idx + 1];

    if freq > 1e-15 {
        (1.0 / freq, power_ratio)
    } else {
        (f64::NAN, 0.0)
    }
}

/// ACF peak component with confidence: returns (period, acf_value_at_peak)
fn sazed_acf_peak_with_confidence(data: &[f64], dt: f64, max_lag: usize) -> (f64, f64) {
    let acf = autocorrelation(data, max_lag);

    match find_first_acf_peak(&acf) {
        Some((peak_lag, acf_value)) => (peak_lag as f64 * dt, acf_value),
        None => (f64::NAN, 0.0),
    }
}

/// ACF average component: weighted mean of ACF peak locations
fn sazed_acf_average(data: &[f64], dt: f64, max_lag: usize) -> f64 {
    let acf = autocorrelation(data, max_lag);

    if acf.len() < 4 {
        return f64::NAN;
    }

    // Find all peaks in ACF
    let peaks = find_peaks_1d(&acf[1..], 1);

    if peaks.is_empty() {
        return f64::NAN;
    }

    // Weight peaks by their ACF value
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;

    for (i, &peak_idx) in peaks.iter().enumerate() {
        let lag = peak_idx + 1;
        let weight = acf[lag].max(0.0);

        if i == 0 {
            // First peak is the fundamental period
            weighted_sum += lag as f64 * weight;
            weight_sum += weight;
        } else {
            // Later peaks: estimate fundamental by dividing by harmonic number
            let expected_fundamental = peaks[0] + 1;
            let harmonic = ((lag as f64 / expected_fundamental as f64) + 0.5) as usize;
            if harmonic > 0 {
                let fundamental_est = lag as f64 / harmonic as f64;
                weighted_sum += fundamental_est * weight;
                weight_sum += weight;
            }
        }
    }

    if weight_sum > 1e-15 {
        weighted_sum / weight_sum * dt
    } else {
        f64::NAN
    }
}

/// Zero-crossing component with confidence: returns (period, consistency)
/// Consistency is how regular the zero crossings are (std/mean of half-periods)
fn sazed_zero_crossing_with_confidence(data: &[f64], dt: f64, max_lag: usize) -> (f64, f64) {
    let acf = autocorrelation(data, max_lag);

    if acf.len() < 4 {
        return (f64::NAN, 0.0);
    }

    // Find zero crossings (sign changes)
    let mut crossings = Vec::new();
    for i in 1..acf.len() {
        if acf[i - 1] * acf[i] < 0.0 {
            // Linear interpolation for more precise crossing
            let frac = acf[i - 1].abs() / (acf[i - 1].abs() + acf[i].abs());
            crossings.push((i - 1) as f64 + frac);
        }
    }

    if crossings.len() < 2 {
        return (f64::NAN, 0.0);
    }

    // Period is twice the distance between consecutive zero crossings
    // (ACF goes through two zero crossings per period)
    let mut half_periods = Vec::new();
    for i in 1..crossings.len() {
        half_periods.push(crossings[i] - crossings[i - 1]);
    }

    if half_periods.is_empty() {
        return (f64::NAN, 0.0);
    }

    // Calculate consistency: 1 - (std/mean) of half-periods
    // High consistency means regular zero crossings
    let mean: f64 = half_periods.iter().sum::<f64>() / half_periods.len() as f64;
    let variance: f64 =
        half_periods.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / half_periods.len() as f64;
    let std = variance.sqrt();
    let consistency = (1.0 - std / mean.max(1e-15)).clamp(0.0, 1.0);

    // Median half-period
    half_periods.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median_half = half_periods[half_periods.len() / 2];

    (2.0 * median_half * dt, consistency)
}

/// Spectral differencing with confidence: returns (period, power_ratio)
fn sazed_spectral_diff_with_confidence(data: &[f64], argvals: &[f64]) -> (f64, f64) {
    if data.len() < 4 {
        return (f64::NAN, 0.0);
    }

    // First difference to remove trend
    let diff: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    let diff_argvals: Vec<f64> = argvals.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

    sazed_spectral_with_confidence(&diff, &diff_argvals)
}
