use std::f64::consts::PI;

use crate::matrix::FdMatrix;

use super::compute_mean_curve;

/// Result of Lomb-Scargle periodogram analysis.
#[derive(Debug, Clone)]
pub struct LombScargleResult {
    /// Test frequencies
    pub frequencies: Vec<f64>,
    /// Corresponding periods (1/frequency)
    pub periods: Vec<f64>,
    /// Normalized Lomb-Scargle power at each frequency
    pub power: Vec<f64>,
    /// Peak period (highest power)
    pub peak_period: f64,
    /// Peak frequency
    pub peak_frequency: f64,
    /// Peak power
    pub peak_power: f64,
    /// False alarm probability at peak (significance level)
    pub false_alarm_probability: f64,
    /// Significance level (1 - FAP)
    pub significance: f64,
}

/// Compute Lomb-Scargle periodogram for irregularly sampled data.
///
/// The Lomb-Scargle periodogram is designed for unevenly-spaced time series
/// and reduces to the standard periodogram for evenly-spaced data.
///
/// # Algorithm
/// Following Scargle (1982) and Horne & Baliunas (1986):
/// 1. For each test frequency omega, compute the phase shift tau
/// 2. Compute the normalized power P(omega)
/// 3. Estimate false alarm probability using the exponential distribution
///
/// # Arguments
/// * `times` - Observation times (not necessarily evenly spaced)
/// * `values` - Observed values at each time
/// * `frequencies` - Optional frequencies to evaluate (cycles per unit time).
///   If None, automatically generates a frequency grid.
/// * `oversampling` - Oversampling factor for auto-generated frequency grid.
///   Default: 4.0. Higher values give finer frequency resolution.
/// * `nyquist_factor` - Maximum frequency as multiple of pseudo-Nyquist.
///   Default: 1.0.
///
/// # Returns
/// `LombScargleResult` with power spectrum and significance estimates.
///
/// # Example
/// ```rust
/// use fdars_core::seasonal::lomb_scargle;
/// use std::f64::consts::PI;
///
/// // Irregularly sampled sine wave
/// let times: Vec<f64> = vec![0.0, 0.3, 0.7, 1.2, 1.5, 2.1, 2.8, 3.0, 3.5, 4.0];
/// let period = 1.5;
/// let values: Vec<f64> = times.iter()
///     .map(|&t| (2.0 * PI * t / period).sin())
///     .collect();
///
/// let result = lomb_scargle(&times, &values, None, None, None);
/// assert!((result.peak_period - period).abs() < 0.2);
/// ```
pub fn lomb_scargle(
    times: &[f64],
    values: &[f64],
    frequencies: Option<&[f64]>,
    oversampling: Option<f64>,
    nyquist_factor: Option<f64>,
) -> LombScargleResult {
    let n = times.len();
    if n != values.len() || n < 3 {
        return LombScargleResult {
            frequencies: vec![],
            periods: vec![],
            power: vec![],
            peak_period: f64::NAN,
            peak_frequency: f64::NAN,
            peak_power: f64::NAN,
            false_alarm_probability: f64::NAN,
            significance: f64::NAN,
        };
    }

    // Compute mean and variance
    let mean_y: f64 = values.iter().sum::<f64>() / n as f64;
    let var_y: f64 = values.iter().map(|&y| (y - mean_y).powi(2)).sum::<f64>() / (n - 1) as f64;

    // Generate frequency grid if not provided
    let freq_vec: Vec<f64>;
    let freqs = if let Some(f) = frequencies {
        f
    } else {
        freq_vec = generate_ls_frequencies(
            times,
            oversampling.unwrap_or(4.0),
            nyquist_factor.unwrap_or(1.0),
        );
        &freq_vec
    };

    // Compute Lomb-Scargle power at each frequency
    let mut power = Vec::with_capacity(freqs.len());

    for &freq in freqs.iter() {
        let omega = 2.0 * PI * freq;
        let p = lomb_scargle_single_freq(times, values, mean_y, var_y, omega);
        power.push(p);
    }

    // Find peak
    let (peak_idx, &peak_power) = power
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or((0, &0.0));

    let peak_frequency = freqs.get(peak_idx).copied().unwrap_or(0.0);
    let peak_period = if peak_frequency > 0.0 {
        1.0 / peak_frequency
    } else {
        f64::INFINITY
    };

    // Compute false alarm probability
    let n_indep = estimate_independent_frequencies(times, freqs.len());
    let fap = lomb_scargle_fap(peak_power, n_indep, n);

    // Compute periods from frequencies
    let periods: Vec<f64> = freqs
        .iter()
        .map(|&f| if f > 0.0 { 1.0 / f } else { f64::INFINITY })
        .collect();

    LombScargleResult {
        frequencies: freqs.to_vec(),
        periods,
        power,
        peak_period,
        peak_frequency,
        peak_power,
        false_alarm_probability: fap,
        significance: 1.0 - fap,
    }
}

/// Compute Lomb-Scargle power at a single frequency.
///
/// Uses the Scargle (1982) normalization.
pub(super) fn lomb_scargle_single_freq(
    times: &[f64],
    values: &[f64],
    mean_y: f64,
    var_y: f64,
    omega: f64,
) -> f64 {
    if var_y <= 0.0 || omega <= 0.0 {
        return 0.0;
    }

    let n = times.len();

    // Compute tau (phase shift) to make sine and cosine terms orthogonal
    let mut sum_sin2 = 0.0;
    let mut sum_cos2 = 0.0;
    for &t in times.iter() {
        let arg = 2.0 * omega * t;
        sum_sin2 += arg.sin();
        sum_cos2 += arg.cos();
    }
    let tau = (sum_sin2).atan2(sum_cos2) / (2.0 * omega);

    // Compute sums for power calculation
    let mut ss = 0.0; // Sum of sin terms
    let mut sc = 0.0; // Sum of cos terms
    let mut css = 0.0; // Sum of cos^2
    let mut sss = 0.0; // Sum of sin^2

    for i in 0..n {
        let y_centered = values[i] - mean_y;
        let arg = omega * (times[i] - tau);
        let c = arg.cos();
        let s = arg.sin();

        sc += y_centered * c;
        ss += y_centered * s;
        css += c * c;
        sss += s * s;
    }

    // Avoid division by zero
    let css = css.max(1e-15);
    let sss = sss.max(1e-15);

    // Lomb-Scargle power (Scargle 1982 normalization)
    0.5 * (sc * sc / css + ss * ss / sss) / var_y
}

/// Generate frequency grid for Lomb-Scargle.
///
/// The grid spans from 1/T_total to f_nyquist with oversampling.
pub(super) fn generate_ls_frequencies(
    times: &[f64],
    oversampling: f64,
    nyquist_factor: f64,
) -> Vec<f64> {
    let n = times.len();
    if n < 2 {
        return vec![0.0];
    }

    // Time span
    let t_min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let t_span = (t_max - t_min).max(1e-10);

    // Minimum frequency: one cycle over the observation span
    let f_min = 1.0 / t_span;

    // Pseudo-Nyquist frequency for irregular data
    // Use average sampling rate as approximation
    let f_nyquist = 0.5 * (n - 1) as f64 / t_span;

    // Maximum frequency
    let f_max = f_nyquist * nyquist_factor;

    // Frequency resolution with oversampling
    let df = f_min / oversampling;

    // Generate frequency grid
    let n_freq = ((f_max - f_min) / df).ceil() as usize + 1;
    let n_freq = n_freq.min(10000); // Cap to prevent memory issues

    (0..n_freq).map(|i| f_min + i as f64 * df).collect()
}

/// Estimate number of independent frequencies (for FAP calculation).
///
/// For irregularly sampled data, this is approximately the number of
/// data points (Horne & Baliunas 1986).
pub(super) fn estimate_independent_frequencies(times: &[f64], n_freq: usize) -> usize {
    // A conservative estimate is min(n_data, n_frequencies)
    let n = times.len();
    n.min(n_freq)
}

/// Compute false alarm probability for Lomb-Scargle peak.
///
/// Uses the exponential distribution approximation:
/// FAP = 1 - (1 - exp(-z))^M
/// where z is the power and M is the number of independent frequencies.
pub(super) fn lomb_scargle_fap(power: f64, n_indep: usize, _n_data: usize) -> f64 {
    if power <= 0.0 || n_indep == 0 {
        return 1.0;
    }

    // Probability that a single frequency has power < z
    let prob_single = 1.0 - (-power).exp();

    // Probability that all M frequencies have power < z
    // FAP = 1 - (1 - exp(-z))^M
    // For numerical stability, use log:
    // 1 - FAP = prob_single^M
    // FAP = 1 - exp(M * ln(prob_single))

    if prob_single >= 1.0 {
        return 0.0; // Very significant
    }
    if prob_single <= 0.0 {
        return 1.0; // Not significant
    }

    let log_prob = prob_single.ln();
    let log_cdf = n_indep as f64 * log_prob;

    if log_cdf < -700.0 {
        0.0 // Numerical underflow, very significant
    } else {
        1.0 - log_cdf.exp()
    }
}

/// Compute Lomb-Scargle periodogram for functional data (multiple curves).
///
/// Computes the periodogram for each curve and returns the result for the
/// mean curve or ensemble statistics.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m) of functional data
/// * `argvals` - Time points of length m
/// * `oversampling` - Oversampling factor. Default: 4.0
/// * `nyquist_factor` - Maximum frequency multiplier. Default: 1.0
///
/// # Returns
/// `LombScargleResult` computed from the mean curve.
pub fn lomb_scargle_fdata(
    data: &FdMatrix,
    argvals: &[f64],
    oversampling: Option<f64>,
    nyquist_factor: Option<f64>,
) -> LombScargleResult {
    // Compute mean curve
    let mean_curve = compute_mean_curve(data);

    // Run Lomb-Scargle on mean curve
    lomb_scargle(argvals, &mean_curve, None, oversampling, nyquist_factor)
}
