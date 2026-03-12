use crate::matrix::FdMatrix;

use super::{
    autocorrelation, cluster_periods, compute_mean_curve, find_spectral_peaks, periodogram,
    refine_period_gradient, validate_period_acf,
};

/// Result of Autoperiod detection.
#[derive(Debug, Clone)]
pub struct AutoperiodResult {
    /// Detected period
    pub period: f64,
    /// Combined confidence (FFT * ACF validation)
    pub confidence: f64,
    /// FFT power at the detected period
    pub fft_power: f64,
    /// ACF validation score (0-1)
    pub acf_validation: f64,
    /// All candidate periods considered
    pub candidates: Vec<AutoperiodCandidate>,
}

/// A candidate period from Autoperiod detection.
#[derive(Debug, Clone)]
pub struct AutoperiodCandidate {
    /// Candidate period
    pub period: f64,
    /// FFT power
    pub fft_power: f64,
    /// ACF validation score
    pub acf_score: f64,
    /// Combined score (power * validation)
    pub combined_score: f64,
}

fn empty_autoperiod_result() -> AutoperiodResult {
    AutoperiodResult {
        period: f64::NAN,
        confidence: 0.0,
        fft_power: 0.0,
        acf_validation: 0.0,
        candidates: Vec::new(),
    }
}

/// Build an autoperiod candidate from a spectral peak, refining with gradient ascent on ACF.
fn build_autoperiod_candidate(
    peak_idx: usize,
    frequencies: &[f64],
    power_no_dc: &[f64],
    acf: &[f64],
    dt: f64,
    steps: usize,
    total_power: f64,
) -> Option<AutoperiodCandidate> {
    let freq = frequencies[peak_idx + 1];
    if freq < 1e-15 {
        return None;
    }
    let fft_power = power_no_dc[peak_idx];
    let normalized_power = fft_power / total_power.max(1e-15);
    let refined_period = refine_period_gradient(acf, 1.0 / freq, dt, steps);
    let refined_acf_score = validate_period_acf(acf, refined_period, dt);
    Some(AutoperiodCandidate {
        period: refined_period,
        fft_power,
        acf_score: refined_acf_score,
        combined_score: normalized_power * refined_acf_score,
    })
}

/// Autoperiod: Hybrid FFT + ACF Period Detection
///
/// Implements the Autoperiod algorithm (Vlachos et al. 2005) which:
/// 1. Computes the periodogram via FFT to find candidate periods
/// 2. Validates each candidate using the autocorrelation function
/// 3. Applies gradient ascent to refine the period estimate
/// 4. Returns the period with the highest combined confidence
///
/// This method is more robust than pure FFT because ACF validation
/// filters out spurious spectral peaks that don't correspond to
/// true periodicity.
///
/// # Arguments
/// * `data` - Input signal (1D time series)
/// * `argvals` - Time points corresponding to data
/// * `n_candidates` - Maximum number of FFT peaks to consider (default: 5)
/// * `gradient_steps` - Number of gradient ascent refinement steps (default: 10)
///
/// # Returns
/// * `AutoperiodResult` containing the best period and validation details
pub fn autoperiod(
    data: &[f64],
    argvals: &[f64],
    n_candidates: Option<usize>,
    gradient_steps: Option<usize>,
) -> AutoperiodResult {
    let n = data.len();
    let max_candidates = n_candidates.unwrap_or(5);
    let steps = gradient_steps.unwrap_or(10);

    if n < 8 || argvals.len() != n {
        return empty_autoperiod_result();
    }

    let dt = (argvals[n - 1] - argvals[0]) / (n - 1) as f64;
    let max_lag = (n / 2).max(4);

    // Step 1: Compute periodogram and find candidate periods
    let (frequencies, power) = periodogram(data, argvals);

    if frequencies.len() < 3 {
        return empty_autoperiod_result();
    }

    // Find top spectral peaks
    let power_no_dc: Vec<f64> = power.iter().skip(1).copied().collect();
    let peak_indices = find_spectral_peaks(&power_no_dc);

    if peak_indices.is_empty() {
        return empty_autoperiod_result();
    }

    // Step 2: Compute ACF for validation
    let acf = autocorrelation(data, max_lag);

    // Step 3: Validate each candidate and refine with gradient ascent
    let total_power: f64 = power_no_dc.iter().sum();
    let candidates: Vec<AutoperiodCandidate> = peak_indices
        .iter()
        .take(max_candidates)
        .filter_map(|&peak_idx| {
            build_autoperiod_candidate(
                peak_idx,
                &frequencies,
                &power_no_dc,
                &acf,
                dt,
                steps,
                total_power,
            )
        })
        .collect();

    if candidates.is_empty() {
        return empty_autoperiod_result();
    }

    // Select best candidate based on combined score
    let best = candidates
        .iter()
        .max_by(|a, b| {
            a.combined_score
                .partial_cmp(&b.combined_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    AutoperiodResult {
        period: best.period,
        confidence: best.combined_score,
        fft_power: best.fft_power,
        acf_validation: best.acf_score,
        candidates,
    }
}

/// Autoperiod for functional data (matrix format)
pub fn autoperiod_fdata(
    data: &FdMatrix,
    argvals: &[f64],
    n_candidates: Option<usize>,
    gradient_steps: Option<usize>,
) -> AutoperiodResult {
    let (n, m) = data.shape();
    if n == 0 || m < 8 || argvals.len() != m {
        return AutoperiodResult {
            period: f64::NAN,
            confidence: 0.0,
            fft_power: 0.0,
            acf_validation: 0.0,
            candidates: Vec::new(),
        };
    }

    let mean_curve = compute_mean_curve(data);
    autoperiod(&mean_curve, argvals, n_candidates, gradient_steps)
}

/// Result of CFDAutoperiod detection.
#[derive(Debug, Clone)]
pub struct CfdAutoperiodResult {
    /// Detected period (primary)
    pub period: f64,
    /// Confidence score
    pub confidence: f64,
    /// ACF validation score for the primary period
    pub acf_validation: f64,
    /// All detected periods (cluster centers)
    pub periods: Vec<f64>,
    /// Confidence for each detected period
    pub confidences: Vec<f64>,
}

/// Convert spectral peak indices to candidate (period, normalized_power) pairs.
fn generate_cfd_candidates(
    frequencies: &[f64],
    power_no_dc: &[f64],
    peak_indices: &[usize],
) -> Vec<(f64, f64)> {
    let total_power: f64 = power_no_dc.iter().sum();
    peak_indices
        .iter()
        .filter_map(|&peak_idx| {
            let freq = frequencies[peak_idx + 1];
            if freq > 1e-15 {
                let period = 1.0 / freq;
                let normalized_power = power_no_dc[peak_idx] / total_power.max(1e-15);
                Some((period, normalized_power))
            } else {
                None
            }
        })
        .collect()
}

/// Validate clustered period candidates using ACF, returning (period, acf_score, power) triples.
pub(super) fn validate_cfd_candidates(
    clusters: &[(f64, f64)],
    acf: &[f64],
    dt: f64,
) -> Vec<(f64, f64, f64)> {
    clusters
        .iter()
        .filter_map(|&(center, power_sum)| {
            let acf_score = validate_period_acf(acf, center, dt);
            if acf_score > 0.1 {
                Some((center, acf_score, power_sum))
            } else {
                None
            }
        })
        .collect()
}

/// Validate cluster candidates with ACF, falling back to best cluster if none pass.
pub(super) fn validate_or_fallback_cfd(
    validated: Vec<(f64, f64, f64)>,
    candidates: &[(f64, f64)],
    tol: f64,
    min_size: usize,
) -> Vec<(f64, f64, f64)> {
    if !validated.is_empty() {
        return validated;
    }
    // Fallback: pick highest-power cluster without ACF validation
    cluster_periods(candidates, tol, min_size)
        .into_iter()
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(center, power_sum)| vec![(center, 0.0, power_sum)])
        .unwrap_or_default()
}

/// Rank validated results by combined score (acf * power).
/// Returns (periods, confidences, top_acf_validation).
pub(super) fn rank_cfd_results(validated: &[(f64, f64, f64)]) -> (Vec<f64>, Vec<f64>, f64) {
    let mut sorted: Vec<_> = validated.to_vec();
    sorted.sort_by(|a, b| {
        (b.1 * b.2)
            .partial_cmp(&(a.1 * a.2))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let top_acf = sorted[0].1;
    let periods = sorted.iter().map(|v| v.0).collect();
    let confidences = sorted.iter().map(|v| v.1 * v.2).collect();
    (periods, confidences, top_acf)
}

pub(super) fn empty_cfd_result() -> CfdAutoperiodResult {
    CfdAutoperiodResult {
        period: f64::NAN,
        confidence: 0.0,
        acf_validation: 0.0,
        periods: Vec::new(),
        confidences: Vec::new(),
    }
}

/// Extract spectral candidates from differenced data: difference, periodogram, peak-find, generate.
fn extract_cfd_spectral_candidates(data: &[f64], argvals: &[f64]) -> Option<Vec<(f64, f64)>> {
    let n = data.len();
    // Linear detrending instead of first-order differencing.
    // Differencing is a high-pass filter that attenuates long-period signals
    // by (2π/period)², making them undetectable. Linear detrending removes
    // the DC/trend component without any frequency-dependent distortion.
    let mean_x: f64 = argvals.iter().sum::<f64>() / n as f64;
    let mean_y: f64 = data.iter().sum::<f64>() / n as f64;
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..n {
        let dx = argvals[i] - mean_x;
        num += dx * (data[i] - mean_y);
        den += dx * dx;
    }
    let slope = if den.abs() > 1e-15 { num / den } else { 0.0 };
    let detrended: Vec<f64> = data
        .iter()
        .zip(argvals.iter())
        .map(|(&y, &x)| y - (mean_y + slope * (x - mean_x)))
        .collect();
    let (frequencies, power) = periodogram(&detrended, argvals);
    if frequencies.len() < 3 {
        return None;
    }
    let power_no_dc: Vec<f64> = power.iter().skip(1).copied().collect();
    let peak_indices = find_spectral_peaks(&power_no_dc);
    if peak_indices.is_empty() {
        return None;
    }
    let candidates = generate_cfd_candidates(&frequencies, &power_no_dc, &peak_indices);
    if candidates.is_empty() {
        None
    } else {
        Some(candidates)
    }
}

/// CFDAutoperiod: Clustered Filtered Detrended Autoperiod
///
/// Implements the CFDAutoperiod algorithm (Puech et al. 2020) which:
/// 1. Applies first-order differencing to remove trends
/// 2. Computes FFT on the detrended signal
/// 3. Identifies candidate periods from periodogram peaks
/// 4. Clusters nearby candidates using density-based clustering
/// 5. Validates cluster centers using ACF on the original signal
///
/// This method is particularly effective for signals with strong trends
/// and handles multiple periodicities by detecting clusters of candidate periods.
///
/// # Arguments
/// * `data` - Input signal (1D time series)
/// * `argvals` - Time points corresponding to data
/// * `cluster_tolerance` - Relative tolerance for clustering periods (default: 0.1 = 10%)
/// * `min_cluster_size` - Minimum number of candidates to form a cluster (default: 1)
///
/// # Returns
/// * `CfdAutoperiodResult` containing detected periods and validation scores
pub fn cfd_autoperiod(
    data: &[f64],
    argvals: &[f64],
    cluster_tolerance: Option<f64>,
    min_cluster_size: Option<usize>,
) -> CfdAutoperiodResult {
    let n = data.len();
    let tol = cluster_tolerance.unwrap_or(0.1);
    let min_size = min_cluster_size.unwrap_or(1);

    if n < 8 || argvals.len() != n {
        return empty_cfd_result();
    }

    let dt = (argvals[n - 1] - argvals[0]) / (n - 1) as f64;
    let max_lag = (n / 2).max(4);

    let Some(candidates) = extract_cfd_spectral_candidates(data, argvals) else {
        return empty_cfd_result();
    };

    let clusters = cluster_periods(&candidates, tol, min_size);
    if clusters.is_empty() {
        return empty_cfd_result();
    }

    let acf = autocorrelation(data, max_lag);
    let validated = validate_cfd_candidates(&clusters, &acf, dt);
    let validated = validate_or_fallback_cfd(validated, &candidates, tol, min_size);
    let (periods, confidences, top_acf) = rank_cfd_results(&validated);

    CfdAutoperiodResult {
        period: periods[0],
        confidence: confidences[0],
        acf_validation: top_acf,
        periods,
        confidences,
    }
}

/// CFDAutoperiod for functional data (matrix format)
pub fn cfd_autoperiod_fdata(
    data: &FdMatrix,
    argvals: &[f64],
    cluster_tolerance: Option<f64>,
    min_cluster_size: Option<usize>,
) -> CfdAutoperiodResult {
    let (n, m) = data.shape();
    if n == 0 || m < 8 || argvals.len() != m {
        return CfdAutoperiodResult {
            period: f64::NAN,
            confidence: 0.0,
            acf_validation: 0.0,
            periods: Vec::new(),
            confidences: Vec::new(),
        };
    }

    let mean_curve = compute_mean_curve(data);
    cfd_autoperiod(&mean_curve, argvals, cluster_tolerance, min_cluster_size)
}
