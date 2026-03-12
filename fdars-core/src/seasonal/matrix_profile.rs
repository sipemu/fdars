use crate::matrix::FdMatrix;

use super::compute_mean_curve;

/// Result of Matrix Profile computation.
#[derive(Debug, Clone)]
pub struct MatrixProfileResult {
    /// The matrix profile (minimum z-normalized distance for each position)
    pub profile: Vec<f64>,
    /// Index of the nearest neighbor for each position
    pub profile_index: Vec<usize>,
    /// Subsequence length used
    pub subsequence_length: usize,
    /// Detected periods from arc analysis
    pub detected_periods: Vec<f64>,
    /// Arc counts at each index distance (for period detection)
    pub arc_counts: Vec<usize>,
    /// Most prominent detected period
    pub primary_period: f64,
    /// Confidence score for primary period (based on arc prominence)
    pub confidence: f64,
}

/// Compute Matrix Profile using STOMP algorithm (Scalable Time series Ordered-search Matrix Profile).
///
/// The Matrix Profile is a data structure that stores the z-normalized Euclidean distance
/// between each subsequence of a time series and its nearest neighbor. It enables efficient
/// motif discovery and anomaly detection.
///
/// # Algorithm (STOMP - Zhu et al. 2016)
/// 1. Pre-compute sliding mean and standard deviation using cumulative sums
/// 2. Use FFT to compute first row of distance matrix
/// 3. Update subsequent rows incrementally using the dot product update rule
/// 4. Track minimum distance and index at each position
///
/// # Arguments
/// * `values` - Time series values
/// * `subsequence_length` - Length of subsequences to compare (window size)
/// * `exclusion_zone` - Fraction of subsequence length to exclude around each position
///   to prevent trivial self-matches. Default: 0.5
///
/// # Returns
/// `MatrixProfileResult` with profile, indices, and detected periods.
///
/// # Example
/// ```rust
/// use fdars_core::seasonal::matrix_profile;
/// use std::f64::consts::PI;
///
/// // Periodic signal
/// let period = 20.0;
/// let values: Vec<f64> = (0..100)
///     .map(|i| (2.0 * PI * i as f64 / period).sin())
///     .collect();
///
/// let result = matrix_profile(&values, Some(15), None);
/// assert!((result.primary_period - period).abs() < 5.0);
/// ```
pub fn matrix_profile(
    values: &[f64],
    subsequence_length: Option<usize>,
    exclusion_zone: Option<f64>,
) -> MatrixProfileResult {
    let n = values.len();

    // Default subsequence length: ~ 1/4 of series length, capped at reasonable range
    let m = subsequence_length.unwrap_or_else(|| {
        let default_m = n / 4;
        default_m.max(4).min(n / 2)
    });

    if m < 3 || m > n / 2 {
        return MatrixProfileResult {
            profile: vec![],
            profile_index: vec![],
            subsequence_length: m,
            detected_periods: vec![],
            arc_counts: vec![],
            primary_period: f64::NAN,
            confidence: 0.0,
        };
    }

    let exclusion_zone = exclusion_zone.unwrap_or(0.5);
    let exclusion_radius = (m as f64 * exclusion_zone).ceil() as usize;

    // Number of subsequences
    let profile_len = n - m + 1;

    // Compute sliding statistics
    let (means, stds) = compute_sliding_stats(values, m);

    // Compute the matrix profile using STOMP
    let (profile, profile_index) = stomp_core(values, m, &means, &stds, exclusion_radius);

    // Perform arc analysis to detect periods
    let (arc_counts, detected_periods, primary_period, confidence) =
        analyze_arcs(&profile_index, profile_len, m);

    MatrixProfileResult {
        profile,
        profile_index,
        subsequence_length: m,
        detected_periods,
        arc_counts,
        primary_period,
        confidence,
    }
}

/// Compute sliding mean and standard deviation using cumulative sums.
///
/// This is O(n) and avoids numerical issues with naive implementations.
fn compute_sliding_stats(values: &[f64], m: usize) -> (Vec<f64>, Vec<f64>) {
    let n = values.len();
    let profile_len = n - m + 1;

    // Compute cumulative sums
    let mut cumsum = vec![0.0; n + 1];
    let mut cumsum_sq = vec![0.0; n + 1];

    for i in 0..n {
        cumsum[i + 1] = cumsum[i] + values[i];
        cumsum_sq[i + 1] = cumsum_sq[i] + values[i] * values[i];
    }

    // Compute means and stds
    let mut means = Vec::with_capacity(profile_len);
    let mut stds = Vec::with_capacity(profile_len);

    let m_f64 = m as f64;

    for i in 0..profile_len {
        let sum = cumsum[i + m] - cumsum[i];
        let sum_sq = cumsum_sq[i + m] - cumsum_sq[i];

        let mean = sum / m_f64;
        let variance = (sum_sq / m_f64) - mean * mean;
        let std = variance.max(0.0).sqrt();

        means.push(mean);
        stds.push(std.max(1e-10)); // Prevent division by zero
    }

    (means, stds)
}

/// Core STOMP algorithm implementation.
///
/// Uses FFT for the first row and incremental updates for subsequent rows.
fn stomp_core(
    values: &[f64],
    m: usize,
    means: &[f64],
    stds: &[f64],
    exclusion_radius: usize,
) -> (Vec<f64>, Vec<usize>) {
    let n = values.len();
    let profile_len = n - m + 1;

    // Initialize profile with infinity and index with 0
    let mut profile = vec![f64::INFINITY; profile_len];
    let mut profile_index = vec![0usize; profile_len];

    // Compute first row using direct computation (could use FFT for large n)
    // QT[0,j] = sum(T[0:m] * T[j:j+m]) for each j
    let mut qt = vec![0.0; profile_len];

    // First query subsequence
    for j in 0..profile_len {
        let mut dot = 0.0;
        for k in 0..m {
            dot += values[k] * values[j + k];
        }
        qt[j] = dot;
    }

    // Process first row
    update_profile_row(
        0,
        &qt,
        means,
        stds,
        m,
        exclusion_radius,
        &mut profile,
        &mut profile_index,
    );

    // Process subsequent rows using incremental updates
    for i in 1..profile_len {
        // Update QT using the sliding dot product update
        // QT[i,j] = QT[i-1,j-1] - T[i-1]*T[j-1] + T[i+m-1]*T[j+m-1]
        let mut qt_new = vec![0.0; profile_len];

        // First element needs direct computation
        let mut dot = 0.0;
        for k in 0..m {
            dot += values[i + k] * values[k];
        }
        qt_new[0] = dot;

        // Update rest using incremental formula
        for j in 1..profile_len {
            qt_new[j] =
                qt[j - 1] - values[i - 1] * values[j - 1] + values[i + m - 1] * values[j + m - 1];
        }

        qt = qt_new;

        // Update profile with this row
        update_profile_row(
            i,
            &qt,
            means,
            stds,
            m,
            exclusion_radius,
            &mut profile,
            &mut profile_index,
        );
    }

    (profile, profile_index)
}

/// Update profile with distances from row i.
fn update_profile_row(
    i: usize,
    qt: &[f64],
    means: &[f64],
    stds: &[f64],
    m: usize,
    exclusion_radius: usize,
    profile: &mut [f64],
    profile_index: &mut [usize],
) {
    let profile_len = profile.len();
    let m_f64 = m as f64;

    for j in 0..profile_len {
        // Skip exclusion zone
        if i.abs_diff(j) <= exclusion_radius {
            continue;
        }

        // Compute z-normalized distance
        // d = sqrt(2*m * (1 - (QT - m*mu_i*mu_j) / (m * sigma_i * sigma_j)))
        let numerator = qt[j] - m_f64 * means[i] * means[j];
        let denominator = m_f64 * stds[i] * stds[j];

        let pearson = if denominator > 0.0 {
            (numerator / denominator).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        let dist_sq = 2.0 * m_f64 * (1.0 - pearson);
        let dist = dist_sq.max(0.0).sqrt();

        // Update profile for position i
        if dist < profile[i] {
            profile[i] = dist;
            profile_index[i] = j;
        }

        // Update profile for position j (symmetric)
        if dist < profile[j] {
            profile[j] = dist;
            profile_index[j] = i;
        }
    }
}

/// Analyze profile index to detect periods using arc counting.
///
/// Arcs connect each position to its nearest neighbor. The distance between
/// connected positions reveals repeating patterns (periods).
fn analyze_arcs(
    profile_index: &[usize],
    profile_len: usize,
    m: usize,
) -> (Vec<usize>, Vec<f64>, f64, f64) {
    // Count arcs at each index distance
    let max_distance = profile_len;
    let mut arc_counts = vec![0usize; max_distance];

    for (i, &j) in profile_index.iter().enumerate() {
        let distance = i.abs_diff(j);
        if distance < max_distance {
            arc_counts[distance] += 1;
        }
    }

    // Find peaks in arc counts (candidate periods)
    let min_period = m / 2; // Minimum meaningful period
    let mut peaks: Vec<(usize, usize)> = Vec::new();

    // Simple peak detection with minimum spacing
    for i in min_period..arc_counts.len().saturating_sub(1) {
        if arc_counts[i] > arc_counts[i.saturating_sub(1)]
            && arc_counts[i] > arc_counts[(i + 1).min(arc_counts.len() - 1)]
            && arc_counts[i] >= 3
        // Minimum count threshold
        {
            peaks.push((i, arc_counts[i]));
        }
    }

    // Sort by count descending
    peaks.sort_by(|a, b| b.1.cmp(&a.1));

    // Extract top periods
    let detected_periods: Vec<f64> = peaks.iter().take(5).map(|(p, _)| *p as f64).collect();

    // Primary period and confidence
    let (primary_period, confidence) = if let Some(&(period, count)) = peaks.first() {
        // Confidence based on relative peak prominence
        let total_arcs: usize = arc_counts[min_period..].iter().sum();
        let conf = if total_arcs > 0 {
            count as f64 / total_arcs as f64
        } else {
            0.0
        };
        (period as f64, conf.min(1.0))
    } else {
        (0.0, 0.0)
    };

    (arc_counts, detected_periods, primary_period, confidence)
}

/// Compute Matrix Profile for functional data (multiple curves).
///
/// Computes the matrix profile for each curve and returns aggregated results.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m) of functional data
/// * `subsequence_length` - Length of subsequences. If None, automatically determined.
/// * `exclusion_zone` - Exclusion zone fraction. Default: 0.5
///
/// # Returns
/// `MatrixProfileResult` computed from the mean curve.
pub fn matrix_profile_fdata(
    data: &FdMatrix,
    subsequence_length: Option<usize>,
    exclusion_zone: Option<f64>,
) -> MatrixProfileResult {
    // Compute mean curve
    let mean_curve = compute_mean_curve(data);

    // Run matrix profile on mean curve
    matrix_profile(&mean_curve, subsequence_length, exclusion_zone)
}

/// Detect seasonality using Matrix Profile analysis.
///
/// Returns true if significant periodicity is detected based on matrix profile analysis.
///
/// # Arguments
/// * `values` - Time series values
/// * `subsequence_length` - Length of subsequences to compare
/// * `confidence_threshold` - Minimum confidence for positive detection. Default: 0.1
///
/// # Returns
/// Tuple of (is_seasonal, detected_period, confidence)
pub fn matrix_profile_seasonality(
    values: &[f64],
    subsequence_length: Option<usize>,
    confidence_threshold: Option<f64>,
) -> (bool, f64, f64) {
    let result = matrix_profile(values, subsequence_length, None);

    let threshold = confidence_threshold.unwrap_or(0.1);
    let is_seasonal = result.confidence >= threshold && result.primary_period > 0.0;

    (is_seasonal, result.primary_period, result.confidence)
}
