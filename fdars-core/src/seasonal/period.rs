use super::strength::seasonal_strength_variance;
use super::{
    compute_mean_curve, find_peaks_1d, fit_and_subtract_sinusoid, periodogram, DetectedPeriod,
    PeriodEstimate,
};
use crate::basis::fourier_basis_with_period;
use crate::matrix::FdMatrix;
use crate::slice_maybe_parallel;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Estimate period using FFT periodogram.
///
/// Finds the dominant frequency in the periodogram (excluding DC) and
/// returns the corresponding period.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points (time values)
///
/// # Returns
/// Period estimate with confidence measure
pub fn estimate_period_fft(data: &FdMatrix, argvals: &[f64]) -> PeriodEstimate {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m {
        return PeriodEstimate {
            period: f64::NAN,
            frequency: f64::NAN,
            power: 0.0,
            confidence: 0.0,
        };
    }

    // Compute mean curve first
    let mean_curve = compute_mean_curve(data);

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
/// * `data` - Column-major matrix (n x m) as flat slice
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
    let mat = FdMatrix::from_slice(data, n, m).unwrap();
    let mean_curve = compute_mean_curve(&mat);

    let acf = super::autocorrelation(&mean_curve, max_lag);

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
/// * `data` - Column-major matrix (n x m) as flat slice
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
    let mat = FdMatrix::from_slice(data, n, m).unwrap();
    let mean_curve = compute_mean_curve(&mat);

    let nbasis = 1 + 2 * n_harmonics;

    // Grid search over candidate periods
    let candidates: Vec<f64> = (0..n_candidates)
        .map(|i| period_min + (period_max - period_min) * i as f64 / (n_candidates - 1) as f64)
        .collect();

    let results: Vec<(f64, f64)> = slice_maybe_parallel!(candidates)
        .map(|&period| {
            let basis = fourier_basis_with_period(argvals, nbasis, period);

            // Simple least squares fit
            let mut rss = 0.0;
            for j in 0..m {
                let mut fitted = 0.0;
                // Simple: use mean of basis function times data as rough fit
                for k in 0..nbasis {
                    let b_val = basis[j + k * m];
                    let coef: f64 = (0..m)
                        .map(|l| mean_curve[l] * basis[l + k * m])
                        .sum::<f64>()
                        / (0..m)
                            .map(|l| basis[l + k * m].powi(2))
                            .sum::<f64>()
                            .max(1e-15);
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
/// * `data` - Column-major matrix (n x m) as flat slice
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
    let mat = FdMatrix::from_slice(data, n, m).unwrap();
    let mean_curve = compute_mean_curve(&mat);

    let mut residual = mean_curve.clone();
    let mut detected = Vec::with_capacity(max_periods);

    for iteration in 1..=max_periods {
        match evaluate_next_period(
            &mut residual,
            m,
            argvals,
            min_confidence,
            min_strength,
            iteration,
        ) {
            Some(period) => detected.push(period),
            None => break,
        }
    }

    detected
}

/// Evaluate and extract the next dominant period from the residual signal.
///
/// Returns `None` if no significant period is found (signals iteration should stop).
fn evaluate_next_period(
    residual: &mut [f64],
    m: usize,
    argvals: &[f64],
    min_confidence: f64,
    min_strength: f64,
    iteration: usize,
) -> Option<DetectedPeriod> {
    let residual_mat = FdMatrix::from_slice(residual, 1, m).unwrap();
    let est = estimate_period_fft(&residual_mat, argvals);

    if est.confidence < min_confidence || est.period.is_nan() || est.period.is_infinite() {
        return None;
    }

    let strength = seasonal_strength_variance(&residual_mat, argvals, est.period, 3);
    if strength < min_strength || strength.is_nan() {
        return None;
    }

    let (_a, _b, amplitude, phase) = fit_and_subtract_sinusoid(residual, argvals, est.period);

    Some(DetectedPeriod {
        period: est.period,
        confidence: est.confidence,
        strength,
        amplitude,
        phase,
        iteration,
    })
}
