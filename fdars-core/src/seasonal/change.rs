use crate::matrix::FdMatrix;
use std::f64::consts::PI;

use super::hilbert::hilbert_transform;
use super::peak::detect_peaks;
use super::strength::{
    seasonal_strength_spectral, seasonal_strength_variance, seasonal_strength_windowed,
};
use super::{
    analyze_amplitude_envelope, compute_cycle_strengths, compute_mean_curve, cwt_morlet_fft,
    detect_threshold_crossings, linear_slope, otsu_threshold, valid_interior_bounds,
    ChangeDetectionResult, PeakTimingResult, StrengthMethod,
};

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

/// Type of amplitude modulation pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModulationType {
    /// Constant amplitude (no modulation)
    Stable,
    /// Amplitude increases over time (seasonality emerges)
    Emerging,
    /// Amplitude decreases over time (seasonality fades)
    Fading,
    /// Amplitude varies non-monotonically
    Oscillating,
    /// No seasonality detected
    NonSeasonal,
}

/// Result of amplitude modulation detection.
#[derive(Debug, Clone)]
pub struct AmplitudeModulationResult {
    /// Whether seasonality is present (using robust spectral method)
    pub is_seasonal: bool,
    /// Overall seasonal strength (spectral method)
    pub seasonal_strength: f64,
    /// Whether amplitude modulation is detected
    pub has_modulation: bool,
    /// Type of amplitude modulation
    pub modulation_type: ModulationType,
    /// Coefficient of variation of time-varying strength (0 = stable, higher = more modulation)
    pub modulation_score: f64,
    /// Trend in amplitude (-1 to 1: negative = fading, positive = emerging)
    pub amplitude_trend: f64,
    /// Time-varying seasonal strength curve
    pub strength_curve: Vec<f64>,
    /// Time points corresponding to strength_curve
    pub time_points: Vec<f64>,
    /// Minimum strength in the curve
    pub min_strength: f64,
    /// Maximum strength in the curve
    pub max_strength: f64,
}

/// Result of wavelet-based amplitude modulation detection.
#[derive(Debug, Clone)]
pub struct WaveletAmplitudeResult {
    /// Whether seasonality is present
    pub is_seasonal: bool,
    /// Overall seasonal strength
    pub seasonal_strength: f64,
    /// Whether amplitude modulation is detected
    pub has_modulation: bool,
    /// Type of amplitude modulation
    pub modulation_type: ModulationType,
    /// Coefficient of variation of wavelet amplitude
    pub modulation_score: f64,
    /// Trend in amplitude (-1 to 1)
    pub amplitude_trend: f64,
    /// Wavelet amplitude at the seasonal frequency over time
    pub wavelet_amplitude: Vec<f64>,
    /// Time points corresponding to wavelet_amplitude
    pub time_points: Vec<f64>,
    /// Scale (period) used for wavelet analysis
    pub scale: f64,
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

/// Detect changes in seasonality.
///
/// Monitors time-varying seasonal strength and detects threshold crossings
/// that indicate onset or cessation of seasonality.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
/// * `threshold` - SS threshold for seasonal/non-seasonal (e.g., 0.3)
/// * `window_size` - Window size for local strength estimation
/// * `min_duration` - Minimum duration to confirm a change
pub fn detect_seasonality_changes(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    threshold: f64,
    window_size: f64,
    min_duration: f64,
) -> ChangeDetectionResult {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m {
        return ChangeDetectionResult {
            change_points: Vec::new(),
            strength_curve: Vec::new(),
        };
    }

    // Compute time-varying seasonal strength
    let strength_curve =
        seasonal_strength_windowed(data, argvals, period, window_size, StrengthMethod::Variance);

    if strength_curve.is_empty() {
        return ChangeDetectionResult {
            change_points: Vec::new(),
            strength_curve: Vec::new(),
        };
    }

    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let min_dur_points = (min_duration / dt).round() as usize;

    let change_points =
        detect_threshold_crossings(&strength_curve, argvals, threshold, min_dur_points);

    ChangeDetectionResult {
        change_points,
        strength_curve,
    }
}

/// Detect amplitude modulation in seasonal time series.
///
/// This function first checks if seasonality exists using the spectral method
/// (which is robust to amplitude modulation), then uses Hilbert transform to
/// extract the amplitude envelope and analyze modulation patterns.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period in argvals units
/// * `modulation_threshold` - CV threshold for detecting modulation (default: 0.15)
/// * `seasonality_threshold` - Strength threshold for seasonality (default: 0.3)
///
/// # Returns
/// `AmplitudeModulationResult` containing detection results and diagnostics
pub fn detect_amplitude_modulation(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    modulation_threshold: f64,
    seasonality_threshold: f64,
) -> AmplitudeModulationResult {
    let (n, m) = data.shape();
    // Default result for invalid input
    let empty_result = AmplitudeModulationResult {
        is_seasonal: false,
        seasonal_strength: 0.0,
        has_modulation: false,
        modulation_type: ModulationType::NonSeasonal,
        modulation_score: 0.0,
        amplitude_trend: 0.0,
        strength_curve: Vec::new(),
        time_points: Vec::new(),
        min_strength: 0.0,
        max_strength: 0.0,
    };

    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return empty_result;
    }

    // Step 1: Check if seasonality exists using spectral method (robust to AM)
    let overall_strength = seasonal_strength_spectral(data, argvals, period);

    if overall_strength < seasonality_threshold {
        return AmplitudeModulationResult {
            is_seasonal: false,
            seasonal_strength: overall_strength,
            has_modulation: false,
            modulation_type: ModulationType::NonSeasonal,
            modulation_score: 0.0,
            amplitude_trend: 0.0,
            strength_curve: Vec::new(),
            time_points: argvals.to_vec(),
            min_strength: 0.0,
            max_strength: 0.0,
        };
    }

    // Step 2: Compute mean curve
    let mean_curve = compute_mean_curve(data);

    // Step 3: Use Hilbert transform to get amplitude envelope
    let dc: f64 = mean_curve.iter().sum::<f64>() / m as f64;
    let detrended: Vec<f64> = mean_curve.iter().map(|&x| x - dc).collect();
    let analytic = hilbert_transform(&detrended);
    let envelope: Vec<f64> = analytic.iter().map(|c| c.norm()).collect();

    if envelope.is_empty() {
        return AmplitudeModulationResult {
            is_seasonal: true,
            seasonal_strength: overall_strength,
            has_modulation: false,
            modulation_type: ModulationType::Stable,
            modulation_score: 0.0,
            amplitude_trend: 0.0,
            strength_curve: Vec::new(),
            time_points: argvals.to_vec(),
            min_strength: 0.0,
            max_strength: 0.0,
        };
    }

    // Step 4: Smooth the envelope to reduce high-frequency noise
    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let smooth_window = ((period / dt) as usize).max(3);
    let half_window = smooth_window / 2;

    let smoothed_envelope: Vec<f64> = (0..m)
        .map(|i| {
            let start = i.saturating_sub(half_window);
            let end = (i + half_window + 1).min(m);
            let sum: f64 = envelope[start..end].iter().sum();
            sum / (end - start) as f64
        })
        .collect();

    // Step 5: Analyze envelope statistics
    let Some((interior_start, interior_end)) = valid_interior_bounds(m, 4) else {
        return AmplitudeModulationResult {
            is_seasonal: true,
            seasonal_strength: overall_strength,
            has_modulation: false,
            modulation_type: ModulationType::Stable,
            modulation_score: 0.0,
            amplitude_trend: 0.0,
            strength_curve: envelope,
            time_points: argvals.to_vec(),
            min_strength: 0.0,
            max_strength: 0.0,
        };
    };

    let stats = analyze_amplitude_envelope(
        &smoothed_envelope[interior_start..interior_end],
        &argvals[interior_start..interior_end],
        modulation_threshold,
    );

    AmplitudeModulationResult {
        is_seasonal: true,
        seasonal_strength: overall_strength,
        has_modulation: stats.has_modulation,
        modulation_type: stats.modulation_type,
        modulation_score: stats.modulation_score,
        amplitude_trend: stats.amplitude_trend,
        strength_curve: envelope,
        time_points: argvals.to_vec(),
        min_strength: stats.min_amp,
        max_strength: stats.max_amp,
    }
}

/// Detect amplitude modulation using Morlet wavelet transform.
///
/// Uses continuous wavelet transform at the seasonal period to extract
/// time-varying amplitude. This method is more robust to noise and can
/// better handle non-stationary signals compared to Hilbert transform.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period in argvals units
/// * `modulation_threshold` - CV threshold for detecting modulation (default: 0.15)
/// * `seasonality_threshold` - Strength threshold for seasonality (default: 0.3)
///
/// # Returns
/// `WaveletAmplitudeResult` containing detection results and wavelet amplitude curve
pub fn detect_amplitude_modulation_wavelet(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    modulation_threshold: f64,
    seasonality_threshold: f64,
) -> WaveletAmplitudeResult {
    let (n, m) = data.shape();
    let empty_result = WaveletAmplitudeResult {
        is_seasonal: false,
        seasonal_strength: 0.0,
        has_modulation: false,
        modulation_type: ModulationType::NonSeasonal,
        modulation_score: 0.0,
        amplitude_trend: 0.0,
        wavelet_amplitude: Vec::new(),
        time_points: Vec::new(),
        scale: 0.0,
    };

    if n == 0 || m < 4 || argvals.len() != m || period <= 0.0 {
        return empty_result;
    }

    // Step 1: Check if seasonality exists using spectral method
    let overall_strength = seasonal_strength_spectral(data, argvals, period);

    if overall_strength < seasonality_threshold {
        return WaveletAmplitudeResult {
            is_seasonal: false,
            seasonal_strength: overall_strength,
            has_modulation: false,
            modulation_type: ModulationType::NonSeasonal,
            modulation_score: 0.0,
            amplitude_trend: 0.0,
            wavelet_amplitude: Vec::new(),
            time_points: argvals.to_vec(),
            scale: 0.0,
        };
    }

    // Step 2: Compute mean curve
    let mean_curve = compute_mean_curve(data);

    // Remove DC component
    let dc: f64 = mean_curve.iter().sum::<f64>() / m as f64;
    let detrended: Vec<f64> = mean_curve.iter().map(|&x| x - dc).collect();

    // Step 3: Compute wavelet transform at the seasonal period
    let omega0 = 6.0; // Standard Morlet parameter
    let scale = period * omega0 / (2.0 * PI);

    // Use FFT-based CWT for efficiency
    let wavelet_coeffs = cwt_morlet_fft(&detrended, argvals, scale, omega0);

    if wavelet_coeffs.is_empty() {
        return WaveletAmplitudeResult {
            is_seasonal: true,
            seasonal_strength: overall_strength,
            has_modulation: false,
            modulation_type: ModulationType::Stable,
            modulation_score: 0.0,
            amplitude_trend: 0.0,
            wavelet_amplitude: Vec::new(),
            time_points: argvals.to_vec(),
            scale,
        };
    }

    // Step 4: Extract amplitude (magnitude of wavelet coefficients)
    let wavelet_amplitude: Vec<f64> = wavelet_coeffs.iter().map(|c| c.norm()).collect();

    // Step 5: Analyze amplitude envelope statistics (skip edges)
    let Some((interior_start, interior_end)) = valid_interior_bounds(m, 4) else {
        return WaveletAmplitudeResult {
            is_seasonal: true,
            seasonal_strength: overall_strength,
            has_modulation: false,
            modulation_type: ModulationType::Stable,
            modulation_score: 0.0,
            amplitude_trend: 0.0,
            wavelet_amplitude,
            time_points: argvals.to_vec(),
            scale,
        };
    };

    let stats = analyze_amplitude_envelope(
        &wavelet_amplitude[interior_start..interior_end],
        &argvals[interior_start..interior_end],
        modulation_threshold,
    );

    WaveletAmplitudeResult {
        is_seasonal: true,
        seasonal_strength: overall_strength,
        has_modulation: stats.has_modulation,
        modulation_type: stats.modulation_type,
        modulation_score: stats.modulation_score,
        amplitude_trend: stats.amplitude_trend,
        wavelet_amplitude,
        time_points: argvals.to_vec(),
        scale,
    }
}

/// Analyze peak timing variability across cycles.
///
/// For short series (e.g., 3-5 years of yearly data), this function detects
/// one peak per cycle and analyzes how peak timing varies between cycles.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
/// * `period` - Known period (e.g., 365 for daily data with yearly seasonality)
/// * `smooth_nbasis` - Number of Fourier basis functions for smoothing.
///   If None, uses GCV for automatic selection.
///
/// # Returns
/// Peak timing result with variability metrics
pub fn analyze_peak_timing(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    smooth_nbasis: Option<usize>,
) -> PeakTimingResult {
    let (n, m) = data.shape();
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
        data,
        argvals,
        Some(min_distance),
        None, // No prominence filter
        true, // Smooth first with Fourier basis
        smooth_nbasis,
    );

    // Use the first sample's peaks (for mean curve analysis)
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
        .map(|&t| crate::utility::f64_to_usize_clamped(((t - t_start) / period).floor()) + 1)
        .collect();

    // Compute statistics
    let n_peaks = normalized_timing.len() as f64;
    let mean_timing = normalized_timing.iter().sum::<f64>() / n_peaks;

    let variance: f64 = normalized_timing
        .iter()
        .map(|&x| (x - mean_timing).powi(2))
        .sum::<f64>()
        / n_peaks;
    let std_timing = variance.sqrt();

    let min_timing = normalized_timing
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let max_timing = normalized_timing
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let range_timing = max_timing - min_timing;

    // Variability score: normalized std deviation
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
/// * `argvals` - Evaluation points
/// * `period` - Known seasonal period
/// * `strength_threshold` - Threshold for seasonal/non-seasonal (default: 0.3)
/// * `timing_threshold` - Max std of normalized timing for "stable" (default: 0.05)
///
/// # Returns
/// Seasonality classification with type and diagnostics
pub fn classify_seasonality(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    strength_threshold: Option<f64>,
    timing_threshold: Option<f64>,
) -> SeasonalityClassification {
    let strength_thresh = strength_threshold.unwrap_or(0.3);
    let timing_thresh = timing_threshold.unwrap_or(0.05);

    let (n, m) = data.shape();
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
    let overall_strength = seasonal_strength_variance(data, argvals, period, 3);

    let (cycle_strengths, weak_seasons) =
        compute_cycle_strengths(data, argvals, period, strength_thresh);
    let n_cycles = cycle_strengths.len();

    // Analyze peak timing
    let peak_timing = analyze_peak_timing(data, argvals, period, None);

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
/// * `argvals` - Evaluation points
/// * `period` - Seasonal period
/// * `threshold_method` - Method for threshold selection
/// * `window_size` - Window size for local strength estimation
/// * `min_duration` - Minimum duration to confirm a change
pub fn detect_seasonality_changes_auto(
    data: &FdMatrix,
    argvals: &[f64],
    period: f64,
    threshold_method: ThresholdMethod,
    window_size: f64,
    min_duration: f64,
) -> ChangeDetectionResult {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m {
        return ChangeDetectionResult {
            change_points: Vec::new(),
            strength_curve: Vec::new(),
        };
    }

    // Compute time-varying seasonal strength
    let strength_curve =
        seasonal_strength_windowed(data, argvals, period, window_size, StrengthMethod::Variance);

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
            let mut sorted: Vec<f64> = strength_curve
                .iter()
                .copied()
                .filter(|x| x.is_finite())
                .collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if sorted.is_empty() {
                0.5
            } else {
                let idx = crate::utility::f64_to_usize_clamped((p / 100.0) * sorted.len() as f64);
                sorted[idx.min(sorted.len() - 1)]
            }
        }
        ThresholdMethod::Otsu => otsu_threshold(&strength_curve),
    };

    // Now use the regular detection with computed threshold
    detect_seasonality_changes(data, argvals, period, threshold, window_size, min_duration)
}
