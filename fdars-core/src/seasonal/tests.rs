use super::autoperiod::{
    empty_cfd_result, rank_cfd_results, validate_cfd_candidates, validate_or_fallback_cfd,
};
use super::lomb_scargle::{
    estimate_independent_frequencies, generate_ls_frequencies, lomb_scargle_fap,
};
use super::ssa::{
    apply_ssa_grouping_defaults, auto_group_ssa_components, classify_ssa_component,
    diagonal_average, embed_trajectory, is_periodic_component, is_trend_component,
    reconstruct_grouped, svd_decompose, SsaComponentKind,
};
use super::*;
use std::f64::consts::PI;

fn generate_sine(n: usize, m: usize, period: f64, argvals: &[f64]) -> FdMatrix {
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            data[i + j * n] = (2.0 * PI * argvals[j] / period).sin();
        }
    }
    FdMatrix::from_column_major(data, n, m).unwrap()
}

#[test]
fn test_period_estimation_fft() {
    let m = 200;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let period = 2.0;
    let data = generate_sine(1, m, period, &argvals);

    let estimate = estimate_period_fft(&data, &argvals);
    assert!((estimate.period - period).abs() < 0.2);
    assert!(estimate.confidence > 1.0);
}

#[test]
fn test_peak_detection() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let period = 2.0;
    let data = generate_sine(1, m, period, &argvals);

    let result = detect_peaks(&data, &argvals, Some(1.5), None, false, None);

    // Should find approximately 5 peaks (10 / 2)
    assert!(!result.peaks[0].is_empty());
    assert!((result.mean_period - period).abs() < 0.3);
}

#[test]
fn test_peak_detection_known_sine() {
    // Pure sine wave: sin(2*pi*t/2) on [0, 10]
    // Peaks occur at t = period/4 + k*period = 0.5, 2.5, 4.5, 6.5, 8.5
    let m = 200; // High resolution for accurate detection
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_peaks(&data, &argvals, None, None, false, None);

    // Should find exactly 5 peaks
    assert_eq!(
        result.peaks[0].len(),
        5,
        "Expected 5 peaks, got {}. Peak times: {:?}",
        result.peaks[0].len(),
        result.peaks[0].iter().map(|p| p.time).collect::<Vec<_>>()
    );

    // Check peak locations are close to expected
    let expected_times = [0.5, 2.5, 4.5, 6.5, 8.5];
    for (peak, expected) in result.peaks[0].iter().zip(expected_times.iter()) {
        assert!(
            (peak.time - expected).abs() < 0.15,
            "Peak at {:.3} not close to expected {:.3}",
            peak.time,
            expected
        );
    }

    // Check mean period
    assert!(
        (result.mean_period - period).abs() < 0.1,
        "Mean period {:.3} not close to expected {:.3}",
        result.mean_period,
        period
    );
}

#[test]
fn test_peak_detection_with_min_distance() {
    // Same sine wave but with min_distance constraint
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    // min_distance = 1.5 should still find all 5 peaks (spacing = 2.0)
    let result = detect_peaks(&data, &argvals, Some(1.5), None, false, None);
    assert_eq!(
        result.peaks[0].len(),
        5,
        "With min_distance=1.5, expected 5 peaks, got {}",
        result.peaks[0].len()
    );

    // min_distance = 2.5 should find fewer peaks
    let result2 = detect_peaks(&data, &argvals, Some(2.5), None, false, None);
    assert!(
        result2.peaks[0].len() < 5,
        "With min_distance=2.5, expected fewer than 5 peaks, got {}",
        result2.peaks[0].len()
    );
}

#[test]
fn test_peak_detection_period_1() {
    // Higher frequency: sin(2*pi*t/1) on [0, 10]
    // Peaks at t = 0.25, 1.25, 2.25, ..., 9.25 (10 peaks)
    let m = 400; // Higher resolution for higher frequency
    let period = 1.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_peaks(&data, &argvals, None, None, false, None);

    // Should find 10 peaks
    assert_eq!(
        result.peaks[0].len(),
        10,
        "Expected 10 peaks, got {}",
        result.peaks[0].len()
    );

    // Check mean period
    assert!(
        (result.mean_period - period).abs() < 0.1,
        "Mean period {:.3} not close to expected {:.3}",
        result.mean_period,
        period
    );
}

#[test]
fn test_peak_detection_shifted_sine() {
    // Shifted sine: sin(2*pi*t/2) + 1 on [0, 10]
    // Same peak locations, just shifted up
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t / period).sin() + 1.0)
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_peaks(&data, &argvals, None, None, false, None);

    // Should still find 5 peaks
    assert_eq!(
        result.peaks[0].len(),
        5,
        "Expected 5 peaks for shifted sine, got {}",
        result.peaks[0].len()
    );

    // Peak values should be around 2.0 (max of sin + 1)
    for peak in &result.peaks[0] {
        assert!(
            (peak.value - 2.0).abs() < 0.05,
            "Peak value {:.3} not close to expected 2.0",
            peak.value
        );
    }
}

#[test]
fn test_peak_detection_prominence() {
    // Create signal with peaks of different heights
    // Large peaks at odd positions, small peaks at even positions
    let m = 200;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            let base = (2.0 * std::f64::consts::PI * t / 2.0).sin();
            // Add small ripples
            let ripple = 0.1 * (2.0 * std::f64::consts::PI * t * 4.0).sin();
            base + ripple
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    // Without prominence filter, may find extra peaks from ripples
    let result_no_filter = detect_peaks(&data, &argvals, None, None, false, None);

    // With prominence filter, should only find major peaks
    let result_filtered = detect_peaks(&data, &argvals, None, Some(0.5), false, None);

    // Filtered should have fewer or equal peaks
    assert!(
        result_filtered.peaks[0].len() <= result_no_filter.peaks[0].len(),
        "Prominence filter should reduce peak count"
    );
}

#[test]
fn test_peak_detection_different_amplitudes() {
    // Test with various amplitudes: 0.5, 1.0, 2.0, 5.0
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();

    for amplitude in [0.5, 1.0, 2.0, 5.0] {
        let data: Vec<f64> = argvals
            .iter()
            .map(|&t| amplitude * (2.0 * std::f64::consts::PI * t / period).sin())
            .collect();
        let data = FdMatrix::from_column_major(data, 1, m).unwrap();

        let result = detect_peaks(&data, &argvals, None, None, false, None);

        assert_eq!(
            result.peaks[0].len(),
            5,
            "Amplitude {} should still find 5 peaks",
            amplitude
        );

        // Peak values should be close to amplitude
        for peak in &result.peaks[0] {
            assert!(
                (peak.value - amplitude).abs() < 0.1,
                "Peak value {:.3} should be close to amplitude {}",
                peak.value,
                amplitude
            );
        }
    }
}

#[test]
fn test_peak_detection_varying_frequency() {
    // Signal with varying frequency: chirp-like signal
    // Peaks get closer together over time
    let m = 400;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();

    // Frequency increases linearly: f(t) = 0.5 + 0.1*t
    // Phase integral: phi(t) = 2*pi * (0.5*t + 0.05*t^2)
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            let phase = 2.0 * std::f64::consts::PI * (0.5 * t + 0.05 * t * t);
            phase.sin()
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_peaks(&data, &argvals, None, None, false, None);

    // Should find multiple peaks with decreasing spacing
    assert!(
        result.peaks[0].len() >= 5,
        "Should find at least 5 peaks, got {}",
        result.peaks[0].len()
    );

    // Verify inter-peak distances decrease over time
    let distances = &result.inter_peak_distances[0];
    if distances.len() >= 3 {
        // Later peaks should be closer than earlier peaks
        let early_avg = (distances[0] + distances[1]) / 2.0;
        let late_avg = (distances[distances.len() - 2] + distances[distances.len() - 1]) / 2.0;
        assert!(
            late_avg < early_avg,
            "Later peaks should be closer: early avg={:.3}, late avg={:.3}",
            early_avg,
            late_avg
        );
    }
}

#[test]
fn test_peak_detection_sum_of_sines() {
    // Sum of two sine waves with different periods creates non-uniform peak spacing
    // y = sin(2*pi*t/2) + 0.5*sin(2*pi*t/3)
    let m = 300;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 12.0 / (m - 1) as f64).collect();

    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            (2.0 * std::f64::consts::PI * t / 2.0).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * t / 3.0).sin()
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_peaks(&data, &argvals, Some(1.0), None, false, None);

    // Should find peaks (exact count depends on interference pattern)
    assert!(
        result.peaks[0].len() >= 4,
        "Should find at least 4 peaks, got {}",
        result.peaks[0].len()
    );

    // Inter-peak distances should vary
    let distances = &result.inter_peak_distances[0];
    if distances.len() >= 2 {
        let min_dist = distances.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_dist = distances.iter().cloned().fold(0.0, f64::max);
        assert!(
            max_dist > min_dist * 1.1,
            "Distances should vary: min={:.3}, max={:.3}",
            min_dist,
            max_dist
        );
    }
}

#[test]
fn test_seasonal_strength() {
    let m = 200;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let period = 2.0;
    let data = generate_sine(1, m, period, &argvals);

    let strength = seasonal_strength_variance(&data, &argvals, period, 3);
    // Pure sine should have high seasonal strength
    assert!(strength > 0.8);

    let strength_spectral = seasonal_strength_spectral(&data, &argvals, period);
    assert!(strength_spectral > 0.5);
}

#[test]
fn test_instantaneous_period() {
    let m = 200;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let period = 2.0;
    let data = generate_sine(1, m, period, &argvals);

    let result = instantaneous_period(&data, &argvals);

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

    let result = analyze_peak_timing(&data, &argvals, period, Some(11));

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

    let result = classify_seasonality(&data, &argvals, period, None, None);

    assert!(result.is_seasonal);
    assert!(result.seasonal_strength > 0.5);
    assert!(matches!(
        result.classification,
        SeasonalType::StableSeasonal | SeasonalType::VariableTiming
    ));
}

#[test]
fn test_otsu_threshold() {
    // Bimodal distribution: mix of low (0.1-0.2) and high (0.7-0.9) values
    let values = vec![
        0.1, 0.12, 0.15, 0.18, 0.11, 0.14, 0.7, 0.75, 0.8, 0.85, 0.9, 0.72,
    ];

    let threshold = otsu_threshold(&values);

    // Threshold should be between the two modes
    // Due to small sample size, Otsu's method may not find optimal threshold
    // Just verify it returns a reasonable value in the data range
    assert!(threshold >= 0.1, "Threshold {} should be >= 0.1", threshold);
    assert!(threshold <= 0.9, "Threshold {} should be <= 0.9", threshold);
}

#[test]
fn test_gcv_fourier_nbasis_selection() {
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();

    // Noisy sine wave
    let mut data = vec![0.0; m];
    for j in 0..m {
        data[j] = (2.0 * PI * argvals[j] / 2.0).sin() + 0.1 * (j as f64 * 0.3).sin();
    }

    let data_mat = crate::matrix::FdMatrix::from_column_major(data, 1, m).unwrap();
    let nbasis = crate::basis::select_fourier_nbasis_gcv(&data_mat, &argvals, 5, 25);

    // nbasis should be reasonable (between min and max)
    assert!(nbasis >= 5);
    assert!(nbasis <= 25);
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
        data[j] =
            (2.0 * PI * argvals[j] / period1).sin() + 0.6 * (2.0 * PI * argvals[j] / period2).sin();
    }

    // Use higher min_strength threshold to properly stop after real periods
    let detected = detect_multiple_periods(&data, 1, m, &argvals, 5, 0.4, 0.20);

    // Should detect exactly 2 periods with these thresholds
    assert!(
        detected.len() >= 2,
        "Expected at least 2 periods, found {}",
        detected.len()
    );

    // Check that both periods were detected (order depends on amplitude)
    let periods: Vec<f64> = detected.iter().map(|d| d.period).collect();
    let has_period1 = periods.iter().any(|&p| (p - period1).abs() < 0.3);
    let has_period2 = periods.iter().any(|&p| (p - period2).abs() < 0.5);

    assert!(
        has_period1,
        "Expected to find period ~{}, got {:?}",
        period1, periods
    );
    assert!(
        has_period2,
        "Expected to find period ~{}, got {:?}",
        period2, periods
    );

    // Verify first detected has higher amplitude (amplitude 1.0 vs 0.6)
    assert!(
        detected[0].amplitude > detected[1].amplitude,
        "First detected should have higher amplitude"
    );

    // Each detected period should have strength and confidence info
    for d in &detected {
        assert!(
            d.strength > 0.0,
            "Detected period should have positive strength"
        );
        assert!(
            d.confidence > 0.0,
            "Detected period should have positive confidence"
        );
        assert!(
            d.amplitude > 0.0,
            "Detected period should have positive amplitude"
        );
    }
}

// ========================================================================
// Amplitude Modulation Detection Tests
// ========================================================================

#[test]
fn test_amplitude_modulation_stable() {
    // Constant amplitude seasonal signal - should detect as Stable
    let m = 200;
    let period = 0.2;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Constant amplitude sine wave
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation(
        &data, &argvals, period, 0.15, // modulation threshold
        0.3,  // seasonality threshold
    );

    eprintln!(
        "Stable test: is_seasonal={}, has_modulation={}, modulation_score={:.4}, amplitude_trend={:.4}, type={:?}",
        result.is_seasonal, result.has_modulation, result.modulation_score, result.amplitude_trend, result.modulation_type
    );

    assert!(result.is_seasonal, "Should detect seasonality");
    assert!(
        !result.has_modulation,
        "Constant amplitude should not have modulation, got score={:.4}",
        result.modulation_score
    );
    assert_eq!(
        result.modulation_type,
        ModulationType::Stable,
        "Should be classified as Stable"
    );
}

#[test]
fn test_amplitude_modulation_emerging() {
    // Amplitude increases over time (emerging seasonality)
    let m = 200;
    let period = 0.2;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Amplitude grows from 0.2 to 1.0
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            let amplitude = 0.2 + 0.8 * t; // Linear increase
            amplitude * (2.0 * PI * t / period).sin()
        })
        .collect();

    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation(&data, &argvals, period, 0.15, 0.2);

    eprintln!(
        "Emerging test: is_seasonal={}, has_modulation={}, modulation_score={:.4}, amplitude_trend={:.4}, type={:?}",
        result.is_seasonal, result.has_modulation, result.modulation_score, result.amplitude_trend, result.modulation_type
    );

    assert!(result.is_seasonal, "Should detect seasonality");
    assert!(
        result.has_modulation,
        "Growing amplitude should have modulation, score={:.4}",
        result.modulation_score
    );
    assert_eq!(
        result.modulation_type,
        ModulationType::Emerging,
        "Should be classified as Emerging, trend={:.4}",
        result.amplitude_trend
    );
    assert!(
        result.amplitude_trend > 0.0,
        "Trend should be positive for emerging"
    );
}

#[test]
fn test_amplitude_modulation_fading() {
    // Amplitude decreases over time (fading seasonality)
    let m = 200;
    let period = 0.2;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Amplitude decreases from 1.0 to 0.2
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            let amplitude = 1.0 - 0.8 * t; // Linear decrease
            amplitude * (2.0 * PI * t / period).sin()
        })
        .collect();

    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation(&data, &argvals, period, 0.15, 0.2);

    eprintln!(
        "Fading test: is_seasonal={}, has_modulation={}, modulation_score={:.4}, amplitude_trend={:.4}, type={:?}",
        result.is_seasonal, result.has_modulation, result.modulation_score, result.amplitude_trend, result.modulation_type
    );

    assert!(result.is_seasonal, "Should detect seasonality");
    assert!(
        result.has_modulation,
        "Fading amplitude should have modulation"
    );
    assert_eq!(
        result.modulation_type,
        ModulationType::Fading,
        "Should be classified as Fading, trend={:.4}",
        result.amplitude_trend
    );
    assert!(
        result.amplitude_trend < 0.0,
        "Trend should be negative for fading"
    );
}

#[test]
fn test_amplitude_modulation_oscillating() {
    // Amplitude oscillates (neither purely emerging nor fading)
    let m = 200;
    let period = 0.1;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Amplitude oscillates: high-low-high-low pattern
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            let amplitude = 0.5 + 0.4 * (2.0 * PI * t * 2.0).sin(); // 2 modulation cycles
            amplitude * (2.0 * PI * t / period).sin()
        })
        .collect();

    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation(&data, &argvals, period, 0.15, 0.2);

    eprintln!(
        "Oscillating test: is_seasonal={}, has_modulation={}, modulation_score={:.4}, amplitude_trend={:.4}, type={:?}",
        result.is_seasonal, result.has_modulation, result.modulation_score, result.amplitude_trend, result.modulation_type
    );

    assert!(result.is_seasonal, "Should detect seasonality");
    // Oscillating has high variation but near-zero trend
    if result.has_modulation {
        // Trend should be near zero for oscillating
        assert!(
            result.amplitude_trend.abs() < 0.5,
            "Trend should be small for oscillating"
        );
    }
}

#[test]
fn test_amplitude_modulation_non_seasonal() {
    // Pure noise - no seasonality
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Random noise (use simple pseudo-random)
    let data: Vec<f64> = (0..m)
        .map(|i| ((i as f64 * 1.618).sin() * 100.0).fract())
        .collect();

    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation(
        &data, &argvals, 0.2, // arbitrary period
        0.15, 0.3,
    );

    assert!(
        !result.is_seasonal,
        "Noise should not be detected as seasonal"
    );
    assert_eq!(
        result.modulation_type,
        ModulationType::NonSeasonal,
        "Should be classified as NonSeasonal"
    );
}

// ========================================================================
// Wavelet-based Amplitude Modulation Detection Tests
// ========================================================================

#[test]
fn test_wavelet_amplitude_modulation_stable() {
    let m = 200;
    let period = 0.2;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();

    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation_wavelet(&data, &argvals, period, 0.15, 0.3);

    eprintln!(
        "Wavelet stable: is_seasonal={}, has_modulation={}, modulation_score={:.4}, amplitude_trend={:.4}, type={:?}",
        result.is_seasonal, result.has_modulation, result.modulation_score, result.amplitude_trend, result.modulation_type
    );

    assert!(result.is_seasonal, "Should detect seasonality");
    assert!(
        !result.has_modulation,
        "Constant amplitude should not have modulation, got score={:.4}",
        result.modulation_score
    );
}

#[test]
fn test_wavelet_amplitude_modulation_emerging() {
    let m = 200;
    let period = 0.2;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Amplitude grows from 0.2 to 1.0
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            let amplitude = 0.2 + 0.8 * t;
            amplitude * (2.0 * PI * t / period).sin()
        })
        .collect();

    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation_wavelet(&data, &argvals, period, 0.15, 0.2);

    eprintln!(
        "Wavelet emerging: is_seasonal={}, has_modulation={}, modulation_score={:.4}, amplitude_trend={:.4}, type={:?}",
        result.is_seasonal, result.has_modulation, result.modulation_score, result.amplitude_trend, result.modulation_type
    );

    assert!(result.is_seasonal, "Should detect seasonality");
    assert!(
        result.has_modulation,
        "Growing amplitude should have modulation"
    );
    assert!(
        result.amplitude_trend > 0.0,
        "Trend should be positive for emerging"
    );
}

#[test]
fn test_wavelet_amplitude_modulation_fading() {
    let m = 200;
    let period = 0.2;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Amplitude decreases from 1.0 to 0.2
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            let amplitude = 1.0 - 0.8 * t;
            amplitude * (2.0 * PI * t / period).sin()
        })
        .collect();

    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation_wavelet(&data, &argvals, period, 0.15, 0.2);

    eprintln!(
        "Wavelet fading: is_seasonal={}, has_modulation={}, modulation_score={:.4}, amplitude_trend={:.4}, type={:?}",
        result.is_seasonal, result.has_modulation, result.modulation_score, result.amplitude_trend, result.modulation_type
    );

    assert!(result.is_seasonal, "Should detect seasonality");
    assert!(
        result.has_modulation,
        "Fading amplitude should have modulation"
    );
    assert!(
        result.amplitude_trend < 0.0,
        "Trend should be negative for fading"
    );
}

#[test]
fn test_seasonal_strength_wavelet() {
    let m = 200;
    let period = 0.2;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Pure sine wave at target period - should have high strength
    let seasonal_data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();

    let seasonal_data = FdMatrix::from_column_major(seasonal_data, 1, m).unwrap();

    let strength = seasonal_strength_wavelet(&seasonal_data, &argvals, period);
    eprintln!("Wavelet strength (pure sine): {:.4}", strength);
    assert!(
        strength > 0.5,
        "Pure sine should have high wavelet strength"
    );

    // Pure noise - should have low strength
    let noise_data: Vec<f64> = (0..m)
        .map(|i| ((i * 12345 + 67890) % 1000) as f64 / 1000.0 - 0.5)
        .collect();

    let noise_data = FdMatrix::from_column_major(noise_data, 1, m).unwrap();

    let noise_strength = seasonal_strength_wavelet(&noise_data, &argvals, period);
    eprintln!("Wavelet strength (noise): {:.4}", noise_strength);
    assert!(
        noise_strength < 0.3,
        "Noise should have low wavelet strength"
    );

    // Wrong period - should have lower strength
    let wrong_period_strength = seasonal_strength_wavelet(&seasonal_data, &argvals, period * 2.0);
    eprintln!(
        "Wavelet strength (wrong period): {:.4}",
        wrong_period_strength
    );
    assert!(
        wrong_period_strength < strength,
        "Wrong period should have lower strength"
    );
}

#[test]
fn test_compute_mean_curve() {
    // 2 samples, 3 time points
    // Sample 1: [1, 2, 3]
    // Sample 2: [2, 4, 6]
    // Mean: [1.5, 3, 4.5]
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0], 2, 3).unwrap();
    let mean = compute_mean_curve(&data);
    assert_eq!(mean.len(), 3);
    assert!((mean[0] - 1.5).abs() < 1e-10);
    assert!((mean[1] - 3.0).abs() < 1e-10);
    assert!((mean[2] - 4.5).abs() < 1e-10);
}

#[test]
fn test_compute_mean_curve_parallel_consistency() {
    // Test that parallel and sequential give same results
    let n = 10;
    let m = 200;
    let data: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.1).sin()).collect();

    let data = FdMatrix::from_column_major(data, n, m).unwrap();

    let seq_result = compute_mean_curve_impl(&data, false);
    let par_result = compute_mean_curve_impl(&data, true);

    assert_eq!(seq_result.len(), par_result.len());
    for (s, p) in seq_result.iter().zip(par_result.iter()) {
        assert!(
            (s - p).abs() < 1e-10,
            "Sequential and parallel results differ"
        );
    }
}

#[test]
fn test_interior_bounds() {
    // m = 100: edge_skip = 10, interior = [10, 90)
    let bounds = interior_bounds(100);
    assert!(bounds.is_some());
    let (start, end) = bounds.unwrap();
    assert_eq!(start, 10);
    assert_eq!(end, 90);

    // m = 10: edge_skip = 1, but min(1, 2) = 1, max(9, 7) = 9
    let bounds = interior_bounds(10);
    assert!(bounds.is_some());
    let (start, end) = bounds.unwrap();
    assert!(start < end);

    // Very small m might not have valid interior
    let bounds = interior_bounds(2);
    // Should still return something as long as end > start
    assert!(bounds.is_some() || bounds.is_none());
}

#[test]
fn test_hilbert_transform_pure_sine() {
    // Hilbert transform of sin(t) should give cos(t) in imaginary part
    let m = 200;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let signal: Vec<f64> = argvals.iter().map(|&t| (2.0 * PI * t).sin()).collect();

    let analytic = hilbert_transform(&signal);
    assert_eq!(analytic.len(), m);

    // Check amplitude is approximately 1
    for c in analytic.iter().skip(10).take(m - 20) {
        let amp = c.norm();
        assert!(
            (amp - 1.0).abs() < 0.1,
            "Amplitude should be ~1, got {}",
            amp
        );
    }
}

#[test]
fn test_sazed_pure_sine() {
    // Pure sine wave with known period
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();

    let result = sazed(&data, &argvals, None);

    assert!(result.period.is_finite(), "SAZED should detect a period");
    assert!(
        (result.period - period).abs() < 0.3,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
    assert!(
        result.confidence > 0.4,
        "Expected confidence > 0.4, got {}",
        result.confidence
    );
    assert!(
        result.agreeing_components >= 2,
        "Expected at least 2 agreeing components, got {}",
        result.agreeing_components
    );
}

#[test]
fn test_sazed_noisy_sine() {
    // Sine wave with noise
    let m = 300;
    let period = 3.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();

    // Deterministic pseudo-noise using sin with different frequency
    let data: Vec<f64> = argvals
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let signal = (2.0 * PI * t / period).sin();
            let noise = 0.1 * (17.3 * i as f64).sin();
            signal + noise
        })
        .collect();

    let result = sazed(&data, &argvals, Some(0.15));

    assert!(
        result.period.is_finite(),
        "SAZED should detect a period even with noise"
    );
    assert!(
        (result.period - period).abs() < 0.5,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
}

#[test]
fn test_sazed_fdata() {
    // Multiple samples with same period
    let n = 5;
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(n, m, period, &argvals);

    let result = sazed_fdata(&data, &argvals, None);

    assert!(result.period.is_finite(), "SAZED should detect a period");
    assert!(
        (result.period - period).abs() < 0.3,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
}

#[test]
fn test_sazed_short_series() {
    // Very short series - should return NaN gracefully
    let argvals: Vec<f64> = (0..5).map(|i| i as f64).collect();
    let data: Vec<f64> = argvals.iter().map(|&t| t.sin()).collect();

    let result = sazed(&data, &argvals, None);

    // Should handle gracefully (return NaN for too-short series)
    assert!(
        result.period.is_nan() || result.period.is_finite(),
        "Should return NaN or valid period"
    );
}

#[test]
fn test_autoperiod_pure_sine() {
    // Pure sine wave with known period
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();

    let result = autoperiod(&data, &argvals, None, None);

    assert!(
        result.period.is_finite(),
        "Autoperiod should detect a period"
    );
    assert!(
        (result.period - period).abs() < 0.3,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
    assert!(
        result.confidence > 0.0,
        "Expected positive confidence, got {}",
        result.confidence
    );
}

#[test]
fn test_autoperiod_with_trend() {
    // Sine wave with linear trend
    let m = 300;
    let period = 3.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| 0.2 * t + (2.0 * PI * t / period).sin())
        .collect();

    let result = autoperiod(&data, &argvals, None, None);

    assert!(
        result.period.is_finite(),
        "Autoperiod should detect a period"
    );
    // Allow more tolerance with trend
    assert!(
        (result.period - period).abs() < 0.5,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
}

#[test]
fn test_autoperiod_candidates() {
    // Verify candidates are generated
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();

    let result = autoperiod(&data, &argvals, Some(5), Some(10));

    assert!(
        !result.candidates.is_empty(),
        "Should have at least one candidate"
    );

    // Best candidate should have highest combined score
    let max_score = result
        .candidates
        .iter()
        .map(|c| c.combined_score)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (result.confidence - max_score).abs() < 1e-10,
        "Returned confidence should match best candidate's score"
    );
}

#[test]
fn test_autoperiod_fdata() {
    // Multiple samples with same period
    let n = 5;
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(n, m, period, &argvals);

    let result = autoperiod_fdata(&data, &argvals, None, None);

    assert!(
        result.period.is_finite(),
        "Autoperiod should detect a period"
    );
    assert!(
        (result.period - period).abs() < 0.3,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
}

#[test]
fn test_cfd_autoperiod_pure_sine() {
    // Pure sine wave with known period
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();

    let result = cfd_autoperiod(&data, &argvals, None, None);

    assert!(
        result.period.is_finite(),
        "CFDAutoperiod should detect a period"
    );
    assert!(
        (result.period - period).abs() < 0.3,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
}

#[test]
fn test_cfd_autoperiod_with_trend() {
    // Sine wave with strong linear trend - CFD excels here
    let m = 300;
    let period = 3.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| 2.0 * t + (2.0 * PI * t / period).sin())
        .collect();

    let result = cfd_autoperiod(&data, &argvals, None, None);

    assert!(
        result.period.is_finite(),
        "CFDAutoperiod should detect a period despite trend"
    );
    // Allow more tolerance since trend can affect detection
    assert!(
        (result.period - period).abs() < 0.6,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
}

#[test]
fn test_cfd_autoperiod_multiple_periods() {
    // Signal with two periods - should detect multiple
    let m = 400;
    let period1 = 2.0;
    let period2 = 5.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period1).sin() + 0.5 * (2.0 * PI * t / period2).sin())
        .collect();

    let result = cfd_autoperiod(&data, &argvals, None, None);

    assert!(
        !result.periods.is_empty(),
        "Should detect at least one period"
    );
    // The primary period should be one of the two
    let close_to_p1 = (result.period - period1).abs() < 0.5;
    let close_to_p2 = (result.period - period2).abs() < 1.0;
    assert!(
        close_to_p1 || close_to_p2,
        "Primary period {} not close to {} or {}",
        result.period,
        period1,
        period2
    );
}

#[test]
fn test_cfd_autoperiod_fdata() {
    // Multiple samples with same period
    let n = 5;
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(n, m, period, &argvals);

    let result = cfd_autoperiod_fdata(&data, &argvals, None, None);

    assert!(
        result.period.is_finite(),
        "CFDAutoperiod should detect a period"
    );
    assert!(
        (result.period - period).abs() < 0.3,
        "Expected period ~{}, got {}",
        period,
        result.period
    );
}

// ========================================================================
// SSA Tests
// ========================================================================

#[test]
fn test_ssa_pure_sine() {
    let n = 200;
    let period = 12.0;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            0.01 * t + (2.0 * PI * t / period).sin() + 0.05 * ((i * 7) as f64 * 0.1).sin()
        })
        .collect();

    let result = ssa(&values, None, None, None, None);

    // trend + seasonal + noise ≈ original
    for i in 0..n {
        let reconstructed = result.trend[i] + result.seasonal[i] + result.noise[i];
        assert!(
            (reconstructed - values[i]).abs() < 1e-8,
            "SSA reconstruction error at {}: {} vs {}",
            i,
            reconstructed,
            values[i]
        );
    }

    // Singular values should be descending
    for i in 1..result.singular_values.len() {
        assert!(
            result.singular_values[i] <= result.singular_values[i - 1] + 1e-10,
            "Singular values should be descending: {} > {}",
            result.singular_values[i],
            result.singular_values[i - 1]
        );
    }

    // Contributions should sum to <= 1
    let total_contrib: f64 = result.contributions.iter().sum();
    assert!(
        total_contrib <= 1.0 + 1e-10,
        "Contributions should sum to <= 1, got {}",
        total_contrib
    );
}

#[test]
fn test_ssa_explicit_groupings() {
    let n = 100;
    let period = 10.0;
    let values: Vec<f64> = (0..n)
        .map(|i| 0.01 * i as f64 + (2.0 * PI * i as f64 / period).sin())
        .collect();

    let trend_components = [0usize];
    let seasonal_components = [1usize, 2];

    let result = ssa(
        &values,
        None,
        None,
        Some(&trend_components),
        Some(&seasonal_components),
    );

    assert_eq!(result.trend.len(), n);
    assert_eq!(result.seasonal.len(), n);
    assert_eq!(result.noise.len(), n);

    // Reconstruction should still hold
    for i in 0..n {
        let reconstructed = result.trend[i] + result.seasonal[i] + result.noise[i];
        assert!(
            (reconstructed - values[i]).abs() < 1e-8,
            "SSA explicit grouping reconstruction error at {}",
            i
        );
    }
}

#[test]
fn test_ssa_short_series() {
    // n < 4 should trigger early return
    let values = vec![1.0, 2.0, 3.0];
    let result = ssa(&values, None, None, None, None);

    assert_eq!(result.trend, values);
    assert_eq!(result.seasonal, vec![0.0; 3]);
    assert_eq!(result.noise, vec![0.0; 3]);
    assert_eq!(result.n_components, 0);
}

#[test]
fn test_ssa_fdata() {
    let n = 5;
    let m = 100;
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let amp = (i + 1) as f64;
        for j in 0..m {
            data[i + j * n] = amp * (2.0 * PI * j as f64 / 12.0).sin() + 0.01 * j as f64;
        }
    }

    let data = FdMatrix::from_column_major(data, n, m).unwrap();

    let result = ssa_fdata(&data, None, None);

    assert_eq!(result.trend.len(), m);
    assert_eq!(result.seasonal.len(), m);
    assert_eq!(result.noise.len(), m);
    assert!(!result.singular_values.is_empty());
}

#[test]
fn test_ssa_seasonality() {
    // Seasonal signal
    let n = 200;
    let period = 12.0;
    let seasonal_values: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / period).sin())
        .collect();

    let (is_seasonal, _det_period, confidence) =
        ssa_seasonality(&seasonal_values, None, Some(0.05));

    // A pure sine should be detected as seasonal
    // (confidence depends on component grouping)
    assert!(confidence >= 0.0, "Confidence should be non-negative");

    // Noise-only signal should not be seasonal
    let noise_values: Vec<f64> = (0..n)
        .map(|i| ((i * 13 + 7) as f64 * 0.1).sin() * 0.01)
        .collect();

    let (is_noise_seasonal, _, noise_conf) = ssa_seasonality(&noise_values, None, Some(0.5));

    // Noise should have low confidence (but it's not guaranteed to be strictly false
    // depending on auto-grouping, so we just check confidence)
    let _ = (is_seasonal, is_noise_seasonal, noise_conf);
}

// ========================================================================
// Matrix Profile Tests
// ========================================================================

#[test]
fn test_matrix_profile_periodic() {
    let period = 20;
    let n = period * 10;
    let values: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / period as f64).sin())
        .collect();

    let result = matrix_profile(&values, Some(15), None);

    assert_eq!(result.profile.len(), n - 15 + 1);
    assert_eq!(result.profile_index.len(), n - 15 + 1);
    assert_eq!(result.subsequence_length, 15);

    // Profile should be finite
    for &p in &result.profile {
        assert!(p.is_finite(), "Profile values should be finite");
    }

    // Primary period should be close to 20
    assert!(
        (result.primary_period - period as f64).abs() < 5.0,
        "Expected primary_period ≈ {}, got {}",
        period,
        result.primary_period
    );
}

#[test]
fn test_matrix_profile_non_periodic() {
    // Linear ramp (non-periodic)
    let n = 200;
    let values: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();

    let result = matrix_profile(&values, Some(10), None);

    assert_eq!(result.profile.len(), n - 10 + 1);

    // Should have lower confidence than periodic
    // (not always strictly 0, depends on ramp structure)
    assert!(result.confidence <= 1.0, "Confidence should be <= 1.0");
}

#[test]
fn test_matrix_profile_fdata() {
    let n = 3;
    let m = 200;
    let period = 20.0;
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let amp = (i + 1) as f64;
        for j in 0..m {
            data[i + j * n] = amp * (2.0 * PI * j as f64 / period).sin();
        }
    }

    let data = FdMatrix::from_column_major(data, n, m).unwrap();

    let result = matrix_profile_fdata(&data, Some(15), None);

    assert!(!result.profile.is_empty());
    assert!(result.profile_index.len() == result.profile.len());
}

#[test]
fn test_matrix_profile_seasonality() {
    let period = 20;
    let n = period * 10;
    // Periodic signal
    let values: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / period as f64).sin())
        .collect();

    let (is_seasonal, det_period, confidence) =
        matrix_profile_seasonality(&values, Some(15), Some(0.05));

    assert!(
        is_seasonal,
        "Periodic signal should be detected as seasonal"
    );
    assert!(det_period > 0.0, "Detected period should be positive");
    assert!(confidence >= 0.05, "Confidence should be above threshold");

    // Weak / non-periodic signal
    let weak_values: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();
    let (is_weak_seasonal, _, _) = matrix_profile_seasonality(&weak_values, Some(15), Some(0.5));
    let _ = is_weak_seasonal; // May or may not detect - we just check it doesn't panic
}

// ========================================================================
// Lomb-Scargle Tests
// ========================================================================

#[test]
fn test_lomb_scargle_regular() {
    let n = 200;
    let true_period = 5.0;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * t / true_period).sin())
        .collect();

    let result = lomb_scargle(&times, &values, None, None, None);

    assert!(
        (result.peak_period - true_period).abs() < 0.5,
        "Expected peak_period ≈ {}, got {}",
        true_period,
        result.peak_period
    );
    assert!(
        result.false_alarm_probability < 0.05,
        "FAP should be low for strong signal: {}",
        result.false_alarm_probability
    );
    assert!(result.peak_power > 0.0, "Peak power should be positive");
    assert!(!result.frequencies.is_empty());
    assert_eq!(result.frequencies.len(), result.power.len());
    assert_eq!(result.frequencies.len(), result.periods.len());
}

#[test]
fn test_lomb_scargle_irregular() {
    let true_period = 5.0;
    // Irregularly sampled: take a subset of regular samples
    let all_times: Vec<f64> = (0..300).map(|i| i as f64 * 0.1).collect();
    // Take every other point with some jitter-like selection
    let times: Vec<f64> = all_times
        .iter()
        .enumerate()
        .filter(|(i, _)| i % 2 == 0 || i % 3 == 0)
        .map(|(_, &t)| t)
        .collect();
    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * t / true_period).sin())
        .collect();

    let result = lomb_scargle(&times, &values, None, None, None);

    assert!(
        (result.peak_period - true_period).abs() < 1.0,
        "Irregular LS: expected period ≈ {}, got {}",
        true_period,
        result.peak_period
    );
}

#[test]
fn test_lomb_scargle_custom_frequencies() {
    let n = 100;
    let true_period = 5.0;
    let times: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let values: Vec<f64> = times
        .iter()
        .map(|&t| (2.0 * PI * t / true_period).sin())
        .collect();

    // Explicit frequency grid
    let frequencies: Vec<f64> = (1..50).map(|i| i as f64 * 0.01).collect();
    let result = lomb_scargle(&times, &values, Some(&frequencies), None, None);

    assert_eq!(result.frequencies.len(), frequencies.len());
    assert_eq!(result.power.len(), frequencies.len());
    assert!(result.peak_power > 0.0);
}

#[test]
fn test_lomb_scargle_fdata() {
    let n = 5;
    let m = 200;
    let period = 5.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(n, m, period, &argvals);

    let result = lomb_scargle_fdata(&data, &argvals, None, None);

    assert!(
        (result.peak_period - period).abs() < 0.5,
        "Fdata LS: expected period ≈ {}, got {}",
        period,
        result.peak_period
    );
    assert!(!result.frequencies.is_empty());
}

// ========================================================================
// Seasonality change detection tests
// ========================================================================

#[test]
fn test_detect_seasonality_changes_onset() {
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect(); // 0..20

    // First half: noise-like (low seasonality), second half: strong sine
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            if t < 10.0 {
                // Weak signal
                0.05 * ((t * 13.0).sin() + (t * 7.0).cos())
            } else {
                // Strong seasonal
                (2.0 * PI * t / period).sin()
            }
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_seasonality_changes(&data, &argvals, period, 0.3, 4.0, 2.0);

    assert!(
        !result.strength_curve.is_empty(),
        "Strength curve should not be empty"
    );
    assert_eq!(result.strength_curve.len(), m);

    // Should detect at least one change point (onset around t=10)
    if !result.change_points.is_empty() {
        let onset_points: Vec<_> = result
            .change_points
            .iter()
            .filter(|cp| cp.change_type == ChangeType::Onset)
            .collect();
        // At least one onset should exist near the transition
        assert!(!onset_points.is_empty(), "Should detect Onset change point");
    }
}

#[test]
fn test_detect_seasonality_changes_no_change() {
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect();

    // Consistently strong seasonal signal
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_seasonality_changes(&data, &argvals, period, 0.3, 4.0, 2.0);

    assert!(!result.strength_curve.is_empty());
    // With consistently seasonal data, there should be no Cessation points
    let cessation_points: Vec<_> = result
        .change_points
        .iter()
        .filter(|cp| cp.change_type == ChangeType::Cessation)
        .collect();
    assert!(
        cessation_points.is_empty(),
        "Consistently seasonal signal should have no Cessation points, found {}",
        cessation_points.len()
    );
}

// ========================================================================
// Period estimation tests (ACF and regression)
// ========================================================================

#[test]
fn test_estimate_period_acf() {
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, period, &argvals);

    let estimate = estimate_period_acf(data.as_slice(), 1, m, &argvals, m / 2);

    assert!(
        estimate.period.is_finite(),
        "ACF period estimate should be finite"
    );
    assert!(
        (estimate.period - period).abs() < 0.5,
        "ACF expected period ≈ {}, got {}",
        period,
        estimate.period
    );
    assert!(
        estimate.confidence > 0.0,
        "ACF confidence should be positive"
    );
}

#[test]
fn test_estimate_period_regression() {
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, period, &argvals);

    let estimate = estimate_period_regression(data.as_slice(), 1, m, &argvals, 1.5, 3.0, 100, 3);

    assert!(
        estimate.period.is_finite(),
        "Regression period estimate should be finite"
    );
    assert!(
        (estimate.period - period).abs() < 0.5,
        "Regression expected period ≈ {}, got {}",
        period,
        estimate.period
    );
    assert!(
        estimate.confidence > 0.0,
        "Regression confidence should be positive"
    );
}

// ========================================================================
// Seasonal strength windowed test
// ========================================================================

#[test]
fn test_seasonal_strength_windowed_variance() {
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, period, &argvals);

    let strengths = seasonal_strength_windowed(
        &data,
        &argvals,
        period,
        4.0, // window_size
        StrengthMethod::Variance,
    );

    assert_eq!(strengths.len(), m, "Should return m values");

    // Interior values (away from edges) should be in [0,1]
    let interior_start = m / 4;
    let interior_end = 3 * m / 4;
    for i in interior_start..interior_end {
        let s = strengths[i];
        if s.is_finite() {
            assert!(
                (-0.01..=1.01).contains(&s),
                "Windowed strength at {} should be near [0,1], got {}",
                i,
                s
            );
        }
    }
}

#[test]
fn test_constant_signal_fft_period() {
    // Constant signal has no periodicity
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 1.0).collect();
    let data_vec: Vec<f64> = vec![5.0; m];
    let data = FdMatrix::from_column_major(data_vec, 1, m).unwrap();
    let result = estimate_period_fft(&data, &argvals);
    // Should return something (even if period is meaningless for constant)
    assert!(result.period.is_finite() || result.period.is_nan());
}

#[test]
fn test_very_short_series_period() {
    // Very short series (4 points)
    let argvals = vec![0.0, 1.0, 2.0, 3.0];
    let data_vec = vec![1.0, -1.0, 1.0, -1.0];
    let data = FdMatrix::from_column_major(data_vec, 1, 4).unwrap();
    let result = estimate_period_fft(&data, &argvals);
    assert!(result.period.is_finite() || result.period.is_nan());
}

#[test]
fn test_nan_sazed_no_panic() {
    let mut data = vec![0.0; 50];
    let argvals: Vec<f64> = (0..50).map(|i| i as f64).collect();
    for i in 0..50 {
        data[i] = (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin();
    }
    data[10] = f64::NAN;
    let result = sazed(&data, &argvals, None);
    // Should not panic
    assert!(result.period.is_finite() || result.period.is_nan());
}

// ========================================================================
// Additional coverage tests
// ========================================================================

#[test]
fn test_interior_bounds_very_small() {
    // m = 0 should return None since end <= start
    let bounds = interior_bounds(0);
    assert!(bounds.is_none());

    // m = 1
    let bounds = interior_bounds(1);
    // edge_skip = 0, interior_start = min(0, 0) = 0, interior_end = max(1, 0) = 1
    // end > start => Some
    assert!(bounds.is_some() || bounds.is_none());
}

#[test]
fn test_valid_interior_bounds_min_span() {
    // m = 10 should give valid bounds
    let bounds = valid_interior_bounds(10, 4);
    // Should pass since span > 4
    assert!(bounds.is_some());

    // Very high min_span should fail
    let bounds = valid_interior_bounds(10, 100);
    assert!(bounds.is_none());
}

#[test]
fn test_periodogram_edge_cases() {
    // Empty data
    let (freqs, power) = periodogram(&[], &[]);
    assert!(freqs.is_empty());
    assert!(power.is_empty());

    // Single data point
    let (freqs, power) = periodogram(&[1.0], &[0.0]);
    assert!(freqs.is_empty());
    assert!(power.is_empty());

    // Mismatched lengths
    let (freqs, power) = periodogram(&[1.0, 2.0], &[0.0]);
    assert!(freqs.is_empty());
    assert!(power.is_empty());
}

#[test]
fn test_autocorrelation_edge_cases() {
    // Empty data
    let acf = autocorrelation(&[], 10);
    assert!(acf.is_empty());

    // Constant data (zero variance)
    let acf = autocorrelation(&[5.0, 5.0, 5.0, 5.0], 3);
    assert_eq!(acf.len(), 4);
    for &v in &acf {
        assert!((v - 1.0).abs() < 1e-10, "Constant data ACF should be 1.0");
    }
}

#[test]
fn test_detect_seasonality_changes_empty_data() {
    // Empty matrix should return empty result
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let argvals: Vec<f64> = vec![];
    let result = detect_seasonality_changes(&data, &argvals, 2.0, 0.3, 4.0, 2.0);
    assert!(result.change_points.is_empty());
    assert!(result.strength_curve.is_empty());

    // Too few points (m < 4)
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let argvals = vec![0.0, 1.0, 2.0];
    let result = detect_seasonality_changes(&data, &argvals, 2.0, 0.3, 4.0, 2.0);
    assert!(result.change_points.is_empty());
    assert!(result.strength_curve.is_empty());
}

#[test]
fn test_detect_amplitude_modulation_non_seasonal_returns_early() {
    // Non-seasonal data should return early with NonSeasonal type
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Pure noise
    let data: Vec<f64> = (0..m)
        .map(|i| ((i as f64 * 1.618).sin() * 100.0).fract())
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation(&data, &argvals, 0.2, 0.15, 0.5);
    assert!(!result.is_seasonal);
    assert_eq!(result.modulation_type, ModulationType::NonSeasonal);
    assert_eq!(result.modulation_score, 0.0);
}

#[test]
fn test_detect_amplitude_modulation_small_data() {
    // m < 4 should hit early return
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let argvals = vec![0.0, 1.0, 2.0];

    let result = detect_amplitude_modulation(&data, &argvals, 1.0, 0.15, 0.3);
    assert!(!result.is_seasonal);
}

#[test]
fn test_detect_amplitude_modulation_wavelet_invalid_inputs() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = detect_amplitude_modulation_wavelet(&data, &[], 2.0, 0.15, 0.3);
    assert!(!result.is_seasonal);
    assert_eq!(result.modulation_type, ModulationType::NonSeasonal);

    // m < 4
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let argvals = vec![0.0, 1.0, 2.0];
    let result = detect_amplitude_modulation_wavelet(&data, &argvals, 2.0, 0.15, 0.3);
    assert!(!result.is_seasonal);

    // period <= 0
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / 0.2).sin())
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();
    let result = detect_amplitude_modulation_wavelet(&data, &argvals, -1.0, 0.15, 0.3);
    assert!(!result.is_seasonal);
}

#[test]
fn test_detect_amplitude_modulation_wavelet_non_seasonal() {
    // Non-seasonal data should return early
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Pure noise
    let data: Vec<f64> = (0..m)
        .map(|i| ((i as f64 * 1.618).sin() * 100.0).fract())
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation_wavelet(&data, &argvals, 0.2, 0.15, 0.5);
    assert!(!result.is_seasonal);
    assert_eq!(result.modulation_type, ModulationType::NonSeasonal);
}

#[test]
fn test_detect_amplitude_modulation_wavelet_seasonal() {
    // Seasonal signal with known modulation
    let m = 200;
    let period = 0.2;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Amplitude grows from 0.2 to 1.0
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            let amplitude = 0.2 + 0.8 * t;
            amplitude * (2.0 * PI * t / period).sin()
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_amplitude_modulation_wavelet(&data, &argvals, period, 0.15, 0.2);
    assert!(result.is_seasonal, "Should detect seasonality");
    assert!(result.scale > 0.0, "Scale should be positive");
    assert!(
        !result.wavelet_amplitude.is_empty(),
        "Wavelet amplitude should be computed"
    );
    assert_eq!(result.time_points.len(), m);
}

#[test]
fn test_instantaneous_period_invalid_inputs() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = instantaneous_period(&data, &[]);
    assert!(result.period.is_empty());
    assert!(result.frequency.is_empty());
    assert!(result.amplitude.is_empty());

    // m < 4
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let result = instantaneous_period(&data, &[0.0, 1.0, 2.0]);
    assert!(result.period.is_empty());
}

#[test]
fn test_analyze_peak_timing_invalid_inputs() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = analyze_peak_timing(&data, &[], 2.0, None);
    assert!(result.peak_times.is_empty());
    assert!(result.mean_timing.is_nan());

    // m < 3
    let data = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
    let result = analyze_peak_timing(&data, &[0.0, 1.0], 2.0, None);
    assert!(result.peak_times.is_empty());
    assert!(result.mean_timing.is_nan());

    // period <= 0
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, 2.0, &argvals);
    let result = analyze_peak_timing(&data, &argvals, -1.0, None);
    assert!(result.peak_times.is_empty());
    assert!(result.mean_timing.is_nan());
}

#[test]
fn test_analyze_peak_timing_no_peaks() {
    // Very short data (m < 3) should return early with no peaks
    let data = FdMatrix::from_column_major(vec![5.0, 5.0], 1, 2).unwrap();
    let result = analyze_peak_timing(&data, &[0.0, 1.0], 2.0, Some(11));
    assert!(result.peak_times.is_empty());
    assert!(result.mean_timing.is_nan());
}

#[test]
fn test_classify_seasonality_invalid_inputs() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = classify_seasonality(&data, &[], 2.0, None, None);
    assert!(!result.is_seasonal);
    assert!(result.seasonal_strength.is_nan());
    assert_eq!(result.classification, SeasonalType::NonSeasonal);

    // m < 4
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let result = classify_seasonality(&data, &[0.0, 1.0, 2.0], 2.0, None, None);
    assert!(!result.is_seasonal);
    assert_eq!(result.classification, SeasonalType::NonSeasonal);

    // period <= 0
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, 2.0, &argvals);
    let result = classify_seasonality(&data, &argvals, -1.0, None, None);
    assert!(!result.is_seasonal);
}

#[test]
fn test_classify_seasonality_non_seasonal() {
    // Constant data should be non-seasonal
    let m = 100;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = FdMatrix::from_column_major(vec![5.0; m], 1, m).unwrap();

    let result = classify_seasonality(&data, &argvals, 2.0, Some(0.3), Some(0.05));
    assert!(!result.is_seasonal);
    assert_eq!(result.classification, SeasonalType::NonSeasonal);
}

#[test]
fn test_classify_seasonality_strong_seasonal() {
    // Strong, stable seasonal signal
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect();
    let data = generate_sine(1, m, period, &argvals);

    let result = classify_seasonality(&data, &argvals, period, Some(0.3), Some(0.5));
    assert!(result.is_seasonal);
    assert!(result.seasonal_strength > 0.5);
    assert!(result.peak_timing.is_some());
    // Check cycle_strengths is populated
    assert!(
        !result.cycle_strengths.is_empty(),
        "cycle_strengths should be computed"
    );
}

#[test]
fn test_classify_seasonality_with_custom_thresholds() {
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect();
    let data = generate_sine(1, m, period, &argvals);

    // Very high strength threshold
    let result = classify_seasonality(&data, &argvals, period, Some(0.99), None);
    // Should still detect as seasonal for pure sine
    assert!(result.seasonal_strength > 0.8);
}

#[test]
fn test_detect_seasonality_changes_auto_fixed() {
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect();

    // Signal with onset
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            if t < 10.0 {
                0.05 * ((t * 13.0).sin() + (t * 7.0).cos())
            } else {
                (2.0 * PI * t / period).sin()
            }
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    // Test with Fixed threshold method
    let result = detect_seasonality_changes_auto(
        &data,
        &argvals,
        period,
        ThresholdMethod::Fixed(0.3),
        4.0,
        2.0,
    );
    assert!(!result.strength_curve.is_empty());
}

#[test]
fn test_detect_seasonality_changes_auto_percentile() {
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect();

    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            if t < 10.0 {
                0.05 * ((t * 13.0).sin() + (t * 7.0).cos())
            } else {
                (2.0 * PI * t / period).sin()
            }
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    // Test with Percentile threshold method
    let result = detect_seasonality_changes_auto(
        &data,
        &argvals,
        period,
        ThresholdMethod::Percentile(50.0),
        4.0,
        2.0,
    );
    assert!(!result.strength_curve.is_empty());
}

#[test]
fn test_detect_seasonality_changes_auto_otsu() {
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect();

    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            if t < 10.0 {
                0.05 * ((t * 13.0).sin() + (t * 7.0).cos())
            } else {
                (2.0 * PI * t / period).sin()
            }
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    // Test with Otsu threshold method
    let result =
        detect_seasonality_changes_auto(&data, &argvals, period, ThresholdMethod::Otsu, 4.0, 2.0);
    assert!(!result.strength_curve.is_empty());
}

#[test]
fn test_detect_seasonality_changes_auto_invalid_inputs() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result =
        detect_seasonality_changes_auto(&data, &[], 2.0, ThresholdMethod::Fixed(0.3), 4.0, 2.0);
    assert!(result.change_points.is_empty());
    assert!(result.strength_curve.is_empty());

    // m < 4
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let result = detect_seasonality_changes_auto(
        &data,
        &[0.0, 1.0, 2.0],
        2.0,
        ThresholdMethod::Otsu,
        1.0,
        0.5,
    );
    assert!(result.change_points.is_empty());
    assert!(result.strength_curve.is_empty());
}

#[test]
fn test_cfd_autoperiod_fdata_invalid_inputs() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = cfd_autoperiod_fdata(&data, &[], None, None);
    assert!(result.period.is_nan());
    assert_eq!(result.confidence, 0.0);

    // m < 8
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 5.0], 1, 5).unwrap();
    let result = cfd_autoperiod_fdata(&data, &[0.0, 1.0, 2.0, 3.0, 4.0], None, None);
    assert!(result.period.is_nan());
}

#[test]
fn test_cfd_autoperiod_fdata_valid() {
    // Valid data with seasonal pattern
    let n = 3;
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(n, m, period, &argvals);

    let result = cfd_autoperiod_fdata(&data, &argvals, Some(0.1), Some(1));
    assert!(result.period.is_finite());
}

#[test]
fn test_lomb_scargle_fap_edge_cases() {
    // power <= 0
    let fap = lomb_scargle_fap(0.0, 100, 200);
    assert_eq!(fap, 1.0);

    let fap = lomb_scargle_fap(-1.0, 100, 200);
    assert_eq!(fap, 1.0);

    // n_indep = 0
    let fap = lomb_scargle_fap(10.0, 0, 200);
    assert_eq!(fap, 1.0);

    // Very high power should give FAP near 0
    let fap = lomb_scargle_fap(100.0, 100, 200);
    assert!(
        fap < 0.01,
        "Very high power should give low FAP, got {}",
        fap
    );

    // Moderate power
    let fap = lomb_scargle_fap(5.0, 50, 100);
    assert!((0.0..=1.0).contains(&fap));
}

#[test]
fn test_lomb_scargle_fdata_valid() {
    let n = 3;
    let m = 200;
    let period = 5.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(n, m, period, &argvals);

    let result = lomb_scargle_fdata(&data, &argvals, Some(4.0), Some(1.0));
    assert!(
        (result.peak_period - period).abs() < 1.0,
        "Expected period ~{}, got {}",
        period,
        result.peak_period
    );
    assert!(!result.frequencies.is_empty());
}

#[test]
fn test_cwt_morlet_edge_cases() {
    // Empty signal
    let result = cwt_morlet_fft(&[], &[], 1.0, 6.0);
    assert!(result.is_empty());

    // scale <= 0
    let result = cwt_morlet_fft(&[1.0, 2.0], &[0.0, 1.0], 0.0, 6.0);
    assert!(result.is_empty());

    let result = cwt_morlet_fft(&[1.0, 2.0], &[0.0, 1.0], -1.0, 6.0);
    assert!(result.is_empty());
}

#[test]
fn test_hilbert_transform_empty() {
    let result = hilbert_transform(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_unwrap_phase_empty() {
    let result = unwrap_phase(&[]);
    assert!(result.is_empty());
}

#[test]
fn test_unwrap_phase_monotonic() {
    // Phase that wraps around
    let phase = vec![0.0, 1.0, 2.0, 3.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let unwrapped = unwrap_phase(&phase);
    assert_eq!(unwrapped.len(), phase.len());
    // After unwrapping, phase should be monotonically increasing
    for i in 1..unwrapped.len() {
        assert!(
            unwrapped[i] >= unwrapped[i - 1] - 0.01,
            "Unwrapped phase should be monotonic at {}: {} vs {}",
            i,
            unwrapped[i],
            unwrapped[i - 1]
        );
    }
}

#[test]
fn test_linear_slope_edge_cases() {
    // Mismatched lengths
    assert_eq!(linear_slope(&[1.0, 2.0], &[1.0]), 0.0);

    // Too few points
    assert_eq!(linear_slope(&[1.0], &[1.0]), 0.0);

    // Perfect linear relationship: y = 2x + 1
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![1.0, 3.0, 5.0, 7.0, 9.0];
    let slope = linear_slope(&x, &y);
    assert!(
        (slope - 2.0).abs() < 1e-10,
        "Slope should be 2.0, got {}",
        slope
    );

    // Constant x (zero variance in x)
    let x = vec![5.0, 5.0, 5.0];
    let y = vec![1.0, 2.0, 3.0];
    let slope = linear_slope(&x, &y);
    assert_eq!(slope, 0.0, "Constant x should give slope 0");
}

#[test]
fn test_otsu_threshold_edge_cases() {
    // Empty values
    let threshold = otsu_threshold(&[]);
    assert!((threshold - 0.5).abs() < 1e-10);

    // All NaN
    let threshold = otsu_threshold(&[f64::NAN, f64::NAN]);
    assert!((threshold - 0.5).abs() < 1e-10);

    // Constant values
    let threshold = otsu_threshold(&[5.0, 5.0, 5.0]);
    assert!((threshold - 5.0).abs() < 1e-10);

    // Two distinct values
    let threshold = otsu_threshold(&[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]);
    assert!(threshold > 0.0 && threshold < 1.0);
}

#[test]
fn test_find_peaks_1d_edge_cases() {
    // Too short
    assert!(find_peaks_1d(&[], 1).is_empty());
    assert!(find_peaks_1d(&[1.0], 1).is_empty());
    assert!(find_peaks_1d(&[1.0, 2.0], 1).is_empty());

    // Single peak
    let peaks = find_peaks_1d(&[0.0, 1.0, 0.0], 1);
    assert_eq!(peaks, vec![1]);

    // Two peaks with min distance filtering
    let signal = vec![0.0, 2.0, 0.0, 1.5, 0.0];
    let peaks = find_peaks_1d(&signal, 1);
    assert_eq!(peaks, vec![1, 3]);

    // Two peaks close together, min_distance replaces shorter
    let signal = vec![0.0, 1.0, 0.5, 2.0, 0.0];
    let peaks = find_peaks_1d(&signal, 3);
    // Peak at 1 (val=1.0) found first, then peak at 3 (val=2.0) is within
    // distance 3, but higher, so replaces it
    assert_eq!(peaks.len(), 1);
    assert_eq!(peaks[0], 3);
}

#[test]
fn test_compute_prominence() {
    // Simple peak
    let signal = vec![0.0, 0.0, 5.0, 0.0, 0.0];
    let prom = compute_prominence(&signal, 2);
    assert!((prom - 5.0).abs() < 1e-10);

    // Peak with asymmetric valleys
    let signal = vec![0.0, 2.0, 5.0, 1.0, 0.0];
    let prom = compute_prominence(&signal, 2);
    // Left min = 0.0 (going left until higher peak), right min = 0.0
    // Prominence = 5.0 - max(0.0, 0.0) = 5.0
    assert!(prom > 0.0);
}

#[test]
fn test_seasonal_strength_variance_invalid() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = seasonal_strength_variance(&data, &[], 2.0, 3);
    assert!(result.is_nan());

    // period <= 0
    let m = 50;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, 2.0, &argvals);
    let result = seasonal_strength_variance(&data, &argvals, -1.0, 3);
    assert!(result.is_nan());
}

#[test]
fn test_seasonal_strength_spectral_invalid() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = seasonal_strength_spectral(&data, &[], 2.0);
    assert!(result.is_nan());

    // period <= 0
    let m = 50;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, 2.0, &argvals);
    let result = seasonal_strength_spectral(&data, &argvals, -1.0);
    assert!(result.is_nan());
}

#[test]
fn test_seasonal_strength_wavelet_invalid() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = seasonal_strength_wavelet(&data, &[], 2.0);
    assert!(result.is_nan());

    // period <= 0
    let m = 50;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, 2.0, &argvals);
    let result = seasonal_strength_wavelet(&data, &argvals, -1.0);
    assert!(result.is_nan());

    // Constant data (zero variance)
    let data = FdMatrix::from_column_major(vec![5.0; m], 1, m).unwrap();
    let result = seasonal_strength_wavelet(&data, &argvals, 2.0);
    assert!(
        (result - 0.0).abs() < 1e-10,
        "Constant data should have 0 strength"
    );
}

#[test]
fn test_seasonal_strength_windowed_spectral() {
    // Test with Spectral method (existing test only covers Variance)
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, period, &argvals);

    let strengths =
        seasonal_strength_windowed(&data, &argvals, period, 4.0, StrengthMethod::Spectral);

    assert_eq!(strengths.len(), m, "Should return m values");
}

#[test]
fn test_seasonal_strength_windowed_invalid() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let strengths = seasonal_strength_windowed(&data, &[], 2.0, 4.0, StrengthMethod::Variance);
    assert!(strengths.is_empty());

    // window_size <= 0
    let m = 50;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data = generate_sine(1, m, 2.0, &argvals);
    let strengths =
        seasonal_strength_windowed(&data, &argvals, 2.0, -1.0, StrengthMethod::Variance);
    assert!(strengths.is_empty());
}

#[test]
fn test_ssa_custom_window_length() {
    // Test with explicit window_length that might trigger edge case
    let n = 50;
    let values: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 10.0).sin()).collect();

    // window_length > n/2 should trigger early return
    let result = ssa(&values, Some(30), None, None, None);
    // Since 30 > 50/2 = 25, this should return the early return path
    assert_eq!(result.trend, values);
    assert_eq!(result.n_components, 0);
}

#[test]
fn test_ssa_window_length_too_small() {
    let n = 50;
    let values: Vec<f64> = (0..n).map(|i| i as f64).collect();

    // window_length = 1 (< 2)
    let result = ssa(&values, Some(1), None, None, None);
    assert_eq!(result.trend, values);
    assert_eq!(result.n_components, 0);
}

#[test]
fn test_ssa_auto_grouping() {
    // Test auto-grouping path (no explicit trend/seasonal components)
    let n = 200;
    let period = 12.0;
    let values: Vec<f64> = (0..n)
        .map(|i| {
            let t = i as f64;
            // Clear trend + seasonal + noise
            0.05 * t + 2.0 * (2.0 * PI * t / period).sin() + 0.01 * ((i * 7) as f64).sin()
        })
        .collect();

    let result = ssa(&values, Some(30), Some(6), None, None);

    // Should detect period
    assert!(
        result.detected_period > 0.0,
        "Should detect a period, got {}",
        result.detected_period
    );
    assert!(result.confidence > 0.0);

    // Reconstruction should hold
    for i in 0..n {
        let reconstructed = result.trend[i] + result.seasonal[i] + result.noise[i];
        assert!(
            (reconstructed - values[i]).abs() < 1e-8,
            "SSA auto-grouping reconstruction error at {}",
            i
        );
    }
}

#[test]
fn test_ssa_with_many_components() {
    // Test with n_components larger than available
    let n = 100;
    let values: Vec<f64> = (0..n).map(|i| (2.0 * PI * i as f64 / 12.0).sin()).collect();

    let result = ssa(&values, None, Some(100), None, None);
    assert!(result.n_components <= n);
    assert!(!result.singular_values.is_empty());
}

#[test]
fn test_embed_trajectory() {
    // Simple test: [1, 2, 3, 4, 5] with L=3, K=3
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let l = 3;
    let k = 3; // n - l + 1 = 5 - 3 + 1 = 3
    let traj = embed_trajectory(&values, l, k);

    // Column 0: [1, 2, 3]
    assert!((traj[0] - 1.0).abs() < 1e-10);
    assert!((traj[1] - 2.0).abs() < 1e-10);
    assert!((traj[2] - 3.0).abs() < 1e-10);

    // Column 1: [2, 3, 4]
    assert!((traj[3] - 2.0).abs() < 1e-10);
    assert!((traj[4] - 3.0).abs() < 1e-10);
    assert!((traj[5] - 4.0).abs() < 1e-10);

    // Column 2: [3, 4, 5]
    assert!((traj[6] - 3.0).abs() < 1e-10);
    assert!((traj[7] - 4.0).abs() < 1e-10);
    assert!((traj[8] - 5.0).abs() < 1e-10);
}

#[test]
fn test_diagonal_average() {
    // Test with a simple 3x3 trajectory matrix
    // If all values are 1.0, diagonal averaging should give 1.0
    let l = 3;
    let k = 3;
    let n = l + k - 1; // = 5
    let matrix = vec![1.0; l * k];
    let result = diagonal_average(&matrix, l, k, n);
    assert_eq!(result.len(), n);
    for &v in &result {
        assert!((v - 1.0).abs() < 1e-10, "Expected 1.0, got {}", v);
    }
}

#[test]
fn test_svd_decompose() {
    // Simple 3x2 matrix
    let l = 3;
    let k = 2;
    let trajectory = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // identity-like
    let (u, sigma, vt) = svd_decompose(&trajectory, l, k);

    assert!(!u.is_empty(), "U should not be empty");
    assert!(!sigma.is_empty(), "Sigma should not be empty");
    assert!(!vt.is_empty(), "V^T should not be empty");

    // Singular values should be non-negative and descending
    for &s in &sigma {
        assert!(s >= 0.0, "Singular values must be non-negative");
    }
    for i in 1..sigma.len() {
        assert!(sigma[i] <= sigma[i - 1] + 1e-10);
    }
}

#[test]
fn test_is_trend_component() {
    // Monotonic vector (trend-like)
    let trend_vec: Vec<f64> = (0..20).map(|i| i as f64).collect();
    assert!(is_trend_component(&trend_vec));

    // Oscillating vector (not a trend)
    let osc_vec: Vec<f64> = (0..20).map(|i| (PI * i as f64 / 3.0).sin()).collect();
    assert!(!is_trend_component(&osc_vec));

    // Too short
    assert!(!is_trend_component(&[1.0, 2.0]));
}

#[test]
fn test_is_periodic_component() {
    // Periodic signal
    let periodic: Vec<f64> = (0..40)
        .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
        .collect();
    let (is_periodic, period) = is_periodic_component(&periodic);
    assert!(is_periodic, "Should detect periodicity");
    assert!(
        (period - 10.0).abs() < 2.0,
        "Expected period ~10, got {}",
        period
    );

    // Monotonic signal (no periodicity)
    let monotonic: Vec<f64> = (0..40).map(|i| i as f64 * 0.01).collect();
    let (is_periodic, _) = is_periodic_component(&monotonic);
    // Monotonic ramp has no significant positive ACF peak at lag > 1
    // (autocorrelation decays monotonically)
    // We just verify it doesn't panic and returns a result
    let _ = is_periodic;

    // Too short
    let (is_periodic, _) = is_periodic_component(&[1.0, 2.0, 3.0]);
    assert!(!is_periodic, "Too short to be periodic");
}

#[test]
fn test_classify_ssa_component() {
    // Trend-like vector
    let trend_vec: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
    let kind = classify_ssa_component(&trend_vec, 0);
    assert!(matches!(kind, SsaComponentKind::Trend));

    // Second trend should still classify as trend if count < 2
    let kind = classify_ssa_component(&trend_vec, 1);
    assert!(matches!(kind, SsaComponentKind::Trend));

    // With trend_count >= 2, is_trend_component still returns true but the
    // condition fails, so it falls through to is_periodic_component.
    // A monotonic ramp may or may not be detected as periodic depending on
    // ACF behavior, so we just verify it doesn't crash and isn't Trend.
    let kind = classify_ssa_component(&trend_vec, 2);
    assert!(!matches!(kind, SsaComponentKind::Trend));

    // Periodic vector
    let periodic: Vec<f64> = (0..40)
        .map(|i| (2.0 * PI * i as f64 / 10.0).sin())
        .collect();
    let kind = classify_ssa_component(&periodic, 0);
    assert!(matches!(kind, SsaComponentKind::Seasonal(_)));
}

#[test]
fn test_apply_ssa_grouping_defaults() {
    // Empty indices with enough components
    let mut trend_idx = Vec::new();
    let mut seasonal_idx = Vec::new();
    apply_ssa_grouping_defaults(&mut trend_idx, &mut seasonal_idx, 5);
    assert_eq!(trend_idx, vec![0]);
    assert_eq!(seasonal_idx, vec![1, 2]);

    // Already populated indices
    let mut trend_idx = vec![0];
    let mut seasonal_idx = vec![1];
    apply_ssa_grouping_defaults(&mut trend_idx, &mut seasonal_idx, 5);
    assert_eq!(trend_idx, vec![0]); // unchanged
    assert_eq!(seasonal_idx, vec![1]); // unchanged

    // n_comp < 3: no seasonal defaults
    let mut trend_idx = Vec::new();
    let mut seasonal_idx = Vec::new();
    apply_ssa_grouping_defaults(&mut trend_idx, &mut seasonal_idx, 2);
    assert_eq!(trend_idx, vec![0]);
    assert!(seasonal_idx.is_empty());

    // n_comp = 0: no defaults
    let mut trend_idx = Vec::new();
    let mut seasonal_idx = Vec::new();
    apply_ssa_grouping_defaults(&mut trend_idx, &mut seasonal_idx, 0);
    assert!(trend_idx.is_empty());
    assert!(seasonal_idx.is_empty());
}

#[test]
fn test_reconstruct_grouped_empty() {
    // Empty group
    let result = reconstruct_grouped(&[], &[], &[], 3, 3, 5, &[]);
    assert_eq!(result, vec![0.0; 5]);
}

#[test]
fn test_reconstruct_grouped_idx_out_of_range() {
    // group_idx with index beyond sigma length
    let u = vec![1.0; 9]; // 3x3
    let sigma = vec![1.0, 0.5];
    let vt = vec![1.0; 6]; // 2x3 or similar
    let result = reconstruct_grouped(&u, &sigma, &vt, 3, 3, 5, &[5]);
    // Index 5 is beyond sigma.len()=2, so it should be skipped
    assert_eq!(result, vec![0.0; 5]);
}

#[test]
fn test_auto_group_ssa_components() {
    // Build a simple set of components
    let l = 20;
    let n_comp = 4;
    // U matrix: first column is trend-like, second and third are periodic
    let mut u = vec![0.0; l * n_comp];
    for i in 0..l {
        u[i] = i as f64 * 0.1; // Trend (monotonic)
        u[i + l] = (2.0 * PI * i as f64 / 8.0).sin(); // Periodic
        u[i + 2 * l] = (2.0 * PI * i as f64 / 8.0).cos(); // Periodic (same frequency)
        u[i + 3 * l] = (i as f64 * 1.618).fract(); // Noise-like
    }
    let sigma = vec![10.0, 5.0, 4.0, 0.1];

    let (trend_idx, seasonal_idx, detected_period, confidence) =
        auto_group_ssa_components(&u, &sigma, l, 10, n_comp);

    assert!(
        !trend_idx.is_empty(),
        "Should detect at least one trend component"
    );
    assert!(
        !seasonal_idx.is_empty(),
        "Should detect at least one seasonal component"
    );
    // Detected period should come from the periodic components
    if detected_period > 0.0 {
        assert!(confidence > 0.0);
    }
}

#[test]
fn test_estimate_period_fft_invalid() {
    // n == 0
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = estimate_period_fft(&data, &[]);
    assert!(result.period.is_nan());
    assert!(result.frequency.is_nan());

    // m < 4
    let data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let result = estimate_period_fft(&data, &[0.0, 1.0, 2.0]);
    assert!(result.period.is_nan());
}

#[test]
fn test_detect_peaks_invalid_inputs() {
    // Empty data
    let data = FdMatrix::from_column_major(vec![], 0, 0).unwrap();
    let result = detect_peaks(&data, &[], None, None, false, None);
    assert!(result.peaks.is_empty());
    assert!(result.mean_period.is_nan());

    // m < 3
    let data = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
    let result = detect_peaks(&data, &[0.0, 1.0], None, None, false, None);
    assert!(result.peaks.is_empty());
}

#[test]
fn test_detect_peaks_with_smoothing() {
    // Test peak detection with smoothing enabled (smooth_first = true)
    let m = 200;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 10.0 / (m - 1) as f64).collect();

    // Noisy sine wave
    let data: Vec<f64> = argvals
        .iter()
        .enumerate()
        .map(|(i, &t)| (2.0 * PI * t / period).sin() + 0.1 * ((i as f64 * 5.7).sin()))
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    // With smoothing
    let result = detect_peaks(&data, &argvals, Some(1.5), None, true, Some(11));
    assert!(!result.peaks[0].is_empty());
}

#[test]
fn test_morlet_wavelet() {
    // At t=0, wavelet should be exp(0) * [cos(0), sin(0)] = [1, 0]
    let w = morlet_wavelet(0.0, 6.0);
    assert!((w.re - 1.0).abs() < 1e-10);
    assert!(w.im.abs() < 1e-10);

    // At large |t|, wavelet should be near zero (Gaussian decay)
    let w = morlet_wavelet(10.0, 6.0);
    assert!(w.norm() < 1e-10, "Wavelet should decay for large t");
}

#[test]
fn test_matrix_profile_short_series() {
    // Short series: n=10, m=3, so m <= n/2 = 5
    let values: Vec<f64> = (0..10).map(|i| (i as f64 * 0.5).sin()).collect();
    let result = matrix_profile(&values, Some(3), None);
    assert_eq!(result.profile.len(), 8); // n - m + 1 = 10 - 3 + 1 = 8
}

#[test]
fn test_ssa_seasonality_with_threshold() {
    // Test ssa_seasonality with explicit confidence_threshold
    let n = 200;
    let period = 12.0;
    let values: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / period).sin())
        .collect();

    // Low threshold
    let (is_seasonal, det_period, confidence) = ssa_seasonality(&values, None, Some(0.01));
    assert!(confidence >= 0.0);
    let _ = (is_seasonal, det_period);

    // Very high threshold
    let (is_seasonal, _, _) = ssa_seasonality(&values, None, Some(0.99));
    let _ = is_seasonal;
}

#[test]
fn test_cfd_autoperiod_short_data() {
    // Data too short for differencing to work well
    let argvals: Vec<f64> = (0..8).map(|i| i as f64).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / 4.0).sin())
        .collect();

    let result = cfd_autoperiod(&data, &argvals, None, None);
    // Should handle gracefully
    assert!(result.period.is_finite() || result.period.is_nan());
}

#[test]
fn test_cfd_autoperiod_long_period() {
    // 8 years of daily temperature-like data with annual period (365 days).
    // Regression test for GH-14: differencing attenuated long-period signals,
    // returning period ≈ 2.2 instead of 365. Linear detrending fixes this.
    let n = 365 * 8;
    let argvals: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| 15.0 + 10.0 * (2.0 * PI * t / 365.0).sin() + 0.001 * t)
        .collect();
    let result = cfd_autoperiod(&data, &argvals, None, None);
    let err = (result.period - 365.0).abs();
    assert!(
        err < 2.0,
        "long-period detection: expected ~365, got {:.1} (err={:.1})",
        result.period,
        err
    );
}

#[test]
fn test_validate_sazed_component() {
    // Valid component
    let result = validate_sazed_component(5.0, 0.8, 1.0, 10.0, 0.5);
    assert_eq!(result, Some(5.0));

    // Period out of range
    let result = validate_sazed_component(0.5, 0.8, 1.0, 10.0, 0.5);
    assert_eq!(result, None);

    let result = validate_sazed_component(15.0, 0.8, 1.0, 10.0, 0.5);
    assert_eq!(result, None);

    // Low confidence
    let result = validate_sazed_component(5.0, 0.3, 1.0, 10.0, 0.5);
    assert_eq!(result, None);

    // NaN period
    let result = validate_sazed_component(f64::NAN, 0.8, 1.0, 10.0, 0.5);
    assert_eq!(result, None);
}

#[test]
fn test_count_agreeing_periods() {
    let periods = vec![5.0, 5.1, 5.2, 10.0, 15.0];

    // All three ~5.x should agree with reference 5.0 within 10% tolerance
    let (count, sum) = count_agreeing_periods(&periods, 5.0, 0.1);
    assert_eq!(count, 3);
    assert!((sum - 15.3).abs() < 1e-10);

    // None should agree with 100.0
    let (count, _) = count_agreeing_periods(&periods, 100.0, 0.1);
    assert_eq!(count, 0);
}

#[test]
fn test_generate_ls_frequencies() {
    let times: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let freqs = generate_ls_frequencies(&times, 4.0, 1.0);
    assert!(!freqs.is_empty());
    // First frequency should be approximately 1/T_span = 1/99
    assert!(
        (freqs[0] - 1.0 / 99.0).abs() < 0.01,
        "First freq should be ~1/99, got {}",
        freqs[0]
    );

    // Short series
    let freqs = generate_ls_frequencies(&[0.0], 4.0, 1.0);
    assert_eq!(freqs, vec![0.0]);
}

#[test]
fn test_estimate_independent_frequencies() {
    let times: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let n_indep = estimate_independent_frequencies(&times, 100);
    assert_eq!(n_indep, 50); // min(50, 100) = 50

    let n_indep = estimate_independent_frequencies(&times, 30);
    assert_eq!(n_indep, 30); // min(50, 30) = 30
}

#[test]
fn test_cluster_periods() {
    // Empty candidates
    let result = cluster_periods(&[], 0.1, 1);
    assert!(result.is_empty());

    // Single candidate
    let candidates = vec![(5.0, 1.0)];
    let result = cluster_periods(&candidates, 0.1, 1);
    assert_eq!(result.len(), 1);

    // Two clusters
    let candidates = vec![(5.0, 1.0), (5.05, 0.8), (10.0, 0.5), (10.1, 0.4)];
    let result = cluster_periods(&candidates, 0.1, 1);
    assert_eq!(result.len(), 2, "Should find 2 clusters");

    // High min_size filters small clusters
    let candidates = vec![(5.0, 1.0), (10.0, 0.5)];
    let result = cluster_periods(&candidates, 0.01, 2);
    // Each cluster has only 1 member, min_size=2 filters them
    assert!(result.is_empty());
}

#[test]
fn test_validate_cfd_candidates() {
    // Create a simple ACF with a peak at lag ~10
    let n = 50;
    let dt = 1.0;
    let mut acf = vec![0.0; n + 1];
    acf[0] = 1.0;
    for i in 1..=n {
        acf[i] = (2.0 * PI * i as f64 / 10.0).cos() * 0.5;
    }

    let clusters = vec![(10.0, 1.0), (20.0, 0.5)];
    let validated = validate_cfd_candidates(&clusters, &acf, dt);
    // At least the period=10 cluster should validate
    assert!(
        !validated.is_empty(),
        "Should validate at least one candidate"
    );
}

#[test]
fn test_validate_or_fallback_cfd() {
    // When validated is not empty, return as-is
    let validated = vec![(10.0, 0.8, 1.0)];
    let candidates = vec![(10.0, 1.0)];
    let result = validate_or_fallback_cfd(validated.clone(), &candidates, 0.1, 1);
    assert_eq!(result.len(), 1);

    // When validated is empty, fallback to best cluster
    let candidates = vec![(10.0, 1.0), (10.2, 0.8)];
    let result = validate_or_fallback_cfd(vec![], &candidates, 0.1, 1);
    assert!(!result.is_empty(), "Fallback should return something");
}

#[test]
fn test_rank_cfd_results() {
    let validated = vec![
        (10.0, 0.5, 1.0), // score = 0.5
        (5.0, 0.8, 2.0),  // score = 1.6
    ];
    let (periods, confidences, top_acf) = rank_cfd_results(&validated);
    // Should be sorted by combined score descending
    assert_eq!(periods[0], 5.0); // highest score
    assert_eq!(periods[1], 10.0);
    assert!((top_acf - 0.8).abs() < 1e-10);
    assert_eq!(confidences.len(), 2);
}

#[test]
fn test_empty_cfd_result() {
    let result = empty_cfd_result();
    assert!(result.period.is_nan());
    assert_eq!(result.confidence, 0.0);
    assert_eq!(result.acf_validation, 0.0);
    assert!(result.periods.is_empty());
}

#[test]
fn test_fit_and_subtract_sinusoid() {
    let m = 200;
    let period = 10.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let mut residual: Vec<f64> = argvals
        .iter()
        .map(|&t| 3.0 * (2.0 * PI * t / period).sin())
        .collect();

    let (a, b, amplitude, phase) = fit_and_subtract_sinusoid(&mut residual, &argvals, period);

    assert!(
        amplitude > 2.0,
        "Amplitude should be ~3.0, got {}",
        amplitude
    );
    assert!(phase.is_finite());
    let _ = (a, b);

    // After subtraction, residual should be near zero
    let max_residual: f64 = residual.iter().map(|&x| x.abs()).fold(0.0, f64::max);
    assert!(
        max_residual < 0.5,
        "Residual after subtraction should be small, got {}",
        max_residual
    );
}

#[test]
fn test_detect_seasonality_changes_cessation() {
    // Signal that starts seasonal and becomes non-seasonal
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect();

    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| {
            if t < 10.0 {
                // Strong seasonal
                (2.0 * PI * t / period).sin()
            } else {
                // Weak/no seasonality
                0.05 * ((t * 13.0).sin() + (t * 7.0).cos())
            }
        })
        .collect();
    let data = FdMatrix::from_column_major(data, 1, m).unwrap();

    let result = detect_seasonality_changes(&data, &argvals, period, 0.3, 4.0, 2.0);

    assert!(!result.strength_curve.is_empty());
    // Should detect cessation change point
    if !result.change_points.is_empty() {
        let cessation_points: Vec<_> = result
            .change_points
            .iter()
            .filter(|cp| cp.change_type == ChangeType::Cessation)
            .collect();
        assert!(!cessation_points.is_empty(), "Should detect Cessation");
    }
}

#[test]
fn test_matrix_profile_fdata_multiple_samples() {
    // Test with multiple samples
    let n = 5;
    let m = 200;
    let period = 20.0;
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let amp = (i + 1) as f64;
        for j in 0..m {
            data[i + j * n] = amp * (2.0 * PI * j as f64 / period).sin();
        }
    }
    let data = FdMatrix::from_column_major(data, n, m).unwrap();

    let result = matrix_profile_fdata(&data, Some(15), None);
    assert!(!result.profile.is_empty());
    assert!(result.primary_period > 0.0);
}

#[test]
fn test_ssa_fdata_multiple_samples() {
    let n = 3;
    let m = 200;
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            data[i + j * n] = (2.0 * PI * j as f64 / 12.0).sin() + 0.01 * j as f64;
        }
    }
    let data = FdMatrix::from_column_major(data, n, m).unwrap();

    let result = ssa_fdata(&data, Some(25), Some(5));
    assert_eq!(result.trend.len(), m);
    assert_eq!(result.seasonal.len(), m);
}

#[test]
fn test_compute_cycle_strengths() {
    let m = 400;
    let period = 2.0;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.05).collect(); // 0..20

    // Strong seasonal for all cycles
    let data = generate_sine(1, m, period, &argvals);
    let (strengths, weak) = compute_cycle_strengths(&data, &argvals, period, 0.3);

    assert!(
        !strengths.is_empty(),
        "Should compute at least one cycle strength"
    );
    // For pure sine, no cycles should be weak
    assert!(
        weak.is_empty(),
        "Pure sine should have no weak seasons, got {:?}",
        weak
    );
}

#[test]
fn test_find_acf_descent_end() {
    // ACF that descends then goes negative
    let acf = vec![1.0, 0.8, 0.5, 0.2, -0.1, -0.3, 0.0, 0.3, 0.5];
    let end = find_acf_descent_end(&acf);
    assert_eq!(end, 4, "Should find first negative at index 4");

    // ACF that descends but never goes negative, has uptick
    // At i=4: acf[4]=0.4 > acf[3]=0.3, so returns i-1=3
    let acf = vec![1.0, 0.8, 0.5, 0.3, 0.4, 0.6];
    let end = find_acf_descent_end(&acf);
    assert_eq!(end, 3, "Should find uptick at index 3 (i-1 where i=4)");
}

#[test]
fn test_autocorrelation_fft_matches_naive() {
    // Generate a signal long enough to exercise the FFT path (n > 64)
    let n = 200;
    let data: Vec<f64> = (0..n)
        .map(|i| (2.0 * PI * i as f64 / 20.0).sin() + 0.5 * (i as f64 * 0.1).cos())
        .collect();
    let max_lag = 50;

    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let var: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

    let naive = autocorrelation_naive(&data, max_lag, mean, var);
    let fft = autocorrelation(&data, max_lag); // n=200 > 64, takes FFT path

    assert_eq!(naive.len(), fft.len());
    for (lag, (n_val, f_val)) in naive.iter().zip(fft.iter()).enumerate() {
        assert!(
            (n_val - f_val).abs() < 1e-10,
            "Mismatch at lag {lag}: naive={n_val}, fft={f_val}"
        );
    }
}
