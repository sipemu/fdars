//! Example 10: Seasonal Analysis
//!
//! Demonstrates period estimation, peak detection, seasonal strength
//! measures, and seasonality change detection on synthetic signals
//! with known periods.

use fdars_core::matrix::FdMatrix;
use fdars_core::seasonal::{
    autoperiod, detect_peaks, detect_seasonality_changes, estimate_period_acf, estimate_period_fft,
    sazed, seasonal_strength_spectral, seasonal_strength_variance, PeriodEstimate,
};

fn print_period_estimate(label: &str, est: &PeriodEstimate) {
    println!(
        "  {label}: period={:.4}, freq={:.4}, power={:.4}, confidence={:.4}",
        est.period, est.frequency, est.power, est.confidence
    );
}

fn main() {
    println!("=== Example 10: Seasonal Analysis ===\n");

    // --- Generate synthetic signals ---
    let m = 200;
    let t: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect(); // 0..20, step 0.1

    // Signal 1: Pure sinusoid with period 2.0
    let signal_pure: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti / 2.0).sin())
        .collect();

    // Signal 2: Two harmonics
    let signal_harmonic: Vec<f64> = t
        .iter()
        .map(|&ti| {
            (2.0 * std::f64::consts::PI * ti / 2.0).sin()
                + 0.5 * (2.0 * std::f64::consts::PI * ti / 1.0).sin()
        })
        .collect();

    // Signal 3: Changing seasonality (strong then weak)
    let signal_changing: Vec<f64> = t
        .iter()
        .map(|&ti| {
            let amplitude = if ti < 10.0 { 1.0 } else { 0.2 };
            amplitude * (2.0 * std::f64::consts::PI * ti / 2.0).sin()
        })
        .collect();

    // For functions expecting functional data format (n=1 curve, m points)
    let n = 1;

    // Wrap signals as FdMatrix for migrated functions
    let mat_pure = FdMatrix::from_slice(&signal_pure, n, m).unwrap();
    let mat_harmonic = FdMatrix::from_slice(&signal_harmonic, n, m).unwrap();
    let mat_changing = FdMatrix::from_slice(&signal_changing, n, m).unwrap();

    // --- Section 1: FFT period estimation ---
    println!("--- FFT Period Estimation ---");
    let fft_pure = estimate_period_fft(&mat_pure, &t);
    let fft_harmonic = estimate_period_fft(&mat_harmonic, &t);
    print_period_estimate("Pure signal (true period=2.0)", &fft_pure);
    print_period_estimate("Two harmonics  (true period=2.0)", &fft_harmonic);

    // --- Section 2: ACF period estimation ---
    println!("\n--- ACF Period Estimation ---");
    let acf_pure = estimate_period_acf(&signal_pure, n, m, &t, m / 2);
    let acf_harmonic = estimate_period_acf(&signal_harmonic, n, m, &t, m / 2);
    print_period_estimate("Pure signal (true period=2.0)", &acf_pure);
    print_period_estimate("Two harmonics  (true period=2.0)", &acf_harmonic);

    // --- Section 3: Autoperiod ---
    // autoperiod takes (data, argvals, n_candidates, gradient_steps) and returns AutoperiodResult
    println!("\n--- Autoperiod ---");
    let auto_pure = autoperiod(&signal_pure, &t, None, None);
    let auto_harmonic = autoperiod(&signal_harmonic, &t, None, None);
    println!(
        "  Pure signal:    period={:.4}, confidence={:.4}",
        auto_pure.period, auto_pure.confidence
    );
    println!(
        "  Two harmonics:  period={:.4}, confidence={:.4}",
        auto_harmonic.period, auto_harmonic.confidence
    );

    // --- Section 4: SAZED ensemble method ---
    // sazed takes (data, argvals, tolerance) and returns SazedResult
    println!("\n--- SAZED Ensemble Detection ---");
    let sazed_pure = sazed(&signal_pure, &t, None);
    let sazed_harmonic = sazed(&signal_harmonic, &t, None);
    println!("  Pure signal:");
    println!(
        "    Period: {:.4}, Confidence: {:.4}",
        sazed_pure.period, sazed_pure.confidence
    );
    println!(
        "    Components: spectral={:.4}, acf_peak={:.4}, zero_cross={:.4}",
        sazed_pure.component_periods.spectral,
        sazed_pure.component_periods.acf_peak,
        sazed_pure.component_periods.zero_crossing
    );
    println!("  Two harmonics:");
    println!(
        "    Period: {:.4}, Confidence: {:.4}",
        sazed_harmonic.period, sazed_harmonic.confidence
    );

    // --- Section 5: Peak detection ---
    println!("\n--- Peak Detection ---");
    let peaks = detect_peaks(&mat_pure, &t, Some(1.0), None, false, None);
    println!("  Pure signal peaks: {} found", peaks.peaks[0].len());
    for (i, peak) in peaks.peaks[0].iter().enumerate().take(5) {
        println!(
            "    Peak {i}: time={:.2}, value={:.4}, prominence={:.4}",
            peak.time, peak.value, peak.prominence
        );
    }
    println!(
        "  Mean inter-peak period: {:.4} (true: 2.0)",
        peaks.mean_period
    );

    // --- Section 6: Seasonal strength ---
    println!("\n--- Seasonal Strength ---");
    let period = 2.0;
    let str_var_pure = seasonal_strength_variance(&mat_pure, &t, period, 3);
    let str_spec_pure = seasonal_strength_spectral(&mat_pure, &t, period);
    let str_var_changing = seasonal_strength_variance(&mat_changing, &t, period, 3);
    let str_spec_changing = seasonal_strength_spectral(&mat_changing, &t, period);

    println!("  Pure signal:     variance={str_var_pure:.4}, spectral={str_spec_pure:.4}");
    println!("  Changing signal: variance={str_var_changing:.4}, spectral={str_spec_changing:.4}");

    // --- Section 7: Seasonality change detection ---
    println!("\n--- Seasonality Change Detection ---");
    let changes = detect_seasonality_changes(&mat_changing, &t, period, 0.3, 4.0, 2.0);
    println!("  Change points found: {}", changes.change_points.len());
    for cp in &changes.change_points {
        println!(
            "    time={:.2}: {:?}, strength {:.4} -> {:.4}",
            cp.time, cp.change_type, cp.strength_before, cp.strength_after
        );
    }
    println!(
        "  Strength curve range: [{:.4}, {:.4}]",
        changes
            .strength_curve
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min),
        changes
            .strength_curve
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max)
    );

    println!("\n=== Done ===");
}
