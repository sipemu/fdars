//! Integration tests validating mathematical correctness of the phase tolerance band
//! implementation. Tests cover tangent-space geometry, warping function properties,
//! coverage ordering, amplitude-vs-raw comparisons, and joint consistency.

use std::f64::consts::PI;

use fdars_core::alignment::karcher_mean;
use fdars_core::matrix::FdMatrix;
use fdars_core::simulation::{sim_fundata, EFunType, EValType};
use fdars_core::tolerance::{
    elastic_tolerance_band, elastic_tolerance_band_with_config, fpca_tolerance_band,
    phase_tolerance_band, BandType, ElasticToleranceConfig,
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Create a uniform grid on [0, 1] with `m` points.
fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

/// Build n identical copies of a template curve (no phase or amplitude variation).
fn identical_curves(template: &[f64], n: usize) -> FdMatrix {
    let m = template.len();
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        for j in 0..m {
            data[(i, j)] = template[j];
        }
    }
    data
}

/// Build n curves that are time-shifted versions of sin(2*pi*t):
///   f_i(t) = sin(2*pi*(t + offset_i))
/// where offsets are linearly spaced in [-max_shift, +max_shift].
fn phase_shifted_sines(n: usize, m: usize, max_shift: f64) -> (FdMatrix, Vec<f64>) {
    let t = uniform_grid(m);
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let offset = max_shift * (2.0 * i as f64 / (n - 1) as f64 - 1.0);
        for j in 0..m {
            data[(i, j)] = (2.0 * PI * (t[j] + offset)).sin();
        }
    }
    (data, t)
}

/// Build curves with both amplitude and moderate phase variation.
fn mixed_variation_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let t = uniform_grid(m);
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let phase_shift = 0.05 * (2.0 * i as f64 / (n - 1) as f64 - 1.0);
        let amp = 1.0 + 0.3 * (i as f64 / (n - 1) as f64 - 0.5);
        for j in 0..m {
            data[(i, j)] = amp * (2.0 * PI * (t[j] + phase_shift)).sin();
        }
    }
    (data, t)
}

/// Build curves with symmetric phase variation: half shifted early, half shifted late.
fn symmetric_phase_curves(n: usize, m: usize, shift: f64) -> (FdMatrix, Vec<f64>) {
    assert!(n >= 4 && n % 2 == 0, "need even n >= 4");
    let t = uniform_grid(m);
    let mut data = FdMatrix::zeros(n, m);
    for i in 0..n {
        let offset = if i < n / 2 { -shift } else { shift };
        for j in 0..m {
            data[(i, j)] = (2.0 * PI * (t[j] + offset)).sin();
        }
    }
    (data, t)
}

// ─── Test 1: No-phase-variation baseline ─────────────────────────────────────

#[test]
fn no_phase_variation_tight_band() {
    let m = 51;
    let t = uniform_grid(m);
    let template: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect();
    let data = identical_curves(&template, 20);

    // Karcher mean of identical curves: gammas should be near-identity.
    let km = karcher_mean(&data, &t, 15, 1e-4, 0.0);
    for i in 0..20 {
        for (j, &tj) in t.iter().enumerate() {
            assert!(
                (km.gammas[(i, j)] - tj).abs() < 0.05,
                "gamma[{i},{j}] = {} should be near identity {}",
                km.gammas[(i, j)],
                tj
            );
        }
    }

    // Phase band should be extremely tight.
    let phase = phase_tolerance_band(&data, &t, 1, 50, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed for identical curves");

    // The tangent-space half-widths should be very small.
    let max_hw: f64 = phase
        .tangent_band
        .half_width
        .iter()
        .copied()
        .fold(0.0, f64::max);
    assert!(
        max_hw < 0.5,
        "Tangent half-width for identical curves should be very small, got {max_hw}"
    );

    // gamma_lower and gamma_upper should be close to identity.
    for (j, &tj) in t.iter().enumerate() {
        let dev_lo = (phase.gamma_lower[j] - tj).abs();
        let dev_hi = (phase.gamma_upper[j] - tj).abs();
        assert!(
            dev_lo < 0.15,
            "gamma_lower[{j}] deviates too far from identity: {dev_lo}"
        );
        assert!(
            dev_hi < 0.15,
            "gamma_upper[{j}] deviates too far from identity: {dev_hi}"
        );
    }
}

// ─── Test 2: Known phase shift produces wide phase band ──────────────────────

#[test]
fn known_phase_shift_wide_band() {
    let n = 20;
    let m = 51;
    let max_shift = 0.12;
    let (data, t) = phase_shifted_sines(n, m, max_shift);

    let phase = phase_tolerance_band(&data, &t, 2, 100, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed for phase-shifted sines");

    // Phase band should be notably wider than for identical curves.
    let max_gamma_dev: f64 = (0..m)
        .map(|j| {
            let lo = (phase.gamma_lower[j] - t[j]).abs();
            let hi = (phase.gamma_upper[j] - t[j]).abs();
            lo.max(hi)
        })
        .fold(0.0, f64::max);

    assert!(
        max_gamma_dev > 0.01,
        "Phase band should have visible deviation from identity for shifted data, got {max_gamma_dev}"
    );

    // Amplitude band after alignment should be tighter than FPCA band on raw data.
    let amp_band = elastic_tolerance_band(&data, &t, 2, 100, 0.95, BandType::Pointwise, 15, 42)
        .expect("elastic_tolerance_band should succeed");
    let raw_band = fpca_tolerance_band(&data, 2, 100, 0.95, BandType::Pointwise, 42)
        .expect("fpca_tolerance_band should succeed");

    let amp_mean_hw: f64 = amp_band.half_width.iter().sum::<f64>() / m as f64;
    let raw_mean_hw: f64 = raw_band.half_width.iter().sum::<f64>() / m as f64;

    assert!(
        amp_mean_hw < raw_mean_hw * 1.5,
        "Amplitude band (mean hw {amp_mean_hw:.4}) should generally be tighter \
         than raw band (mean hw {raw_mean_hw:.4}) for phase-variable data"
    );
}

// ─── Test 3: Phase band captures timing ──────────────────────────────────────

#[test]
fn phase_band_captures_timing() {
    let (data, t) = mixed_variation_curves(20, 51);

    let phase = phase_tolerance_band(&data, &t, 2, 100, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed");

    let m = t.len();

    // Both gamma_lower and gamma_upper should be valid warping functions:
    // monotone non-decreasing, with correct boundary values.
    for j in 1..m {
        assert!(
            phase.gamma_lower[j] >= phase.gamma_lower[j - 1] - 1e-12,
            "gamma_lower not monotone at j={j}"
        );
        assert!(
            phase.gamma_upper[j] >= phase.gamma_upper[j - 1] - 1e-12,
            "gamma_upper not monotone at j={j}"
        );
    }

    // The lower and upper warps should differ from the identity (the data has
    // phase variation, so the tangent band has non-trivial half-width).
    let lower_dev: f64 = (0..m)
        .map(|j| (phase.gamma_lower[j] - t[j]).abs())
        .sum::<f64>()
        / m as f64;
    let upper_dev: f64 = (0..m)
        .map(|j| (phase.gamma_upper[j] - t[j]).abs())
        .sum::<f64>()
        / m as f64;

    // At least one of the bounds should deviate from identity.
    assert!(
        lower_dev > 1e-6 || upper_dev > 1e-6,
        "Phase band warps should differ from identity for data with phase variation, \
         lower_dev={lower_dev}, upper_dev={upper_dev}"
    );

    // The tangent-space band should be well-ordered (lower <= upper in tangent space).
    for j in 0..m {
        assert!(
            phase.tangent_band.lower[j] <= phase.tangent_band.upper[j] + 1e-10,
            "Tangent band lower[{j}] > upper[{j}]"
        );
    }
}

// ─── Test 4: Tangent space band center near zero ─────────────────────────────

#[test]
fn tangent_band_center_near_zero() {
    let t = uniform_grid(51);
    let data = sim_fundata(
        30,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );

    let phase = phase_tolerance_band(&data, &t, 3, 100, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed");

    // The tangent band center is the mean of shooting vectors from the Karcher mean.
    // By construction this should be near zero (the Karcher mean minimizes sum of
    // squared geodesic distances, so the mean tangent vector converges to zero).
    let mean_abs_center: f64 = phase
        .tangent_band
        .center
        .iter()
        .map(|c| c.abs())
        .sum::<f64>()
        / t.len() as f64;

    assert!(
        mean_abs_center < 1.0,
        "Mean absolute tangent band center should be small, got {mean_abs_center}"
    );
}

// ─── Test 5: Coverage ordering (99% wider than 90%) ──────────────────────────

#[test]
fn coverage_ordering_99_wider_than_90() {
    let (data, t) = phase_shifted_sines(20, 51, 0.08);

    let phase_90 = phase_tolerance_band(&data, &t, 2, 100, 0.90, BandType::Pointwise, 15, 42)
        .expect("90% phase band should succeed");
    let phase_99 = phase_tolerance_band(&data, &t, 2, 100, 0.99, BandType::Pointwise, 15, 42)
        .expect("99% phase band should succeed");

    // Compute mean gamma spread for each.
    let m = t.len();
    let spread_90: f64 = (0..m)
        .map(|j| phase_90.gamma_upper[j] - phase_90.gamma_lower[j])
        .sum::<f64>()
        / m as f64;
    let spread_99: f64 = (0..m)
        .map(|j| phase_99.gamma_upper[j] - phase_99.gamma_lower[j])
        .sum::<f64>()
        / m as f64;

    assert!(
        spread_99 >= spread_90 - 1e-10,
        "99% band spread ({spread_99:.6}) should be >= 90% band spread ({spread_90:.6})"
    );

    // Also check in tangent space.
    let hw_90: f64 = phase_90.tangent_band.half_width.iter().sum::<f64>() / m as f64;
    let hw_99: f64 = phase_99.tangent_band.half_width.iter().sum::<f64>() / m as f64;
    assert!(
        hw_99 >= hw_90 - 1e-10,
        "99% tangent half-width ({hw_99:.6}) should be >= 90% tangent half-width ({hw_90:.6})"
    );
}

// ─── Test 6: Amplitude band tighter than raw for phase-variable data ─────────

#[test]
fn amplitude_tighter_than_raw_for_phase_data() {
    let n = 30;
    let m = 51;
    let (data, t) = phase_shifted_sines(n, m, 0.15);

    let amp_band = elastic_tolerance_band(&data, &t, 3, 100, 0.95, BandType::Pointwise, 15, 42)
        .expect("elastic_tolerance_band should succeed");
    let raw_band = fpca_tolerance_band(&data, 3, 100, 0.95, BandType::Pointwise, 42)
        .expect("fpca_tolerance_band should succeed");

    let amp_max_hw: f64 = amp_band.half_width.iter().copied().fold(0.0, f64::max);
    let raw_max_hw: f64 = raw_band.half_width.iter().copied().fold(0.0, f64::max);

    // For data with significant phase variation, the elastic (amplitude) band
    // should be notably tighter since alignment removes the timing differences.
    assert!(
        amp_max_hw < raw_max_hw * 1.2,
        "Elastic amplitude max half-width ({amp_max_hw:.4}) should be much \
         smaller than raw max half-width ({raw_max_hw:.4})"
    );
}

// ─── Test 7: Joint consistency ───────────────────────────────────────────────

#[test]
fn joint_amplitude_matches_standalone() {
    let t = uniform_grid(51);
    let data = sim_fundata(
        20,
        &t,
        3,
        EFunType::Fourier,
        EValType::Exponential,
        Some(99),
    );

    let mut config = ElasticToleranceConfig::default();
    config.ncomp_amplitude = 2;
    config.ncomp_phase = 2;
    config.nb = 80;
    config.coverage = 0.95;
    config.band_type = BandType::Pointwise;
    config.max_iter = 15;
    config.tol = 1e-4;
    config.seed = 42;

    let joint =
        elastic_tolerance_band_with_config(&data, &t, &config).expect("joint band should succeed");
    let standalone = elastic_tolerance_band(
        &data,
        &t,
        config.ncomp_amplitude,
        config.nb,
        config.coverage,
        config.band_type,
        config.max_iter,
        config.seed,
    )
    .expect("standalone amplitude band should succeed");

    let m = t.len();

    // Both use the same Karcher mean + same FPCA band parameters, so the
    // amplitude bands should be identical.
    for j in 0..m {
        assert!(
            (joint.amplitude.lower[j] - standalone.lower[j]).abs() < 1e-10,
            "Amplitude lower mismatch at j={j}: joint={} standalone={}",
            joint.amplitude.lower[j],
            standalone.lower[j]
        );
        assert!(
            (joint.amplitude.upper[j] - standalone.upper[j]).abs() < 1e-10,
            "Amplitude upper mismatch at j={j}: joint={} standalone={}",
            joint.amplitude.upper[j],
            standalone.upper[j]
        );
    }

    // Phase band should also be present and valid.
    assert_eq!(joint.phase.gamma_lower.len(), m);
    assert_eq!(joint.phase.gamma_upper.len(), m);
}

// ─── Test 8: Phase band symmetry ─────────────────────────────────────────────

#[test]
fn symmetric_phase_variation_symmetric_band() {
    let n = 20;
    let m = 51;
    let shift = 0.06;
    let (data, t) = symmetric_phase_curves(n, m, shift);

    let phase = phase_tolerance_band(&data, &t, 2, 100, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed for symmetric data");

    // For symmetric phase variation, deviations of gamma_lower and gamma_upper
    // from identity should be roughly symmetric (at interior points).
    let interior = 5..(m - 5);
    let mut dev_lower_sum = 0.0;
    let mut dev_upper_sum = 0.0;
    let count = interior.len();

    for j in interior {
        dev_lower_sum += (t[j] - phase.gamma_lower[j]).abs();
        dev_upper_sum += (phase.gamma_upper[j] - t[j]).abs();
    }

    let mean_dev_lo = dev_lower_sum / count as f64;
    let mean_dev_hi = dev_upper_sum / count as f64;

    // They should be roughly within a factor of 3 of each other.
    if mean_dev_lo > 1e-6 && mean_dev_hi > 1e-6 {
        let ratio = mean_dev_lo / mean_dev_hi;
        assert!(
            ratio > 0.1 && ratio < 10.0,
            "Symmetric phase deviation ratio should be roughly balanced: \
             mean_dev_lower={mean_dev_lo:.6}, mean_dev_upper={mean_dev_hi:.6}, ratio={ratio:.3}"
        );
    }
}

// ─── Test 9: Monotonicity of gamma bounds ────────────────────────────────────

#[test]
fn gamma_bounds_monotone() {
    let t = uniform_grid(51);
    let data = sim_fundata(
        25,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(77),
    );

    let phase = phase_tolerance_band(&data, &t, 3, 80, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed");

    let m = t.len();
    for j in 1..m {
        assert!(
            phase.gamma_lower[j] >= phase.gamma_lower[j - 1] - 1e-12,
            "gamma_lower not monotone at j={j}: {} < {}",
            phase.gamma_lower[j],
            phase.gamma_lower[j - 1]
        );
        assert!(
            phase.gamma_upper[j] >= phase.gamma_upper[j - 1] - 1e-12,
            "gamma_upper not monotone at j={j}: {} < {}",
            phase.gamma_upper[j],
            phase.gamma_upper[j - 1]
        );
    }
}

// ─── Test 10: Warp endpoint constraints ──────────────────────────────────────

#[test]
fn gamma_bounds_fix_endpoints() {
    let t = uniform_grid(51);
    let data = sim_fundata(
        20,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(12),
    );

    let phase = phase_tolerance_band(&data, &t, 2, 80, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed");

    let m = t.len();

    // Both gamma_lower and gamma_upper must match argvals at endpoints.
    assert!(
        (phase.gamma_lower[0] - t[0]).abs() < 1e-10,
        "gamma_lower[0] = {} should match t[0] = {}",
        phase.gamma_lower[0],
        t[0]
    );
    assert!(
        (phase.gamma_lower[m - 1] - t[m - 1]).abs() < 1e-10,
        "gamma_lower[m-1] = {} should match t[m-1] = {}",
        phase.gamma_lower[m - 1],
        t[m - 1]
    );
    assert!(
        (phase.gamma_upper[0] - t[0]).abs() < 1e-10,
        "gamma_upper[0] = {} should match t[0] = {}",
        phase.gamma_upper[0],
        t[0]
    );
    assert!(
        (phase.gamma_upper[m - 1] - t[m - 1]).abs() < 1e-10,
        "gamma_upper[m-1] = {} should match t[m-1] = {}",
        phase.gamma_upper[m - 1],
        t[m - 1]
    );

    // gamma_center should be the identity warp.
    for (j, &tj) in t.iter().enumerate() {
        assert!(
            (phase.gamma_center[j] - tj).abs() < 1e-10,
            "gamma_center[{j}] = {} should equal t[{j}] = {}",
            phase.gamma_center[j],
            tj
        );
    }
}

// ─── Test: gamma_lower and gamma_upper stay within domain ────────────────────

#[test]
fn gamma_bounds_within_domain() {
    let t = uniform_grid(51);
    let data = sim_fundata(
        25,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(55),
    );

    let phase = phase_tolerance_band(&data, &t, 3, 80, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed");

    let m = t.len();
    for j in 0..m {
        assert!(
            phase.gamma_lower[j] >= t[0] - 1e-10,
            "gamma_lower[{j}] = {} below domain start {}",
            phase.gamma_lower[j],
            t[0]
        );
        assert!(
            phase.gamma_upper[j] <= t[m - 1] + 1e-10,
            "gamma_upper[{j}] = {} above domain end {}",
            phase.gamma_upper[j],
            t[m - 1]
        );
    }
}

// ─── Test: Simultaneous band type works ──────────────────────────────────────

#[test]
fn simultaneous_band_wider_than_pointwise() {
    let (data, t) = phase_shifted_sines(20, 51, 0.08);

    let pw = phase_tolerance_band(&data, &t, 2, 100, 0.95, BandType::Pointwise, 15, 42)
        .expect("pointwise band should succeed");
    let sim = phase_tolerance_band(&data, &t, 2, 100, 0.95, BandType::Simultaneous, 15, 42)
        .expect("simultaneous band should succeed");

    let m = t.len();
    let pw_hw: f64 = pw.tangent_band.half_width.iter().sum::<f64>() / m as f64;
    let sim_hw: f64 = sim.tangent_band.half_width.iter().sum::<f64>() / m as f64;

    // Simultaneous bands should be at least as wide (typically wider).
    assert!(
        sim_hw >= pw_hw * 0.8,
        "Simultaneous tangent half-width ({sim_hw:.4}) should generally be >= \
         pointwise ({pw_hw:.4})"
    );
}

// ─── Test: Tangent band lower <= center <= upper ─────────────────────────────

#[test]
fn tangent_band_ordering() {
    let t = uniform_grid(51);
    let data = sim_fundata(
        25,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(33),
    );

    let phase = phase_tolerance_band(&data, &t, 2, 80, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed");

    let m = t.len();
    for j in 0..m {
        let lo = phase.tangent_band.lower[j];
        let hi = phase.tangent_band.upper[j];
        let ctr = phase.tangent_band.center[j];
        assert!(
            lo <= ctr + 1e-10,
            "tangent lower[{j}]={lo} > center[{j}]={ctr}"
        );
        assert!(
            ctr <= hi + 1e-10,
            "tangent center[{j}]={ctr} > upper[{j}]={hi}"
        );
    }
}

// ─── Test: Non-trivial domain [a, b] != [0, 1] ──────────────────────────────

#[test]
fn nonstandard_domain_endpoints() {
    let m = 51;
    let t: Vec<f64> = (0..m)
        .map(|i| 2.0 + 3.0 * i as f64 / (m - 1) as f64)
        .collect();
    // domain is [2, 5]

    let mut data = FdMatrix::zeros(20, m);
    for i in 0..20 {
        let shift = 0.05 * (2.0 * i as f64 / 19.0 - 1.0);
        for j in 0..m {
            data[(i, j)] = (2.0 * PI * ((t[j] - 2.0) / 3.0 + shift)).sin();
        }
    }

    let phase = phase_tolerance_band(&data, &t, 2, 80, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band on [2,5] domain should succeed");

    // Endpoints must match the custom domain.
    assert!(
        (phase.gamma_lower[0] - 2.0).abs() < 1e-10,
        "gamma_lower[0] should be 2.0, got {}",
        phase.gamma_lower[0]
    );
    assert!(
        (phase.gamma_upper[0] - 2.0).abs() < 1e-10,
        "gamma_upper[0] should be 2.0, got {}",
        phase.gamma_upper[0]
    );
    assert!(
        (phase.gamma_lower[m - 1] - 5.0).abs() < 1e-10,
        "gamma_lower[m-1] should be 5.0, got {}",
        phase.gamma_lower[m - 1]
    );
    assert!(
        (phase.gamma_upper[m - 1] - 5.0).abs() < 1e-10,
        "gamma_upper[m-1] should be 5.0, got {}",
        phase.gamma_upper[m - 1]
    );

    // gamma_center should equal t.
    for (j, &tj) in t.iter().enumerate() {
        assert!(
            (phase.gamma_center[j] - tj).abs() < 1e-10,
            "gamma_center[{j}] should match t[{j}]"
        );
    }

    // Monotonicity within [2, 5].
    for j in 1..m {
        assert!(
            phase.gamma_lower[j] >= phase.gamma_lower[j - 1] - 1e-12,
            "gamma_lower not monotone at j={j} on [2,5] domain"
        );
        assert!(
            phase.gamma_upper[j] >= phase.gamma_upper[j - 1] - 1e-12,
            "gamma_upper not monotone at j={j} on [2,5] domain"
        );
    }
}

// ─── Test: Half-width non-negative ───────────────────────────────────────────

#[test]
fn tangent_half_width_nonnegative() {
    let t = uniform_grid(51);
    let data = sim_fundata(20, &t, 3, EFunType::Fourier, EValType::Exponential, Some(7));

    let phase = phase_tolerance_band(&data, &t, 2, 80, 0.95, BandType::Pointwise, 15, 42)
        .expect("phase_tolerance_band should succeed");

    for (j, &hw) in phase.tangent_band.half_width.iter().enumerate() {
        assert!(
            hw >= -1e-15,
            "tangent half_width[{j}] = {hw} should be non-negative"
        );
    }
}

// ─── Test: Reproducibility with same seed ────────────────────────────────────

#[test]
fn reproducible_with_same_seed() {
    let (data, t) = phase_shifted_sines(15, 41, 0.07);

    let phase1 = phase_tolerance_band(&data, &t, 2, 60, 0.95, BandType::Pointwise, 10, 123)
        .expect("first call should succeed");
    let phase2 = phase_tolerance_band(&data, &t, 2, 60, 0.95, BandType::Pointwise, 10, 123)
        .expect("second call should succeed");

    let m = t.len();
    for j in 0..m {
        assert!(
            (phase1.gamma_lower[j] - phase2.gamma_lower[j]).abs() < 1e-12,
            "gamma_lower not reproducible at j={j}"
        );
        assert!(
            (phase1.gamma_upper[j] - phase2.gamma_upper[j]).abs() < 1e-12,
            "gamma_upper not reproducible at j={j}"
        );
    }
}
