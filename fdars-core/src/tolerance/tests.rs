use super::*;
use crate::fdata::mean_1d;
use crate::matrix::FdMatrix;
use crate::simulation::{sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn make_test_data() -> FdMatrix {
    let m = 50;
    let t = uniform_grid(m);
    sim_fundata(
        50,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    )
}

// ── normal_quantile tests ──

#[test]
fn test_normal_quantile_symmetry() {
    use super::helpers::normal_quantile;
    for &p in &[0.1, 0.2, 0.3, 0.4] {
        let q_low = normal_quantile(p);
        let q_high = normal_quantile(1.0 - p);
        assert!(
            (q_low + q_high).abs() < 1e-6,
            "q({p}) + q({}) = {} (expected ~0)",
            1.0 - p,
            q_low + q_high
        );
    }
}

#[test]
fn test_normal_quantile_known_values() {
    use super::helpers::normal_quantile;
    let q975 = normal_quantile(0.975);
    assert!(
        (q975 - 1.96).abs() < 0.01,
        "q(0.975) = {q975}, expected ~1.96"
    );

    let q50 = normal_quantile(0.5);
    assert!(q50.abs() < 1e-10, "q(0.5) = {q50}, expected 0.0");

    let q_invalid = normal_quantile(0.0);
    assert!(q_invalid.is_nan());
    let q_invalid2 = normal_quantile(1.0);
    assert!(q_invalid2.is_nan());
}

// ── FPCA tolerance band tests ──

#[test]
fn test_fpca_band_valid_output() {
    let data = make_test_data();
    let m = data.ncols();

    let band = fpca_tolerance_band(&data, 3, 100, 0.95, BandType::Pointwise, 42);
    let band = band.expect("FPCA band should succeed");

    assert_eq!(band.lower.len(), m);
    assert_eq!(band.upper.len(), m);
    assert_eq!(band.center.len(), m);
    assert_eq!(band.half_width.len(), m);
}

#[test]
fn test_fpca_band_lower_less_than_upper() {
    let data = make_test_data();
    let band = fpca_tolerance_band(&data, 3, 100, 0.95, BandType::Pointwise, 42).unwrap();

    for j in 0..band.lower.len() {
        assert!(
            band.lower[j] < band.upper[j],
            "lower[{j}] = {} >= upper[{j}] = {}",
            band.lower[j],
            band.upper[j]
        );
    }
}

#[test]
fn test_fpca_band_deterministic() {
    let data = make_test_data();
    let b1 = fpca_tolerance_band(&data, 3, 50, 0.95, BandType::Pointwise, 123).unwrap();
    let b2 = fpca_tolerance_band(&data, 3, 50, 0.95, BandType::Pointwise, 123).unwrap();

    for j in 0..b1.lower.len() {
        assert_eq!(b1.lower[j], b2.lower[j]);
        assert_eq!(b1.upper[j], b2.upper[j]);
    }
}

#[test]
fn test_fpca_simultaneous_wider_than_pointwise() {
    let data = make_test_data();
    let pw = fpca_tolerance_band(&data, 3, 200, 0.95, BandType::Pointwise, 42).unwrap();
    let sim = fpca_tolerance_band(&data, 3, 200, 0.95, BandType::Simultaneous, 42).unwrap();

    let pw_mean_hw: f64 = pw.half_width.iter().sum::<f64>() / pw.half_width.len() as f64;
    let sim_mean_hw: f64 = sim.half_width.iter().sum::<f64>() / sim.half_width.len() as f64;

    assert!(
        sim_mean_hw > pw_mean_hw,
        "Simultaneous mean half-width ({sim_mean_hw}) should exceed pointwise ({pw_mean_hw})"
    );
}

#[test]
fn test_fpca_higher_coverage_wider() {
    let data = make_test_data();
    let b90 = fpca_tolerance_band(&data, 3, 200, 0.90, BandType::Pointwise, 42).unwrap();
    let b99 = fpca_tolerance_band(&data, 3, 200, 0.99, BandType::Pointwise, 42).unwrap();

    let hw90: f64 = b90.half_width.iter().sum::<f64>();
    let hw99: f64 = b99.half_width.iter().sum::<f64>();

    assert!(
        hw99 > hw90,
        "99% band total half-width ({hw99}) should exceed 90% ({hw90})"
    );
}

#[test]
fn test_fpca_band_invalid_input() {
    let data = make_test_data();
    assert!(fpca_tolerance_band(&data, 0, 100, 0.95, BandType::Pointwise, 42).is_err());
    assert!(fpca_tolerance_band(&data, 3, 0, 0.95, BandType::Pointwise, 42).is_err());
    assert!(fpca_tolerance_band(&data, 3, 100, 0.0, BandType::Pointwise, 42).is_err());
    assert!(fpca_tolerance_band(&data, 3, 100, 1.0, BandType::Pointwise, 42).is_err());

    let tiny = FdMatrix::zeros(2, 5);
    assert!(fpca_tolerance_band(&tiny, 1, 10, 0.95, BandType::Pointwise, 42).is_err());
}

// ── Conformal prediction band tests ──

#[test]
fn test_conformal_band_valid_output() {
    let data = make_test_data();
    let m = data.ncols();

    let band = conformal_prediction_band(&data, 0.2, 0.95, NonConformityScore::SupNorm, 42);
    let band = band.expect("Conformal band should succeed");

    assert_eq!(band.lower.len(), m);
    assert_eq!(band.upper.len(), m);
}

#[test]
fn test_conformal_supnorm_constant_width() {
    let data = make_test_data();
    let band =
        conformal_prediction_band(&data, 0.3, 0.95, NonConformityScore::SupNorm, 42).unwrap();

    let first = band.half_width[0];
    for &hw in &band.half_width {
        assert!(
            (hw - first).abs() < 1e-12,
            "SupNorm band should have constant width"
        );
    }
}

#[test]
fn test_conformal_l2_constant_width() {
    let data = make_test_data();
    let band = conformal_prediction_band(&data, 0.3, 0.95, NonConformityScore::L2, 42).unwrap();

    let first = band.half_width[0];
    for &hw in &band.half_width {
        assert!(
            (hw - first).abs() < 1e-12,
            "L2 band should have constant width"
        );
    }
}

#[test]
fn test_conformal_coverage_monotonicity() {
    let data = make_test_data();
    let b80 = conformal_prediction_band(&data, 0.3, 0.80, NonConformityScore::SupNorm, 42).unwrap();
    let b95 = conformal_prediction_band(&data, 0.3, 0.95, NonConformityScore::SupNorm, 42).unwrap();

    assert!(
        b95.half_width[0] >= b80.half_width[0],
        "Higher coverage should give wider band"
    );
}

#[test]
fn test_conformal_invalid_input() {
    let data = make_test_data();
    assert!(conformal_prediction_band(&data, 0.0, 0.95, NonConformityScore::SupNorm, 42).is_none());
    assert!(conformal_prediction_band(&data, 1.0, 0.95, NonConformityScore::SupNorm, 42).is_none());
    assert!(conformal_prediction_band(&data, 0.2, 0.0, NonConformityScore::SupNorm, 42).is_none());

    let tiny = FdMatrix::zeros(3, 5);
    assert!(conformal_prediction_band(&tiny, 0.2, 0.95, NonConformityScore::SupNorm, 42).is_none());
}

// ── SCB mean Degras tests ──

#[test]
fn test_scb_mean_valid_output() {
    let data = make_test_data();
    let m = data.ncols();
    let t = uniform_grid(m);

    let band = scb_mean_degras(&data, &t, 0.2, 100, 0.95, MultiplierDistribution::Gaussian);
    let band = band.expect("SCB mean should succeed");

    assert_eq!(band.lower.len(), m);
    assert_eq!(band.upper.len(), m);
    for j in 0..m {
        assert!(band.lower[j] < band.upper[j]);
    }
}

#[test]
fn test_scb_gaussian_vs_rademacher() {
    let data = make_test_data();
    let m = data.ncols();
    let t = uniform_grid(m);

    let gauss =
        scb_mean_degras(&data, &t, 0.2, 200, 0.95, MultiplierDistribution::Gaussian).unwrap();
    let radem = scb_mean_degras(
        &data,
        &t,
        0.2,
        200,
        0.95,
        MultiplierDistribution::Rademacher,
    )
    .unwrap();

    // Both should produce valid bands; widths may differ but both should be positive
    let gauss_mean_hw: f64 = gauss.half_width.iter().sum::<f64>() / m as f64;
    let radem_mean_hw: f64 = radem.half_width.iter().sum::<f64>() / m as f64;
    assert!(gauss_mean_hw > 0.0);
    assert!(radem_mean_hw > 0.0);
}

#[test]
fn test_scb_narrows_with_more_data() {
    let m = 50;
    let t = uniform_grid(m);

    let data_small = sim_fundata(
        20,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    let data_large = sim_fundata(
        200,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );

    let band_small = scb_mean_degras(
        &data_small,
        &t,
        0.2,
        100,
        0.95,
        MultiplierDistribution::Gaussian,
    )
    .unwrap();
    let band_large = scb_mean_degras(
        &data_large,
        &t,
        0.2,
        100,
        0.95,
        MultiplierDistribution::Gaussian,
    )
    .unwrap();

    let hw_small: f64 = band_small.half_width.iter().sum::<f64>() / m as f64;
    let hw_large: f64 = band_large.half_width.iter().sum::<f64>() / m as f64;

    assert!(
        hw_large < hw_small,
        "SCB should narrow with more data: hw_small={hw_small}, hw_large={hw_large}"
    );
}

#[test]
fn test_scb_invalid_input() {
    let data = make_test_data();
    let t = uniform_grid(data.ncols());

    assert!(scb_mean_degras(&data, &t, 0.0, 100, 0.95, MultiplierDistribution::Gaussian).is_err());
    assert!(scb_mean_degras(&data, &t, 0.2, 0, 0.95, MultiplierDistribution::Gaussian).is_err());
    assert!(scb_mean_degras(&data, &t, 0.2, 100, 0.0, MultiplierDistribution::Gaussian).is_err());
    // Wrong argvals length
    let wrong_t = uniform_grid(data.ncols() + 1);
    assert!(scb_mean_degras(
        &data,
        &wrong_t,
        0.2,
        100,
        0.95,
        MultiplierDistribution::Gaussian
    )
    .is_err());
}

// ── Exponential family tests ──

#[test]
fn test_exp_family_gaussian_matches_fpca() {
    let data = make_test_data();

    let exp_band =
        exponential_family_tolerance_band(&data, ExponentialFamily::Gaussian, 3, 100, 0.95, 42)
            .unwrap();

    let fpca_band = fpca_tolerance_band(&data, 3, 100, 0.95, BandType::Simultaneous, 42).unwrap();

    // Gaussian family with identity link should produce the same band
    for j in 0..data.ncols() {
        assert!(
            (exp_band.lower[j] - fpca_band.lower[j]).abs() < 1e-10,
            "Gaussian family should match FPCA at point {j}"
        );
        assert!(
            (exp_band.upper[j] - fpca_band.upper[j]).abs() < 1e-10,
            "Gaussian family should match FPCA at point {j}"
        );
    }
}

#[test]
fn test_exp_family_poisson() {
    // Create data that looks like Poisson counts (positive)
    let m = 30;
    let t = uniform_grid(m);
    let raw = sim_fundata(
        40,
        &t,
        3,
        EFunType::Fourier,
        EValType::Exponential,
        Some(99),
    );

    // Shift to positive range and add offset
    let mut data = FdMatrix::zeros(40, m);
    for j in 0..m {
        for i in 0..40 {
            data[(i, j)] = (raw[(i, j)] + 5.0).max(0.1); // ensure positive
        }
    }

    let band =
        exponential_family_tolerance_band(&data, ExponentialFamily::Poisson, 3, 50, 0.95, 42);
    let band = band.expect("Poisson band should succeed");

    // All bounds should be positive (exp of anything is positive)
    for j in 0..m {
        assert!(
            band.lower[j] > 0.0,
            "Poisson lower bound should be positive"
        );
        assert!(
            band.upper[j] > 0.0,
            "Poisson upper bound should be positive"
        );
    }
}

#[test]
fn test_exp_family_binomial() {
    // Create data in (0, 1) range
    let m = 30;
    let t = uniform_grid(m);
    let raw = sim_fundata(
        40,
        &t,
        3,
        EFunType::Fourier,
        EValType::Exponential,
        Some(77),
    );

    let mut data = FdMatrix::zeros(40, m);
    for j in 0..m {
        for i in 0..40 {
            // Map to (0, 1) via sigmoid
            data[(i, j)] = 1.0 / (1.0 + (-raw[(i, j)]).exp());
        }
    }

    let band =
        exponential_family_tolerance_band(&data, ExponentialFamily::Binomial, 3, 50, 0.95, 42);
    let band = band.expect("Binomial band should succeed");

    // All bounds should be in (0, 1) (inverse logit maps to (0, 1))
    for j in 0..m {
        assert!(
            band.lower[j] > 0.0 && band.lower[j] < 1.0,
            "Binomial lower bound at {j} = {} should be in (0,1)",
            band.lower[j]
        );
        assert!(
            band.upper[j] > 0.0 && band.upper[j] < 1.0,
            "Binomial upper bound at {j} = {} should be in (0,1)",
            band.upper[j]
        );
    }
}

#[test]
fn test_exp_family_invalid_input() {
    let data = make_test_data();
    assert!(exponential_family_tolerance_band(
        &data,
        ExponentialFamily::Gaussian,
        0,
        100,
        0.95,
        42
    )
    .is_err());
    assert!(
        exponential_family_tolerance_band(&data, ExponentialFamily::Gaussian, 3, 0, 0.95, 42)
            .is_err()
    );
}

// ── Elastic tolerance band tests ──

fn make_elastic_test_data() -> (FdMatrix, Vec<f64>) {
    let m = 50;
    let t = uniform_grid(m);
    let data = sim_fundata(
        30,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    (data, t)
}

#[test]
fn test_elastic_band_valid_output() {
    let (data, t) = make_elastic_test_data();
    let m = t.len();

    let band = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 42);
    let band = band.expect("Elastic band should succeed");

    assert_eq!(band.lower.len(), m);
    assert_eq!(band.upper.len(), m);
    assert_eq!(band.center.len(), m);
    assert_eq!(band.half_width.len(), m);
}

#[test]
fn test_elastic_band_lower_less_than_upper() {
    let (data, t) = make_elastic_test_data();
    let m = t.len();

    let band = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 42).unwrap();

    for j in 0..m {
        assert!(
            band.lower[j] < band.upper[j],
            "lower[{j}] = {} >= upper[{j}] = {}",
            band.lower[j],
            band.upper[j]
        );
    }
}

#[test]
fn test_elastic_band_center_within_bounds() {
    let (data, t) = make_elastic_test_data();
    let m = t.len();

    let band = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 42).unwrap();

    for j in 0..m {
        assert!(
            band.center[j] >= band.lower[j] && band.center[j] <= band.upper[j],
            "center[{j}]={} should be in [{}, {}]",
            band.center[j],
            band.lower[j],
            band.upper[j]
        );
    }
}

#[test]
fn test_elastic_band_half_width_positive() {
    let (data, t) = make_elastic_test_data();

    let band = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 42).unwrap();

    for (j, &hw) in band.half_width.iter().enumerate() {
        assert!(hw > 0.0, "half_width[{j}] should be positive, got {hw}");
    }
}

#[test]
fn test_elastic_band_simultaneous() {
    let (data, t) = make_elastic_test_data();
    let m = t.len();

    let band = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Simultaneous, 5, 42);
    let band = band.expect("Elastic simultaneous band should succeed");

    assert_eq!(band.lower.len(), m);
    for j in 0..m {
        assert!(band.lower[j] < band.upper[j]);
    }
}

#[test]
fn test_elastic_band_deterministic() {
    let (data, t) = make_elastic_test_data();

    let b1 = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 123).unwrap();
    let b2 = elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 5, 123).unwrap();

    for j in 0..t.len() {
        assert_eq!(
            b1.lower[j], b2.lower[j],
            "lower[{j}] should be deterministic"
        );
        assert_eq!(
            b1.upper[j], b2.upper[j],
            "upper[{j}] should be deterministic"
        );
    }
}

#[test]
fn test_elastic_band_higher_coverage_wider() {
    let (data, t) = make_elastic_test_data();

    let b90 = elastic_tolerance_band(&data, &t, 3, 100, 0.90, BandType::Pointwise, 5, 42).unwrap();
    let b99 = elastic_tolerance_band(&data, &t, 3, 100, 0.99, BandType::Pointwise, 5, 42).unwrap();

    let hw90: f64 = b90.half_width.iter().sum();
    let hw99: f64 = b99.half_width.iter().sum();

    assert!(
        hw99 > hw90,
        "99% coverage band should be wider than 90%: hw99={hw99:.4}, hw90={hw90:.4}"
    );
}

#[test]
fn test_elastic_band_invalid_input() {
    let (data, t) = make_elastic_test_data();

    // Wrong argvals length
    let wrong_t = uniform_grid(t.len() + 1);
    assert!(
        elastic_tolerance_band(&data, &wrong_t, 3, 50, 0.95, BandType::Pointwise, 5, 42).is_err()
    );

    // max_iter = 0
    assert!(elastic_tolerance_band(&data, &t, 3, 50, 0.95, BandType::Pointwise, 0, 42).is_err());

    // ncomp = 0
    assert!(elastic_tolerance_band(&data, &t, 0, 50, 0.95, BandType::Pointwise, 5, 42).is_err());

    // nb = 0
    assert!(elastic_tolerance_band(&data, &t, 3, 0, 0.95, BandType::Pointwise, 5, 42).is_err());

    // coverage out of range
    assert!(elastic_tolerance_band(&data, &t, 3, 50, 0.0, BandType::Pointwise, 5, 42).is_err());
    assert!(elastic_tolerance_band(&data, &t, 3, 50, 1.0, BandType::Pointwise, 5, 42).is_err());

    // Too few observations
    let tiny = FdMatrix::zeros(2, t.len());
    assert!(elastic_tolerance_band(&tiny, &t, 1, 10, 0.95, BandType::Pointwise, 5, 42).is_err());
}

// ── Equivalence test (TOST) tests ──

fn make_equivalent_groups() -> (FdMatrix, FdMatrix) {
    let m = 50;
    let t = uniform_grid(m);
    let d1 = sim_fundata(
        30,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    let d2 = sim_fundata(
        30,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(142),
    );
    (d1, d2)
}

fn make_shifted_groups() -> (FdMatrix, FdMatrix) {
    let m = 50;
    let t = uniform_grid(m);
    let d1 = sim_fundata(
        30,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    let mut d2 = sim_fundata(
        30,
        &t,
        5,
        EFunType::Fourier,
        EValType::Exponential,
        Some(142),
    );
    let (n2, m2) = d2.shape();
    for i in 0..n2 {
        for j in 0..m2 {
            d2[(i, j)] += 10.0;
        }
    }
    (d1, d2)
}

#[test]
fn test_equivalence_invalid_inputs() {
    let (data1, data2) = make_equivalent_groups();
    let bs = EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian);

    let tiny = FdMatrix::zeros(2, 50);
    assert!(equivalence_test(&tiny, &data2, 1.0, 0.05, 100, bs, 42).is_none());
    assert!(equivalence_test(&data1, &tiny, 1.0, 0.05, 100, bs, 42).is_none());

    let wrong_m = FdMatrix::zeros(30, 40);
    assert!(equivalence_test(&data1, &wrong_m, 1.0, 0.05, 100, bs, 42).is_none());

    assert!(equivalence_test(&data1, &data2, 0.0, 0.05, 100, bs, 42).is_none());
    assert!(equivalence_test(&data1, &data2, -1.0, 0.05, 100, bs, 42).is_none());
    assert!(equivalence_test(&data1, &data2, 1.0, 0.0, 100, bs, 42).is_none());
    assert!(equivalence_test(&data1, &data2, 1.0, 0.5, 100, bs, 42).is_none());
    assert!(equivalence_test(&data1, &data2, 1.0, 0.05, 0, bs, 42).is_none());
}

#[test]
fn test_equivalence_deterministic() {
    let (data1, data2) = make_equivalent_groups();
    let bs = EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian);

    let r1 = equivalence_test(&data1, &data2, 5.0, 0.05, 100, bs, 42).unwrap();
    let r2 = equivalence_test(&data1, &data2, 5.0, 0.05, 100, bs, 42).unwrap();

    assert_eq!(r1.test_statistic, r2.test_statistic);
    assert_eq!(r1.p_value, r2.p_value);
    assert_eq!(r1.critical_value, r2.critical_value);
    assert_eq!(r1.equivalent, r2.equivalent);
}

#[test]
fn test_equivalence_identical_groups() {
    let data = make_test_data();
    let r = equivalence_test(
        &data,
        &data,
        10.0,
        0.05,
        200,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .unwrap();
    assert!(
        r.equivalent,
        "Identical groups with large delta should be equivalent"
    );
}

#[test]
fn test_equivalence_different_groups() {
    let (data1, data2) = make_shifted_groups();
    let r = equivalence_test(
        &data1,
        &data2,
        0.5,
        0.05,
        200,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .unwrap();
    assert!(
        !r.equivalent,
        "Shifted groups with small delta should not be equivalent"
    );
}

#[test]
fn test_equivalence_scb_properties() {
    let (data1, data2) = make_equivalent_groups();
    let r = equivalence_test(
        &data1,
        &data2,
        5.0,
        0.05,
        200,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .unwrap();

    for j in 0..r.scb.lower.len() {
        assert!(
            r.scb.lower[j] < r.scb.center[j],
            "lower[{j}] should be < center[{j}]"
        );
        assert!(
            r.scb.center[j] < r.scb.upper[j],
            "center[{j}] should be < upper[{j}]"
        );
    }
}

#[test]
fn test_equivalence_larger_delta_easier() {
    let (data1, data2) = make_equivalent_groups();
    let bs = EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian);

    let r_small = equivalence_test(&data1, &data2, 1.0, 0.05, 200, bs, 42).unwrap();
    let r_large = equivalence_test(&data1, &data2, 100.0, 0.05, 200, bs, 42).unwrap();

    assert!(
        r_large.equivalent || !r_small.equivalent,
        "Larger delta should be at least as likely equivalent"
    );
    assert!(
        r_large.p_value <= r_small.p_value + 1e-10,
        "Larger delta p-value ({}) should be <= smaller delta p-value ({})",
        r_large.p_value,
        r_small.p_value
    );
}

#[test]
fn test_equivalence_pvalue_range() {
    let (data1, data2) = make_equivalent_groups();
    let r = equivalence_test(
        &data1,
        &data2,
        5.0,
        0.05,
        200,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .unwrap();
    assert!(r.p_value >= 0.0, "p_value should be >= 0");
    assert!(r.p_value <= 1.0, "p_value should be <= 1");
}

#[test]
fn test_equivalence_pvalue_consistent() {
    let (data1, data2) = make_equivalent_groups();
    let bs = EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian);

    let r = equivalence_test(&data1, &data2, 100.0, 0.05, 500, bs, 42).unwrap();
    if r.equivalent {
        assert!(
            r.p_value < r.alpha,
            "equivalent=true should imply p_value ({}) < alpha ({})",
            r.p_value,
            r.alpha
        );
    }

    let r2 = equivalence_test(&data1, &data2, 0.001, 0.05, 500, bs, 42).unwrap();
    if !r2.equivalent {
        assert!(
            r2.p_value >= r2.alpha,
            "equivalent=false should imply p_value ({}) >= alpha ({})",
            r2.p_value,
            r2.alpha
        );
    }
}

#[test]
fn test_equivalence_percentile() {
    let (data1, data2) = make_equivalent_groups();
    let r = equivalence_test(
        &data1,
        &data2,
        5.0,
        0.05,
        200,
        EquivalenceBootstrap::Percentile,
        42,
    )
    .unwrap();

    assert!(r.test_statistic >= 0.0);
    assert!(r.p_value >= 0.0 && r.p_value <= 1.0);
    assert!(r.critical_value >= 0.0);
}

#[test]
fn test_equivalence_one_sample_equivalent() {
    let data = make_test_data();
    let mu0 = mean_1d(&data);

    let r = equivalence_test_one_sample(
        &data,
        &mu0,
        10.0,
        0.05,
        200,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .unwrap();
    assert!(
        r.equivalent,
        "Data vs its own mean with large delta should be equivalent"
    );
}

#[test]
fn test_equivalence_one_sample_shifted() {
    let data = make_test_data();
    let mu0 = vec![100.0; data.ncols()];

    let r = equivalence_test_one_sample(
        &data,
        &mu0,
        0.5,
        0.05,
        200,
        EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian),
        42,
    )
    .unwrap();
    assert!(
        !r.equivalent,
        "Data vs far-away mu0 should not be equivalent"
    );
}

#[test]
fn test_equivalence_one_sample_invalid() {
    let data = make_test_data();
    let mu0 = vec![0.0; data.ncols()];
    let bs = EquivalenceBootstrap::Multiplier(MultiplierDistribution::Gaussian);

    let tiny = FdMatrix::zeros(2, data.ncols());
    assert!(equivalence_test_one_sample(&tiny, &mu0, 1.0, 0.05, 100, bs, 42).is_none());
    assert!(equivalence_test_one_sample(&data, &[0.0; 10], 1.0, 0.05, 100, bs, 42).is_none());
    assert!(equivalence_test_one_sample(&data, &mu0, 0.0, 0.05, 100, bs, 42).is_none());
    assert!(equivalence_test_one_sample(&data, &mu0, 1.0, 0.5, 100, bs, 42).is_none());
}

#[test]
fn test_constant_data_fpca_tolerance() {
    let n = 10;
    let m = 20;
    let data = FdMatrix::from_column_major(vec![5.0; n * m], n, m).unwrap();
    // Constant data: FPCA tolerance band should be tight around 5.0
    let band = fpca_tolerance_band(&data, 2, 200, 0.95, BandType::Pointwise, 42);
    // Constant data may cause FPCA to fail (zero variance), so handle both cases
    if let Ok(band) = band {
        assert_eq!(band.lower.len(), m);
        assert_eq!(band.upper.len(), m);
        for j in 0..m {
            assert!(band.lower[j].is_finite());
            assert!(band.upper[j].is_finite());
        }
    }
}

#[test]
fn test_n3_fpca_tolerance() {
    // Minimum viable: 3 curves
    let n = 3;
    let m = 20;
    let data_vec: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.1).sin()).collect();
    let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();
    let band = fpca_tolerance_band(&data, 2, 100, 0.90, BandType::Pointwise, 42);
    if let Ok(band) = band {
        assert_eq!(band.lower.len(), m);
        assert_eq!(band.upper.len(), m);
    }
}

#[test]
fn test_delta_zero_equivalence() {
    // delta=0 means testing exact equality (should always reject / return None due to invalid params)
    let n = 10;
    let m = 20;
    let data1_vec: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.1).sin()).collect();
    let data2_vec: Vec<f64> = (0..n * m).map(|i| (i as f64 * 0.1).sin() + 0.5).collect();
    let data1 = FdMatrix::from_column_major(data1_vec, n, m).unwrap();
    let data2 = FdMatrix::from_column_major(data2_vec, n, m).unwrap();
    let result = equivalence_test(
        &data1,
        &data2,
        0.0,
        0.05,
        199,
        EquivalenceBootstrap::Percentile,
        42,
    );
    // With delta=0, valid_equivalence_params returns false, so result should be None
    assert!(result.is_none());
}
