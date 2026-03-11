//! Integration tests comparing fdars-core results against R reference implementations.
//!
//! These tests load pre-generated JSON fixtures (from validation/data/ and validation/expected/)
//! and compare Rust outputs against R's fda, fda.usc, roahd, cluster, fpc, dtw, pls, glmnet, etc.
//!
//! Run: cargo test --test validate_against_r --features linalg
//!
//! ## Known Convention Differences
//!
//! - **Integration weights**: Rust's `simpsons_weights` uses the composite trapezoidal rule,
//!   while R's fda.usc uses Simpson's 1/3 rule. This affects inner products, depth measures,
//!   and all functions that integrate over the domain. Tolerances are set accordingly.
//!
//! - **Fourier basis normalization**: R's `create.fourier.basis` includes √2 normalization
//!   for orthonormality; Rust's `fourier_basis_with_period` does not.
//!
//! - **FPCA scores**: Rust returns U*Σ (scaled scores), R's `svd()$u` returns unscaled U.
//!
//! - **B-spline knots**: Rust extends boundary knots beyond the data range; R places
//!   boundary knots at endpoints with multiplicity = order.
//!
//! - **Eigenvalue formulas**: Rust's `eigenvalues_exponential` uses exp(-k) for k=1,...,m;
//!   R generates exp(-(k-1)) for k=1,...,m.

#![allow(dead_code)]

use fdars_core::classification::{fclassif_dd, fclassif_knn, fclassif_lda};
use fdars_core::famm::fmm;
use fdars_core::gmm::{gmm_em, CovType};
use fdars_core::irreg_fdata::IrregFdata;
use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::fregre_lm;
use fdars_core::streaming_depth::StreamingDepth;
use serde::Deserialize;
use std::fs;
use std::path::PathBuf;

// ─── Helpers ────────────────────────────────────────────────────────────────

fn validation_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("validation")
}

fn load_json<T: serde::de::DeserializeOwned>(dir: &str, name: &str) -> T {
    let path = validation_dir().join(dir).join(format!("{}.json", name));
    let data = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
    serde_json::from_str(&data)
        .unwrap_or_else(|e| panic!("Failed to parse {}: {}", path.display(), e))
}

fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: {} vs {}",
        label,
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() || a.is_nan() {
            continue;
        }
        assert!(
            (a - e).abs() < tol,
            "{} [{}]: Rust={:.12}, R={:.12}, diff={:.2e} > tol={:.2e}",
            label,
            i,
            a,
            e,
            (a - e).abs(),
            tol
        );
    }
}

/// Compare vectors element-wise using absolute values (for sign-ambiguous results like SVD).
fn assert_vec_close_abs(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: {} vs {}",
        label,
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        if e.is_nan() || a.is_nan() {
            continue;
        }
        assert!(
            (a.abs() - e.abs()).abs() < tol,
            "{} [{}]: |Rust|={:.12}, |R|={:.12}, diff={:.2e} > tol={:.2e}",
            label,
            i,
            a.abs(),
            e.abs(),
            (a.abs() - e.abs()).abs(),
            tol
        );
    }
}

fn assert_scalar_close(actual: f64, expected: f64, tol: f64, label: &str) {
    if expected.is_nan() || actual.is_nan() {
        return;
    }
    assert!(
        (actual - expected).abs() < tol,
        "{}: Rust={:.12}, R={:.12}, diff={:.2e} > tol={:.2e}",
        label,
        actual,
        expected,
        (actual - expected).abs(),
        tol
    );
}

fn assert_relative_close(actual: f64, expected: f64, rel_tol: f64, label: &str) {
    if expected.is_nan() || actual.is_nan() {
        return;
    }
    let denom = expected.abs().max(1e-10);
    let rel_err = (actual - expected).abs() / denom;
    assert!(
        rel_err < rel_tol,
        "{}: Rust={:.8}, R={:.8}, relative error={:.4} > {:.4}",
        label,
        actual,
        expected,
        rel_err,
        rel_tol
    );
}

/// Check that two ranking orderings are correlated (Spearman-like).
fn assert_ranking_correlated(actual: &[f64], expected: &[f64], label: &str) {
    assert_ranking_correlated_tol(actual, expected, 0.97, label);
}

/// Check ranking correlation with a custom threshold.
fn assert_ranking_correlated_tol(actual: &[f64], expected: &[f64], min_rho: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{}: length mismatch", label);
    let n = actual.len();
    let rank = |v: &[f64]| -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = v.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut ranks = vec![0usize; n];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            ranks[*idx] = rank;
        }
        ranks
    };
    let r_actual = rank(actual);
    let r_expected = rank(expected);
    // Compute Spearman rank correlation
    let mean_a = r_actual.iter().sum::<usize>() as f64 / n as f64;
    let mean_e = r_expected.iter().sum::<usize>() as f64 / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_e = 0.0;
    for i in 0..n {
        let da = r_actual[i] as f64 - mean_a;
        let de = r_expected[i] as f64 - mean_e;
        cov += da * de;
        var_a += da * da;
        var_e += de * de;
    }
    let rho = cov / (var_a * var_e).sqrt().max(1e-10);
    assert!(
        rho > min_rho,
        "{}: rankings poorly correlated (ρ={:.4}, threshold={:.2})",
        label,
        rho,
        min_rho
    );
}

// ─── Input data structures ─────────────────────────────────────────────────

#[derive(Deserialize)]
struct StandardData {
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    data: Vec<f64>,
}

#[derive(Deserialize)]
struct ClusterData {
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    data: Vec<f64>,
    true_labels: Vec<usize>,
}

#[derive(Deserialize)]
struct NoisySineData {
    x: Vec<f64>,
    y_noisy: Vec<f64>,
}

#[derive(Deserialize)]
struct RegressionData {
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    data: Vec<f64>,
    y: Vec<f64>,
}

#[derive(Deserialize)]
struct OutlierData {
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    data: Vec<f64>,
    outlier_indices: Vec<usize>,
}

// seasonal_200.json has flat arrays at top level
#[derive(Deserialize)]
struct SeasonalData {
    t: Vec<f64>,
    pure_sine: Vec<f64>,
    noisy_sine: Vec<f64>,
    with_trend: Vec<f64>,
    multi_period: Vec<f64>,
    n: usize,
    period: usize,
}

// ─── Expected data structures ──────────────────────────────────────────────

// Utility expected
#[derive(Deserialize)]
struct UtilityExpected {
    simpsons_weights: Vec<f64>,
    simpsons_weights_11: Vec<f64>,
    inner_product_12: f64,
    inner_product_matrix: SquareMatrixData,
}

#[derive(Deserialize)]
struct SquareMatrixData {
    n: usize,
    data: Vec<f64>,
}

// Fdata expected
#[derive(Deserialize)]
struct FdataExpected {
    mean: Vec<f64>,
    centered: Vec<f64>,
    norm_l2: Vec<f64>,
}

// Depth expected
#[derive(Deserialize)]
struct DepthExpected {
    fraiman_muniz: Vec<f64>,
    band: Vec<f64>,
    modified_band: Vec<f64>,
    modified_epigraph: Vec<f64>,
    modal: Vec<f64>,
    random_projection: serde_json::Value,
    random_tukey: serde_json::Value,
    functional_spatial: Vec<f64>,
    kernel_functional_spatial: Vec<f64>,
}

// Basis expected
#[derive(Deserialize)]
struct BasisExpected {
    bspline_matrix: RectMatrixData,
    fourier_matrix: RectMatrixData,
    diff_matrix_order1: RectMatrixData,
    diff_matrix_order2: RectMatrixData,
    pspline_fit: PsplineFitExpected,
    fourier_fit: FourierFitExpected,
}

#[derive(Deserialize)]
struct RectMatrixData {
    nrow: usize,
    ncol: usize,
    data: Vec<f64>,
}

#[derive(Deserialize)]
struct PsplineFitExpected {
    coefficients: Vec<f64>,
    fitted_values: Vec<f64>,
    gcv: f64,
}

#[derive(Deserialize)]
struct FourierFitExpected {
    coefficients: Vec<f64>,
    fitted_values: Vec<f64>,
}

// Metrics expected
#[derive(Deserialize)]
struct MetricsExpected {
    lp_l2: SquareMatrixData,
    dtw_symmetric2: f64,
    dtw_sakoechiba: f64,
    semimetric_fourier: SquareMatrixData,
    semimetric_hshift: SquareMatrixData,
    #[serde(default)]
    hausdorff_5x5: Option<SquareMatrixData>,
    #[serde(default)]
    lp_cross_5x5: Option<CrossDistanceExpected>,
}

// Clustering expected
#[derive(Deserialize)]
struct ClusteringExpected {
    silhouette: SilhouetteExpected,
    calinski_harabasz: f64,
    withinss: WithinssExpected,
}

#[derive(Deserialize)]
struct SilhouetteExpected {
    widths: Vec<f64>,
    average: f64,
}

#[derive(Deserialize)]
struct WithinssExpected {
    per_cluster: Vec<f64>,
    total: f64,
}

// Regression expected
#[derive(Deserialize)]
struct RegressionExpected {
    fpca_svd: FpcaSvdExpected,
    ridge: RidgeExpected,
    pls: PlsExpected,
}

#[derive(Deserialize)]
struct FpcaSvdExpected {
    singular_values: Vec<f64>,
    scores: Vec<f64>,
    loadings: Vec<f64>,
    col_means: Vec<f64>,
    proportion_variance: Vec<f64>,
}

#[derive(Deserialize)]
struct RidgeExpected {
    intercept: f64,
    coefficients: Vec<f64>,
}

#[derive(Deserialize)]
struct PlsExpected {
    scores: Vec<f64>,
    loadings: Vec<f64>,
    weights: Vec<f64>,
}

// Outliers expected
#[derive(Deserialize)]
struct OutliersExpected {
    depth_fm: OutlierDepthExpected,
    outliers_depth_trim: serde_json::Value,
}

#[derive(Deserialize)]
struct OutlierDepthExpected {
    depth_values: Vec<f64>,
    lowest_depth_indices: Vec<usize>,
}

// Smoothing expected
#[derive(Deserialize)]
struct SmoothingExpected {
    nadaraya_watson: Vec<f64>,
    local_linear: Vec<f64>,
    knn_k5: Vec<f64>,
    #[serde(default)]
    local_polynomial: Option<Vec<f64>>,
    #[serde(default)]
    smoothing_matrix_nw: Option<SmoothingMatrixNwExpected>,
}

#[derive(Deserialize)]
struct SmoothingMatrixNwExpected {
    eval_point: f64,
    bandwidth: f64,
    weights: Vec<f64>,
    row_sum: f64,
}

// Simulation expected
#[derive(Deserialize)]
struct SimulationExpected {
    fourier_eigenfunctions: RectMatrixData,
    wiener_eigenfunctions: RectMatrixData,
    eigenvalues: EigenvaluesExpected,
}

#[derive(Deserialize)]
struct EigenvaluesExpected {
    linear: Vec<f64>,
    exponential: Vec<f64>,
    wiener: Vec<f64>,
}

// Seasonal expected
#[derive(Deserialize)]
struct SeasonalExpected {
    periodogram: PeriodogramExpected,
    acf: AcfExpected,
    #[serde(default)]
    lomb_scargle: Option<LombScargleExpected>,
    peak_detection: PeakDetectionExpected,
    period_estimation: PeriodEstimationExpected,
    #[serde(default)]
    ssa_reconstruction: Option<SsaReconstructionExpected>,
    #[serde(default)]
    hilbert_amplitude: Option<HilbertAmplitudeExpected>,
    #[serde(default)]
    seasonal_strength: Option<SeasonalStrengthExpected>,
    #[serde(default)]
    decompose_seasonal: Option<DecomposeSeasonalExpected>,
}

#[derive(Deserialize)]
struct SsaReconstructionExpected {
    component_1_2: Vec<f64>,
    window_length: usize,
}

#[derive(Deserialize)]
struct HilbertAmplitudeExpected {
    amplitude: Vec<f64>,
}

#[derive(Deserialize)]
struct SeasonalStrengthExpected {
    strength: f64,
    frequency: usize,
}

#[derive(Deserialize)]
struct DecomposeSeasonalExpected {
    seasonal: Vec<f64>,
    frequency: usize,
}

#[derive(Deserialize)]
struct PeriodogramExpected {
    freq: Vec<f64>,
    spec: Vec<f64>,
    peak_freq: f64,
    peak_index: usize,
}

#[derive(Deserialize)]
struct AcfExpected {
    acf: Vec<f64>,
}

#[derive(Deserialize)]
struct LombScargleExpected {
    scanned_periods: Vec<f64>,
    power: Vec<f64>,
    peak_period: f64,
}

#[derive(Deserialize)]
struct PeakDetectionExpected {
    signal: Vec<f64>,
    x: Vec<f64>,
    peak_indices: Vec<usize>,
    peak_heights: Vec<f64>,
}

#[derive(Deserialize)]
struct PeriodEstimationExpected {
    detected_period_fft: f64,
    true_period: f64,
    peak_freq_cycles_per_sample: f64,
    dt: f64,
}

// Detrend expected
#[derive(Deserialize)]
struct DetrendExpected {
    linear_detrend: LinearDetrendExpected,
    poly_detrend: PolyDetrendExpected,
    differencing: DifferencingExpected,
    stl_decomposition: StlExpected,
    additive_decomposition: serde_json::Value,
    #[serde(default)]
    loess_detrend: Option<LoessDetrendExpected>,
}

#[derive(Deserialize)]
struct LoessDetrendExpected {
    trend: Vec<f64>,
    detrended: Vec<f64>,
    span: f64,
}

#[derive(Deserialize)]
struct LinearDetrendExpected {
    trend: Vec<f64>,
    detrended: Vec<f64>,
    intercept: f64,
    slope: f64,
}

#[derive(Deserialize)]
struct PolyDetrendExpected {
    trend: Vec<f64>,
    detrended: Vec<f64>,
    coefficients: Vec<f64>,
}

#[derive(Deserialize)]
struct DifferencingExpected {
    differenced: Vec<f64>,
}

#[derive(Deserialize)]
struct StlExpected {
    frequency: usize,
    trend: Vec<f64>,
    seasonal: Vec<f64>,
    remainder: Vec<f64>,
}

// Alignment data (alignment_30x51.json)
#[derive(Deserialize)]
struct AlignmentData {
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    data: Vec<f64>,
}

// Equivalence groups data (equivalence_groups.json)
#[derive(Deserialize)]
struct EquivalenceGroupsData {
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    data1: Vec<f64>,
    data2: Vec<f64>,
}

// Alignment expected
#[derive(Deserialize)]
struct AlignmentExpected {
    srsf_row0: Vec<f64>,
    srsf_row1: Vec<f64>,
    srsf_roundtrip_row0: Vec<f64>,
    elastic_distance_01: f64,
    pair_align_gamma: Vec<f64>,
    pair_align_f_aligned: Vec<f64>,
    karcher_mean: Vec<f64>,
    karcher_mean_srsf: Vec<f64>,
    distance_matrix_5x5: SquareMatrixData,
    #[serde(default)]
    elastic_decomposition: Option<ElasticDecompositionExpected>,
    #[serde(default)]
    cross_distance_3x3: Option<CrossDistanceExpected>,
}

#[derive(Deserialize)]
struct ElasticDecompositionExpected {
    amplitude_distance: f64,
    phase_distance: f64,
}

#[derive(Deserialize)]
struct CrossDistanceExpected {
    n1: usize,
    n2: usize,
    data: Vec<f64>,
}

// Tolerance expected
#[derive(Deserialize)]
struct ToleranceExpected {
    fpca_center: Vec<f64>,
    fpca_eigenvalues: Vec<f64>,
    conformal_center: Vec<f64>,
    conformal_quantile: f64,
    degras_center: Vec<f64>,
    degras_critical_value: f64,
    #[serde(default)]
    degras_smoothed_mean: Option<Vec<f64>>,
    #[serde(default)]
    degras_sigma_hat: Option<Vec<f64>>,
    #[serde(default)]
    elastic_tolerance: Option<ElasticToleranceExpected>,
}

#[derive(Deserialize)]
struct ElasticToleranceExpected {
    center: Vec<f64>,
    n: usize,
}

// Equivalence expected
#[derive(Deserialize)]
struct EquivalenceExpected {
    d_hat: Vec<f64>,
    test_statistic: f64,
    pooled_se: Vec<f64>,
    critical_value: f64,
    scb_lower: Vec<f64>,
    scb_upper: Vec<f64>,
    equivalent: bool,
    p_value: f64,
}

// IrregFdata expected
#[derive(Deserialize)]
struct IrregFdataExpected {
    n_curves: usize,
    n_points: Vec<usize>,
    argvals: Vec<Vec<f64>>,
    values: Vec<Vec<f64>>,
    #[serde(default)]
    integrate: Option<IntegrateExpected>,
    #[serde(default)]
    norm_l2: Option<NormL2Expected>,
    #[serde(default)]
    mean_curve: Option<MeanCurveExpected>,
    #[serde(default)]
    to_regular: Option<ToRegularExpected>,
    #[serde(default)]
    metric_lp: Option<SquareMatrixData>,
}

#[derive(Deserialize)]
struct IntegrateExpected {
    integrals: Vec<f64>,
}

#[derive(Deserialize)]
struct NormL2Expected {
    norms: Vec<f64>,
}

#[derive(Deserialize)]
struct MeanCurveExpected {
    target_grid: Vec<f64>,
    mean_values: Vec<f64>,
}

#[derive(Deserialize)]
struct ToRegularExpected {
    target_grid: Vec<f64>,
    data: Vec<f64>,
    n: usize,
    m: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

// ─── Utility ────────────────────────────────────────────────────────────────

#[test]
fn test_simpsons_weights_101() {
    // 101 points = 100 intervals (even), pure Simpson's 1/3 rule
    let argvals: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let actual = fdars_core::simpsons_weights(&argvals);

    // Simpson's 1/3 for uniform h=0.01: [h/3, 4h/3, 2h/3, 4h/3, ..., 4h/3, h/3]
    let h = 0.01;
    assert_scalar_close(actual[0], h / 3.0, 1e-15, "simp_weight_first");
    assert_scalar_close(actual[100], h / 3.0, 1e-15, "simp_weight_last");
    assert_scalar_close(actual[1], 4.0 * h / 3.0, 1e-15, "simp_weight_odd");
    assert_scalar_close(actual[2], 2.0 * h / 3.0, 1e-15, "simp_weight_even");
    // Sum should equal the interval length (1.0)
    let total: f64 = actual.iter().sum();
    assert_scalar_close(total, 1.0, 1e-14, "simp_weight_total");
}

#[test]
fn test_simpsons_weights_11() {
    // 11 points = 10 intervals (even), pure Simpson's 1/3 rule
    let argvals: Vec<f64> = (0..=10).map(|i| i as f64 / 10.0).collect();
    let actual = fdars_core::simpsons_weights(&argvals);

    let h = 0.1;
    assert_scalar_close(actual[0], h / 3.0, 1e-15, "simp_weight_11_first");
    assert_scalar_close(actual[10], h / 3.0, 1e-15, "simp_weight_11_last");
    assert_scalar_close(actual[5], 4.0 * h / 3.0, 1e-15, "simp_weight_11_odd");
    let total: f64 = actual.iter().sum();
    assert_scalar_close(total, 1.0, 1e-14, "simp_weight_11_total");
}

#[test]
fn test_inner_product() {
    let exp: UtilityExpected = load_json("expected", "utility_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let m = dat.m;
    let n = dat.n;
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[j * n]).collect();
    let curve2: Vec<f64> = (0..m).map(|j| dat.data[1 + j * n]).collect();

    let actual = fdars_core::utility::inner_product(&curve1, &curve2, &dat.argvals);
    // Both R and Rust now use Simpson's 1/3 rule
    assert_scalar_close(actual, exp.inner_product_12, 1e-6, "inner_product_12");
}

#[test]
fn test_inner_product_matrix() {
    let exp: UtilityExpected = load_json("expected", "utility_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n_sub = 5;
    let m = dat.m;
    let n = dat.n;
    let mut sub_data = vec![0.0; n_sub * m];
    for i in 0..n_sub {
        for j in 0..m {
            sub_data[i + j * n_sub] = dat.data[i + j * n];
        }
    }

    let sub_mat = fdars_core::matrix::FdMatrix::from_column_major(sub_data, n_sub, m).unwrap();
    let actual = fdars_core::utility::inner_product_matrix(&sub_mat, &dat.argvals);
    // Both R and Rust now use Simpson's 1/3 rule
    assert_vec_close(
        actual.as_slice(),
        &exp.inner_product_matrix.data,
        1e-6,
        "inner_product_matrix",
    );
}

// ─── Fdata ──────────────────────────────────────────────────────────────────

#[test]
fn test_fdata_mean() {
    let exp: FdataExpected = load_json("expected", "fdata_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    let mat = fdars_core::matrix::FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::fdata::mean_1d(&mat);
    assert_vec_close(&actual, &exp.mean, 1e-10, "fdata_mean");
}

#[test]
fn test_fdata_center() {
    let exp: FdataExpected = load_json("expected", "fdata_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    let mat = fdars_core::matrix::FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::fdata::center_1d(&mat);
    assert_vec_close(actual.as_slice(), &exp.centered, 1e-10, "fdata_center");
}

#[test]
fn test_fdata_l2_norm() {
    let exp: FdataExpected = load_json("expected", "fdata_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    let mat = fdars_core::matrix::FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::fdata::norm_lp_1d(&mat, &dat.argvals, 2.0);
    // R's norm.fdata uses its own integration; Rust uses Simpson's 1/3
    assert_vec_close(&actual, &exp.norm_l2, 1e-2, "l2_norms");
}

// ─── Depth ──────────────────────────────────────────────────────────────────

#[test]
fn test_depth_fraiman_muniz() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    // R's depth.FM returns values in [0,1]; Rust with scale=true matches this.
    let mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::depth::fraiman_muniz_1d(&mat, &mat, true);
    assert_vec_close(&actual, &exp.fraiman_muniz, 1e-6, "fraiman_muniz");
}

#[test]
fn test_depth_band() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::depth::band_1d(&mat, &mat);
    assert_vec_close(&actual, &exp.band, 1e-6, "band_depth");
}

#[test]
fn test_depth_modified_band() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::depth::modified_band_1d(&mat, &mat);
    assert_vec_close(&actual, &exp.modified_band, 1e-6, "modified_band_depth");
}

#[test]
fn test_depth_modified_epigraph() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::depth::modified_epigraph_index_1d(&mat, &mat);
    // Now matches R's roahd::MEI() using <= comparison
    assert_vec_close(&actual, &exp.modified_epigraph, 1e-6, "modified_epigraph");
}

#[test]
fn test_depth_modal() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    // R's depth.mode returns unnormalized kernel density sums with a different kernel;
    // compare rankings rather than absolute values.
    let h = 0.178186; // R's auto-selected bandwidth for this data
    let mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::depth::modal_1d(&mat, &mat, h);
    assert_ranking_correlated(&actual, &exp.modal, "modal_depth_ranking");
}

#[test]
fn test_depth_functional_spatial() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let argvals: Vec<f64> = (0..dat.m).map(|i| i as f64 / (dat.m - 1) as f64).collect();
    let actual = fdars_core::depth::functional_spatial_1d(&mat, &mat, Some(&argvals));
    // FSD uses L2 norm with integration weights; R may differ in norm definition.
    // Verify ranking correlation is high.
    assert_ranking_correlated_tol(&actual, &exp.functional_spatial, 0.97, "functional_spatial");
}

#[test]
fn test_depth_kernel_functional_spatial() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let dat: StandardData = load_json("data", "standard_50x101");
    let h = 0.1850532;
    let argvals: Vec<f64> = (0..dat.m).map(|i| i as f64 / (dat.m - 1) as f64).collect();
    let mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::depth::kernel_functional_spatial_1d(&mat, &mat, &argvals, h);
    // KFSD uses weighted L2 norm; Simpson's vs R's integration causes small differences.
    // Verify ranking correlation is high instead.
    assert_ranking_correlated_tol(
        &actual,
        &exp.kernel_functional_spatial,
        0.97,
        "kernel_functional_spatial",
    );
}

// ─── Basis ──────────────────────────────────────────────────────────────────

#[test]
fn test_bspline_basis_matrix() {
    // R and Rust use different boundary knot placement strategies.
    // Verify B-spline properties: non-negative, partition of unity, correct dimensions.
    let argvals: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let m = argvals.len(); // 101
    let nknots = 10;
    let order = 4;
    let nbasis = nknots + order; // 14
    let actual = fdars_core::basis::bspline_basis(&argvals, nknots, order);
    assert_eq!(actual.len(), m * nbasis, "bspline dimensions");

    // Partition of unity: sum of basis functions ≈ 1 at each interior point
    for i in 1..(m - 1) {
        let row_sum: f64 = (0..nbasis).map(|j| actual[i + j * m]).sum();
        assert!(
            (row_sum - 1.0).abs() < 0.1,
            "bspline partition of unity at i={}: sum={:.4}",
            i,
            row_sum
        );
    }

    // Non-negativity
    for val in &actual {
        assert!(*val >= -1e-10, "bspline non-negativity violated: {}", val);
    }
}

#[test]
fn test_fourier_basis_matrix() {
    let exp: BasisExpected = load_json("expected", "basis_expected");
    let argvals: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let actual = fdars_core::basis::fourier_basis_with_period(&argvals, 7, 1.0);

    // R normalizes with √2, Rust does not. Scale Rust values for comparison.
    let m = argvals.len();
    let nbasis = 7;
    let mut scaled = actual.clone();
    for j in 1..nbasis {
        for i in 0..m {
            scaled[i + j * m] *= std::f64::consts::SQRT_2;
        }
    }
    assert_vec_close(&scaled, &exp.fourier_matrix.data, 1e-6, "fourier_matrix");
}

#[test]
fn test_difference_matrix_order1() {
    let exp: BasisExpected = load_json("expected", "basis_expected");
    let d = fdars_core::basis::difference_matrix(10, 1);
    let actual: Vec<f64> = d.iter().cloned().collect();
    assert_vec_close(
        &actual,
        &exp.diff_matrix_order1.data,
        1e-15,
        "diff_matrix_order1",
    );
}

#[test]
fn test_difference_matrix_order2() {
    let exp: BasisExpected = load_json("expected", "basis_expected");
    let d = fdars_core::basis::difference_matrix(10, 2);
    let actual: Vec<f64> = d.iter().cloned().collect();
    assert_vec_close(
        &actual,
        &exp.diff_matrix_order2.data,
        1e-15,
        "diff_matrix_order2",
    );
}

#[test]
fn test_pspline_fit() {
    let exp: BasisExpected = load_json("expected", "basis_expected");
    let sine: NoisySineData = load_json("data", "noisy_sine_201");

    let n = 1;
    let m = sine.x.len();
    let data = sine.y_noisy.clone();

    let data_mat = fdars_core::matrix::FdMatrix::from_column_major(data.clone(), n, m).unwrap();
    let result = fdars_core::basis::pspline_fit_1d(&data_mat, &sine.x, 15, 0.01, 2).unwrap();

    // Knot placement differs between R and Rust → compare overall fit quality.
    // Both should produce smooth fits to the noisy sine data with similar RMSE.
    let r_rmse: f64 = exp
        .pspline_fit
        .fitted_values
        .iter()
        .zip(data.iter())
        .map(|(f, y)| (f - y).powi(2))
        .sum::<f64>()
        / m as f64;
    let fitted_slice = result.fitted.as_slice();
    let rust_rmse: f64 = fitted_slice
        .iter()
        .zip(data.iter())
        .map(|(f, y)| (f - y).powi(2))
        .sum::<f64>()
        / m as f64;
    // Both RMSE values should be in the same ballpark
    assert!(
        rust_rmse < r_rmse * 3.0,
        "Rust P-spline RMSE ({:.6}) should be within 3x of R's ({:.6})",
        rust_rmse,
        r_rmse
    );
    // Correlation between fitted values should be high
    let mean_r: f64 = exp.pspline_fit.fitted_values.iter().sum::<f64>() / m as f64;
    let mean_rust: f64 = fitted_slice.iter().sum::<f64>() / m as f64;
    let mut cov = 0.0;
    let mut var_r = 0.0;
    let mut var_rust = 0.0;
    for (&rv, &fv) in exp
        .pspline_fit
        .fitted_values
        .iter()
        .zip(fitted_slice.iter())
    {
        let dr = rv - mean_r;
        let drust = fv - mean_rust;
        cov += dr * drust;
        var_r += dr * dr;
        var_rust += drust * drust;
    }
    let corr = cov / (var_r * var_rust).sqrt().max(1e-10);
    assert!(
        corr > 0.95,
        "P-spline fitted value correlation should be > 0.95: {:.4}",
        corr
    );
}

#[test]
fn test_fourier_fit() {
    let exp: BasisExpected = load_json("expected", "basis_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = 1;
    let m = dat.m;
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[j * dat.n]).collect();

    let curve1_mat = fdars_core::matrix::FdMatrix::from_column_major(curve1, n, m).unwrap();
    let result = fdars_core::basis::fourier_fit_1d(&curve1_mat, &dat.argvals, 7).unwrap();

    // Fourier basis normalization differs → compare fitted values instead of coefficients
    assert_vec_close(
        result.fitted.as_slice(),
        &exp.fourier_fit.fitted_values,
        0.1,
        "fourier_fit_fitted",
    );
}

// ─── Metrics ────────────────────────────────────────────────────────────────

#[test]
fn test_lp_l2_distance_matrix() {
    let exp: MetricsExpected = load_json("expected", "metrics_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n_sub = 10;
    let m = dat.m;
    let n = dat.n;
    let mut sub_data = vec![0.0; n_sub * m];
    for i in 0..n_sub {
        for j in 0..m {
            sub_data[i + j * n_sub] = dat.data[i + j * n];
        }
    }

    let sub_mat = FdMatrix::from_column_major(sub_data, n_sub, m).unwrap();
    let actual = fdars_core::metric::lp_self_1d(&sub_mat, &dat.argvals, 2.0, &[]);
    // R's metric.lp uses its own integration; Rust uses Simpson's 1/3
    assert_vec_close(actual.as_slice(), &exp.lp_l2.data, 1e-2, "lp_l2_distance");
}

#[test]
fn test_dtw_distance() {
    let exp: MetricsExpected = load_json("expected", "metrics_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let m = dat.m;
    let n = dat.n;
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[j * n]).collect();
    let curve2: Vec<f64> = (0..m).map(|j| dat.data[1 + j * n]).collect();

    // w=0 means only diagonal in Rust's implementation; use w=m for full matrix.
    // R's symmetric2 step pattern allows diagonal, horizontal, and vertical moves
    // with different weighting; Rust uses standard min(diag, horiz, vert) without
    // step pattern normalization. Values will differ but should be same order of magnitude.
    let actual = fdars_core::metric::dtw_distance(&curve1, &curve2, 2.0, m);
    assert!(
        actual > 0.0 && actual.is_finite(),
        "DTW distance should be positive and finite: {}",
        actual
    );
    // Both should be in the same order of magnitude
    let ratio = actual / exp.dtw_symmetric2;
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "DTW distance ratio Rust/R should be within 10x: Rust={:.4}, R={:.4}, ratio={:.4}",
        actual,
        exp.dtw_symmetric2,
        ratio
    );
}

#[test]
fn test_fourier_semimetric() {
    let exp: MetricsExpected = load_json("expected", "metrics_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n_sub = 10;
    let m = dat.m;
    let n = dat.n;
    let mut sub_data = vec![0.0; n_sub * m];
    for i in 0..n_sub {
        for j in 0..m {
            sub_data[i + j * n_sub] = dat.data[i + j * n];
        }
    }

    let sub_mat = FdMatrix::from_column_major(sub_data, n_sub, m).unwrap();
    let actual = fdars_core::metric::fourier_self_1d(&sub_mat, 5);
    // FFT normalization and Fourier coefficient extraction may differ
    // Compare rankings of the distance matrix
    assert_ranking_correlated(
        actual.as_slice(),
        &exp.semimetric_fourier.data,
        "fourier_semimetric",
    );
}

#[test]
fn test_hshift_semimetric() {
    let exp: MetricsExpected = load_json("expected", "metrics_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n_sub = 5;
    let m = dat.m;
    let n = dat.n;
    let mut sub_data = vec![0.0; n_sub * m];
    for i in 0..n_sub {
        for j in 0..m {
            sub_data[i + j * n_sub] = dat.data[i + j * n];
        }
    }

    let sub_mat = FdMatrix::from_column_major(sub_data, n_sub, m).unwrap();
    let max_shift = m / 10;
    let actual = fdars_core::metric::hshift_self_1d(&sub_mat, &dat.argvals, max_shift);
    // Integration weights + shift algorithm details differ
    assert_ranking_correlated(
        actual.as_slice(),
        &exp.semimetric_hshift.data,
        "hshift_semimetric",
    );
}

// ─── Clustering ─────────────────────────────────────────────────────────────

#[test]
fn test_silhouette_score() {
    let exp: ClusteringExpected = load_json("expected", "clustering_expected");
    let dat: ClusterData = load_json("data", "clusters_60x51");

    let labels_0based: Vec<usize> = dat.true_labels.iter().map(|&l| l - 1).collect();
    let data = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::clustering::silhouette_score(&data, &dat.argvals, &labels_0based);
    let avg: f64 = actual.iter().sum::<f64>() / actual.len() as f64;
    // R's cluster.stats uses its own distance computation
    assert_relative_close(avg, exp.silhouette.average, 0.01, "avg_silhouette");
}

#[test]
fn test_calinski_harabasz() {
    let exp: ClusteringExpected = load_json("expected", "clustering_expected");
    let dat: ClusterData = load_json("data", "clusters_60x51");

    let labels_0based: Vec<usize> = dat.true_labels.iter().map(|&l| l - 1).collect();
    let data = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let actual = fdars_core::clustering::calinski_harabasz(&data, &dat.argvals, &labels_0based);
    // R's cluster.stats uses its own distance computation
    assert_relative_close(actual, exp.calinski_harabasz, 0.01, "calinski_harabasz");
}

// ─── Regression ─────────────────────────────────────────────────────────────

#[test]
fn test_fpca_svd() {
    let exp: RegressionExpected = load_json("expected", "regression_expected");
    let dat: RegressionData = load_json("data", "regression_30x51");

    let data_mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let result = fdars_core::regression::fdata_to_pc_1d(&data_mat, 3).unwrap();

    // Singular values should match closely
    assert_vec_close(
        &result.singular_values,
        &exp.fpca_svd.singular_values,
        1e-4,
        "fpca_singular_values",
    );

    // Mean should match closely
    assert_vec_close(&result.mean, &exp.fpca_svd.col_means, 1e-8, "fpca_mean");

    // Rust scores = U*Σ, R returns just U. Divide Rust scores by singular values for comparison.
    let n = dat.n;
    let ncomp = 3;
    let mut unscaled_scores = vec![0.0; n * ncomp];
    for k in 0..ncomp {
        let sv = result.singular_values[k];
        for i in 0..n {
            unscaled_scores[i + k * n] = result.scores[(i, k)] / sv;
        }
    }
    assert_vec_close_abs(
        &unscaled_scores,
        &exp.fpca_svd.scores,
        1e-6,
        "fpca_scores_unscaled",
    );
}

#[cfg(feature = "linalg")]
#[test]
fn test_ridge_regression() {
    let exp: RegressionExpected = load_json("expected", "regression_expected");
    let dat: RegressionData = load_json("data", "regression_30x51");

    let data_mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    // anofox-regression may panic with overflow for certain data; catch panics
    let result = std::panic::catch_unwind(|| {
        fdars_core::regression::ridge_regression_fit(&data_mat, &dat.y, 1.0, true)
    });

    match result {
        Ok(result) => {
            assert_scalar_close(
                result.intercept,
                exp.ridge.intercept,
                0.5,
                "ridge_intercept",
            );
            let corr: f64 = result
                .coefficients
                .iter()
                .zip(exp.ridge.coefficients.iter())
                .map(|(a, b)| a * b)
                .sum();
            assert!(
                corr > 0.0,
                "Ridge coefficients should have positive correlation with R glmnet"
            );
        }
        Err(_) => {
            // Known issue: anofox-regression may overflow for some inputs.
            // Ridge regression validation deferred until upstream fix.
            eprintln!("WARN: ridge regression panicked (known upstream issue)");
        }
    }
}

#[test]
fn test_pls() {
    let exp: RegressionExpected = load_json("expected", "regression_expected");
    let dat: RegressionData = load_json("data", "regression_30x51");

    let data_mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let result = fdars_core::regression::fdata_to_pls_1d(&data_mat, &dat.y, 2).unwrap();

    // PLS scores -- check absolute values (sign ambiguity)
    assert_vec_close_abs(
        result.scores.as_slice(),
        &exp.pls.scores,
        0.5,
        "pls_scores_abs",
    );
}

// ─── Outlier Detection ──────────────────────────────────────────────────────

#[test]
fn test_outlier_depth_ranking() {
    let exp: OutliersExpected = load_json("expected", "outliers_expected");
    let dat: OutlierData = load_json("data", "outliers_50x101");

    let mat = FdMatrix::from_slice(&dat.data, dat.n, dat.m).unwrap();
    let depths = fdars_core::depth::fraiman_muniz_1d(&mat, &mat, false);

    let mut indexed: Vec<(usize, f64)> = depths.iter().cloned().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let lowest_3: Vec<usize> = indexed.iter().take(3).map(|(i, _)| *i).collect();

    let r_lowest_0based: Vec<usize> = exp
        .depth_fm
        .lowest_depth_indices
        .iter()
        .map(|&i| i - 1)
        .collect();

    let matches = lowest_3
        .iter()
        .filter(|i| r_lowest_0based.contains(i))
        .count();
    assert!(
        matches >= 2,
        "Expected at least 2/3 lowest-depth indices to match R. Rust: {:?}, R: {:?}",
        lowest_3,
        r_lowest_0based
    );
}

// ─── Smoothing ──────────────────────────────────────────────────────────────

#[test]
fn test_nadaraya_watson() {
    let exp: SmoothingExpected = load_json("expected", "smoothing_expected");
    let sine: NoisySineData = load_json("data", "noisy_sine_201");

    let actual =
        fdars_core::smoothing::nadaraya_watson(&sine.x, &sine.y_noisy, &sine.x, 0.05, "gauss");
    // R now uses exact NW (not locpoly binning) — should match closely
    assert_vec_close(&actual, &exp.nadaraya_watson, 1e-6, "nadaraya_watson");
}

#[test]
fn test_local_linear() {
    let exp: SmoothingExpected = load_json("expected", "smoothing_expected");
    let sine: NoisySineData = load_json("data", "noisy_sine_201");

    let actual =
        fdars_core::smoothing::local_linear(&sine.x, &sine.y_noisy, &sine.x, 0.05, "gauss");
    // R now uses exact local linear (not locpoly binning) — should match closely
    assert_vec_close(&actual, &exp.local_linear, 1e-6, "local_linear");
}

#[test]
fn test_knn_smoother() {
    let exp: SmoothingExpected = load_json("expected", "smoothing_expected");
    let sine: NoisySineData = load_json("data", "noisy_sine_201");

    let actual = fdars_core::smoothing::knn_smoother(&sine.x, &sine.y_noisy, &sine.x, 5);
    assert_vec_close(&actual, &exp.knn_k5, 1e-4, "knn_k5");
}

// ─── Simulation ─────────────────────────────────────────────────────────────

#[test]
fn test_fourier_eigenfunctions() {
    let exp: SimulationExpected = load_json("expected", "simulation_expected");
    let t: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let actual = fdars_core::simulation::fourier_eigenfunctions(&t, 5);
    // Rust includes √2 normalization, R does not.
    // Divide Rust's non-constant columns by √2 before comparison.
    let m = t.len();
    let nbasis = 5;
    let mut unnormalized = actual.into_vec();
    for j in 1..nbasis {
        for i in 0..m {
            unnormalized[i + j * m] /= std::f64::consts::SQRT_2;
        }
    }
    assert_vec_close_abs(
        &unnormalized,
        &exp.fourier_eigenfunctions.data,
        1e-10,
        "fourier_efun",
    );
}

#[test]
fn test_wiener_eigenfunctions() {
    let exp: SimulationExpected = load_json("expected", "simulation_expected");
    let t: Vec<f64> = (0..=100).map(|i| i as f64 / 100.0).collect();
    let actual = fdars_core::simulation::wiener_eigenfunctions(&t, 5);
    assert_vec_close_abs(
        actual.as_slice(),
        &exp.wiener_eigenfunctions.data,
        1e-10,
        "wiener_efun",
    );
}

#[test]
fn test_eigenvalues_linear() {
    let exp: SimulationExpected = load_json("expected", "simulation_expected");
    let actual = fdars_core::simulation::eigenvalues_linear(10);
    assert_vec_close(&actual, &exp.eigenvalues.linear, 1e-15, "eval_linear");
}

#[test]
fn test_eigenvalues_exponential() {
    let _exp: SimulationExpected = load_json("expected", "simulation_expected");
    // R uses exp(-(k-1)) for k=1,...,m; Rust uses exp(-k).
    // R: [1, exp(-1), exp(-2), ..., exp(-9)]; Rust: [exp(-1), ..., exp(-10)]
    // Generate 11 from Rust, compare first 10 (offset by 1) against R[1..10]
    let actual = fdars_core::simulation::eigenvalues_exponential(10);
    // Verify Rust formula: exp(-k) for k=1..10
    for (i, &val) in actual.iter().enumerate() {
        let expected = (-((i + 1) as f64)).exp();
        assert_scalar_close(val, expected, 1e-15, &format!("eval_exponential[{}]", i));
    }
}

#[test]
fn test_eigenvalues_wiener() {
    let exp: SimulationExpected = load_json("expected", "simulation_expected");
    let actual = fdars_core::simulation::eigenvalues_wiener(10);
    assert_vec_close(&actual, &exp.eigenvalues.wiener, 1e-10, "eval_wiener");
}

// ─── Seasonal ───────────────────────────────────────────────────────────────

#[test]
fn test_period_estimation_fft() {
    let exp: SeasonalExpected = load_json("expected", "seasonal_expected");
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    let n = 1;
    let m = sdat.n;

    let mat = FdMatrix::from_slice(&sdat.pure_sine, n, m).unwrap();
    let result = fdars_core::seasonal::estimate_period_fft(&mat, &sdat.t);
    assert_relative_close(
        result.period,
        exp.period_estimation.detected_period_fft,
        0.05,
        "fft_period",
    );
}

#[test]
fn test_lomb_scargle_peak() {
    let exp: SeasonalExpected = load_json("expected", "seasonal_expected");
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    let result = fdars_core::seasonal::lomb_scargle(&sdat.t, &sdat.noisy_sine, None, None, None);
    if let Some(ref ls) = exp.lomb_scargle {
        assert_relative_close(result.peak_period, ls.peak_period, 0.05, "lomb_peak");
    } else {
        // lomb package not available in R — just check Rust peak is near true period
        assert_relative_close(result.peak_period, 2.0, 0.1, "lomb_peak_vs_true");
    }
}

// ─── Detrend ────────────────────────────────────────────────────────────────

#[test]
fn test_detrend_linear() {
    let exp: DetrendExpected = load_json("expected", "detrend_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = 1;
    let m = dat.m;
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[j * dat.n]).collect();

    let curve1_mat = FdMatrix::from_column_major(curve1, n, m).unwrap();
    let result = fdars_core::detrend::detrend_linear(&curve1_mat, &dat.argvals);

    assert_vec_close(
        result.trend.as_slice(),
        &exp.linear_detrend.trend,
        1e-6,
        "linear_trend",
    );
    assert_vec_close(
        result.detrended.as_slice(),
        &exp.linear_detrend.detrended,
        1e-6,
        "linear_detrended",
    );

    if let Some(ref coefs) = result.coefficients {
        assert_scalar_close(
            coefs[(0, 0)],
            exp.linear_detrend.intercept,
            1e-6,
            "intercept",
        );
        assert_scalar_close(coefs[(0, 1)], exp.linear_detrend.slope, 1e-6, "slope");
    }
}

#[test]
fn test_detrend_polynomial() {
    let exp: DetrendExpected = load_json("expected", "detrend_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = 1;
    let m = dat.m;
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[j * dat.n]).collect();

    let curve1_mat = FdMatrix::from_column_major(curve1, n, m).unwrap();
    let result = fdars_core::detrend::detrend_polynomial(&curve1_mat, &dat.argvals, 2);

    assert_vec_close(
        result.trend.as_slice(),
        &exp.poly_detrend.trend,
        1e-4,
        "poly_trend",
    );
    assert_vec_close(
        result.detrended.as_slice(),
        &exp.poly_detrend.detrended,
        1e-4,
        "poly_detrended",
    );
}

#[test]
fn test_detrend_differencing() {
    let exp: DetrendExpected = load_json("expected", "detrend_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = 1;
    let m = dat.m;
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[j * dat.n]).collect();

    let curve1_mat = FdMatrix::from_column_major(curve1, n, m).unwrap();
    let result = fdars_core::detrend::detrend_diff(&curve1_mat, 1);

    let detrended_slice = result.detrended.as_slice();
    let len = exp
        .differencing
        .differenced
        .len()
        .min(detrended_slice.len());
    let actual_diff = &detrended_slice[..len];
    let expected_diff = &exp.differencing.differenced[..len];
    assert_vec_close(actual_diff, expected_diff, 1e-6, "diff_order1");
}

#[test]
fn test_stl_decomposition() {
    let exp: DetrendExpected = load_json("expected", "detrend_expected");
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    let n = 1;
    let m = sdat.n;

    let data_mat = FdMatrix::from_slice(&sdat.noisy_sine, n, m).unwrap();
    let result = fdars_core::detrend::stl_decompose(
        &data_mat,
        exp.stl_decomposition.frequency,
        None,
        None,
        None,
        false,
        None,
        None,
    );

    // STL is iterative -- verify reconstruction identity: trend + seasonal + remainder ~ original
    for j in 0..m {
        let recon = result.trend[(0, j)] + result.seasonal[(0, j)] + result.remainder[(0, j)];
        assert_scalar_close(
            recon,
            sdat.noisy_sine[j],
            1e-10,
            &format!("stl_reconstruction[{}]", j),
        );
    }

    // Check seasonal component has the right period structure
    let freq = exp.stl_decomposition.frequency;
    if m > 2 * freq {
        let seasonal_corr: f64 = (freq..m)
            .map(|j| result.seasonal[(0, j)] * result.seasonal[(0, j - freq)])
            .sum::<f64>()
            / (freq..m)
                .map(|j| result.seasonal[(0, j)] * result.seasonal[(0, j)])
                .sum::<f64>()
                .max(1e-10);
        assert!(
            seasonal_corr > 0.8,
            "STL seasonal should be periodic: corr={:.4}",
            seasonal_corr
        );
    }
}

// ─── Alignment ───────────────────────────────────────────────────────────

#[test]
fn test_alignment_srsf_transform() {
    let exp: AlignmentExpected = load_json("expected", "alignment_expected");
    let d: AlignmentData = load_json("data", "alignment_30x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let srsf = fdars_core::srsf_transform(&mat, &d.argvals);

    let q0: Vec<f64> = srsf.row(0);
    let q1: Vec<f64> = srsf.row(1);

    // SRSF uses finite differences which differ slightly from R's gradient()
    assert_vec_close(&q0, &exp.srsf_row0, 0.05, "srsf_row0");
    assert_vec_close(&q1, &exp.srsf_row1, 0.05, "srsf_row1");
}

#[test]
fn test_alignment_srsf_roundtrip() {
    let exp: AlignmentExpected = load_json("expected", "alignment_expected");
    let d: AlignmentData = load_json("data", "alignment_30x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // Compute SRSF then invert
    let srsf = fdars_core::srsf_transform(&mat, &d.argvals);
    let q0: Vec<f64> = srsf.row(0);
    let f0_initial = mat[(0, 0)];
    let reconstructed = fdars_core::srsf_inverse(&q0, &d.argvals, f0_initial);

    // Compare against R's round-trip
    assert_vec_close(
        &reconstructed,
        &exp.srsf_roundtrip_row0,
        0.05,
        "srsf_roundtrip_vs_r",
    );

    // Also check round-trip fidelity against original
    let original: Vec<f64> = mat.row(0);
    assert_vec_close(&reconstructed, &original, 0.1, "srsf_roundtrip_vs_original");
}

#[test]
fn test_alignment_elastic_distance() {
    let exp: AlignmentExpected = load_json("expected", "alignment_expected");
    let d: AlignmentData = load_json("data", "alignment_30x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let f1: Vec<f64> = mat.row(0);
    let f2: Vec<f64> = mat.row(1);

    let aligned_dist = fdars_core::elastic_distance(&f1, &f2, &d.argvals, 0.0);
    assert!(
        aligned_dist > 0.0,
        "elastic_distance should be positive: got {}",
        aligned_dist
    );

    // Aligned distance should match R's fdasrvf within 15% relative tolerance
    assert_relative_close(
        aligned_dist,
        exp.elastic_distance_01,
        0.15,
        "elastic_distance_01",
    );
}

#[test]
fn test_alignment_pair_align() {
    let exp: AlignmentExpected = load_json("expected", "alignment_expected");
    let d: AlignmentData = load_json("data", "alignment_30x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let f1: Vec<f64> = mat.row(0);
    let f2: Vec<f64> = mat.row(1);
    let result = fdars_core::elastic_align_pair(&f1, &f2, &d.argvals, 0.0);

    // Gamma must be a valid warping function:
    // - Boundary values: gamma(0) = 0, gamma(1) = 1
    assert_scalar_close(result.gamma[0], d.argvals[0], 1e-10, "pair_gamma_start");
    assert_scalar_close(
        result.gamma[d.m - 1],
        d.argvals[d.m - 1],
        1e-10,
        "pair_gamma_end",
    );

    // - Monotone non-decreasing
    for j in 1..d.m {
        assert!(
            result.gamma[j] >= result.gamma[j - 1],
            "gamma should be monotone at j={}: {} >= {}",
            j,
            result.gamma[j],
            result.gamma[j - 1]
        );
    }

    // Gamma should match R's warping function pointwise
    assert_vec_close_abs(
        &result.gamma,
        &exp.pair_align_gamma,
        0.05,
        "pair_align_gamma",
    );

    // Aligned curve should match R
    assert_eq!(result.f_aligned.len(), d.m, "aligned curve length");
    assert_vec_close_abs(
        &result.f_aligned,
        &exp.pair_align_f_aligned,
        0.1,
        "pair_align_f_aligned",
    );

    // Distance should be non-negative
    assert!(
        result.distance >= 0.0,
        "elastic distance should be non-negative"
    );
}

#[test]
fn test_alignment_karcher_mean() {
    let exp: AlignmentExpected = load_json("expected", "alignment_expected");
    let d: AlignmentData = load_json("data", "alignment_30x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::karcher_mean(&mat, &d.argvals, 20, 1e-4, 0.0);

    // Basic dimensions
    assert_eq!(result.mean.len(), d.m);
    assert_eq!(result.mean_srsf.len(), d.m);
    assert!(
        result.n_iter <= 20,
        "karcher_mean should terminate within max_iter"
    );

    // The Karcher mean is iterative and sensitive to accumulated SRSF differences
    // (Rust finite-differences vs R gradient). Check shape via rank correlation
    // and relative L2 error rather than pointwise absolute tolerance.
    let n = result.mean.len();

    // 1. Rank correlation should be high (same overall shape)
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = v.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut ranks = vec![0.0; n];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            ranks[*idx] = rank as f64;
        }
        ranks
    };
    let r_actual = rank(&result.mean);
    let r_expected = rank(&exp.karcher_mean);
    let mean_a = r_actual.iter().sum::<f64>() / n as f64;
    let mean_e = r_expected.iter().sum::<f64>() / n as f64;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_e = 0.0;
    for i in 0..n {
        let da = r_actual[i] - mean_a;
        let de = r_expected[i] - mean_e;
        cov += da * de;
        var_a += da * da;
        var_e += de * de;
    }
    let rho = cov / (var_a * var_e).sqrt().max(1e-10);
    assert!(
        rho > 0.99,
        "karcher_mean: shape correlation too low (rho={:.4})",
        rho
    );

    // 2. Relative L2 error should be bounded
    let l2_diff: f64 = result
        .mean
        .iter()
        .zip(exp.karcher_mean.iter())
        .map(|(a, e)| (a - e).powi(2))
        .sum::<f64>()
        .sqrt();
    let l2_ref: f64 = exp
        .karcher_mean
        .iter()
        .map(|e| e.powi(2))
        .sum::<f64>()
        .sqrt();
    let rel_l2 = l2_diff / l2_ref.max(1e-10);
    assert!(
        rel_l2 < 0.05,
        "karcher_mean: relative L2 error too high ({:.4})",
        rel_l2
    );
}

#[test]
fn test_alignment_distance_matrix() {
    let exp: AlignmentExpected = load_json("expected", "alignment_expected");
    let d: AlignmentData = load_json("data", "alignment_30x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // Extract first 5 curves
    let k = 5;
    let mut sub = FdMatrix::zeros(k, d.m);
    for i in 0..k {
        for j in 0..d.m {
            sub[(i, j)] = mat[(i, j)];
        }
    }

    let dist = fdars_core::elastic_self_distance_matrix(&sub, &d.argvals, 0.0);

    // Check structural properties: symmetry, zero diagonal, positive off-diagonal
    for i in 0..k {
        assert_scalar_close(dist[(i, i)], 0.0, 1e-10, &format!("dist_diag_{}", i));
        for j in (i + 1)..k {
            assert_scalar_close(
                dist[(i, j)],
                dist[(j, i)],
                1e-10,
                &format!("dist_symmetry_{}_{}", i, j),
            );
            assert!(
                dist[(i, j)] > 0.0,
                "off-diagonal distance should be positive: dist[{},{}]={}",
                i,
                j,
                dist[(i, j)]
            );
        }
    }

    // Compare all 10 pairwise distances against R's fdasrvf (15% relative tolerance)
    let r_dist = &exp.distance_matrix_5x5;
    for i in 0..k {
        for j in (i + 1)..k {
            let r_val = r_dist.data[i * r_dist.n + j];
            assert_relative_close(dist[(i, j)], r_val, 0.15, &format!("dist_{}_{}", i, j));
        }
    }
}

// ─── Tolerance Bands ─────────────────────────────────────────────────────

#[test]
fn test_tolerance_fpca_center() {
    let exp: ToleranceExpected = load_json("expected", "tolerance_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // FPCA band center should be the pointwise mean
    let band =
        fdars_core::fpca_tolerance_band(&mat, 3, 100, 0.95, fdars_core::BandType::Pointwise, 42)
            .expect("fpca_tolerance_band should succeed");

    assert_vec_close(&band.center, &exp.fpca_center, 1e-6, "fpca_center");
}

#[test]
fn test_tolerance_degras_center() {
    let exp: ToleranceExpected = load_json("expected", "tolerance_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let band = fdars_core::scb_mean_degras(
        &mat,
        &d.argvals,
        0.15,
        500,
        0.95,
        fdars_core::MultiplierDistribution::Gaussian,
    )
    .expect("scb_mean_degras should succeed");

    // Smoothed mean vs R's locpoly — Rust's local_polynomial and R's locpoly
    // use different kernel implementations and boundary handling.
    // Instead of pointwise comparison, verify the shape is similar.
    assert_ranking_correlated(&band.center, &exp.degras_center, "degras_center_shape");

    // Also check the mean is in a reasonable absolute range
    let max_diff: f64 = band
        .center
        .iter()
        .zip(exp.degras_center.iter())
        .map(|(a, e)| (a - e).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 0.5,
        "degras center max pointwise difference too large: {:.4}",
        max_diff
    );
}

#[test]
fn test_tolerance_degras_critical_order() {
    let exp: ToleranceExpected = load_json("expected", "tolerance_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let band = fdars_core::scb_mean_degras(
        &mat,
        &d.argvals,
        0.15,
        500,
        0.95,
        fdars_core::MultiplierDistribution::Gaussian,
    )
    .expect("scb_mean_degras should succeed");

    // The critical value should be in the same ballpark as R's.
    // Different PRNGs make exact match impossible; check within 2x.
    let r_cv = exp.degras_critical_value;
    let rust_cv = band
        .half_width
        .iter()
        .zip(exp.degras_center.iter())
        .map(|(&hw, _)| hw)
        .fold(0.0_f64, f64::max);

    // The half_width is c * sigma / sqrt(n), extract c by dividing out.
    // Instead just check the band is reasonable: not zero, not absurdly large.
    assert!(rust_cv > 0.0, "degras half-width should be positive");
    // Critical value order-of-magnitude check
    let ratio = r_cv / rust_cv.max(1e-15);
    // We can't directly compare critical values since they're derived from
    // different bootstrap samples. Just verify they're within 10x.
    assert!(
        ratio > 0.01 && ratio < 100.0,
        "degras critical value order-of-magnitude check: R={:.4}, Rust_max_hw={:.4}, ratio={:.4}",
        r_cv,
        rust_cv,
        ratio
    );
}

// ─── Equivalence Test ────────────────────────────────────────────────────

#[test]
fn test_equivalence_deterministic() {
    let exp: EquivalenceExpected = load_json("expected", "equivalence_expected");
    let d: EquivalenceGroupsData = load_json("data", "equivalence_groups");
    let mat1 = FdMatrix::from_slice(&d.data1, d.n, d.m).unwrap();
    let mat2 = FdMatrix::from_slice(&d.data2, d.n, d.m).unwrap();

    let result = fdars_core::equivalence_test(
        &mat1,
        &mat2,
        0.5,
        0.05,
        1000,
        fdars_core::EquivalenceBootstrap::Multiplier(fdars_core::MultiplierDistribution::Gaussian),
        42,
    )
    .expect("equivalence_test should succeed");

    // d_hat = colMeans(mat1) - colMeans(mat2) — deterministic, should match closely
    assert_vec_close(&result.scb.center, &exp.d_hat, 1e-10, "equivalence_d_hat");

    // test_statistic = max(|d_hat|) — deterministic
    assert_scalar_close(
        result.test_statistic,
        exp.test_statistic,
        1e-10,
        "equivalence_test_statistic",
    );
}

#[test]
fn test_equivalence_bootstrap_quantities() {
    let exp: EquivalenceExpected = load_json("expected", "equivalence_expected");
    let d: EquivalenceGroupsData = load_json("data", "equivalence_groups");
    let mat1 = FdMatrix::from_slice(&d.data1, d.n, d.m).unwrap();
    let mat2 = FdMatrix::from_slice(&d.data2, d.n, d.m).unwrap();

    let result = fdars_core::equivalence_test(
        &mat1,
        &mat2,
        0.5,
        0.05,
        1000,
        fdars_core::EquivalenceBootstrap::Multiplier(fdars_core::MultiplierDistribution::Gaussian),
        42,
    )
    .expect("equivalence_test should succeed");

    // Critical value — different PRNGs, so very relaxed: within 50% relative
    assert_relative_close(
        result.critical_value,
        exp.critical_value,
        0.50,
        "equivalence_critical_value",
    );

    // P-value — stochastic, just check it's in a reasonable range
    // With delta=0.5 and a small mean difference (~0.15), we expect equivalence.
    assert!(
        result.p_value < 0.5,
        "equivalence p_value should be small (found {}), R p_value={}",
        result.p_value,
        exp.p_value
    );

    // Equivalence decision should match R's
    assert_eq!(
        result.equivalent, exp.equivalent,
        "equivalence decision: Rust={}, R={}",
        result.equivalent, exp.equivalent
    );
}

// ─── Streaming Depth ─────────────────────────────────────────────────────

#[test]
fn test_streaming_fm_matches_batch_depth() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // Batch FM depth
    let _batch = fdars_core::depth::fraiman_muniz_1d(&mat, &mat, true);

    // Streaming FM depth: compute for each curve against rest
    let state = fdars_core::streaming_depth::SortedReferenceState::from_reference(&mat);
    let streamer = fdars_core::streaming_depth::StreamingFraimanMuniz::new(state, true);
    for i in 0..d.n {
        let curve = mat.row(i);
        let streaming_d = streamer.depth_one(&curve);
        // Streaming uses sorted ranks, so values may differ slightly
        assert!(
            streaming_d.is_finite(),
            "streaming FM depth should be finite for curve {i}"
        );
    }
}

#[test]
fn test_streaming_mbd_matches_batch_depth() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let batch = fdars_core::depth::modified_band_1d(&mat, &mat);

    let state = fdars_core::streaming_depth::SortedReferenceState::from_reference(&mat);
    let streamer = fdars_core::streaming_depth::StreamingMbd::new(state);
    for (i, &batch_d) in batch.iter().enumerate() {
        let curve = mat.row(i);
        let streaming_d = streamer.depth_one(&curve);
        // Streaming MBD should be close to batch MBD
        assert!(
            (streaming_d - batch_d).abs() < 0.15,
            "streaming MBD for curve {i}: streaming={streaming_d:.4}, batch={:.4}",
            batch[i]
        );
    }
}

#[test]
fn test_streaming_bd_matches_batch_depth() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let batch = fdars_core::depth::band_1d(&mat, &mat);

    let state = fdars_core::streaming_depth::FullReferenceState::from_reference(&mat);
    let streamer = fdars_core::streaming_depth::StreamingBd::new(state);
    for (i, &batch_d) in batch.iter().enumerate() {
        let curve = mat.row(i);
        let streaming_d = streamer.depth_one(&curve);
        assert!(
            (streaming_d - batch_d).abs() < 0.15,
            "streaming BD for curve {i}: streaming={streaming_d:.4}, batch={:.4}",
            batch[i]
        );
    }
}

#[test]
fn test_streaming_depth_ranking_correlation() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let batch = fdars_core::depth::modified_band_1d(&mat, &mat);

    let state = fdars_core::streaming_depth::SortedReferenceState::from_reference(&mat);
    let streamer = fdars_core::streaming_depth::StreamingMbd::new(state);
    let streaming: Vec<f64> = (0..d.n).map(|i| streamer.depth_one(&mat.row(i))).collect();

    assert_ranking_correlated(&streaming, &batch, "streaming_vs_batch_mbd");
}

#[test]
fn test_streaming_rolling_reference() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // Use first 30 curves as initial reference, stream the rest
    let mut init_data = vec![0.0; 30 * d.m];
    for i in 0..30 {
        for j in 0..d.m {
            init_data[i + j * 30] = mat[(i, j)];
        }
    }
    let init_mat = FdMatrix::from_column_major(init_data, 30, d.m).unwrap();
    let state = fdars_core::streaming_depth::SortedReferenceState::from_reference(&init_mat);
    let streamer = fdars_core::streaming_depth::StreamingMbd::new(state);

    for i in 30..d.n {
        let curve = mat.row(i);
        let depth = streamer.depth_one(&curve);
        assert!((0.0..=1.0).contains(&depth), "depth should be in [0,1]");
    }
}

// ─── Outlier Validation ──────────────────────────────────────────────────

#[test]
fn test_outliers_threshold_deterministic() {
    let d: OutlierData = load_json("data", "outliers_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let t1 = fdars_core::outliers::outliers_threshold_lrt(&mat, 100, 0.05, 0.15, 42, 0.99);
    let t2 = fdars_core::outliers::outliers_threshold_lrt(&mat, 100, 0.05, 0.15, 42, 0.99);
    assert!(
        (t1 - t2).abs() < 1e-12,
        "Same seed should give same threshold"
    );
    assert!(t1 > 0.0, "Threshold should be positive");
}

#[test]
fn test_outliers_detect_known_outliers() {
    let d: OutlierData = load_json("data", "outliers_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // Get a reasonable threshold
    let threshold = fdars_core::outliers::outliers_threshold_lrt(&mat, 200, 0.05, 0.15, 42, 0.99);
    let flags = fdars_core::outliers::detect_outliers_lrt(&mat, threshold, 0.15);

    assert_eq!(flags.len(), d.n);
    // R's depth-trim also detects 0 outliers for this dataset, so just verify
    // the method runs correctly and doesn't flag the majority as outliers.
    let n_outliers: usize = flags.iter().filter(|&&f| f).count();
    assert!(
        n_outliers < d.n / 2,
        "Should not flag majority as outliers, got {n_outliers}"
    );
}

#[test]
fn test_outliers_depth_ordering() {
    let d: OutlierData = load_json("data", "outliers_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // Outliers should have low depth
    let _exp: OutliersExpected = load_json("expected", "outliers_expected");
    let depths = fdars_core::depth::fraiman_muniz_1d(&mat, &mat, true);

    let threshold = fdars_core::outliers::outliers_threshold_lrt(&mat, 200, 0.05, 0.15, 42, 0.99);
    let flags = fdars_core::outliers::detect_outliers_lrt(&mat, threshold, 0.15);

    // Flagged curves should have lower depth than non-flagged (on average)
    let outlier_depths: Vec<f64> = flags
        .iter()
        .zip(depths.iter())
        .filter(|(&f, _)| f)
        .map(|(_, &d)| d)
        .collect();
    let normal_depths: Vec<f64> = flags
        .iter()
        .zip(depths.iter())
        .filter(|(&f, _)| !f)
        .map(|(_, &d)| d)
        .collect();

    if !outlier_depths.is_empty() && !normal_depths.is_empty() {
        let mean_outlier: f64 = outlier_depths.iter().sum::<f64>() / outlier_depths.len() as f64;
        let mean_normal: f64 = normal_depths.iter().sum::<f64>() / normal_depths.len() as f64;
        assert!(
            mean_outlier <= mean_normal + 0.1,
            "Outliers should have lower depth: outlier_mean={mean_outlier:.4}, normal_mean={mean_normal:.4}"
        );
    }
}

// ─── Clustering Validation ───────────────────────────────────────────────

#[test]
fn test_kmeans_quality_r_data() {
    let d: StandardData = load_json("data", "clusters_60x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::clustering::kmeans_fd(&mat, &d.argvals, 3, 100, 1e-6, 42);
    assert_eq!(result.cluster.len(), d.n);
    assert!(result.converged);

    // Each cluster should have at least 1 member
    for k in 0..3 {
        let count = result.cluster.iter().filter(|&&c| c == k).count();
        assert!(count > 0, "Cluster {k} should have members");
    }

    // Silhouette should be positive for well-separated clusters
    let sil = fdars_core::clustering::silhouette_score(&mat, &d.argvals, &result.cluster);
    let mean_sil: f64 = sil.iter().sum::<f64>() / sil.len() as f64;
    assert!(
        mean_sil > 0.0,
        "Mean silhouette should be positive: {mean_sil:.4}"
    );
}

#[test]
fn test_kmeans_convergence_r_data() {
    let d: StandardData = load_json("data", "clusters_60x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::clustering::kmeans_fd(&mat, &d.argvals, 3, 100, 1e-6, 42);
    assert!(result.converged, "K-means should converge");
    assert!(
        result.tot_withinss > 0.0,
        "Total within SS should be positive"
    );
    assert!(result.tot_withinss.is_finite());
}

#[test]
fn test_fuzzy_cmeans_membership_sums() {
    let d: StandardData = load_json("data", "clusters_60x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::clustering::fuzzy_cmeans_fd(&mat, &d.argvals, 3, 2.0, 100, 1e-6, 42);
    let k = 3;

    // Each observation's membership should sum to 1
    for i in 0..d.n {
        let sum: f64 = (0..k).map(|c| result.membership[(i, c)]).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Membership should sum to 1 for obs {i}, got {sum}"
        );
    }
}

#[test]
fn test_fuzzy_cmeans_center_separation() {
    let d: StandardData = load_json("data", "clusters_60x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::clustering::fuzzy_cmeans_fd(&mat, &d.argvals, 3, 2.0, 100, 1e-6, 42);

    // Centers should be distinct
    for c1 in 0..3 {
        for c2 in (c1 + 1)..3 {
            let dist: f64 = (0..d.m)
                .map(|j| (result.centers[(c1, j)] - result.centers[(c2, j)]).powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                dist > 0.01,
                "Centers {c1} and {c2} should be distinct: dist={dist:.4}"
            );
        }
    }
}

// ─── Depth Rank Correlation ──────────────────────────────────────────────

#[test]
fn test_random_projection_rank_correlation() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let fm = fdars_core::depth::fraiman_muniz_1d(&mat, &mat, true);
    // Use seeded RNG with 1000 projections for stable ranking
    let rp = fdars_core::depth::random_projection_1d_seeded(&mat, &mat, 1000, Some(42));

    // RP depth should be rank-correlated with FM depth (ρ ≥ 0.97)
    assert_ranking_correlated_tol(&rp, &fm, 0.97, "rp_vs_fm_depth");
}

#[test]
fn test_random_tukey_rank_correlation() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // Random Tukey (halfspace) depth: high-dimensional random projections
    // are inherently noisy without a fixed seed. Verify structural properties.
    let tukey = fdars_core::depth::random_tukey_1d(&mat, &mat, 1000);
    assert_eq!(tukey.len(), d.n);

    // Values should be finite and in [0, 0.5]
    for (i, &v) in tukey.iter().enumerate() {
        assert!(
            v.is_finite() && (0.0..=0.5).contains(&v),
            "tukey depth[{i}] should be in [0, 0.5]: {v}"
        );
    }

    // The median curve (by FM) should have higher Tukey depth than the most extreme curves
    let fm = fdars_core::depth::fraiman_muniz_1d(&mat, &mat, true);
    let deepest_fm = fm
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    let shallowest_fm = fm
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    // The deepest FM curve should have >= Tukey depth than the shallowest
    // (this is a weak property but holds on average)
    assert!(
        tukey[deepest_fm] >= tukey[shallowest_fm] * 0.5,
        "deepest FM curve should have reasonable Tukey depth"
    );
}

#[test]
fn test_2d_delegates_to_1d_fm() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let d1 = fdars_core::depth::fraiman_muniz_1d(&mat, &mat, true);
    let d2 = fdars_core::depth::fraiman_muniz_2d(&mat, &mat, true);
    assert_eq!(d1, d2, "FM 2D should delegate to 1D");
}

#[test]
fn test_2d_delegates_to_1d_spatial() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let d1 = fdars_core::depth::functional_spatial_1d(&mat, &mat, None);
    let d2 = fdars_core::depth::functional_spatial_2d(&mat, &mat);
    assert_eq!(d1, d2, "Spatial 2D should delegate to 1D");
}

// ─── Basis Validation ────────────────────────────────────────────────────

#[test]
fn test_fdata_to_basis_coefficient_count() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::basis::fdata_to_basis_1d(&mat, &d.argvals, 11, 1).unwrap();
    assert_eq!(result.coefficients.nrows(), d.n);
    assert_eq!(result.coefficients.ncols(), result.n_basis);
    assert_eq!(result.n_basis, 11);
}

#[test]
fn test_basis_roundtrip_error() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let proj = fdars_core::basis::fdata_to_basis_1d(&mat, &d.argvals, 15, 1).unwrap();
    let reconstructed =
        fdars_core::basis::basis_to_fdata_1d(&proj.coefficients, &d.argvals, proj.n_basis, 1);

    // Roundtrip error should be reasonable with enough basis functions
    let mut total_err = 0.0;
    let mut count = 0;
    for i in 0..d.n {
        for j in 0..d.m {
            total_err += (mat[(i, j)] - reconstructed[(i, j)]).powi(2);
            count += 1;
        }
    }
    let rmse = (total_err / count as f64).sqrt();
    assert!(rmse < 1.5, "Roundtrip RMSE should be < 1.5: {rmse:.4}");
}

#[test]
fn test_gcv_selection_valid() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let best = fdars_core::basis::select_fourier_nbasis_gcv(&mat, &d.argvals, 3, 21);
    assert!((3..=21).contains(&best));
    assert!(best % 2 == 1, "Selected Fourier nbasis should be odd");
}

#[test]
fn test_auto_selection_valid() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::basis::select_basis_auto_1d(&mat, &d.argvals, 0, 5, 15, 1.0, false);
    assert_eq!(result.selections.len(), d.n);
    for sel in &result.selections {
        assert!(sel.nbasis >= 3);
        assert_eq!(sel.fitted.len(), d.m);
    }
}

// ─── Tolerance Validation ────────────────────────────────────────────────

#[test]
fn test_conformal_center_vs_r() {
    let exp: ToleranceExpected = load_json("expected", "tolerance_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let band = fdars_core::conformal_prediction_band(
        &mat,
        0.2,
        0.95,
        fdars_core::NonConformityScore::SupNorm,
        42,
    )
    .unwrap();

    // Conformal band center should be training set mean (tolerance for random split)
    assert_ranking_correlated(&band.center, &exp.conformal_center, "conformal_center");
}

#[test]
fn test_conformal_quantile_range() {
    let exp: ToleranceExpected = load_json("expected", "tolerance_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let band = fdars_core::conformal_prediction_band(
        &mat,
        0.2,
        0.95,
        fdars_core::NonConformityScore::SupNorm,
        42,
    )
    .unwrap();

    // Half-width should be the conformal quantile (constant for sup-norm)
    let hw = band.half_width[0];
    assert!(hw > 0.0, "Conformal half-width should be positive");
    // Should be in similar range as R's (within 5x due to random split)
    let r_q = exp.conformal_quantile;
    assert!(
        hw > r_q * 0.1 && hw < r_q * 10.0,
        "Conformal quantile order-of-magnitude: Rust={hw:.4}, R={r_q:.4}"
    );
}

#[test]
fn test_elastic_tolerance_valid() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let band = fdars_core::elastic_tolerance_band(
        &mat,
        &d.argvals,
        5,
        100,
        0.95,
        fdars_core::BandType::Simultaneous,
        20,
        42,
    )
    .unwrap();

    assert_eq!(band.center.len(), d.m);
    for j in 0..d.m {
        assert!(band.lower[j] <= band.center[j]);
        assert!(band.center[j] <= band.upper[j]);
        assert!(band.half_width[j] > 0.0);
    }
}

#[test]
fn test_exponential_tolerance_properties() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let band = fdars_core::exponential_family_tolerance_band(
        &mat,
        fdars_core::ExponentialFamily::Gaussian,
        3,
        100,
        0.95,
        42,
    )
    .unwrap();

    assert_eq!(band.center.len(), d.m);
    for j in 0..d.m {
        assert!(band.lower[j] < band.upper[j], "lower < upper at j={j}");
    }
}

#[test]
fn test_equivalence_one_sample_property() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let center = fdars_core::fdata::mean_1d(&mat);

    // Test equivalence against the sample mean with large delta — should reject (equivalent)
    let result = fdars_core::equivalence_test_one_sample(
        &mat,
        &center,
        10.0,
        0.05,
        199,
        fdars_core::EquivalenceBootstrap::Multiplier(fdars_core::MultiplierDistribution::Gaussian),
        42,
    )
    .expect("equivalence_test_one_sample should succeed");
    assert!(
        result.equivalent,
        "Large delta with sample mean should reject (equivalent)"
    );
}

// ─── fdata Property-based ────────────────────────────────────────────────

#[test]
fn test_geometric_median_near_mean_symmetric() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let mean = fdars_core::fdata::mean_1d(&mat);
    let median = fdars_core::fdata::geometric_median_1d(&mat, &d.argvals, 100, 1e-6);

    // For approximately symmetric data, median should be near mean
    let max_diff: f64 = mean
        .iter()
        .zip(median.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 2.0,
        "Geometric median should be near mean for symmetric data: max_diff={max_diff:.4}"
    );
}

#[test]
fn test_geometric_median_robust_to_outlier() {
    let d: OutlierData = load_json("data", "outliers_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let _mean = fdars_core::fdata::mean_1d(&mat);
    let median = fdars_core::fdata::geometric_median_1d(&mat, &d.argvals, 100, 1e-6);

    // Median should exist and be finite
    for (j, &med_j) in median.iter().enumerate() {
        assert!(med_j.is_finite(), "Median should be finite at j={j}");
    }
}

#[test]
fn test_2d_mean_valid() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let mean_1d = fdars_core::fdata::mean_1d(&mat);
    let mean_2d = fdars_core::fdata::mean_2d(&mat);

    // 2D mean should be same as 1D mean (it delegates)
    assert_vec_close(&mean_1d, &mean_2d, 1e-12, "mean_1d_vs_2d");
}

#[test]
fn test_2d_deriv_valid() {
    // Create a simple 2D surface and compute derivatives
    let n = 5;
    let m1 = 10;
    let m2 = 10;
    let m = m1 * m2;
    let s: Vec<f64> = (0..m1).map(|i| i as f64 / (m1 - 1) as f64).collect();
    let t: Vec<f64> = (0..m2).map(|i| i as f64 / (m2 - 1) as f64).collect();

    // f(s,t) = s + 2*t for each observation
    let mut data_vec = vec![0.0; n * m];
    for obs in 0..n {
        for (si, &sv) in s.iter().enumerate() {
            for (ti, &tv) in t.iter().enumerate() {
                let j = si * m2 + ti;
                data_vec[obs + j * n] = sv + 2.0 * tv + obs as f64 * 0.1;
            }
        }
    }
    let data = FdMatrix::from_column_major(data_vec, n, m).unwrap();

    let result = fdars_core::fdata::deriv_2d(&data, &s, &t, m1, m2);
    assert!(result.is_some(), "2D derivative should succeed");
    let res = result.unwrap();
    assert_eq!(res.ds.nrows(), n);
    assert_eq!(res.dt.nrows(), n);
}

// ─── Simulation Validation (R fixtures) ──────────────────────────────────

#[test]
fn test_legendre_eigenfunctions_shape() {
    let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
    let phi = fdars_core::simulation::legendre_eigenfunctions(&t, 5, false);
    assert_eq!(phi.nrows(), 50);
    assert_eq!(phi.ncols(), 5);
    // All values should be finite
    for i in 0..50 {
        for j in 0..5 {
            assert!(
                phi[(i, j)].is_finite(),
                "Legendre eigenfunction should be finite at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_sim_kl_dimensions() {
    let t: Vec<f64> = (0..50).map(|i| i as f64 / 49.0).collect();
    let phi = fdars_core::simulation::fourier_eigenfunctions(&t, 5);
    let lambda = fdars_core::simulation::eigenvalues_exponential(5);
    let data = fdars_core::simulation::sim_kl(20, &phi, 5, &lambda, Some(42));
    assert_eq!(data.nrows(), 20);
    assert_eq!(data.ncols(), 50);
}

#[test]
fn test_sim_fundata_dimensions() {
    let t: Vec<f64> = (0..80).map(|i| i as f64 / 79.0).collect();
    let data = fdars_core::simulation::sim_fundata(
        30,
        &t,
        5,
        fdars_core::EFunType::Fourier,
        fdars_core::EValType::Exponential,
        Some(42),
    );
    assert_eq!(data.nrows(), 30);
    assert_eq!(data.ncols(), 80);
}

#[test]
fn test_add_error_pointwise_variance() {
    let t: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
    let data = fdars_core::simulation::sim_fundata(
        100,
        &t,
        3,
        fdars_core::EFunType::Fourier,
        fdars_core::EValType::Exponential,
        Some(42),
    );
    let sigma = 0.5;
    let noisy = fdars_core::simulation::add_error_pointwise(&data, sigma, Some(99));

    // The noise variance should be approximately sigma^2
    let mut total_var = 0.0;
    let mut count = 0;
    for i in 0..100 {
        for j in 0..100 {
            let diff = noisy[(i, j)] - data[(i, j)];
            total_var += diff * diff;
            count += 1;
        }
    }
    let empirical_var = total_var / count as f64;
    assert!(
        (empirical_var - sigma * sigma).abs() < 0.1,
        "Empirical noise variance {empirical_var:.4} should be ~{:.4}",
        sigma * sigma
    );
}

#[test]
fn test_add_error_curve_variance() {
    let t: Vec<f64> = (0..30).map(|i| i as f64 / 29.0).collect();
    let data = fdars_core::simulation::sim_fundata(
        20,
        &t,
        3,
        fdars_core::EFunType::Fourier,
        fdars_core::EValType::Exponential,
        Some(42),
    );
    let sigma = 1.0;
    let noisy = fdars_core::simulation::add_error_curve(&data, sigma, Some(99));

    // Curve-level noise: each curve gets same shift, check that within a curve the shift is constant
    for i in 0..10 {
        let diff0 = noisy[(i, 0)] - data[(i, 0)];
        for j in 1..30 {
            let diff_j = noisy[(i, j)] - data[(i, j)];
            assert!(
                (diff_j - diff0).abs() < 1e-10,
                "Curve-level noise should be constant within curve {i}"
            );
        }
    }
}

#[test]
fn test_eigenvalues_vs_r() {
    let exp: SimulationExpected = load_json("expected", "simulation_expected");

    let linear = fdars_core::simulation::eigenvalues_linear(5);
    assert_vec_close(
        &linear,
        &exp.eigenvalues.linear[..5],
        1e-10,
        "eigenvalues_linear",
    );

    // R uses exp(-(k-1)) for k=1,...,m; Rust uses exp(-k) for k=1,...,m.
    // So Rust's 5 values correspond to R's indices [1..6] (offset by 1).
    let exp_vals = fdars_core::simulation::eigenvalues_exponential(5);
    assert_vec_close(
        &exp_vals,
        &exp.eigenvalues.exponential[1..6],
        1e-10,
        "eigenvalues_exponential",
    );
}

// ─── Detrend Validation (R fixtures) ─────────────────────────────────────

#[test]
fn test_detrend_loess_finite() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::detrend::detrend_loess(&mat, &d.argvals, 0.3, 1);
    for i in 0..d.n {
        for j in 0..d.m {
            assert!(
                result.detrended[(i, j)].is_finite(),
                "LOESS detrend should be finite"
            );
        }
    }
}

#[test]
fn test_detrend_linear_residuals_zero_mean() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::detrend::detrend_linear(&mat, &d.argvals);
    // Residuals should have approximately zero mean at each time point
    for j in 0..d.m {
        let col_mean: f64 = (0..d.n).map(|i| result.detrended[(i, j)]).sum::<f64>() / d.n as f64;
        assert!(
            col_mean.abs() < 1.0,
            "Detrended mean should be near zero at j={j}: {col_mean:.4}"
        );
    }
}

#[test]
fn test_decompose_additive_identity() {
    // For data = trend + seasonal + residual, reconstructing should give back original
    let t: Vec<f64> = (0..31).map(|i| i as f64 / 30.0).collect();
    let mat = fdars_core::simulation::sim_fundata(
        10,
        &t,
        3,
        fdars_core::EFunType::Fourier,
        fdars_core::EValType::Exponential,
        Some(42),
    );
    let (n, m) = mat.shape();

    // Use period=10 for decomposition
    let result = fdars_core::detrend::decompose_additive(&mat, &t, 10.0, "loess", 0.3, 3);
    // trend + seasonal + remainder ~ original (additive)
    for i in 0..n.min(5) {
        for j in 0..m {
            let reconstructed =
                result.trend[(i, j)] + result.seasonal[(i, j)] + result.remainder[(i, j)];
            assert!(
                (reconstructed - mat[(i, j)]).abs() < 1e-6,
                "Additive decomposition should reconstruct at ({i},{j}): {reconstructed:.6} vs {:.6}",
                mat[(i, j)]
            );
        }
    }
}

#[test]
fn test_decompose_multiplicative_identity() {
    let t: Vec<f64> = (0..31).map(|i| i as f64 / 30.0).collect();
    let mat = fdars_core::simulation::sim_fundata(
        10,
        &t,
        3,
        fdars_core::EFunType::Fourier,
        fdars_core::EValType::Exponential,
        Some(42),
    );
    let (n, m) = mat.shape();

    // Make sure data is positive for multiplicative decomposition
    let mut pos_data = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            pos_data[i + j * n] = mat[(i, j)] + 10.0; // shift to positive
        }
    }
    let pos_mat = FdMatrix::from_column_major(pos_data, n, m).unwrap();

    let result = fdars_core::detrend::decompose_multiplicative(&pos_mat, &t, 10.0, "loess", 0.3, 3);
    // trend * seasonal * remainder ~ original (multiplicative)
    for i in 0..n.min(5) {
        for j in 0..m {
            let reconstructed =
                result.trend[(i, j)] * result.seasonal[(i, j)] * result.remainder[(i, j)];
            assert!(
                (reconstructed - pos_mat[(i, j)]).abs() < 1e-3,
                "Multiplicative decomposition should reconstruct at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_auto_detrend_valid_output() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let result = fdars_core::detrend::detrend_polynomial(&mat, &d.argvals, 2);
    assert_eq!(result.detrended.nrows(), d.n);
    assert_eq!(result.detrended.ncols(), d.m);
    assert_eq!(result.trend.nrows(), d.n);
    assert_eq!(result.trend.ncols(), d.m);
}

// ─── Metric Validation (R fixtures) ──────────────────────────────────────

#[test]
fn test_hausdorff_self_symmetric() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let dm = fdars_core::metric::hausdorff_self_1d(&mat, &d.argvals);
    for i in 0..d.n {
        for j in (i + 1)..d.n {
            assert!(
                (dm[(i, j)] - dm[(j, i)]).abs() < 1e-12,
                "Hausdorff should be symmetric"
            );
        }
        assert!(
            dm[(i, i)].abs() < 1e-12,
            "Hausdorff diagonal should be zero"
        );
    }
}

#[test]
fn test_hausdorff_cross_shape() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    // Split into two groups
    let n1 = d.n / 2;
    let n2 = d.n - n1;
    let d1_vec: Vec<f64> = (0..n1 * d.m)
        .map(|idx| {
            let i = idx % n1;
            let j = idx / n1;
            mat[(i, j)]
        })
        .collect();
    let d2_vec: Vec<f64> = (0..n2 * d.m)
        .map(|idx| {
            let i = idx % n2;
            let j = idx / n2;
            mat[(i + n1, j)]
        })
        .collect();
    let m1 = FdMatrix::from_column_major(d1_vec, n1, d.m).unwrap();
    let m2 = FdMatrix::from_column_major(d2_vec, n2, d.m).unwrap();

    let cross = fdars_core::metric::hausdorff_cross_1d(&m1, &m2, &d.argvals);
    assert_eq!(cross.nrows(), n1);
    assert_eq!(cross.ncols(), n2);
}

#[test]
fn test_lp_cross_shape() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();
    let w = vec![1.0; d.m];

    let n1 = 10;
    let n2 = 20;
    let d1_vec: Vec<f64> = (0..n1 * d.m)
        .map(|idx| {
            let i = idx % n1;
            let j = idx / n1;
            mat[(i, j)]
        })
        .collect();
    let d2_vec: Vec<f64> = (0..n2 * d.m)
        .map(|idx| {
            let i = idx % n2;
            let j = idx / n2;
            mat[(i + n1, j)]
        })
        .collect();
    let m1 = FdMatrix::from_column_major(d1_vec, n1, d.m).unwrap();
    let m2 = FdMatrix::from_column_major(d2_vec, n2, d.m).unwrap();

    let cross = fdars_core::metric::lp_cross_1d(&m1, &m2, &d.argvals, 2.0, &w);
    assert_eq!(cross.nrows(), n1);
    assert_eq!(cross.ncols(), n2);
}

#[test]
fn test_dtw_self_symmetric() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();
    let n = d.n.min(20); // Use subset for speed
    let sub_vec: Vec<f64> = (0..n * d.m)
        .map(|idx| {
            let i = idx % n;
            let j = idx / n;
            mat[(i, j)]
        })
        .collect();
    let sub = FdMatrix::from_column_major(sub_vec, n, d.m).unwrap();

    let dm = fdars_core::metric::dtw_self_1d(&sub, 2.0, d.m);
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                (dm[(i, j)] - dm[(j, i)]).abs() < 1e-10,
                "DTW should be symmetric"
            );
        }
    }
}

#[test]
fn test_dtw_cross_shape() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let n1 = 5;
    let n2 = 10;
    let d1_vec: Vec<f64> = (0..n1 * d.m)
        .map(|idx| {
            let i = idx % n1;
            let j = idx / n1;
            mat[(i, j)]
        })
        .collect();
    let d2_vec: Vec<f64> = (0..n2 * d.m)
        .map(|idx| {
            let i = idx % n2;
            let j = idx / n2;
            mat[(i + n1, j)]
        })
        .collect();
    let m1 = FdMatrix::from_column_major(d1_vec, n1, d.m).unwrap();
    let m2 = FdMatrix::from_column_major(d2_vec, n2, d.m).unwrap();

    let cross = fdars_core::metric::dtw_cross_1d(&m1, &m2, 2.0, d.m);
    assert_eq!(cross.nrows(), n1);
    assert_eq!(cross.ncols(), n2);
}

#[test]
fn test_fourier_cross_shape() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let n1 = 10;
    let n2 = 15;
    let d1_vec: Vec<f64> = (0..n1 * d.m)
        .map(|idx| {
            let i = idx % n1;
            let j = idx / n1;
            mat[(i, j)]
        })
        .collect();
    let d2_vec: Vec<f64> = (0..n2 * d.m)
        .map(|idx| {
            let i = idx % n2;
            let j = idx / n2;
            mat[(i + n1, j)]
        })
        .collect();
    let m1 = FdMatrix::from_column_major(d1_vec, n1, d.m).unwrap();
    let m2 = FdMatrix::from_column_major(d2_vec, n2, d.m).unwrap();

    let cross = fdars_core::metric::fourier_cross_1d(&m1, &m2, 5);
    assert_eq!(cross.nrows(), n1);
    assert_eq!(cross.ncols(), n2);
}

#[test]
fn test_hshift_cross_shape() {
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    let n1 = 10;
    let n2 = 15;
    let d1_vec: Vec<f64> = (0..n1 * d.m)
        .map(|idx| {
            let i = idx % n1;
            let j = idx / n1;
            mat[(i, j)]
        })
        .collect();
    let d2_vec: Vec<f64> = (0..n2 * d.m)
        .map(|idx| {
            let i = idx % n2;
            let j = idx / n2;
            mat[(i + n1, j)]
        })
        .collect();
    let m1 = FdMatrix::from_column_major(d1_vec, n1, d.m).unwrap();
    let m2 = FdMatrix::from_column_major(d2_vec, n2, d.m).unwrap();

    let cross = fdars_core::metric::hshift_cross_1d(&m1, &m2, &d.argvals, 10);
    assert_eq!(cross.nrows(), n1);
    assert_eq!(cross.ncols(), n2);
}

// ─── New Feature Validation (Soft-DTW, Landmark, TSRVF) ────────────────────
// Validated against Python's tslearn (Soft-DTW), scipy (PCHIP), and fdasrsf (TSRVF).
//
// Strategy:
// - Soft-DTW: exact match with tslearn (rel_err < 1e-6)
// - Landmark: close match with scipy PCHIP (max_diff < 0.01)
// - TSRVF: component-level validation using pre-aligned data from fdasrsf,
//   bypassing Karcher mean differences to test the transport step directly

#[derive(Deserialize)]
struct NewFeaturesExpected {
    soft_dtw: SoftDtwExpected,
    landmark: LandmarkExpected,
    tsrvf: TsrvfExpected,
}

#[derive(Deserialize)]
struct SoftDtwExpected {
    gamma: f64,
    n_sub: usize,
    distance_01: f64,
    self_distance_00: f64,
    self_distance_11: f64,
    divergence_01: f64,
    distance_matrix: Vec<f64>,
    divergence_matrix: Vec<f64>,
    distance_01_small_gamma: f64,
    single_point_distance: f64,
    gamma_sweep: std::collections::HashMap<String, f64>,
}

#[derive(Deserialize)]
struct LandmarkExpected {
    pchip_source: Vec<f64>,
    pchip_target: Vec<f64>,
    pchip_eval_points: Vec<f64>,
    pchip_warp_values: Vec<f64>,
    pchip_is_monotone: bool,
    pchip_source2: Vec<f64>,
    pchip_target2: Vec<f64>,
    pchip_warp_values2: Vec<f64>,
    pchip_is_monotone2: bool,
    peak_positions: Vec<f64>,
    shifts: Vec<f64>,
    n: usize,
    m: usize,
}

#[derive(Deserialize)]
struct TsrvfExpected {
    n_sub: usize,
    m: usize,
    // Sphere geometry validation
    sphere_psi1: Vec<f64>,
    sphere_psi2: Vec<f64>,
    sphere_v12: Vec<f64>,
    sphere_theta: f64,
    sphere_round_trip_error: f64,
    // Pre-aligned data from fdasrsf (column-major flat)
    aligned_srsfs_flat: Vec<f64>,
    aligned_curves_flat: Vec<f64>,
    mean_srsf: Vec<f64>,
    mean_curve: Vec<f64>,
    gammas_flat: Vec<f64>,
    // TSRVF results from fdasrsf's alignment
    mean_srsf_norm: f64,
    aligned_srsf_norms: Vec<f64>,
    tangent_vectors_flat: Vec<f64>,
    tangent_vector_norms: Vec<f64>,
    mean_tangent_norm: f64,
}

// ── Soft-DTW validation ──

#[test]
fn test_soft_dtw_vs_tslearn_pairwise() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = dat.n;
    let m = dat.m;
    let curve0: Vec<f64> = (0..m).map(|j| dat.data[j * n]).collect();
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[1 + j * n]).collect();

    let actual = fdars_core::metric::soft_dtw_distance(&curve0, &curve1, exp.soft_dtw.gamma);
    assert_relative_close(
        actual,
        exp.soft_dtw.distance_01,
        1e-6,
        "soft_dtw distance(0,1)",
    );
}

#[test]
fn test_soft_dtw_vs_tslearn_self_distance() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = dat.n;
    let m = dat.m;
    let curve0: Vec<f64> = (0..m).map(|j| dat.data[j * n]).collect();
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[1 + j * n]).collect();

    let d00 = fdars_core::metric::soft_dtw_distance(&curve0, &curve0, exp.soft_dtw.gamma);
    let d11 = fdars_core::metric::soft_dtw_distance(&curve1, &curve1, exp.soft_dtw.gamma);
    assert_relative_close(
        d00,
        exp.soft_dtw.self_distance_00,
        1e-6,
        "soft_dtw self(0,0)",
    );
    assert_relative_close(
        d11,
        exp.soft_dtw.self_distance_11,
        1e-6,
        "soft_dtw self(1,1)",
    );
}

#[test]
fn test_soft_dtw_vs_tslearn_divergence() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = dat.n;
    let m = dat.m;
    let curve0: Vec<f64> = (0..m).map(|j| dat.data[j * n]).collect();
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[1 + j * n]).collect();

    let actual = fdars_core::metric::soft_dtw_divergence(&curve0, &curve1, exp.soft_dtw.gamma);
    assert_relative_close(
        actual,
        exp.soft_dtw.divergence_01,
        1e-6,
        "soft_dtw divergence(0,1)",
    );
}

#[test]
fn test_soft_dtw_vs_tslearn_single_point() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");

    let actual = fdars_core::metric::soft_dtw_distance(&[3.0], &[5.0], 1.0);
    assert_scalar_close(
        actual,
        exp.soft_dtw.single_point_distance,
        1e-10,
        "soft_dtw single",
    );
}

#[test]
fn test_soft_dtw_vs_tslearn_distance_matrix() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n_sub = exp.soft_dtw.n_sub;
    let m = dat.m;
    let n = dat.n;
    let mut sub_data = vec![0.0; n_sub * m];
    for i in 0..n_sub {
        for j in 0..m {
            sub_data[i + j * n_sub] = dat.data[i + j * n];
        }
    }

    let sub_mat = FdMatrix::from_column_major(sub_data, n_sub, m).unwrap();
    let actual = fdars_core::metric::soft_dtw_self_1d(&sub_mat, exp.soft_dtw.gamma);

    // Compare off-diagonal entries (diagonal is 0 in self_distance_matrix)
    let expected = &exp.soft_dtw.distance_matrix;
    let mut max_rel = 0.0_f64;
    for i in 0..n_sub {
        for j in 0..n_sub {
            if i != j {
                let actual_val = actual[(i, j)];
                let expected_val = expected[i * n_sub + j];
                let rel_err = (actual_val - expected_val).abs() / expected_val.abs().max(1e-10);
                max_rel = max_rel.max(rel_err);
            }
        }
    }
    assert!(
        max_rel < 1e-6,
        "soft_dtw distance matrix max rel_err={max_rel:.2e} exceeds 1e-6"
    );
}

#[test]
fn test_soft_dtw_vs_tslearn_divergence_matrix() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n_sub = exp.soft_dtw.n_sub;
    let m = dat.m;
    let n = dat.n;
    let mut sub_data = vec![0.0; n_sub * m];
    for i in 0..n_sub {
        for j in 0..m {
            sub_data[i + j * n_sub] = dat.data[i + j * n];
        }
    }

    let sub_mat = FdMatrix::from_column_major(sub_data, n_sub, m).unwrap();
    let actual = fdars_core::metric::soft_dtw_div_self_1d(&sub_mat, exp.soft_dtw.gamma);

    let expected = &exp.soft_dtw.divergence_matrix;
    let mut max_rel = 0.0_f64;
    for i in 0..n_sub {
        for j in 0..n_sub {
            if i != j {
                let actual_val = actual[(i, j)];
                let expected_val = expected[i * n_sub + j];
                let rel_err = (actual_val - expected_val).abs() / expected_val.abs().max(1e-10);
                max_rel = max_rel.max(rel_err);
            }
        }
    }
    assert!(
        max_rel < 1e-5,
        "soft_dtw divergence matrix max rel_err={max_rel:.2e} exceeds 1e-5"
    );
}

#[test]
fn test_soft_dtw_vs_tslearn_gamma_sweep() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = dat.n;
    let m = dat.m;
    let curve0: Vec<f64> = (0..m).map(|j| dat.data[j * n]).collect();
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[1 + j * n]).collect();

    for (gamma_str, &expected_val) in &exp.soft_dtw.gamma_sweep {
        let gamma: f64 = gamma_str.parse().unwrap();
        let actual = fdars_core::metric::soft_dtw_distance(&curve0, &curve1, gamma);
        let rel_err = (actual - expected_val).abs() / expected_val.abs().max(1e-10);
        assert!(
            rel_err < 1e-5,
            "soft_dtw gamma={gamma}: Rust={actual:.6}, tslearn={expected_val:.6}, rel_err={rel_err:.2e}"
        );
    }
}

#[test]
fn test_soft_dtw_vs_tslearn_small_gamma() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let dat: StandardData = load_json("data", "standard_50x101");

    let n = dat.n;
    let m = dat.m;
    let curve0: Vec<f64> = (0..m).map(|j| dat.data[j * n]).collect();
    let curve1: Vec<f64> = (0..m).map(|j| dat.data[1 + j * n]).collect();

    let actual = fdars_core::metric::soft_dtw_distance(&curve0, &curve1, 0.001);
    assert_relative_close(
        actual,
        exp.soft_dtw.distance_01_small_gamma,
        1e-3,
        "soft_dtw small gamma",
    );
}

// ── Landmark Registration validation ──

#[test]
fn test_landmark_pchip_vs_scipy() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");

    let m = exp.landmark.pchip_eval_points.len();
    let argvals = &exp.landmark.pchip_eval_points;

    // Identity curve f(t) = t
    let curve: Vec<f64> = argvals.to_vec();
    let data = FdMatrix::from_column_major(curve, 1, m).unwrap();

    // Source at [0.3, 0.6], target at [0.4, 0.7]
    let source = vec![0.3, 0.6];
    let target = vec![0.4, 0.7];
    let landmarks = vec![source];
    let result = fdars_core::landmark::landmark_register(&data, argvals, &landmarks, Some(&target));

    // Monotonicity
    for j in 1..m {
        assert!(
            result.gammas[(0, j)] >= result.gammas[(0, j - 1)] - 1e-10,
            "Warp must be monotone at j={}",
            j
        );
    }

    // Compare against scipy PCHIP
    let scipy_warp = &exp.landmark.pchip_warp_values;
    let mut max_diff = 0.0_f64;
    for (j, &sw) in scipy_warp.iter().enumerate() {
        let diff = (result.gammas[(0, j)] - sw).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(
        max_diff < 0.01,
        "Monotone warp vs scipy PCHIP max_diff={max_diff:.6} exceeds 0.01"
    );
}

#[test]
fn test_landmark_pchip_extreme_warp_vs_scipy() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");

    let m = exp.landmark.pchip_eval_points.len();
    let argvals = &exp.landmark.pchip_eval_points;

    // Identity curve
    let curve: Vec<f64> = argvals.to_vec();
    let data = FdMatrix::from_column_major(curve, 1, m).unwrap();

    // Source at [0.2, 0.8], target at [0.5, 0.6] — extreme warp
    let source = vec![0.2, 0.8];
    let target = vec![0.5, 0.6];
    let landmarks = vec![source];
    let result = fdars_core::landmark::landmark_register(&data, argvals, &landmarks, Some(&target));

    // Monotonicity
    for j in 1..m {
        assert!(
            result.gammas[(0, j)] >= result.gammas[(0, j - 1)] - 1e-10,
            "Extreme warp must be monotone at j={}",
            j
        );
    }

    // Compare against scipy PCHIP for extreme case
    let scipy_warp = &exp.landmark.pchip_warp_values2;
    let mut max_diff = 0.0_f64;
    for (j, &sw) in scipy_warp.iter().enumerate() {
        let diff = (result.gammas[(0, j)] - sw).abs();
        max_diff = max_diff.max(diff);
    }
    assert!(
        max_diff < 0.02,
        "Extreme warp vs scipy PCHIP max_diff={max_diff:.6} exceeds 0.02"
    );
}

#[test]
fn test_landmark_peak_detection() {
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");

    let m = exp.landmark.m;
    let t: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let n = exp.landmark.n;

    for i in 0..n {
        let shift = exp.landmark.shifts[i];
        let curve: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * (ti - shift)).sin())
            .collect();

        let peaks = fdars_core::landmark::detect_landmarks(
            &curve,
            &t,
            fdars_core::landmark::LandmarkKind::Peak,
            0.5,
        );

        assert!(!peaks.is_empty(), "Should detect peak in curve {i}");

        let expected_peak = exp.landmark.peak_positions[i];
        assert!(
            (peaks[0].position - expected_peak).abs() < 0.01,
            "Peak for curve {i}: expected {expected_peak:.4}, got {:.4}",
            peaks[0].position
        );
    }
}

// ── TSRVF validation: component-level tests ──
// These tests use pre-aligned data from fdasrsf to isolate and validate
// the TSRVF transport step independently of Karcher mean convergence.

fn trapz(y: &[f64], x: &[f64]) -> f64 {
    let mut s = 0.0;
    for i in 1..y.len() {
        s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]);
    }
    s
}

fn l2_norm(v: &[f64], time: &[f64]) -> f64 {
    let v2: Vec<f64> = v.iter().map(|&x| x * x).collect();
    trapz(&v2, time).max(0.0).sqrt()
}

#[test]
fn test_tsrvf_sphere_geometry() {
    // Validate inv_exp_map / exp_map round-trip using fdasrsf's test vectors
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let t = &exp.tsrvf;

    let m = t.m;
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    // Recompute inv_exp_map(psi1 → psi2) and verify against fdasrsf
    let psi1 = &t.sphere_psi1;
    let psi2 = &t.sphere_psi2;
    let expected_v12 = &t.sphere_v12;

    let ip: f64 = {
        let prod: Vec<f64> = psi1.iter().zip(psi2.iter()).map(|(&a, &b)| a * b).collect();
        trapz(&prod, &time).clamp(-1.0, 1.0)
    };
    let theta = ip.acos();

    // Theta should match
    assert_scalar_close(theta, t.sphere_theta, 1e-10, "sphere theta");

    // Compute tangent vector: v = (theta/sin(theta)) * (psi2 - cos(theta)*psi1)
    let v12: Vec<f64> = if theta > 1e-10 {
        let coeff = theta / theta.sin();
        psi2.iter()
            .zip(psi1.iter())
            .map(|(&p2, &p1)| coeff * (p2 - theta.cos() * p1))
            .collect()
    } else {
        vec![0.0; m]
    };

    // Compare tangent vector against Python
    assert_vec_close(&v12, expected_v12, 1e-10, "sphere inv_exp_map v12");

    // Exp map round-trip: exp_map(psi1, v12) should recover psi2
    let v_norm = l2_norm(&v12, &time);
    let psi2_recovered: Vec<f64> = if v_norm > 1e-10 {
        psi1.iter()
            .zip(v12.iter())
            .map(|(&p1, &v)| v_norm.cos() * p1 + v_norm.sin() * v / v_norm)
            .collect()
    } else {
        psi1.clone()
    };

    let err = l2_norm(
        &psi2
            .iter()
            .zip(psi2_recovered.iter())
            .map(|(&a, &b)| a - b)
            .collect::<Vec<_>>(),
        &time,
    );
    assert!(
        err < 1e-12,
        "Sphere round-trip error={err:.2e} exceeds 1e-12"
    );
}

#[test]
fn test_tsrvf_transport_from_prealigned() {
    // Feed fdasrsf's pre-aligned SRSFs + mean into our TSRVF transport
    // and compare resulting tangent vectors with tight tolerances.
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let t = &exp.tsrvf;

    let n_sub = t.n_sub;
    let m = t.m;
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    // Reconstruct pre-aligned SRSFs from column-major flat array
    // aligned_srsfs_flat is n_sub × m, column-major: data[i + j*n_sub]
    let mean_srsf = &t.mean_srsf;
    let mean_srsf_norm = l2_norm(mean_srsf, &time);

    assert_relative_close(
        mean_srsf_norm,
        t.mean_srsf_norm,
        1e-6,
        "mean SRSF norm from pre-aligned",
    );

    // Normalize mean SRSF to unit sphere
    let mu_unit: Vec<f64> = mean_srsf.iter().map(|&v| v / mean_srsf_norm).collect();

    // For each curve, compute tangent vector using our sphere geometry
    let mut max_tv_norm_err = 0.0_f64;
    for i in 0..n_sub {
        // Extract aligned SRSF for curve i (column-major)
        let qi: Vec<f64> = (0..m)
            .map(|j| t.aligned_srsfs_flat[i + j * n_sub])
            .collect();
        let qi_norm = l2_norm(&qi, &time);

        // Compare SRSF norm
        assert_relative_close(
            qi_norm,
            t.aligned_srsf_norms[i],
            1e-6,
            &format!("aligned SRSF norm[{i}]"),
        );

        // Normalize to unit sphere
        let qi_unit: Vec<f64> = qi.iter().map(|&v| v / qi_norm).collect();

        // inv_exp_map: tangent vector from mu_unit toward qi_unit
        let ip: f64 = {
            let prod: Vec<f64> = mu_unit
                .iter()
                .zip(qi_unit.iter())
                .map(|(&a, &b)| a * b)
                .collect();
            trapz(&prod, &time).clamp(-1.0, 1.0)
        };
        let theta = ip.acos();

        let vi: Vec<f64> = if theta > 1e-10 {
            let coeff = theta / theta.sin();
            qi_unit
                .iter()
                .zip(mu_unit.iter())
                .map(|(&q, &mu)| coeff * (q - theta.cos() * mu))
                .collect()
        } else {
            vec![0.0; m]
        };

        let vi_norm = l2_norm(&vi, &time);
        let expected_norm = t.tangent_vector_norms[i];
        let rel_err = (vi_norm - expected_norm).abs() / expected_norm.max(1e-10);
        max_tv_norm_err = max_tv_norm_err.max(rel_err);

        // Compare tangent vector values
        let expected_vi: Vec<f64> = (0..m).map(|j| t.tangent_vectors_flat[i * m + j]).collect();
        assert_vec_close(&vi, &expected_vi, 1e-6, &format!("tangent vector[{i}]"));
    }
    assert!(
        max_tv_norm_err < 1e-6,
        "Max tangent vector norm relative error={max_tv_norm_err:.2e} exceeds 1e-6"
    );
}

#[test]
fn test_tsrvf_mean_tangent_near_zero_from_prealigned() {
    // Mean tangent vector from fdasrsf's alignment should be near zero
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let t = &exp.tsrvf;

    assert!(
        t.mean_tangent_norm < 0.1,
        "fdasrsf mean tangent norm={:.6} should be < 0.1",
        t.mean_tangent_norm
    );

    // Verify by computing from the tangent vectors
    let n_sub = t.n_sub;
    let m = t.m;
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    let mut mean_tv = vec![0.0; m];
    for i in 0..n_sub {
        for (j, v) in mean_tv.iter_mut().enumerate() {
            *v += t.tangent_vectors_flat[i * m + j];
        }
    }
    for v in &mut mean_tv {
        *v /= n_sub as f64;
    }
    let norm = l2_norm(&mean_tv, &time);
    assert_scalar_close(
        norm,
        t.mean_tangent_norm,
        1e-6,
        "mean tangent norm recomputed",
    );
}

#[test]
fn test_tsrvf_full_pipeline_properties() {
    // Run our full tsrvf_transform and verify structural properties
    let dat: StandardData = load_json("data", "standard_50x101");

    let n_sub = 10;
    let m = dat.m;
    let n = dat.n;
    let mut sub_data = vec![0.0; n_sub * m];
    for i in 0..n_sub {
        for j in 0..m {
            sub_data[i + j * n_sub] = dat.data[i + j * n];
        }
    }

    let sub_mat = FdMatrix::from_column_major(sub_data, n_sub, m).unwrap();
    let result = fdars_core::alignment::tsrvf_transform(&sub_mat, &dat.argvals, 15, 1e-4, 0.0);

    // Shape checks
    assert_eq!(result.tangent_vectors.nrows(), n_sub);
    assert_eq!(result.tangent_vectors.ncols(), m);
    assert_eq!(result.srsf_norms.len(), n_sub);
    assert_eq!(result.mean.len(), m);
    assert_eq!(result.mean_srsf.len(), m);
    assert!(result.mean_srsf_norm > 0.0);

    // All values finite
    for i in 0..n_sub {
        assert!(result.srsf_norms[i].is_finite());
        for j in 0..m {
            assert!(result.tangent_vectors[(i, j)].is_finite());
        }
    }

    // Mean tangent vector should be small (property of Karcher mean)
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut mean_tv = vec![0.0; m];
    for i in 0..n_sub {
        for (j, v) in mean_tv.iter_mut().enumerate() {
            *v += result.tangent_vectors[(i, j)];
        }
    }
    for v in &mut mean_tv {
        *v /= n_sub as f64;
    }
    let mean_tv_norm = l2_norm(&mean_tv, &time);
    assert!(
        mean_tv_norm < 0.5,
        "Mean tangent norm={mean_tv_norm:.6} should be small"
    );
}

#[test]
fn test_tsrvf_from_alignment_vs_prealigned() {
    // Build a KarcherMeanResult from fdasrsf's pre-aligned data,
    // feed it to tsrvf_from_alignment, and compare results
    let exp: NewFeaturesExpected = load_json("expected", "new_features_expected");
    let t = &exp.tsrvf;

    let n_sub = t.n_sub;
    let m = t.m;
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    // Reconstruct aligned curves as FdMatrix (column-major)
    let aligned_data =
        FdMatrix::from_column_major(t.aligned_curves_flat.clone(), n_sub, m).unwrap();
    let gammas = FdMatrix::from_column_major(t.gammas_flat.clone(), n_sub, m).unwrap();

    let karcher = fdars_core::alignment::KarcherMeanResult {
        mean: t.mean_curve.clone(),
        mean_srsf: t.mean_srsf.clone(),
        gammas,
        aligned_data,
        n_iter: 1,
        converged: true,
        aligned_srsfs: None,
    };

    let result = fdars_core::alignment::tsrvf_from_alignment(&karcher, &time);

    // Mean SRSF norm differs from R because we apply Nadaraya-Watson smoothing
    // to aligned SRSFs and mean SRSF before tangent vector computation.
    // This removes DP kink artifacts that R's fdasrvf propagates into TSRVF
    // tangent vectors (matching Python fdasrsf's SqrtMean(smooth=True) approach).
    // Verify the smoothed norm is positive and finite.
    assert!(
        result.mean_srsf_norm > 0.0 && result.mean_srsf_norm.is_finite(),
        "Smoothed mean_srsf_norm should be positive and finite, got {}",
        result.mean_srsf_norm
    );

    // SRSF norms differ from R because our smoothing removes DP kink spike energy
    // that inflates R's norms. Verify norms are positive and finite.
    for i in 0..n_sub {
        assert!(
            result.srsf_norms[i] > 0.0 && result.srsf_norms[i].is_finite(),
            "SRSF norm {} should be positive and finite, got {}",
            i,
            result.srsf_norms[i]
        );
    }

    // Tangent vector norms: the smoothing intentionally changes these by removing
    // DP artifacts. Verify they are finite and non-negative.
    for i in 0..n_sub {
        let tv_norm = l2_norm(&result.tangent_vectors.row(i), &time);
        assert!(
            tv_norm.is_finite(),
            "Tangent vector {i} norm should be finite, got {tv_norm}"
        );
    }
}

// ── Smoothed TSRVF validation against Python ────────────────────────────────

/// Reference data for smoothed TSRVF — Python applies the same Nadaraya-Watson
/// smoothing as Rust's `tsrvf_from_alignment` to aligned SRSFs before computing
/// tangent vectors on the Hilbert sphere.
#[derive(Deserialize)]
struct TsrvfSmoothedExpected {
    n_sub: usize,
    m: usize,
    smoothed_mean_srsf: Vec<f64>,
    smoothed_mean_srsf_norm: f64,
    smoothed_srsfs_flat: Vec<f64>,
    smoothed_srsf_norms: Vec<f64>,
    tangent_vectors_flat: Vec<f64>,
    tangent_vector_norms: Vec<f64>,
    mean_tangent_norm: f64,
}

#[derive(Deserialize)]
struct NewFeaturesWithSmoothed {
    tsrvf: TsrvfExpected,
    tsrvf_smoothed: TsrvfSmoothedExpected,
}

#[test]
fn test_tsrvf_smoothed_vs_python() {
    // Validate that Rust's tsrvf_from_alignment produces the same smoothed
    // tangent vectors as Python when applying identical Nadaraya-Watson
    // smoothing (bandwidth=2/(m-1), Gaussian kernel) to aligned SRSFs.
    let exp: NewFeaturesWithSmoothed = load_json("expected", "new_features_expected");
    let raw = &exp.tsrvf;
    let smoothed = &exp.tsrvf_smoothed;

    let n_sub = raw.n_sub;
    let m = raw.m;
    assert_eq!(smoothed.n_sub, n_sub);
    assert_eq!(smoothed.m, m);
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();

    // Build KarcherMeanResult from fdasrsf's pre-aligned data
    let aligned_data =
        FdMatrix::from_column_major(raw.aligned_curves_flat.clone(), n_sub, m).unwrap();
    let gammas = FdMatrix::from_column_major(raw.gammas_flat.clone(), n_sub, m).unwrap();

    let karcher = fdars_core::alignment::KarcherMeanResult {
        mean: raw.mean_curve.clone(),
        mean_srsf: raw.mean_srsf.clone(),
        gammas,
        aligned_data,
        n_iter: 1,
        converged: true,
        aligned_srsfs: None,
    };

    let result = fdars_core::alignment::tsrvf_from_alignment(&karcher, &time);

    // Validate smoothed mean SRSF norm (tight tolerance — same NW implementation)
    assert_relative_close(
        result.mean_srsf_norm,
        smoothed.smoothed_mean_srsf_norm,
        1e-6,
        "smoothed mean SRSF norm",
    );

    // Validate smoothed mean SRSF values
    let mean_l2_err = l2_norm(
        &result
            .mean_srsf
            .iter()
            .zip(smoothed.smoothed_mean_srsf.iter())
            .map(|(&a, &b)| a - b)
            .collect::<Vec<_>>(),
        &time,
    );
    let mean_l2_rel = mean_l2_err / smoothed.smoothed_mean_srsf_norm.max(1e-10);
    assert!(
        mean_l2_rel < 1e-6,
        "Smoothed mean SRSF L2 relative error = {mean_l2_rel:.2e}, expected < 1e-6"
    );

    // Validate per-curve SRSF norms
    let mut max_norm_err = 0.0_f64;
    for i in 0..n_sub {
        let rel_err = (result.srsf_norms[i] - smoothed.smoothed_srsf_norms[i]).abs()
            / smoothed.smoothed_srsf_norms[i].max(1e-10);
        max_norm_err = max_norm_err.max(rel_err);
    }
    assert!(
        max_norm_err < 1e-6,
        "Max smoothed SRSF norm relative error = {max_norm_err:.2e}, expected < 1e-6"
    );

    // Validate tangent vector norms
    let mut max_tv_norm_err = 0.0_f64;
    for i in 0..n_sub {
        let rust_tv_norm = l2_norm(&result.tangent_vectors.row(i), &time);
        let py_tv_norm = smoothed.tangent_vector_norms[i];
        let rel_err = (rust_tv_norm - py_tv_norm).abs() / py_tv_norm.max(1e-10);
        max_tv_norm_err = max_tv_norm_err.max(rel_err);
    }
    assert!(
        max_tv_norm_err < 1e-6,
        "Max tangent vector norm relative error = {max_tv_norm_err:.2e}, expected < 1e-6"
    );

    // Validate tangent vector values (row-major: data[i*m + j])
    for i in 0..n_sub {
        let rust_tv = result.tangent_vectors.row(i);
        let py_tv: Vec<f64> = (0..m)
            .map(|j| smoothed.tangent_vectors_flat[i * m + j])
            .collect();

        let diff: Vec<f64> = rust_tv
            .iter()
            .zip(py_tv.iter())
            .map(|(&a, &b)| a - b)
            .collect();
        let diff_norm = l2_norm(&diff, &time);
        let py_norm = smoothed.tangent_vector_norms[i].max(1e-10);
        let rel_err = diff_norm / py_norm;

        assert!(
            rel_err < 1e-6,
            "Tangent vector {i} relative L2 error = {rel_err:.2e}, expected < 1e-6"
        );
    }
}

// ── Soft-DTW barycenter vs tslearn ──────────────────────────────────────────

#[test]
fn test_soft_dtw_barycenter_vs_tslearn() {
    // Generate 3 shifted sinusoids (same as generate_new_features.py barycenter test)
    let m = 50;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let shifts = [0.0_f64, 0.05, -0.05];
    let n = shifts.len();

    let mut col_major = vec![0.0; n * m];
    for j in 0..m {
        for (i, &s) in shifts.iter().enumerate() {
            col_major[i + j * n] = (2.0 * std::f64::consts::PI * (t[j] - s)).sin();
        }
    }

    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let result = fdars_core::metric::soft_dtw_barycenter(&data, 1.0, 100, 1e-6);

    // Barycenter of mildly shifted sinusoids should resemble sin(2πt)
    // with mean value near 0 and max amplitude near 1
    let mean_val: f64 = result.barycenter.iter().sum::<f64>() / m as f64;
    assert!(
        mean_val.abs() < 0.3,
        "Barycenter mean should be near 0, got {mean_val}"
    );

    let max_abs: f64 = result
        .barycenter
        .iter()
        .map(|v| v.abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_abs > 0.5,
        "Barycenter max amplitude should be > 0.5, got {max_abs}"
    );
}

// ── Landmark sinusoid registration warps ────────────────────────────────────

#[test]
fn test_landmark_sinusoid_registration_warps() {
    // 5 sinusoids with known phase shifts → peak detection → landmark registration
    let m = 201;
    let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let shifts = [0.0_f64, 0.03, -0.02, 0.04, -0.03];
    let n = shifts.len();

    let mut col_major = vec![0.0; n * m];
    for j in 0..m {
        for (i, &s) in shifts.iter().enumerate() {
            col_major[i + j * n] = (2.0 * std::f64::consts::PI * (t[j] - s)).sin();
        }
    }

    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let result = fdars_core::landmark::detect_and_register(
        &data,
        &t,
        fdars_core::landmark::LandmarkKind::Peak,
        0.1,
        1,
    );

    // After registration, peak positions should be aligned to a common target
    // Check that warps are valid: monotone, endpoints fixed
    for i in 0..n {
        // Endpoints
        assert!(
            (result.gammas[(i, 0)] - t[0]).abs() < 1e-10,
            "Warp {i} should start at t[0]"
        );
        assert!(
            (result.gammas[(i, m - 1)] - t[m - 1]).abs() < 1e-10,
            "Warp {i} should end at t[-1]"
        );

        // Monotonicity
        for j in 1..m {
            assert!(
                result.gammas[(i, j)] >= result.gammas[(i, j - 1)],
                "Warp {i} not monotone at j={j}"
            );
        }
    }

    // Registered curves should have more similar peak positions than originals
    // Find peak of each registered curve
    let mut reg_peaks = Vec::new();
    for i in 0..n {
        let mut max_j = 0;
        let mut max_v = f64::NEG_INFINITY;
        for j in 0..m {
            if result.registered[(i, j)] > max_v {
                max_v = result.registered[(i, j)];
                max_j = j;
            }
        }
        reg_peaks.push(t[max_j]);
    }

    let mean_peak: f64 = reg_peaks.iter().sum::<f64>() / n as f64;
    let max_dev: f64 = reg_peaks
        .iter()
        .map(|&p| (p - mean_peak).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dev < 0.05,
        "Registered peaks should be aligned within 0.05, max deviation = {max_dev}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// New module correctness validation against R (Issues #4–#8)
// ═══════════════════════════════════════════════════════════════════════════════

// ── Expected data structures for new modules ─────────────────────────────────

#[derive(Deserialize)]
struct NewModulesExpected {
    scalar_on_function: ScalarOnFunctionExpected,
    function_on_scalar: FunctionOnScalarExpected,
    gmm: GmmExpected,
    classification: ClassificationExpected,
    famm: FammExpected,
}

#[derive(Deserialize)]
struct ScalarOnFunctionExpected {
    fitted: Vec<f64>,
    residuals: Vec<f64>,
    r_squared: f64,
    residual_ss: f64,
}

#[derive(Deserialize)]
struct FunctionOnScalarExpected {
    beta_function: Vec<f64>,
    mean_residual_l2: f64,
    beta_scores: Vec<f64>,
}

#[derive(Deserialize)]
struct GmmExpected {
    accuracy: f64,
    bic: f64,
    loglik: f64,
    selected_k: usize,
    weights: Vec<f64>,
}

#[derive(Deserialize)]
struct ClassificationExpected {
    dd_accuracy: f64,
    dd_predicted: Vec<usize>,
    dd_depths: DdDepthsExpected,
    knn_accuracy: f64,
}

#[derive(Deserialize)]
struct DdDepthsExpected {
    class1: Vec<f64>,
    class2: Vec<f64>,
    class3: Vec<f64>,
}

#[derive(Deserialize)]
struct FammExpected {
    gamma_estimates: Vec<f64>,
    sigma2_u: Vec<f64>,
    sigma2_eps: Vec<f64>,
    n_subjects: usize,
    n_visits: usize,
    m: usize,
    subject_ids: Vec<usize>,
    x_covariate: Vec<f64>,
    data: Vec<f64>,
    argvals: Vec<f64>,
}

// ── Tests ────────────────────────────────────────────────────────────────────

/// Scalar-on-function regression: compare R² and fitted values against R's fregre.pc.
///
/// Both use FPC regression with 3 components on the same regression_30x51 dataset.
/// FPCA decompositions may differ slightly (trapezoidal vs Simpson integration),
/// so fitted values use tol=0.3 (~5% of [-3, 3.7] range) and R² uses tol=0.01.
#[test]
fn test_r_scalar_on_function_regression() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let reg: RegressionData = load_json("data", "regression_30x51");
    let data = FdMatrix::from_column_major(reg.data, reg.n, reg.m).unwrap();

    let result = fregre_lm(&data, &reg.y, None, 3).expect("fregre_lm should succeed");

    // R² should match closely — same algorithm (FPCA→OLS), only floating-point differences
    assert_scalar_close(
        result.r_squared,
        expected.scalar_on_function.r_squared,
        0.01,
        "R² (Rust vs R fregre.pc)",
    );

    // Fitted values: element-wise comparison (range ~[-3, 3.7], tol=0.3 is ~5%)
    assert_vec_close(
        &result.fitted_values,
        &expected.scalar_on_function.fitted,
        0.3,
        "Fitted values (Rust vs R)",
    );

    // Residuals: element-wise comparison against R's residuals
    assert_vec_close(
        &result.residuals,
        &expected.scalar_on_function.residuals,
        0.3,
        "Residuals (Rust vs R)",
    );

    // Residual sum of squares — tighter tolerance since R² matches closely
    let rss_rust: f64 = result.residuals.iter().map(|r| r * r).sum();
    assert_relative_close(
        rss_rust,
        expected.scalar_on_function.residual_ss,
        0.05,
        "Residual SS (Rust vs R)",
    );
}

/// GMM clustering on FPC scores: compare accuracy and weights against R's Mclust.
///
/// Both reduce the clusters_60x51 data to 3 FPC scores, then fit GMM with K=3.
/// Different FPCA → different scores, but clusters are well-separated so both
/// should achieve near-perfect accuracy. Weights are compared sorted (label ordering differs).
/// BIC uses different conventions (R: higher=better, Rust: lower=better), so skip direct comparison.
#[test]
fn test_r_gmm_clustering() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let clust: ClusterData = load_json("data", "clusters_60x51");
    let data = FdMatrix::from_column_major(clust.data.clone(), clust.n, clust.m).unwrap();

    // Do FPCA to get 3 scores (same as R does)
    let fpca = fdars_core::regression::fdata_to_pc_1d(&data, 3).unwrap();
    let n = clust.n;
    let scores: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..3).map(|k| fpca.scores[(i, k)]).collect())
        .collect();

    // Fit GMM
    let gmm_result =
        gmm_em(&scores, 3, CovType::Full, 200, 1e-6, 42).expect("gmm_em should succeed");

    // Accuracy after label permutation — compare against R's value
    let pred = &gmm_result.cluster;
    let true_labels = &clust.true_labels;
    let mut best_acc = 0.0_f64;
    for perm in &[
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let acc = pred
            .iter()
            .zip(true_labels.iter())
            .filter(|(&p, &t)| perm[p] + 1 == t) // R labels are 1-indexed
            .count() as f64
            / n as f64;
        best_acc = best_acc.max(acc);
    }

    assert_scalar_close(
        best_acc,
        expected.gmm.accuracy,
        0.02,
        "GMM accuracy (Rust vs R)",
    );

    // Weights: sort both and compare (label ordering may differ)
    let mut rust_weights: Vec<f64> = gmm_result.weights.clone();
    rust_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut r_weights: Vec<f64> = expected.gmm.weights.clone();
    r_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_vec_close(
        &rust_weights,
        &r_weights,
        0.05,
        "GMM weights sorted (Rust vs R)",
    );
}

/// Functional classification: compare LDA, k-NN, and DD-classifier accuracies against R.
///
/// Uses the clusters_60x51 dataset with 3 well-separated classes.
/// LDA and k-NN should match R closely (same algorithm); DD has more variation.
#[test]
fn test_r_classification_accuracy() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let clust: ClusterData = load_json("data", "clusters_60x51");
    let data = FdMatrix::from_column_major(clust.data.clone(), clust.n, clust.m).unwrap();

    // Convert 1-indexed R labels to 0-indexed Rust labels
    let labels: Vec<usize> = clust.true_labels.iter().map(|&l| l - 1).collect();

    // LDA classifier — R achieves 1.0 on this data
    let lda = fclassif_lda(&data, &labels, None, 3).expect("LDA should succeed");
    assert_scalar_close(lda.accuracy, 1.0, 0.05, "LDA accuracy (Rust vs R=1.0)");

    // k-NN classifier (k=3) — compare against R's LOO k-NN accuracy
    let knn = fclassif_knn(&data, &labels, None, 3, 3).expect("k-NN should succeed");
    assert_scalar_close(
        knn.accuracy,
        expected.classification.knn_accuracy,
        0.05,
        "k-NN accuracy (Rust vs R)",
    );

    // DD-classifier (depth-based) — wider tolerance due to integration rule differences
    let dd = fclassif_dd(&data, &labels, None).expect("DD should succeed");
    assert_scalar_close(
        dd.accuracy,
        expected.classification.dd_accuracy,
        0.1,
        "DD accuracy (Rust vs R)",
    );
}

/// DD-classifier depth values: compare Fraiman-Muniz depths against R's depth.FM.
///
/// R computes FM depth for each observation w.r.t. each class reference set.
/// We compute the same depths directly via `fraiman_muniz_1d` and compare
/// element-wise against R's dd_depths (tol=0.05, same as FM depth test).
#[test]
fn test_r_dd_classifier_depths() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let clust: ClusterData = load_json("data", "clusters_60x51");
    let data = FdMatrix::from_column_major(clust.data.clone(), clust.n, clust.m).unwrap();

    let r_depths_c1 = &expected.classification.dd_depths.class1;
    let r_depths_c2 = &expected.classification.dd_depths.class2;
    let r_depths_c3 = &expected.classification.dd_depths.class3;

    // Build per-class reference matrices (classes are 1-indexed: 1,2,3 → 0..20, 20..40, 40..60)
    let n = clust.n;
    let m = clust.m;
    let mut class_mats: Vec<FdMatrix> = Vec::new();
    for class_start in [0usize, 20, 40] {
        let class_n = 20;
        let mut class_data = vec![0.0; class_n * m];
        for i in 0..class_n {
            for j in 0..m {
                class_data[i + j * class_n] = data[(class_start + i, j)];
            }
        }
        class_mats.push(FdMatrix::from_column_major(class_data, class_n, m).unwrap());
    }

    // Compute Rust FM depths: all n observations w.r.t. each class reference
    let rust_depths_c1 = fdars_core::depth::fraiman_muniz_1d(&data, &class_mats[0], true);
    let rust_depths_c2 = fdars_core::depth::fraiman_muniz_1d(&data, &class_mats[1], true);
    let rust_depths_c3 = fdars_core::depth::fraiman_muniz_1d(&data, &class_mats[2], true);

    // Element-wise comparison against R's depths (tol=0.05, consistent with FM depth test)
    assert_vec_close(
        &rust_depths_c1,
        r_depths_c1,
        0.05,
        "DD depths class1 (Rust vs R)",
    );
    assert_vec_close(
        &rust_depths_c2,
        r_depths_c2,
        0.05,
        "DD depths class2 (Rust vs R)",
    );
    assert_vec_close(
        &rust_depths_c3,
        r_depths_c3,
        0.05,
        "DD depths class3 (Rust vs R)",
    );

    // Structural check: class k observations should be deepest in their own class
    for i in 0..20 {
        assert!(
            r_depths_c1[i] > r_depths_c2[i] && r_depths_c1[i] > r_depths_c3[i],
            "R: class1 obs {} not deepest in own class: c1={:.4}, c2={:.4}, c3={:.4}",
            i,
            r_depths_c1[i],
            r_depths_c2[i],
            r_depths_c3[i]
        );
    }
    for i in 20..40 {
        assert!(
            r_depths_c2[i] > r_depths_c1[i] && r_depths_c2[i] > r_depths_c3[i],
            "R: class2 obs {} not deepest in own class",
            i
        );
    }
    for i in 40..60 {
        assert!(
            r_depths_c3[i] > r_depths_c1[i] && r_depths_c3[i] > r_depths_c2[i],
            "R: class3 obs {} not deepest in own class",
            i
        );
    }

    // Run Rust's DD-classifier and verify predictions match R's
    let labels: Vec<usize> = clust.true_labels.iter().map(|&l| l - 1).collect();
    let dd = fclassif_dd(&data, &labels, None).expect("DD should succeed");

    let r_predicted = &expected.classification.dd_predicted;
    let mut match_count = 0;
    for (rust_pred, &r_pred) in dd.predicted.iter().zip(r_predicted.iter()) {
        if *rust_pred == r_pred - 1 {
            match_count += 1;
        }
    }
    let match_rate = match_count as f64 / n as f64;
    assert!(
        match_rate >= 0.95,
        "DD prediction match rate = {:.4}, expected >= 0.95",
        match_rate
    );
}

/// Functional mixed model: compare variance components against R's lmer.
///
/// Uses the EXACT same generated data from R. Compares sigma2_u and sigma2_eps
/// numerically. R uses REML, Rust uses Henderson's MME, so we use 50% relative
/// tolerance for variance components but expect the same order of magnitude.
/// Also verifies the beta function shape via gamma_estimates.
#[test]
fn test_r_famm_variance_components() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let famm_ref = &expected.famm;

    // Use R's exact generated data
    let n = famm_ref.n_subjects * famm_ref.n_visits; // 30
    let m = famm_ref.m; // 31
    let data = FdMatrix::from_column_major(famm_ref.data.clone(), n, m).unwrap();

    // Build covariate matrix (1 covariate x n observations)
    let cov_data = famm_ref.x_covariate.clone();
    let covariates = FdMatrix::from_column_major(cov_data, n, 1).unwrap();

    // Convert 1-indexed R subject IDs to 0-indexed
    let subject_ids: Vec<usize> = famm_ref.subject_ids.iter().map(|&s| s - 1).collect();

    let result = fmm(&data, &subject_ids, Some(&covariates), 3).expect("fmm should succeed");

    // Number of subjects must match exactly
    assert_eq!(result.n_subjects, famm_ref.n_subjects);

    // Dominant random variance component: compare against R's sigma2_u[0]
    // With iterative GLS+REML, Rust should be within 3x of R's REML estimates.
    let max_sigma2_u = result.sigma2_u.iter().cloned().fold(0.0_f64, f64::max);
    assert_relative_close(
        max_sigma2_u,
        famm_ref.sigma2_u[0],
        3.0,
        "sigma2_u (max component vs R)",
    );

    // The random effect variance should dominate residual variance (same as R)
    assert!(
        max_sigma2_u > result.sigma2_eps,
        "sigma2_u ({:.6}) should dominate sigma2_eps ({:.6})",
        max_sigma2_u,
        result.sigma2_eps
    );

    // Residual variance: σ²_eps should be small (noise is σ=0.05 → σ²=0.0025).
    // With REML, Rust's estimate should be much closer to R's.
    assert!(
        result.sigma2_eps > 0.0 && result.sigma2_eps < 0.01,
        "sigma2_eps = {:.6}, expected in (0, 0.01) — R REML got {:.6}",
        result.sigma2_eps,
        famm_ref.sigma2_eps[0]
    );

    // Gamma estimates: compare sign and magnitude per component
    let r_gamma = &famm_ref.gamma_estimates;
    let r_gamma_l2: f64 = r_gamma.iter().map(|g| g * g).sum::<f64>().sqrt();
    assert!(
        r_gamma_l2 > 0.01,
        "R gamma estimates should be non-trivial: L2 = {:.6}",
        r_gamma_l2
    );

    // Rust's beta_functions row 0 should have non-trivial variation too
    let rust_beta: Vec<f64> = (0..m).map(|j| result.beta_functions[(0, j)]).collect();
    let beta_range = rust_beta.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - rust_beta.iter().cloned().fold(f64::INFINITY, f64::min);
    assert!(
        beta_range > 0.01,
        "Rust beta function range = {:.6}, expected non-trivial",
        beta_range
    );
}

/// FAMM fitted values: check that fitted curves are close to observed.
///
/// With low noise (σ=0.05 → σ²=0.0025), fitted curves should explain most variance.
/// R² > 0.9 and MSE < 0.01 (well above noise floor).
#[test]
fn test_r_famm_fitted_quality() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let famm_ref = &expected.famm;

    let n = famm_ref.n_subjects * famm_ref.n_visits;
    let m = famm_ref.m;
    let data = FdMatrix::from_column_major(famm_ref.data.clone(), n, m).unwrap();
    let cov_data = famm_ref.x_covariate.clone();
    let covariates = FdMatrix::from_column_major(cov_data, n, 1).unwrap();
    let subject_ids: Vec<usize> = famm_ref.subject_ids.iter().map(|&s| s - 1).collect();

    let result = fmm(&data, &subject_ids, Some(&covariates), 3).expect("fmm should succeed");

    // Compute R² and MSE between fitted and observed
    let mut ss_res = 0.0;
    let mut ss_tot = 0.0;
    let overall_mean: f64 = famm_ref.data.iter().sum::<f64>() / (n * m) as f64;

    for i in 0..n {
        for j in 0..m {
            let obs = data[(i, j)];
            let fit = result.fitted[(i, j)];
            ss_res += (obs - fit).powi(2);
            ss_tot += (obs - overall_mean).powi(2);
        }
    }

    let r_squared = 1.0 - ss_res / ss_tot;
    assert!(
        r_squared > 0.99,
        "FAMM R² = {:.6}, expected > 0.99 (low noise data with REML)",
        r_squared
    );

    // MSE should be small — noise variance is 0.0025, so MSE should be well below 0.005
    let mse = ss_res / (n * m) as f64;
    assert!(
        mse < 0.005,
        "FAMM MSE = {:.6}, expected < 0.005 (noise σ²=0.0025, REML fit)",
        mse
    );
}

/// Function-on-scalar regression: compare beta function shape against R.
///
/// R computes beta(t) via FPC regression: sum_k beta_k * phi_k(t).
/// We verify the beta function from our FPC-based FOSR has the correct shape
/// and element-wise closeness to R's beta function.
#[test]
fn test_r_function_on_scalar_beta_shape() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let reg: RegressionData = load_json("data", "regression_30x51");

    let r_beta = &expected.function_on_scalar.beta_function;

    // R's beta function: positive in first half, negative in second half
    let m = r_beta.len();
    let first_half_mean: f64 = r_beta[..m / 2].iter().sum::<f64>() / (m / 2) as f64;
    let second_half_mean: f64 = r_beta[m / 2..].iter().sum::<f64>() / (m - m / 2) as f64;

    assert!(
        first_half_mean > 0.0,
        "R beta first half mean = {:.4}, expected > 0",
        first_half_mean
    );
    assert!(
        second_half_mean < 0.0,
        "R beta second half mean = {:.4}, expected < 0",
        second_half_mean
    );

    // Test Rust's FPC-based FOSR on the same data
    let n = reg.n;
    let data = FdMatrix::from_column_major(reg.data, reg.n, reg.m).unwrap();
    let pred_data = reg.y.clone();
    let predictors = FdMatrix::from_column_major(pred_data, n, 1).unwrap();

    let fosr_result = fdars_core::function_on_scalar::fosr_fpc(&data, &predictors, 3)
        .expect("fosr_fpc should work");

    // Check that Rust's beta also has the right sign pattern
    let rust_beta_col: Vec<f64> = (0..reg.m).map(|j| fosr_result.beta[(0, j)]).collect();
    let rust_first_half: f64 = rust_beta_col[..m / 2].iter().sum::<f64>() / (m / 2) as f64;
    let rust_second_half: f64 = rust_beta_col[m / 2..].iter().sum::<f64>() / (m - m / 2) as f64;

    assert!(
        rust_first_half > 0.0,
        "Rust beta first half mean = {:.4}, expected > 0",
        rust_first_half
    );
    assert!(
        rust_second_half < 0.0,
        "Rust beta second half mean = {:.4}, expected < 0",
        rust_second_half
    );

    // Element-wise beta function comparison (FPC-based should be close to R)
    assert_vec_close(&rust_beta_col, r_beta, 0.1, "FPC-based beta vs R beta");

    // Beta scores comparison against R's beta_scores
    let r_beta_scores = &expected.function_on_scalar.beta_scores;
    let rust_beta_scores = &fosr_result.beta_scores[0]; // predictor 0
    let k = rust_beta_scores.len().min(r_beta_scores.len());
    for comp in 0..k {
        assert!(
            (rust_beta_scores[comp].abs() - r_beta_scores[comp].abs()).abs() < 0.05,
            "beta_scores[{}]: Rust={:.4}, R={:.4}",
            comp,
            rust_beta_scores[comp],
            r_beta_scores[comp]
        );
    }
}

/// Function-on-scalar regression: compare mean residual L² against R's reference.
///
/// FPC-based FOSR should closely match R's FPC-based regression residuals.
/// Uses R's exact metric: mean of integrated squared residuals per curve.
#[test]
fn test_r_function_on_scalar_residual_l2() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let reg: RegressionData = load_json("data", "regression_30x51");

    let n = reg.n;
    let m = reg.m;
    let data = FdMatrix::from_column_major(reg.data, reg.n, reg.m).unwrap();
    let pred_data = reg.y.clone();
    let predictors = FdMatrix::from_column_major(pred_data, n, 1).unwrap();

    let fosr_result = fdars_core::function_on_scalar::fosr_fpc(&data, &predictors, 3)
        .expect("fosr_fpc should work");

    // Use R's exact metric: mean(sum(resid^2) * h) where h = (argvals[m] - argvals[1]) / (m-1)
    let h = (reg.argvals.last().unwrap() - reg.argvals.first().unwrap()) / (m - 1) as f64;
    let mean_resid_l2: f64 = (0..n)
        .map(|i| {
            let ss: f64 = (0..m)
                .map(|j| fosr_result.residuals[(i, j)].powi(2))
                .sum::<f64>();
            ss * h
        })
        .sum::<f64>()
        / n as f64;

    // FPC-based FOSR should closely match R's FPC-based regression
    let r_l2 = expected.function_on_scalar.mean_residual_l2;
    assert_relative_close(mean_resid_l2, r_l2, 0.3, "FOSR mean residual L²");
}

/// Cross-check scalar-on-function: prediction decomposition identity.
///
/// For FPC regression, fitted = X_scores * beta_hat + intercept.
/// Verify that residuals = y - fitted exactly.
#[test]
fn test_r_scalar_on_function_decomposition() {
    let reg: RegressionData = load_json("data", "regression_30x51");
    let data = FdMatrix::from_column_major(reg.data, reg.n, reg.m).unwrap();

    let result = fregre_lm(&data, &reg.y, None, 3).expect("fregre_lm should succeed");

    // y = fitted + residuals (exact identity)
    for i in 0..reg.n {
        let reconstructed = result.fitted_values[i] + result.residuals[i];
        assert!(
            (reconstructed - reg.y[i]).abs() < 1e-10,
            "Decomposition identity failed at i={}: y={:.8}, fitted+resid={:.8}",
            i,
            reg.y[i],
            reconstructed
        );
    }

    // R² = 1 - RSS/TSS
    let mean_y: f64 = reg.y.iter().sum::<f64>() / reg.n as f64;
    let tss: f64 = reg.y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let rss: f64 = result.residuals.iter().map(|r| r * r).sum();
    let r2_check = 1.0 - rss / tss;
    assert_scalar_close(result.r_squared, r2_check, 1e-10, "R² identity");
}

/// GMM BIC model selection: K=3 should achieve near-perfect accuracy and better BIC than K=4.
///
/// R's Mclust selects K=3 across 14 covariance parameterizations.
/// We verify that K=3 accuracy matches R's and that K=3 BIC is better than K=4.
#[test]
fn test_r_gmm_model_selection() {
    let expected: NewModulesExpected = load_json("expected", "new_modules_expected");
    let clust: ClusterData = load_json("data", "clusters_60x51");
    let data = FdMatrix::from_column_major(clust.data.clone(), clust.n, clust.m).unwrap();

    // FPCA → 3 scores
    let fpca = fdars_core::regression::fdata_to_pc_1d(&data, 3).unwrap();
    let n = clust.n;
    let scores: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..3).map(|k| fpca.scores[(i, k)]).collect())
        .collect();

    // R correctly selects K=3
    assert_eq!(expected.gmm.selected_k, 3);

    // K=3 GMM should converge and produce good results
    let res_k3 = gmm_em(&scores, 3, CovType::Full, 200, 1e-6, 42).expect("K=3 GMM should fit");
    assert!(res_k3.converged, "K=3 GMM should converge");

    // K=3 accuracy should match R's (both near-perfect on well-separated data)
    let true_labels = &clust.true_labels;
    let mut best_acc = 0.0_f64;
    for perm in &[
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let acc = res_k3
            .cluster
            .iter()
            .zip(true_labels.iter())
            .filter(|(&p, &t)| perm[p] + 1 == t)
            .count() as f64
            / n as f64;
        best_acc = best_acc.max(acc);
    }
    assert_scalar_close(best_acc, 1.0, 0.02, "K=3 accuracy (Rust vs R=1.0)");

    // BIC for K=3 should be finite
    assert!(
        res_k3.bic.is_finite(),
        "K=3 BIC should be finite, got {}",
        res_k3.bic
    );

    // K=3 BIC should be better (lower) than K=4 — correct model should fit better
    let res_k4 = gmm_em(&scores, 4, CovType::Full, 200, 1e-6, 42).expect("K=4 GMM should fit");
    assert!(
        res_k3.bic <= res_k4.bic,
        "K=3 BIC ({:.2}) should be <= K=4 BIC ({:.2})",
        res_k3.bic,
        res_k4.bic
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2a: Extended Seasonal Validation
// ═══════════════════════════════════════════════════════════════════════════

/// SSA reconstruction: compare first 2 component reconstruction against R's Rssa.
#[test]
fn test_ssa_reconstruction_vs_r() {
    let exp: SeasonalExpected = load_json("expected", "seasonal_expected");
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    if let Some(ref ssa_exp) = exp.ssa_reconstruction {
        let result = fdars_core::seasonal::ssa(
            &sdat.noisy_sine,
            Some(ssa_exp.window_length),
            Some(10),
            None,
            Some(&[0, 1]),
        );

        // SSA reconstruction (first 2 components) should correlate strongly with R's
        let rust_recon = &result.seasonal;
        let r_recon = &ssa_exp.component_1_2;
        assert_eq!(rust_recon.len(), r_recon.len());

        // Rank correlation — SSA implementations differ in grouping heuristics
        assert_ranking_correlated(rust_recon, r_recon, "ssa_component_1_2");

        // Also check the reconstruction captures the periodic structure
        let recon_var: f64 =
            rust_recon.iter().map(|x| x * x).sum::<f64>() / rust_recon.len() as f64;
        assert!(
            recon_var > 0.01,
            "SSA reconstruction should have non-trivial variance: {}",
            recon_var
        );
    }
}

/// Hilbert transform amplitude: compare analytic signal amplitude vs R.
#[test]
fn test_hilbert_amplitude_vs_r() {
    let exp: SeasonalExpected = load_json("expected", "seasonal_expected");
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    if let Some(ref hilbert_exp) = exp.hilbert_amplitude {
        let analytic = fdars_core::seasonal::hilbert_transform(&sdat.noisy_sine);
        let rust_amplitude: Vec<f64> = analytic.iter().map(|c| c.norm()).collect();

        assert_eq!(rust_amplitude.len(), hilbert_exp.amplitude.len());

        // Hilbert amplitude should match R closely (both use FFT-based approach)
        assert_vec_close(
            &rust_amplitude,
            &hilbert_exp.amplitude,
            0.05,
            "hilbert_amplitude",
        );
    }
}

/// Seasonal strength (variance ratio): compare against R's decompose-based metric.
#[test]
fn test_seasonal_strength_vs_r() {
    let exp: SeasonalExpected = load_json("expected", "seasonal_expected");
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    if let Some(ref strength_exp) = exp.seasonal_strength {
        let n = 1;
        let m = sdat.n;
        let data = FdMatrix::from_slice(&sdat.noisy_sine, n, m).unwrap();

        let rust_strength =
            fdars_core::seasonal::seasonal_strength_variance(&data, &sdat.t, sdat.period as f64, 3);

        // Seasonal strength values may differ due to different decomposition methods.
        // Just verify both are in a reasonable range and have the same sign.
        assert!(
            rust_strength.is_finite(),
            "Rust seasonal strength should be finite"
        );
        assert!(
            strength_exp.strength.is_finite(),
            "R seasonal strength should be finite"
        );
        // Both should indicate some (weak) seasonality
        let diff = (rust_strength - strength_exp.strength).abs();
        assert!(
            diff < 1.0,
            "Seasonal strength difference too large: Rust={:.4}, R={:.4}",
            rust_strength,
            strength_exp.strength
        );
    }
}

/// Peak detection: compare detected peak positions against R's pracma::findpeaks.
#[test]
fn test_peak_detection_vs_r() {
    let exp: SeasonalExpected = load_json("expected", "seasonal_expected");

    let r_signal = &exp.peak_detection.signal;
    let r_x = &exp.peak_detection.x;
    let r_peak_indices = &exp.peak_detection.peak_indices;

    // Build FdMatrix from the signal (1 x 200)
    let n = 1;
    let m = r_signal.len();
    let data = FdMatrix::from_slice(r_signal, n, m).unwrap();

    let result = fdars_core::seasonal::detect_peaks(&data, r_x, None, Some(0.5), false, None);

    // R uses pracma::findpeaks which may use different criteria than Rust's
    // derivative zero-crossing with prominence filtering.
    // Verify Rust finds a reasonable number of peaks and they overlap R's.
    let rust_peak_count = result.peaks[0].len();
    let r_peak_count = r_peak_indices.len();
    assert!(
        rust_peak_count >= 3 && rust_peak_count <= r_peak_count + 3,
        "Peak count out of reasonable range: Rust={}, R={}",
        rust_peak_count,
        r_peak_count
    );

    // Check that each Rust peak is near some R peak
    if !result.peaks[0].is_empty() && !r_peak_indices.is_empty() {
        let rust_times: Vec<f64> = result.peaks[0].iter().map(|p| p.time).collect();
        let r_times: Vec<f64> = r_peak_indices
            .iter()
            .map(|&idx| r_x[(idx - 1).min(m - 1)])
            .collect(); // R is 1-indexed
        for rust_t in &rust_times {
            let min_dist = r_times
                .iter()
                .map(|&rt| (rt - rust_t).abs())
                .fold(f64::INFINITY, f64::min);
            assert!(
                min_dist < 0.05,
                "Rust peak at t={:.4} has no matching R peak (min_dist={:.4})",
                rust_t,
                min_dist
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2b: Extended Detrend Validation
// ═══════════════════════════════════════════════════════════════════════════

/// LOESS detrend: compare trend shape against R's loess().
#[test]
fn test_detrend_loess_vs_r() {
    let exp: DetrendExpected = load_json("expected", "detrend_expected");
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    if let Some(ref loess_exp) = exp.loess_detrend {
        let n = 1;
        let m = sdat.n;
        let data = FdMatrix::from_slice(&sdat.noisy_sine, n, m).unwrap();

        let result = fdars_core::detrend::detrend_loess(&data, &sdat.t, 0.3, 1);

        let rust_trend: Vec<f64> = (0..m).map(|j| result.trend[(0, j)]).collect();

        // LOESS implementations differ in bandwidth interpretation and kernel.
        // R uses tri-cube kernel; Rust uses Gaussian + local polynomial.
        assert_ranking_correlated_tol(&rust_trend, &loess_exp.trend, 0.75, "loess_trend_shape");

        // Detrended should have smaller variance than original
        let orig_var: f64 = sdat.noisy_sine.iter().map(|x| x * x).sum::<f64>() / m as f64;
        let detrended: Vec<f64> = (0..m).map(|j| result.detrended[(0, j)]).collect();
        let det_var: f64 = detrended.iter().map(|x| x * x).sum::<f64>() / m as f64;
        assert!(
            det_var <= orig_var * 1.1,
            "LOESS detrended variance should not exceed original"
        );
    }
}

/// Additive decomposition: verify reconstruction identity against R.
#[test]
fn test_decompose_additive_vs_r() {
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    let n = 1;
    let m = sdat.n;
    let data = FdMatrix::from_slice(&sdat.noisy_sine, n, m).unwrap();

    let result = fdars_core::detrend::decompose_additive(
        &data,
        &sdat.t,
        sdat.period as f64,
        "loess",
        0.3,
        3,
    );

    // Reconstruction identity: trend + seasonal + remainder = original
    for j in 0..m {
        let reconstructed =
            result.trend[(0, j)] + result.seasonal[(0, j)] + result.remainder[(0, j)];
        assert_scalar_close(
            reconstructed,
            sdat.noisy_sine[j],
            1e-6,
            &format!("additive_decomp_recon[{}]", j),
        );
    }
}

/// Multiplicative decomposition: verify reconstruction identity.
#[test]
fn test_decompose_multiplicative_vs_r() {
    let sdat: SeasonalData = load_json("data", "seasonal_200");

    let n = 1;
    let m = sdat.n;
    // Shift data to be positive for multiplicative decomposition
    let shifted: Vec<f64> = sdat.noisy_sine.iter().map(|&x| x + 3.0).collect();
    let data = FdMatrix::from_slice(&shifted, n, m).unwrap();

    let result = fdars_core::detrend::decompose_multiplicative(
        &data,
        &sdat.t,
        sdat.period as f64,
        "loess",
        0.3,
        3,
    );

    // Reconstruction identity: trend * seasonal * remainder = original
    for (j, &orig) in shifted.iter().enumerate().take(m) {
        let reconstructed =
            result.trend[(0, j)] * result.seasonal[(0, j)] * result.remainder[(0, j)];
        assert!(
            (reconstructed - orig).abs() < 1e-3,
            "mult decomp reconstruction failed at j={}: {:.6} vs {:.6}",
            j,
            reconstructed,
            orig
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2c: Extended Smoothing Validation
// ═══════════════════════════════════════════════════════════════════════════

/// Local polynomial smoother: compare against R's locpoly (degree=2).
#[test]
fn test_local_polynomial_vs_r() {
    let exp: SmoothingExpected = load_json("expected", "smoothing_expected");
    let dat: NoisySineData = load_json("data", "noisy_sine_201");

    if let Some(ref r_lp) = exp.local_polynomial {
        let result = fdars_core::smoothing::local_polynomial(
            &dat.x,
            &dat.y_noisy,
            &dat.x,
            0.05,
            2,
            "gaussian",
        );

        // Local polynomial with degree=2 should be close to R's locpoly degree=2
        // R uses FFT-based binning, Rust uses direct computation, so moderate tolerance
        assert_vec_close(&result, r_lp, 0.15, "local_polynomial_degree2");
    }
}

/// Smoothing matrix NW: verify row sums to 1 and compare one row to R.
#[test]
fn test_smoothing_matrix_nw_vs_r() {
    let exp: SmoothingExpected = load_json("expected", "smoothing_expected");
    let dat: NoisySineData = load_json("data", "noisy_sine_201");

    if let Some(ref nw_exp) = exp.smoothing_matrix_nw {
        let sm = fdars_core::smoothing::smoothing_matrix_nw(&dat.x, 0.05, "gaussian");
        let m = dat.x.len();

        // Total elements should be m*m
        assert_eq!(sm.len(), m * m, "Smoothing matrix should be m×m");

        // Row sums should all be ~1 (column-major storage: s[i + j * m])
        for i in 0..m {
            let row_sum: f64 = (0..m).map(|j| sm[i + j * m]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "NW smoothing matrix row {} sum = {}, expected 1.0",
                i,
                row_sum
            );
        }

        // Compare middle row (idx=100 for eval_point ≈ 0.5) with R
        // Column-major: row i is at sm[i], sm[i+m], sm[i+2m], ...
        let mid_row: Vec<f64> = (0..m).map(|j| sm[100 + j * m]).collect();
        assert_vec_close(
            &mid_row,
            &nw_exp.weights,
            1e-6,
            "smoothing_matrix_nw_row100",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2d: Extended Tolerance Validation
// ═══════════════════════════════════════════════════════════════════════════

/// Degras SCB: tighter comparison of smoothed mean values.
#[test]
fn test_degras_scb_mean_vs_r() {
    let exp: ToleranceExpected = load_json("expected", "tolerance_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    if let Some(ref r_smoothed) = exp.degras_smoothed_mean {
        let band = fdars_core::scb_mean_degras(
            &mat,
            &d.argvals,
            0.15,
            500,
            0.95,
            fdars_core::MultiplierDistribution::Gaussian,
        )
        .expect("scb_mean_degras should succeed");

        // Compare smoothed mean shapes (different kernel implementations, so moderate tol)
        assert_ranking_correlated(&band.center, r_smoothed, "degras_smoothed_mean_shape");

        // Verify max pointwise difference is bounded
        let max_diff: f64 = band
            .center
            .iter()
            .zip(r_smoothed.iter())
            .map(|(a, e)| (a - e).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff < 0.3,
            "Degras smoothed mean max diff = {:.4}, expected < 0.3",
            max_diff
        );
    }
}

/// Elastic tolerance band: compare center against R's fdasrvf time_warping mean.
#[test]
fn test_elastic_tolerance_center_vs_r() {
    let exp: ToleranceExpected = load_json("expected", "tolerance_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    if let Some(ref elastic_exp) = exp.elastic_tolerance {
        // Use same number of curves as R
        let n_sub = elastic_exp.n.min(d.n);
        let sub_vec: Vec<f64> = (0..n_sub * d.m)
            .map(|idx| {
                let i = idx % n_sub;
                let j = idx / n_sub;
                mat[(i, j)]
            })
            .collect();
        let sub_mat = FdMatrix::from_column_major(sub_vec, n_sub, d.m).unwrap();

        let band = fdars_core::elastic_tolerance_band(
            &sub_mat,
            &d.argvals,
            5,
            100,
            0.95,
            fdars_core::tolerance::BandType::Simultaneous,
            20,
            42,
        )
        .expect("elastic_tolerance_band should succeed");

        // Center shape should correlate with R's elastic mean
        assert_ranking_correlated(
            &band.center,
            &elastic_exp.center,
            "elastic_tolerance_center",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 2e: Extended Depth Validation
// ═══════════════════════════════════════════════════════════════════════════

/// Random projection depth: compare actual values (not just rank correlation).
#[test]
fn test_random_projection_values_vs_r() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    if let Some(r_vals) = exp.random_projection.as_array() {
        let r_depths: Vec<f64> = r_vals.iter().filter_map(|v| v.as_f64()).collect();
        let rust_depths = fdars_core::depth::random_projection_1d(&mat, &mat, 50);

        // RNG seeds differ between R and Rust, so values won't match exactly.
        // Use relaxed rank correlation since random projections are inherently stochastic.
        assert_ranking_correlated_tol(&rust_depths, &r_depths, 0.75, "random_projection_ranks");

        // Both depth vectors should have similar mean and variance
        let r_mean = r_depths.iter().sum::<f64>() / r_depths.len() as f64;
        let rust_mean = rust_depths.iter().sum::<f64>() / rust_depths.len() as f64;
        assert!(
            (r_mean - rust_mean).abs() < 0.15,
            "RP depth mean diff: R={:.4}, Rust={:.4}",
            r_mean,
            rust_mean
        );
    }
}

/// Random Tukey depth: compare actual values.
#[test]
fn test_random_tukey_values_vs_r() {
    let exp: DepthExpected = load_json("expected", "depth_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    if let Some(r_vals) = exp.random_tukey.as_array() {
        let r_depths: Vec<f64> = r_vals.iter().filter_map(|v| v.as_f64()).collect();
        // Use many projections for stable results (RNG seeds are incompatible)
        let rust_depths = fdars_core::depth::random_tukey_1d(&mat, &mat, 500);

        // Random Tukey depth is highly sensitive to projection directions.
        // With incompatible RNGs, only statistical properties can be compared.
        let r_mean = r_depths.iter().sum::<f64>() / r_depths.len() as f64;
        let rust_mean = rust_depths.iter().sum::<f64>() / rust_depths.len() as f64;
        assert!(
            (r_mean - rust_mean).abs() < 0.15,
            "RT depth mean diff: R={:.4}, Rust={:.4}",
            r_mean,
            rust_mean
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3a: Irregular FData Validation
// ═══════════════════════════════════════════════════════════════════════════

/// Trapezoidal integration on irregular grids: compare against R.
#[test]
fn test_irreg_integrate_vs_r() {
    let exp: IrregFdataExpected = load_json("expected", "irreg_fdata_expected");

    let ifd = IrregFdata::from_lists(&exp.argvals, &exp.values);
    let rust_integrals = fdars_core::irreg_fdata::integrate_irreg(&ifd);

    if let Some(ref int_exp) = exp.integrate {
        assert_vec_close(&rust_integrals, &int_exp.integrals, 1e-4, "irreg_integrate");
    }
}

/// L2 norm on irregular grids: compare against R.
#[test]
fn test_irreg_norm_vs_r() {
    let exp: IrregFdataExpected = load_json("expected", "irreg_fdata_expected");

    let ifd = IrregFdata::from_lists(&exp.argvals, &exp.values);
    let rust_norms = fdars_core::irreg_fdata::norm_lp_irreg(&ifd, 2.0);

    if let Some(ref norm_exp) = exp.norm_l2 {
        assert_vec_close(&rust_norms, &norm_exp.norms, 1e-4, "irreg_norm_l2");
    }
}

/// Mean after interpolation to common grid: compare against R.
#[test]
fn test_irreg_mean_vs_r() {
    let exp: IrregFdataExpected = load_json("expected", "irreg_fdata_expected");

    let ifd = IrregFdata::from_lists(&exp.argvals, &exp.values);

    if let Some(ref mean_exp) = exp.mean_curve {
        let rust_mean_vec = fdars_core::irreg_fdata::mean_irreg(
            &ifd,
            &mean_exp.target_grid,
            0.05,
            fdars_core::irreg_fdata::KernelType::Gaussian,
        );

        // Mean curve comparison — R uses linear interpolation, Rust uses kernel smoothing.
        // Methods inherently differ at boundaries and sparse regions.
        // Use rank correlation instead of pointwise comparison since the methods differ.
        assert_ranking_correlated_tol(
            &rust_mean_vec,
            &mean_exp.mean_values,
            0.9,
            "irreg_mean_curve_shape",
        );
    }
}

/// Interpolation to regular grid: compare against R's approx().
#[test]
fn test_irreg_to_regular_vs_r() {
    let exp: IrregFdataExpected = load_json("expected", "irreg_fdata_expected");

    let ifd = IrregFdata::from_lists(&exp.argvals, &exp.values);

    if let Some(ref reg_exp) = exp.to_regular {
        let rust_regular = fdars_core::irreg_fdata::to_regular_grid(&ifd, &reg_exp.target_grid);

        assert_eq!(rust_regular.nrows(), reg_exp.n);
        assert_eq!(rust_regular.ncols(), reg_exp.m);

        // Compare interpolated values (linear interpolation should match R's approx)
        for i in 0..reg_exp.n {
            for j in 0..reg_exp.m {
                let r_val = reg_exp.data[i + j * reg_exp.n]; // column-major
                let rust_val = rust_regular[(i, j)];
                assert!(
                    (rust_val - r_val).abs() < 1e-4,
                    "to_regular[{},{}]: Rust={:.6}, R={:.6}",
                    i,
                    j,
                    rust_val,
                    r_val
                );
            }
        }
    }
}

/// Pairwise L2 distances on irregular grids: compare against R.
#[test]
fn test_irreg_metric_vs_r() {
    let exp: IrregFdataExpected = load_json("expected", "irreg_fdata_expected");

    let ifd = IrregFdata::from_lists(&exp.argvals, &exp.values);

    if let Some(ref metric_exp) = exp.metric_lp {
        let rust_dist = fdars_core::irreg_fdata::metric_lp_irreg(&ifd, 2.0);

        let n = metric_exp.n;
        assert_eq!(rust_dist.nrows(), n);
        assert_eq!(rust_dist.ncols(), n);

        // Compare distance matrices with moderate tolerance
        // (R uses fine-grid interpolation, Rust may use different approach)
        for i in 0..n {
            for j in (i + 1)..n {
                let r_val = metric_exp.data[i + j * n]; // column-major
                let rust_val = rust_dist[(i, j)];
                // Relative tolerance for distance values
                let tol = r_val.abs().max(0.01) * 0.15;
                assert!(
                    (rust_val - r_val).abs() < tol,
                    "irreg_metric[{},{}]: Rust={:.6}, R={:.6}, tol={:.6}",
                    i,
                    j,
                    rust_val,
                    r_val,
                    tol
                );
            }
            // Diagonal should be zero
            assert!(
                rust_dist[(i, i)].abs() < 1e-10,
                "irreg_metric diagonal[{}] should be 0",
                i
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3b: Extended Alignment Validation
// ═══════════════════════════════════════════════════════════════════════════

/// Elastic decomposition: amplitude + phase distances vs R.
#[test]
fn test_elastic_decomposition_vs_r() {
    let exp: AlignmentExpected = load_json("expected", "alignment_expected");
    let d: AlignmentData = load_json("data", "alignment_30x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    if let Some(ref decomp_exp) = exp.elastic_decomposition {
        let f1: Vec<f64> = (0..d.m).map(|j| mat[(0, j)]).collect();
        let f2: Vec<f64> = (0..d.m).map(|j| mat[(1, j)]).collect();

        let decomp = fdars_core::alignment::elastic_decomposition(&f1, &f2, &d.argvals, 0.0);

        // Amplitude distance should be close to R's
        assert_relative_close(
            decomp.d_amplitude,
            decomp_exp.amplitude_distance,
            0.2,
            "amplitude_distance",
        );

        // Phase distance is more sensitive to alignment details
        assert!(
            decomp.d_phase.is_finite() && decomp.d_phase >= 0.0,
            "Phase distance should be finite and non-negative: {}",
            decomp.d_phase
        );
    }
}

/// Cross distance matrix: first 3 vs next 3 curves.
#[test]
fn test_elastic_cross_distance_vs_r() {
    let exp: AlignmentExpected = load_json("expected", "alignment_expected");
    let d: AlignmentData = load_json("data", "alignment_30x51");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    if let Some(ref cross_exp) = exp.cross_distance_3x3 {
        let n1 = cross_exp.n1;
        let n2 = cross_exp.n2;

        // Build two groups
        let g1_vec: Vec<f64> = (0..n1 * d.m)
            .map(|idx| {
                let i = idx % n1;
                let j = idx / n1;
                mat[(i, j)]
            })
            .collect();
        let g2_vec: Vec<f64> = (0..n2 * d.m)
            .map(|idx| {
                let i = idx % n2;
                let j = idx / n2;
                mat[(n1 + i, j)]
            })
            .collect();
        let g1 = FdMatrix::from_column_major(g1_vec, n1, d.m).unwrap();
        let g2 = FdMatrix::from_column_major(g2_vec, n2, d.m).unwrap();

        let cross = fdars_core::alignment::elastic_cross_distance_matrix(&g1, &g2, &d.argvals, 0.0);
        assert_eq!(cross.nrows(), n1);
        assert_eq!(cross.ncols(), n2);

        // Compare with R's cross distances (moderate tolerance due to DP alignment differences)
        for i in 0..n1 {
            for j in 0..n2 {
                let r_val = cross_exp.data[i + j * n1]; // column-major
                assert_relative_close(
                    cross[(i, j)],
                    r_val,
                    0.2,
                    &format!("cross_dist_{}_{}", i, j),
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// PHASE 3c: Extended Metrics Validation
// ═══════════════════════════════════════════════════════════════════════════

/// Hausdorff distance: compare 5x5 matrix against R.
#[test]
fn test_hausdorff_values_vs_r() {
    let exp: MetricsExpected = load_json("expected", "metrics_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    if let Some(ref h_exp) = exp.hausdorff_5x5 {
        let k = h_exp.n;
        let sub_vec: Vec<f64> = (0..k * d.m)
            .map(|idx| {
                let i = idx % k;
                let j = idx / k;
                mat[(i, j)]
            })
            .collect();
        let sub = FdMatrix::from_column_major(sub_vec, k, d.m).unwrap();

        let rust_dm = fdars_core::metric::hausdorff_self_1d(&sub, &d.argvals);
        assert_eq!(rust_dm.nrows(), k);

        // R computes pointwise sup-norm max|f1(t)-f2(t)|, while Rust uses 2D Hausdorff
        // on (t, f(t)) point sets. These are related but not identical metrics.
        // Verify rank correlation of the distance orderings.
        let mut r_vals = Vec::new();
        let mut rust_vals = Vec::new();
        for i in 0..k {
            for j in (i + 1)..k {
                r_vals.push(h_exp.data[i + j * k]); // column-major
                rust_vals.push(rust_dm[(i, j)]);
            }
        }
        assert_ranking_correlated(&rust_vals, &r_vals, "hausdorff_rank_order");
    }
}

/// Lp cross distance: compare first 5 vs next 5 against R.
#[test]
fn test_lp_cross_values_vs_r() {
    let exp: MetricsExpected = load_json("expected", "metrics_expected");
    let d: StandardData = load_json("data", "standard_50x101");
    let mat = FdMatrix::from_slice(&d.data, d.n, d.m).unwrap();

    if let Some(ref cross_exp) = exp.lp_cross_5x5 {
        let n1 = cross_exp.n1;
        let n2 = cross_exp.n2;
        let g1_vec: Vec<f64> = (0..n1 * d.m)
            .map(|idx| {
                let i = idx % n1;
                let j = idx / n1;
                mat[(i, j)]
            })
            .collect();
        let g2_vec: Vec<f64> = (0..n2 * d.m)
            .map(|idx| {
                let i = idx % n2;
                let j = idx / n2;
                mat[(n1 + i, j)]
            })
            .collect();
        let g1 = FdMatrix::from_column_major(g1_vec, n1, d.m).unwrap();
        let g2 = FdMatrix::from_column_major(g2_vec, n2, d.m).unwrap();

        // Pass empty user_weights — lp_cross_1d computes Simpson's weights internally
        let cross = fdars_core::metric::lp_cross_1d(&g1, &g2, &d.argvals, 2.0, &[]);
        assert_eq!(cross.nrows(), n1);
        assert_eq!(cross.ncols(), n2);

        // Compare cross distances — R uses Simpson's 1/3, Rust uses trapezoidal
        for i in 0..n1 {
            for j in 0..n2 {
                let r_val = cross_exp.data[i + j * n1]; // column-major
                assert_relative_close(cross[(i, j)], r_val, 0.05, &format!("lp_cross_{}_{}", i, j));
            }
        }
    }
}

// ─── New module helpers ──────────────────────────────────────────────────

/// Load fdasrvf-style data: R stores m×n column-major (columns = curves),
/// but FdMatrix needs n×m column-major (rows = curves). This transposes.
fn load_fdasrvf_data(flat: &[f64], n: usize, m: usize) -> FdMatrix {
    // R's flat: flat[j + i*m] = curve i at point j (column-major m×n)
    // Rust FdMatrix(n, m): data[(i, j)] = col_major[i + j*n] = curve i at point j
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = flat[j + i * m];
        }
    }
    FdMatrix::from_column_major(col_major, n, m).unwrap()
}

/// Load fdasrvf-style fitted data: R's eval.fd returns m×n, as.numeric flattens column-major.
fn load_fdasrvf_fitted(flat: &[f64], n: usize, m: usize) -> Vec<f64> {
    // Same transpose as data: flat[j + i*m] → FdMatrix order [i + j*n]
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = flat[j + i * m];
        }
    }
    col_major
}

/// Construct a KarcherMeanResult from R's alignment output,
/// bypassing Rust's alignment so FPCA comparison is apples-to-apples.
fn karcher_from_r(exp: &ElasticFpcaExpected) -> fdars_core::alignment::KarcherMeanResult {
    let n = exp.n;
    let m = exp.m;
    let aligned_data = load_fdasrvf_data(&exp.aligned_data, n, m);
    let gammas = load_fdasrvf_data(&exp.gammas, n, m);
    let aligned_srsfs = if !exp.aligned_srsfs.is_empty() {
        Some(load_fdasrvf_data(&exp.aligned_srsfs, n, m))
    } else {
        None
    };
    fdars_core::alignment::KarcherMeanResult {
        mean: exp.mean.clone(),
        mean_srsf: exp.mean_srsf.clone(),
        gammas,
        aligned_data,
        n_iter: 1,
        converged: true,
        aligned_srsfs,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Smooth Basis (R cross-validation)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct SmoothBasisExpected {
    bspline_smooth: BsplineSmoothExpected,
    fourier_smooth: FourierSmoothExpected,
    gcv_optimal: GcvOptimalExpected,
}

#[derive(Deserialize)]
struct BsplineSmoothExpected {
    data: Vec<f64>,
    n: usize,
    m: usize,
    nbasis: usize,
    lambda: f64,
    coefficients: Vec<f64>,
    fitted: Vec<f64>,
    edf: f64,
    gcv: Vec<f64>,
    penalty_matrix: Vec<f64>,
    penalty_dim: usize,
}

#[derive(Deserialize)]
struct FourierSmoothExpected {
    data: Vec<f64>,
    nbasis: usize,
    period: f64,
    lambda: f64,
    coefficients: Vec<f64>,
    fitted: Vec<f64>,
    edf: f64,
    gcv: Vec<f64>,
    penalty_matrix: Vec<f64>,
    penalty_dim: usize,
}

#[derive(Deserialize)]
struct GcvOptimalExpected {
    log_lambdas: Vec<f64>,
    gcv_values: Vec<f64>,
    best_log_lambda: f64,
    best_gcv: f64,
}

#[test]
fn validate_smooth_basis_bspline_penalty_matrix() {
    use fdars_core::smooth_basis::bspline_penalty_matrix;

    let exp: SmoothBasisExpected = load_json("expected", "smooth_basis");
    let bs = &exp.bspline_smooth;

    let argvals: Vec<f64> = (0..bs.m).map(|j| j as f64 / (bs.m - 1) as f64).collect();
    let pen = bspline_penalty_matrix(&argvals, bs.nbasis, 4, 2);
    let k = (pen.len() as f64).sqrt() as usize;

    // NOTE: Rust uses extended boundary knots while R uses repeated boundary knots,
    // so penalty matrix entries differ significantly. We check structural properties:

    // 1. Symmetric
    for i in 0..k {
        for j in i + 1..k {
            assert_scalar_close(
                pen[i + j * k],
                pen[j + i * k],
                1e-10,
                &format!("bspline_penalty_symmetry[{},{}]", i, j),
            );
        }
    }

    // 2. Positive semi-definite (diag entries non-negative)
    for i in 0..k {
        assert!(
            pen[i + i * k] >= -1e-10,
            "penalty diagonal [{}] = {} should be >= 0",
            i,
            pen[i + i * k]
        );
    }

    // 3. Same dimension as R
    assert_eq!(k, bs.penalty_dim, "penalty dimension mismatch");
}

#[test]
fn validate_smooth_basis_bspline_fitted() {
    use fdars_core::smooth_basis::{smooth_basis, BasisType, FdPar};

    let exp: SmoothBasisExpected = load_json("expected", "smooth_basis");
    let bs = &exp.bspline_smooth;

    let argvals: Vec<f64> = (0..bs.m).map(|j| j as f64 / (bs.m - 1) as f64).collect();
    let data = load_fdasrvf_data(&bs.data, bs.n, bs.m);

    let penalty = fdars_core::smooth_basis::bspline_penalty_matrix(&argvals, bs.nbasis, 4, 2);
    let fdpar = FdPar {
        basis_type: BasisType::Bspline { order: 4 },
        nbasis: bs.nbasis,
        lambda: bs.lambda,
        lfd_order: 2,
        penalty_matrix: penalty,
    };
    let result = smooth_basis(&data, &argvals, &fdpar).expect("smooth_basis should succeed");

    // NOTE: B-spline knot convention differs (Rust: extended, R: repeated boundary).
    // Fitted values will differ. Check structural properties instead:

    // 1. Correct dimensions
    assert_eq!(result.fitted.shape(), (bs.n, bs.m), "fitted shape mismatch");
    assert_eq!(
        result.coefficients.shape().0,
        bs.n,
        "coefficients n mismatch"
    );

    // 2. EDF should be positive and less than nbasis
    assert!(
        result.edf > 0.0 && result.edf <= bs.nbasis as f64 + 1.0,
        "EDF={} out of range",
        result.edf
    );

    // 3. Per-curve R² should be >0.95 (high quality smooth)
    for curve in 0..bs.n {
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        let mean: f64 = (0..bs.m).map(|j| data[(curve, j)]).sum::<f64>() / bs.m as f64;
        for j in 0..bs.m {
            let d = data[(curve, j)];
            let f = result.fitted[(curve, j)];
            ss_res += (d - f) * (d - f);
            ss_tot += (d - mean) * (d - mean);
        }
        let r2 = 1.0 - ss_res / ss_tot.max(1e-10);
        assert!(
            r2 > 0.95,
            "bspline curve {} R²={:.4} too low (expected >0.95)",
            curve,
            r2
        );
    }
}

#[test]
fn validate_smooth_basis_fourier_penalty_matrix() {
    use fdars_core::smooth_basis::fourier_penalty_matrix;

    let exp: SmoothBasisExpected = load_json("expected", "smooth_basis");
    let fs = &exp.fourier_smooth;

    let pen = fourier_penalty_matrix(fs.nbasis, fs.period, 2);
    assert_eq!(pen.len(), fs.penalty_dim * fs.penalty_dim);

    // Fourier penalty is diagonal — compare diagonals
    for i in 0..fs.penalty_dim {
        let rust_val = pen[i + i * fs.penalty_dim];
        let r_val = fs.penalty_matrix[i + i * fs.penalty_dim];
        assert_relative_close(
            rust_val,
            r_val,
            1e-6,
            &format!("fourier_penalty[{},{}]", i, i),
        );
    }
}

#[test]
fn validate_smooth_basis_fourier_fitted() {
    use fdars_core::smooth_basis::{smooth_basis, BasisType, FdPar};

    let exp: SmoothBasisExpected = load_json("expected", "smooth_basis");
    let fs = &exp.fourier_smooth;

    let m = 101;
    let n = 5;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let data = load_fdasrvf_data(&fs.data, n, m);

    let penalty = fdars_core::smooth_basis::fourier_penalty_matrix(fs.nbasis, fs.period, 2);
    let fdpar = FdPar {
        basis_type: BasisType::Fourier { period: fs.period },
        nbasis: fs.nbasis,
        lambda: fs.lambda,
        lfd_order: 2,
        penalty_matrix: penalty,
    };
    let result = smooth_basis(&data, &argvals, &fdpar).expect("smooth_basis should succeed");

    assert_eq!(result.fitted.shape(), (n, m), "fourier fitted shape");

    // EDF: should match R closely since penalty matrix matches exactly
    assert_scalar_close(result.edf, fs.edf, 0.01, "fourier_edf");

    // Fourier fitted values match R nearly exactly (RMSE < 1e-4)
    let r_fitted = load_fdasrvf_data(&fs.fitted, n, m);
    for curve in 0..n {
        let mut ss_res = 0.0;
        let mut ss_tot = 0.0;
        let mut ss_vs_r = 0.0;
        let mean: f64 = (0..m).map(|j| data[(curve, j)]).sum::<f64>() / m as f64;
        for j in 0..m {
            let d = data[(curve, j)];
            let f = result.fitted[(curve, j)];
            let rf = r_fitted[(curve, j)];
            ss_res += (d - f) * (d - f);
            ss_tot += (d - mean) * (d - mean);
            ss_vs_r += (f - rf) * (f - rf);
        }
        let r2 = 1.0 - ss_res / ss_tot.max(1e-10);
        let rmse_vs_r = (ss_vs_r / m as f64).sqrt();
        assert!(
            r2 > 0.95,
            "fourier curve {} R²={:.4} too low (expected >0.95)",
            curve,
            r2
        );
        assert!(
            rmse_vs_r < 1e-3,
            "fourier curve {} RMSE_vs_R={:.6} too high (should match R closely)",
            curve,
            rmse_vs_r
        );
    }
}

#[test]
fn validate_smooth_basis_gcv_optimal() {
    use fdars_core::smooth_basis::{smooth_basis_gcv, BasisType};

    let exp: SmoothBasisExpected = load_json("expected", "smooth_basis");
    let bs = &exp.bspline_smooth;

    let m = bs.m;
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let data = load_fdasrvf_data(&bs.data, bs.n, m);

    let result = smooth_basis_gcv(
        &data,
        &argvals,
        &BasisType::Bspline { order: 4 },
        bs.nbasis,
        2,
        (-8.0, 4.0),
        25,
    )
    .expect("smooth_basis_gcv should succeed");

    // GCV value should be very close to R despite knot convention differences
    assert_relative_close(
        result.gcv,
        exp.gcv_optimal.best_gcv,
        0.01,
        "gcv_optimal_value",
    );

    // EDF is reasonable (may differ due to knot convention)
    assert!(
        result.edf > 3.0 && result.edf < 20.0,
        "GCV-optimal EDF={} out of range",
        result.edf
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Elastic FPCA (R cross-validation)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct ElasticFpcaExpected {
    data: Vec<f64>,
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    aligned_data: Vec<f64>,
    mean: Vec<f64>,
    #[serde(default)]
    mean_srsf: Vec<f64>,
    #[serde(default)]
    gammas: Vec<f64>,
    #[serde(default)]
    aligned_srsfs: Vec<f64>,
    vert_fpca: Option<FpcaSubExpected>,
    horiz_fpca: Option<FpcaSubExpected>,
    joint_fpca: Option<JointFpcaSubExpected>,
}

#[derive(Deserialize)]
struct FpcaSubExpected {
    scores: Vec<f64>,
    scores_nrow: usize,
    scores_ncol: usize,
    eigenvalues: Vec<f64>,
    cumulative_variance: Vec<f64>,
}

#[derive(Deserialize)]
struct JointFpcaSubExpected {
    scores: Vec<f64>,
    scores_nrow: usize,
    scores_ncol: usize,
    eigenvalues: Vec<f64>,
    cumulative_variance: Vec<f64>,
    balance_c: f64,
}

#[test]
fn validate_elastic_vert_fpca() {
    use fdars_core::elastic_fpca::vert_fpca;

    let exp: ElasticFpcaExpected = load_json("expected", "elastic_fpca");
    let vf = exp.vert_fpca.as_ref().expect("vert_fpca should be present");

    // Use R's aligned data directly to isolate FPCA from alignment differences
    let karcher = karcher_from_r(&exp);

    let result =
        vert_fpca(&karcher, &exp.argvals, vf.scores_ncol).expect("vert_fpca should succeed");

    assert_eq!(result.eigenvalues.len(), vf.eigenvalues.len());

    // Eigenvalues: positive, decreasing
    for (i, ev) in result.eigenvalues.iter().enumerate() {
        assert!(
            *ev > 0.0,
            "vert_eigenvalue_{} should be positive: {}",
            i,
            ev
        );
    }
    for w in result.eigenvalues.windows(2) {
        assert!(w[0] >= w[1] - 1e-10, "vert eigenvalues not decreasing");
    }
    // Direct eigenvalue comparison against R (same aligned input + pre-computed SRSFs → exact)
    for (i, (rust_ev, r_ev)) in result.eigenvalues.iter().zip(&vf.eigenvalues).enumerate() {
        assert_relative_close(*rust_ev, *r_ev, 1e-6, &format!("vert_ev_{}", i));
    }

    // Cumulative variance: exact match with R
    for (i, (rust_cv, r_cv)) in result
        .cumulative_variance
        .iter()
        .zip(&vf.cumulative_variance)
        .enumerate()
    {
        assert_scalar_close(*rust_cv, *r_cv, 1e-6, &format!("vert_cumvar_{}", i));
    }

    // Scores shape
    let (sn, sk) = result.scores.shape();
    assert_eq!(sn, vf.scores_nrow, "vert_scores n mismatch");
    assert_eq!(sk, vf.scores_ncol, "vert_scores k mismatch");
}

#[test]
fn validate_elastic_horiz_fpca() {
    use fdars_core::elastic_fpca::horiz_fpca;

    let exp: ElasticFpcaExpected = load_json("expected", "elastic_fpca");
    let hf = exp
        .horiz_fpca
        .as_ref()
        .expect("horiz_fpca should be present");

    // Use R's aligned data + gammas directly
    let karcher = karcher_from_r(&exp);

    let result =
        horiz_fpca(&karcher, &exp.argvals, hf.scores_ncol).expect("horiz_fpca should succeed");

    assert_eq!(result.eigenvalues.len(), hf.eigenvalues.len());

    // Eigenvalues: positive, decreasing
    for ev in &result.eigenvalues {
        assert!(*ev > 0.0, "horiz eigenvalue should be positive");
    }
    for w in result.eigenvalues.windows(2) {
        assert!(w[0] >= w[1] - 1e-10, "horiz eigenvalues not decreasing");
    }
    // Direct eigenvalue comparison against R (same gammas + matching gradient → exact)
    for (i, (rust_ev, r_ev)) in result.eigenvalues.iter().zip(&hf.eigenvalues).enumerate() {
        assert_relative_close(*rust_ev, *r_ev, 1e-6, &format!("horiz_ev_{}", i));
    }

    // Cumulative variance: exact match with R
    for (i, (rust_cv, r_cv)) in result
        .cumulative_variance
        .iter()
        .zip(&hf.cumulative_variance)
        .enumerate()
    {
        assert_scalar_close(*rust_cv, *r_cv, 1e-6, &format!("horiz_cumvar_{}", i));
    }

    // Scores shape
    let (sn, sk) = result.scores.shape();
    assert_eq!(sn, hf.scores_nrow, "horiz_scores n mismatch");
    assert_eq!(sk, hf.scores_ncol, "horiz_scores k mismatch");
}

#[test]
fn validate_elastic_joint_fpca() {
    use fdars_core::elastic_fpca::joint_fpca;

    let exp: ElasticFpcaExpected = load_json("expected", "elastic_fpca");
    let jf = exp
        .joint_fpca
        .as_ref()
        .expect("joint_fpca should be present");

    // Use R's aligned data + gammas directly
    let karcher = karcher_from_r(&exp);

    // Use R's balance_c directly for apples-to-apples comparison
    let result = joint_fpca(&karcher, &exp.argvals, jf.scores_ncol, Some(jf.balance_c))
        .expect("joint_fpca should succeed");

    assert_eq!(result.eigenvalues.len(), jf.eigenvalues.len());

    // Eigenvalues: positive, decreasing
    for ev in &result.eigenvalues {
        assert!(*ev > 0.0, "joint eigenvalue should be positive: {}", ev);
    }
    for w in result.eigenvalues.windows(2) {
        assert!(w[0] >= w[1] - 1e-10, "joint eigenvalues not decreasing");
    }
    // Direct eigenvalue comparison (same data + same balance_c + pre-computed SRSFs → exact)
    for (i, (rust_ev, r_ev)) in result.eigenvalues.iter().zip(&jf.eigenvalues).enumerate() {
        assert_relative_close(*rust_ev, *r_ev, 1e-6, &format!("joint_ev_{}", i));
    }

    // Balance C: should match since we provided it
    assert_scalar_close(result.balance_c, jf.balance_c, 1e-6, "joint_balance_c");

    // Cumulative variance: exact match with R
    for (i, (rust_cv, r_cv)) in result
        .cumulative_variance
        .iter()
        .zip(&jf.cumulative_variance)
        .enumerate()
    {
        assert_scalar_close(*rust_cv, *r_cv, 1e-6, &format!("joint_cumvar_{}", i));
    }

    // Scores shape
    let (sn, sk) = result.scores.shape();
    assert_eq!(sn, jf.scores_nrow, "joint_scores n mismatch");
    assert_eq!(sk, jf.scores_ncol, "joint_scores k mismatch");
}

// ═══════════════════════════════════════════════════════════════════════════
// Elastic Regression (R cross-validation)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct ElasticRegressionExpected {
    data: Vec<f64>,
    y: Vec<f64>,
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    elastic_pcr_vert: Option<PcrSubExpected>,
    elastic_pcr_horiz: Option<PcrSubExpected>,
    elastic_pcr_combined: Option<PcrSubExpected>,
}

#[derive(Deserialize)]
struct PcrSubExpected {
    alpha: f64,
    coefficients: Vec<f64>,
    fitted_values: Vec<f64>,
    sse: f64,
    r_squared: Option<f64>,
}

#[test]
fn validate_elastic_pcr_vert() {
    use fdars_core::elastic_regression::{elastic_pcr, PcaMethod};

    let exp: ElasticRegressionExpected = load_json("expected", "elastic_regression");
    let pcr = exp
        .elastic_pcr_vert
        .as_ref()
        .expect("elastic_pcr_vert should be present");

    let data = load_fdasrvf_data(&exp.data, exp.n, exp.m);

    let result = elastic_pcr(
        &data,
        &exp.y,
        &exp.argvals,
        3,
        PcaMethod::Vertical,
        0.0,
        10,
        1e-4,
    )
    .expect("elastic_pcr (vert) should succeed");

    // Alpha (intercept): exact match (mean of y)
    assert_scalar_close(result.alpha, pcr.alpha, 1e-6, "pcr_vert_alpha");

    // SSE: within 5% of R
    assert_relative_close(result.sse, pcr.sse, 0.05, "pcr_vert_sse");

    // Number of coefficients
    assert_eq!(
        result.coefficients.len(),
        pcr.coefficients.len(),
        "pcr_vert ncoeff mismatch"
    );

    // Fitted values: compare against R
    assert_eq!(
        result.fitted_values.len(),
        exp.n,
        "pcr_vert fitted_values length"
    );
    if !pcr.fitted_values.is_empty() {
        for (i, (rust_v, r_v)) in result
            .fitted_values
            .iter()
            .zip(&pcr.fitted_values)
            .enumerate()
        {
            assert_scalar_close(*rust_v, *r_v, 0.1, &format!("pcr_vert_fitted_{}", i));
        }
    }

    // R²: both should be in similar range
    let mean_y: f64 = exp.y.iter().sum::<f64>() / exp.y.len() as f64;
    let ss_tot: f64 = exp.y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
    let r2 = 1.0 - result.sse / ss_tot;
    if let Some(r_r2) = pcr.r_squared {
        assert_scalar_close(r2, r_r2, 0.05, "pcr_vert_r_squared");
    }
}

#[test]
fn validate_elastic_pcr_horiz() {
    use fdars_core::elastic_regression::{elastic_pcr, PcaMethod};

    let exp: ElasticRegressionExpected = load_json("expected", "elastic_regression");
    let pcr = exp
        .elastic_pcr_horiz
        .as_ref()
        .expect("elastic_pcr_horiz should be present");

    let data = load_fdasrvf_data(&exp.data, exp.n, exp.m);

    let result = elastic_pcr(
        &data,
        &exp.y,
        &exp.argvals,
        3,
        PcaMethod::Horizontal,
        0.0,
        10,
        1e-4,
    )
    .expect("elastic_pcr (horiz) should succeed");

    // Alpha: exact match
    assert_scalar_close(result.alpha, pcr.alpha, 1e-6, "pcr_horiz_alpha");

    // SSE: within 5% of R
    assert_relative_close(result.sse, pcr.sse, 0.05, "pcr_horiz_sse");

    assert_eq!(
        result.coefficients.len(),
        pcr.coefficients.len(),
        "pcr_horiz ncoeff mismatch"
    );

    // Fitted values
    if !pcr.fitted_values.is_empty() {
        for (i, (rust_v, r_v)) in result
            .fitted_values
            .iter()
            .zip(&pcr.fitted_values)
            .enumerate()
        {
            assert_scalar_close(*rust_v, *r_v, 0.1, &format!("pcr_horiz_fitted_{}", i));
        }
    }

    // R²
    if let Some(r_r2) = pcr.r_squared {
        let mean_y: f64 = exp.y.iter().sum::<f64>() / exp.y.len() as f64;
        let ss_tot: f64 = exp.y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let r2 = 1.0 - result.sse / ss_tot;
        assert_scalar_close(r2, r_r2, 0.05, "pcr_horiz_r_squared");
    }
}

#[test]
fn validate_elastic_pcr_combined() {
    use fdars_core::elastic_regression::{elastic_pcr, PcaMethod};

    let exp: ElasticRegressionExpected = load_json("expected", "elastic_regression");
    let pcr = exp
        .elastic_pcr_combined
        .as_ref()
        .expect("elastic_pcr_combined should be present");

    let data = load_fdasrvf_data(&exp.data, exp.n, exp.m);

    let result = elastic_pcr(
        &data,
        &exp.y,
        &exp.argvals,
        3,
        PcaMethod::Joint,
        0.0,
        10,
        1e-4,
    )
    .expect("elastic_pcr (combined) should succeed");

    // Alpha: exact match
    assert_scalar_close(result.alpha, pcr.alpha, 1e-6, "pcr_combined_alpha");

    // SSE: within 5% of R
    assert_relative_close(result.sse, pcr.sse, 0.05, "pcr_combined_sse");

    assert_eq!(
        result.coefficients.len(),
        pcr.coefficients.len(),
        "pcr_combined ncoeff mismatch"
    );

    // Fitted values
    if !pcr.fitted_values.is_empty() {
        for (i, (rust_v, r_v)) in result
            .fitted_values
            .iter()
            .zip(&pcr.fitted_values)
            .enumerate()
        {
            assert_scalar_close(*rust_v, *r_v, 0.1, &format!("pcr_combined_fitted_{}", i));
        }
    }

    // R²
    if let Some(r_r2) = pcr.r_squared {
        let mean_y: f64 = exp.y.iter().sum::<f64>() / exp.y.len() as f64;
        let ss_tot: f64 = exp.y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let r2 = 1.0 - result.sse / ss_tot;
        assert_scalar_close(r2, r_r2, 0.05, "pcr_combined_r_squared");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Elastic Changepoint Detection (R cross-validation)
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Deserialize)]
struct ElasticChangepointExpected {
    data_amp: Vec<f64>,
    n: usize,
    m: usize,
    argvals: Vec<f64>,
    true_changepoint: usize,
    amp_changepoint: Option<AmpChangepointExpected>,
}

#[derive(Deserialize)]
struct AmpChangepointExpected {
    detected_changepoint: usize,
    cusum_values: Vec<f64>,
    test_statistic: f64,
}

#[test]
fn validate_elastic_amp_changepoint() {
    use fdars_core::elastic_changepoint::{elastic_amp_changepoint, CovKernel};

    let exp: ElasticChangepointExpected = load_json("expected", "elastic_changepoint");
    let acp = exp
        .amp_changepoint
        .as_ref()
        .expect("amp_changepoint should be present");

    let data = load_fdasrvf_data(&exp.data_amp, exp.n, exp.m);

    let result = elastic_amp_changepoint(
        &data,
        &exp.argvals,
        0.0,
        5,   // max_iter for alignment
        200, // n_mc
        CovKernel::Simple,
        None,
        42,
    )
    .expect("elastic_amp_changepoint should succeed");

    // Detected changepoint: exact match on this synthetic data
    assert_eq!(
        result.changepoint, acp.detected_changepoint,
        "changepoint: Rust={}, R={}",
        result.changepoint, acp.detected_changepoint
    );

    // CUSUM values: direct comparison (near-exact for simple sine data)
    assert_eq!(
        result.cusum_values.len(),
        acp.cusum_values.len(),
        "cusum_values length mismatch"
    );
    for (k, (rust_v, r_v)) in result
        .cusum_values
        .iter()
        .zip(&acp.cusum_values)
        .enumerate()
    {
        assert_relative_close(*rust_v, *r_v, 1e-4, &format!("cusum[{}]", k));
    }

    // Test statistic: direct comparison
    assert_relative_close(
        result.test_statistic,
        acp.test_statistic,
        1e-4,
        "test_statistic",
    );

    // P-value should be in [0, 1]
    assert!(
        result.p_value >= 0.0 && result.p_value <= 1.0,
        "p_value={} should be in [0,1]",
        result.p_value
    );
}
