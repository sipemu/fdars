//! Automatic basis type and parameter selection.

use super::bspline::bspline_basis;
use super::fourier::fourier_basis;
use super::helpers::{compute_model_criterion, svd_pseudoinverse};
use super::projection::ProjectionBasisType;
use super::pspline::difference_matrix;
use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;

/// Result of automatic basis selection for a single curve.
#[derive(Debug, Clone)]
pub struct SingleCurveSelection {
    /// Selected basis type.
    pub basis_type: ProjectionBasisType,
    /// Selected number of basis functions
    pub nbasis: usize,
    /// Best criterion score (GCV, AIC, or BIC)
    pub score: f64,
    /// Coefficients for the selected basis
    pub coefficients: Vec<f64>,
    /// Fitted values
    pub fitted: Vec<f64>,
    /// Effective degrees of freedom
    pub edf: f64,
    /// Whether seasonal pattern was detected (if use_seasonal_hint)
    pub seasonal_detected: bool,
    /// Lambda value (for P-spline, NaN for Fourier)
    pub lambda: f64,
}

/// Result of automatic basis selection for all curves.
#[derive(Debug, Clone)]
pub struct BasisAutoSelectionResult {
    /// Per-curve selection results
    pub selections: Vec<SingleCurveSelection>,
    /// Criterion used (0=GCV, 1=AIC, 2=BIC)
    pub criterion: i32,
}

/// Detect if a curve has seasonal/periodic pattern using FFT.
///
/// Returns true if the peak power in the periodogram is significantly
/// above the mean power level.
fn detect_seasonality_fft(curve: &[f64]) -> bool {
    use rustfft::{num_complex::Complex, FftPlanner};

    let n = curve.len();
    if n < 8 {
        return false;
    }

    // Remove mean
    let mean: f64 = curve.iter().sum::<f64>() / n as f64;
    let mut input: Vec<Complex<f64>> = curve.iter().map(|&x| Complex::new(x - mean, 0.0)).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    fft.process(&mut input);

    // Compute power spectrum (skip DC component and Nyquist)
    let powers: Vec<f64> = input[1..n / 2]
        .iter()
        .map(nalgebra::Complex::norm_sqr)
        .collect();

    if powers.is_empty() {
        return false;
    }

    let max_power = powers.iter().copied().fold(0.0_f64, f64::max);
    let mean_power = powers.iter().sum::<f64>() / powers.len() as f64;

    // Seasonal if peak is significantly above mean
    max_power > 2.0 * mean_power
}

/// Fit a single curve with Fourier basis and compute criterion score.
fn fit_curve_fourier(
    curve: &[f64],
    m: usize,
    argvals: &[f64],
    nbasis: usize,
    criterion: i32,
) -> Option<(f64, Vec<f64>, Vec<f64>, f64)> {
    let nbasis = if nbasis % 2 == 0 { nbasis + 1 } else { nbasis };

    let basis = fourier_basis(argvals, nbasis);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let btb = &b_mat.transpose() * &b_mat;
    let btb_inv = svd_pseudoinverse(&btb)?;
    let proj = &btb_inv * b_mat.transpose();
    let h_mat = &b_mat * &proj;
    let edf: f64 = (0..m).map(|i| h_mat[(i, i)]).sum();

    let curve_vec = DVector::from_column_slice(curve);
    let coefs = &btb_inv * (b_mat.transpose() * &curve_vec);
    let fitted = &b_mat * &coefs;

    let rss: f64 = (0..m).map(|j| (curve[j] - fitted[j]).powi(2)).sum();
    let score = compute_model_criterion(rss, m as f64, edf, criterion);

    let coef_vec: Vec<f64> = (0..actual_nbasis).map(|k| coefs[k]).collect();
    let fitted_vec: Vec<f64> = (0..m).map(|j| fitted[j]).collect();

    Some((score, coef_vec, fitted_vec, edf))
}

/// Fit a single curve with P-spline basis and compute criterion score.
fn fit_curve_pspline(
    curve: &[f64],
    m: usize,
    argvals: &[f64],
    nbasis: usize,
    lambda: f64,
    order: usize,
    criterion: i32,
) -> Option<(f64, Vec<f64>, Vec<f64>, f64)> {
    let basis = bspline_basis(argvals, nbasis.saturating_sub(4).max(2), 4);
    let actual_nbasis = basis.len() / m;
    let b_mat = DMatrix::from_column_slice(m, actual_nbasis, &basis);

    let d = difference_matrix(actual_nbasis, order);
    let penalty = &d.transpose() * &d;
    let btb = &b_mat.transpose() * &b_mat;
    let btb_penalized = &btb + lambda * &penalty;

    let btb_inv = svd_pseudoinverse(&btb_penalized)?;
    let proj = &btb_inv * b_mat.transpose();
    let h_mat = &b_mat * &proj;
    let edf: f64 = (0..m).map(|i| h_mat[(i, i)]).sum();

    let curve_vec = DVector::from_column_slice(curve);
    let coefs = &btb_inv * (b_mat.transpose() * &curve_vec);
    let fitted = &b_mat * &coefs;

    let rss: f64 = (0..m).map(|j| (curve[j] - fitted[j]).powi(2)).sum();
    let score = compute_model_criterion(rss, m as f64, edf, criterion);

    let coef_vec: Vec<f64> = (0..actual_nbasis).map(|k| coefs[k]).collect();
    let fitted_vec: Vec<f64> = (0..m).map(|j| fitted[j]).collect();

    Some((score, coef_vec, fitted_vec, edf))
}

/// Result of a basis search for a single curve.
struct BasisSearchResult {
    score: f64,
    nbasis: usize,
    coefs: Vec<f64>,
    fitted: Vec<f64>,
    edf: f64,
    lambda: f64,
}

/// Search over Fourier basis sizes for the best fit.
fn search_fourier_basis(
    curve: &[f64],
    m: usize,
    argvals: &[f64],
    fourier_min: usize,
    fourier_max: usize,
    seasonal: bool,
    criterion: i32,
) -> Option<BasisSearchResult> {
    let fourier_start = if seasonal {
        fourier_min.max(5)
    } else {
        fourier_min
    };
    let mut nb = if fourier_start % 2 == 0 {
        fourier_start + 1
    } else {
        fourier_start
    };

    let mut best: Option<BasisSearchResult> = None;
    while nb <= fourier_max {
        if let Some((score, coefs, fitted, edf)) =
            fit_curve_fourier(curve, m, argvals, nb, criterion)
        {
            if score.is_finite() && best.as_ref().map_or(true, |b| score < b.score) {
                best = Some(BasisSearchResult {
                    score,
                    nbasis: nb,
                    coefs,
                    fitted,
                    edf,
                    lambda: f64::NAN,
                });
            }
        }
        nb += 2;
    }
    best
}

/// Try a single P-spline fit and update best if it improves the score.
fn try_pspline_fit_update(
    curve: &[f64],
    m: usize,
    argvals: &[f64],
    nb: usize,
    lam: f64,
    criterion: i32,
    best: &mut Option<BasisSearchResult>,
) {
    if let Some((score, coefs, fitted, edf)) =
        fit_curve_pspline(curve, m, argvals, nb, lam, 2, criterion)
    {
        if score.is_finite() && best.as_ref().map_or(true, |b| score < b.score) {
            *best = Some(BasisSearchResult {
                score,
                nbasis: nb,
                coefs,
                fitted,
                edf,
                lambda: lam,
            });
        }
    }
}

/// Search over P-spline basis sizes (and optionally lambda) for the best fit.
fn search_pspline_basis(
    curve: &[f64],
    m: usize,
    argvals: &[f64],
    pspline_min: usize,
    pspline_max: usize,
    lambda_grid: &[f64],
    auto_lambda: bool,
    lambda: f64,
    criterion: i32,
) -> Option<BasisSearchResult> {
    let mut best: Option<BasisSearchResult> = None;
    for nb in pspline_min..=pspline_max {
        let lambdas: Box<dyn Iterator<Item = f64>> = if auto_lambda {
            Box::new(lambda_grid.iter().copied())
        } else {
            Box::new(std::iter::once(lambda))
        };
        for lam in lambdas {
            try_pspline_fit_update(curve, m, argvals, nb, lam, criterion, &mut best);
        }
    }
    best
}

/// Select optimal basis type and parameters for each curve individually.
///
/// This function compares Fourier and P-spline bases for each curve,
/// selecting the optimal basis type and number of basis functions using
/// model selection criteria (GCV, AIC, or BIC).
///
/// # Arguments
/// * `data` - Column-major FdMatrix (n x m)
/// * `argvals` - Evaluation points
/// * `criterion` - Model selection criterion: 0=GCV, 1=AIC, 2=BIC
/// * `nbasis_min` - Minimum number of basis functions (0 for auto)
/// * `nbasis_max` - Maximum number of basis functions (0 for auto)
/// * `lambda_pspline` - Smoothing parameter for P-spline (negative for auto-select)
/// * `use_seasonal_hint` - Whether to use FFT to detect seasonality
///
/// # Returns
/// BasisAutoSelectionResult with per-curve selections
pub fn select_basis_auto_1d(
    data: &FdMatrix,
    argvals: &[f64],
    criterion: i32,
    nbasis_min: usize,
    nbasis_max: usize,
    lambda_pspline: f64,
    use_seasonal_hint: bool,
) -> BasisAutoSelectionResult {
    let n = data.nrows();
    let m = data.ncols();
    let fourier_min = if nbasis_min > 0 { nbasis_min.max(3) } else { 3 };
    let fourier_max = if nbasis_max > 0 {
        nbasis_max.min(m / 3).min(25)
    } else {
        (m / 3).min(25)
    };
    let pspline_min = if nbasis_min > 0 { nbasis_min.max(6) } else { 6 };
    let pspline_max = if nbasis_max > 0 {
        nbasis_max.min(m / 2).min(40)
    } else {
        (m / 2).min(40)
    };

    let lambda_grid = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0];
    let auto_lambda = lambda_pspline < 0.0;

    let selections: Vec<SingleCurveSelection> = iter_maybe_parallel!(0..n)
        .map(|i| {
            let curve: Vec<f64> = (0..m).map(|j| data[(i, j)]).collect();
            let seasonal_detected = if use_seasonal_hint {
                detect_seasonality_fft(&curve)
            } else {
                false
            };

            let fourier_best = search_fourier_basis(
                &curve,
                m,
                argvals,
                fourier_min,
                fourier_max,
                seasonal_detected,
                criterion,
            );
            let pspline_best = search_pspline_basis(
                &curve,
                m,
                argvals,
                pspline_min,
                pspline_max,
                &lambda_grid,
                auto_lambda,
                lambda_pspline,
                criterion,
            );

            // Pick the best overall result
            let (basis_type, result) = match (fourier_best, pspline_best) {
                (Some(f), Some(p)) => {
                    if f.score <= p.score {
                        (ProjectionBasisType::Fourier, f)
                    } else {
                        (ProjectionBasisType::Bspline, p)
                    }
                }
                (Some(f), None) => (ProjectionBasisType::Fourier, f),
                (None, Some(p)) => (ProjectionBasisType::Bspline, p),
                (None, None) => {
                    return SingleCurveSelection {
                        basis_type: ProjectionBasisType::Bspline,
                        nbasis: pspline_min,
                        score: f64::INFINITY,
                        coefficients: Vec::new(),
                        fitted: Vec::new(),
                        edf: 0.0,
                        seasonal_detected,
                        lambda: f64::NAN,
                    };
                }
            };

            SingleCurveSelection {
                basis_type,
                nbasis: result.nbasis,
                score: result.score,
                coefficients: result.coefs,
                fitted: result.fitted,
                edf: result.edf,
                seasonal_detected,
                lambda: result.lambda,
            }
        })
        .collect();

    BasisAutoSelectionResult {
        selections,
        criterion,
    }
}
