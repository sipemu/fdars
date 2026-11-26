//! Optional pre-smoothing of curves before the SRSF transform (ALN-HYGIENE #56).
//!
//! The SRSF `q = sign(f')·sqrt(|f'|)` differentiates the data, so measurement
//! noise in `f` is amplified in `q`. fdasrvf offers two remedies, mirrored here:
//!
//! * [`SrsfSmoothing::BoxFilter`] — fdasrvf `smooth.data(f, sparam)`: `passes`
//!   sweeps of the `(1, 2, 1)/4` filter over the interior points, endpoints
//!   fixed. Matches fdasrvf 2.6.0 (unchanged since 2.5.0) to round-off.
//! * [`SrsfSmoothing::SmoothingSpline`] — the semantics of fdasrvf (≥ 2.5.1,
//!   validated against 2.6.0) `f_to_srvf(f, time, smooth = TRUE)`, which differentiates
//!   `stats::smooth.spline(time, f)` (a cubic smoothing spline with a GCV-chosen
//!   penalty, chosen per curve). Here each curve is fitted with the crate's
//!   penalised cubic B-spline smoother ([`crate::smooth_basis::smooth_basis`],
//!   `∫(f'')²` penalty) with its own GCV-optimal `λ`, then passed through the
//!   standard SRSF derivative. Differences from R: uniform knots (R uses data
//!   quantiles), a log-λ grid search (R uses Brent on `spar`) and a numerical
//!   rather than analytic derivative of the smooth — so values agree in
//!   accuracy, not bit-for-bit.
//!
//! [`SrsfSmoothing::None`] (the default) is exactly [`srsf_transform`].

use super::srsf::srsf_transform;
use crate::error::FdarError;
use crate::matrix::FdMatrix;
use crate::smooth_basis::{bspline_penalty_matrix, smooth_basis, BasisType, FdPar};

/// Pre-smoothing applied to the curves before the SRSF transform.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum SrsfSmoothing {
    /// No smoothing: identical to [`srsf_transform`] (default).
    #[default]
    None,
    /// fdasrvf `smooth.data(f, sparam = passes)`: `passes` sweeps of
    /// `f[j] ← (f[j−1] + 2 f[j] + f[j+1]) / 4` over the interior points
    /// (all updated from the previous sweep), endpoints unchanged. Needs
    /// `m >= 3`.
    BoxFilter {
        /// Number of filter sweeps (fdasrvf `sparam`; fdasrvf's
        /// `time_warping` default is 25). `0` leaves the data unchanged.
        passes: usize,
    },
    /// Per-curve GCV cubic smoothing spline, the semantics of fdasrvf (≥ 2.5.1)
    /// `f_to_srvf(smooth = TRUE)` (see the module docs). Needs `m >= 4`.
    SmoothingSpline,
}

/// Apply the pre-smoothing of `smoothing` to every curve (row) of `data`.
///
/// Returns the smoothed curves (same shape). This is the `f` that
/// [`srsf_transform_smoothed`] differentiates; it is exposed so the smoothed
/// curves can be reused (e.g. for plotting or as alignment input).
///
/// # Errors
/// [`FdarError::InvalidDimension`] if `argvals.len() != m`, `n == 0`, or `m`
/// is too small for the method (`BoxFilter`: `m >= 3`; `SmoothingSpline`:
/// `m >= 4`); [`FdarError::ComputationFailed`] if no spline fit succeeds.
///
/// # Example
/// ```
/// use fdars_core::alignment::{srsf_presmooth, SrsfSmoothing};
/// use fdars_core::matrix::FdMatrix;
///
/// let t: Vec<f64> = (0..5).map(|j| j as f64 / 4.0).collect();
/// let data = FdMatrix::from_column_major(vec![0.0, 0.0, 4.0, 0.0, 0.0], 1, 5).unwrap();
/// let s = srsf_presmooth(&data, &t, &SrsfSmoothing::BoxFilter { passes: 1 }).unwrap();
/// assert_eq!(s.row(0), vec![0.0, 1.0, 2.0, 1.0, 0.0]);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn srsf_presmooth(
    data: &FdMatrix,
    argvals: &[f64],
    smoothing: &SrsfSmoothing,
) -> Result<FdMatrix, FdarError> {
    validate(data, argvals, smoothing)?;
    match *smoothing {
        SrsfSmoothing::None => Ok(data.clone()),
        SrsfSmoothing::BoxFilter { passes } => Ok(box_filter(data, passes)),
        SrsfSmoothing::SmoothingSpline => spline_smooth(data, argvals),
    }
}

/// SRSF transform with optional pre-smoothing of the curves.
///
/// With [`SrsfSmoothing::None`] the output is bit-identical to
/// [`srsf_transform`]; otherwise the curves are first smoothed by
/// [`srsf_presmooth`] and the smoothed curves are transformed.
///
/// # Errors
/// As [`srsf_presmooth`].
///
/// # Example
/// ```
/// use fdars_core::alignment::{srsf_transform, srsf_transform_smoothed, SrsfSmoothing};
/// use fdars_core::matrix::FdMatrix;
///
/// let m = 101;
/// let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
/// // A sine observed with high-frequency noise.
/// let f: Vec<f64> = (0..m)
///     .map(|j| (6.283 * t[j]).sin() + 0.05 * (707.1 * (j + 1) as f64).sin())
///     .collect();
/// let data = FdMatrix::from_column_major(f, 1, m).unwrap();
/// let q_raw = srsf_transform(&data, &t);
/// let q_smooth = srsf_transform_smoothed(&data, &t, &SrsfSmoothing::SmoothingSpline).unwrap();
/// assert_eq!(q_smooth.shape(), q_raw.shape());
/// let none = srsf_transform_smoothed(&data, &t, &SrsfSmoothing::None).unwrap();
/// assert_eq!(none, q_raw);
/// ```
#[must_use = "expensive computation whose result should not be discarded"]
pub fn srsf_transform_smoothed(
    data: &FdMatrix,
    argvals: &[f64],
    smoothing: &SrsfSmoothing,
) -> Result<FdMatrix, FdarError> {
    if matches!(smoothing, SrsfSmoothing::None) {
        validate(data, argvals, smoothing)?;
        return Ok(srsf_transform(data, argvals));
    }
    let smoothed = srsf_presmooth(data, argvals, smoothing)?;
    Ok(srsf_transform(&smoothed, argvals))
}

fn validate(data: &FdMatrix, argvals: &[f64], smoothing: &SrsfSmoothing) -> Result<(), FdarError> {
    let (n, m) = data.shape();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("length {m}"),
            actual: format!("length {}", argvals.len()),
        });
    }
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "nrows > 0".to_string(),
            actual: "nrows = 0".to_string(),
        });
    }
    let min_m = match smoothing {
        SrsfSmoothing::None => 1,
        SrsfSmoothing::BoxFilter { .. } => 3,
        SrsfSmoothing::SmoothingSpline => 4,
    };
    if m < min_m {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: format!("ncols >= {min_m} for {smoothing:?}"),
            actual: format!("ncols = {m}"),
        });
    }
    Ok(())
}

/// fdasrvf `smooth.data`: vectorised `(1,2,1)/4` sweeps, endpoints fixed.
fn box_filter(data: &FdMatrix, passes: usize) -> FdMatrix {
    let (n, m) = data.shape();
    let mut out = data.clone();
    let mut row = vec![0.0; m];
    let mut next = vec![0.0; m];
    for i in 0..n {
        for j in 0..m {
            row[j] = out[(i, j)];
        }
        for _ in 0..passes {
            next[0] = row[0];
            next[m - 1] = row[m - 1];
            for j in 1..m - 1 {
                next[j] = (row[j - 1] + 2.0 * row[j] + row[j + 1]) / 4.0;
            }
            std::mem::swap(&mut row, &mut next);
        }
        for j in 0..m {
            out[(i, j)] = row[j];
        }
    }
    out
}

/// Knot count of R's `smooth.spline` (`.nknots.smspl`, all knots below 50).
fn r_nknots(m: usize) -> usize {
    if m < 50 {
        return m;
    }
    let (a1, a2, a3, a4) = (50f64.log2(), 100f64.log2(), 140f64.log2(), 200f64.log2());
    let x = m as f64;
    let k = if m < 200 {
        (a1 + (a2 - a1) * (x - 50.0) / 150.0).exp2()
    } else if m < 800 {
        (a2 + (a3 - a2) * (x - 200.0) / 600.0).exp2()
    } else if m < 3200 {
        (a3 + (a4 - a3) * (x - 800.0) / 2400.0).exp2()
    } else {
        200.0 + (x - 3200.0).powf(0.2)
    };
    k as usize
}

/// Per-curve GCV-optimal penalised cubic B-spline smooth (`∫(f'')²` penalty).
///
/// The basis, penalty and hat matrix depend only on `argvals` and `λ`, so each
/// `λ` of the log grid is fitted for all curves in one call; every curve then
/// keeps the fit with its own smallest GCV `(RSS/m) / (1 − edf/m)²`.
fn spline_smooth(data: &FdMatrix, argvals: &[f64]) -> Result<FdMatrix, FdarError> {
    const ORDER: usize = 4;
    const LOG_LAMBDA: (f64, f64) = (-12.0, 4.0);
    const N_GRID: usize = 65;

    let (n, m) = data.shape();
    // natural cubic spline on K knots has K + 2 B-spline dof; at least 6 so
    // the basis has two interior knots.
    let nbasis = (r_nknots(m) + 2).clamp(6, m.max(6));
    let penalty = bspline_penalty_matrix(argvals, nbasis, ORDER, 2);
    // Rescale the λ grid by the domain length^3 so it is invariant to the
    // units of `argvals` (∫(f'')² scales as L^-3 for a fixed shape).
    let span = argvals[m - 1] - argvals[0];
    let scale = span.powi(3);

    let mut best_gcv = vec![f64::INFINITY; n];
    let mut out = data.clone();
    let mut any_ok = false;
    for g in 0..N_GRID {
        let log_lam = LOG_LAMBDA.0 + (LOG_LAMBDA.1 - LOG_LAMBDA.0) * g as f64 / (N_GRID - 1) as f64;
        let fdpar = FdPar {
            basis_type: BasisType::Bspline { order: ORDER },
            nbasis,
            lambda: 10f64.powf(log_lam) * scale,
            lfd_order: 2,
            penalty_matrix: penalty.clone(),
        };
        let Ok(fit) = smooth_basis(data, argvals, &fdpar) else {
            continue;
        };
        let denom = 1.0 - fit.edf / m as f64;
        if denom.abs() <= 1e-10 {
            continue;
        }
        any_ok = true;
        for i in 0..n {
            let rss: f64 = (0..m)
                .map(|j| (data[(i, j)] - fit.fitted[(i, j)]).powi(2))
                .sum();
            let gcv = (rss / m as f64) / (denom * denom);
            if gcv < best_gcv[i] {
                best_gcv[i] = gcv;
                for j in 0..m {
                    out[(i, j)] = fit.fitted[(i, j)];
                }
            }
        }
    }
    if !any_ok {
        return Err(FdarError::ComputationFailed {
            operation: "srsf_presmooth",
            detail: "no smoothing-spline fit succeeded on the lambda grid".to_string(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::simpsons_weights;

    fn grid(m: usize) -> Vec<f64> {
        (0..m).map(|j| j as f64 / (m - 1) as f64).collect()
    }

    /// Golden values: R fdasrvf 2.6.0 (CRAN; bit-identical to the 2.5.0 output
    /// first used, GH #74),
    /// `t <- seq(0, 1, length.out = 9); f <- cbind(sin(3*t) + t^2, cos(5*t));
    ///  smooth.data(f, 3)` (`.planning/quick/261126-mun-*/golden_srsf_smoothing_260.R`).
    #[test]
    #[allow(clippy::excessive_precision)] // R's %.17g output, verbatim
    fn box_filter_matches_fdasrvf_smooth_data() {
        let t = grid(9);
        let mut d = FdMatrix::zeros(2, 9);
        for j in 0..9 {
            d[(0, j)] = (3.0 * t[j]).sin() + t[j] * t[j];
            d[(1, j)] = (5.0 * t[j]).cos();
        }
        let r1 = [
            0.0,
            0.363_583_728_105_155_71,
            0.698_475_159_693_621_73,
            0.975_509_080_350_860_92,
            1.170_526_014_638_696_7,
            1.272_111_320_451_747_1,
            1.285_509_100_322_616_2,
            1.231_404_669_483_914_2,
            1.141_120_008_059_867_1,
        ];
        let r2 = [
            1.0,
            0.658_900_813_967_537_97,
            0.240_003_192_027_680_06,
            -0.222_374_119_247_887_97,
            -0.594_770_208_397_203_29,
            -0.742_299_287_933_240_96,
            -0.607_508_777_192_535_93,
            -0.229_629_484_560_902_6,
            0.283_662_185_463_226_25,
        ];
        let s = srsf_presmooth(&d, &t, &SrsfSmoothing::BoxFilter { passes: 3 }).unwrap();
        for j in 0..9 {
            assert!((s[(0, j)] - r1[j]).abs() < 1e-14, "row0 j={j}");
            assert!((s[(1, j)] - r2[j]).abs() < 1e-14, "row1 j={j}");
        }
    }

    #[test]
    fn none_is_bit_identical_to_srsf_transform() {
        let t = grid(30);
        let mut d = FdMatrix::zeros(3, 30);
        for i in 0..3 {
            for j in 0..30 {
                d[(i, j)] = (t[j] * (2.0 + i as f64)).sin();
            }
        }
        let a = srsf_transform_smoothed(&d, &t, &SrsfSmoothing::None).unwrap();
        assert_eq!(a, srsf_transform(&d, &t));
        // passes = 0 is the identity as well.
        let b = srsf_transform_smoothed(&d, &t, &SrsfSmoothing::BoxFilter { passes: 0 }).unwrap();
        assert_eq!(b, srsf_transform(&d, &t));
    }

    #[test]
    fn linear_curves_are_preserved() {
        // Both smoothers reproduce straight lines (box filter exactly; the
        // spline because lines are in the basis and carry zero ∫(f'')² penalty).
        let t = grid(40);
        let mut d = FdMatrix::zeros(2, 40);
        for j in 0..40 {
            d[(0, j)] = 2.0 * t[j] - 0.5;
            d[(1, j)] = -3.0 * t[j] + 1.0;
        }
        let b = srsf_presmooth(&d, &t, &SrsfSmoothing::BoxFilter { passes: 25 }).unwrap();
        let s = srsf_presmooth(&d, &t, &SrsfSmoothing::SmoothingSpline).unwrap();
        for i in 0..2 {
            for j in 0..40 {
                assert!((b[(i, j)] - d[(i, j)]).abs() < 1e-13);
                assert!((s[(i, j)] - d[(i, j)]).abs() < 1e-6, "spline i={i} j={j}");
            }
        }
    }

    /// fdasrvf 2.6.0 `f_to_srvf(f, t, smooth = TRUE)` on the data of
    /// [`smoothing_spline_denoises_like_fdasrvf`] (`golden_srsf_smoothing_260.R`).
    #[allow(clippy::excessive_precision, clippy::unreadable_literal)] // R's %.17g output
    const R_Q_SMOOTH: [f64; 101] = [
        2.4335729219668845,
        2.432608177479838,
        2.4293233686795683,
        2.422682082791531,
        2.4122930688274371,
        2.3972215674308304,
        2.3768537545833195,
        2.3510103572371679,
        2.3194661667707108,
        2.2817737695401328,
        2.237815800284197,
        2.1876324049908136,
        2.1309155807089502,
        2.0675397821480441,
        1.9971571308282723,
        1.9193865892239319,
        1.8336961785455035,
        1.7395429932032651,
        1.6363196136203053,
        1.5223821962274016,
        1.3952738238349101,
        1.252081732571896,
        1.0875139213216756,
        0.89057943384416782,
        0.63281383851507522,
        0.077554733356226743,
        -0.62276292275147827,
        -0.88323535472749726,
        -1.0813755821993782,
        -1.2466811206344375,
        -1.3901915528584958,
        -1.5172954382114612,
        -1.6316932065655676,
        -1.7358486472075447,
        -1.8308936845356105,
        -1.9176399200681911,
        -1.9969357215061931,
        -2.0694125097119427,
        -2.1354873762855453,
        -2.1954264784924575,
        -2.249679100923947,
        -2.298272684656848,
        -2.341278827812006,
        -2.3790867809548413,
        -2.411661476396683,
        -2.4389361829006808,
        -2.4613104912547965,
        -2.4787997434491187,
        -2.491169377254971,
        -2.4985449164465376,
        -2.5010199797479733,
        -2.4985205214773449,
        -2.4910858031356105,
        -2.4788152376671082,
        -2.4614758077523886,
        -2.4390468393952824,
        -2.4116387858357453,
        -2.3790829193120873,
        -2.3411678997185255,
        -2.2980256565134911,
        -2.249647385947116,
        -2.1955704148670114,
        -2.1355023060547809,
        -2.0693546662015221,
        -1.9968221863845612,
        -1.9175205985352468,
        -1.8309267483864358,
        -1.7360588606558773,
        -1.6319627264920713,
        -1.5171400579753622,
        -1.3897471229355736,
        -1.246706907365416,
        -1.0816754065592189,
        -0.88368584118082694,
        -0.62384929146902302,
        0.069910333640615455,
        0.63182121970737415,
        0.8898569776294698,
        1.0867769870402066,
        1.2510671422293618,
        1.3940939869491364,
        1.5211810963449668,
        1.635595586730298,
        1.7391457443209133,
        1.83306631551688,
        1.9188209464834785,
        1.9968669546217173,
        2.0675028750159732,
        2.1314571309158352,
        2.1888341621357661,
        2.2396771430330511,
        2.2843786044176739,
        2.3230519306354847,
        2.3557986565584947,
        2.38296646059118,
        2.4048468203086739,
        2.4214327525745767,
        2.433261444543112,
        2.4409787603312032,
        2.4453330210300774,
        2.4470510700414958,
    ];

    /// Reference: fdasrvf 2.6.0 `f_to_srvf(f, t, smooth = TRUE)` (CRAN; it is
    /// `predict(smooth.spline(t, f), t, deriv = 1)` then `g/sqrt(|g|+eps)`, equal
    /// to 0 ulp to the hand-written route that stood in for it under 2.5.0, GH #74).
    /// For `t = seq(0,1,len=101)`, `f = sin(2πt) + 0.05 sin(1000·(j+1)·0.7071)`
    /// (j = 0..100), R 4.x gives an L2 error vs the noise-free SRSF of
    /// 0.0236 (smoothed) and 1.0957 (unsmoothed `f_to_srvf`, interpolating spline).
    #[test]
    #[allow(clippy::approx_constant)] // 0.7071 is the literal used in the R reference
    fn smoothing_spline_denoises_like_fdasrvf() {
        let m = 101;
        let t = grid(m);
        let tau = 2.0 * std::f64::consts::PI;
        let mut d = FdMatrix::zeros(1, m);
        let mut q_true = vec![0.0; m];
        for j in 0..m {
            d[(0, j)] = (tau * t[j]).sin() + 0.05 * (1000.0 * (j + 1) as f64 * 0.7071).sin();
            let g = tau * (tau * t[j]).cos();
            q_true[j] = g.signum() * g.abs().sqrt();
        }
        let w = simpsons_weights(&t);
        let err = |q: &FdMatrix| -> f64 {
            (0..m)
                .map(|j| w[j] * (q[(0, j)] - q_true[j]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let e_raw = err(&srsf_transform(&d, &t));
        let e_spl = err(&srsf_transform_smoothed(&d, &t, &SrsfSmoothing::SmoothingSpline).unwrap());
        // Same order of accuracy as R's smooth.spline route (0.0236), and a
        // large improvement over the raw transform. (fdars' raw finite-difference
        // derivative gives 0.41 here, R's interpolating-spline one 1.10.)
        assert!(e_raw > 0.3, "raw error {e_raw}");
        // fdars: 0.0270. Bound: within 1.5x of the R reference error.
        assert!(
            e_spl < 1.5 * 0.0236,
            "spline error {e_spl} (R fdasrvf: 0.0236)"
        );
        let e_box = err(
            &srsf_transform_smoothed(&d, &t, &SrsfSmoothing::BoxFilter { passes: 25 }).unwrap(),
        );
        assert!(e_box < e_raw / 3.0, "box error {e_box} vs raw {e_raw}");
        let q_spl = srsf_transform_smoothed(&d, &t, &SrsfSmoothing::SmoothingSpline).unwrap();
        let dist: f64 = (0..m)
            .map(|j| w[j] * (q_spl[(0, j)] - R_Q_SMOOTH[j]).powi(2))
            .sum::<f64>()
            .sqrt();
        let e_r: f64 = (0..m)
            .map(|j| w[j] * (R_Q_SMOOTH[j] - q_true[j]).powi(2))
            .sum::<f64>()
            .sqrt();
        // Direct agreement with fdasrvf 2.6.0's smoothed SRSF (added in GH #74;
        // before, only the accuracy against the truth was asserted): fdars'
        // uniform-knot B-spline + GCV grid vs R's smooth.spline + Brent differ by
        // an L2 distance of 2.7e-3 (max pointwise 9.6e-3), a ninth of either
        // method's error against the noise-free SRSF (~0.024).
        assert!(
            dist < 4e-3,
            "L2 distance to fdasrvf f_to_srvf(smooth=TRUE): {dist}"
        );
        assert!(e_r < 0.025, "R reference error {e_r} (Simpson weights)");
    }

    #[test]
    fn spline_lambda_grid_is_unit_invariant() {
        // Rescaling time (t -> 10 t) must give the same smoothed values.
        let m = 60;
        let t = grid(m);
        let t10: Vec<f64> = t.iter().map(|x| 10.0 * x).collect();
        let mut d = FdMatrix::zeros(1, m);
        for j in 0..m {
            d[(0, j)] = (5.0 * t[j]).sin() + 0.03 * (313.7 * (j + 1) as f64).sin();
        }
        let a = srsf_presmooth(&d, &t, &SrsfSmoothing::SmoothingSpline).unwrap();
        let b = srsf_presmooth(&d, &t10, &SrsfSmoothing::SmoothingSpline).unwrap();
        for j in 0..m {
            assert!((a[(0, j)] - b[(0, j)]).abs() < 1e-6, "j={j}");
        }
    }

    #[test]
    fn validation_errors() {
        let t = grid(3);
        let d = FdMatrix::zeros(1, 3);
        assert!(srsf_presmooth(&d, &t[..2], &SrsfSmoothing::None).is_err());
        assert!(srsf_presmooth(&d, &t, &SrsfSmoothing::SmoothingSpline).is_err());
        assert!(srsf_presmooth(&d, &t, &SrsfSmoothing::BoxFilter { passes: 1 }).is_ok());
        let t2 = grid(2);
        let d2 = FdMatrix::zeros(1, 2);
        assert!(srsf_presmooth(&d2, &t2, &SrsfSmoothing::BoxFilter { passes: 1 }).is_err());
        let empty = FdMatrix::zeros(0, 3);
        assert!(srsf_transform_smoothed(&empty, &t, &SrsfSmoothing::None).is_err());
    }
}
