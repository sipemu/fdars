//! Elastic shape analysis: quotient space operations and orbit representatives.
//!
//! Extends elastic alignment to work in quotient spaces where curves are
//! considered equivalent up to reparameterization, translation, and/or scaling.

use super::karcher::karcher_mean;
use super::pairwise::{elastic_align_pair, elastic_self_distance_matrix};
use super::srsf::srsf_single;
use super::{AlignmentResult, KarcherMeanResult};
use crate::error::FdarError;
use crate::helpers::simpsons_weights;
use crate::matrix::FdMatrix;
use crate::warping::l2_norm_l2;

// ─── Types ──────────────────────────────────────────────────────────────────

/// Quotient space for shape analysis.
///
/// Determines which transformations are factored out when comparing curves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ShapeQuotient {
    /// Quotient by reparameterization only (elastic distance).
    #[default]
    Reparameterization,
    /// Quotient by reparameterization and vertical translation.
    ReparameterizationTranslation,
    /// Quotient by reparameterization, translation, and scale.
    ReparameterizationTranslationScale,
}

/// A canonical representative of a shape orbit.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OrbitRepresentative {
    /// The pre-processed curve (centered, scaled, etc.).
    pub representative: Vec<f64>,
    /// SRSF of the representative curve.
    pub representative_srsf: Vec<f64>,
    /// Warping function applied (identity for orbit_representative).
    pub gamma: Vec<f64>,
    /// Vertical translation removed.
    pub translation: f64,
    /// Scale factor removed.
    pub scale: f64,
}

/// Result of computing the elastic shape distance between two curves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ShapeDistanceResult {
    /// Shape distance in the quotient space.
    pub distance: f64,
    /// Optimal warping function (length m).
    pub gamma: Vec<f64>,
    /// Second curve aligned to the first.
    pub f2_aligned: Vec<f64>,
}

/// Result of computing the shape mean of a set of curves.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct ShapeMeanResult {
    /// Shape mean curve.
    pub mean: Vec<f64>,
    /// SRSF of the shape mean.
    pub mean_srsf: Vec<f64>,
    /// Warping functions (n x m).
    pub gammas: FdMatrix,
    /// Curves aligned to the mean (n x m).
    pub aligned_data: FdMatrix,
    /// Number of iterations used.
    pub n_iter: usize,
    /// Whether the algorithm converged.
    pub converged: bool,
}

// ─── Pre-processing Helpers ─────────────────────────────────────────────────

/// Compute the integral mean of a curve using Simpson's weights.
fn integral_mean(f: &[f64], argvals: &[f64]) -> f64 {
    let w = simpsons_weights(argvals);
    let total_w: f64 = w.iter().sum();
    if total_w <= 0.0 {
        return 0.0;
    }
    let wsum: f64 = f.iter().zip(w.iter()).map(|(&fi, &wi)| fi * wi).sum();
    wsum / total_w
}

/// Pre-process a curve according to the quotient type.
///
/// Returns `(processed_curve, translation, scale)`.
fn preprocess_curve(f: &[f64], argvals: &[f64], quotient: ShapeQuotient) -> (Vec<f64>, f64, f64) {
    let mut curve = f.to_vec();
    let mut translation = 0.0;
    let mut scale = 1.0;

    match quotient {
        ShapeQuotient::Reparameterization => {
            // No pre-processing needed.
        }
        ShapeQuotient::ReparameterizationTranslation => {
            // Subtract integral mean.
            let mean_val = integral_mean(&curve, argvals);
            translation = mean_val;
            for v in &mut curve {
                *v -= mean_val;
            }
        }
        ShapeQuotient::ReparameterizationTranslationScale => {
            // Subtract integral mean, then scale by SRSF L2 norm.
            let mean_val = integral_mean(&curve, argvals);
            translation = mean_val;
            for v in &mut curve {
                *v -= mean_val;
            }

            // Compute L2 norm of the SRSF for scale normalization.
            let q = srsf_single(&curve, argvals);
            // Use a uniform [0,1] time grid for the L2 norm computation.
            let m = argvals.len();
            let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1).max(1) as f64).collect();
            let norm = l2_norm_l2(&q, &time);

            if norm > 1e-10 {
                scale = norm;
                for v in &mut curve {
                    *v /= norm;
                }
            }
        }
    }

    (curve, translation, scale)
}

/// Pre-process all rows of a data matrix according to the quotient type.
fn preprocess_data(data: &FdMatrix, argvals: &[f64], quotient: ShapeQuotient) -> FdMatrix {
    let (n, m) = data.shape();
    let mut result = FdMatrix::zeros(n, m);
    for i in 0..n {
        let row = data.row(i);
        let (processed, _, _) = preprocess_curve(&row, argvals, quotient);
        for j in 0..m {
            result[(i, j)] = processed[j];
        }
    }
    result
}

// ─── Public API ─────────────────────────────────────────────────────────────

/// Compute the canonical orbit representative of a curve.
///
/// Applies the quotient transformations (centering, scaling) and computes
/// the SRSF of the result. The warping function is the identity.
///
/// # Arguments
/// * `f`        — Curve values (length m).
/// * `argvals`  — Evaluation points (length m).
/// * `quotient` — Which transformations to factor out.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if lengths do not match or `m < 2`.
pub fn orbit_representative(
    f: &[f64],
    argvals: &[f64],
    quotient: ShapeQuotient,
) -> Result<OrbitRepresentative, FdarError> {
    let m = f.len();
    if m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "f",
            expected: format!("length {}", argvals.len()),
            actual: format!("length {m}"),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "f",
            expected: "length >= 2".to_string(),
            actual: format!("length {m}"),
        });
    }

    let (representative, translation, scale) = preprocess_curve(f, argvals, quotient);
    let representative_srsf = srsf_single(&representative, argvals);
    let gamma = argvals.to_vec(); // identity warp

    Ok(OrbitRepresentative {
        representative,
        representative_srsf,
        gamma,
        translation,
        scale,
    })
}

/// Compute the elastic shape distance between two curves.
///
/// Pre-processes both curves according to the quotient type, then computes
/// the elastic distance after optimal alignment.
///
/// # Arguments
/// * `f1`       — First curve (length m).
/// * `f2`       — Second curve (length m).
/// * `argvals`  — Evaluation points (length m).
/// * `quotient` — Which transformations to factor out.
/// * `lambda`   — Roughness penalty for alignment.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if lengths do not match or `m < 2`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_distance(
    f1: &[f64],
    f2: &[f64],
    argvals: &[f64],
    quotient: ShapeQuotient,
    lambda: f64,
) -> Result<ShapeDistanceResult, FdarError> {
    let m = f1.len();
    if m != f2.len() || m != argvals.len() {
        return Err(FdarError::InvalidDimension {
            parameter: "f1/f2",
            expected: format!("matching lengths == argvals.len() ({})", argvals.len()),
            actual: format!("f1.len()={}, f2.len()={}", f1.len(), f2.len()),
        });
    }
    if m < 2 {
        return Err(FdarError::InvalidDimension {
            parameter: "f1",
            expected: "length >= 2".to_string(),
            actual: format!("length {m}"),
        });
    }

    let (f1_pre, _, _) = preprocess_curve(f1, argvals, quotient);
    let (f2_pre, _, _) = preprocess_curve(f2, argvals, quotient);

    let AlignmentResult {
        gamma,
        f_aligned,
        distance,
    } = elastic_align_pair(&f1_pre, &f2_pre, argvals, lambda);

    Ok(ShapeDistanceResult {
        distance,
        gamma,
        f2_aligned: f_aligned,
    })
}

/// Compute the pairwise shape distance matrix for a set of curves.
///
/// Pre-processes all curves according to the quotient type, then delegates to
/// the elastic self-distance matrix computation.
///
/// # Arguments
/// * `data`     — Functional data matrix (n x m).
/// * `argvals`  — Evaluation points (length m).
/// * `quotient` — Which transformations to factor out.
/// * `lambda`   — Roughness penalty for alignment.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_self_distance_matrix(
    data: &FdMatrix,
    argvals: &[f64],
    quotient: ShapeQuotient,
    lambda: f64,
) -> Result<FdMatrix, FdarError> {
    let (_n, m) = data.shape();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }

    let preprocessed = preprocess_data(data, argvals, quotient);
    Ok(elastic_self_distance_matrix(&preprocessed, argvals, lambda))
}

/// Compute the Karcher (Frechet) mean in the elastic shape space.
///
/// Pre-processes all curves according to the quotient type, then computes
/// the Karcher mean on the preprocessed data.
///
/// # Arguments
/// * `data`     — Functional data matrix (n x m).
/// * `argvals`  — Evaluation points (length m).
/// * `quotient` — Which transformations to factor out.
/// * `lambda`   — Roughness penalty for alignment.
/// * `max_iter` — Maximum number of Karcher iterations.
/// * `tol`      — Convergence tolerance.
///
/// # Errors
/// Returns [`FdarError::InvalidDimension`] if `argvals` length does not match `m`
/// or `n < 1`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn shape_mean(
    data: &FdMatrix,
    argvals: &[f64],
    quotient: ShapeQuotient,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> Result<ShapeMeanResult, FdarError> {
    let (n, m) = data.shape();
    if argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "argvals",
            expected: format!("{m}"),
            actual: format!("{}", argvals.len()),
        });
    }
    if n < 1 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "at least 1 row".to_string(),
            actual: format!("{n} rows"),
        });
    }

    let preprocessed = preprocess_data(data, argvals, quotient);
    let KarcherMeanResult {
        mean,
        mean_srsf,
        gammas,
        aligned_data,
        n_iter,
        converged,
        ..
    } = karcher_mean(&preprocessed, argvals, max_iter, tol, lambda);

    Ok(ShapeMeanResult {
        mean,
        mean_srsf,
        gammas,
        aligned_data,
        n_iter,
        converged,
    })
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulation::{sim_fundata, EFunType, EValType};
    use crate::test_helpers::uniform_grid;

    fn make_data(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        let t = uniform_grid(m);
        let data = sim_fundata(n, &t, 3, EFunType::Fourier, EValType::Exponential, Some(99));
        (data, t)
    }

    // ── orbit_representative ──

    #[test]
    fn orbit_representative_reparam_only() {
        let t = uniform_grid(30);
        let f: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin()).collect();
        let rep = orbit_representative(&f, &t, ShapeQuotient::Reparameterization).unwrap();
        // No transformation: representative should match original.
        assert_eq!(rep.representative.len(), 30);
        for i in 0..30 {
            assert!(
                (rep.representative[i] - f[i]).abs() < 1e-12,
                "reparameterization-only orbit should not change the curve"
            );
        }
        assert!((rep.translation - 0.0).abs() < f64::EPSILON);
        assert!((rep.scale - 1.0).abs() < f64::EPSILON);
        assert_eq!(rep.gamma, t);
    }

    #[test]
    fn orbit_representative_translation() {
        let t = uniform_grid(30);
        let offset = 5.0;
        let f: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin() + offset).collect();
        let rep =
            orbit_representative(&f, &t, ShapeQuotient::ReparameterizationTranslation).unwrap();
        // The integral mean should be removed; representative should be roughly centered.
        let mean_after = integral_mean(&rep.representative, &t);
        assert!(
            mean_after.abs() < 1e-10,
            "translation quotient should center the curve, mean={mean_after}"
        );
    }

    #[test]
    fn orbit_representative_translation_scale() {
        let t = uniform_grid(50);
        let f: Vec<f64> = t.iter().map(|&x| 10.0 * (x * 4.0).sin() + 3.0).collect();
        let rep = orbit_representative(&f, &t, ShapeQuotient::ReparameterizationTranslationScale)
            .unwrap();
        assert!(rep.scale > 0.0, "scale factor should be positive");

        // Scaling a curve by alpha should produce the same representative (up to sign).
        let f2: Vec<f64> = t.iter().map(|&x| 20.0 * (x * 4.0).sin() + 3.0).collect();
        let rep2 = orbit_representative(&f2, &t, ShapeQuotient::ReparameterizationTranslationScale)
            .unwrap();

        // The representatives should be proportional (same shape); check correlation.
        let dot: f64 = rep
            .representative
            .iter()
            .zip(rep2.representative.iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let n1: f64 = rep
            .representative
            .iter()
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt();
        let n2: f64 = rep2
            .representative
            .iter()
            .map(|&v| v * v)
            .sum::<f64>()
            .sqrt();
        let corr = if n1 > 1e-10 && n2 > 1e-10 {
            dot / (n1 * n2)
        } else {
            1.0
        };
        assert!(
            corr > 0.99,
            "scaled curves should have nearly identical representatives, corr={corr}"
        );
    }

    #[test]
    fn orbit_representative_length_mismatch() {
        let t = uniform_grid(30);
        let f = vec![1.0; 20];
        assert!(orbit_representative(&f, &t, ShapeQuotient::Reparameterization).is_err());
    }

    #[test]
    fn orbit_representative_too_short() {
        let f = vec![1.0];
        let t = vec![0.0];
        assert!(orbit_representative(&f, &t, ShapeQuotient::Reparameterization).is_err());
    }

    // ── shape_distance ──

    #[test]
    fn shape_distance_identical_curves() {
        let t = uniform_grid(30);
        let f: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin()).collect();
        let result = shape_distance(&f, &f, &t, ShapeQuotient::Reparameterization, 0.0).unwrap();
        assert!(
            result.distance < 0.1,
            "distance between identical curves should be near zero, got {}",
            result.distance
        );
        assert_eq!(result.gamma.len(), 30);
        assert_eq!(result.f2_aligned.len(), 30);
    }

    #[test]
    fn shape_distance_translated_curves() {
        let t = uniform_grid(30);
        let f1: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin()).collect();
        let f2: Vec<f64> = t.iter().map(|&x| (x * 6.0).sin() + 5.0).collect();

        // Without translation quotient: distance should be large.
        let d_no_trans =
            shape_distance(&f1, &f2, &t, ShapeQuotient::Reparameterization, 0.0).unwrap();
        // With translation quotient: distance should be much smaller.
        let d_trans = shape_distance(
            &f1,
            &f2,
            &t,
            ShapeQuotient::ReparameterizationTranslation,
            0.0,
        )
        .unwrap();

        assert!(
            d_trans.distance < d_no_trans.distance + 0.01,
            "translation quotient should not increase distance: d_trans={}, d_no_trans={}",
            d_trans.distance,
            d_no_trans.distance
        );
    }

    #[test]
    fn shape_distance_length_mismatch() {
        let t = uniform_grid(30);
        let f1 = vec![0.0; 30];
        let f2 = vec![0.0; 20];
        assert!(shape_distance(&f1, &f2, &t, ShapeQuotient::Reparameterization, 0.0).is_err());
    }

    // ── shape_self_distance_matrix ──

    #[test]
    fn shape_distance_matrix_smoke() {
        let (data, t) = make_data(5, 20);
        let dmat =
            shape_self_distance_matrix(&data, &t, ShapeQuotient::Reparameterization, 0.0).unwrap();
        assert_eq!(dmat.shape(), (5, 5));
        // Diagonal should be zero.
        for i in 0..5 {
            assert!(
                dmat[(i, i)].abs() < 1e-10,
                "diagonal should be zero, got {}",
                dmat[(i, i)]
            );
        }
        // Should be symmetric.
        for i in 0..5 {
            for j in (i + 1)..5 {
                assert!(
                    (dmat[(i, j)] - dmat[(j, i)]).abs() < 1e-10,
                    "distance matrix should be symmetric"
                );
            }
        }
    }

    #[test]
    fn shape_distance_matrix_argvals_mismatch() {
        let (data, _) = make_data(5, 20);
        let bad_t = uniform_grid(15);
        assert!(
            shape_self_distance_matrix(&data, &bad_t, ShapeQuotient::Reparameterization, 0.0)
                .is_err()
        );
    }

    // ── shape_mean ──

    #[test]
    fn shape_mean_smoke() {
        let (data, t) = make_data(6, 25);
        let result =
            shape_mean(&data, &t, ShapeQuotient::Reparameterization, 0.0, 5, 1e-2).unwrap();
        assert_eq!(result.mean.len(), 25);
        assert_eq!(result.mean_srsf.len(), 25);
        assert_eq!(result.gammas.shape(), (6, 25));
        assert_eq!(result.aligned_data.shape(), (6, 25));
        assert!(result.n_iter >= 1);
    }

    #[test]
    fn shape_mean_translation_quotient() {
        let (data, t) = make_data(6, 25);
        let result = shape_mean(
            &data,
            &t,
            ShapeQuotient::ReparameterizationTranslation,
            0.0,
            5,
            1e-2,
        )
        .unwrap();
        assert_eq!(result.mean.len(), 25);
    }

    #[test]
    fn shape_mean_full_quotient() {
        let (data, t) = make_data(6, 25);
        let result = shape_mean(
            &data,
            &t,
            ShapeQuotient::ReparameterizationTranslationScale,
            0.0,
            5,
            1e-2,
        )
        .unwrap();
        assert_eq!(result.mean.len(), 25);
    }

    #[test]
    fn shape_mean_argvals_mismatch() {
        let (data, _) = make_data(5, 25);
        let bad_t = uniform_grid(15);
        assert!(shape_mean(
            &data,
            &bad_t,
            ShapeQuotient::Reparameterization,
            0.0,
            5,
            1e-2
        )
        .is_err());
    }

    #[test]
    fn default_quotient() {
        assert_eq!(ShapeQuotient::default(), ShapeQuotient::Reparameterization);
    }
}
