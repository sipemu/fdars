//! Public fit→transform seam for joint functional PCA (jfPCA).
//!
//! This module exposes a reusable [`JfpcaModel`] trained by [`jfpca_fit`] that
//! can project out-of-sample curves onto the trained joint-FPCA basis in the
//! trained coordinate system via [`JfpcaModel::transform`].
//!
//! # Overview
//!
//! Joint FPCA (Tucker et al.) decomposes functional curves into amplitude
//! (vertical) and phase (horizontal) variability after elastic alignment.
//! This module wraps the existing [`joint_fpca`] machinery into a persistent
//! model that stores all state needed for out-of-sample projection.
//!
//! # Example
//!
//! ```
//! use fdars_core::{jfpca_fit, JfpcaModel, JfpcaTransform};
//! use fdars_core::matrix::FdMatrix;
//! use std::f64::consts::PI;
//!
//! // Build a small spanning multi-frequency FdMatrix (n=6, m=12)
//! let n = 6;
//! let m = 12;
//! let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
//! let mut data = FdMatrix::zeros(n, m);
//! for i in 0..n {
//!     for j in 0..m {
//!         let t = argvals[j];
//!         let amp1 = 1.0 + 0.3 * i as f64;
//!         let amp2 = 0.5 - 0.1 * i as f64;
//!         let amp3 = 0.3 + 0.05 * i as f64;
//!         data[(i, j)] = amp1 * (2.0 * PI * t).sin()
//!             + amp2 * (4.0 * PI * t).cos()
//!             + amp3 * (6.0 * PI * t).sin();
//!     }
//! }
//!
//! let ncomp = 3;
//! let model = jfpca_fit(&data, &argvals, ncomp, None, 0.0, 20)?;
//! let transform = model.transform(&data)?;
//! assert_eq!(transform.scores.shape(), (n, model.ncomp));
//! # Ok::<(), fdars_core::FdarError>(())
//! ```

use crate::alignment::{align_to_target, karcher_mean, srsf_transform};
use crate::elastic_fpca::{
    build_augmented_srsfs, center_matrix, horiz_fpca, joint_fpca, shooting_vectors_from_psis,
    warps_to_normalized_psi, JointFpcaResult,
};
use crate::matrix::FdMatrix;
use crate::FdarError;

/// Trained joint-FPCA model that stores the full basis for out-of-sample projection.
///
/// Created by [`jfpca_fit`]; used via [`JfpcaModel::transform`].
///
/// All fields are public and `#[non_exhaustive]` — new fields may be added in
/// future minor versions without breaking existing destructuring.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JfpcaModel {
    /// Trained Karcher-mean template curve (length m).
    ///
    /// New curves are aligned to this fixed template during [`JfpcaModel::transform`].
    pub karcher_mean: Vec<f64>,
    /// Mean of augmented SRSF matrix (length m+1).
    ///
    /// Used to center new curves' augmented SRSFs using the *training-time* mean,
    /// preserving coordinate-system alignment.
    pub mean_q: Vec<f64>,
    /// Trained ψ Karcher mean on the Hilbert sphere (length m).
    ///
    /// Used to compute shooting vectors for new curves. Captured from
    /// [`horiz_fpca`] at fit time (discarded by [`joint_fpca`]).
    pub mean_psi: Vec<f64>,
    /// Vertical (amplitude) eigenvector component (ncomp × (m+1)).
    ///
    /// Rows are the amplitude rows of the joint right-singular vectors V^T.
    pub vert_component: FdMatrix,
    /// Horizontal (phase) eigenvector component (ncomp × m).
    ///
    /// Rows are the phase rows of the joint right-singular vectors V^T.
    pub horiz_component: FdMatrix,
    /// Phase-vs-amplitude balance weight used at training time.
    pub balance_c: f64,
    /// Evaluation grid (length m).
    ///
    /// New curves passed to [`JfpcaModel::transform`] must have the same number
    /// of columns; a mismatch returns [`FdarError::InvalidDimension`].
    pub argvals: Vec<f64>,
    /// Eigenvalues (variance explained, length `ncomp`).
    pub eigenvalues: Vec<f64>,
    /// Number of principal components (clamped to `n-1` at training time).
    pub ncomp: usize,
    /// Full joint FPCA result from training (contains training scores).
    pub joint_result: JointFpcaResult,
    /// Warp penalty weight used at training time.
    ///
    /// Stored so the transform step can reproduce the same alignment geodesic.
    pub lambda: f64,
}

/// Output of [`JfpcaModel::transform`] — projections of new curves onto the
/// trained joint-FPCA basis.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct JfpcaTransform {
    /// PC scores in the trained coordinate system (n_new × ncomp).
    pub scores: FdMatrix,
    /// New curves aligned to the trained Karcher-mean template (n_new × m).
    pub aligned: FdMatrix,
    /// Warping functions mapping new curves to the template (n_new × m).
    pub warping: FdMatrix,
}

/// Fit a joint-FPCA model on a set of functional curves.
///
/// Runs elastic alignment (Karcher mean), joint FPCA, and captures all state
/// needed for out-of-sample projection via [`JfpcaModel::transform`].
///
/// Training scores in the returned [`JfpcaModel`] reproduce those from a direct
/// call to [`joint_fpca`] with the same arguments within 1e-8 element-wise.
///
/// # Arguments
///
/// * `data` — Functional data matrix (n × m); rows are curves, columns are
///   evaluation points.
/// * `argvals` — Evaluation grid (length m). Must satisfy `argvals.len() == m`.
/// * `ncomp` — Number of principal components to extract (≥ 1). Clamped to
///   `n-1` internally; the stored [`JfpcaModel::ncomp`] reflects the clamped value.
/// * `balance_c` — Phase-vs-amplitude balance weight. `None` triggers golden-
///   section optimization (matches [`joint_fpca`] default).
/// * `lambda` — Warp regularization penalty for Karcher alignment (0.0 = no
///   penalty; must match any subsequent manual alignment calls).
/// * `max_iter` — Maximum Karcher-mean iterations (20 is a safe default).
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] when:
/// - `n < 2` or `m < 2`
/// - `ncomp < 1`
/// - `argvals.len() != m`
///
/// # Examples
///
/// See the [module-level example](self).
#[must_use = "expensive computation: fit returns a trained JfpcaModel; use it to project curves"]
pub fn jfpca_fit(
    data: &FdMatrix,
    argvals: &[f64],
    ncomp: usize,
    balance_c: Option<f64>,
    lambda: f64,
    max_iter: usize,
) -> Result<JfpcaModel, FdarError> {
    let (n, m) = data.shape();
    if n < 2 || m < 2 || ncomp < 1 || argvals.len() != m {
        return Err(FdarError::InvalidDimension {
            parameter: "data/argvals/ncomp",
            expected: "n >= 2, m >= 2, ncomp >= 1, argvals.len() == m".to_string(),
            actual: format!(
                "n={}, m={}, ncomp={}, argvals.len()={}",
                n,
                m,
                ncomp,
                argvals.len()
            ),
        });
    }

    // Step 1: Karcher-mean alignment
    let karcher = karcher_mean(data, argvals, max_iter, 1e-4, lambda);

    // Step 2: Joint FPCA — training scores live here; reuse directly so scores
    // are bit-identical (not merely close) to a standalone joint_fpca call.
    let joint_result = joint_fpca(&karcher, argvals, ncomp, balance_c)?;

    // Step 3: Horizontal FPCA to capture mean_psi.
    // joint_fpca internally calls horiz_fpca but discards its mean_psi; we
    // must capture it here for out-of-sample shooting-vector computation.
    let horiz = horiz_fpca(&karcher, argvals, ncomp)?;

    // Step 4: Recompute mean_q using the same path as joint_fpca.
    // joint_fpca discards _mean_q (prefixed with _); we must store it for
    // out-of-sample centering.
    let (n_k, m_k) = karcher.aligned_data.shape();
    let m_aug = m_k + 1;
    let qn = match &karcher.aligned_srsfs {
        Some(srsfs) => srsfs.clone(),
        None => srsf_transform(&karcher.aligned_data, argvals),
    };
    let q_aug = build_augmented_srsfs(&qn, &karcher.aligned_data, n_k, m_k);
    let (_, mean_q) = center_matrix(&q_aug, n_k, m_aug);

    // Step 5: Clamped ncomp from the joint_result (joint_fpca clamps to n-1)
    let ncomp_actual = joint_result.eigenvalues.len();

    Ok(JfpcaModel {
        karcher_mean: karcher.mean.clone(),
        mean_q,
        mean_psi: horiz.mean_psi,
        vert_component: joint_result.vert_component.clone(),
        horiz_component: joint_result.horiz_component.clone(),
        balance_c: joint_result.balance_c,
        argvals: argvals.to_vec(),
        eigenvalues: joint_result.eigenvalues.clone(),
        ncomp: ncomp_actual,
        joint_result,
        lambda,
    })
}

impl JfpcaModel {
    /// Project new curves onto the trained joint-FPCA basis.
    ///
    /// Aligns each new curve to the **trained Karcher-mean template** (does NOT
    /// re-run a fresh Karcher mean), centers with the trained `mean_q`, computes
    /// shooting vectors from the trained `mean_psi`, and scores via the exact
    /// dot-product formula derived from the right singular vectors.
    ///
    /// For a training-set round-trip (`model.transform(&training_curves)`), the
    /// returned scores reproduce [`JfpcaModel::joint_result`]`.scores` within
    /// 1e-8 element-wise.
    ///
    /// # Arguments
    ///
    /// * `new_curves` — Curves to project (n_new × m). Must have the same number
    ///   of columns as the training grid (`self.argvals.len()`).
    ///
    /// # Errors
    ///
    /// Returns [`FdarError::InvalidDimension`] when:
    /// - `new_curves.ncols() != self.argvals.len()` (grid mismatch)
    /// - `n_new < 1`
    pub fn transform(&self, new_curves: &FdMatrix) -> Result<JfpcaTransform, FdarError> {
        let (n_new, m_new) = new_curves.shape();
        let m = self.argvals.len();

        // Grid-mismatch check — first validation before any alignment work.
        if m_new != m {
            return Err(FdarError::InvalidDimension {
                parameter: "new_curves columns",
                expected: format!("== {} (trained argvals length)", m),
                actual: format!("{}", m_new),
            });
        }
        if n_new < 1 {
            return Err(FdarError::InvalidDimension {
                parameter: "new_curves rows",
                expected: ">= 1".to_string(),
                actual: format!("{}", n_new),
            });
        }

        // Step 1: Align new curves to the FIXED trained Karcher-mean template.
        // Do NOT run a fresh karcher_mean — that changes the coordinate system.
        let aln = align_to_target(new_curves, &self.karcher_mean, &self.argvals, self.lambda);

        // Step 2: Psi-space shooting vectors for new warps.
        let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let psis = warps_to_normalized_psi(&aln.gammas, &self.argvals);
        let shooting = shooting_vectors_from_psis(&psis, &self.mean_psi, &time);

        // Step 3: Augmented SRSF matrix for the newly aligned curves.
        let qn_new = srsf_transform(&aln.aligned_data, &self.argvals);
        let q_aug = build_augmented_srsfs(&qn_new, &aln.aligned_data, n_new, m);

        // Step 4: Center with the TRAINED mean_q — do NOT recompute from new data.
        let m_aug = m + 1;
        let mut q_aug_centered = q_aug;
        for i in 0..n_new {
            for j in 0..m_aug {
                q_aug_centered[(i, j)] -= self.mean_q[j];
            }
        }

        // Step 5: Score via the verified joint-FPCA dot-product formula (RESEARCH §9).
        // score_i_k = Σ_j q_aug_centered[i,j] * vert_component[k,j]
        //           + balance_c * Σ_j shooting[i,j] * horiz_component[k,j]
        // This is NOT project_onto_eigenvectors (which uses covariance SVD U and
        // gives numerically different results for joint FPCA).
        let scores = self.project_joint(&q_aug_centered, &shooting, n_new);

        Ok(JfpcaTransform {
            scores,
            aligned: aln.aligned_data,
            warping: aln.gammas,
        })
    }

    /// Inner dot-product projection onto the joint right-singular vectors.
    ///
    /// `combined_i = [q_aug_centered_i | balance_c * shooting_i]`
    /// `score_i_k  = dot(q_aug_centered_i, vert_component[k]) + balance_c * dot(shooting_i, horiz_component[k])`
    fn project_joint(&self, q_aug_centered: &FdMatrix, shooting: &FdMatrix, n: usize) -> FdMatrix {
        let m = self.argvals.len();
        let m_aug = m + 1;
        let ncomp = self.ncomp;
        let mut scores = FdMatrix::zeros(n, ncomp);
        for k in 0..ncomp {
            for i in 0..n {
                let mut s = 0.0;
                // Amplitude part: (m+1) dimensions
                for j in 0..m_aug {
                    s += q_aug_centered[(i, j)] * self.vert_component[(k, j)];
                }
                // Phase part: m dimensions scaled by balance_c
                for j in 0..m {
                    s += self.balance_c * shooting[(i, j)] * self.horiz_component[(k, j)];
                }
                scores[(i, k)] = s;
            }
        }
        scores
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    /// Build a spanning multi-frequency fixture.
    ///
    /// Curves are combinations of 3+ harmonics with per-curve random-but-
    /// deterministic amplitudes, ensuring the augmented representation is
    /// effectively full-rank up to ncomp.
    ///
    /// IMPORTANT: Do NOT use the single-freq phase-shifted sinusoid generator
    /// from elastic_fpca::tests — those span only a 2-D subspace and mask bugs.
    fn spanning_fixture(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
        // Deterministic amplitude table: each curve has a distinct combination of
        // 4 harmonics so the design matrix is full rank for any ncomp <= n-1.
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let mut data = FdMatrix::zeros(n, m);
        for i in 0..n {
            let fi = i as f64;
            let a1 = 1.0 + 0.4 * fi;
            let a2 = 0.6 - 0.08 * fi;
            let a3 = 0.35 + 0.07 * fi;
            let a4 = 0.2 - 0.03 * fi;
            for j in 0..m {
                let t = argvals[j];
                data[(i, j)] = a1 * (2.0 * PI * t).sin()
                    + a2 * (4.0 * PI * t).cos()
                    + a3 * (6.0 * PI * t).sin()
                    + a4 * (8.0 * PI * t).cos();
            }
        }
        (data, argvals)
    }

    /// Compute max absolute element-wise difference between two FdMatrices of the
    /// same shape.
    fn max_abs_diff(a: &FdMatrix, b: &FdMatrix) -> f64 {
        let (na, ma) = a.shape();
        let (nb, mb) = b.shape();
        assert_eq!((na, ma), (nb, mb), "shape mismatch in max_abs_diff");
        let mut max_d = 0.0_f64;
        for i in 0..na {
            for j in 0..ma {
                max_d = max_d.max((a[(i, j)] - b[(i, j)]).abs());
            }
        }
        max_d
    }

    /// Tracer test: end-to-end fit -> transform on the spanning fixture.
    /// Verifies the architecture path works (real formula, real error handling).
    /// Numerical precision gates are in Task 2 tests.
    #[test]
    fn tracer() {
        let n = 12;
        let m = 15;
        let ncomp = 4;
        let (data, argvals) = spanning_fixture(n, m);

        let model = jfpca_fit(&data, &argvals, ncomp, None, 0.0, 20)
            .expect("jfpca_fit should succeed on spanning fixture");

        // Transform the training curves through the trained model
        let transform = model
            .transform(&data)
            .expect("transform should succeed on training curves");

        // Basic shape and finite checks
        let (s_rows, s_cols) = transform.scores.shape();
        assert_eq!(s_rows, n, "scores should have n rows");
        assert_eq!(s_cols, model.ncomp, "scores should have ncomp cols");
        assert_eq!(transform.aligned.shape(), (n, m), "aligned shape mismatch");
        assert_eq!(transform.warping.shape(), (n, m), "warping shape mismatch");

        for i in 0..s_rows {
            for j in 0..s_cols {
                assert!(
                    transform.scores[(i, j)].is_finite(),
                    "score [{i},{j}] is not finite"
                );
            }
        }
    }

    /// Gate 1a (VEE-01a): training scores reproduce joint_fpca within 1e-8.
    /// Since jfpca_fit delegates to joint_fpca directly, diff should be ~0.
    #[test]
    fn test_fit_scores_match_joint_fpca() {
        use crate::alignment::karcher_mean;
        use crate::elastic_fpca::joint_fpca;

        let n = 12;
        let m = 15;
        let ncomp = 4;
        let (data, argvals) = spanning_fixture(n, m);

        let model =
            jfpca_fit(&data, &argvals, ncomp, None, 0.0, 20).expect("jfpca_fit should succeed");

        // Independent reference path with the same arguments
        let karcher_ref = karcher_mean(&data, &argvals, 20, 1e-4, 0.0);
        let joint_ref = joint_fpca(&karcher_ref, &argvals, ncomp, None)
            .expect("joint_fpca reference should succeed");

        let diff = max_abs_diff(&model.joint_result.scores, &joint_ref.scores);
        assert!(
            diff < 1e-8,
            "training scores differ from joint_fpca by {diff} (tolerance 1e-8)"
        );
    }

    /// Gate 1b (VEE-01b): all model fields have correct shapes and clamped ncomp.
    #[test]
    fn test_model_fields_populated() {
        let n = 12;
        let m = 15;
        let ncomp = 4;
        let (data, argvals) = spanning_fixture(n, m);

        let model =
            jfpca_fit(&data, &argvals, ncomp, None, 0.0, 20).expect("jfpca_fit should succeed");

        assert_eq!(model.mean_psi.len(), m, "mean_psi should have length m");
        assert_eq!(model.mean_q.len(), m + 1, "mean_q should have length m+1");
        assert_eq!(
            model.vert_component.shape(),
            (model.ncomp, m + 1),
            "vert_component shape mismatch"
        );
        assert_eq!(
            model.horiz_component.shape(),
            (model.ncomp, m),
            "horiz_component shape mismatch"
        );
        assert_eq!(
            model.eigenvalues.len(),
            model.ncomp,
            "eigenvalues length should equal ncomp"
        );
        assert_eq!(model.argvals, argvals, "argvals mismatch");
        // ncomp stored in model must equal joint_result.eigenvalues.len() (clamp respected)
        assert_eq!(
            model.ncomp,
            model.joint_result.eigenvalues.len(),
            "ncomp must equal joint_result.eigenvalues.len() (clamp check)"
        );
        // ncomp must be <= n-1 (clamped)
        assert!(model.ncomp <= n - 1, "ncomp must be clamped to n-1");
    }

    /// Gate 2a (VEE-02a): fit->transform round-trip on training curves within 1e-8.
    ///
    /// Achieved tolerance documented in comment below.
    #[test]
    fn test_roundtrip_training_curves() {
        let n = 12;
        let m = 15;
        let ncomp = 4;
        let (data, argvals) = spanning_fixture(n, m);

        let model =
            jfpca_fit(&data, &argvals, ncomp, None, 0.0, 20).expect("jfpca_fit should succeed");

        // Transform the ORIGINAL training FdMatrix (raw curves, not aligned)
        let transform = model
            .transform(&data)
            .expect("transform should succeed on training curves");

        // Achieved round-trip tolerance: the transform aligns to the trained
        // Karcher mean (same template as fit), uses the exact stored mean_q
        // and mean_psi, and applies the identical dot-product formula.
        // For a spanning multi-harmonic fixture with deterministic alignment
        // this should be within floating-point noise (< 1e-12 typically).
        let diff = max_abs_diff(&transform.scores, &model.joint_result.scores);
        assert!(
            diff < 1e-8,
            "round-trip diff = {diff} exceeds 1e-8; check: \
            (a) dot-product vs project_onto_eigenvectors, \
            (b) trained mean_q vs recomputed mean, \
            (c) align_to_target vs fresh karcher_mean, \
            (d) lambda mismatch"
        );
    }

    /// Gate 2b (VEE-02b): grid-mismatch returns FdarError::InvalidDimension, no panic.
    #[test]
    fn test_transform_grid_mismatch_error() {
        let n = 12;
        let m = 15;
        let ncomp = 4;
        let (data, argvals) = spanning_fixture(n, m);

        let model =
            jfpca_fit(&data, &argvals, ncomp, None, 0.0, 20).expect("jfpca_fit should succeed");

        // Curves with wrong number of columns
        let bad_m = m + 3;
        let wrong_curves = FdMatrix::zeros(n, bad_m);
        let res = model.transform(&wrong_curves);
        assert!(
            matches!(res, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension on grid mismatch, got: {:?}",
            res
        );
    }

    /// Gate 3 (VEE-02b + entry validation): jfpca_fit rejects degenerate inputs.
    #[test]
    fn test_fit_rejects_degenerate() {
        let m = 10;
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

        // argvals.len() != data ncols
        let data_ok = FdMatrix::zeros(3, m);
        let bad_argvals: Vec<f64> = (0..(m + 1)).map(|i| i as f64 / m as f64).collect();
        let res = jfpca_fit(&data_ok, &bad_argvals, 2, None, 0.0, 5);
        assert!(
            matches!(res, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension for argvals length mismatch"
        );

        // ncomp == 0
        let res2 = jfpca_fit(&data_ok, &argvals, 0, None, 0.0, 5);
        assert!(
            matches!(res2, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension for ncomp=0"
        );

        // n < 2
        let data_tiny = FdMatrix::zeros(1, m);
        let res3 = jfpca_fit(&data_tiny, &argvals, 2, None, 0.0, 5);
        assert!(
            matches!(res3, Err(FdarError::InvalidDimension { .. })),
            "expected InvalidDimension for n<2"
        );
    }
}
