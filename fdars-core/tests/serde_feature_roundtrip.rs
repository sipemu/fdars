//! Serde round-trip regression test.
//!
//! Ensures that `ClassifFit` (and transitively `ClassifMethod`, `ClassifResult`)
//! compile under the `serde` feature and that a constructed value survives a
//! `serde_json::to_string` → `serde_json::from_str` round-trip with field-level
//! equality.  JSON represents f64 with limited decimal digits (up to ~17), so
//! the comparison uses a tight relative tolerance (1e-12) for floating-point
//! fields — matching the convention in `src/spm/phase.rs:777`.
//! A future missing-derive gap will cause a compilation failure in this file.

#[cfg(feature = "serde")]
mod serde_roundtrip {
    use fdars_core::classification::{fclassif_lda_fit, ClassifFit, ClassifMethod};
    use fdars_core::matrix::FdMatrix;
    use std::f64::consts::PI;

    /// Construct a deterministic 2-class functional dataset (n×m).
    fn make_two_class_data(n_per_class: usize, m: usize) -> (FdMatrix, Vec<usize>) {
        let n = 2 * n_per_class;
        let mut data = FdMatrix::zeros(n, m);
        let mut labels = vec![0usize; n];
        let t: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
        // Class 0: sine curves.
        for i in 0..n_per_class {
            let phase = i as f64 * 0.3;
            for j in 0..m {
                data[(i, j)] = (2.0 * PI * t[j] + phase).sin();
            }
            labels[i] = 0;
        }
        // Class 1: cosine curves offset to ensure clean separation.
        for i in 0..n_per_class {
            let phase = i as f64 * 0.3;
            for j in 0..m {
                data[(n_per_class + i, j)] = (2.0 * PI * t[j] + phase).cos() + 3.0;
            }
            labels[n_per_class + i] = 1;
        }
        (data, labels)
    }

    fn vec_approx_eq(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len(), "vector length mismatch");
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (ai - bi).abs() <= tol,
                "element {i} differs: {ai} vs {bi} (diff={diff})",
                diff = (ai - bi).abs()
            );
        }
    }

    fn fdmatrix_approx_eq(a: &FdMatrix, b: &FdMatrix, tol: f64) {
        assert_eq!(a.nrows(), b.nrows(), "nrows mismatch");
        assert_eq!(a.ncols(), b.ncols(), "ncols mismatch");
        for r in 0..a.nrows() {
            for c in 0..a.ncols() {
                let ai = a[(r, c)];
                let bi = b[(r, c)];
                assert!(
                    (ai - bi).abs() <= tol,
                    "FdMatrix[{r},{c}] differs: {ai} vs {bi} (diff={diff})",
                    diff = (ai - bi).abs()
                );
            }
        }
    }

    #[test]
    fn classiffit_serde_roundtrip() {
        let (data, labels) = make_two_class_data(10, 20);
        let fit = fclassif_lda_fit(&data, &labels, None, 2).expect("LDA fit failed");

        let json = serde_json::to_string(&fit).expect("serialization failed");
        let roundtrip: ClassifFit = serde_json::from_str(&json).expect("deserialization failed");

        // Structural / integer fields must be exactly equal.
        assert_eq!(fit.ncomp, roundtrip.ncomp, "ncomp mismatch");
        assert_eq!(
            fit.result.predicted, roundtrip.result.predicted,
            "predicted labels mismatch"
        );
        assert_eq!(
            fit.result.n_classes, roundtrip.result.n_classes,
            "n_classes mismatch"
        );
        assert_eq!(
            fit.result.ncomp, roundtrip.result.ncomp,
            "result.ncomp mismatch"
        );
        assert_eq!(
            fit.result.confusion, roundtrip.result.confusion,
            "confusion matrix mismatch"
        );
        assert_eq!(
            fit.result.probabilities.is_none(),
            roundtrip.result.probabilities.is_none()
        );

        // Floating-point fields: tight tolerance (JSON has ~15-16 significant digits).
        const TOL: f64 = 1e-12;
        assert!(
            (fit.result.accuracy - roundtrip.result.accuracy).abs() <= TOL,
            "accuracy mismatch: {} vs {}",
            fit.result.accuracy,
            roundtrip.result.accuracy
        );
        vec_approx_eq(&fit.fpca_mean, &roundtrip.fpca_mean, TOL);
        vec_approx_eq(&fit.fpca_int_weights, &roundtrip.fpca_int_weights, TOL);
        fdmatrix_approx_eq(&fit.fpca_rotation, &roundtrip.fpca_rotation, TOL);
        fdmatrix_approx_eq(&fit.fpca_scores, &roundtrip.fpca_scores, TOL);

        // ClassifMethod discriminant must match.
        match (&fit.method, &roundtrip.method) {
            (
                ClassifMethod::Lda {
                    n_classes: nc_a,
                    priors: p_a,
                    ..
                },
                ClassifMethod::Lda {
                    n_classes: nc_b,
                    priors: p_b,
                    ..
                },
            ) => {
                assert_eq!(nc_a, nc_b, "LDA n_classes mismatch");
                vec_approx_eq(p_a, p_b, TOL);
            }
            _ => panic!(
                "ClassifMethod variant changed after round-trip: {:?} vs {:?}",
                fit.method, roundtrip.method
            ),
        }
    }
}
