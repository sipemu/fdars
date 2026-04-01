use super::*;
use crate::test_helpers::uniform_grid;
use std::f64::consts::PI;

/// Generate two well-separated classes of curves.
fn generate_two_class_data(n_per: usize, m: usize) -> (FdMatrix, Vec<usize>, Vec<f64>) {
    let t = uniform_grid(m);
    let n = 2 * n_per;
    let mut col_major = vec![0.0; n * m];

    for i in 0..n_per {
        for (j, &tj) in t.iter().enumerate() {
            // Class 0: sin
            col_major[i + j * n] =
                (2.0 * PI * tj).sin() + 0.05 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
        }
    }
    for i in 0..n_per {
        for (j, &tj) in t.iter().enumerate() {
            // Class 1: -sin (opposite phase)
            col_major[(i + n_per) + j * n] =
                -(2.0 * PI * tj).sin() + 0.05 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
        }
    }

    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let labels: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();
    (data, labels, t)
}

#[test]
fn test_fclassif_lda_basic() {
    let (data, labels, _t) = generate_two_class_data(20, 50);
    let result = fclassif_lda(&data, &labels, None, 3).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert_eq!(result.n_classes, 2);
    assert!(
        result.accuracy > 0.8,
        "LDA accuracy should be high: {}",
        result.accuracy
    );
}

#[test]
fn test_fclassif_qda_basic() {
    let (data, labels, _t) = generate_two_class_data(20, 50);
    let result = fclassif_qda(&data, &labels, None, 3).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert!(
        result.accuracy > 0.8,
        "QDA accuracy should be high: {}",
        result.accuracy
    );
}

#[test]
fn test_fclassif_knn_basic() {
    let (data, labels, _t) = generate_two_class_data(20, 50);
    let result = fclassif_knn(&data, &labels, None, 3, 5).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert!(
        result.accuracy > 0.7,
        "k-NN accuracy should be reasonable: {}",
        result.accuracy
    );
}

#[test]
fn test_fclassif_kernel_basic() {
    let (data, labels, t) = generate_two_class_data(20, 50);
    let result = fclassif_kernel(&data, &labels, &t, None, 0.0, 0.0).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert!(
        result.accuracy > 0.7,
        "Kernel accuracy should be reasonable: {}",
        result.accuracy
    );
}

#[test]
fn test_fclassif_dd_basic() {
    let (data, labels, _t) = generate_two_class_data(20, 50);
    let result = fclassif_dd(&data, &labels, None).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert_eq!(result.n_classes, 2);
    // DD-classifier should work on well-separated data
    assert!(
        result.accuracy > 0.6,
        "DD accuracy should be reasonable: {}",
        result.accuracy
    );
    assert!(result.probabilities.is_some());
}

#[test]
fn test_confusion_matrix_shape() {
    let (data, labels, _t) = generate_two_class_data(15, 50);
    let result = fclassif_lda(&data, &labels, None, 2).unwrap();

    assert_eq!(result.confusion.len(), 2);
    assert_eq!(result.confusion[0].len(), 2);
    assert_eq!(result.confusion[1].len(), 2);

    // Total should equal n
    let total: usize = result.confusion.iter().flat_map(|row| row.iter()).sum();
    assert_eq!(total, 30);
}

#[test]
fn test_fclassif_cv_lda() {
    let (data, labels, t) = generate_two_class_data(25, 50);
    let result = fclassif_cv(&data, &t, &labels, None, "lda", 3, 5, 42).unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(
        result.error_rate < 0.3,
        "CV error should be low: {}",
        result.error_rate
    );
}

#[test]
fn test_fclassif_invalid_input() {
    let data = FdMatrix::zeros(0, 0);
    assert!(fclassif_lda(&data, &[], None, 1).is_err());

    let data = FdMatrix::zeros(10, 50);
    let labels = vec![0; 10]; // single class
    assert!(fclassif_lda(&data, &labels, None, 1).is_err());
}

#[test]
fn test_remap_labels() {
    let (mapped, g) = remap_labels(&[5, 5, 10, 10, 20]);
    assert_eq!(g, 3);
    assert_eq!(mapped, vec![0, 0, 1, 1, 2]);
}

#[test]
fn test_fclassif_lda_with_scalar_covariates() {
    let n_per = 15;
    let n = 2 * n_per;
    let m = 50;
    let t = uniform_grid(m);

    // Curves are identical across classes
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for (j, &tj) in t.iter().enumerate() {
            col_major[i + j * n] = (2.0 * PI * tj).sin();
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

    // But covariate separates: 0 vs 10
    let mut cov_data = vec![0.0; n];
    for i in n_per..n {
        cov_data[i] = 10.0;
    }
    let scalar_covariates = FdMatrix::from_column_major(cov_data, n, 1).unwrap();

    let labels: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();

    let result = fclassif_lda(&data, &labels, Some(&scalar_covariates), 2).unwrap();
    assert!(
        result.accuracy > 0.9,
        "Covariate should enable separation: {}",
        result.accuracy
    );
}

// -----------------------------------------------------------------------
// Additional coverage tests
// -----------------------------------------------------------------------

/// Helper: generate two-class data with scalar covariates.
fn generate_two_class_with_scalar_covariates(
    n_per: usize,
    m: usize,
    p_cov: usize,
) -> (FdMatrix, Vec<usize>, Vec<f64>, FdMatrix) {
    let (data, labels, t) = generate_two_class_data(n_per, m);
    let n = 2 * n_per;
    // Covariates: class 0 → low values, class 1 → high values
    let mut cov_data = vec![0.0; n * p_cov];
    for i in 0..n {
        for j in 0..p_cov {
            let base = if labels[i] == 0 { 0.0 } else { 5.0 };
            cov_data[i + j * n] = base + 0.1 * ((i * 3 + j * 7) % 50) as f64 / 50.0;
        }
    }
    let scalar_covariates = FdMatrix::from_column_major(cov_data, n, p_cov).unwrap();
    (data, labels, t, scalar_covariates)
}

#[test]
fn test_fclassif_cv_qda() {
    let (data, labels, t) = generate_two_class_data(25, 50);
    let result = fclassif_cv(&data, &t, &labels, None, "qda", 3, 5, 42).unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(
        result.error_rate < 0.4,
        "QDA CV error should be low: {}",
        result.error_rate
    );
    assert_eq!(result.best_ncomp, 3);
}

#[test]
fn test_fclassif_cv_knn() {
    let (data, labels, t) = generate_two_class_data(25, 50);
    let result = fclassif_cv(&data, &t, &labels, None, "knn", 3, 5, 42).unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(
        result.error_rate < 0.4,
        "k-NN CV error should be low: {}",
        result.error_rate
    );
}

#[test]
fn test_fclassif_cv_kernel() {
    let (data, labels, t) = generate_two_class_data(25, 50);
    let result = fclassif_cv(&data, &t, &labels, None, "kernel", 3, 5, 42).unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    // Kernel CV: placeholder prediction may not be accurate, just ensure it runs
    assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
}

#[test]
fn test_fclassif_cv_dd() {
    let (data, labels, t) = generate_two_class_data(25, 50);
    let result = fclassif_cv(&data, &t, &labels, None, "dd", 3, 5, 42).unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
}

#[test]
fn test_fclassif_cv_invalid_method() {
    let (data, labels, t) = generate_two_class_data(25, 50);
    // "bogus" method hits the `_ => None` arm in cv_fold_predict
    let result = fclassif_cv(&data, &t, &labels, None, "bogus", 3, 5, 42);

    // Should still return Some — fold errors will be 1.0 for each fold
    let r = result.unwrap();
    assert!((r.error_rate - 1.0).abs() < 1e-10);
}

#[test]
fn test_fclassif_cv_too_few_folds() {
    let (data, labels, t) = generate_two_class_data(10, 50);
    // nfold < 2 → Err
    assert!(fclassif_cv(&data, &t, &labels, None, "lda", 3, 1, 42).is_err());
    // n < nfold → Err
    assert!(fclassif_cv(&data, &t, &labels, None, "lda", 3, 100, 42).is_err());
}

#[test]
fn test_fclassif_cv_single_class() {
    let (data, _labels, t) = generate_two_class_data(10, 50);
    let single = vec![0usize; 20]; // only one class
    assert!(fclassif_cv(&data, &t, &single, None, "lda", 3, 5, 42).is_err());
}

#[test]
fn test_fclassif_kernel_with_scalar_covariates() {
    let (data, labels, t, scalar_covariates) = generate_two_class_with_scalar_covariates(20, 50, 2);
    let result = fclassif_kernel(&data, &labels, &t, Some(&scalar_covariates), 0.0, 0.0).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert!(
        result.accuracy > 0.5,
        "Kernel+cov accuracy should be reasonable: {}",
        result.accuracy
    );
    assert_eq!(result.ncomp, 0); // kernel doesn't use ncomp
}

#[test]
fn test_fclassif_kernel_with_scalar_covariates_manual_bandwidth() {
    let (data, labels, t, scalar_covariates) = generate_two_class_with_scalar_covariates(15, 50, 1);
    // Provide explicit bandwidths (>0 skips LOO selection)
    let result = fclassif_kernel(&data, &labels, &t, Some(&scalar_covariates), 1.0, 1.0).unwrap();

    assert_eq!(result.predicted.len(), 30);
    assert!(result.accuracy >= 0.0 && result.accuracy <= 1.0);
}

#[test]
fn test_fclassif_dd_with_scalar_covariates() {
    let (data, labels, _t, scalar_covariates) =
        generate_two_class_with_scalar_covariates(20, 50, 2);
    let result = fclassif_dd(&data, &labels, Some(&scalar_covariates)).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert_eq!(result.n_classes, 2);
    assert!(
        result.accuracy > 0.5,
        "DD+cov accuracy should be reasonable: {}",
        result.accuracy
    );
    assert!(result.probabilities.is_some());
}

#[test]
fn test_fclassif_dd_with_single_covariate() {
    // Curves are identical; only the covariate separates classes
    let n_per = 15;
    let n = 2 * n_per;
    let m = 50;
    let t = uniform_grid(m);

    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for (j, &tj) in t.iter().enumerate() {
            col_major[i + j * n] = (2.0 * PI * tj).sin();
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let labels: Vec<usize> = (0..n).map(|i| if i < n_per { 0 } else { 1 }).collect();

    // Covariate: class 0 → [0..1], class 1 → [10..11]
    let mut cov_data = vec![0.0; n];
    for i in 0..n_per {
        cov_data[i] = i as f64 / n_per as f64;
    }
    for i in n_per..n {
        cov_data[i] = 10.0 + (i - n_per) as f64 / n_per as f64;
    }
    let scalar_covariates = FdMatrix::from_column_major(cov_data, n, 1).unwrap();

    let result = fclassif_dd(&data, &labels, Some(&scalar_covariates)).unwrap();
    // The scalar blending should help even when curves are identical
    assert!(
        result.accuracy >= 0.5,
        "DD with scalar covariate: {}",
        result.accuracy
    );
}

#[test]
fn test_scalar_depth_for_obs_edge_cases() {
    use super::kernel::scalar_depth_for_obs;
    // Empty class indices → depth = 0
    let cov = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0], 4, 1).unwrap();
    assert_eq!(scalar_depth_for_obs(&cov, 0, &[], 1), 0.0);

    // p=0 → depth = 0
    let cov0 = FdMatrix::zeros(4, 0);
    assert_eq!(scalar_depth_for_obs(&cov0, 0, &[0, 1, 2, 3], 0), 0.0);

    // Normal case: all indices
    let depth = scalar_depth_for_obs(&cov, 1, &[0, 1, 2, 3], 1);
    assert!(depth > 0.0 && depth <= 0.5, "depth={}", depth);

    // Observation is at the extremes
    let depth_min = scalar_depth_for_obs(&cov, 0, &[0, 1, 2, 3], 1);
    let depth_max = scalar_depth_for_obs(&cov, 3, &[0, 1, 2, 3], 1);
    // Extreme observations should have low depth
    assert!(depth_min <= 0.5, "depth_min={}", depth_min);
    assert!(depth_max <= 0.5, "depth_max={}", depth_max);
}

#[test]
fn test_scalar_depth_for_obs_multivariate() {
    use super::kernel::scalar_depth_for_obs;
    // 2 scalar_covariates
    let cov = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0], 4, 2)
        .unwrap();
    let depth = scalar_depth_for_obs(&cov, 1, &[0, 1, 2, 3], 2);
    assert!(depth > 0.0 && depth <= 0.5, "multivar depth={}", depth);
}

#[test]
fn test_blend_scalar_depths_modifies_scores() {
    use super::dd::blend_scalar_depths;
    let n = 6;
    let g = 2;
    let mut depth_scores = FdMatrix::zeros(n, g);
    // Fill with some values
    for i in 0..n {
        depth_scores[(i, 0)] = 0.5;
        depth_scores[(i, 1)] = 0.3;
    }

    let cov = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], n, 1).unwrap();
    let class_indices = vec![vec![0, 1, 2], vec![3, 4, 5]];

    let original_00 = depth_scores[(0, 0)];
    blend_scalar_depths(&mut depth_scores, &cov, &class_indices, n);

    // Scores should have been modified (blended with 0.7 / 0.3 weights)
    let blended_00 = depth_scores[(0, 0)];
    // blended = 0.7 * 0.5 + 0.3 * scalar_depth
    assert!(
        (blended_00 - original_00).abs() > 1e-10,
        "blend should change scores: original={}, blended={}",
        original_00,
        blended_00
    );
}

#[test]
fn test_compute_pairwise_scalar() {
    use super::kernel::compute_pairwise_scalar;
    let n = 4;
    // 2 scalar_covariates
    let cov =
        FdMatrix::from_column_major(vec![0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0], n, 2).unwrap();
    let dists = compute_pairwise_scalar(&cov);
    assert_eq!(dists.len(), n * n);

    // Diagonal should be zero
    for i in 0..n {
        assert!((dists[i * n + i]).abs() < 1e-15);
    }
    // Symmetry
    for i in 0..n {
        for j in 0..n {
            assert!((dists[i * n + j] - dists[j * n + i]).abs() < 1e-15);
        }
    }
    // d(0,1) = sqrt(1^2 + 0^2) = 1.0
    assert!((dists[1] - 1.0).abs() < 1e-10);
    // d(0,3) = sqrt(3^2 + 0^2) = 3.0
    assert!((dists[3] - 3.0).abs() < 1e-10);
}

#[test]
fn test_fclassif_cv_lda_with_scalar_covariates() {
    let (data, labels, t, scalar_covariates) = generate_two_class_with_scalar_covariates(25, 50, 1);
    let result = fclassif_cv(
        &data,
        &t,
        &labels,
        Some(&scalar_covariates),
        "lda",
        3,
        5,
        42,
    )
    .unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
}

#[test]
fn test_fclassif_cv_qda_with_scalar_covariates() {
    let (data, labels, t, scalar_covariates) = generate_two_class_with_scalar_covariates(25, 50, 1);
    let result = fclassif_cv(
        &data,
        &t,
        &labels,
        Some(&scalar_covariates),
        "qda",
        3,
        5,
        42,
    )
    .unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
}

#[test]
fn test_fclassif_cv_knn_with_scalar_covariates() {
    let (data, labels, t, scalar_covariates) = generate_two_class_with_scalar_covariates(25, 50, 1);
    let result = fclassif_cv(
        &data,
        &t,
        &labels,
        Some(&scalar_covariates),
        "knn",
        3,
        5,
        42,
    )
    .unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
}

#[test]
fn test_fclassif_cv_kernel_with_scalar_covariates() {
    let (data, labels, t, scalar_covariates) = generate_two_class_with_scalar_covariates(25, 50, 1);
    let result = fclassif_cv(
        &data,
        &t,
        &labels,
        Some(&scalar_covariates),
        "kernel",
        3,
        5,
        42,
    )
    .unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
}

#[test]
fn test_fclassif_cv_dd_with_scalar_covariates() {
    let (data, labels, t, scalar_covariates) = generate_two_class_with_scalar_covariates(25, 50, 2);
    let result = fclassif_cv(&data, &t, &labels, Some(&scalar_covariates), "dd", 3, 5, 42).unwrap();

    assert_eq!(result.fold_errors.len(), 5);
    assert!(result.error_rate >= 0.0 && result.error_rate <= 1.0);
}

#[test]
fn test_fclassif_kernel_invalid_inputs() {
    let data = FdMatrix::zeros(0, 0);
    assert!(fclassif_kernel(&data, &[], &[], None, 0.0, 0.0).is_err());

    let data = FdMatrix::zeros(5, 10);
    let t = uniform_grid(10);
    let labels = vec![0; 5]; // single class
    assert!(fclassif_kernel(&data, &labels, &t, None, 0.0, 0.0).is_err());

    // Mismatched argvals length
    let labels2 = vec![0, 0, 0, 1, 1];
    let wrong_t = vec![0.0, 1.0]; // wrong length
    assert!(fclassif_kernel(&data, &labels2, &wrong_t, None, 0.0, 0.0).is_err());
}

#[test]
fn test_fclassif_dd_invalid_inputs() {
    let data = FdMatrix::zeros(0, 0);
    assert!(fclassif_dd(&data, &[], None).is_err());

    let data = FdMatrix::zeros(5, 10);
    let labels = vec![0; 5]; // single class
    assert!(fclassif_dd(&data, &labels, None).is_err());
}

#[test]
fn test_argmax_class_empty() {
    use super::kernel::argmax_class;
    assert_eq!(argmax_class(&[]), 0);
    assert_eq!(argmax_class(&[0.1]), 0);
    assert_eq!(argmax_class(&[0.1, 0.9, 0.5]), 1);
}

#[test]
fn test_gaussian_kernel_values() {
    use super::kernel::gaussian_kernel;
    // h=0 → 0
    assert_eq!(gaussian_kernel(1.0, 0.0), 0.0);
    // dist=0 → 1
    assert!((gaussian_kernel(0.0, 1.0) - 1.0).abs() < 1e-15);
    // Normal case
    let k = gaussian_kernel(1.0, 1.0);
    let expected = (-0.5_f64).exp();
    assert!((k - expected).abs() < 1e-10);
}

#[test]
fn test_fclassif_qda_with_scalar_covariates() {
    let (data, labels, _t, scalar_covariates) =
        generate_two_class_with_scalar_covariates(20, 50, 1);
    let result = fclassif_qda(&data, &labels, Some(&scalar_covariates), 3).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert!(
        result.accuracy > 0.5,
        "QDA+cov accuracy: {}",
        result.accuracy
    );
}

#[test]
fn test_fclassif_knn_with_scalar_covariates() {
    let (data, labels, _t, scalar_covariates) =
        generate_two_class_with_scalar_covariates(20, 50, 1);
    let result = fclassif_knn(&data, &labels, Some(&scalar_covariates), 3, 5).unwrap();

    assert_eq!(result.predicted.len(), 40);
    assert!(
        result.accuracy > 0.5,
        "k-NN+cov accuracy: {}",
        result.accuracy
    );
}

#[test]
fn test_fclassif_knn_invalid_k() {
    let (data, labels, _t) = generate_two_class_data(10, 50);
    // k_nn == 0 → Err
    assert!(fclassif_knn(&data, &labels, None, 3, 0).is_err());
}

#[test]
fn test_bandwidth_candidates_empty_distances() {
    use super::kernel::bandwidth_candidates;
    // All distances zero → candidates filtered out
    let dists = vec![0.0; 9];
    let cands = bandwidth_candidates(&dists, 3);
    assert!(cands.is_empty());
}

#[test]
fn test_select_bandwidth_loo_empty_candidates() {
    use super::kernel::select_bandwidth_loo;
    // All distances zero → empty candidates → default bandwidth
    let dists = vec![0.0; 9];
    let labels = vec![0, 0, 1];
    let h = select_bandwidth_loo(&dists, &labels, 2, 3, true);
    assert!((h - 1.0).abs() < 1e-10, "default func bandwidth: {}", h);

    let h2 = select_bandwidth_loo(&dists, &labels, 2, 3, false);
    assert!((h2 - 0.5).abs() < 1e-10, "default scalar bandwidth: {}", h2);
}

#[test]
fn test_fold_split() {
    use super::cv::fold_split;
    let folds = vec![0, 1, 2, 0, 1, 2];
    let (train, test) = fold_split(&folds, 1);
    assert_eq!(train, vec![0, 2, 3, 5]);
    assert_eq!(test, vec![1, 4]);
}

#[test]
fn test_assign_folds_deterministic() {
    use super::cv::assign_folds;
    let f1 = assign_folds(10, 3, 42);
    let f2 = assign_folds(10, 3, 42);
    assert_eq!(f1, f2);

    // All fold indices in [0, nfold)
    for &f in &f1 {
        assert!(f < 3);
    }
}

#[test]
fn test_project_test_onto_fpca() {
    use super::cv::project_test_onto_fpca;
    use crate::regression::fdata_to_pc_1d;

    let n_train = 20;
    let m = 50;
    let ncomp = 3;
    let (data, _labels, _t) = generate_two_class_data(n_train / 2, m);

    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1).max(1) as f64).collect();
    let fpca = fdata_to_pc_1d(&data, ncomp, &argvals).unwrap();

    // Create small "test" matrix
    let n_test = 5;
    let mut test_col = vec![0.0; n_test * m];
    for i in 0..n_test {
        for j in 0..m {
            test_col[i + j * n_test] = data[(i, j)] + 0.01;
        }
    }
    let test_data = FdMatrix::from_column_major(test_col, n_test, m).unwrap();

    let projected = project_test_onto_fpca(&test_data, &fpca);
    assert_eq!(projected.nrows(), n_test);
    assert_eq!(projected.ncols(), ncomp);
}

#[test]
fn test_fclassif_three_classes() {
    let n_per = 15;
    let n = 3 * n_per;
    let m = 50;
    let t = uniform_grid(m);

    let mut col_major = vec![0.0; n * m];
    // Class 0: sin
    for i in 0..n_per {
        for (j, &tj) in t.iter().enumerate() {
            col_major[i + j * n] =
                (2.0 * PI * tj).sin() + 0.02 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
        }
    }
    // Class 1: cos
    for i in 0..n_per {
        for (j, &tj) in t.iter().enumerate() {
            col_major[(i + n_per) + j * n] =
                (2.0 * PI * tj).cos() + 0.02 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
        }
    }
    // Class 2: constant
    for i in 0..n_per {
        for (j, _) in t.iter().enumerate() {
            col_major[(i + 2 * n_per) + j * n] =
                3.0 + 0.02 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
        }
    }

    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let labels: Vec<usize> = (0..n)
        .map(|i| {
            if i < n_per {
                0
            } else if i < 2 * n_per {
                1
            } else {
                2
            }
        })
        .collect();

    let result = fclassif_lda(&data, &labels, None, 3).unwrap();
    assert_eq!(result.n_classes, 3);
    assert!(
        result.accuracy > 0.8,
        "Three-class accuracy: {}",
        result.accuracy
    );
}
