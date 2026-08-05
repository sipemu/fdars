use super::cluster::run_multiple_inits;
use super::em::count_params;
use super::init::build_features;
use super::*;
use crate::basis::projection::ProjectionBasisType;
use crate::matrix::FdMatrix;
use crate::test_helpers::uniform_grid;
use rand::prelude::*;
use std::f64::consts::PI;

/// Generate two clearly separated clusters of curves.
fn generate_two_clusters(n_per: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let t = uniform_grid(m);
    let n = 2 * n_per;
    let mut col_major = vec![0.0; n * m];

    for i in 0..n_per {
        for (j, &tj) in t.iter().enumerate() {
            // Cluster 0: sin with small noise
            col_major[i + j * n] =
                (2.0 * PI * tj).sin() + 0.05 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
        }
    }
    for i in 0..n_per {
        for (j, &tj) in t.iter().enumerate() {
            // Cluster 1: shifted sin
            col_major[(i + n_per) + j * n] =
                (2.0 * PI * tj).sin() + 5.0 + 0.05 * ((i * 7 + j * 3) % 100) as f64 / 100.0;
        }
    }
    (FdMatrix::from_column_major(col_major, n, m).unwrap(), t)
}

#[test]
fn test_gmm_em_basic() {
    let (data, t) = generate_two_clusters(15, 50);
    let features = build_features(&data, &t, None, 8, ProjectionBasisType::Bspline, 1.0)
        .expect("Feature extraction failed");
    let result = gmm_em(&features.0, 2, CovType::Full, 100, 1e-6, 42).unwrap();

    assert_eq!(result.cluster.len(), 30);
    assert_eq!(result.k, 2);
    assert!(result.iterations > 0);
}

#[test]
fn test_gmm_em_finds_clusters() {
    // Test on synthetic Gaussian features directly (bypasses basis projection)
    let n_per = 30;
    let n = 2 * n_per;
    let mut features = Vec::with_capacity(n);

    // Cluster 0: mean = [0, 0, 0]
    for i in 0..n_per {
        let noise = (i as f64 * 0.1).sin() * 0.3;
        features.push(vec![noise, noise * 0.5, -noise * 0.7]);
    }
    // Cluster 1: mean = [5, 5, 5]
    for i in 0..n_per {
        let noise = (i as f64 * 0.1).sin() * 0.3;
        features.push(vec![5.0 + noise, 5.0 + noise * 0.5, 5.0 - noise * 0.7]);
    }

    let result = gmm_em(&features, 2, CovType::Diagonal, 200, 1e-6, 42).unwrap();

    let c0 = result.cluster[0];
    let c1 = result.cluster[n_per];
    assert_ne!(c0, c1, "Two clusters should be separated");

    let correct_first = (0..n_per).filter(|&i| result.cluster[i] == c0).count();
    let correct_second = (n_per..2 * n_per)
        .filter(|&i| result.cluster[i] == c1)
        .count();
    assert!(
        correct_first >= n_per - 1,
        "First cluster mostly correct: {}/{}",
        correct_first,
        n_per
    );
    assert!(
        correct_second >= n_per - 1,
        "Second cluster mostly correct: {}/{}",
        correct_second,
        n_per
    );
}

#[test]
fn test_gmm_em_diagonal_covariance() {
    let (data, t) = generate_two_clusters(15, 50);
    let (features, _d) =
        build_features(&data, &t, None, 8, ProjectionBasisType::Bspline, 1.0).unwrap();

    let result = gmm_em(&features, 2, CovType::Diagonal, 100, 1e-6, 42).unwrap();
    assert_eq!(result.cluster.len(), 30);

    // Diagonal should have fewer parameters → BIC advantage for simpler models
    let result_full = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();
    // Both should find clusters
    assert_eq!(result_full.cluster.len(), 30);
}

#[test]
fn test_gmm_membership_sums_to_one() {
    let (data, t) = generate_two_clusters(10, 50);
    let (features, _d) =
        build_features(&data, &t, None, 6, ProjectionBasisType::Bspline, 1.0).unwrap();

    let result = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();

    let n = result.membership.nrows();
    let k = result.membership.ncols();
    for i in 0..n {
        let sum: f64 = (0..k).map(|c| result.membership[(i, c)]).sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Membership should sum to 1, got {}",
            sum
        );
    }
}

#[test]
fn test_gmm_bic_icl_finite() {
    let (data, t) = generate_two_clusters(10, 50);
    let (features, _d) =
        build_features(&data, &t, None, 6, ProjectionBasisType::Bspline, 1.0).unwrap();

    let result = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();

    assert!(result.bic.is_finite(), "BIC should be finite");
    assert!(result.icl.is_finite(), "ICL should be finite");
    assert!(
        result.icl >= result.bic,
        "ICL >= BIC (ICL adds entropy penalty)"
    );
}

#[test]
fn test_gmm_cluster_auto_k() {
    // Use pseudo-random Gaussian noise via Box-Muller
    let n_per = 50;
    let n = 2 * n_per;
    let mut features = Vec::with_capacity(n);
    let mut rng = StdRng::seed_from_u64(99);

    for _ in 0..n_per {
        let u1: f64 = rng.gen::<f64>().max(1e-15);
        let u2: f64 = rng.gen::<f64>();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
        features.push(vec![z1 * 0.5, z2 * 0.5]);
    }
    for _ in 0..n_per {
        let u1: f64 = rng.gen::<f64>().max(1e-15);
        let u2: f64 = rng.gen::<f64>();
        let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        let z2 = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).sin();
        features.push(vec![6.0 + z1 * 0.5, 6.0 + z2 * 0.5]);
    }

    let mut best_bic = f64::INFINITY;
    let mut best_k = 0;
    let mut bic_values = Vec::new();
    for k in 1..=4 {
        let r = run_multiple_inits(&features, k, CovType::Diagonal, 100, 1e-6, 3, 42).unwrap();
        bic_values.push((k, r.bic));
        if r.bic < best_bic {
            best_bic = r.bic;
            best_k = k;
        }
    }

    assert_eq!(bic_values.len(), 4);
    assert_eq!(
        best_k, 2,
        "Should select K=2 for well-separated data, BICs: {:?}",
        bic_values
    );
}

#[test]
fn test_gmm_with_covariates() {
    // Test that appending a covariate dimension to features separates clusters
    let n_per = 25;
    let n = 2 * n_per;

    // Base features identical for both groups, covariate separates them
    let mut features = Vec::with_capacity(n);
    for i in 0..n_per {
        let noise = (i as f64 * 0.1).sin() * 0.1;
        features.push(vec![noise, noise * 0.5, 0.0]); // covariate = 0
    }
    for i in 0..n_per {
        let noise = (i as f64 * 0.1).sin() * 0.1;
        features.push(vec![noise, noise * 0.5, 10.0]); // covariate = 10
    }

    let result = gmm_em(&features, 2, CovType::Diagonal, 100, 1e-6, 42).unwrap();
    assert_eq!(result.cluster.len(), n);

    let c0 = result.cluster[0];
    let correct = (0..n_per).filter(|&i| result.cluster[i] == c0).count()
        + (n_per..n).filter(|&i| result.cluster[i] != c0).count();
    assert!(
        correct >= n - 2,
        "Covariate-based separation: {}/{} correct",
        correct,
        n
    );
}

#[test]
fn test_predict_gmm() {
    let n_per = 15;
    let (data, t) = generate_two_clusters(n_per, 50);
    let nbasis = 8;
    let basis_type = ProjectionBasisType::Bspline;

    let result = gmm_cluster(
        &data,
        &t,
        None,
        &[2],
        nbasis,
        basis_type,
        CovType::Diagonal,
        1.0,
        100,
        1e-6,
        1,
        42,
        false,
    )
    .unwrap();

    // Predict on training data — should mostly match
    let (pred_cluster, pred_mem) = predict_gmm(
        &data,
        &t,
        None,
        &result.best,
        nbasis,
        basis_type,
        1.0,
        CovType::Diagonal,
    )
    .unwrap();

    assert_eq!(pred_cluster.len(), 2 * n_per);
    assert_eq!(pred_mem.nrows(), 2 * n_per);
    assert_eq!(pred_mem.ncols(), 2);

    let matching: usize = pred_cluster
        .iter()
        .zip(&result.best.cluster)
        .filter(|(&a, &b)| a == b)
        .count();
    assert!(
        matching >= 2 * n_per - 3,
        "Predict on training data should mostly match: {}/{}",
        matching,
        2 * n_per
    );
}

#[test]
fn test_gmm_em_invalid_input() {
    let empty: Vec<Vec<f64>> = Vec::new();
    assert!(gmm_em(&empty, 2, CovType::Full, 100, 1e-6, 42).is_err());

    let features = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    // k > n
    assert!(gmm_em(&features, 5, CovType::Full, 100, 1e-6, 42).is_err());
    // k = 0
    assert!(gmm_em(&features, 0, CovType::Full, 100, 1e-6, 42).is_err());
}

#[test]
fn test_gmm_deterministic() {
    let (data, t) = generate_two_clusters(10, 50);
    let (features, _d) =
        build_features(&data, &t, None, 6, ProjectionBasisType::Bspline, 1.0).unwrap();

    let r1 = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();
    let r2 = gmm_em(&features, 2, CovType::Full, 100, 1e-6, 42).unwrap();

    assert_eq!(r1.cluster, r2.cluster);
    assert!((r1.log_likelihood - r2.log_likelihood).abs() < 1e-10);
}

#[test]
fn test_count_params() {
    // K=2, d=3, Full: means=6, weights=1, cov=2*(3*4/2)=12 → 19
    assert_eq!(count_params(2, 3, CovType::Full), 19);
    // K=2, d=3, Diagonal: means=6, weights=1, cov=2*3=6 → 13
    assert_eq!(count_params(2, 3, CovType::Diagonal), 13);
}

#[test]
fn test_gmm_k1() {
    let (data, t) = generate_two_clusters(10, 50);
    let (features, _d) =
        build_features(&data, &t, None, 6, ProjectionBasisType::Bspline, 1.0).unwrap();

    let result = gmm_em(&features, 1, CovType::Full, 100, 1e-6, 42).unwrap();
    assert!(result.cluster.iter().all(|&c| c == 0));
    assert!(result.converged);
}

#[test]
fn test_gmm_weights_sum_to_one() {
    let (data, t) = generate_two_clusters(10, 50);
    let (features, _d) =
        build_features(&data, &t, None, 6, ProjectionBasisType::Bspline, 1.0).unwrap();

    let result = gmm_em(&features, 3, CovType::Diagonal, 100, 1e-6, 42).unwrap();
    let sum: f64 = result.weights.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-6,
        "Mixing weights should sum to 1, got {}",
        sum
    );
}

/// Build two Gaussian clusters directly in feature space with a controllable
/// separation, feature scale, and within-cluster spread. Returns row-major
/// feature vectors (n = 2 * n_per rows, `d` columns).
fn make_gaussian_feature_clusters(
    n_per: usize,
    d: usize,
    sep: f64,
    scale: f64,
    sd: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut gaussian = |rng: &mut StdRng| -> f64 {
        let u1: f64 = rng.gen::<f64>().max(1e-15);
        let u2: f64 = rng.gen::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    };
    let mut feats = Vec::with_capacity(2 * n_per);
    for g in 0..2 {
        let center = g as f64 * sep;
        for _ in 0..n_per {
            let row = (0..d)
                .map(|_| scale * (center + sd * gaussian(&mut rng)))
                .collect();
            feats.push(row);
        }
    }
    feats
}

/// Regression test for GH #3 (over-splitting / no BIC minimum).
///
/// Two well-separated Gaussian clusters with a LARGE feature scale (variance of
/// order 400, matching real basis-coefficient magnitudes). Before the
/// data-scaled covariance regularization fix, the fixed absolute `reg = 1e-6`
/// floor was negligible at this scale, so components collapsed onto a few points
/// (variance → 0), the log-likelihood exploded, and BIC kept decreasing past the
/// true K. With a data-scaled floor, BIC must attain its minimum at the true K=2.
#[test]
fn test_gmm_bic_minimum_at_true_k_large_scale() {
    // scale=20 => per-dimension variance ~ (20 * 1.0)^2 = 400.
    let features = make_gaussian_feature_clusters(60, 5, 6.0, 20.0, 1.0, 7);

    let mut bic_values = Vec::new();
    let mut best_bic = f64::INFINITY;
    let mut best_k = 0;
    for k in 1..=6 {
        let r = run_multiple_inits(&features, k, CovType::Diagonal, 200, 1e-6, 3, 42).unwrap();
        bic_values.push((k, r.bic));
        // No component covariance should collapse to the regularization floor and
        // inflate the log-likelihood: a finite, sane LL is expected at every K.
        assert!(
            r.log_likelihood.is_finite(),
            "LL must stay finite at K={k}, got {}",
            r.log_likelihood
        );
        if r.bic < best_bic {
            best_bic = r.bic;
            best_k = k;
        }
    }

    assert_eq!(
        best_k, 2,
        "BIC should minimize at true K=2 on large-scale data; got K={best_k}, BICs: {bic_values:?}"
    );
    // BIC must increase after the minimum (a real minimum, not monotone decrease).
    let bic_k2 = bic_values[1].1;
    let bic_k3 = bic_values[2].1;
    assert!(
        bic_k3 > bic_k2,
        "BIC should rise past the true K (K3 > K2); BICs: {bic_values:?}"
    );
}

/// Regression test for GH #3 (membership effectively hard).
///
/// For genuinely overlapping clusters the returned membership must contain
/// GRADED posteriors (not a near one-hot of the hard assignment). We check that
/// at least some observations near the boundary have a max responsibility well
/// below 1, which is only possible if soft posteriors are exposed.
#[test]
fn test_gmm_membership_graded_on_overlap() {
    // sep=2, sd=1 => substantial overlap between the two Gaussians.
    let features = make_gaussian_feature_clusters(80, 4, 2.0, 1.0, 1.0, 11);
    let result = gmm_em(&features, 2, CovType::Diagonal, 200, 1e-6, 42).unwrap();

    let mem = &result.membership;
    let (n, k) = mem.shape();
    let mut min_max_resp = 1.0_f64;
    for i in 0..n {
        let mut mx = 0.0_f64;
        for c in 0..k {
            mx = mx.max(mem[(i, c)]);
        }
        min_max_resp = min_max_resp.min(mx);
    }
    assert!(
        min_max_resp < 0.9,
        "Overlapping clusters should yield graded posteriors (some max-resp < 0.9), \
         got min max-resp = {min_max_resp:.4}"
    );
}

/// Unit test for the data-scaled covariance regularization floor.
#[test]
fn test_data_scaled_reg_tracks_feature_scale() {
    use super::covariance::{data_scaled_reg, REG_REL};

    // Two features with variance ~100 each => mean variance ~100 => reg ~ 1e-4.
    let features = make_gaussian_feature_clusters(200, 2, 0.0, 10.0, 1.0, 3);
    let reg = data_scaled_reg(&features, 2);
    // Should be orders of magnitude larger than the old fixed 1e-6 floor
    // (feature variance ~100 => reg ~ 1e-4).
    assert!(
        reg > 10.0 * REG_REL,
        "data-scaled reg should reflect feature variance, got {reg}"
    );
    // Degenerate (zero-variance) data falls back to REG_REL, never zero.
    let flat = vec![vec![1.0, 1.0]; 10];
    assert_eq!(data_scaled_reg(&flat, 2), REG_REL);
    assert_eq!(data_scaled_reg(&[], 2), REG_REL);
}
