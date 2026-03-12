//! Example 22: Model-Based Functional Clustering via GMM
//!
//! Clusters functional data using Gaussian mixture models:
//! - Direct EM on FPC scores (`gmm_em`)
//! - Full pipeline with automatic K selection (`gmm_cluster`)
//! - Prediction on new data (`predict_gmm`)

use fdars_core::basis::ProjectionBasisType;
use fdars_core::gmm::{gmm_cluster, gmm_em, predict_gmm, CovType};
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc_1d;
use std::f64::consts::PI;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    let n = 60;
    let m = 51;
    let t = uniform_grid(m);

    // Generate 3 clusters with distinct curve shapes
    let mut col_major = vec![0.0; n * m];
    let true_labels: Vec<usize> = (0..n).map(|i| i % 3).collect();

    for i in 0..n {
        let noise = 0.1 * ((i * 13 + 7) % 100) as f64 / 100.0;
        for (j, &tj) in t.iter().enumerate() {
            let curve = match true_labels[i] {
                0 => (2.0 * PI * tj).sin() + noise,
                1 => (2.0 * PI * tj).cos() + 0.5 * tj + noise,
                _ => 0.5 * (4.0 * PI * tj).sin() - tj + noise,
            };
            col_major[i + j * n] = curve;
        }
    }

    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

    // ── 1. GMM on FPC scores ──────────────────────────────────────────────
    println!("=== GMM on FPC Scores ===");
    let fpca = fdata_to_pc_1d(&data, 3).unwrap();
    let scores: Vec<Vec<f64>> = (0..n)
        .map(|i| (0..3).map(|k| fpca.scores[(i, k)]).collect())
        .collect();

    let gmm = gmm_em(&scores, 3, CovType::Full, 200, 1e-6, 42).unwrap();
    println!(
        "  Converged: {} ({} iterations)",
        gmm.converged, gmm.iterations
    );
    println!("  Log-likelihood: {:.4}", gmm.log_likelihood);
    println!("  BIC: {:.4}", gmm.bic);

    // Accuracy (accounting for label permutation)
    let accuracy = compute_accuracy(&gmm.cluster, &true_labels, 3);
    println!("  Accuracy: {:.1}%", accuracy * 100.0);

    // ── 2. Full pipeline with K selection ─────────────────────────────────
    println!("\n=== GMM Pipeline with Automatic K Selection ===");
    let result = gmm_cluster(
        &data,
        &t,
        None,       // no scalar covariates
        &[2, 3, 4], // test K=2,3,4
        5,          // 5 basis functions
        ProjectionBasisType::Bspline,
        CovType::Full,
        1.0,   // covariate weight
        200,   // max EM iterations
        1e-6,  // tolerance
        3,     // 3 random initializations
        42,    // seed
        false, // use BIC (not ICL)
    )
    .unwrap();

    println!("  Selected K = {}", result.best.k);
    println!(
        "  BIC values: {:?}",
        result
            .bic_values
            .iter()
            .map(|(k, b)| format!("K={k}: {b:.1}"))
            .collect::<Vec<_>>()
    );

    // ── 3. Predict on new data ────────────────────────────────────────────
    println!("\n=== Predict Cluster for New Curve ===");
    let new_col = t.iter().map(|&tj| (2.0 * PI * tj).sin()).collect();
    let new_data = FdMatrix::from_column_major(new_col, 1, m).unwrap();
    let (labels, probs) =
        predict_gmm(&new_data, &t, None, &result.best, 5, ProjectionBasisType::Bspline, 1.0, CovType::Full).unwrap();
    println!("  Assigned cluster: {}", labels[0]);
    println!(
        "  Membership: [{}]",
        (0..result.best.k)
            .map(|k| format!("{:.3}", probs[(0, k)]))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

/// Find the best unused (row, col) entry in the confusion matrix.
fn find_best_match(conf: &[Vec<usize>], used: &[bool], k: usize) -> (usize, usize, usize) {
    let mut best = (0, 0, 0);
    for (r, row) in conf.iter().enumerate() {
        for (c, &val) in row.iter().enumerate() {
            if !used[c] && val > best.2 {
                best = (r, c, val);
            }
        }
    }
    let _ = k;
    best
}

/// Compute accuracy with best label permutation (greedy matching).
fn compute_accuracy(predicted: &[usize], true_labels: &[usize], k: usize) -> f64 {
    let n = predicted.len();
    let mut conf = vec![vec![0usize; k]; k];
    for i in 0..n {
        if predicted[i] < k && true_labels[i] < k {
            conf[true_labels[i]][predicted[i]] += 1;
        }
    }
    let mut used = vec![false; k];
    let mut correct = 0;
    for _ in 0..k {
        let best = find_best_match(&conf, &used, k);
        used[best.1] = true;
        correct += best.2;
        conf[best.0] = vec![0; k];
    }
    correct as f64 / n as f64
}
