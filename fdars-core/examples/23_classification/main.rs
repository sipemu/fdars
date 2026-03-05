//! Example 23: Functional Classification
//!
//! Classifies functional data into discrete groups using:
//! - FPC + LDA/QDA (`fclassif_lda`, `fclassif_qda`)
//! - FPC + k-NN (`fclassif_knn`)
//! - Depth-based DD-classifier (`fclassif_dd`)
//! - Cross-validated error rates (`fclassif_cv`)

use fdars_core::classification::{
    fclassif_cv, fclassif_dd, fclassif_knn, fclassif_lda, fclassif_qda,
};
use fdars_core::matrix::FdMatrix;
use std::f64::consts::PI;

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    let n = 60;
    let m = 51;
    let t = uniform_grid(m);

    // Generate 3-class functional data
    let mut col_major = vec![0.0; n * m];
    let labels: Vec<usize> = (0..n).map(|i| i % 3).collect();

    for i in 0..n {
        let noise = 0.05 * ((i * 13 + 7) % 100) as f64 / 100.0;
        for (j, &tj) in t.iter().enumerate() {
            col_major[i + j * n] = match labels[i] {
                0 => (2.0 * PI * tj).sin() + noise,
                1 => (2.0 * PI * tj).cos() + noise,
                _ => 0.5 * (4.0 * PI * tj).sin() + 0.3 * tj + noise,
            };
        }
    }

    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let ncomp = 3;

    println!("=== Functional Classification ({n} curves, 3 classes) ===\n");

    // ── LDA ───────────────────────────────────────────────────────────────
    let lda = fclassif_lda(&data, &labels, None, ncomp).unwrap();
    println!("LDA:  accuracy = {:.1}%", lda.accuracy * 100.0);

    // ── QDA ───────────────────────────────────────────────────────────────
    let qda = fclassif_qda(&data, &labels, None, ncomp).unwrap();
    println!("QDA:  accuracy = {:.1}%", qda.accuracy * 100.0);

    // ── k-NN ──────────────────────────────────────────────────────────────
    let knn = fclassif_knn(&data, &labels, None, ncomp, 5).unwrap();
    println!("k-NN: accuracy = {:.1}%  (k=5)", knn.accuracy * 100.0);

    // ── DD-classifier ─────────────────────────────────────────────────────
    let dd = fclassif_dd(&data, &labels, None).unwrap();
    println!("DD:   accuracy = {:.1}%", dd.accuracy * 100.0);

    // ── Confusion matrix (LDA) ────────────────────────────────────────────
    println!("\nLDA Confusion Matrix:");
    println!("         Pred 0  Pred 1  Pred 2");
    for (r, row) in lda.confusion.iter().enumerate() {
        println!(
            "  True {}: {:>5}   {:>5}   {:>5}",
            r, row[0], row[1], row[2]
        );
    }

    // ── Cross-validation ──────────────────────────────────────────────────
    println!("\n=== 5-fold Cross-Validation ===");
    let cv = fclassif_cv(&data, &t, &labels, None, "lda", ncomp, 5, 42).unwrap();
    println!("  LDA CV error rate: {:.1}%", cv.error_rate * 100.0);
    println!(
        "  Per-fold errors: [{}]",
        cv.fold_errors
            .iter()
            .map(|e| format!("{:.1}%", e * 100.0))
            .collect::<Vec<_>>()
            .join(", ")
    );
}
