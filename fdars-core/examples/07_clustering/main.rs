//! Example 07: Functional Data Clustering
//!
//! Demonstrates k-means and fuzzy c-means clustering of functional data.
//! Creates three groups of curves from different generative processes,
//! combines them, and evaluates clustering quality using silhouette
//! scores and the Calinski-Harabasz index.

use fdars_core::clustering::{calinski_harabasz, fuzzy_cmeans_fd, kmeans_fd, silhouette_score};
use fdars_core::matrix::FdMatrix;
use fdars_core::simulation::{sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 07: Functional Data Clustering ===\n");

    let n_per_group = 15;
    let n = n_per_group * 3;
    let m = 50;
    let big_m = 5;
    let t = uniform_grid(m);

    // --- Generate 3 groups of curves ---
    println!("--- Generating 3 Groups ---");
    let group1 = sim_fundata(
        n_per_group,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    let group2 = sim_fundata(
        n_per_group,
        &t,
        big_m,
        EFunType::Poly,
        EValType::Linear,
        Some(43),
    );
    let group3 = sim_fundata(
        n_per_group,
        &t,
        big_m,
        EFunType::Wiener,
        EValType::Wiener,
        Some(44),
    );

    // Add offsets to separate groups
    let mut combined = vec![0.0; n * m];
    for j in 0..m {
        for i in 0..n_per_group {
            combined[i + j * n] = group1[(i, j)];
            combined[(i + n_per_group) + j * n] = group2[(i, j)] + 3.0;
            combined[(i + 2 * n_per_group) + j * n] = group3[(i, j)] - 3.0;
        }
    }

    let combined = FdMatrix::from_column_major(combined, n, m).unwrap();

    let true_labels: Vec<usize> = (0..n).map(|i| i / n_per_group).collect();
    println!("  Group 1 (Fourier/Exp): curves 0-{}", n_per_group - 1);
    println!(
        "  Group 2 (Poly/Linear + offset): curves {}-{}",
        n_per_group,
        2 * n_per_group - 1
    );
    println!(
        "  Group 3 (Wiener/Wiener - offset): curves {}-{}",
        2 * n_per_group,
        n - 1
    );
    println!("  Total: {n} curves, {m} grid points");

    // --- Section 1: K-means clustering ---
    println!("\n--- K-Means Clustering (k=3) ---");
    let km = kmeans_fd(&combined, &t, 3, 100, 1e-6, 42);
    println!("  Converged: {} (iter: {})", km.converged, km.iter);
    println!("  Total within-SS: {:.4}", km.tot_withinss);
    println!(
        "  Within-SS per cluster: {:?}",
        km.withinss
            .iter()
            .map(|x| format!("{x:.4}"))
            .collect::<Vec<_>>()
    );
    println!("  Cluster assignments: {:?}", km.cluster);

    // Compare with true labels
    let accuracy = compute_clustering_accuracy(&km.cluster, &true_labels, 3);
    println!("  Clustering accuracy: {:.1}%", accuracy * 100.0);

    // --- Section 2: Silhouette scores ---
    println!("\n--- Silhouette Analysis ---");
    let sil = silhouette_score(&combined, &t, &km.cluster);
    let mean_sil: f64 = sil.iter().sum::<f64>() / sil.len() as f64;
    println!("  Mean silhouette score: {mean_sil:.4}");
    for k in 0..3 {
        let cluster_sil: Vec<&f64> = sil
            .iter()
            .enumerate()
            .filter(|(i, _)| km.cluster[*i] == k)
            .map(|(_, s)| s)
            .collect();
        let cluster_mean = cluster_sil.iter().copied().sum::<f64>() / cluster_sil.len() as f64;
        println!(
            "  Cluster {k} mean silhouette: {cluster_mean:.4} ({} members)",
            cluster_sil.len()
        );
    }

    // --- Section 3: Calinski-Harabasz index ---
    println!("\n--- Calinski-Harabasz Index ---");
    let ch = calinski_harabasz(&combined, &t, &km.cluster);
    println!("  CH index (k=3): {ch:.4}");
    println!("  (Higher values indicate better-defined clusters)");

    // --- Section 4: Choosing k ---
    println!("\n--- Choosing k (Silhouette + CH for k=2..5) ---");
    for k in 2..=5 {
        let km_k = kmeans_fd(&combined, &t, k, 100, 1e-6, 42);
        let sil_k = silhouette_score(&combined, &t, &km_k.cluster);
        let mean_sil_k: f64 = sil_k.iter().sum::<f64>() / sil_k.len() as f64;
        let ch_k = calinski_harabasz(&combined, &t, &km_k.cluster);
        println!(
            "  k={k}: silhouette={mean_sil_k:.4}, CH={ch_k:.2}, within-SS={:.2}",
            km_k.tot_withinss
        );
    }

    // --- Section 5: Fuzzy C-means ---
    println!("\n--- Fuzzy C-Means (k=3, fuzziness=2.0) ---");
    let fcm = fuzzy_cmeans_fd(&combined, &t, 3, 2.0, 100, 1e-6, 42);
    println!("  Converged: {} (iter: {})", fcm.converged, fcm.iter);

    // Show membership matrix for first few curves
    println!("  Membership degrees (first 5 curves from each group):");
    println!("  {:>5} {:>8} {:>8} {:>8}", "Curve", "C0", "C1", "C2");
    for &i in &[0, 1, 2, 3, 4, 15, 16, 17, 18, 19, 30, 31, 32, 33, 34] {
        println!(
            "  {:>5} {:>8.4} {:>8.4} {:>8.4}",
            i,
            fcm.membership[(i, 0)],
            fcm.membership[(i, 1)],
            fcm.membership[(i, 2)]
        );
    }

    println!("\n=== Done ===");
}

/// Compute clustering accuracy using best-match permutation of labels
fn compute_clustering_accuracy(predicted: &[usize], true_labels: &[usize], k: usize) -> f64 {
    // Try all permutations for small k (here k=3, so 6 permutations)
    let mut best_acc = 0.0;
    let perms = permutations(k);
    for perm in &perms {
        let correct = predicted
            .iter()
            .zip(true_labels.iter())
            .filter(|(&p, &t)| perm[p] == t)
            .count();
        let acc = correct as f64 / predicted.len() as f64;
        if acc > best_acc {
            best_acc = acc;
        }
    }
    best_acc
}

fn permutations(n: usize) -> Vec<Vec<usize>> {
    if n == 0 {
        return vec![vec![]];
    }
    let mut result = Vec::new();
    let sub = permutations(n - 1);
    for perm in &sub {
        for i in 0..=perm.len() {
            let mut new_perm = perm.clone();
            new_perm.insert(i, n - 1);
            result.push(new_perm);
        }
    }
    result
}
