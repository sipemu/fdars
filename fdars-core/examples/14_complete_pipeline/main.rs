//! Example 14: Complete FDA Pipeline
//!
//! Demonstrates an end-to-end functional data analysis workflow:
//! simulate → smooth → detect outliers → FPCA → cluster → characterize.
//! This integrates concepts from all previous examples into a cohesive
//! analysis pipeline.

use fdars_core::basis::pspline_fit_1d;
use fdars_core::clustering::{kmeans_fd, silhouette_score, KmeansResult};
use fdars_core::depth::modified_band_1d;
use fdars_core::matrix::FdMatrix;
use fdars_core::outliers::{detect_outliers_lrt, outliers_threshold_lrt};
use fdars_core::regression::{fdata_to_pc_1d, FpcaResult};
use fdars_core::simulation::{add_error_pointwise, sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn generate_data(t: &[f64], m: usize, big_m: usize) -> (FdMatrix, usize, usize, usize) {
    let n_per_group = 20;
    let n_outliers = 3;

    let group_a = sim_fundata(
        n_per_group,
        t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    let group_b = sim_fundata(
        n_per_group,
        t,
        big_m,
        EFunType::Poly,
        EValType::Linear,
        Some(43),
    );

    let n_total = 2 * n_per_group + n_outliers;
    let mut clean = FdMatrix::zeros(n_total, m);
    for j in 0..m {
        for i in 0..n_per_group {
            clean[(i, j)] = group_a[(i, j)];
            clean[(i + n_per_group, j)] = group_b[(i, j)] + 2.0;
        }
        clean[(2 * n_per_group, j)] = group_a[(0, j)] + 5.0;
        clean[(2 * n_per_group + 1, j)] =
            group_a[(1, j)] + 3.0 * (15.0 * std::f64::consts::PI * t[j]).sin();
        clean[(2 * n_per_group + 2, j)] = group_b[(0, j)] - 4.0;
    }

    let data = add_error_pointwise(&clean, 0.2, Some(42));
    println!("  Group A: {n_per_group} Fourier/Exp curves");
    println!("  Group B: {n_per_group} Poly/Linear curves (+2.0 offset)");
    println!("  Outliers: {n_outliers} injected");
    println!("  Noise: sd=0.2 pointwise");
    println!("  Total: {n_total} curves, {m} grid points");

    (data, n_total, n_per_group, n_outliers)
}

fn smooth_data(data: &FdMatrix, t: &[f64], n_total: usize, m: usize) -> FdMatrix {
    let nbasis = 20;
    let lambda = 0.1;
    if let Some(pspline) = pspline_fit_1d(data, t, nbasis, lambda, 2) {
        println!("  P-spline: nbasis={nbasis}, λ={lambda}");
        println!("  EDF: {:.1}, GCV: {:.6}", pspline.edf, pspline.gcv);
        let noise_reduction: f64 = data
            .as_slice()
            .iter()
            .zip(pspline.fitted.as_slice().iter())
            .map(|(d, f)| (d - f).powi(2))
            .sum::<f64>()
            / (n_total * m) as f64;
        println!("  Mean squared smoothing change: {noise_reduction:.6}");
        pspline.fitted
    } else {
        println!("  P-spline fitting failed, using raw data");
        data.clone()
    }
}

fn detect_and_remove_outliers(
    smoothed: &FdMatrix,
    n_total: usize,
    m: usize,
    n_per_group: usize,
) -> (Vec<usize>, FdMatrix) {
    let threshold = outliers_threshold_lrt(smoothed, 100, 0.05, 0.1, 42, 0.99);
    let is_outlier = detect_outliers_lrt(smoothed, threshold, 0.1);
    let outlier_indices: Vec<usize> = is_outlier
        .iter()
        .enumerate()
        .filter(|(_, &o)| o)
        .map(|(i, _)| i)
        .collect();
    println!("  Threshold: {threshold:.4}");
    println!("  Detected outliers: {outlier_indices:?}");
    println!(
        "  True outlier indices: [{}, {}, {}]",
        2 * n_per_group,
        2 * n_per_group + 1,
        2 * n_per_group + 2
    );

    let clean_indices: Vec<usize> = (0..n_total).filter(|i| !is_outlier[*i]).collect();
    let n_clean = clean_indices.len();
    let mut clean_data = vec![0.0; n_clean * m];
    for (new_i, &old_i) in clean_indices.iter().enumerate() {
        for j in 0..m {
            clean_data[new_i + j * n_clean] = smoothed[(old_i, j)];
        }
    }
    println!("  After removal: {n_clean} curves remaining");

    let clean_mat = FdMatrix::from_slice(&clean_data, n_clean, m).unwrap();
    (clean_indices, clean_mat)
}

fn print_clustering_results(
    km: &KmeansResult,
    clean_mat: &FdMatrix,
    t: &[f64],
    clean_indices: &[usize],
    n_per_group: usize,
    n_clean: usize,
    k: usize,
) {
    println!("  k={k}, converged: {} (iter: {})", km.converged, km.iter);
    println!("  Within-SS: {:.4}", km.tot_withinss);

    for c in 0..k {
        let size = km.cluster.iter().filter(|&&x| x == c).count();
        println!("  Cluster {c}: {size} curves");
    }

    let sil = silhouette_score(clean_mat, t, &km.cluster);
    let mean_sil: f64 = sil.iter().sum::<f64>() / sil.len() as f64;
    println!("  Mean silhouette: {mean_sil:.4}");

    let true_labels: Vec<usize> = clean_indices
        .iter()
        .map(|&i| if i < n_per_group { 0 } else { 1 })
        .collect();
    let match_a: usize = (0..n_clean)
        .filter(|&i| km.cluster[i] == true_labels[i])
        .count();
    let match_b: usize = (0..n_clean)
        .filter(|&i| km.cluster[i] != true_labels[i])
        .count();
    let accuracy = match_a.max(match_b) as f64 / n_clean as f64;
    println!("  Clustering accuracy: {:.1}%", accuracy * 100.0);
}

fn print_depth_characterization(
    clean_mat: &FdMatrix,
    km: &KmeansResult,
    fpca: &FpcaResult,
    n_clean: usize,
    ncomp: usize,
    k: usize,
) {
    let depths = modified_band_1d(clean_mat, clean_mat);

    for c in 0..k {
        let cluster_depths: Vec<f64> = (0..n_clean)
            .filter(|&i| km.cluster[i] == c)
            .map(|i| depths[i])
            .collect();
        let mean_d: f64 = cluster_depths.iter().sum::<f64>() / cluster_depths.len() as f64;
        let min_d = cluster_depths.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_d = cluster_depths
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        println!("  Cluster {c}: mean_depth={mean_d:.4}, range=[{min_d:.4}, {max_d:.4}]");
    }

    let deepest_idx = depths
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    let deepest_cluster = km.cluster[deepest_idx];
    println!("  Most central curve: index {deepest_idx} (cluster {deepest_cluster})");

    println!("\n--- PC Score Means by Cluster ---");
    println!(
        "  {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Cluster", "PC1", "PC2", "PC3", "PC4"
    );
    for c in 0..k {
        let members: Vec<usize> = (0..n_clean).filter(|&i| km.cluster[i] == c).collect();
        let nc = members.len() as f64;
        print!("  {:>8}", c);
        for comp in 0..ncomp {
            let mean_score: f64 = members.iter().map(|&i| fpca.scores[(i, comp)]).sum::<f64>() / nc;
            print!(" {:>10.4}", mean_score);
        }
        println!();
    }
}

fn main() {
    println!("=== Example 14: Complete FDA Pipeline ===\n");

    let m = 80;
    let big_m = 5;
    let t = uniform_grid(m);

    // Step 1: Simulate multi-group functional data
    println!("--- Step 1: Data Generation ---");
    let (data, n_total, n_per_group, _n_outliers) = generate_data(&t, m, big_m);

    // Step 2: Smooth the data
    println!("\n--- Step 2: P-spline Smoothing ---");
    let smoothed = smooth_data(&data, &t, n_total, m);

    // Step 3: Outlier detection
    println!("\n--- Step 3: Outlier Detection (LRT) ---");
    let (clean_indices, clean_mat) = detect_and_remove_outliers(&smoothed, n_total, m, n_per_group);
    let n_clean = clean_indices.len();

    // Step 4: Functional PCA
    println!("\n--- Step 4: Functional PCA ---");
    let ncomp = 4;
    if let Some(fpca) = fdata_to_pc_1d(&clean_mat, ncomp) {
        let total_var: f64 = fpca.singular_values.iter().map(|s| s * s).sum();
        let mut cumvar = 0.0;
        for (k, sv) in fpca.singular_values.iter().enumerate() {
            cumvar += sv * sv;
            let cumprop = cumvar / total_var * 100.0;
            println!(
                "  PC{}: var explained = {:.1}% (cumulative: {:.1}%)",
                k + 1,
                sv * sv / total_var * 100.0,
                cumprop
            );
        }

        // Step 5: Clustering
        println!("\n--- Step 5: K-Means Clustering ---");
        let k = 2;
        let km = kmeans_fd(&clean_mat, &t, k, 100, 1e-6, 42);
        print_clustering_results(&km, &clean_mat, &t, &clean_indices, n_per_group, n_clean, k);

        // Step 6: Depth-based characterization
        println!("\n--- Step 6: Depth Characterization ---");
        print_depth_characterization(&clean_mat, &km, &fpca, n_clean, ncomp, k);
    }

    // Summary
    let nbasis = 20;
    let lambda = 0.1;
    println!("\n--- Pipeline Summary ---");
    println!("  1. Generated {n_total} curves (2 groups + outliers + noise)");
    println!("  2. Smoothed with P-splines (nbasis={nbasis}, λ={lambda})");
    println!("  3. Detected and removed {} outlier(s)", n_total - n_clean);
    println!("  4. FPCA: extracted {ncomp} principal components");
    println!("  5. K-means clustering into 2 groups");
    println!("  6. Characterized clusters with depth measures and PC scores");

    println!("\n=== Done ===");
}
