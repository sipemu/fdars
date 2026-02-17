//! Example 06: Distances and Metrics for Functional Data
//!
//! Demonstrates computing pairwise distance matrices using different
//! metrics: Lp, Hausdorff, DTW, Fourier-based semimetric, and
//! horizontal shift semimetric. Compares how different metrics
//! capture different notions of curve similarity.

use fdars_core::metric::{
    dtw_self_1d, fourier_self_1d, hausdorff_self_1d, hshift_self_1d, lp_cross_1d, lp_self_1d,
};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

/// Print a portion of an n x n distance matrix (column-major)
fn print_dist_matrix(dists: &[f64], n: usize, max_show: usize) {
    let show = n.min(max_show);
    print!("       ");
    for j in 0..show {
        print!("{j:>8}");
    }
    println!();
    for i in 0..show {
        print!("  {i:>3}: ");
        for j in 0..show {
            if i == j {
                print!("   ---  ");
            } else {
                print!("{:>8.4}", dists[i + j * n]);
            }
        }
        println!();
    }
}

fn main() {
    println!("=== Example 06: Distances and Metrics ===\n");

    let n = 10;
    let m = 50;
    let big_m = 5;
    let t = uniform_grid(m);
    let empty_weights: Vec<f64> = vec![];

    let data = sim_fundata(
        n,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );

    // --- Section 1: L2 pairwise distances ---
    // Self-distance functions return n x n matrices (column-major), symmetric with zero diagonal
    println!("--- L2 Distance Matrix ---");
    let l2_dists = lp_self_1d(&data, n, m, &t, 2.0, &empty_weights);
    println!("  Matrix size: {} ({n} x {n})", l2_dists.len());
    print_dist_matrix(&l2_dists, n, 5);

    // --- Section 2: L1 distances ---
    println!("\n--- L1 Distance Matrix ---");
    let l1_dists = lp_self_1d(&data, n, m, &t, 1.0, &empty_weights);
    print_dist_matrix(&l1_dists, n, 5);

    // --- Section 3: L-infinity distances ---
    println!("\n--- L∞ Distance Matrix ---");
    let linf_dists = lp_self_1d(&data, n, m, &t, f64::INFINITY, &empty_weights);
    print_dist_matrix(&linf_dists, n, 5);

    // --- Section 4: Hausdorff distances ---
    println!("\n--- Hausdorff Distance Matrix ---");
    let haus_dists = hausdorff_self_1d(&data, n, m, &t);
    print_dist_matrix(&haus_dists, n, 5);

    // --- Section 5: DTW distances ---
    println!("\n--- DTW Distance Matrix (p=2, window=5) ---");
    let dtw_dists = dtw_self_1d(&data, n, m, 2.0, 5);
    print_dist_matrix(&dtw_dists, n, 5);

    // --- Section 6: Fourier-based semimetric ---
    println!("\n--- Fourier Semimetric (5 frequencies) ---");
    let fourier_dists = fourier_self_1d(&data, n, m, 5);
    print_dist_matrix(&fourier_dists, n, 5);

    // --- Section 7: Horizontal shift semimetric ---
    println!("\n--- Horizontal Shift Semimetric (max_shift=5) ---");
    let hshift_dists = hshift_self_1d(&data, n, m, &t, 5);
    print_dist_matrix(&hshift_dists, n, 5);

    // --- Section 8: Cross-distance matrix ---
    println!("\n--- Cross-Distance Matrix (L2, first 5 vs last 5) ---");
    let data1: Vec<f64> = (0..5 * m)
        .map(|idx| {
            let i = idx % 5;
            let j = idx / 5;
            data[i + j * n]
        })
        .collect();
    let data2: Vec<f64> = (0..5 * m)
        .map(|idx| {
            let i = idx % 5;
            let j = idx / 5;
            data[(i + 5) + j * n]
        })
        .collect();
    let cross = lp_cross_1d(&data1, &data2, 5, 5, m, &t, 2.0, &empty_weights);
    println!("  Cross-distance matrix: {} elements (5 x 5)", cross.len());
    for i in 0..5 {
        print!("  Row {i}: ");
        for j in 0..5 {
            print!("{:.4} ", cross[i + j * 5]);
        }
        println!();
    }

    // --- Section 9: Metric comparison ---
    // dists[i + j * n] = distance between curve i and curve j
    println!("\n--- Metric Comparison (distance between curves 0 and 1) ---");
    println!("  L1:        {:.6}", l1_dists[n]);
    println!("  L2:        {:.6}", l2_dists[n]);
    println!("  L∞:        {:.6}", linf_dists[n]);
    println!("  Hausdorff: {:.6}", haus_dists[n]);
    println!("  DTW:       {:.6}", dtw_dists[n]);
    println!("  Fourier:   {:.6}", fourier_dists[n]);
    println!("  H-shift:   {:.6}", hshift_dists[n]);

    println!("\n=== Done ===");
}
