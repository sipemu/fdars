//! Example 06: Distances and Metrics for Functional Data
//!
//! Demonstrates computing pairwise distance matrices using different
//! metrics: Lp, Hausdorff, DTW, Fourier-based semimetric, and
//! horizontal shift semimetric. Compares how different metrics
//! capture different notions of curve similarity.

use fdars_core::matrix::FdMatrix;
use fdars_core::metric::{
    dtw_self_1d, fourier_self_1d, hausdorff_self_1d, hshift_self_1d, lp_cross_1d, lp_self_1d,
};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

/// Print a portion of an n x n distance matrix (FdMatrix)
fn print_dist_matrix(dists: &FdMatrix, max_show: usize) {
    let n = dists.nrows();
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
                print!("{:>8.4}", dists[(i, j)]);
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
    // Self-distance functions return n x n matrices (symmetric with zero diagonal)
    println!("--- L2 Distance Matrix ---");
    let l2_dists = lp_self_1d(&data, &t, 2.0, &empty_weights);
    println!("  Matrix size: {}x{}", l2_dists.nrows(), l2_dists.ncols());
    print_dist_matrix(&l2_dists, 5);

    // --- Section 2: L1 distances ---
    println!("\n--- L1 Distance Matrix ---");
    let l1_dists = lp_self_1d(&data, &t, 1.0, &empty_weights);
    print_dist_matrix(&l1_dists, 5);

    // --- Section 3: L-infinity distances ---
    println!("\n--- L-inf Distance Matrix ---");
    let linf_dists = lp_self_1d(&data, &t, f64::INFINITY, &empty_weights);
    print_dist_matrix(&linf_dists, 5);

    // --- Section 4: Hausdorff distances ---
    println!("\n--- Hausdorff Distance Matrix ---");
    let haus_dists = hausdorff_self_1d(&data, &t);
    print_dist_matrix(&haus_dists, 5);

    // --- Section 5: DTW distances ---
    println!("\n--- DTW Distance Matrix (p=2, window=5) ---");
    let dtw_dists = dtw_self_1d(&data, 2.0, 5);
    print_dist_matrix(&dtw_dists, 5);

    // --- Section 6: Fourier-based semimetric ---
    println!("\n--- Fourier Semimetric (5 frequencies) ---");
    let fourier_dists = fourier_self_1d(&data, 5);
    print_dist_matrix(&fourier_dists, 5);

    // --- Section 7: Horizontal shift semimetric ---
    println!("\n--- Horizontal Shift Semimetric (max_shift=5) ---");
    let hshift_dists = hshift_self_1d(&data, &t, 5);
    print_dist_matrix(&hshift_dists, 5);

    // --- Section 8: Cross-distance matrix ---
    println!("\n--- Cross-Distance Matrix (L2, first 5 vs last 5) ---");
    // Extract first 5 and last 5 curves as sub-matrices
    let mut flat1 = vec![0.0; 5 * m];
    let mut flat2 = vec![0.0; 5 * m];
    for j in 0..m {
        for i in 0..5 {
            flat1[i + j * 5] = data[(i, j)];
            flat2[i + j * 5] = data[(i + 5, j)];
        }
    }
    let data1 = FdMatrix::from_column_major(flat1, 5, m).unwrap();
    let data2 = FdMatrix::from_column_major(flat2, 5, m).unwrap();
    let cross = lp_cross_1d(&data1, &data2, &t, 2.0, &empty_weights);
    println!(
        "  Cross-distance matrix: {}x{} (5 x 5)",
        cross.nrows(),
        cross.ncols()
    );
    for i in 0..5 {
        print!("  Row {i}: ");
        for j in 0..5 {
            print!("{:.4} ", cross[(i, j)]);
        }
        println!();
    }

    // --- Section 9: Metric comparison ---
    // dist[(0, 1)] = distance between curve 0 and curve 1
    println!("\n--- Metric Comparison (distance between curves 0 and 1) ---");
    println!("  L1:        {:.6}", l1_dists[(0, 1)]);
    println!("  L2:        {:.6}", l2_dists[(0, 1)]);
    println!("  L-inf:     {:.6}", linf_dists[(0, 1)]);
    println!("  Hausdorff: {:.6}", haus_dists[(0, 1)]);
    println!("  DTW:       {:.6}", dtw_dists[(0, 1)]);
    println!("  Fourier:   {:.6}", fourier_dists[(0, 1)]);
    println!("  H-shift:   {:.6}", hshift_dists[(0, 1)]);

    println!("\n=== Done ===");
}
