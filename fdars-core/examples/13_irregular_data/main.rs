//! Example 13: Irregular Functional Data
//!
//! Demonstrates working with irregularly sampled functional data using
//! the CSR-like (Compressed Sparse Row) storage format. Shows how to
//! construct, query, integrate, compute norms, estimate mean functions,
//! compute distances, and regularize to a common grid.

use fdars_core::irreg_fdata::{
    integrate_irreg, mean_irreg, metric_lp_irreg, norm_lp_irreg, to_regular_grid, IrregFdata,
};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};

fn main() {
    println!("=== Example 13: Irregular Functional Data ===\n");

    // --- Section 1: Construct irregular data ---
    println!("--- Constructing Irregular Data ---");
    let mut rng = StdRng::seed_from_u64(42);
    let n = 5;

    // Each curve has a different number of observation points
    let mut argvals_list = Vec::new();
    let mut values_list = Vec::new();
    let point_dist = Uniform::new(15, 40);

    for i in 0..n {
        let n_points = point_dist.sample(&mut rng) as usize;
        // Random observation times, sorted
        let uniform = Uniform::new(0.0_f64, 1.0);
        let mut times: Vec<f64> = (0..n_points).map(|_| uniform.sample(&mut rng)).collect();
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Function values: sin(2πt) + curve-specific offset
        let offset = i as f64 * 0.5;
        let vals: Vec<f64> = times
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin() + offset)
            .collect();
        println!(
            "  Curve {i}: {n_points} points in [{:.3}, {:.3}], offset={offset:.1}",
            times.first().unwrap(),
            times.last().unwrap()
        );
        argvals_list.push(times);
        values_list.push(vals);
    }

    // --- Section 2: IrregFdata construction ---
    println!("\n--- IrregFdata Object ---");
    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);
    println!("  Number of observations: {}", ifd.n_obs());
    println!("  Total points: {}", ifd.total_points());
    println!("  Min points per curve: {}", ifd.min_obs());
    println!("  Max points per curve: {}", ifd.max_obs());
    println!("  Range: [{:.3}, {:.3}]", ifd.rangeval[0], ifd.rangeval[1]);
    println!("  Observation counts: {:?}", ifd.obs_counts());

    // --- Section 3: Accessing individual observations ---
    println!("\n--- Accessing Observations ---");
    for i in 0..n {
        let (times, vals) = ifd.get_obs(i);
        println!(
            "  Curve {i}: {} points, first=({:.3}, {:.3}), last=({:.3}, {:.3})",
            ifd.n_points(i),
            times[0],
            vals[0],
            times[times.len() - 1],
            vals[vals.len() - 1]
        );
    }

    // --- Section 4: Integration ---
    println!("\n--- Integration (trapezoidal) ---");
    let integrals = integrate_irreg(&ifd.offsets, &ifd.argvals, &ifd.values);
    for (i, &val) in integrals.iter().enumerate() {
        println!("  Curve {i}: integral = {val:.6}");
    }

    // --- Section 5: Lp norms ---
    println!("\n--- Lp Norms ---");
    let l2_norms = norm_lp_irreg(&ifd.offsets, &ifd.argvals, &ifd.values, 2.0);
    let l1_norms = norm_lp_irreg(&ifd.offsets, &ifd.argvals, &ifd.values, 1.0);
    for i in 0..n {
        println!("  Curve {i}: L1={:.4}, L2={:.4}", l1_norms[i], l2_norms[i]);
    }

    // --- Section 6: Mean function estimation ---
    println!("\n--- Mean Function Estimation ---");
    let target_m = 50;
    let target_grid: Vec<f64> = (0..target_m)
        .map(|i| i as f64 / (target_m - 1) as f64)
        .collect();
    let bandwidth = 0.1;
    let kernel_type = 0; // Gaussian kernel
    let mean_fn = mean_irreg(
        &ifd.offsets,
        &ifd.argvals,
        &ifd.values,
        &target_grid,
        bandwidth,
        kernel_type,
    );
    println!("  Target grid: {target_m} equally spaced points");
    println!("  Bandwidth: {bandwidth}");
    println!("  Mean at t=0.0: {:.4}", mean_fn[0]);
    println!("  Mean at t=0.5: {:.4}", mean_fn[target_m / 2]);
    println!("  Mean at t=1.0: {:.4}", mean_fn[target_m - 1]);

    // --- Section 7: Pairwise distances ---
    println!("\n--- Pairwise Lp Distances ---");
    let dists = metric_lp_irreg(&ifd.offsets, &ifd.argvals, &ifd.values, 2.0);
    let n_pairs = n * (n - 1) / 2;
    println!("  Number of pairs: {n_pairs}");
    let mut pair_idx = 0;
    for i in 0..n {
        for j in (i + 1)..n {
            if pair_idx < dists.len() {
                println!("  d({i},{j}) = {:.4}", dists[pair_idx]);
            }
            pair_idx += 1;
        }
    }

    // --- Section 8: Regularize to common grid ---
    println!("\n--- Regularization to Common Grid ---");
    let regular = to_regular_grid(&ifd.offsets, &ifd.argvals, &ifd.values, &target_grid);
    let n_regular = regular.len() / target_m;
    println!("  Regular grid: {target_m} points");
    println!("  Regularized data: {n_regular} curves x {target_m} points");
    for i in 0..n {
        let first = regular[i];
        let mid = regular[i + (target_m / 2) * n_regular];
        let last = regular[i + (target_m - 1) * n_regular];
        println!("  Curve {i}: first={first:.4}, mid={mid:.4}, last={last:.4}");
    }

    println!("\n=== Done ===");
}
