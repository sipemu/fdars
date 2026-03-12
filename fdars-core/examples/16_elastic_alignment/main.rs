//! Example: Elastic alignment, SRSF transforms, and warping utilities.
//!
//! Demonstrates the full elastic analysis pipeline:
//! 1. SRSF transform and inverse round-trip
//! 2. Pairwise elastic alignment
//! 3. Warping composition and reparameterization
//! 4. Karcher mean computation
//! 5. Aligning curves to a target
//! 6. Elastic distance matrix
//! 7. Cross distance matrix
//! 8. Elastic tolerance bands (pointwise and simultaneous)

use fdars_core::alignment::{
    align_to_target, compose_warps, elastic_align_pair, elastic_cross_distance_matrix,
    elastic_distance, elastic_self_distance_matrix, karcher_mean, reparameterize_curve,
    srsf_inverse, srsf_transform,
};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};
use fdars_core::tolerance::{elastic_tolerance_band, BandType};
use fdars_core::{l2_distance, simpsons_weights, FdMatrix};

fn main() {
    let m = 50;
    let t: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let n = 20;
    let data = sim_fundata(n, &t, 5, EFunType::Fourier, EValType::Exponential, Some(42));

    // ── 1. SRSF transform and inverse round-trip ──
    println!("=== 1. SRSF Transform ===");
    let q = srsf_transform(&data, &t);
    println!(
        "Input shape: {:?}, SRSF shape: {:?}",
        data.shape(),
        q.shape()
    );

    // Round-trip reconstruction for each of the first 3 curves
    for i in 0..3 {
        let fi = data.row(i);
        let qi = q.row(i);
        let f_recon = srsf_inverse(&qi, &t, fi[0]);
        let max_err: f64 = fi
            .iter()
            .zip(f_recon.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("  Curve {i} round-trip max error: {max_err:.6}");
    }

    // Show SRSF of a linear function: f(t) = 2t → q(t) = sqrt(2)
    let f_linear: Vec<f64> = t.iter().map(|&ti| 2.0 * ti).collect();
    let q_linear_mat = srsf_transform(&FdMatrix::from_slice(&f_linear, 1, m).unwrap(), &t);
    let q_linear = q_linear_mat.row(0);
    println!(
        "  SRSF of f(t)=2t at midpoint: {:.4} (expected sqrt(2) = {:.4})",
        q_linear[m / 2],
        2.0_f64.sqrt()
    );

    // ── 2. Pairwise alignment ──
    println!("\n=== 2. Pairwise Alignment ===");
    let f1 = data.row(0);
    let f2 = data.row(1);

    // L2 distance before alignment
    let weights = simpsons_weights(&t);
    let l2_before = l2_distance(&f1, &f2, &weights);

    let result = elastic_align_pair(&f1, &f2, &t, 0.0);
    let l2_after = l2_distance(&f1, &result.f_aligned, &weights);

    println!("  Elastic distance: {:.6}", result.distance);
    println!("  L2 before alignment: {l2_before:.6}");
    println!("  L2 after alignment:  {l2_after:.6}");
    println!(
        "  Warp range: [{:.4}, {:.4}]",
        result.gamma[0],
        result.gamma[m - 1]
    );
    println!(
        "  Warp deviation from identity: {:.6}",
        result
            .gamma
            .iter()
            .zip(t.iter())
            .map(|(g, ti)| (g - ti).abs())
            .fold(0.0_f64, f64::max)
    );

    // Alignment of identical curves
    let self_result = elastic_align_pair(&f1, &f1, &t, 0.0);
    println!(
        "  Self-alignment distance: {:.6} (should be ~0)",
        self_result.distance
    );

    // ── 3. Warping utilities ──
    println!("\n=== 3. Warping Utilities ===");

    // Reparameterize: apply a quadratic warp to a linear function
    let f_id: Vec<f64> = t.clone();
    let gamma_quad: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
    let f_warped = reparameterize_curve(&f_id, &t, &gamma_quad);
    println!(
        "  f(t)=t with gamma=t^2: f(gamma(0.5)) = {:.4} (expected 0.25)",
        f_warped[m / 2]
    );

    // Compose warps: gamma1(t) = sqrt(t), gamma2(t) = t^2 → (g1 o g2)(t) = t
    let g1: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();
    let g2: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
    let composed = compose_warps(&g1, &g2, &t);
    let compose_err: f64 = composed
        .iter()
        .zip(t.iter())
        .map(|(c, ti)| (c - ti).abs())
        .fold(0.0_f64, f64::max);
    println!("  sqrt(t) o t^2 max deviation from identity: {compose_err:.6}");

    // ── 4. Karcher mean ──
    println!("\n=== 4. Karcher Mean ===");
    let km = karcher_mean(&data, &t, 15, 1e-4, 0.0);
    println!(
        "  Converged: {} after {} iterations",
        km.converged, km.n_iter
    );
    println!(
        "  Mean range: [{:.4}, {:.4}]",
        km.mean.iter().cloned().fold(f64::INFINITY, f64::min),
        km.mean.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );

    // Check that warping functions are valid
    let max_warp_dev: f64 = (0..n)
        .map(|i| {
            (0..m)
                .map(|j| (km.gammas[(i, j)] - t[j]).abs())
                .fold(0.0_f64, f64::max)
        })
        .fold(0.0_f64, f64::max);
    println!("  Max warp deviation from identity: {max_warp_dev:.4}");

    // Variance reduction after alignment
    let var_before = pointwise_variance(&data, m);
    let var_after = pointwise_variance(&km.aligned_data, m);
    println!("  Mean pointwise variance before: {var_before:.6}");
    println!("  Mean pointwise variance after:  {var_after:.6}");

    // ── 5. Align all curves to Karcher mean ──
    println!("\n=== 5. Align to Target (Karcher mean) ===");
    let aligned = align_to_target(&data, &km.mean, &t, 0.0);
    println!(
        "  Mean elastic distance: {:.6}",
        aligned.distances.iter().sum::<f64>() / n as f64
    );
    println!(
        "  Max elastic distance: {:.6}",
        aligned.distances.iter().cloned().fold(0.0_f64, f64::max)
    );
    println!(
        "  Min elastic distance: {:.6}",
        aligned
            .distances
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
    );

    // ── 6. Elastic self distance matrix ──
    println!("\n=== 6. Elastic Distance Matrix (first 5 curves) ===");
    let small_n = 5;
    let small_data = subset_data(&data, small_n, m);
    let dm = elastic_self_distance_matrix(&small_data, &t, 0.0);
    for i in 0..small_n {
        let row: Vec<String> = (0..small_n).map(|j| format!("{:.4}", dm[(i, j)])).collect();
        println!("  [{}]", row.join(", "));
    }

    // Verify properties
    println!(
        "  Diagonal zeros: {}",
        (0..small_n).all(|i| dm[(i, i)] < 1e-12)
    );
    println!(
        "  Symmetric: {}",
        (0..small_n).all(|i| ((i + 1)..small_n).all(|j| (dm[(i, j)] - dm[(j, i)]).abs() < 1e-12))
    );

    // Direct pairwise check
    let f0 = small_data.row(0);
    let f1_small = small_data.row(1);
    let d01 = elastic_distance(&f0, &f1_small, &t, 0.0);
    println!(
        "  Matrix (0,1) = {:.4}, pairwise d(0,1) = {d01:.4}",
        dm[(0, 1)]
    );

    // ── 7. Cross distance matrix ──
    println!("\n=== 7. Cross Distance Matrix ===");
    let data_a = subset_data(&data, 3, m);
    let data_b = {
        let mut d = FdMatrix::zeros(4, m);
        for i in 0..4 {
            for j in 0..m {
                d[(i, j)] = data[(i + 3, j)];
            }
        }
        d
    };
    let cdm = elastic_cross_distance_matrix(&data_a, &data_b, &t, 0.0);
    println!("  Shape: {:?}", cdm.shape());
    for i in 0..3 {
        let row: Vec<String> = (0..4).map(|j| format!("{:.4}", cdm[(i, j)])).collect();
        println!("  [{}]", row.join(", "));
    }

    // ── 8. Elastic tolerance bands ──
    println!("\n=== 8. Elastic Tolerance Bands ===");

    // Pointwise band
    let pw_band = elastic_tolerance_band(&data, &t, 3, 100, 0.95, BandType::Pointwise, 10, 42);
    match pw_band {
        Ok(b) => {
            let mean_hw: f64 = b.half_width.iter().sum::<f64>() / m as f64;
            println!("  Pointwise 95% band:");
            println!(
                "    Center range: [{:.4}, {:.4}]",
                b.center.iter().cloned().fold(f64::INFINITY, f64::min),
                b.center.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            );
            println!("    Mean half-width: {mean_hw:.4}");
            println!(
                "    All lower < upper: {}",
                b.lower.iter().zip(b.upper.iter()).all(|(l, u)| l < u)
            );
        }
        Err(e) => println!("  Pointwise band: failed ({e})"),
    }

    // Simultaneous band
    let sim_band = elastic_tolerance_band(&data, &t, 3, 100, 0.95, BandType::Simultaneous, 10, 42);
    match sim_band {
        Ok(b) => {
            let mean_hw: f64 = b.half_width.iter().sum::<f64>() / m as f64;
            println!("  Simultaneous 95% band:");
            println!("    Mean half-width: {mean_hw:.4}");
        }
        Err(e) => println!("  Simultaneous band: failed ({e})"),
    }

    // Compare coverage levels
    let b90 = elastic_tolerance_band(&data, &t, 3, 100, 0.90, BandType::Pointwise, 10, 42).unwrap();
    let b99 = elastic_tolerance_band(&data, &t, 3, 100, 0.99, BandType::Pointwise, 10, 42).unwrap();
    let hw90: f64 = b90.half_width.iter().sum::<f64>() / m as f64;
    let hw99: f64 = b99.half_width.iter().sum::<f64>() / m as f64;
    println!("  90% mean half-width: {hw90:.4}");
    println!("  99% mean half-width: {hw99:.4}");
    println!("  99% wider than 90%: {}", hw99 > hw90);

    println!("\nDone.");
}

/// Compute mean pointwise variance across all evaluation points.
fn pointwise_variance(data: &FdMatrix, m: usize) -> f64 {
    let n = data.nrows();
    let nf = n as f64;
    let mut total_var = 0.0;
    for j in 0..m {
        let col = data.column(j);
        let mean = col.iter().sum::<f64>() / nf;
        let var: f64 = col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0);
        total_var += var;
    }
    total_var / m as f64
}

/// Extract first `k` rows of data as a new FdMatrix.
fn subset_data(data: &FdMatrix, k: usize, m: usize) -> FdMatrix {
    let mut d = FdMatrix::zeros(k, m);
    for i in 0..k {
        for j in 0..m {
            d[(i, j)] = data[(i, j)];
        }
    }
    d
}
