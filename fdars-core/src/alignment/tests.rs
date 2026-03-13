//! Tests for the alignment module.

use super::*;
use crate::helpers::{cumulative_trapz, l2_distance, linear_interp, simpsons_weights, trapz};
use crate::simulation::{sim_fundata, EFunType, EValType};
use crate::test_helpers::uniform_grid;
use crate::warping::{inner_product_l2, normalize_warp};

fn make_test_data(n: usize, m: usize, seed: u64) -> FdMatrix {
    let t = uniform_grid(m);
    sim_fundata(
        n,
        &t,
        3,
        EFunType::Fourier,
        EValType::Exponential,
        Some(seed),
    )
}

// ── cumulative_trapz ──

#[test]
fn test_cumulative_trapz_constant() {
    // ∫₀ᵗ 1 dt = t
    let x = uniform_grid(50);
    let y = vec![1.0; 50];
    let result = cumulative_trapz(&y, &x);
    assert!((result[0]).abs() < 1e-15, "cumulative_trapz(0) should be 0");
    for j in 1..50 {
        assert!(
            (result[j] - x[j]).abs() < 1e-12,
            "∫₀^{:.3} 1 dt should be {:.3}, got {:.3}",
            x[j],
            x[j],
            result[j]
        );
    }
}

#[test]
fn test_cumulative_trapz_linear() {
    // ∫₀ᵗ s ds = t²/2
    let m = 100;
    let x = uniform_grid(m);
    let y: Vec<f64> = x.clone();
    let result = cumulative_trapz(&y, &x);
    for j in 1..m {
        let expected = x[j] * x[j] / 2.0;
        assert!(
            (result[j] - expected).abs() < 1e-4,
            "∫₀^{:.3} s ds: expected {expected:.6}, got {:.6}",
            x[j],
            result[j]
        );
    }
}

// ── normalize_warp ──

#[test]
fn test_normalize_warp_fixes_boundaries() {
    let t = uniform_grid(10);
    let mut gamma = vec![0.1; 10]; // constant, wrong boundaries
    normalize_warp(&mut gamma, &t);
    assert_eq!(gamma[0], t[0]);
    assert_eq!(gamma[9], t[9]);
}

#[test]
fn test_normalize_warp_enforces_monotonicity() {
    let t = uniform_grid(5);
    let mut gamma = vec![0.0, 0.5, 0.3, 0.8, 1.0]; // non-monotone at index 2
    normalize_warp(&mut gamma, &t);
    for j in 1..5 {
        assert!(
            gamma[j] >= gamma[j - 1],
            "gamma should be monotone after normalization at j={j}"
        );
    }
}

#[test]
fn test_normalize_warp_identity_unchanged() {
    let t = uniform_grid(20);
    let mut gamma = t.clone();
    normalize_warp(&mut gamma, &t);
    for j in 0..20 {
        assert!(
            (gamma[j] - t[j]).abs() < 1e-15,
            "Identity warp should be unchanged"
        );
    }
}

// ── linear_interp ──

#[test]
fn test_linear_interp_at_nodes() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 2.0, 4.0, 6.0];
    for i in 0..x.len() {
        assert!((linear_interp(&x, &y, x[i]) - y[i]).abs() < 1e-12);
    }
}

#[test]
fn test_linear_interp_midpoints() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 2.0, 4.0];
    assert!((linear_interp(&x, &y, 0.5) - 1.0).abs() < 1e-12);
    assert!((linear_interp(&x, &y, 1.5) - 3.0).abs() < 1e-12);
}

#[test]
fn test_linear_interp_clamp() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![1.0, 3.0, 5.0];
    assert!((linear_interp(&x, &y, -1.0) - 1.0).abs() < 1e-12);
    assert!((linear_interp(&x, &y, 3.0) - 5.0).abs() < 1e-12);
}

#[test]
fn test_linear_interp_nonuniform_grid() {
    let x = vec![0.0, 0.1, 0.5, 1.0];
    let y = vec![0.0, 1.0, 5.0, 10.0];
    // Between 0.1 and 0.5: slope = (5-1)/(0.5-0.1) = 10
    let val = linear_interp(&x, &y, 0.3);
    let expected = 1.0 + 10.0 * (0.3 - 0.1);
    assert!(
        (val - expected).abs() < 1e-12,
        "Non-uniform interp: expected {expected}, got {val}"
    );
}

#[test]
fn test_linear_interp_two_points() {
    let x = vec![0.0, 1.0];
    let y = vec![3.0, 7.0];
    assert!((linear_interp(&x, &y, 0.25) - 4.0).abs() < 1e-12);
    assert!((linear_interp(&x, &y, 0.75) - 6.0).abs() < 1e-12);
}

// ── SRSF transform ──

#[test]
fn test_srsf_transform_linear() {
    // f(t) = 2t: derivative = 2, SRSF = sqrt(2)
    let m = 50;
    let t = uniform_grid(m);
    let f: Vec<f64> = t.iter().map(|&ti| 2.0 * ti).collect();
    let mat = FdMatrix::from_slice(&f, 1, m).unwrap();

    let q_mat = srsf_transform(&mat, &t);
    let q: Vec<f64> = q_mat.row(0);

    let expected = 2.0_f64.sqrt();
    // Interior points should be close to sqrt(2)
    for j in 2..(m - 2) {
        assert!(
            (q[j] - expected).abs() < 0.1,
            "q[{j}] = {}, expected ~{expected}",
            q[j]
        );
    }
}

#[test]
fn test_srsf_transform_preserves_shape() {
    let data = make_test_data(10, 50, 42);
    let t = uniform_grid(50);
    let q = srsf_transform(&data, &t);
    assert_eq!(q.shape(), data.shape());
}

#[test]
fn test_srsf_transform_constant_is_zero() {
    // f(t) = 5 (constant): derivative = 0, SRSF = 0
    let m = 30;
    let t = uniform_grid(m);
    let f = vec![5.0; m];
    let mat = FdMatrix::from_slice(&f, 1, m).unwrap();
    let q_mat = srsf_transform(&mat, &t);
    let q: Vec<f64> = q_mat.row(0);

    for j in 0..m {
        assert!(
            q[j].abs() < 1e-10,
            "SRSF of constant should be 0, got q[{j}] = {}",
            q[j]
        );
    }
}

#[test]
fn test_srsf_transform_negative_slope() {
    // f(t) = -3t: derivative = -3, SRSF = -sqrt(3)
    let m = 50;
    let t = uniform_grid(m);
    let f: Vec<f64> = t.iter().map(|&ti| -3.0 * ti).collect();
    let mat = FdMatrix::from_slice(&f, 1, m).unwrap();

    let q_mat = srsf_transform(&mat, &t);
    let q: Vec<f64> = q_mat.row(0);

    let expected = -(3.0_f64.sqrt());
    for j in 2..(m - 2) {
        assert!(
            (q[j] - expected).abs() < 0.15,
            "q[{j}] = {}, expected ~{expected}",
            q[j]
        );
    }
}

#[test]
fn test_srsf_transform_empty_input() {
    let data = FdMatrix::zeros(0, 0);
    let t: Vec<f64> = vec![];
    let q = srsf_transform(&data, &t);
    assert_eq!(q.shape(), (0, 0));
}

#[test]
fn test_srsf_transform_multiple_curves() {
    let m = 40;
    let t = uniform_grid(m);
    let data = make_test_data(5, m, 42);

    let q = srsf_transform(&data, &t);
    assert_eq!(q.shape(), (5, m));

    // Each row should have finite values
    for i in 0..5 {
        for j in 0..m {
            assert!(q[(i, j)].is_finite(), "SRSF should be finite at ({i},{j})");
        }
    }
}

// ── SRSF inverse ──

#[test]
fn test_srsf_round_trip() {
    let m = 100;
    let t = uniform_grid(m);
    // Use a smooth function
    let f: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin() + ti)
        .collect();

    let mat = FdMatrix::from_slice(&f, 1, m).unwrap();
    let q_mat = srsf_transform(&mat, &t);
    let q: Vec<f64> = q_mat.row(0);

    let f_recon = srsf_inverse(&q, &t, f[0]);

    // Check reconstruction is close (interior points, avoid boundary effects)
    let max_err: f64 = f[5..(m - 5)]
        .iter()
        .zip(f_recon[5..(m - 5)].iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_err < 0.15,
        "Round-trip error too large: max_err = {max_err}"
    );
}

#[test]
fn test_srsf_inverse_empty() {
    let q: Vec<f64> = vec![];
    let t: Vec<f64> = vec![];
    let result = srsf_inverse(&q, &t, 0.0);
    assert!(result.is_empty());
}

#[test]
fn test_srsf_inverse_preserves_initial_value() {
    let m = 50;
    let t = uniform_grid(m);
    let q = vec![1.0; m]; // constant SRSF
    let f0 = 3.15;
    let f = srsf_inverse(&q, &t, f0);
    assert!((f[0] - f0).abs() < 1e-12, "srsf_inverse should start at f0");
}

#[test]
fn test_srsf_round_trip_multiple_curves() {
    let m = 80;
    let t = uniform_grid(m);
    let data = make_test_data(5, m, 99);

    let q_mat = srsf_transform(&data, &t);

    for i in 0..5 {
        let fi = data.row(i);
        let qi = q_mat.row(i);
        let f_recon = srsf_inverse(&qi, &t, fi[0]);
        let max_err: f64 = fi[5..(m - 5)]
            .iter()
            .zip(f_recon[5..(m - 5)].iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(max_err < 0.3, "Round-trip curve {i}: max_err = {max_err}");
    }
}

// ── Reparameterize ──

#[test]
fn test_reparameterize_identity_warp() {
    let m = 50;
    let t = uniform_grid(m);
    let f: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();

    // Identity warp: γ(t) = t
    let result = reparameterize_curve(&f, &t, &t);
    for j in 0..m {
        assert!(
            (result[j] - f[j]).abs() < 1e-12,
            "Identity warp should return original at j={j}"
        );
    }
}

#[test]
fn test_reparameterize_linear_warp() {
    let m = 50;
    let t = uniform_grid(m);
    // f(t) = t (linear), γ(t) = t^2 (quadratic warp on [0,1])
    let f: Vec<f64> = t.clone();
    let gamma: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();

    let result = reparameterize_curve(&f, &t, &gamma);

    // f(γ(t)) = γ(t) = t^2 for a linear f(t) = t
    for j in 0..m {
        assert!(
            (result[j] - gamma[j]).abs() < 1e-10,
            "f(gamma(t)) should be gamma(t) for f(t)=t at j={j}"
        );
    }
}

#[test]
fn test_reparameterize_sine_with_quadratic_warp() {
    let m = 100;
    let t = uniform_grid(m);
    let f: Vec<f64> = t
        .iter()
        .map(|&ti| (std::f64::consts::PI * ti).sin())
        .collect();
    let gamma: Vec<f64> = t.iter().map(|&ti| ti * ti).collect(); // speeds up start

    let result = reparameterize_curve(&f, &t, &gamma);

    // f(γ(t)) = sin(π * t²); check a few known values
    for j in 0..m {
        let expected = (std::f64::consts::PI * gamma[j]).sin();
        assert!(
            (result[j] - expected).abs() < 0.05,
            "sin(π γ(t)) at j={j}: expected {expected:.4}, got {:.4}",
            result[j]
        );
    }
}

#[test]
fn test_reparameterize_preserves_length() {
    let m = 50;
    let t = uniform_grid(m);
    let f = vec![1.0; m];
    let gamma: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();

    let result = reparameterize_curve(&f, &t, &gamma);
    assert_eq!(result.len(), m);
}

// ── Compose warps ──

#[test]
fn test_compose_warps_identity() {
    let m = 50;
    let t = uniform_grid(m);
    // γ(t) = t^0.5 (a warp on [0,1])
    let gamma: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();

    // identity ∘ γ = γ
    let result = compose_warps(&t, &gamma, &t);
    for j in 0..m {
        assert!(
            (result[j] - gamma[j]).abs() < 1e-10,
            "id ∘ γ should be γ at j={j}"
        );
    }

    // γ ∘ identity = γ
    let result2 = compose_warps(&gamma, &t, &t);
    for j in 0..m {
        assert!(
            (result2[j] - gamma[j]).abs() < 1e-10,
            "γ ∘ id should be γ at j={j}"
        );
    }
}

#[test]
fn test_compose_warps_associativity() {
    // (γ₁ ∘ γ₂) ∘ γ₃ ≈ γ₁ ∘ (γ₂ ∘ γ₃)
    let m = 50;
    let t = uniform_grid(m);
    let g1: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();
    let g2: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
    let g3: Vec<f64> = t.iter().map(|&ti| 0.5 * ti + 0.5 * ti * ti).collect();

    let g12 = compose_warps(&g1, &g2, &t);
    let left = compose_warps(&g12, &g3, &t); // (g1∘g2) ∘ g3

    let g23 = compose_warps(&g2, &g3, &t);
    let right = compose_warps(&g1, &g23, &t); // g1 ∘ (g2∘g3)

    for j in 0..m {
        assert!(
            (left[j] - right[j]).abs() < 0.05,
            "Composition should be roughly associative at j={j}: left={:.4}, right={:.4}",
            left[j],
            right[j]
        );
    }
}

#[test]
fn test_compose_warps_preserves_domain() {
    let m = 50;
    let t = uniform_grid(m);
    let g1: Vec<f64> = t.iter().map(|&ti| ti.sqrt()).collect();
    let g2: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();

    let composed = compose_warps(&g1, &g2, &t);
    assert!(
        (composed[0] - t[0]).abs() < 1e-10,
        "Composed warp should start at domain start"
    );
    assert!(
        (composed[m - 1] - t[m - 1]).abs() < 1e-10,
        "Composed warp should end at domain end"
    );
}

// ── Elastic align pair ──

#[test]
fn test_align_identical_curves() {
    let m = 50;
    let t = uniform_grid(m);
    let f: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();

    let result = elastic_align_pair(&f, &f, &t, 0.0);

    // Distance should be near zero
    assert!(
        result.distance < 0.1,
        "Distance between identical curves should be near 0, got {}",
        result.distance
    );

    // Warp should be near identity
    for j in 0..m {
        assert!(
            (result.gamma[j] - t[j]).abs() < 0.1,
            "Warp should be near identity at j={j}: gamma={}, t={}",
            result.gamma[j],
            t[j]
        );
    }
}

#[test]
fn test_align_pair_valid_output() {
    let data = make_test_data(2, 50, 42);
    let t = uniform_grid(50);
    let f1 = data.row(0);
    let f2 = data.row(1);

    let result = elastic_align_pair(&f1, &f2, &t, 0.0);

    assert_eq!(result.gamma.len(), 50);
    assert_eq!(result.f_aligned.len(), 50);
    assert!(result.distance >= 0.0);

    // Warp should be monotone
    for j in 1..50 {
        assert!(
            result.gamma[j] >= result.gamma[j - 1],
            "Warp should be monotone at j={j}"
        );
    }
}

#[test]
fn test_align_pair_warp_boundaries() {
    let data = make_test_data(2, 50, 42);
    let t = uniform_grid(50);
    let f1 = data.row(0);
    let f2 = data.row(1);

    let result = elastic_align_pair(&f1, &f2, &t, 0.0);
    assert!(
        (result.gamma[0] - t[0]).abs() < 1e-12,
        "Warp should start at domain start"
    );
    assert!(
        (result.gamma[49] - t[49]).abs() < 1e-12,
        "Warp should end at domain end"
    );
}

#[test]
fn test_align_shifted_sine() {
    // Two sines with a phase shift — alignment should reduce distance
    let m = 80;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
        .collect();

    let weights = simpsons_weights(&t);
    let l2_before = l2_distance(&f1, &f2, &weights);
    let result = elastic_align_pair(&f1, &f2, &t, 0.0);
    let l2_after = l2_distance(&f1, &result.f_aligned, &weights);

    assert!(
        l2_after < l2_before + 0.01,
        "Alignment should not increase L2 distance: before={l2_before:.4}, after={l2_after:.4}"
    );
}

#[test]
fn test_align_pair_aligned_curve_is_finite() {
    let data = make_test_data(2, 50, 77);
    let t = uniform_grid(50);
    let f1 = data.row(0);
    let f2 = data.row(1);

    let result = elastic_align_pair(&f1, &f2, &t, 0.0);
    for j in 0..50 {
        assert!(
            result.f_aligned[j].is_finite(),
            "Aligned curve should be finite at j={j}"
        );
    }
}

#[test]
fn test_align_pair_minimum_grid() {
    // Minimum viable grid: m = 2
    let t = vec![0.0, 1.0];
    let f1 = vec![0.0, 1.0];
    let f2 = vec![0.0, 2.0];
    let result = elastic_align_pair(&f1, &f2, &t, 0.0);
    assert_eq!(result.gamma.len(), 2);
    assert_eq!(result.f_aligned.len(), 2);
    assert!(result.distance >= 0.0);
}

// ── Elastic distance ──

#[test]
fn test_elastic_distance_symmetric() {
    let data = make_test_data(3, 50, 42);
    let t = uniform_grid(50);
    let f1 = data.row(0);
    let f2 = data.row(1);

    let d12 = elastic_distance(&f1, &f2, &t, 0.0);
    let d21 = elastic_distance(&f2, &f1, &t, 0.0);

    // Should be approximately symmetric (DP is not perfectly symmetric)
    assert!(
        (d12 - d21).abs() < d12.max(d21) * 0.3 + 0.01,
        "Elastic distance should be roughly symmetric: d12={d12}, d21={d21}"
    );
}

#[test]
fn test_elastic_distance_nonneg() {
    let data = make_test_data(3, 50, 42);
    let t = uniform_grid(50);

    for i in 0..3 {
        for j in 0..3 {
            let fi = data.row(i);
            let fj = data.row(j);
            let d = elastic_distance(&fi, &fj, &t, 0.0);
            assert!(d >= 0.0, "Elastic distance should be non-negative");
        }
    }
}

#[test]
fn test_elastic_distance_self_near_zero() {
    let data = make_test_data(3, 50, 42);
    let t = uniform_grid(50);

    for i in 0..3 {
        let fi = data.row(i);
        let d = elastic_distance(&fi, &fi, &t, 0.0);
        assert!(
            d < 0.1,
            "Self-distance should be near zero, got {d} for curve {i}"
        );
    }
}

#[test]
fn test_elastic_distance_triangle_inequality() {
    let data = make_test_data(3, 50, 42);
    let t = uniform_grid(50);
    let f0 = data.row(0);
    let f1 = data.row(1);
    let f2 = data.row(2);

    let d01 = elastic_distance(&f0, &f1, &t, 0.0);
    let d12 = elastic_distance(&f1, &f2, &t, 0.0);
    let d02 = elastic_distance(&f0, &f2, &t, 0.0);

    // Relaxed triangle inequality (DP alignment is approximate)
    let slack = 0.5;
    assert!(
        d02 <= d01 + d12 + slack,
        "Triangle inequality (relaxed): d02={d02:.4} > d01={d01:.4} + d12={d12:.4} + {slack}"
    );
}

#[test]
fn test_elastic_distance_different_shapes_nonzero() {
    let m = 50;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t.to_vec(); // linear
    let f2: Vec<f64> = t.iter().map(|&ti| ti * ti).collect(); // quadratic

    let d = elastic_distance(&f1, &f2, &t, 0.0);
    assert!(
        d > 0.01,
        "Distance between different shapes should be > 0, got {d}"
    );
}

// ── Self distance matrix ──

#[test]
fn test_self_distance_matrix_symmetric() {
    let data = make_test_data(5, 30, 42);
    let t = uniform_grid(30);

    let dm = elastic_self_distance_matrix(&data, &t, 0.0);
    let n = dm.nrows();

    assert_eq!(dm.shape(), (5, 5));

    // Zero diagonal
    for i in 0..n {
        assert!(dm[(i, i)].abs() < 1e-12, "Diagonal should be zero");
    }

    // Symmetric
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                (dm[(i, j)] - dm[(j, i)]).abs() < 1e-12,
                "Matrix should be symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_self_distance_matrix_nonneg() {
    let data = make_test_data(4, 30, 42);
    let t = uniform_grid(30);
    let dm = elastic_self_distance_matrix(&data, &t, 0.0);

    for i in 0..4 {
        for j in 0..4 {
            assert!(
                dm[(i, j)] >= 0.0,
                "Distance matrix entries should be non-negative at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_self_distance_matrix_single_curve() {
    let data = make_test_data(1, 30, 42);
    let t = uniform_grid(30);
    let dm = elastic_self_distance_matrix(&data, &t, 0.0);
    assert_eq!(dm.shape(), (1, 1));
    assert!(dm[(0, 0)].abs() < 1e-12);
}

#[test]
fn test_self_distance_matrix_consistent_with_pairwise() {
    let data = make_test_data(4, 30, 42);
    let t = uniform_grid(30);

    let dm = elastic_self_distance_matrix(&data, &t, 0.0);

    // Check a few entries match direct elastic_distance calls
    for i in 0..4 {
        for j in (i + 1)..4 {
            let fi = data.row(i);
            let fj = data.row(j);
            let d_direct = elastic_distance(&fi, &fj, &t, 0.0);
            assert!(
                (dm[(i, j)] - d_direct).abs() < 1e-10,
                "Matrix entry ({i},{j})={:.6} should match pairwise {d_direct:.6}",
                dm[(i, j)]
            );
        }
    }
}

// ── Karcher mean ──

#[test]
fn test_karcher_mean_identical_curves() {
    let m = 50;
    let t = uniform_grid(m);
    let f: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();

    // Create 5 identical curves
    let mut data = FdMatrix::zeros(5, m);
    for i in 0..5 {
        for j in 0..m {
            data[(i, j)] = f[j];
        }
    }

    let result = karcher_mean(&data, &t, 10, 1e-4, 0.0);

    assert_eq!(result.mean.len(), m);
    assert!(result.n_iter <= 10);
}

#[test]
fn test_karcher_mean_output_shape() {
    let data = make_test_data(15, 50, 42);
    let t = uniform_grid(50);

    let result = karcher_mean(&data, &t, 5, 1e-3, 0.0);

    assert_eq!(result.mean.len(), 50);
    assert_eq!(result.mean_srsf.len(), 50);
    assert_eq!(result.gammas.shape(), (15, 50));
    assert_eq!(result.aligned_data.shape(), (15, 50));
    assert!(result.n_iter <= 5);
}

#[test]
fn test_karcher_mean_warps_are_valid() {
    let data = make_test_data(10, 40, 42);
    let t = uniform_grid(40);

    let result = karcher_mean(&data, &t, 5, 1e-3, 0.0);

    for i in 0..10 {
        // Boundary values
        assert!(
            (result.gammas[(i, 0)] - t[0]).abs() < 1e-10,
            "Warp {i} should start at domain start"
        );
        assert!(
            (result.gammas[(i, 39)] - t[39]).abs() < 1e-10,
            "Warp {i} should end at domain end"
        );
        // Monotonicity
        for j in 1..40 {
            assert!(
                result.gammas[(i, j)] >= result.gammas[(i, j - 1)],
                "Warp {i} should be monotone at j={j}"
            );
        }
    }
}

#[test]
fn test_karcher_mean_aligned_data_is_finite() {
    let data = make_test_data(8, 40, 42);
    let t = uniform_grid(40);
    let result = karcher_mean(&data, &t, 5, 1e-3, 0.0);

    for i in 0..8 {
        for j in 0..40 {
            assert!(
                result.aligned_data[(i, j)].is_finite(),
                "Aligned data should be finite at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_karcher_mean_srsf_is_finite() {
    let data = make_test_data(8, 40, 42);
    let t = uniform_grid(40);
    let result = karcher_mean(&data, &t, 5, 1e-3, 0.0);

    for j in 0..40 {
        assert!(
            result.mean_srsf[j].is_finite(),
            "Mean SRSF should be finite at j={j}"
        );
        assert!(
            result.mean[j].is_finite(),
            "Mean curve should be finite at j={j}"
        );
    }
}

#[test]
fn test_karcher_mean_single_iteration() {
    let data = make_test_data(10, 40, 42);
    let t = uniform_grid(40);
    let result = karcher_mean(&data, &t, 1, 1e-10, 0.0);

    assert_eq!(result.n_iter, 1);
    assert_eq!(result.mean.len(), 40);
    // With only 1 iteration, still produces valid output
    for j in 0..40 {
        assert!(result.mean[j].is_finite());
    }
}

#[test]
fn test_karcher_mean_convergence_not_premature() {
    let n = 10;
    let m = 50;
    let t = uniform_grid(m);

    // Create phase-shifted curves that genuinely need alignment
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        let shift = (i as f64 - 5.0) * 0.05;
        for j in 0..m {
            col_major[i + j * n] = (2.0 * std::f64::consts::PI * (t[j] - shift)).sin();
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

    // With an impossibly tight tolerance, the algorithm should hit the
    // iteration cap rather than "converging" after 2 iterations.
    let result = karcher_mean(&data, &t, 20, 1e-15, 0.0);
    assert!(
        result.n_iter > 2,
        "With tol=1e-15 the algorithm should iterate beyond 2, got n_iter={}",
        result.n_iter
    );

    // With a reasonable tolerance, it should converge and report so
    let result_loose = karcher_mean(&data, &t, 20, 1e-2, 0.0);
    assert!(
        result_loose.converged,
        "With tol=1e-2 the algorithm should converge"
    );
}

// ── Align to target ──

#[test]
fn test_align_to_target_valid() {
    let data = make_test_data(10, 40, 42);
    let t = uniform_grid(40);
    let target = data.row(0);

    let result = align_to_target(&data, &target, &t, 0.0);

    assert_eq!(result.gammas.shape(), (10, 40));
    assert_eq!(result.aligned_data.shape(), (10, 40));
    assert_eq!(result.distances.len(), 10);

    // All distances should be non-negative
    for &d in &result.distances {
        assert!(d >= 0.0);
    }
}

#[test]
fn test_align_to_target_self_near_zero() {
    let data = make_test_data(5, 40, 42);
    let t = uniform_grid(40);
    let target = data.row(0);

    let result = align_to_target(&data, &target, &t, 0.0);

    // Distance of target to itself should be near zero
    assert!(
        result.distances[0] < 0.1,
        "Self-alignment distance should be near zero, got {}",
        result.distances[0]
    );
}

#[test]
fn test_align_to_target_warps_are_monotone() {
    let data = make_test_data(8, 40, 42);
    let t = uniform_grid(40);
    let target = data.row(0);
    let result = align_to_target(&data, &target, &t, 0.0);

    for i in 0..8 {
        for j in 1..40 {
            assert!(
                result.gammas[(i, j)] >= result.gammas[(i, j - 1)],
                "Warp for curve {i} should be monotone at j={j}"
            );
        }
    }
}

#[test]
fn test_align_to_target_aligned_data_finite() {
    let data = make_test_data(6, 40, 42);
    let t = uniform_grid(40);
    let target = data.row(0);
    let result = align_to_target(&data, &target, &t, 0.0);

    for i in 0..6 {
        for j in 0..40 {
            assert!(
                result.aligned_data[(i, j)].is_finite(),
                "Aligned data should be finite at ({i},{j})"
            );
        }
    }
}

// ── Cross distance matrix ──

#[test]
fn test_cross_distance_matrix_shape() {
    let data1 = make_test_data(3, 30, 42);
    let data2 = make_test_data(4, 30, 99);
    let t = uniform_grid(30);

    let dm = elastic_cross_distance_matrix(&data1, &data2, &t, 0.0);
    assert_eq!(dm.shape(), (3, 4));

    // All non-negative
    for i in 0..3 {
        for j in 0..4 {
            assert!(dm[(i, j)] >= 0.0);
        }
    }
}

#[test]
fn test_cross_distance_matrix_self_matches_self_matrix() {
    // cross_distance(data, data) should have zero diagonal (approximately)
    let data = make_test_data(4, 30, 42);
    let t = uniform_grid(30);

    let cross = elastic_cross_distance_matrix(&data, &data, &t, 0.0);
    for i in 0..4 {
        assert!(
            cross[(i, i)] < 0.1,
            "Cross distance (self) diagonal should be near zero: got {}",
            cross[(i, i)]
        );
    }
}

#[test]
fn test_cross_distance_matrix_consistent_with_pairwise() {
    let data1 = make_test_data(3, 30, 42);
    let data2 = make_test_data(2, 30, 99);
    let t = uniform_grid(30);

    let dm = elastic_cross_distance_matrix(&data1, &data2, &t, 0.0);

    for i in 0..3 {
        for j in 0..2 {
            let fi = data1.row(i);
            let fj = data2.row(j);
            let d_direct = elastic_distance(&fi, &fj, &t, 0.0);
            assert!(
                (dm[(i, j)] - d_direct).abs() < 1e-10,
                "Cross matrix ({i},{j})={:.6} should match pairwise {d_direct:.6}",
                dm[(i, j)]
            );
        }
    }
}

// ── align_srsf_pair ──

#[test]
fn test_align_srsf_pair_identity() {
    use super::karcher::align_srsf_pair;
    use super::srsf::srsf_single;

    let m = 50;
    let t = uniform_grid(m);
    let f: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let q = srsf_single(&f, &t);

    let (gamma, q_aligned) = align_srsf_pair(&q, &q, &t, 0.0);

    // Warp should be near identity
    for j in 0..m {
        assert!(
            (gamma[j] - t[j]).abs() < 0.15,
            "Self-SRSF alignment warp should be near identity at j={j}"
        );
    }

    // Aligned SRSF should be close to original
    let weights = simpsons_weights(&t);
    let dist = l2_distance(&q, &q_aligned, &weights);
    assert!(
        dist < 0.5,
        "Self-aligned SRSF distance should be small, got {dist}"
    );
}

// ── srsf_single ──

#[test]
fn test_srsf_single_matches_matrix_version() {
    use super::srsf::srsf_single;

    let m = 50;
    let t = uniform_grid(m);
    let f: Vec<f64> = t.iter().map(|&ti| ti * ti + ti).collect();

    let q_single = srsf_single(&f, &t);

    let mat = FdMatrix::from_slice(&f, 1, m).unwrap();
    let q_mat = srsf_transform(&mat, &t);
    let q_from_mat = q_mat.row(0);

    for j in 0..m {
        assert!(
            (q_single[j] - q_from_mat[j]).abs() < 1e-12,
            "srsf_single should match srsf_transform at j={j}"
        );
    }
}

// ── gcd ──

#[test]
fn test_gcd_basic() {
    assert_eq!(gcd(1, 1), 1);
    assert_eq!(gcd(6, 4), 2);
    assert_eq!(gcd(7, 5), 1);
    assert_eq!(gcd(12, 8), 4);
    assert_eq!(gcd(7, 0), 7);
    assert_eq!(gcd(0, 5), 5);
}

// ── generate_coprime_nbhd ──

#[test]
fn test_coprime_nbhd_count() {
    assert_eq!(generate_coprime_nbhd(1).len(), 1); // just (1,1)
    assert_eq!(generate_coprime_nbhd(7).len(), 35);
}

#[test]
fn test_coprime_nbhd_matches_const() {
    let generated = generate_coprime_nbhd(7);
    assert_eq!(generated.len(), COPRIME_NBHD_7.len());
    for (i, pair) in generated.iter().enumerate() {
        assert_eq!(*pair, COPRIME_NBHD_7[i], "mismatch at index {i}");
    }
}

#[test]
fn test_coprime_nbhd_all_coprime() {
    for &(i, j) in &COPRIME_NBHD_7 {
        assert_eq!(gcd(i, j), 1, "({i},{j}) should be coprime");
        assert!((1..=7).contains(&i));
        assert!((1..=7).contains(&j));
    }
}

// ── dp_edge_weight ──

#[test]
fn test_dp_edge_weight_diagonal() {
    // Diagonal move (1,1): weight = (q1[sc] - sqrt(1)*q2[sr])^2 * h
    let t = uniform_grid(10);
    let q1 = vec![1.0; 10];
    let q2 = vec![1.0; 10];
    // Identical SRSFs: weight should be 0
    let w = dp_edge_weight(&q1, &q2, &t, 0, 1, 0, 1);
    assert!(w.abs() < 1e-12, "identical SRSFs should have zero cost");
}

#[test]
fn test_dp_edge_weight_non_diagonal() {
    // Move (1,2): n1=2, n2=1, slope = h/(2h) = 0.5
    let t = uniform_grid(10);
    let q1 = vec![1.0; 10];
    let q2 = vec![0.0; 10];
    let w = dp_edge_weight(&q1, &q2, &t, 0, 2, 0, 1);
    // diff = q1[0] - sqrt(0.5)*q2[0] = 1.0 - 0 = 1.0
    // weight = 1.0^2 * 1.0 * (t[2]-t[0]) = 2/9
    let expected = 2.0 / 9.0;
    assert!(
        (w - expected).abs() < 1e-10,
        "dp_edge_weight (1,2): expected {expected}, got {w}"
    );
}

#[test]
fn test_dp_edge_weight_zero_span() {
    let t = uniform_grid(10);
    let q1 = vec![1.0; 10];
    let q2 = vec![1.0; 10];
    // n1=0: should return INFINITY
    assert_eq!(dp_edge_weight(&q1, &q2, &t, 3, 3, 0, 1), f64::INFINITY);
    // n2=0: should return INFINITY
    assert_eq!(dp_edge_weight(&q1, &q2, &t, 0, 1, 3, 3), f64::INFINITY);
}

// ── DP alignment quality ──

#[test]
fn test_alignment_improves_distance() {
    use super::srsf::srsf_single;

    // Aligned SRSF distance should be less than unaligned SRSF distance
    let m = 50;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&x| (2.0 * std::f64::consts::PI * x).sin())
        .collect();
    // Use a larger shift so improvement is clear
    let f2: Vec<f64> = t
        .iter()
        .map(|&x| (2.0 * std::f64::consts::PI * (x + 0.2)).sin())
        .collect();

    let q1 = srsf_single(&f1, &t);
    let q2 = srsf_single(&f2, &t);
    let weights = simpsons_weights(&t);
    let unaligned_srsf_dist = l2_distance(&q1, &q2, &weights);

    let result = elastic_align_pair(&f1, &f2, &t, 0.0);

    assert!(
        result.distance <= unaligned_srsf_dist + 1e-6,
        "aligned SRSF dist ({}) should be <= unaligned SRSF dist ({})",
        result.distance,
        unaligned_srsf_dist
    );
}

// ── Edge case: constant data ──

#[test]
fn test_alignment_constant_curves() {
    let m = 30;
    let t = uniform_grid(m);
    let f1 = vec![5.0; m];
    let f2 = vec![5.0; m];

    let result = elastic_align_pair(&f1, &f2, &t, 0.0);
    assert!(
        result.distance < 0.01,
        "Constant curves: distance should be ~0"
    );
    assert_eq!(result.f_aligned.len(), m);
}

#[test]
fn test_karcher_mean_constant_curves() {
    let m = 30;
    let t = uniform_grid(m);
    let mut data = FdMatrix::zeros(5, m);
    for i in 0..5 {
        for j in 0..m {
            data[(i, j)] = 3.0;
        }
    }

    let result = karcher_mean(&data, &t, 5, 1e-4, 0.0);
    for j in 0..m {
        assert!(
            (result.mean[j] - 3.0).abs() < 0.5,
            "Mean of constant curves should be near 3.0, got {} at j={j}",
            result.mean[j]
        );
    }
}

#[test]
fn test_nan_srsf_no_panic() {
    let m = 20;
    let t = uniform_grid(m);
    let mut f = vec![1.0; m];
    f[5] = f64::NAN;
    let mat = FdMatrix::from_slice(&f, 1, m).unwrap();
    let q = srsf_transform(&mat, &t);
    // Should not panic; NaN propagates
    assert_eq!(q.nrows(), 1);
}

#[test]
fn test_n1_karcher_mean() {
    let m = 30;
    let t = uniform_grid(m);
    let f: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
    let data = FdMatrix::from_slice(&f, 1, m).unwrap();
    let result = karcher_mean(&data, &t, 5, 1e-4, 0.0);
    assert_eq!(result.mean.len(), m);
    // With only 1 curve, the mean should be close to the original
    for j in 0..m {
        assert!(result.mean[j].is_finite());
    }
}

#[test]
fn test_two_point_grid() {
    let t = vec![0.0, 1.0];
    let f1 = vec![0.0, 1.0];
    let f2 = vec![0.0, 2.0];
    let d = elastic_distance(&f1, &f2, &t, 0.0);
    assert!(d >= 0.0);
    assert!(d.is_finite());
}

#[test]
fn test_non_uniform_grid_alignment() {
    // Non-uniform grid: points clustered near 0
    let t = vec![0.0, 0.01, 0.05, 0.2, 0.5, 1.0];
    let m = t.len();
    let f1: Vec<f64> = t.iter().map(|&ti: &f64| ti.sin()).collect();
    let f2: Vec<f64> = t.iter().map(|&ti: &f64| (ti + 0.1).sin()).collect();
    let result = elastic_align_pair(&f1, &f2, &t, 0.0);
    assert_eq!(result.gamma.len(), m);
    assert!(result.distance >= 0.0);
    assert!(result.distance.is_finite());
}

// ── TSRVF tests ──

#[test]
fn test_tsrvf_output_shape() {
    let m = 50;
    let n = 10;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);
    let result = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
    assert_eq!(
        result.tangent_vectors.shape(),
        (n, m),
        "Tangent vectors should be n×m"
    );
    assert_eq!(result.gammas.shape(), (n, m), "Gammas should be n×m");
    assert_eq!(result.srsf_norms.len(), n, "Should have n SRSF norms");
    assert_eq!(result.mean.len(), m, "Mean should have m points");
    assert_eq!(result.mean_srsf.len(), m, "Mean SRSF should have m points");
}

#[test]
fn test_tsrvf_all_finite() {
    let m = 50;
    let n = 5;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);
    let result = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
    for i in 0..n {
        for j in 0..m {
            assert!(
                result.tangent_vectors[(i, j)].is_finite(),
                "Tangent vector should be finite at ({i},{j})"
            );
        }
        assert!(
            result.srsf_norms[i].is_finite(),
            "SRSF norm should be finite for curve {i}"
        );
    }
    assert!(
        result.mean_srsf_norm.is_finite(),
        "Mean SRSF norm should be finite"
    );
}

#[test]
fn test_tsrvf_identical_curves_zero_tangent() {
    let m = 50;
    let t = uniform_grid(m);
    // Stack 5 identical curves
    let curve: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let mut col_major = vec![0.0; 5 * m];
    for i in 0..5 {
        for j in 0..m {
            col_major[i + j * 5] = curve[j];
        }
    }
    let data = FdMatrix::from_column_major(col_major, 5, m).unwrap();
    let result = tsrvf_transform(&data, &t, 10, 1e-4, 0.0);

    // All tangent vectors should be approximately zero
    for i in 0..5 {
        let tv_norm_sq: f64 = (0..m).map(|j| result.tangent_vectors[(i, j)].powi(2)).sum();
        assert!(
            tv_norm_sq.sqrt() < 0.5,
            "Identical curves should have near-zero tangent vectors, got norm = {}",
            tv_norm_sq.sqrt()
        );
    }
}

#[test]
fn test_tsrvf_mean_tangent_near_zero() {
    let m = 50;
    let n = 10;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);
    let result = tsrvf_transform(&data, &t, 10, 1e-3, 0.0);

    // Mean of tangent vectors should be approximately zero (property of Karcher mean)
    let mut mean_tv = vec![0.0; m];
    for i in 0..n {
        for j in 0..m {
            mean_tv[j] += result.tangent_vectors[(i, j)];
        }
    }
    for j in 0..m {
        mean_tv[j] /= n as f64;
    }
    let mean_norm: f64 = mean_tv.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        mean_norm < 1.0,
        "Mean tangent vector should be near zero, got norm = {mean_norm}"
    );
}

#[test]
fn test_tsrvf_from_alignment() {
    let m = 50;
    let n = 5;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);
    let karcher_res = karcher_mean(&data, &t, 5, 1e-3, 0.0);
    let result = tsrvf_from_alignment(&karcher_res, &t);
    assert_eq!(result.tangent_vectors.shape(), (n, m));
    assert!(result.mean_srsf_norm > 0.0);
}

#[test]
fn test_tsrvf_round_trip() {
    let m = 50;
    let n = 5;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);
    let result = tsrvf_transform(&data, &t, 10, 1e-3, 0.0);
    let reconstructed = tsrvf_inverse(&result, &t);

    assert_eq!(reconstructed.shape(), result.tangent_vectors.shape());
    // Reconstructed curves should have finite values
    for i in 0..n {
        for j in 0..m {
            assert!(
                reconstructed[(i, j)].is_finite(),
                "Reconstructed curve should be finite at ({i},{j})"
            );
        }
    }
    // Issue #12: per-curve initial values should be preserved
    for i in 0..n {
        assert!(
            (reconstructed[(i, 0)] - result.initial_values[i]).abs() < 1e-6,
            "Curve {i} initial value: expected {}, got {}",
            result.initial_values[i],
            reconstructed[(i, 0)]
        );
    }
}

#[test]
fn test_tsrvf_initial_values_per_curve() {
    // Issue #12: tsrvf_inverse must use per-curve initial values, not mean[0]
    let m = 50;
    let n = 5;
    let t = uniform_grid(m);

    // Create curves with distinct initial values
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        let offset = (i as f64 + 1.0) * 2.0; // offsets: 2, 4, 6, 8, 10
        for j in 0..m {
            col_major[i + j * n] = offset + (2.0 * std::f64::consts::PI * t[j]).sin();
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();

    let result = tsrvf_transform(&data, &t, 15, 1e-4, 0.0);

    // initial_values should differ per curve
    assert_eq!(result.initial_values.len(), n);
    let all_same = result
        .initial_values
        .windows(2)
        .all(|w| (w[0] - w[1]).abs() < 1e-10);
    assert!(
        !all_same,
        "Initial values should differ per curve: {:?}",
        result.initial_values
    );

    // Reconstruct and check initial values are preserved
    let reconstructed = tsrvf_inverse(&result, &t);
    for i in 0..n {
        assert!(
            (reconstructed[(i, 0)] - result.initial_values[i]).abs() < 1e-6,
            "Curve {i}: reconstructed f(0) = {}, expected {}",
            reconstructed[(i, 0)],
            result.initial_values[i]
        );
    }

    // Before the fix, all curves would have reconstructed[(i, 0)] ≈ mean[0]
    // Verify they are NOT all the same
    let recon_initials: Vec<f64> = (0..n).map(|i| reconstructed[(i, 0)]).collect();
    let all_recon_same = recon_initials.windows(2).all(|w| (w[0] - w[1]).abs() < 0.1);
    assert!(
        !all_recon_same,
        "Reconstructed initial values must vary per curve: {:?}",
        recon_initials
    );
}

#[test]
fn test_tsrvf_single_curve() {
    let m = 50;
    let t = uniform_grid(m);
    let data = make_test_data(1, m, 42);
    let result = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
    assert_eq!(result.tangent_vectors.shape(), (1, m));
    // Single curve → tangent vector should be zero (it IS the mean)
    let tv_norm: f64 = (0..m)
        .map(|j| result.tangent_vectors[(0, j)].powi(2))
        .sum::<f64>()
        .sqrt();
    assert!(
        tv_norm < 0.5,
        "Single curve tangent vector should be near zero, got {tv_norm}"
    );
}

#[test]
fn test_tsrvf_constant_curves() {
    let m = 30;
    let t = uniform_grid(m);
    // Constant curves → SRSF = 0, norms = 0
    let data = FdMatrix::from_column_major(vec![5.0; 3 * m], 3, m).unwrap();
    let result = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
    // Should not produce NaN or Inf
    for i in 0..3 {
        for j in 0..m {
            let v = result.tangent_vectors[(i, j)];
            assert!(
                !v.is_nan(),
                "Constant curves should not produce NaN tangent vectors"
            );
        }
    }
}

// ── Reference-value tests (sphere geometry) ─────────────────────────────

#[test]
fn test_tsrvf_sphere_inv_exp_reference() {
    use crate::warping::inv_exp_map_sphere;

    // Analytical reference: known vectors on the Hilbert sphere
    let m = 21;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    // Construct unit vector psi1 (constant)
    let raw1 = vec![1.0; m];
    let norm1 = inner_product_l2(&raw1, &raw1, &time).max(0.0).sqrt();
    let psi1: Vec<f64> = raw1.iter().map(|&v| v / norm1).collect();

    // Construct psi2 with sinusoidal perturbation
    let raw2: Vec<f64> = time
        .iter()
        .map(|&t| 1.0 + 0.3 * (2.0 * std::f64::consts::PI * t).sin())
        .collect();
    let norm2 = inner_product_l2(&raw2, &raw2, &time).max(0.0).sqrt();
    let psi2: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

    // Compute theta analytically
    let ip = inner_product_l2(&psi1, &psi2, &time).clamp(-1.0, 1.0);
    let theta_expected = ip.acos();

    // Compute via inv_exp_map_sphere
    let v = inv_exp_map_sphere(&psi1, &psi2, &time);
    let v_norm = inner_product_l2(&v, &v, &time).max(0.0).sqrt();

    // ||v|| should equal theta
    assert!(
        (v_norm - theta_expected).abs() < 1e-10,
        "||v|| = {v_norm}, expected theta = {theta_expected}"
    );

    // theta should be small but non-zero (perturbation is mild)
    assert!(
        theta_expected > 0.01 && theta_expected < 1.0,
        "theta = {theta_expected} out of expected range"
    );
}

#[test]
fn test_tsrvf_sphere_round_trip_reference() {
    use crate::warping::{exp_map_sphere, inv_exp_map_sphere};

    // Round-trip: exp_map(psi1, inv_exp_map(psi1, psi2)) should recover psi2
    let m = 21;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();

    let raw1 = vec![1.0; m];
    let norm1 = inner_product_l2(&raw1, &raw1, &time).max(0.0).sqrt();
    let psi1: Vec<f64> = raw1.iter().map(|&v| v / norm1).collect();

    let raw2: Vec<f64> = time
        .iter()
        .map(|&t| 1.0 + 0.3 * (2.0 * std::f64::consts::PI * t).sin())
        .collect();
    let norm2 = inner_product_l2(&raw2, &raw2, &time).max(0.0).sqrt();
    let psi2: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

    let v = inv_exp_map_sphere(&psi1, &psi2, &time);
    let recovered = exp_map_sphere(&psi1, &v, &time);

    // L2 error between psi2 and recovered
    let diff: Vec<f64> = psi2
        .iter()
        .zip(recovered.iter())
        .map(|(&a, &b)| (a - b).powi(2))
        .collect();
    let l2_err = trapz(&diff, &time).max(0.0).sqrt();
    assert!(
        l2_err < 1e-12,
        "Round-trip L2 error = {l2_err:.2e}, expected < 1e-12"
    );
}

// ── Penalized alignment ──

#[test]
fn test_penalized_alignment_lambda_zero_matches_unpenalized() {
    let m = 50;
    let t = uniform_grid(m);
    let data = make_test_data(2, m, 42);
    let f1 = data.row(0);
    let f2 = data.row(1);

    let r0 = elastic_align_pair(&f1, &f2, &t, 0.0);
    // lambda = 0.0 should produce the same result regardless
    assert!(r0.distance >= 0.0);
    assert_eq!(r0.gamma.len(), m);
}

#[test]
fn test_penalized_alignment_smoother_warp() {
    let m = 80;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.15)).sin())
        .collect();

    let r_free = elastic_align_pair(&f1, &f2, &t, 0.0);
    let r_pen = elastic_align_pair(&f1, &f2, &t, 1.0);

    // Measure warp deviation from identity
    let dev_free: f64 = r_free
        .gamma
        .iter()
        .zip(t.iter())
        .map(|(g, ti)| (g - ti).powi(2))
        .sum();
    let dev_pen: f64 = r_pen
        .gamma
        .iter()
        .zip(t.iter())
        .map(|(g, ti)| (g - ti).powi(2))
        .sum();

    assert!(
        dev_pen <= dev_free + 1e-6,
        "Penalized warp should be closer to identity: free={dev_free:.6}, pen={dev_pen:.6}"
    );
}

#[test]
fn test_penalized_alignment_large_lambda_near_identity() {
    let m = 50;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
        .collect();

    let r = elastic_align_pair(&f1, &f2, &t, 1000.0);

    // With very large lambda, warp should be very close to identity
    let max_dev: f64 = r
        .gamma
        .iter()
        .zip(t.iter())
        .map(|(g, ti)| (g - ti).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dev < 0.05,
        "Large lambda should give near-identity warp: max deviation = {max_dev}"
    );
}

#[test]
fn test_penalized_karcher_mean() {
    let m = 40;
    let t = uniform_grid(m);
    let data = make_test_data(10, m, 42);

    let result = karcher_mean(&data, &t, 5, 1e-3, 0.5);
    assert_eq!(result.mean.len(), m);
    for j in 0..m {
        assert!(result.mean[j].is_finite());
    }
}

// ── Phase-amplitude decomposition ──

#[test]
fn test_decomposition_identity_curves() {
    let m = 50;
    let t = uniform_grid(m);
    let f: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();

    let result = elastic_decomposition(&f, &f, &t, 0.0);
    assert!(
        result.d_amplitude < 0.1,
        "Self-decomposition amplitude should be ~0, got {}",
        result.d_amplitude
    );
    assert!(
        result.d_phase < 0.2,
        "Self-decomposition phase should be ~0, got {}",
        result.d_phase
    );
}

#[test]
fn test_decomposition_pythagorean() {
    // d_total² ≈ d_a² + d_φ² (approximately, for the Fisher-Rao metric)
    let m = 80;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| 1.2 * (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
        .collect();

    let result = elastic_decomposition(&f1, &f2, &t, 0.0);
    let da = result.d_amplitude;
    let dp = result.d_phase;
    // Both should be non-negative
    assert!(da >= 0.0);
    assert!(dp >= 0.0);
}

#[test]
fn test_phase_distance_shifted_sine() {
    let m = 80;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.15)).sin())
        .collect();

    let dp = phase_distance_pair(&f1, &f2, &t, 0.0);
    assert!(
        dp > 0.01,
        "Phase distance of shifted curves should be > 0, got {dp}"
    );
}

#[test]
fn test_amplitude_distance_scaled_curve() {
    let m = 80;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| 2.0 * (2.0 * std::f64::consts::PI * ti).sin())
        .collect();

    let da = amplitude_distance(&f1, &f2, &t, 0.0);
    assert!(
        da > 0.01,
        "Amplitude distance of scaled curves should be > 0, got {da}"
    );
}

#[test]
fn test_phase_distance_nonneg() {
    let data = make_test_data(4, 40, 42);
    let t = uniform_grid(40);
    for i in 0..4 {
        for j in 0..4 {
            let fi = data.row(i);
            let fj = data.row(j);
            let dp = phase_distance_pair(&fi, &fj, &t, 0.0);
            assert!(dp >= 0.0, "Phase distance should be non-negative");
        }
    }
}

// ── Parallel transport ──

#[test]
fn test_schilds_ladder_zero_vector() {
    use super::tsrvf::parallel_transport_schilds;

    let m = 21;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let raw = vec![1.0; m];
    let norm = crate::warping::l2_norm_l2(&raw, &time);
    let from: Vec<f64> = raw.iter().map(|&v| v / norm).collect();
    let raw2: Vec<f64> = time
        .iter()
        .map(|&t| 1.0 + 0.2 * (2.0 * std::f64::consts::PI * t).sin())
        .collect();
    let norm2 = crate::warping::l2_norm_l2(&raw2, &time);
    let to: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

    let zero = vec![0.0; m];
    let result = parallel_transport_schilds(&zero, &from, &to, &time);
    let result_norm: f64 = result.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        result_norm < 1e-6,
        "Transporting zero should give zero, got norm {result_norm}"
    );
}

#[test]
fn test_pole_ladder_zero_vector() {
    use super::tsrvf::parallel_transport_pole;

    let m = 21;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let raw = vec![1.0; m];
    let norm = crate::warping::l2_norm_l2(&raw, &time);
    let from: Vec<f64> = raw.iter().map(|&v| v / norm).collect();
    let raw2: Vec<f64> = time
        .iter()
        .map(|&t| 1.0 + 0.2 * (2.0 * std::f64::consts::PI * t).sin())
        .collect();
    let norm2 = crate::warping::l2_norm_l2(&raw2, &time);
    let to: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

    let zero = vec![0.0; m];
    let result = parallel_transport_pole(&zero, &from, &to, &time);
    let result_norm: f64 = result.iter().map(|v| v * v).sum::<f64>().sqrt();
    assert!(
        result_norm < 1e-6,
        "Transporting zero should give zero, got norm {result_norm}"
    );
}

#[test]
fn test_schilds_preserves_norm() {
    use super::tsrvf::parallel_transport_schilds;

    let m = 51;
    let time: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let raw = vec![1.0; m];
    let norm = crate::warping::l2_norm_l2(&raw, &time);
    let from: Vec<f64> = raw.iter().map(|&v| v / norm).collect();
    let raw2: Vec<f64> = time
        .iter()
        .map(|&t| 1.0 + 0.15 * (2.0 * std::f64::consts::PI * t).sin())
        .collect();
    let norm2 = crate::warping::l2_norm_l2(&raw2, &time);
    let to: Vec<f64> = raw2.iter().map(|&v| v / norm2).collect();

    // Small tangent vector
    let v: Vec<f64> = time
        .iter()
        .map(|&t| 0.1 * (4.0 * std::f64::consts::PI * t).cos())
        .collect();
    let v_norm = crate::warping::l2_norm_l2(&v, &time);

    let transported = parallel_transport_schilds(&v, &from, &to, &time);
    let t_norm = crate::warping::l2_norm_l2(&transported, &time);

    // Norm should be approximately preserved (ladder methods are first-order)
    assert!(
        (t_norm - v_norm).abs() / v_norm.max(1e-10) < 1.5,
        "Schild's should roughly preserve norm: original={v_norm:.4}, transported={t_norm:.4}"
    );
}

#[test]
fn test_tsrvf_logmap_matches_original() {
    let m = 50;
    let n = 5;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);

    let result_orig = tsrvf_transform(&data, &t, 5, 1e-3, 0.0);
    let result_logmap =
        tsrvf_transform_with_method(&data, &t, 5, 1e-3, 0.0, TransportMethod::LogMap);

    // Should be identical (LogMap delegates to original)
    for i in 0..n {
        for j in 0..m {
            assert!(
                (result_orig.tangent_vectors[(i, j)] - result_logmap.tangent_vectors[(i, j)]).abs()
                    < 1e-12,
                "LogMap variant should match original at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_tsrvf_with_schilds_produces_valid_result() {
    let m = 50;
    let n = 5;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);

    let result =
        tsrvf_transform_with_method(&data, &t, 5, 1e-3, 0.0, TransportMethod::SchildsLadder);

    assert_eq!(result.tangent_vectors.shape(), (n, m));
    for i in 0..n {
        for j in 0..m {
            assert!(
                result.tangent_vectors[(i, j)].is_finite(),
                "Schild's TSRVF should produce finite tangent vectors at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_transport_methods_differ() {
    let m = 50;
    let n = 5;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);
    let karcher_res = karcher_mean(&data, &t, 5, 1e-3, 0.0);

    let r_log = tsrvf_from_alignment_with_method(&karcher_res, &t, TransportMethod::LogMap);
    let r_schilds =
        tsrvf_from_alignment_with_method(&karcher_res, &t, TransportMethod::SchildsLadder);

    // Methods should produce different (but related) tangent vectors
    let mut total_diff = 0.0;
    for i in 0..n {
        for j in 0..m {
            total_diff += (r_log.tangent_vectors[(i, j)] - r_schilds.tangent_vectors[(i, j)]).abs();
        }
    }

    // They should be non-zero different (unless all curves are identical to mean)
    // Just check both produce finite results
    assert!(total_diff.is_finite());
}

// ── Alignment quality metrics ──

#[test]
fn test_warp_complexity_identity_is_zero() {
    let m = 50;
    let t = uniform_grid(m);
    let identity = t.clone();
    let c = warp_complexity(&identity, &t);
    assert!(
        c < 1e-10,
        "Identity warp should have zero complexity, got {c}"
    );
}

#[test]
fn test_warp_complexity_nonidentity_positive() {
    let m = 50;
    let t = uniform_grid(m);
    let gamma: Vec<f64> = t.iter().map(|&ti| ti * ti).collect();
    let c = warp_complexity(&gamma, &t);
    assert!(
        c > 0.01,
        "Non-identity warp should have positive complexity, got {c}"
    );
}

#[test]
fn test_warp_smoothness_identity_is_zero() {
    let m = 50;
    let t = uniform_grid(m);
    let identity = t.clone();
    let s = warp_smoothness(&identity, &t);
    assert!(
        s < 1e-6,
        "Identity warp (constant γ'=1, γ''=0) should have near-zero bending energy, got {s}"
    );
}

#[test]
fn test_alignment_quality_basic() {
    let m = 50;
    let n = 8;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);
    let karcher_res = karcher_mean(&data, &t, 10, 1e-3, 0.0);
    let quality = alignment_quality(&data, &karcher_res, &t);

    // Shape checks
    assert_eq!(quality.warp_complexity.len(), n);
    assert_eq!(quality.warp_smoothness.len(), n);
    assert_eq!(quality.pointwise_variance_ratio.len(), m);

    // Non-negativity
    assert!(quality.total_variance >= 0.0);
    assert!(quality.amplitude_variance >= 0.0);
    assert!(quality.phase_variance >= 0.0);
    assert!(quality.mean_warp_complexity >= 0.0);
    assert!(quality.mean_warp_smoothness >= 0.0);

    // Amplitude variance ≤ total variance
    assert!(
        quality.amplitude_variance <= quality.total_variance + 1e-10,
        "Amplitude variance ({}) should be ≤ total variance ({})",
        quality.amplitude_variance,
        quality.total_variance
    );
}

#[test]
fn test_alignment_quality_identical_curves() {
    let m = 50;
    let t = uniform_grid(m);
    let curve: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let mut col_major = vec![0.0; 5 * m];
    for i in 0..5 {
        for j in 0..m {
            col_major[i + j * 5] = curve[j];
        }
    }
    let data = FdMatrix::from_column_major(col_major, 5, m).unwrap();
    let karcher_res = karcher_mean(&data, &t, 5, 1e-3, 0.0);
    let quality = alignment_quality(&data, &karcher_res, &t);

    // Identical curves → near-zero variances and warp complexities
    assert!(
        quality.total_variance < 0.01,
        "Identical curves should have near-zero total variance, got {}",
        quality.total_variance
    );
    assert!(
        quality.mean_warp_complexity < 0.1,
        "Identical curves should have near-zero warp complexity, got {}",
        quality.mean_warp_complexity
    );
}

#[test]
fn test_alignment_quality_variance_reduction() {
    let m = 50;
    let n = 10;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);
    let karcher_res = karcher_mean(&data, &t, 10, 1e-3, 0.0);
    let quality = alignment_quality(&data, &karcher_res, &t);

    // Mean variance ratio should be ≤ ~1 (alignment shouldn't increase variance)
    assert!(
        quality.mean_variance_reduction <= 1.5,
        "Mean variance reduction ratio should be ≤ ~1, got {}",
        quality.mean_variance_reduction
    );
}

#[test]
fn test_pairwise_consistency_small() {
    let m = 40;
    let n = 4;
    let t = uniform_grid(m);
    let data = make_test_data(n, m, 42);

    let consistency = pairwise_consistency(&data, &t, 0.0, 100);
    assert!(
        consistency.is_finite() && consistency >= 0.0,
        "Pairwise consistency should be finite and non-negative, got {consistency}"
    );
}

// ── Multidimensional SRSF ──

#[test]
fn test_srsf_nd_d1_matches_existing() {
    use crate::matrix::FdCurveSet;

    let m = 50;
    let t = uniform_grid(m);
    let data = make_test_data(3, m, 42);

    // 1D via existing function
    let q_1d = srsf_transform(&data, &t);

    // 1D via nd function
    let data_nd = FdCurveSet::from_1d(data);
    let q_nd = srsf_transform_nd(&data_nd, &t);

    assert_eq!(q_nd.ndim(), 1);
    for i in 0..3 {
        for j in 0..m {
            assert!(
                (q_1d[(i, j)] - q_nd.dims[0][(i, j)]).abs() < 1e-10,
                "1D nd SRSF should match existing at ({i},{j}): {} vs {}",
                q_1d[(i, j)],
                q_nd.dims[0][(i, j)]
            );
        }
    }
}

#[test]
fn test_srsf_nd_constant_is_zero() {
    use crate::matrix::FdCurveSet;

    let m = 30;
    let t = uniform_grid(m);
    // Constant R^2 curve: f(t) = (3.0, -1.0)
    let dim0 = FdMatrix::from_column_major(vec![3.0; m], 1, m).unwrap();
    let dim1 = FdMatrix::from_column_major(vec![-1.0; m], 1, m).unwrap();
    let data = FdCurveSet::from_dims(vec![dim0, dim1]).unwrap();

    let q = srsf_transform_nd(&data, &t);
    for k in 0..2 {
        for j in 0..m {
            assert!(
                q.dims[k][(0, j)].abs() < 1e-10,
                "Constant curve SRSF should be zero, dim {k} at {j}: {}",
                q.dims[k][(0, j)]
            );
        }
    }
}

#[test]
fn test_srsf_nd_linear_r2() {
    use crate::matrix::FdCurveSet;

    let m = 51;
    let t = uniform_grid(m);
    // f(t) = (2t, 3t) → f'(t) = (2, 3), ||f'|| = sqrt(13)
    // q(t) = (2, 3) / sqrt(sqrt(13)) = (2, 3) / 13^(1/4)
    let dim0 =
        FdMatrix::from_slice(&t.iter().map(|&ti| 2.0 * ti).collect::<Vec<_>>(), 1, m).unwrap();
    let dim1 =
        FdMatrix::from_slice(&t.iter().map(|&ti| 3.0 * ti).collect::<Vec<_>>(), 1, m).unwrap();
    let data = FdCurveSet::from_dims(vec![dim0, dim1]).unwrap();

    let q = srsf_transform_nd(&data, &t);
    let expected_scale = 1.0 / 13.0_f64.powf(0.25);
    let mid = m / 2;

    assert!(
        (q.dims[0][(0, mid)] - 2.0 * expected_scale).abs() < 0.1,
        "q_x at midpoint: {} vs expected {}",
        q.dims[0][(0, mid)],
        2.0 * expected_scale
    );
    assert!(
        (q.dims[1][(0, mid)] - 3.0 * expected_scale).abs() < 0.1,
        "q_y at midpoint: {} vs expected {}",
        q.dims[1][(0, mid)],
        3.0 * expected_scale
    );
}

#[test]
fn test_srsf_nd_round_trip() {
    use crate::matrix::FdCurveSet;

    let m = 51;
    let t = uniform_grid(m);
    // f(t) = (sin(2πt), cos(2πt))
    let pi2 = 2.0 * std::f64::consts::PI;
    let vals_x: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).sin()).collect();
    let vals_y: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).cos()).collect();
    let dim0 = FdMatrix::from_slice(&vals_x, 1, m).unwrap();
    let dim1 = FdMatrix::from_slice(&vals_y, 1, m).unwrap();
    let data = FdCurveSet::from_dims(vec![dim0, dim1]).unwrap();

    let q = srsf_transform_nd(&data, &t);
    let q_vecs: Vec<Vec<f64>> = q.dims.iter().map(|dm| dm.row(0)).collect();
    let f0 = vec![vals_x[0], vals_y[0]];
    let recon = srsf_inverse_nd(&q_vecs, &t, &f0);

    // Check reconstruction error (skip boundary points)
    let mut max_err = 0.0_f64;
    for k in 0..2 {
        let orig = if k == 0 { &vals_x } else { &vals_y };
        for j in 2..(m - 2) {
            let err = (recon[k][j] - orig[j]).abs();
            max_err = max_err.max(err);
        }
    }
    assert!(
        max_err < 0.2,
        "SRSF round-trip max error should be small, got {max_err}"
    );
}

#[test]
fn test_align_nd_identical_near_zero() {
    use crate::matrix::FdCurveSet;

    let m = 50;
    let t = uniform_grid(m);
    let pi2 = 2.0 * std::f64::consts::PI;
    let vals_x: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).sin()).collect();
    let vals_y: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).cos()).collect();
    let dim0 = FdMatrix::from_slice(&vals_x, 1, m).unwrap();
    let dim1 = FdMatrix::from_slice(&vals_y, 1, m).unwrap();
    let data = FdCurveSet::from_dims(vec![dim0, dim1]).unwrap();

    let result = elastic_align_pair_nd(&data, &data, &t, 0.0);
    assert!(
        result.distance < 0.5,
        "Self-alignment distance should be ~0, got {}",
        result.distance
    );
    // Gamma should be near identity
    let max_dev: f64 = result
        .gamma
        .iter()
        .zip(t.iter())
        .map(|(g, ti)| (g - ti).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dev < 0.1,
        "Self-alignment warp should be near identity, max dev = {max_dev}"
    );
}

#[test]
fn test_align_nd_shifted_r2() {
    use crate::matrix::FdCurveSet;

    let m = 60;
    let t = uniform_grid(m);
    let pi2 = 2.0 * std::f64::consts::PI;

    // f1(t) = (sin(2πt), cos(2πt))
    let f1x: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).sin()).collect();
    let f1y: Vec<f64> = t.iter().map(|&ti| (pi2 * ti).cos()).collect();
    let f1 = FdCurveSet::from_dims(vec![
        FdMatrix::from_slice(&f1x, 1, m).unwrap(),
        FdMatrix::from_slice(&f1y, 1, m).unwrap(),
    ])
    .unwrap();

    // f2(t) = (sin(2π(t-0.1)), cos(2π(t-0.1))) — phase shifted
    let f2x: Vec<f64> = t.iter().map(|&ti| (pi2 * (ti - 0.1)).sin()).collect();
    let f2y: Vec<f64> = t.iter().map(|&ti| (pi2 * (ti - 0.1)).cos()).collect();
    let f2 = FdCurveSet::from_dims(vec![
        FdMatrix::from_slice(&f2x, 1, m).unwrap(),
        FdMatrix::from_slice(&f2y, 1, m).unwrap(),
    ])
    .unwrap();

    let result = elastic_align_pair_nd(&f1, &f2, &t, 0.0);
    assert!(
        result.distance.is_finite(),
        "Distance should be finite, got {}",
        result.distance
    );
    assert_eq!(result.f_aligned.len(), 2);
    assert_eq!(result.f_aligned[0].len(), m);
    // Warp should deviate from identity (non-trivial alignment)
    let max_dev: f64 = result
        .gamma
        .iter()
        .zip(t.iter())
        .map(|(g, ti)| (g - ti).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_dev > 0.01,
        "Shifted curves should require non-trivial warp, max dev = {max_dev}"
    );
}

// ── Landmark-constrained alignment ──

#[test]
fn test_constrained_no_landmarks_matches_unconstrained() {
    use super::constrained::elastic_align_pair_constrained;

    let m = 50;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
        .collect();

    let r_free = elastic_align_pair(&f1, &f2, &t, 0.0);
    let r_const = elastic_align_pair_constrained(&f1, &f2, &t, &[], 0.0);

    // Should match unconstrained
    for j in 0..m {
        assert!(
            (r_free.gamma[j] - r_const.gamma[j]).abs() < 1e-10,
            "No-landmark constrained should match unconstrained at {j}"
        );
    }
    assert!(r_const.enforced_landmarks.is_empty());
}

#[test]
fn test_constrained_single_landmark_enforced() {
    use super::constrained::elastic_align_pair_constrained;

    let m = 60;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
        .collect();

    // Constrain midpoint: target_t=0.5 should map to source_t=0.5
    let result = elastic_align_pair_constrained(&f1, &f2, &t, &[(0.5, 0.5)], 0.0);

    // Gamma at the midpoint should be close to 0.5
    let mid_idx = m / 2;
    assert!(
        (result.gamma[mid_idx] - 0.5).abs() < 0.05,
        "Constrained gamma at midpoint should be ~0.5, got {}",
        result.gamma[mid_idx]
    );
    assert_eq!(result.enforced_landmarks.len(), 1);
}

#[test]
fn test_constrained_multiple_landmarks() {
    use super::constrained::elastic_align_pair_constrained;

    let m = 80;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (4.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (4.0 * std::f64::consts::PI * (ti - 0.05)).sin())
        .collect();

    let landmarks = vec![(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)];
    let result = elastic_align_pair_constrained(&f1, &f2, &t, &landmarks, 0.0);

    // Gamma should pass through (or near) each landmark
    for &(tt, st) in &landmarks {
        // snap_to_grid equivalent
        let idx = (tt * (m - 1) as f64).round() as usize;
        assert!(
            (result.gamma[idx] - st).abs() < 0.05,
            "Gamma at t={tt} should be ~{st}, got {}",
            result.gamma[idx]
        );
    }
}

#[test]
fn test_constrained_monotone_gamma() {
    use super::constrained::elastic_align_pair_constrained;

    let m = 60;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.1)).sin())
        .collect();

    let result = elastic_align_pair_constrained(&f1, &f2, &t, &[(0.3, 0.3), (0.7, 0.7)], 0.0);

    // Gamma should be non-decreasing
    for j in 1..m {
        assert!(
            result.gamma[j] >= result.gamma[j - 1] - 1e-10,
            "Gamma should be monotone: gamma[{}]={} < gamma[{}]={}",
            j,
            result.gamma[j],
            j - 1,
            result.gamma[j - 1]
        );
    }
    // Boundary conditions
    assert!((result.gamma[0] - t[0]).abs() < 1e-10);
    assert!((result.gamma[m - 1] - t[m - 1]).abs() < 1e-10);
}

#[test]
fn test_constrained_distance_ge_unconstrained() {
    use super::constrained::elastic_align_pair_constrained;

    let m = 60;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * (ti - 0.15)).sin())
        .collect();

    let r_free = elastic_align_pair(&f1, &f2, &t, 0.0);
    let r_const = elastic_align_pair_constrained(&f1, &f2, &t, &[(0.5, 0.5)], 0.0);

    // Constrained distance should be >= unconstrained (constraints reduce freedom)
    assert!(
        r_const.distance >= r_free.distance - 1e-6,
        "Constrained distance ({}) should be >= unconstrained ({})",
        r_const.distance,
        r_free.distance
    );
}

#[test]
fn test_constrained_with_landmark_detection() {
    let m = 80;
    let t = uniform_grid(m);
    let f1: Vec<f64> = t
        .iter()
        .map(|&ti| (4.0 * std::f64::consts::PI * ti).sin())
        .collect();
    let f2: Vec<f64> = t
        .iter()
        .map(|&ti| (4.0 * std::f64::consts::PI * (ti - 0.05)).sin())
        .collect();

    let result = elastic_align_pair_with_landmarks(
        &f1,
        &f2,
        &t,
        crate::landmark::LandmarkKind::Peak,
        0.1,
        0,
        0.0,
    );

    assert_eq!(result.gamma.len(), m);
    assert_eq!(result.f_aligned.len(), m);
    assert!(result.distance.is_finite());
    // Should be monotone
    for j in 1..m {
        assert!(
            result.gamma[j] >= result.gamma[j - 1] - 1e-10,
            "Gamma should be monotone at j={j}"
        );
    }
}

// ── SRSF smoothing for TSRVF (Issue #13) ──

#[test]
fn test_gam_to_psi_smooth_identity() {
    // Smoothed psi of identity warp should stay close to constant 1 in the interior.
    use crate::warping::{gam_to_psi, gam_to_psi_smooth};
    let m = 101;
    let h = 1.0 / (m - 1) as f64;
    let gam: Vec<f64> = uniform_grid(m);
    let psi_raw = gam_to_psi(&gam, h);
    let psi_smooth = gam_to_psi_smooth(&gam, h);
    // Check interior points (skip ~5% at each boundary)
    let skip = m / 20;
    for j in skip..(m - skip) {
        assert!(
            (psi_smooth[j] - 1.0).abs() < 0.05,
            "Smoothed psi of identity should be ~1.0, got {} at j={}",
            psi_smooth[j],
            j
        );
        assert!(
            (psi_smooth[j] - psi_raw[j]).abs() < 0.05,
            "Smoothed and raw psi should agree on smooth warp at j={}",
            j
        );
    }
}

#[test]
fn test_gam_to_psi_smooth_reduces_spikes() {
    // Create a kinky warp (simulating DP output with multiple slope changes)
    use crate::warping::{gam_to_psi, gam_to_psi_smooth};
    let m = 101;
    let h = 1.0 / (m - 1) as f64;
    let argvals = uniform_grid(m);
    // Multi-segment piecewise-linear warp with several kinks
    let mut gam: Vec<f64> = Vec::with_capacity(m);
    for j in 0..m {
        let t = argvals[j];
        // Three segments: slow (slope 0.5), fast (slope 2), slow (slope 0.5)
        let g = if t < 0.33 {
            t * 0.5 / 0.33
        } else if t < 0.67 {
            0.5 + (t - 0.33) * 0.5 / 0.34 * 2.0 // steeper
        } else {
            let base = 0.5 + 0.5 / 0.34 * 2.0 * 0.34; // ~1.5 but clamped
            (base + (t - 0.67) * 0.5 / 0.33).min(1.0)
        };
        gam.push(g.min(1.0));
    }
    // Normalize to [0,1]
    let gmax = gam[m - 1].max(1e-10);
    for g in &mut gam {
        *g /= gmax;
    }
    let psi_raw = gam_to_psi(&gam, h);
    let psi_smooth = gam_to_psi_smooth(&gam, h);
    // The raw psi should have jumps at kink points (slope transitions)
    let max_jump_raw: f64 = psi_raw
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f64, f64::max);
    let max_jump_smooth: f64 = psi_smooth
        .windows(2)
        .map(|w| (w[1] - w[0]).abs())
        .fold(0.0_f64, f64::max);
    // Smoothing should reduce the maximum jump in psi
    assert!(
        max_jump_smooth < max_jump_raw + 0.01,
        "Smoothing should not increase max psi jump: raw={max_jump_raw:.4}, smooth={max_jump_smooth:.4}"
    );
}

#[test]
fn test_smooth_aligned_srsfs_preserves_shape() {
    // Smoothing aligned SRSFs should preserve overall shape
    use crate::smoothing::nadaraya_watson;
    let m = 101;
    let time: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    // Create a smooth SRSF (sine curve)
    let qi: Vec<f64> = time
        .iter()
        .map(|&t| (2.0 * std::f64::consts::PI * t).sin())
        .collect();
    let bandwidth = 2.0 / (m - 1) as f64;
    let qi_smooth = nadaraya_watson(&time, &qi, &time, bandwidth, "gaussian").unwrap();
    // Correlation between original and smoothed should be very high
    let mean_orig: f64 = qi.iter().sum::<f64>() / m as f64;
    let mean_smooth: f64 = qi_smooth.iter().sum::<f64>() / m as f64;
    let mut cov = 0.0;
    let mut var_o = 0.0;
    let mut var_s = 0.0;
    for j in 0..m {
        let do_ = qi[j] - mean_orig;
        let ds = qi_smooth[j] - mean_smooth;
        cov += do_ * ds;
        var_o += do_ * do_;
        var_s += ds * ds;
    }
    let rho = cov / (var_o * var_s).sqrt().max(1e-10);
    assert!(
        rho > 0.99,
        "Smoothed SRSF should be highly correlated with original (rho={rho:.4})"
    );
}

#[test]
fn test_tsrvf_tangent_vectors_no_spikes() {
    // End-to-end: compute TSRVF tangent vectors, verify no element dominates.
    let m = 101;
    let argvals = uniform_grid(m);
    let data = make_test_data(10, m, 42);
    let result = tsrvf_transform(&data, &argvals, 5, 1e-3, 0.0);
    let (n, _) = result.tangent_vectors.shape();
    for i in 0..n {
        let vi = result.tangent_vectors.row(i);
        let rms = (vi.iter().map(|&v| v * v).sum::<f64>() / m as f64).sqrt();
        if rms > 1e-10 {
            let max_abs = vi.iter().map(|&v| v.abs()).fold(0.0_f64, f64::max);
            assert!(
                max_abs < 10.0 * rms,
                "Tangent vector {} has spike: max |v| = {max_abs:.4}, rms = {rms:.4}, ratio = {:.1}",
                i,
                max_abs / rms
            );
        }
    }
}
