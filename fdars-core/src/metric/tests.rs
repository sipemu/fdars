use super::*;
use crate::test_helpers::uniform_grid;
use std::f64::consts::PI;

#[test]
fn test_lp_self_distance() {
    let data = FdMatrix::from_column_major(vec![0.0, 1.0, 1.0, 2.0], 2, 2).unwrap();
    let argvals = vec![0.0, 1.0];
    let dist = lp_self_1d(&data, &argvals, 2.0, &[]);
    assert!((dist[(0, 1)] - 1.0).abs() < 0.1);
}

#[test]
fn test_lp_self_symmetric() {
    let n = 5;
    let m = 20;
    let argvals = uniform_grid(m);
    let mut flat = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            flat[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin();
        }
    }
    let data = FdMatrix::from_column_major(flat, n, m).unwrap();
    let dist = lp_self_1d(&data, &argvals, 2.0, &[]);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Distance matrix should be symmetric"
            );
        }
    }
}

#[test]
fn test_lp_self_diagonal_zero() {
    let n = 4;
    let m = 15;
    let argvals = uniform_grid(m);
    let flat: Vec<f64> = (0..(n * m)).map(|i| i as f64 * 0.1).collect();
    let data = FdMatrix::from_column_major(flat, n, m).unwrap();
    let dist = lp_self_1d(&data, &argvals, 2.0, &[]);
    for i in 0..n {
        assert!(dist[(i, i)].abs() < 1e-10, "Self-distance should be zero");
    }
}

#[test]
fn test_lp_cross_shape() {
    let n1 = 3;
    let n2 = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let data1 = FdMatrix::from_column_major((0..(n1 * m)).map(|i| i as f64 * 0.1).collect(), n1, m)
        .unwrap();
    let data2 = FdMatrix::from_column_major((0..(n2 * m)).map(|i| i as f64 * 0.2).collect(), n2, m)
        .unwrap();
    let dist = lp_cross_1d(&data1, &data2, &argvals, 2.0, &[]);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
}

#[test]
fn test_lp_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(lp_self_1d(&empty, &[], 2.0, &[]).is_empty());
    assert!(lp_cross_1d(&empty, &empty, &[], 2.0, &[]).is_empty());
}

#[test]
fn test_dtw_distance() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0];
    let dist = dtw_distance(&x, &y, 2.0, 10);
    assert!((dist - 0.0).abs() < 1e-10);
}

#[test]
fn test_dtw_distance_different() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![2.0, 3.0, 4.0];
    let dist = dtw_distance(&x, &y, 1.0, 10);
    assert!(
        dist > 0.0,
        "Different curves should have positive DTW distance"
    );
}

#[test]
fn test_dtw_self_symmetric() {
    let n = 4;
    let m = 15;
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| i as f64 * 0.1).collect(), n, m).unwrap();
    let dist = dtw_self_1d(&data, 2.0, 5);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "DTW matrix should be symmetric"
            );
        }
    }
}

#[test]
fn test_dtw_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(dtw_self_1d(&empty, 2.0, 5).is_empty());
}

#[test]
fn test_hausdorff_self_symmetric() {
    let n = 4;
    let m = 15;
    let argvals = uniform_grid(m);
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let dist = hausdorff_self_1d(&data, &argvals);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Hausdorff matrix should be symmetric"
            );
        }
    }
}

#[test]
fn test_hausdorff_self_diagonal_zero() {
    let n = 3;
    let m = 10;
    let argvals = uniform_grid(m);
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| i as f64 * 0.1).collect(), n, m).unwrap();
    let dist = hausdorff_self_1d(&data, &argvals);
    for i in 0..n {
        assert!(dist[(i, i)].abs() < 1e-10, "Self-distance should be zero");
    }
}

#[test]
fn test_hausdorff_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(hausdorff_self_1d(&empty, &[]).is_empty());
}

#[test]
fn test_fourier_self_symmetric() {
    let n = 4;
    let m = 32;
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let dist = fourier_self_1d(&data, 5);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Fourier distance should be symmetric"
            );
        }
    }
}

#[test]
fn test_fourier_self_diagonal_zero() {
    let n = 3;
    let m = 32;
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.2).cos()).collect(), n, m)
            .unwrap();
    let dist = fourier_self_1d(&data, 8);
    for i in 0..n {
        assert!(dist[(i, i)].abs() < 1e-10, "Self-distance should be zero");
    }
}

#[test]
fn test_fourier_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(fourier_self_1d(&empty, 5).is_empty());
}

#[test]
fn test_hshift_self_symmetric() {
    let n = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let dist = hshift_self_1d(&data, &argvals, 3);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Hshift distance should be symmetric"
            );
        }
    }
}

#[test]
fn test_hshift_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(hshift_self_1d(&empty, &[], 3).is_empty());
}

#[test]
fn test_hausdorff_3d_identical() {
    let points1 = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
    let points2 = vec![(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)];
    let dist = hausdorff_3d(&points1, &points2);
    assert!(
        dist.abs() < 1e-10,
        "Identical point sets should have zero distance"
    );
}

#[test]
fn test_hausdorff_3d_different() {
    let points1 = vec![(0.0, 0.0, 0.0)];
    let points2 = vec![(1.0, 1.0, 1.0)];
    let dist = hausdorff_3d(&points1, &points2);
    let expected = (3.0_f64).sqrt();
    assert!(
        (dist - expected).abs() < 1e-10,
        "Expected {}, got {}",
        expected,
        dist
    );
}

#[test]
fn test_lp_2d_symmetric() {
    let n = 3;
    let m1 = 4;
    let m2 = 5;
    let argvals_s = uniform_grid(m1);
    let argvals_t = uniform_grid(m2);
    let n_points = m1 * m2;
    let data = FdMatrix::from_column_major(
        (0..(n * n_points)).map(|i| i as f64 * 0.1).collect(),
        n,
        n_points,
    )
    .unwrap();
    let dist = lp_self_2d(&data, &argvals_s, &argvals_t, 2.0, &[]);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "2D Lp distance should be symmetric"
            );
        }
    }
}

#[test]
fn test_lp_2d_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(lp_self_2d(&empty, &[], &[], 2.0, &[]).is_empty());
}

#[test]
fn test_hausdorff_2d_symmetric() {
    let n = 3;
    let m1 = 4;
    let m2 = 5;
    let argvals_s = uniform_grid(m1);
    let argvals_t = uniform_grid(m2);
    let n_points = m1 * m2;
    let data = FdMatrix::from_column_major(
        (0..(n * n_points))
            .map(|i| (i as f64 * 0.1).sin())
            .collect(),
        n,
        n_points,
    )
    .unwrap();
    let dist = hausdorff_self_2d(&data, &argvals_s, &argvals_t);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "2D Hausdorff should be symmetric"
            );
        }
    }
}

#[test]
fn test_hausdorff_2d_invalid() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(hausdorff_self_2d(&empty, &[], &[]).is_empty());
}

#[test]
fn test_hausdorff_cross_1d() {
    let n1 = 3;
    let n2 = 4;
    let m = 15;
    let argvals = uniform_grid(m);
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
        n1,
        m,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
        n2,
        m,
    )
    .unwrap();
    let dist = hausdorff_cross_1d(&data1, &data2, &argvals);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
    for j in 0..n2 {
        for i in 0..n1 {
            assert!(
                dist[(i, j)] >= 0.0,
                "Hausdorff cross distance should be non-negative"
            );
            assert!(
                dist[(i, j)].is_finite(),
                "Hausdorff cross distance should be finite"
            );
        }
    }
    let self_dist = hausdorff_self_1d(&data1, &argvals);
    let cross_self = hausdorff_cross_1d(&data1, &data1, &argvals);
    for i in 0..n1 {
        assert!(
            (cross_self[(i, i)] - self_dist[(i, i)]).abs() < 1e-10,
            "Cross-self diagonal should match self diagonal at {}",
            i
        );
    }
}

#[test]
fn test_dtw_cross_1d() {
    let n1 = 3;
    let n2 = 4;
    let m = 15;
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
        n1,
        m,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
        n2,
        m,
    )
    .unwrap();
    let dist = dtw_cross_1d(&data1, &data2, 2.0, 5);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
    for j in 0..n2 {
        for i in 0..n1 {
            assert!(
                dist[(i, j)] >= 0.0,
                "DTW cross distance should be non-negative"
            );
            assert!(
                dist[(i, j)].is_finite(),
                "DTW cross distance should be finite"
            );
        }
    }
    let data_same = FdMatrix::from_column_major(
        (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
        n1,
        m,
    )
    .unwrap();
    let cross_self = dtw_cross_1d(&data_same, &data_same, 2.0, 5);
    for i in 0..n1 {
        assert!(
            cross_self[(i, i)] < 1e-8,
            "DTW self-distance should be ~0, got {}",
            cross_self[(i, i)]
        );
    }
}

#[test]
fn test_fourier_cross_1d() {
    let n1 = 3;
    let n2 = 4;
    let m = 32;
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
        n1,
        m,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
        n2,
        m,
    )
    .unwrap();
    let dist = fourier_cross_1d(&data1, &data2, 5);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
    for j in 0..n2 {
        for i in 0..n1 {
            assert!(
                dist[(i, j)] >= 0.0,
                "Fourier cross distance should be non-negative"
            );
            assert!(
                dist[(i, j)].is_finite(),
                "Fourier cross distance should be finite"
            );
        }
    }
}

#[test]
fn test_hshift_cross_1d() {
    let n1 = 3;
    let n2 = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
        n1,
        m,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
        n2,
        m,
    )
    .unwrap();
    let dist = hshift_cross_1d(&data1, &data2, &argvals, 3);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
    for j in 0..n2 {
        for i in 0..n1 {
            assert!(
                dist[(i, j)] >= 0.0,
                "Hshift cross distance should be non-negative"
            );
            assert!(
                dist[(i, j)].is_finite(),
                "Hshift cross distance should be finite"
            );
        }
    }
}

#[test]
fn test_hausdorff_self_2d_properties() {
    let n = 3;
    let m1 = 4;
    let m2 = 5;
    let argvals_s = uniform_grid(m1);
    let argvals_t = uniform_grid(m2);
    let n_points = m1 * m2;
    let data = FdMatrix::from_column_major(
        (0..(n * n_points))
            .map(|i| (i as f64 * 0.1).sin())
            .collect(),
        n,
        n_points,
    )
    .unwrap();
    let dist = hausdorff_self_2d(&data, &argvals_s, &argvals_t);
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "Hausdorff 2D self-distance should be zero on diagonal"
        );
    }
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Hausdorff 2D should be symmetric"
            );
        }
    }
}

#[test]
fn test_hausdorff_cross_2d() {
    let n1 = 2;
    let n2 = 3;
    let m1 = 4;
    let m2 = 5;
    let argvals_s = uniform_grid(m1);
    let argvals_t = uniform_grid(m2);
    let n_points = m1 * m2;
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * n_points))
            .map(|i| (i as f64 * 0.1).sin())
            .collect(),
        n1,
        n_points,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * n_points))
            .map(|i| (i as f64 * 0.2).cos())
            .collect(),
        n2,
        n_points,
    )
    .unwrap();
    let dist = hausdorff_cross_2d(&data1, &data2, &argvals_s, &argvals_t);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
    for j in 0..n2 {
        for i in 0..n1 {
            assert!(
                dist[(i, j)] >= 0.0,
                "Hausdorff cross 2D should be non-negative"
            );
            assert!(
                dist[(i, j)].is_finite(),
                "Hausdorff cross 2D should be finite"
            );
        }
    }
}

#[test]
fn test_lp_cross_2d() {
    let n1 = 2;
    let n2 = 3;
    let m1 = 4;
    let m2 = 5;
    let argvals_s = uniform_grid(m1);
    let argvals_t = uniform_grid(m2);
    let n_points = m1 * m2;
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * n_points))
            .map(|i| (i as f64 * 0.1).sin())
            .collect(),
        n1,
        n_points,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * n_points))
            .map(|i| (i as f64 * 0.2).cos())
            .collect(),
        n2,
        n_points,
    )
    .unwrap();
    let dist = lp_cross_2d(&data1, &data2, &argvals_s, &argvals_t, 2.0, &[]);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
    for j in 0..n2 {
        for i in 0..n1 {
            assert!(dist[(i, j)] >= 0.0, "Lp cross 2D should be non-negative");
            assert!(dist[(i, j)].is_finite(), "Lp cross 2D should be finite");
        }
    }
    let user_weights: Vec<f64> = vec![1.0; n_points];
    let dist_w = lp_cross_2d(&data1, &data2, &argvals_s, &argvals_t, 2.0, &user_weights);
    assert_eq!(dist_w.nrows(), n1);
    assert_eq!(dist_w.ncols(), n2);
}

#[test]
fn test_lp_self_2d_with_user_weights() {
    let n = 3;
    let m1 = 4;
    let m2 = 5;
    let argvals_s = uniform_grid(m1);
    let argvals_t = uniform_grid(m2);
    let n_points = m1 * m2;
    let data = FdMatrix::from_column_major(
        (0..(n * n_points)).map(|i| i as f64 * 0.1).collect(),
        n,
        n_points,
    )
    .unwrap();
    let user_weights: Vec<f64> = vec![2.0; n_points];
    let dist = lp_self_2d(&data, &argvals_s, &argvals_t, 2.0, &user_weights);
    assert_eq!(dist.nrows(), n);
    assert_eq!(dist.ncols(), n);
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "Weighted Lp 2D self-distance should be zero on diagonal"
        );
    }
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Weighted Lp 2D should be symmetric"
            );
        }
    }
}

#[test]
fn test_nan_lp_no_panic() {
    let m = 10;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let mut data_vec = vec![1.0; 3 * m];
    data_vec[5] = f64::NAN;
    let data = FdMatrix::from_column_major(data_vec, 3, m).unwrap();
    let w = vec![1.0; m];
    let dm = lp_self_1d(&data, &argvals, 2.0, &w);
    assert_eq!(dm.nrows(), 3);
}

#[test]
fn test_n1_self_metric() {
    let m = 10;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let data = FdMatrix::from_column_major(vec![1.0; m], 1, m).unwrap();
    let w = vec![1.0; m];
    let dm = lp_self_1d(&data, &argvals, 2.0, &w);
    assert_eq!(dm.shape(), (1, 1));
    assert!(dm[(0, 0)].abs() < 1e-12);
}

#[test]
fn test_inf_hausdorff() {
    let m = 10;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let mut data_vec = vec![1.0; 2 * m];
    data_vec[0] = f64::INFINITY;
    let data = FdMatrix::from_column_major(data_vec, 2, m).unwrap();
    let dm = hausdorff_self_1d(&data, &argvals);
    assert_eq!(dm.nrows(), 2);
    // Should not panic
}

#[test]
fn test_non_uniform_lp() {
    // Non-uniform grid
    let argvals = vec![0.0, 0.1, 0.5, 1.0];
    let m = argvals.len();
    let data_vec: Vec<f64> = vec![
        1.0, 2.0, // col 0
        1.0, 2.0, // col 1
        1.0, 2.0, // col 2
        1.0, 2.0, // col 3
    ];
    let data = FdMatrix::from_column_major(data_vec, 2, m).unwrap();
    let w = vec![1.0; m];
    let dm = lp_self_1d(&data, &argvals, 2.0, &w);
    assert_eq!(dm.shape(), (2, 2));
    // Constant offset curves: distance should be > 0
    assert!(dm[(0, 1)] > 0.0);
}

// -- Soft-DTW tests --

#[test]
fn test_softmin3_approaches_hard_min() {
    let (a, b, c) = (1.0, 3.0, 5.0);
    // Small gamma -> hard min
    let result = soft_dtw::softmin3(a, b, c, 0.001);
    assert!(
        (result - 1.0).abs() < 0.01,
        "softmin3 with small gamma should approach hard min, got {result}"
    );
}

#[test]
fn test_softmin3_large_gamma() {
    let (a, b, c) = (1.0, 3.0, 5.0);
    // Large gamma -> approaches negative bias from ln(3)
    let result = soft_dtw::softmin3(a, b, c, 1000.0);
    // With very large gamma, softmin3 ~ min - gamma * ln(3)
    assert!(
        result.is_finite(),
        "softmin3 with large gamma should be finite"
    );
}

#[test]
fn test_softmin3_stability_large_values() {
    let result = soft_dtw::softmin3(1e300, 1e300 + 1.0, 1e300 + 2.0, 1.0);
    assert!(
        result.is_finite(),
        "softmin3 should handle large values without overflow"
    );
}

#[test]
fn test_soft_dtw_identical_series() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let div = soft_dtw_divergence(&x, &x, 1.0);
    assert!(
        div.abs() < 1e-10,
        "Divergence of identical series should be ~0, got {div}"
    );
}

#[test]
fn test_soft_dtw_converges_to_hard_dtw() {
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let y = vec![2.0, 3.0, 4.0, 5.0];
    let hard = dtw_distance(&x, &y, 2.0, 10);
    let soft = soft_dtw_distance(&x, &y, 0.001);
    assert!(
        (soft - hard).abs() < 0.1,
        "Soft-DTW with small gamma should approach hard DTW\u{b2}: soft={soft}, hard={hard}"
    );
}

#[test]
fn test_soft_dtw_self_symmetric() {
    let n = 4;
    let m = 10;
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.2).sin()).collect(), n, m)
            .unwrap();
    let dist = soft_dtw_self_1d(&data, 1.0);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Soft-DTW matrix should be symmetric"
            );
        }
    }
}

#[test]
fn test_soft_dtw_cross_vs_self() {
    let n = 3;
    let m = 10;
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.2).sin()).collect(), n, m)
            .unwrap();
    let cross = soft_dtw_cross_1d(&data, &data, 1.0);
    let self_mat = soft_dtw_self_1d(&data, 1.0);
    // Off-diagonal entries should match (diagonal differs because self_distance_matrix
    // leaves diagonal as 0, but sdtw(x,x) > 0 for soft-DTW)
    for i in 0..n {
        for j in 0..n {
            if i != j {
                assert!(
                    (cross[(i, j)] - self_mat[(i, j)]).abs() < 1e-10,
                    "Cross(data,data) should match self at ({i},{j})"
                );
            }
        }
    }
}

#[test]
fn test_soft_dtw_divergence_nonneg() {
    let n = 4;
    let m = 10;
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.3).sin()).collect(), n, m)
            .unwrap();
    let div = soft_dtw_div_self_1d(&data, 1.0);
    for i in 0..n {
        for j in 0..n {
            assert!(
                div[(i, j)] >= -1e-10,
                "Divergence should be non-negative, got {} at ({i},{j})",
                div[(i, j)]
            );
        }
    }
}

#[test]
fn test_soft_dtw_single_point() {
    let x = vec![3.0];
    let y = vec![5.0];
    let dist = soft_dtw_distance(&x, &y, 1.0);
    let expected = (3.0 - 5.0_f64).powi(2);
    assert!(
        (dist - expected).abs() < 1e-10,
        "Single-point sdtw should be (a-b)\u{b2}, got {dist}, expected {expected}"
    );
}

#[test]
fn test_soft_dtw_barycenter_identical() {
    let m = 10;
    let series: Vec<f64> = (0..m).map(|i| (i as f64 * 0.3).sin()).collect();
    // Stack 5 identical copies
    let mut flat = Vec::with_capacity(5 * m);
    for _ in 0..5 {
        flat.extend_from_slice(&series);
    }
    // Column-major: for n=5, m=10
    let mut col_major = vec![0.0; 5 * m];
    for i in 0..5 {
        for j in 0..m {
            col_major[i + j * 5] = flat[i * m + j];
        }
    }
    let data = FdMatrix::from_column_major(col_major, 5, m).unwrap();
    let result = soft_dtw_barycenter(&data, 1.0, 50, 1e-6);
    for j in 0..m {
        assert!(
            (result.barycenter[j] - series[j]).abs() < 0.5,
            "Barycenter of identical series should be close to the series at j={j}"
        );
    }
}

#[test]
fn test_soft_dtw_barycenter_shifted() {
    let m = 20;
    let n = 3;
    // Create shifted copies: sin(t), sin(t)+1, sin(t)+2
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            let t = j as f64 / (m - 1) as f64;
            col_major[i + j * n] = (2.0 * PI * t).sin() + i as f64;
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let result = soft_dtw_barycenter(&data, 1.0, 100, 1e-6);
    // Barycenter should be approximately sin(t)+1 (the middle)
    let mean_val: f64 = result.barycenter.iter().sum::<f64>() / m as f64;
    assert!(
        (mean_val - 1.0).abs() < 0.5,
        "Barycenter mean should be ~1.0 (middle of shifts), got {mean_val}"
    );
}

#[test]
fn test_soft_dtw_empty() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(soft_dtw_self_1d(&empty, 1.0).is_empty());
    assert!(soft_dtw_cross_1d(&empty, &empty, 1.0).is_empty());
    assert!(soft_dtw_div_self_1d(&empty, 1.0).is_empty());
}

#[test]
fn test_soft_dtw_gamma_effect() {
    let x = vec![0.0, 1.0, 0.0];
    let y = vec![0.0, 0.0, 1.0];
    let d_small = soft_dtw_distance(&x, &y, 0.01);
    let d_large = soft_dtw_distance(&x, &y, 10.0);
    // Larger gamma produces more smoothing (smaller soft-dtw value due to more averaging)
    assert!(
        d_small > d_large || (d_small - d_large).abs() < 1e-5,
        "Larger gamma should generally produce smaller or equal soft-DTW: small={d_small}, large={d_large}"
    );
}

// -- Reference-value tests (tslearn) --

#[test]
fn test_soft_dtw_reference_tslearn_pairwise() {
    // Reference: tslearn.metrics.soft_dtw with (n_timestamps, 1) arrays
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

    let dxy = soft_dtw_distance(&x, &y, 1.0);
    let dxx = soft_dtw_distance(&x, &x, 1.0);
    let dyy = soft_dtw_distance(&y, &y, 1.0);

    // tslearn reference values (gamma=1.0)
    let dxy_ref = -0.277175357237551;
    let dxx_ref = -2.488408256052583;
    let dyy_ref = -2.488408256052583;

    let rel = |a: f64, b: f64| (a - b).abs() / b.abs().max(1e-10);
    assert!(
        rel(dxy, dxy_ref) < 1e-6,
        "d(x,y): got {dxy}, expected {dxy_ref}"
    );
    assert!(
        rel(dxx, dxx_ref) < 1e-6,
        "d(x,x): got {dxx}, expected {dxx_ref}"
    );
    assert!(
        rel(dyy, dyy_ref) < 1e-6,
        "d(y,y): got {dyy}, expected {dyy_ref}"
    );

    // Divergence = d(x,y) - 0.5*(d(x,x) + d(y,y))
    let div = dxy - 0.5 * (dxx + dyy);
    let div_ref = 2.211232898815032;
    assert!(
        rel(div, div_ref) < 1e-6,
        "divergence: got {div}, expected {div_ref}"
    );
}

#[test]
fn test_soft_dtw_reference_tslearn_gamma_sweep() {
    // Reference: tslearn.metrics.soft_dtw at multiple gamma values
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

    let cases: [(f64, f64); 3] = [
        (0.1, 1.999963681086807),
        (1.0, -0.277175357237551),
        (10.0, -46.741_092_332_890_21),
    ];

    for (gamma, expected) in cases {
        let actual = soft_dtw_distance(&x, &y, gamma);
        let denom = expected.abs().max(1e-10);
        assert!(
            (actual - expected).abs() / denom < 1e-5,
            "gamma={gamma}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn test_soft_dtw_divergence_reference_tslearn() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let y = vec![2.0, 3.0, 4.0, 5.0, 6.0];

    let div_xy = soft_dtw_divergence(&x, &y, 1.0);
    let div_ref = 2.211232898815032;
    assert!(
        (div_xy - div_ref).abs() / div_ref.abs() < 1e-6,
        "divergence(x,y): got {div_xy}, expected {div_ref}"
    );

    let div_xx = soft_dtw_divergence(&x, &x, 1.0);
    assert!(
        div_xx.abs() < 1e-6,
        "divergence(x,x) should be ~0, got {div_xx}"
    );
}

// ---------------------------------------------------------------------------
// KL divergence tests
// ---------------------------------------------------------------------------

#[test]
fn test_kl_identical_curves_zero() {
    let m = 20;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let n = 4;
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin().abs() + 0.1;
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let dist = kl_self_1d(&data, &argvals, 1e-10);
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "KL self-distance should be zero on diagonal, got {} at ({i},{i})",
            dist[(i, i)]
        );
    }
}

#[test]
fn test_kl_symmetric() {
    let m = 20;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let n = 4;
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin().abs() + 0.1;
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let dist = kl_self_1d(&data, &argvals, 1e-10);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "KL distance should be symmetric at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_kl_nonnegative() {
    let m = 20;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let n = 5;
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).cos().abs() + 0.05;
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let dist = kl_self_1d(&data, &argvals, 1e-10);
    for i in 0..n {
        for j in 0..n {
            assert!(
                dist[(i, j)] >= -1e-12,
                "KL divergence should be non-negative, got {} at ({i},{j})",
                dist[(i, j)]
            );
        }
    }
}

#[test]
fn test_kl_epsilon_handles_zeros() {
    let m = 10;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let n = 3;
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = if j % (i + 2) == 0 { 0.0 } else { 1.0 };
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let dist = kl_self_1d(&data, &argvals, 1e-10);
    for i in 0..n {
        for j in 0..n {
            assert!(
                dist[(i, j)].is_finite(),
                "KL with epsilon should produce finite values at ({i},{j})"
            );
            assert!(
                dist[(i, j)] >= -1e-12,
                "KL with epsilon should be non-negative at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_kl_cross_dimensions() {
    let m = 15;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let n1 = 3;
    let n2 = 5;
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * m))
            .map(|i| (i as f64 * 0.1).sin().abs() + 0.01)
            .collect(),
        n1,
        m,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * m))
            .map(|i| (i as f64 * 0.2).cos().abs() + 0.01)
            .collect(),
        n2,
        m,
    )
    .unwrap();
    let dist = kl_cross_1d(&data1, &data2, &argvals, 1e-10);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
    for i in 0..n1 {
        for j in 0..n2 {
            assert!(
                dist[(i, j)].is_finite(),
                "KL cross distance should be finite at ({i},{j})"
            );
            assert!(
                dist[(i, j)] >= -1e-12,
                "KL cross distance should be non-negative at ({i},{j})"
            );
        }
    }
}

#[test]
fn test_kl_known_distributions() {
    let m = 101;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let mut col_major = vec![0.0; 2 * m];
    for j in 0..m {
        let t = argvals[j];
        col_major[0 + j * 2] = 0.5 + t;
        col_major[1 + j * 2] = 1.5 - t;
    }
    let data = FdMatrix::from_column_major(col_major, 2, m).unwrap();
    let dist = kl_self_1d(&data, &argvals, 1e-12);
    assert!(
        dist[(0, 1)] > 0.0,
        "KL between different distributions should be positive, got {}",
        dist[(0, 1)]
    );
    assert!(dist[(0, 1)].is_finite());
    assert!(dist[(0, 0)].abs() < 1e-10);
    assert!((dist[(0, 1)] - dist[(1, 0)]).abs() < 1e-10);
}

#[test]
fn test_kl_empty_input() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(kl_self_1d(&empty, &[], 1e-10).is_empty());
    assert!(kl_cross_1d(&empty, &empty, &[], 1e-10).is_empty());
}

#[test]
fn test_kl_cross_self_consistent() {
    let m = 20;
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let n = 4;
    let mut col_major = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            col_major[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin().abs() + 0.1;
        }
    }
    let data = FdMatrix::from_column_major(col_major, n, m).unwrap();
    let cross = kl_cross_1d(&data, &data, &argvals, 1e-10);
    for i in 0..n {
        assert!(
            cross[(i, i)].abs() < 1e-10,
            "Cross-self diagonal should be ~0, got {} at ({i},{i})",
            cross[(i, i)]
        );
    }
}

// ---------------------------------------------------------------------------
// PCA-based semimetric tests
// ---------------------------------------------------------------------------

#[test]
fn test_pca_self_identical_zero() {
    let n = 5;
    let m = 20;
    let argvals = uniform_grid(m);
    let mut flat = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            flat[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin();
        }
    }
    let data = FdMatrix::from_column_major(flat, n, m).unwrap();
    let dist = pca_self_1d(&data, 2).unwrap();
    assert_eq!(dist.shape(), (n, n));
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "PCA self-distance should be zero on diagonal, got {}",
            dist[(i, i)]
        );
    }
}

#[test]
fn test_pca_self_symmetric() {
    let n = 4;
    let m = 15;
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let dist = pca_self_1d(&data, 2).unwrap();
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "PCA distance should be symmetric"
            );
        }
    }
}

#[test]
fn test_pca_self_ncomp_zero_error() {
    let data = FdMatrix::from_column_major(vec![1.0; 20], 2, 10).unwrap();
    assert!(pca_self_1d(&data, 0).is_err());
}

#[test]
fn test_pca_self_ncomp_too_large_error() {
    let data = FdMatrix::from_column_major(vec![1.0; 20], 2, 10).unwrap();
    assert!(pca_self_1d(&data, 3).is_err());
}

#[test]
fn test_pca_self_too_few_rows_error() {
    let data = FdMatrix::from_column_major(vec![1.0; 10], 1, 10).unwrap();
    assert!(pca_self_1d(&data, 1).is_err());
}

#[test]
fn test_pca_cross_dimensions() {
    let n1 = 3;
    let n2 = 4;
    let m = 20;
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
        n1,
        m,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
        n2,
        m,
    )
    .unwrap();
    let dist = pca_cross_1d(&data1, &data2, 2).unwrap();
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
}

#[test]
fn test_pca_cross_self_consistent() {
    let n = 5;
    let m = 20;
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let self_dist = pca_self_1d(&data, 2).unwrap();
    let cross_dist = pca_cross_1d(&data, &data, 2).unwrap();
    for i in 0..n {
        assert!(
            cross_dist[(i, i)].abs() < 1e-8,
            "PCA cross-self diagonal should be ~0, got {}",
            cross_dist[(i, i)]
        );
    }
    for i in 0..n {
        for j in (i + 1)..n {
            assert!(
                (self_dist[(i, j)] - cross_dist[(i, j)]).abs() < 1e-8,
                "PCA cross(data,data) should match self at ({i},{j})"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Derivative-based semimetric tests
// ---------------------------------------------------------------------------

#[test]
fn test_deriv_self_identical_zero() {
    let n = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let mut flat = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            flat[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin();
        }
    }
    let data = FdMatrix::from_column_major(flat, n, m).unwrap();
    let dist = deriv_self_1d(&data, &argvals, 1, &[]);
    assert_eq!(dist.shape(), (n, n));
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "Derivative self-distance should be zero on diagonal, got {}",
            dist[(i, i)]
        );
    }
}

#[test]
fn test_deriv_self_symmetric() {
    let n = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let dist = deriv_self_1d(&data, &argvals, 1, &[]);
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Derivative distance should be symmetric"
            );
        }
    }
}

#[test]
fn test_deriv_self_second_derivative() {
    let n = 3;
    let m = 30;
    let argvals = uniform_grid(m);
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let dist = deriv_self_1d(&data, &argvals, 2, &[]);
    assert_eq!(dist.shape(), (n, n));
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "Second derivative self-distance should be zero"
        );
    }
}

#[test]
fn test_deriv_cross_dimensions() {
    let n1 = 3;
    let n2 = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
        n1,
        m,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
        n2,
        m,
    )
    .unwrap();
    let dist = deriv_cross_1d(&data1, &data2, &argvals, 1, &[]);
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
}

#[test]
fn test_deriv_empty() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(deriv_self_1d(&empty, &[], 1, &[]).is_empty());
    assert!(deriv_cross_1d(&empty, &empty, &[], 1, &[]).is_empty());
}

// ---------------------------------------------------------------------------
// Basis coefficient semimetric tests
// ---------------------------------------------------------------------------

#[test]
fn test_basis_coef_self_identical_zero() {
    let n = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let mut flat = vec![0.0; n * m];
    for i in 0..n {
        for j in 0..m {
            flat[i + j * n] = (2.0 * PI * argvals[j] * (i as f64 + 1.0)).sin();
        }
    }
    let data = FdMatrix::from_column_major(flat, n, m).unwrap();
    let dist = basis_coef_self_1d(
        &data,
        &argvals,
        7,
        crate::basis::ProjectionBasisType::Fourier,
    );
    assert_eq!(dist.shape(), (n, n));
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "Basis coef self-distance should be zero on diagonal, got {}",
            dist[(i, i)]
        );
    }
}

#[test]
fn test_basis_coef_self_symmetric() {
    let n = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let dist = basis_coef_self_1d(
        &data,
        &argvals,
        7,
        crate::basis::ProjectionBasisType::Fourier,
    );
    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Basis coef distance should be symmetric"
            );
        }
    }
}

#[test]
fn test_basis_coef_cross_dimensions() {
    let n1 = 3;
    let n2 = 4;
    let m = 20;
    let argvals = uniform_grid(m);
    let data1 = FdMatrix::from_column_major(
        (0..(n1 * m)).map(|i| (i as f64 * 0.1).sin()).collect(),
        n1,
        m,
    )
    .unwrap();
    let data2 = FdMatrix::from_column_major(
        (0..(n2 * m)).map(|i| (i as f64 * 0.2).cos()).collect(),
        n2,
        m,
    )
    .unwrap();
    let dist = basis_coef_cross_1d(
        &data1,
        &data2,
        &argvals,
        7,
        crate::basis::ProjectionBasisType::Fourier,
    );
    assert_eq!(dist.nrows(), n1);
    assert_eq!(dist.ncols(), n2);
}

#[test]
fn test_basis_coef_bspline() {
    let n = 3;
    let m = 20;
    let argvals = uniform_grid(m);
    let data =
        FdMatrix::from_column_major((0..(n * m)).map(|i| (i as f64 * 0.1).sin()).collect(), n, m)
            .unwrap();
    let dist = basis_coef_self_1d(
        &data,
        &argvals,
        7,
        crate::basis::ProjectionBasisType::Bspline,
    );
    assert_eq!(dist.shape(), (n, n));
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "Bspline basis coef self-distance should be zero"
        );
    }
}

#[test]
fn test_basis_coef_empty() {
    let empty = FdMatrix::zeros(0, 0);
    assert!(
        basis_coef_self_1d(&empty, &[], 7, crate::basis::ProjectionBasisType::Fourier).is_empty()
    );
    assert!(basis_coef_cross_1d(
        &empty,
        &empty,
        &[],
        7,
        crate::basis::ProjectionBasisType::Fourier
    )
    .is_empty());
}
