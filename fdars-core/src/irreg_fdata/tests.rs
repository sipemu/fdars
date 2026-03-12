use super::*;

fn make_ifd(offsets: Vec<usize>, argvals: Vec<f64>, values: Vec<f64>) -> IrregFdata {
    let range_min = argvals.iter().cloned().fold(f64::INFINITY, f64::min);
    let range_max = argvals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    IrregFdata::from_flat(offsets, argvals, values, [range_min, range_max]).unwrap()
}

#[test]
fn test_from_lists() {
    let argvals_list = vec![vec![0.0, 0.5, 1.0], vec![0.0, 1.0]];
    let values_list = vec![vec![1.0, 2.0, 3.0], vec![1.0, 3.0]];

    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

    assert_eq!(ifd.n_obs(), 2);
    assert_eq!(ifd.n_points(0), 3);
    assert_eq!(ifd.n_points(1), 2);
    assert_eq!(ifd.total_points(), 5);
}

#[test]
fn test_get_obs() {
    let argvals_list = vec![vec![0.0, 0.5, 1.0], vec![0.0, 1.0]];
    let values_list = vec![vec![1.0, 2.0, 3.0], vec![1.0, 3.0]];

    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

    let (t0, x0) = ifd.get_obs(0);
    assert_eq!(t0, &[0.0, 0.5, 1.0]);
    assert_eq!(x0, &[1.0, 2.0, 3.0]);

    let (t1, x1) = ifd.get_obs(1);
    assert_eq!(t1, &[0.0, 1.0]);
    assert_eq!(x1, &[1.0, 3.0]);
}

#[test]
fn test_integrate_irreg() {
    // Integrate constant function = 1 over [0, 1]
    let ifd = make_ifd(
        vec![0, 3, 6],
        vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
    );

    let integrals = integrate_irreg(&ifd);

    assert!((integrals[0] - 1.0).abs() < 1e-10);
    assert!((integrals[1] - 2.0).abs() < 1e-10);
}

#[test]
fn test_norm_lp_irreg() {
    // L2 norm of constant function = c is c (on \[0,1\])
    let ifd = make_ifd(vec![0, 3], vec![0.0, 0.5, 1.0], vec![2.0, 2.0, 2.0]);

    let norms = norm_lp_irreg(&ifd, 2.0);

    assert!((norms[0] - 2.0).abs() < 1e-10);
}

#[test]
fn test_linear_interp() {
    let t = vec![0.0, 1.0, 2.0];
    let x = vec![0.0, 2.0, 4.0];

    assert!((linear_interp(&t, &x, 0.5) - 1.0).abs() < 1e-10);
    assert!((linear_interp(&t, &x, 1.5) - 3.0).abs() < 1e-10);
}

#[test]
fn test_mean_irreg() {
    // Two identical curves should give exact mean
    let ifd = make_ifd(
        vec![0, 3, 6],
        vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    );

    let target = vec![0.0, 0.5, 1.0];
    let mean = mean_irreg(&ifd, &target, 0.5, KernelType::Gaussian);

    // Mean should be close to the common values
    assert!((mean[1] - 1.0).abs() < 0.3);
}

// ========================================================================
// Tests for from_flat and accessors
// ========================================================================

#[test]
fn test_from_flat() {
    let offsets = vec![0, 3, 5, 10];
    let argvals = vec![0.0, 0.5, 1.0, 0.0, 1.0, 0.0, 0.2, 0.4, 0.6, 0.8];
    let values = vec![1.0, 2.0, 3.0, 1.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    let rangeval = [0.0, 1.0];

    let ifd = IrregFdata::from_flat(offsets.clone(), argvals.clone(), values.clone(), rangeval)
        .unwrap();

    assert_eq!(ifd.n_obs(), 3);
    assert_eq!(ifd.offsets, offsets);
    assert_eq!(ifd.argvals, argvals);
    assert_eq!(ifd.values, values);
    assert_eq!(ifd.rangeval, rangeval);
}

#[test]
fn test_from_flat_invalid() {
    // Empty offsets
    assert!(IrregFdata::from_flat(vec![], vec![], vec![], [0.0, 1.0]).is_err());
    // Mismatched argvals/values lengths
    assert!(IrregFdata::from_flat(vec![0, 2], vec![0.0, 1.0], vec![1.0], [0.0, 1.0]).is_err());
    // Last offset doesn't match argvals length
    assert!(
        IrregFdata::from_flat(vec![0, 5], vec![0.0, 1.0], vec![1.0, 2.0], [0.0, 1.0]).is_err()
    );
    // Non-monotonic offsets
    assert!(IrregFdata::from_flat(
        vec![0, 3, 1],
        vec![0.0, 1.0, 2.0],
        vec![1.0, 2.0, 3.0],
        [0.0, 2.0]
    )
    .is_err());
}

#[test]
fn test_accessors_n_obs_n_points_total() {
    let argvals_list = vec![
        vec![0.0, 0.5, 1.0],             // 3 points
        vec![0.0, 1.0],                  // 2 points
        vec![0.0, 0.25, 0.5, 0.75, 1.0], // 5 points
    ];
    let values_list = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.0, 3.0],
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
    ];

    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

    // Test n_obs
    assert_eq!(ifd.n_obs(), 3);

    // Test n_points for each curve
    assert_eq!(ifd.n_points(0), 3);
    assert_eq!(ifd.n_points(1), 2);
    assert_eq!(ifd.n_points(2), 5);

    // Test total_points
    assert_eq!(ifd.total_points(), 10);
}

#[test]
fn test_obs_counts() {
    let argvals_list = vec![
        vec![0.0, 0.5, 1.0],             // 3 points
        vec![0.0, 1.0],                  // 2 points
        vec![0.0, 0.25, 0.5, 0.75, 1.0], // 5 points
    ];
    let values_list = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.0, 3.0],
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
    ];

    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);
    let counts = ifd.obs_counts();

    assert_eq!(counts, vec![3, 2, 5]);
}

#[test]
fn test_min_max_obs() {
    let argvals_list = vec![
        vec![0.0, 0.5, 1.0],             // 3 points
        vec![0.0, 1.0],                  // 2 points
        vec![0.0, 0.25, 0.5, 0.75, 1.0], // 5 points
    ];
    let values_list = vec![
        vec![1.0, 2.0, 3.0],
        vec![1.0, 3.0],
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
    ];

    let ifd = IrregFdata::from_lists(&argvals_list, &values_list);

    assert_eq!(ifd.min_obs(), 2);
    assert_eq!(ifd.max_obs(), 5);
}

#[test]
fn test_min_max_obs_empty() {
    let ifd = IrregFdata::from_lists(&[], &[]);
    assert_eq!(ifd.min_obs(), 0);
    assert_eq!(ifd.max_obs(), 0);
}

// ========================================================================
// Tests for cov_irreg
// ========================================================================

#[test]
fn test_cov_irreg_identical_curves() {
    // Two identical curves should have zero covariance (no variability)
    let ifd = make_ifd(
        vec![0, 5, 10],
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
        vec![1.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 2.0, 1.0],
    );

    let grid = vec![0.25, 0.5, 0.75];
    let cov = cov_irreg(&ifd, &grid, &grid, 0.3);

    // Covariance should be close to 0 (identical curves)
    assert_eq!(cov.nrows(), 3);
    assert_eq!(cov.ncols(), 3);
    // Diagonal should be variance (close to 0 for identical curves)
    for i in 0..3 {
        assert!(
            cov[(i, i)].abs() < 0.5,
            "Diagonal cov[{},{}] = {} should be near 0",
            i,
            i,
            cov[(i, i)]
        );
    }
}

#[test]
fn test_cov_irreg_symmetry() {
    // Covariance matrix should be symmetric
    let ifd = make_ifd(
        vec![0, 5, 10, 15],
        vec![
            0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
        ],
        vec![
            1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0,
        ],
    );

    let grid = vec![0.25, 0.5, 0.75];
    let cov = cov_irreg(&ifd, &grid, &grid, 0.3);

    // Check symmetry: cov[i,j] = cov[j,i]
    for i in 0..3 {
        for j in 0..3 {
            assert!(
                (cov[(i, j)] - cov[(j, i)]).abs() < 1e-10,
                "Cov[{},{}] = {} != Cov[{},{}] = {}",
                i,
                j,
                cov[(i, j)],
                j,
                i,
                cov[(j, i)]
            );
        }
    }
}

#[test]
fn test_cov_irreg_diagonal_positive() {
    // Diagonal (variances) should be non-negative
    let ifd = make_ifd(
        vec![0, 5, 10, 15],
        vec![
            0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
        ],
        vec![
            1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 2.0, 3.0, 2.0, 3.0, 2.0,
        ],
    );

    let grid = vec![0.25, 0.5, 0.75];
    let cov = cov_irreg(&ifd, &grid, &grid, 0.3);

    for i in 0..3 {
        assert!(
            cov[(i, i)] >= -1e-10,
            "Variance at {} should be non-negative: {}",
            i,
            cov[(i, i)]
        );
    }
}

#[test]
fn test_cov_irreg_different_grids() {
    let ifd = make_ifd(
        vec![0, 5, 10],
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
        vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
    );

    let s_grid = vec![0.25, 0.5];
    let t_grid = vec![0.5, 0.75];
    let cov = cov_irreg(&ifd, &s_grid, &t_grid, 0.3);

    // Should produce a 2x2 matrix
    assert_eq!(cov.nrows(), 2);
    assert_eq!(cov.ncols(), 2);
}

// ========================================================================
// Tests for metric_lp_irreg
// ========================================================================

#[test]
fn test_metric_lp_irreg_self_distance_zero() {
    // Distance to self should be 0
    let ifd = make_ifd(
        vec![0, 5, 10],
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
        vec![1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0],
    );

    let dist = metric_lp_irreg(&ifd, 2.0);

    // Diagonal should be 0
    let n = 2;
    for i in 0..n {
        assert!(
            dist[(i, i)].abs() < 1e-10,
            "Self-distance d[{},{}] = {} should be 0",
            i,
            i,
            dist[(i, i)]
        );
    }
}

#[test]
fn test_metric_lp_irreg_symmetry() {
    // Distance matrix should be symmetric
    let ifd = make_ifd(
        vec![0, 5, 10, 15],
        vec![
            0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
        ],
        vec![
            1.0, 2.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 2.0, 3.0, 4.0, 3.0, 2.0,
        ],
    );

    let dist = metric_lp_irreg(&ifd, 2.0);
    let n = 3;

    for i in 0..n {
        for j in 0..n {
            assert!(
                (dist[(i, j)] - dist[(j, i)]).abs() < 1e-10,
                "Dist[{},{}] = {} != Dist[{},{}] = {}",
                i,
                j,
                dist[(i, j)],
                j,
                i,
                dist[(j, i)]
            );
        }
    }
}

#[test]
fn test_metric_lp_irreg_triangle_inequality() {
    // Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
    let ifd = make_ifd(
        vec![0, 5, 10, 15],
        vec![
            0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0,
        ],
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, // curve a
            1.0, 1.0, 1.0, 1.0, 1.0, // curve b
            2.0, 2.0, 2.0, 2.0, 2.0, // curve c
        ],
    );

    let dist = metric_lp_irreg(&ifd, 2.0);

    // d(a,c) <= d(a,b) + d(b,c)
    let d_ac = dist[(0, 2)];
    let d_ab = dist[(0, 1)];
    let d_bc = dist[(1, 2)];

    assert!(
        d_ac <= d_ab + d_bc + 1e-10,
        "Triangle inequality violated: {} > {} + {}",
        d_ac,
        d_ab,
        d_bc
    );
}

// ========================================================================
// Tests for to_regular_grid
// ========================================================================

#[test]
fn test_to_regular_grid_basic() {
    let ifd = make_ifd(
        vec![0, 5],
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![0.0, 1.0, 2.0, 3.0, 4.0],
    );

    let grid = vec![0.0, 0.5, 1.0];
    let result = to_regular_grid(&ifd, &grid);

    // Should produce 1 curve x 3 points
    assert_eq!(result.nrows(), 1);
    assert_eq!(result.ncols(), 3);

    // Check interpolated values
    assert!(
        (result[(0, 0)] - 0.0).abs() < 1e-10,
        "At t=0: {}",
        result[(0, 0)]
    );
    assert!(
        (result[(0, 1)] - 2.0).abs() < 1e-10,
        "At t=0.5: {}",
        result[(0, 1)]
    );
    assert!(
        (result[(0, 2)] - 4.0).abs() < 1e-10,
        "At t=1: {}",
        result[(0, 2)]
    );
}

#[test]
fn test_to_regular_grid_multiple_curves() {
    let ifd = make_ifd(
        vec![0, 5, 10],
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
        vec![
            0.0, 1.0, 2.0, 3.0, 4.0, // Linear: y = 4t
            4.0, 3.0, 2.0, 1.0, 0.0, // Linear: y = 4 - 4t
        ],
    );

    let grid = vec![0.0, 0.5, 1.0];
    let result = to_regular_grid(&ifd, &grid);

    // Should produce 2 curves x 3 points
    assert_eq!(result.nrows(), 2);
    assert_eq!(result.ncols(), 3);

    // Curve 0 at t=0.5 should be 2.0
    assert!((result[(0, 1)] - 2.0).abs() < 1e-10);
    // Curve 1 at t=0.5 should be 2.0
    assert!((result[(1, 1)] - 2.0).abs() < 1e-10);
}

#[test]
fn test_to_regular_grid_boundary_nan() {
    let ifd = make_ifd(
        vec![0, 3],
        vec![0.2, 0.5, 0.8], // Curve only defined on [0.2, 0.8]
        vec![1.0, 2.0, 3.0],
    );

    let grid = vec![0.0, 0.5, 1.0]; // Grid extends beyond curve range
    let result = to_regular_grid(&ifd, &grid);

    // At t=0.0 (before curve starts), should be NaN
    assert!(result[(0, 0)].is_nan(), "t=0 should be NaN");
    // At t=0.5 (within range), should be valid
    assert!(
        (result[(0, 1)] - 2.0).abs() < 1e-10,
        "t=0.5: {}",
        result[(0, 1)]
    );
    // At t=1.0 (after curve ends), should be NaN
    assert!(result[(0, 2)].is_nan(), "t=1 should be NaN");
}

#[test]
fn test_integrate_single_point_curve() {
    // Single-point curve: integral should be 0
    let ifd = make_ifd(
        vec![0, 1, 4],
        vec![0.5, 0.0, 0.5, 1.0],
        vec![1.0, 0.0, 1.0, 2.0],
    );
    let integrals = integrate_irreg(&ifd);
    assert_eq!(integrals.len(), 2);
    assert!(
        (integrals[0]).abs() < 1e-10,
        "Single-point integral should be 0"
    );
    assert!((integrals[1] - 1.0).abs() < 1e-10);
}

#[test]
fn test_norm_lp_l1() {
    // L1 norm of constant function = c is c on [0,1]
    let ifd = make_ifd(vec![0, 3], vec![0.0, 0.5, 1.0], vec![2.0, 2.0, 2.0]);
    let norms = norm_lp_irreg(&ifd, 1.0);
    assert!(
        (norms[0] - 2.0).abs() < 1e-10,
        "L1 norm of 2 = {}",
        norms[0]
    );
}

#[test]
fn test_norm_lp_general_p() {
    // L3 norm of constant function = c is c on [0,1]
    let ifd = make_ifd(vec![0, 3], vec![0.0, 0.5, 1.0], vec![3.0, 3.0, 3.0]);
    let norms = norm_lp_irreg(&ifd, 3.0);
    assert!((norms[0] - 3.0).abs() < 0.1, "L3 norm of 3 = {}", norms[0]);
}

#[test]
fn test_norm_lp_single_point() {
    // Single-point curve: norm should be 0
    let ifd = make_ifd(vec![0, 1], vec![0.5], vec![5.0]);
    let norms = norm_lp_irreg(&ifd, 2.0);
    assert!((norms[0]).abs() < 1e-10);
}

#[test]
fn test_mean_irreg_epanechnikov() {
    let ifd = make_ifd(
        vec![0, 3, 6],
        vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0],
        vec![0.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    );
    let target = vec![0.0, 0.5, 1.0];
    let mean = mean_irreg(&ifd, &target, 0.5, KernelType::Epanechnikov);
    // Mean of identical curves should match the values
    assert!(
        (mean[1] - 1.0).abs() < 0.5,
        "Epanechnikov mean at 0.5: {}",
        mean[1]
    );
}

#[test]
fn test_mean_irreg_zero_weight() {
    // Tiny bandwidth far from data -> zero weights -> NaN
    let ifd = make_ifd(vec![0, 3], vec![0.0, 0.5, 1.0], vec![1.0, 2.0, 3.0]);
    let target = vec![100.0]; // far from data
    let mean = mean_irreg(&ifd, &target, 0.001, KernelType::Epanechnikov);
    assert!(mean[0].is_nan(), "Should be NaN with zero weights");
}

#[test]
fn test_metric_lp_l1() {
    let ifd = make_ifd(
        vec![0, 5, 10],
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let dist = metric_lp_irreg(&ifd, 1.0);
    assert!(dist[(0, 1)] > 0.0, "L1 distance should be positive");
    assert!(
        (dist[(0, 1)] - 1.0).abs() < 1e-10,
        "L1 distance of 0 and 1: {}",
        dist[(0, 1)]
    );
}

#[test]
fn test_metric_lp_general_p() {
    let ifd = make_ifd(
        vec![0, 5, 10],
        vec![0.0, 0.25, 0.5, 0.75, 1.0, 0.0, 0.25, 0.5, 0.75, 1.0],
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    );
    let dist = metric_lp_irreg(&ifd, 3.0);
    assert!(dist[(0, 1)] > 0.0, "L3 distance should be positive");
    // L3 distance of |0-1|^3 integrated = 1, then ^(1/3) = 1
    assert!(
        (dist[(0, 1)] - 1.0).abs() < 0.1,
        "L3 distance: {}",
        dist[(0, 1)]
    );
}

#[test]
fn test_metric_lp_non_overlapping_curves() {
    // Curves on non-overlapping ranges: common_t < 2 -> NaN
    let ifd = IrregFdata::from_lists(
        &[vec![0.0, 0.5], vec![2.0, 3.0]],
        &[vec![1.0, 2.0], vec![3.0, 4.0]],
    );
    let dist = metric_lp_irreg(&ifd, 2.0);
    assert!(dist[(0, 1)].is_nan(), "Non-overlapping should give NaN");
}

#[test]
fn test_empty_from_lists() {
    let ifd = IrregFdata::from_lists(&[], &[]);
    assert_eq!(ifd.n_obs(), 0);
}

#[test]
fn test_single_point_curve() {
    let argvals = vec![vec![0.5]];
    let values = vec![vec![1.0]];
    let ifd = IrregFdata::from_lists(&argvals, &values);
    assert_eq!(ifd.n_obs(), 1);
    assert_eq!(ifd.n_points(0), 1);
    let (a, v) = ifd.get_obs(0);
    assert_eq!(a, &[0.5]);
    assert_eq!(v, &[1.0]);
}

#[test]
fn test_nan_values_irreg() {
    let argvals = vec![vec![0.0, 0.5, 1.0]];
    let values = vec![vec![1.0, f64::NAN, 3.0]];
    let ifd = IrregFdata::from_lists(&argvals, &values);
    let integrals = integrate_irreg(&ifd);
    assert_eq!(integrals.len(), 1);
    // NaN should propagate
    assert!(integrals[0].is_nan());
}
