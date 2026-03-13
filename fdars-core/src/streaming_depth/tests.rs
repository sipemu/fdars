use super::*;
use crate::depth::{band_1d, fraiman_muniz_1d, modified_band_1d};
use crate::matrix::FdMatrix;
use crate::test_helpers::uniform_grid;
use std::f64::consts::PI;

fn generate_centered_data(n: usize, m: usize) -> Vec<f64> {
    let argvals = uniform_grid(m);
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let offset = (i as f64 - n as f64 / 2.0) / (n as f64);
        for j in 0..m {
            data[i + j * n] = (2.0 * PI * argvals[j]).sin() + offset;
        }
    }
    data
}

/// Extract a single curve (row i) from column-major data into row layout.
fn extract_curve(data: &[f64], i: usize, n: usize, m: usize) -> Vec<f64> {
    (0..m).map(|t| data[i + t * n]).collect()
}

// ============== Rank correctness ==============

#[test]
fn test_rank_basic() {
    // 5 reference curves, 3 time points
    // Column 0: [1, 2, 3, 4, 5]
    let data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, // t=0
        10.0, 20.0, 30.0, 40.0, 50.0, // t=1
        100.0, 200.0, 300.0, 400.0, 500.0, // t=2
    ];
    let mat = FdMatrix::from_column_major(data, 5, 3).unwrap();
    let state = SortedReferenceState::from_reference(&mat);

    // At t=0, x=3.0: below=2 (1,2), above=2 (4,5)
    let (below, above) = state.rank_at(0, 3.0);
    assert_eq!(below, 2);
    assert_eq!(above, 2);

    // At t=1, x=25.0: below=2 (10,20), above=3 (30,40,50)
    let (below, above) = state.rank_at(1, 25.0);
    assert_eq!(below, 2);
    assert_eq!(above, 3);
}

#[test]
fn test_rank_boundary_values() {
    // All values identical
    let data = vec![5.0, 5.0, 5.0, 5.0];
    let mat = FdMatrix::from_column_major(data, 4, 1).unwrap();
    let state = SortedReferenceState::from_reference(&mat);

    // x=5.0 exactly: none strictly below, none strictly above
    let (below, above) = state.rank_at(0, 5.0);
    assert_eq!(below, 0);
    assert_eq!(above, 0);

    // x < all: below=0, above=4
    let (below, above) = state.rank_at(0, 3.0);
    assert_eq!(below, 0);
    assert_eq!(above, 4);

    // x > all: below=4, above=0
    let (below, above) = state.rank_at(0, 7.0);
    assert_eq!(below, 4);
    assert_eq!(above, 0);
}

#[test]
fn test_rank_duplicates() {
    // Values with duplicates: [1, 2, 2, 3, 3, 3]
    let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
    let mat = FdMatrix::from_column_major(data, 6, 1).unwrap();
    let state = SortedReferenceState::from_reference(&mat);

    // x=2.0: below=1 (just 1), above=3 (three 3s)
    let (below, above) = state.rank_at(0, 2.0);
    assert_eq!(below, 1);
    assert_eq!(above, 3);

    // x=3.0: below=3 (1,2,2), above=0
    let (below, above) = state.rank_at(0, 3.0);
    assert_eq!(below, 3);
    assert_eq!(above, 0);
}

// ============== Batch equivalence ==============

#[test]
fn test_streaming_mbd_matches_batch() {
    let n = 15;
    let m = 20;
    let data = generate_centered_data(n, m);

    let mat = FdMatrix::from_slice(&data, n, m).unwrap();
    let batch = modified_band_1d(&mat, &mat);
    let state = SortedReferenceState::from_reference(&mat);
    let streaming = StreamingMbd::new(state);
    let streaming_result = streaming.depth_batch(&mat);

    assert_eq!(batch.len(), streaming_result.len());
    for (b, s) in batch.iter().zip(streaming_result.iter()) {
        assert!(
            (b - s).abs() < 1e-10,
            "MBD mismatch: batch={}, streaming={}",
            b,
            s
        );
    }
}

#[test]
fn test_streaming_fm_matches_batch() {
    let n = 15;
    let m = 20;
    let data = generate_centered_data(n, m);

    let mat = FdMatrix::from_slice(&data, n, m).unwrap();
    for scale in [true, false] {
        let batch = fraiman_muniz_1d(&mat, &mat, scale);
        let state = SortedReferenceState::from_reference(&mat);
        let streaming = StreamingFraimanMuniz::new(state, scale);
        let streaming_result = streaming.depth_batch(&mat);

        assert_eq!(batch.len(), streaming_result.len());
        for (b, s) in batch.iter().zip(streaming_result.iter()) {
            assert!(
                (b - s).abs() < 1e-10,
                "FM mismatch (scale={}): batch={}, streaming={}",
                scale,
                b,
                s
            );
        }
    }
}

#[test]
fn test_streaming_bd_matches_batch() {
    let n = 10;
    let m = 20;
    let data = generate_centered_data(n, m);

    let mat = FdMatrix::from_slice(&data, n, m).unwrap();
    let batch = band_1d(&mat, &mat);
    let full_state = FullReferenceState::from_reference(&mat);
    let streaming = StreamingBd::new(full_state);
    let streaming_result = streaming.depth_batch(&mat);

    assert_eq!(batch.len(), streaming_result.len());
    for (b, s) in batch.iter().zip(streaming_result.iter()) {
        assert!(
            (b - s).abs() < 1e-10,
            "BD mismatch: batch={}, streaming={}",
            b,
            s
        );
    }
}

// ============== Rolling reference ==============

#[test]
fn test_rolling_sorted_columns_maintained() {
    let mut rolling = RollingReference::new(3, 2);

    rolling.push(&[1.0, 10.0]);
    assert_eq!(rolling.snapshot().sorted_columns[0], vec![1.0]);
    assert_eq!(rolling.snapshot().sorted_columns[1], vec![10.0]);

    rolling.push(&[3.0, 5.0]);
    assert_eq!(rolling.snapshot().sorted_columns[0], vec![1.0, 3.0]);
    assert_eq!(rolling.snapshot().sorted_columns[1], vec![5.0, 10.0]);

    rolling.push(&[2.0, 7.0]);
    assert_eq!(rolling.snapshot().sorted_columns[0], vec![1.0, 2.0, 3.0]);
    assert_eq!(rolling.snapshot().sorted_columns[1], vec![5.0, 7.0, 10.0]);

    // Push a 4th -- evicts [1.0, 10.0]
    let evicted = rolling.push(&[0.5, 8.0]);
    assert_eq!(evicted, Some(vec![1.0, 10.0]));
    assert_eq!(rolling.snapshot().sorted_columns[0], vec![0.5, 2.0, 3.0]);
    assert_eq!(rolling.snapshot().sorted_columns[1], vec![5.0, 7.0, 8.0]);
}

#[test]
fn test_rolling_mbd_matches_batch() {
    let n = 10;
    let m = 15;
    let data = generate_centered_data(n, m);

    // Fill a rolling window with the same curves
    let mut rolling = RollingReference::new(n, m);
    for i in 0..n {
        let curve = extract_curve(&data, i, n, m);
        rolling.push(&curve);
    }

    // mbd_one should match batch for each curve
    let mat = FdMatrix::from_slice(&data, n, m).unwrap();
    let batch = modified_band_1d(&mat, &mat);
    for i in 0..n {
        let curve = extract_curve(&data, i, n, m);
        let rolling_depth = rolling.mbd_one(&curve);
        assert!(
            (batch[i] - rolling_depth).abs() < 1e-10,
            "Rolling MBD mismatch at i={}: batch={}, rolling={}",
            i,
            batch[i],
            rolling_depth
        );
    }
}

#[test]
fn test_rolling_eviction_correctness() {
    let m = 5;
    let mut rolling = RollingReference::new(3, m);

    // Push 5 curves -- window should only contain the last 3
    let curves: Vec<Vec<f64>> = (0..5)
        .map(|i| (0..m).map(|t| (i * m + t) as f64).collect())
        .collect();

    for c in &curves {
        rolling.push(c);
    }

    assert_eq!(rolling.len(), 3);

    // Snapshot should match manually-built state from curves 2,3,4
    let snapshot = rolling.snapshot();
    assert_eq!(snapshot.nori(), 3);

    // Build reference data manually from curves 2..5
    let mut ref_data = vec![0.0; 3 * m];
    for (idx, ci) in (2..5).enumerate() {
        for t in 0..m {
            ref_data[idx + t * 3] = curves[ci][t];
        }
    }
    let ref_mat = FdMatrix::from_column_major(ref_data, 3, m).unwrap();
    let expected = SortedReferenceState::from_reference(&ref_mat);

    for t in 0..m {
        assert_eq!(
            snapshot.sorted_columns[t], expected.sorted_columns[t],
            "sorted columns differ at t={}",
            t
        );
    }
}

// ============== Properties ==============

#[test]
fn test_depth_in_unit_interval() {
    let n = 20;
    let m = 30;
    let data = generate_centered_data(n, m);
    let mat = FdMatrix::from_slice(&data, n, m).unwrap();

    let state_mbd = SortedReferenceState::from_reference(&mat);
    let mbd = StreamingMbd::new(state_mbd);
    for d in mbd.depth_batch(&mat) {
        assert!((0.0..=1.0).contains(&d), "MBD out of range: {}", d);
    }

    let state_fm = SortedReferenceState::from_reference(&mat);
    let fm = StreamingFraimanMuniz::new(state_fm, true);
    for d in fm.depth_batch(&mat) {
        assert!((0.0..=1.0).contains(&d), "FM out of range: {}", d);
    }

    let full = FullReferenceState::from_reference(&mat);
    let bd = StreamingBd::new(full);
    for d in bd.depth_batch(&mat) {
        assert!((0.0..=1.0).contains(&d), "BD out of range: {}", d);
    }
}

#[test]
fn test_central_curves_deeper() {
    let n = 20;
    let m = 30;
    let data = generate_centered_data(n, m);
    let mat = FdMatrix::from_slice(&data, n, m).unwrap();

    let state = SortedReferenceState::from_reference(&mat);
    let mbd = StreamingMbd::new(state);
    let depths = mbd.depth_batch(&mat);

    let central_depth = depths[n / 2];
    let edge_depth = depths[0];
    assert!(
        central_depth > edge_depth,
        "Central curve should be deeper: {} > {}",
        central_depth,
        edge_depth
    );
}

#[test]
fn test_empty_inputs() {
    let empty = FdMatrix::zeros(0, 0);
    let state = SortedReferenceState::from_reference(&empty);
    let mbd = StreamingMbd::new(state);
    assert_eq!(mbd.depth_one(&[]), 0.0);

    let state = SortedReferenceState::from_reference(&empty);
    let fm = StreamingFraimanMuniz::new(state, true);
    assert_eq!(fm.depth_one(&[]), 0.0);
}

#[test]
fn test_depth_one_matches_depth_batch_single() {
    let n = 10;
    let m = 15;
    let data = generate_centered_data(n, m);
    let mat = FdMatrix::from_slice(&data, n, m).unwrap();

    // Build a 1-curve column-major "matrix" from curve 3
    let curve = extract_curve(&data, 3, n, m);
    let single_mat = FdMatrix::from_column_major(curve.clone(), 1, m).unwrap();

    let state = SortedReferenceState::from_reference(&mat);
    let mbd = StreamingMbd::new(state);

    let one = mbd.depth_one(&curve);
    let batch = mbd.depth_batch(&single_mat);
    assert!(
        (one - batch[0]).abs() < 1e-14,
        "depth_one ({}) != depth_batch ({}) for single curve",
        one,
        batch[0]
    );
}

// ============== Thread safety ==============

#[test]
fn test_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SortedReferenceState>();
    assert_send_sync::<StreamingMbd>();
    assert_send_sync::<StreamingFraimanMuniz>();
    assert_send_sync::<FullReferenceState>();
    assert_send_sync::<StreamingBd>();
    assert_send_sync::<RollingReference>();
}

// ============== Edge cases ==============

#[test]
fn test_single_reference_curve() {
    // nori=1: C(1,2) = 0, MBD is undefined -> returns 0
    let data = vec![1.0, 2.0, 3.0]; // 1 curve, 3 time points
    let mat = FdMatrix::from_column_major(data, 1, 3).unwrap();
    let state = SortedReferenceState::from_reference(&mat);
    let mbd = StreamingMbd::new(state);
    assert_eq!(mbd.depth_one(&[1.0, 2.0, 3.0]), 0.0);

    // BD also needs at least 2
    let full = FullReferenceState::from_reference(&mat);
    let bd = StreamingBd::new(full);
    assert_eq!(bd.depth_one(&[1.0, 2.0, 3.0]), 0.0);
}

#[test]
fn test_capacity_one_window() {
    let mut rolling = RollingReference::new(1, 3);

    rolling.push(&[1.0, 2.0, 3.0]);
    assert_eq!(rolling.len(), 1);
    // MBD with 1 curve -> 0
    assert_eq!(rolling.mbd_one(&[1.0, 2.0, 3.0]), 0.0);

    let evicted = rolling.push(&[4.0, 5.0, 6.0]);
    assert_eq!(evicted, Some(vec![1.0, 2.0, 3.0]));
    assert_eq!(rolling.len(), 1);
}

#[test]
#[should_panic(expected = "curve length")]
fn test_curve_length_mismatch() {
    let mat = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0, 4.0], 2, 2).unwrap();
    let state = SortedReferenceState::from_reference(&mat);
    let mbd = StreamingMbd::new(state);
    // Curve has 3 elements but n_points is 2 -- should ideally be caught.
    // depth_one doesn't assert length (it just indexes), but rolling does.
    let mut rolling = RollingReference::new(5, 2);
    rolling.push(&[1.0, 2.0, 3.0]); // panics: length mismatch
    let _ = mbd; // suppress unused warning
}

// ============== Additional: snapshot-based streaming ==============

#[test]
fn test_rolling_snapshot_produces_valid_mbd() {
    let n = 8;
    let m = 10;
    let data = generate_centered_data(n, m);

    let mut rolling = RollingReference::new(n, m);
    for i in 0..n {
        let curve = extract_curve(&data, i, n, m);
        rolling.push(&curve);
    }

    let snapshot = rolling.snapshot();
    let mbd = StreamingMbd::new(snapshot);

    let mat = FdMatrix::from_slice(&data, n, m).unwrap();
    let batch_depths = modified_band_1d(&mat, &mat);
    let streaming_depths = mbd.depth_batch(&mat);

    for (b, s) in batch_depths.iter().zip(streaming_depths.iter()) {
        assert!(
            (b - s).abs() < 1e-10,
            "Snapshot MBD mismatch: batch={}, streaming={}",
            b,
            s
        );
    }
}

#[test]
fn test_nan_reference_streaming() {
    // Reference data with NaN
    let n_ref = 5;
    let m = 10;
    let mut ref_data = vec![0.0; n_ref * m];
    ref_data[3] = f64::NAN;
    let ref_mat = FdMatrix::from_column_major(ref_data, n_ref, m).unwrap();
    let state = SortedReferenceState::from_reference(&ref_mat);
    let streamer = StreamingMbd::new(state);
    let new_curve: Vec<f64> = vec![1.0; m];
    let depth = streamer.depth_one(&new_curve);
    // Should not panic, depth may be NaN
    let _ = depth;
}

#[test]
fn test_n2_mbd_streaming() {
    // Minimum reference set: 2 curves
    let ref_data = FdMatrix::from_column_major(vec![0.0, 1.0, 0.0, 1.0], 2, 2).unwrap();
    let state = SortedReferenceState::from_reference(&ref_data);
    let streamer = StreamingMbd::new(state);
    let depth = streamer.depth_one(&[0.5, 0.5]);
    assert!(depth.is_finite());
    assert!((0.0..=1.0).contains(&depth));
}

#[test]
fn test_trait_n_points_n_reference() {
    let n = 5;
    let m = 3;
    let data = generate_centered_data(n, m);
    let mat = FdMatrix::from_slice(&data, n, m).unwrap();

    let state = SortedReferenceState::from_reference(&mat);
    assert_eq!(state.nori(), n);
    assert_eq!(state.n_points(), m);

    let mbd = StreamingMbd::new(SortedReferenceState::from_reference(&mat));
    assert_eq!(mbd.n_points(), m);
    assert_eq!(mbd.n_reference(), n);

    let fm = StreamingFraimanMuniz::new(SortedReferenceState::from_reference(&mat), true);
    assert_eq!(fm.n_points(), m);
    assert_eq!(fm.n_reference(), n);

    let full = FullReferenceState::from_reference(&mat);
    let bd = StreamingBd::new(full);
    assert_eq!(bd.n_points(), m);
    assert_eq!(bd.n_reference(), n);
}

#[test]
fn test_bd_depth_one_matches_batch() {
    let n = 8;
    let m = 10;
    let data = generate_centered_data(n, m);
    let mat = FdMatrix::from_slice(&data, n, m).unwrap();
    let full = FullReferenceState::from_reference(&mat);
    let bd = StreamingBd::new(full);

    // depth_one should match depth_batch for single curve
    let curve = extract_curve(&data, 3, n, m);
    let single_mat = FdMatrix::from_column_major(curve.clone(), 1, m).unwrap();
    let one = bd.depth_one(&curve);
    let batch = bd.depth_batch(&single_mat);
    assert!((one - batch[0]).abs() < 1e-14);
}

#[test]
fn test_bd_batch_empty() {
    let ref_data = FdMatrix::from_column_major(vec![0.0, 1.0, 0.0, 1.0], 2, 2).unwrap();
    let full = FullReferenceState::from_reference(&ref_data);
    let bd = StreamingBd::new(full);

    // Empty query matrix
    let empty = FdMatrix::zeros(0, 2);
    assert!(bd.depth_batch(&empty).is_empty());
}

#[test]
fn test_bd_single_ref_returns_zero() {
    let ref_data = FdMatrix::from_column_major(vec![1.0, 2.0, 3.0], 1, 3).unwrap();
    let full = FullReferenceState::from_reference(&ref_data);
    let bd = StreamingBd::new(full);
    assert_eq!(bd.depth_one(&[1.5, 2.5, 2.0]), 0.0);
}

#[test]
fn test_mbd_batch_empty_ref() {
    let mat = FdMatrix::zeros(0, 0);
    let state = SortedReferenceState::from_reference(&mat);
    let mbd = StreamingMbd::new(state);
    let query = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
    let result = mbd.depth_batch(&query);
    assert_eq!(result, vec![0.0]);
}

#[test]
fn test_fm_batch_empty_ref() {
    let mat = FdMatrix::zeros(0, 0);
    let state = SortedReferenceState::from_reference(&mat);
    let fm = StreamingFraimanMuniz::new(state, false);
    let query = FdMatrix::from_column_major(vec![1.0, 2.0], 1, 2).unwrap();
    let result = fm.depth_batch(&query);
    assert_eq!(result, vec![0.0]);
}

#[test]
fn test_fm_unscaled_depth() {
    // Test with scale=false
    let n = 10;
    let m = 15;
    let data = generate_centered_data(n, m);
    let mat = FdMatrix::from_slice(&data, n, m).unwrap();
    let state = SortedReferenceState::from_reference(&mat);
    let fm = StreamingFraimanMuniz::new(state, false);
    let depths = fm.depth_batch(&mat);
    // All depths should be in [0, 0.5] for unscaled
    for d in &depths {
        assert!(
            (0.0..=0.5 + 1e-10).contains(d),
            "Unscaled FM depth {} > 0.5",
            d
        );
    }
}

#[test]
fn test_rolling_is_empty_and_capacity() {
    let rolling = RollingReference::new(5, 3);
    assert!(rolling.is_empty());
    assert_eq!(rolling.capacity(), 5);
    assert_eq!(rolling.len(), 0);
}

#[test]
fn test_rolling_mbd_with_two_curves() {
    let mut rolling = RollingReference::new(5, 3);
    rolling.push(&[0.0, 0.0, 0.0]);
    rolling.push(&[1.0, 1.0, 1.0]);
    // Query a curve in between
    let d = rolling.mbd_one(&[0.5, 0.5, 0.5]);
    assert!(d.is_finite());
    assert!((0.0..=1.0).contains(&d));
}

#[test]
fn test_single_timepoint_streaming() {
    // m=1: single time point
    let ref_data = FdMatrix::from_column_major(vec![0.0, 1.0, 2.0], 3, 1).unwrap();
    let state = SortedReferenceState::from_reference(&ref_data);
    let streamer = StreamingMbd::new(state);
    let depth = streamer.depth_one(&[1.0]);
    assert!(depth.is_finite());
}
