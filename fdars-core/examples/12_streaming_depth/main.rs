//! Example 12: Streaming/Online Depth Computation
//!
//! Demonstrates online depth computation where new curves are evaluated
//! against a pre-built reference set. This is efficient for real-time
//! monitoring: the reference is built once, then each new curve is
//! scored in O(m log n) rather than recomputing from scratch.

use fdars_core::depth::{band_1d, fraiman_muniz_1d, modified_band_1d};
use fdars_core::simulation::{sim_fundata, EFunType, EValType};
use fdars_core::streaming_depth::{
    FullReferenceState, RollingReference, SortedReferenceState, StreamingBd, StreamingDepth,
    StreamingFraimanMuniz, StreamingMbd,
};

fn uniform_grid(m: usize) -> Vec<f64> {
    (0..m).map(|i| i as f64 / (m - 1) as f64).collect()
}

fn main() {
    println!("=== Example 12: Streaming Depth Computation ===\n");

    let n_ref = 40;
    let n_new = 5;
    let m = 50;
    let big_m = 5;
    let t = uniform_grid(m);

    // Reference data and new curves
    let ref_mat = sim_fundata(
        n_ref,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(42),
    );
    let new_mat = sim_fundata(
        n_new,
        &t,
        big_m,
        EFunType::Fourier,
        EValType::Exponential,
        Some(99),
    );
    // --- Section 1: Build reference state ---
    println!("--- Building Reference State ---");
    let sorted_state = SortedReferenceState::from_reference(&ref_mat);
    let full_state = FullReferenceState::from_reference(&ref_mat);
    println!("  Reference curves: {}", sorted_state.nori());
    println!("  Grid points: {}", sorted_state.n_points());

    // --- Section 2: Streaming Modified Band Depth ---
    println!("\n--- Streaming Modified Band Depth ---");
    let streaming_mbd = StreamingMbd::new(SortedReferenceState::from_reference(&ref_mat));

    // Score individual curves
    for i in 0..n_new {
        let curve: Vec<f64> = (0..m).map(|j| new_mat[(i, j)]).collect();
        let depth = streaming_mbd.depth_one(&curve);
        println!("  New curve {i}: streaming MBD = {depth:.6}");
    }

    // Score a batch
    let batch_depths = streaming_mbd.depth_batch(&new_mat);
    println!(
        "  Batch depths: {:?}",
        batch_depths
            .iter()
            .map(|d| format!("{d:.4}"))
            .collect::<Vec<_>>()
    );

    // --- Section 3: Streaming Fraiman-Muniz ---
    println!("\n--- Streaming Fraiman-Muniz Depth ---");
    let streaming_fm =
        StreamingFraimanMuniz::new(SortedReferenceState::from_reference(&ref_mat), true);
    let fm_depths = streaming_fm.depth_batch(&new_mat);
    println!(
        "  Batch depths: {:?}",
        fm_depths
            .iter()
            .map(|d| format!("{d:.4}"))
            .collect::<Vec<_>>()
    );

    // --- Section 4: Streaming Band Depth ---
    println!("\n--- Streaming Band Depth ---");
    let streaming_bd = StreamingBd::new(full_state);
    let bd_depths = streaming_bd.depth_batch(&new_mat);
    println!(
        "  Batch depths: {:?}",
        bd_depths
            .iter()
            .map(|d| format!("{d:.4}"))
            .collect::<Vec<_>>()
    );

    // --- Section 5: Compare streaming vs batch ---
    println!("\n--- Streaming vs Batch Comparison ---");
    // Batch: combine reference + new, compute depth of new w.r.t. reference
    let batch_mbd = modified_band_1d(&new_mat, &ref_mat);
    let batch_fm = fraiman_muniz_1d(&new_mat, &ref_mat, true);
    let batch_bd = band_1d(&new_mat, &ref_mat);

    println!(
        "  {:>5} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}",
        "Curve", "S-MBD", "B-MBD", "S-FM", "B-FM", "S-BD", "B-BD"
    );
    let stream_mbd_vals = streaming_mbd.depth_batch(&new_mat);
    for i in 0..n_new {
        println!(
            "  {:>5} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6} {:>12.6}",
            i,
            stream_mbd_vals[i],
            batch_mbd[i],
            fm_depths[i],
            batch_fm[i],
            bd_depths[i],
            batch_bd[i]
        );
    }

    // --- Section 6: Rolling reference window ---
    println!("\n--- Rolling Reference Window ---");
    let window_size = 20;
    let mut rolling = RollingReference::new(window_size, m);

    // Fill the window with reference curves
    for i in 0..window_size {
        let curve: Vec<f64> = (0..m).map(|j| ref_mat[(i, j)]).collect();
        rolling.push(&curve);
    }
    println!(
        "  Window capacity: {}, current length: {}",
        rolling.capacity(),
        rolling.len()
    );

    // Score a new curve against the rolling window using snapshot()
    let test_curve: Vec<f64> = (0..m).map(|j| new_mat[(0, j)]).collect();

    // Build a streaming depth from the rolling window's snapshot
    let window_state = rolling.snapshot();
    let window_mbd = StreamingMbd::new(window_state);
    let window_depth = window_mbd.depth_one(&test_curve);
    println!("  Depth of test curve against window: {window_depth:.6}");

    // Can also use mbd_one directly on the rolling reference
    let direct_depth = rolling.mbd_one(&test_curve);
    println!("  Direct mbd_one: {direct_depth:.6}");

    // Slide the window by pushing a new curve (oldest is evicted)
    let new_curve: Vec<f64> = (0..m).map(|j| ref_mat[(20, j)]).collect();
    let evicted = rolling.push(&new_curve);
    println!("  After sliding: evicted a curve = {}", evicted.is_some());
    println!("  Window length: {}", rolling.len());

    println!("\n=== Done ===");
}
