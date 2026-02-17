# Example 12: Streaming/Online Depth Computation

## What this demonstrates

Online depth computation where the reference set is pre-processed once, then new curves are scored individually without recomputing from scratch. This is efficient for real-time monitoring scenarios where new observations arrive one at a time.

Three streaming depth methods are shown (Modified Band, Fraiman-Muniz, Band), along with batch comparison and rolling reference windows.

## API functions used

- `streaming_depth::SortedReferenceState::from_reference()` — build sorted reference state
- `streaming_depth::FullReferenceState::from_reference()` — build full reference state (for Band depth)
- `streaming_depth::StreamingMbd::new()` — streaming Modified Band Depth
- `streaming_depth::StreamingFraimanMuniz::new()` — streaming Fraiman-Muniz depth
- `streaming_depth::StreamingBd::new()` — streaming Band Depth
- `StreamingDepth::depth_one()` — score a single curve
- `StreamingDepth::depth_batch()` — score multiple curves
- `streaming_depth::RollingReference::new()` — sliding window reference

## How to run

```bash
cargo run --example streaming_depth
```

## Expected output

Individual and batch streaming depths, comparison with full batch computation (values should match closely), and rolling window demonstration.

## Key concepts

- **Pre-built reference**: sort reference data per time point once, enabling O(m log n) queries
- **StreamingDepth trait**: unified interface for `depth_one()` and `depth_batch()`
- **Rolling window**: fixed-capacity FIFO buffer for adaptive reference sets
- **Streaming vs batch**: results should be identical or very close when using the same reference
