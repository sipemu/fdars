# Phase 01: Measurement Discipline & Baselines — Pattern Map

**Mapped:** 2026-08-07
**Files analyzed:** 1 (new code file: `fdars-core/benches/audit_hotpaths.rs`)
**Analogs found:** 2 primary / 1 supplementary (regression_benchmarks.rs, alignment_benchmarks.rs, depth_benchmarks.rs)

---

## File Classification

| New/Modified File | Role | Data Flow | Closest Analog | Match Quality |
|-------------------|------|-----------|----------------|---------------|
| `fdars-core/benches/audit_hotpaths.rs` | bench | batch (N×M synthetic inputs → Criterion timing) | `fdars-core/benches/regression_benchmarks.rs` | exact — same: criterion harness, FdMatrix construction, `fdata_to_pc_1d` sentinel |
| `fdars-core/Cargo.toml` (append only) | config | — | Lines 54–88 of existing `[[bench]]` block | exact — same `[[bench]] / name / harness = false` triple |

---

## Pattern Assignments

### `fdars-core/benches/audit_hotpaths.rs` (bench, batch)

**Primary analog:** `fdars-core/benches/regression_benchmarks.rs`
**Secondary analog:** `fdars-core/benches/alignment_benchmarks.rs`
**Supplementary (streaming depth + FM depth):** `fdars-core/benches/depth_benchmarks.rs`

---

#### Imports pattern

**Source:** `fdars-core/benches/regression_benchmarks.rs` lines 8–12 (FPCA/SVD sentinel imports) and `fdars-core/benches/alignment_benchmarks.rs` lines 8–14 (elastic imports) and `fdars-core/benches/depth_benchmarks.rs` lines 5–18 (depth + streaming imports) and `fdars-core/benches/smoothing_benchmarks.rs` lines 10–13.

Copy this combined import block into `audit_hotpaths.rs`:

```rust
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::alignment::elastic_self_distance_matrix;
use fdars_core::classification::fclassif_cv;
use fdars_core::depth::fraiman_muniz_1d;
use fdars_core::matrix::FdMatrix;
use fdars_core::regression::fdata_to_pc_1d;
use fdars_core::smoothing::nadaraya_watson;
use fdars_core::streaming_depth::{SortedReferenceState, StreamingDepth, StreamingFraimanMuniz};
use std::f64::consts::PI;
```

No `rand`/`rand_distr` import is needed if the deterministic-trig generator (Section 7 of RESEARCH.md) is used — it requires only `std::f64::consts::PI`.

---

#### `[[bench]]` entry in `fdars-core/Cargo.toml`

**Source:** `fdars-core/Cargo.toml` lines 54–56 (any existing bench entry — all identical in structure).

Append immediately after line 88 (the last existing `[[bench]]` block):

```toml
[[bench]]
name = "audit_hotpaths"
harness = false
```

The `name` field must match the filename stem (`benches/audit_hotpaths.rs`) exactly. All 9 existing entries follow this three-line pattern with no extra keys.

---

#### Seeded column-major input generator

**Source:** `fdars-core/benches/regression_benchmarks.rs` lines 18–44 (canonical generator) and `fdars-core/benches/alignment_benchmarks.rs` lines 17–31 (same pattern, simpler form).

The alignment bench form (lines 17–31) is the cleanest template:

```rust
/// Generate synthetic functional data (n curves, m time points).
fn generate_curves(n: usize, m: usize) -> (FdMatrix, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        // Deterministic phase/amplitude variation per curve
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        for j in 0..m {
            let t = argvals[j];
            data[i + j * n] = amp * (2.0 * PI * (t + phase)).sin();
        }
    }
    let mat = FdMatrix::from_column_major(data, n, m).unwrap();
    (mat, argvals)
}
```

**Column-major layout rule** (verified: `alignment_benchmarks.rs` line 26):
`data[i + j * n]` = observation `i` at evaluation point `j`.

**Constructor** (verified: `regression_benchmarks.rs` line 43):
`FdMatrix::from_column_major(data, n, m).unwrap()` — `.unwrap()` is correct in non-fallible setup code outside `b.iter()`.

For the CV loops sentinel, also add:

```rust
/// Build alternating binary class labels for n curves (0 / 1).
fn make_class_labels(n: usize) -> Vec<usize> {
    (0..n).map(|i| i % 2).collect()
}
```

For the smoothing sentinel, the existing bench uses plain `Vec<f64>` (not `FdMatrix`) — copy the generator from `smoothing_benchmarks.rs` lines 17–29:

```rust
fn generate_smoothing_data(n: usize, m: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| i as f64 / (n - 1) as f64).collect();
    let y: Vec<f64> = x
        .iter()
        .enumerate()
        .map(|(i, &xi)| {
            let noise = ((i as f64 * 17.3 + 0.5).sin()) * 0.3;
            (2.0 * PI * xi).sin() + 0.5 * (4.0 * PI * xi).cos() + noise
        })
        .collect();
    let x_new: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    (x, y, x_new)
}
```

---

#### Criterion group scaffold — single-cell `bench_function` form

**Source:** `fdars-core/benches/regression_benchmarks.rs` lines 58–74 (`bench_fpca`) and `fdars-core/benches/alignment_benchmarks.rs` lines 84–110 (`bench_self_distance_matrix`).

For Phase 1, each audit bench function benchmarks **one baseline cell** (not a loop over sizes — that belongs to Phases 3–6). Use `bench_function` (not `bench_with_input`) for named single-cell groups. Pattern from `alignment_benchmarks.rs` lines 118–130:

```rust
fn bench_fpca_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_fpca");
    // Tune for audit cell: SVD O(m³) at m=200 costs ~100ms/iter
    group.sample_size(20);
    group.measurement_time(std::time::Duration::from_secs(20));
    group.warm_up_time(std::time::Duration::from_secs(5));

    // Build input OUTSIDE b.iter() to avoid measuring the allocator
    let (data, argvals) = generate_curves(500, 200);
    group.bench_function("n500_m200", |b| {
        b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(5usize), black_box(&argvals)))
    });

    group.finish();  // REQUIRED — verified: regression_benchmarks.rs:73, alignment_benchmarks.rs:81
}
```

**Critical:** `group.finish()` must always be called. Every existing bench file ends each group function with `group.finish()` (verified: `regression_benchmarks.rs` line 73; `alignment_benchmarks.rs` lines 81, 109, 146).

---

#### `black_box` on both inputs and outputs

**Source:** `fdars-core/benches/regression_benchmarks.rs` line 68 and `fdars-core/benches/depth_benchmarks.rs` lines 39–40.

```rust
// For Result<T, E> returns (large struct) — wrap inputs; Criterion drop is sufficient for output:
b.iter(|| fdata_to_pc_1d(black_box(&data), black_box(5usize), black_box(&argvals)));

// For Vec<f64> / primitive returns — also wrap output:
b.iter(|| black_box(fraiman_muniz_1d(black_box(data), black_box(data), true)));

// For Result<Vec<f64>, _> returns — unwrap and wrap:
b.iter(|| {
    nadaraya_watson(
        black_box(&x),
        black_box(&y),
        black_box(&x_new),
        black_box(bandwidth),
        black_box("gaussian"),
    )
    .unwrap()
});
```

The `.unwrap()` inside `b.iter()` is the exact pattern used in `smoothing_benchmarks.rs` line 49 — it is correct because a bench panicking on setup data indicates a bug, not a measurement concern.

---

#### Streaming depth sentinel pattern

**Source:** `fdars-core/benches/depth_benchmarks.rs` lines 58–87 (`bench_streaming_construction_and_query`).

```rust
fn bench_streaming_sentinel(c: &mut Criterion) {
    let mut group = c.benchmark_group("audit_streaming_depth");
    group.sample_size(30);
    group.measurement_time(std::time::Duration::from_secs(15));
    group.warm_up_time(std::time::Duration::from_secs(3));

    let data = generate_centered_data(500, 200);
    group.bench_function("n500_m200", |b, | {
        b.iter(|| {
            let state = SortedReferenceState::from_reference(black_box(&data));
            let fm = StreamingFraimanMuniz::new(state, true);
            fm.depth_batch(black_box(&data))
        })
    });

    group.finish();
}
```

**Note:** `from_reference` + `depth_batch` are benchmarked as a single unit inside `b.iter()`, matching the exact pattern in `depth_benchmarks.rs` lines 67–72. This measures the combined construct-then-query cost, which is the representative use case.

---

#### `criterion_group!` / `criterion_main!` macros

**Source:** `fdars-core/benches/regression_benchmarks.rs` lines 142–148 and `fdars-core/benches/alignment_benchmarks.rs` lines 149–155.

```rust
criterion_group!(
    benches,
    bench_fpca_sentinel,
    bench_elastic_sentinel,
    bench_depth_sentinel,
    bench_cv_sentinel,
    bench_streaming_sentinel,
    bench_smooth_sentinel,
);
criterion_main!(benches);
```

This is the exact two-macro pattern every existing bench uses. `criterion_main!(benches)` must reference the name used in `criterion_group!` (here `benches`).

---

## Sentinel Function to Public Path Map

| Module | Sentinel Function | Public Import Path | Analog Bench (call site) |
|--------|------------------|--------------------|--------------------------|
| Elastic alignment | `elastic_self_distance_matrix` | `fdars_core::alignment::elastic_self_distance_matrix` | `alignment_benchmarks.rs` line 93 |
| FPCA/SVD | `fdata_to_pc_1d` | `fdars_core::regression::fdata_to_pc_1d` | `regression_benchmarks.rs` line 68 |
| Depth & distance | `fraiman_muniz_1d` | `fdars_core::depth::fraiman_muniz_1d` | `depth_benchmarks.rs` line 40 |
| CV loops | `fclassif_cv` | `fdars_core::classification::fclassif_cv` | no existing bench — use signature from RESEARCH.md §6 |
| Streaming depth | `StreamingFraimanMuniz::depth_batch` | `fdars_core::streaming_depth::{SortedReferenceState, StreamingDepth, StreamingFraimanMuniz}` | `depth_benchmarks.rs` lines 67–72 |
| Smoothing | `nadaraya_watson` | `fdars_core::smoothing::nadaraya_watson` | `smoothing_benchmarks.rs` lines 41–51 |

---

## Shared Patterns

### Per-group `sample_size` / `measurement_time` tuning

**Source:** `fdars-core/benches/smoothing_benchmarks.rs` (implied by `sample_size(20)` at line 144 for expensive `optim_bandwidth` group). Pattern is per-group, not global.

**Apply to:** Every audit bench group (prevents polluting CI bench timing with audit-sized inputs).

```rust
group.sample_size(10);                                             // minimum for Criterion 0.5
group.measurement_time(std::time::Duration::from_secs(60));        // elastic large cells
group.warm_up_time(std::time::Duration::from_secs(10));
```

Recommended per-module settings (from RESEARCH.md §1 Pattern 3):

| Sentinel | sample_size | measurement_time | warm_up_time |
|----------|-------------|------------------|--------------|
| Elastic N=100, M=50 | 20 | 20s | 5s |
| Elastic N=500, M=200 | 10 | 60s | 10s |
| FPCA N=500, M=200 | 20 | 20s | 5s |
| Depth N=500, M=200 | 30 | 15s | 3s |
| CV N=100, M=50 | 15 | 20s | 5s |
| Streaming N=500, M=200 | 30 | 15s | 3s |
| Smoothing N=500, M=200 | 30 | 10s | 3s |

### Inputs built outside `b.iter()`

**Source:** `fdars-core/benches/regression_benchmarks.rs` lines 63–65 and `fdars-core/benches/alignment_benchmarks.rs` lines 89–90.

**Apply to:** Every bench function. Build all `FdMatrix` / `Vec<f64>` inputs before `group.bench_function(...)`. Only the sentinel function call goes inside `b.iter(|| ...)`.

### Module-level doc comment

**Source:** `fdars-core/benches/regression_benchmarks.rs` lines 1–6 (module-level `//!` block).

**Apply to:** Top of `audit_hotpaths.rs`:

```rust
//! Audit hot-path benchmarks for fdars-core.
//!
//! One representative sentinel per hot-path module at Phase 1 baseline cell.
//! Sizes, caps, and sample_size rationale: see .planning/phases/01-.../01-RESEARCH.md §8.
//! Raw output saved to .planning/research/bench/ per D-06.
```

---

## No Analog Found

| File | Role | Data Flow | Reason |
|------|------|-----------|--------|
| `fclassif_cv` call site | bench call | batch | No existing bench file calls `fclassif_cv`; the `classification_benchmarks.rs` file benchmarks only `fclassif_lda` fit/predict directly. The function signature is available from RESEARCH.md §6 Module 4 — use that. |

---

## Metadata

**Analog search scope:** `fdars-core/benches/` (all 9 existing bench files)
**Files read:** `regression_benchmarks.rs` (148 lines), `alignment_benchmarks.rs` (155 lines), `depth_benchmarks.rs` (100 lines, first section), `smoothing_benchmarks.rs` (56 lines, first section), `fdars-core/Cargo.toml` (lines 54–89)
**Pattern extraction date:** 2026-08-07
