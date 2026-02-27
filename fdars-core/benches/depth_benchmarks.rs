//! Benchmarks for depth computation methods
//!
//! Compares performance of FM depth, MBD, and outlier detection at various sizes.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::depth::{
    band_1d, fraiman_muniz_1d, functional_spatial_1d, modified_band_1d, random_projection_1d,
};
use fdars_core::matrix::FdMatrix;
use fdars_core::fdata::norm_lp_1d;
use fdars_core::metric::{dtw_self_1d, fourier_self_1d, hausdorff_self_1d, lp_self_1d};
use fdars_core::outliers::outliers_threshold_lrt;
use fdars_core::streaming_depth::{
    SortedReferenceState, StreamingDepth, StreamingFraimanMuniz, StreamingMbd,
};
use std::f64::consts::PI;

/// Generate deterministic centered functional data (n curves, m time points).
fn generate_centered_data(n: usize, m: usize) -> FdMatrix {
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    let mut data = vec![0.0; n * m];
    for i in 0..n {
        let offset = (i as f64 - n as f64 / 2.0) / (n as f64);
        for j in 0..m {
            data[i + j * n] = (2.0 * PI * argvals[j]).sin() + offset;
        }
    }
    FdMatrix::from_column_major(data, n, m).unwrap()
}

fn bench_fraiman_muniz(c: &mut Criterion) {
    let mut group = c.benchmark_group("fraiman_muniz_1d");
    let t = 200;
    for &n in &[50, 200, 500, 1000, 2300] {
        let data = generate_centered_data(n, t);
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| fraiman_muniz_1d(black_box(data), black_box(data), true))
        });
    }
    group.finish();
}

fn bench_modified_band(c: &mut Criterion) {
    let mut group = c.benchmark_group("modified_band_1d");
    let t = 200;
    for &n in &[50, 200, 500, 1000, 2300] {
        let data = generate_centered_data(n, t);
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| modified_band_1d(black_box(data), black_box(data)))
        });
    }
    group.finish();
}

fn bench_streaming_construction_and_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_depth");
    let t = 200;
    for &n in &[50, 200, 500, 1000, 2300] {
        let data = generate_centered_data(n, t);
        group.bench_with_input(BenchmarkId::new("FM_construct+query/N", n), &data, |b, data| {
            b.iter(|| {
                let state = SortedReferenceState::from_reference(black_box(data));
                let fm = StreamingFraimanMuniz::new(state, true);
                fm.depth_batch(black_box(data))
            })
        });
        group.bench_with_input(BenchmarkId::new("MBD_construct+query/N", n), &data, |b, data| {
            b.iter(|| {
                let state = SortedReferenceState::from_reference(black_box(data));
                let mbd = StreamingMbd::new(state);
                mbd.depth_batch(black_box(data))
            })
        });
    }
    group.finish();
}

fn bench_outliers(c: &mut Criterion) {
    let mut group = c.benchmark_group("outliers_threshold_lrt");
    let n = 100;
    let t = 200;
    let data = generate_centered_data(n, t);
    for &nb in &[10, 50] {
        group.bench_with_input(BenchmarkId::new("nb", nb), &nb, |b, &nb| {
            b.iter(|| {
                outliers_threshold_lrt(black_box(&data), nb, 0.1, 0.1, 42, 0.95)
            })
        });
    }
    group.finish();
}

fn bench_random_projection(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_projection_1d");
    let t = 200;
    let nproj = 100;
    for &n in &[50, 200, 500] {
        let data = generate_centered_data(n, t);
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| random_projection_1d(black_box(data), black_box(data), nproj))
        });
    }
    group.finish();
}

fn bench_functional_spatial(c: &mut Criterion) {
    let mut group = c.benchmark_group("functional_spatial_1d");
    let t = 200;
    for &n in &[50, 200, 500] {
        let data = generate_centered_data(n, t);
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| functional_spatial_1d(black_box(data), black_box(data)))
        });
    }
    group.finish();
}

fn bench_band_depth(c: &mut Criterion) {
    let mut group = c.benchmark_group("band_1d");
    let t = 200;
    for &n in &[50, 200] {
        let data = generate_centered_data(n, t);
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| band_1d(black_box(data), black_box(data)))
        });
    }
    group.finish();
}

fn bench_dtw(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtw_self_1d");
    let m = 100;
    for &n in &[20, 50] {
        let data = generate_centered_data(n, m);
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| dtw_self_1d(black_box(data), 2.0, 10))
        });
    }
    group.finish();
}

fn bench_hausdorff(c: &mut Criterion) {
    let mut group = c.benchmark_group("hausdorff_self_1d");
    let m = 100;
    for &n in &[20, 50] {
        let data = generate_centered_data(n, m);
        let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| hausdorff_self_1d(black_box(data), black_box(&argvals)))
        });
    }
    group.finish();
}

fn bench_fourier(c: &mut Criterion) {
    let mut group = c.benchmark_group("fourier_self_1d");
    let m = 100;
    for &n in &[50, 200] {
        let data = generate_centered_data(n, m);
        group.bench_with_input(BenchmarkId::new("N", n), &data, |b, data| {
            b.iter(|| fourier_self_1d(black_box(data), 10))
        });
    }
    group.finish();
}

fn bench_lp_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("lp_self_1d");
    let m = 200;
    let n = 100;
    let data = generate_centered_data(n, m);
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    for &p in &[1.0, 2.0, 3.0] {
        group.bench_with_input(BenchmarkId::new("p", p as i32), &p, |b, &p| {
            b.iter(|| lp_self_1d(black_box(&data), black_box(&argvals), p, &[]))
        });
    }
    group.finish();
}

fn bench_norm_lp(c: &mut Criterion) {
    let mut group = c.benchmark_group("norm_lp_1d");
    let m = 200;
    let n = 500;
    let data = generate_centered_data(n, m);
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
    for &p in &[1.0, 2.0, 3.0] {
        group.bench_with_input(BenchmarkId::new("p", p as i32), &p, |b, &p| {
            b.iter(|| norm_lp_1d(black_box(&data), black_box(&argvals), p))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_fraiman_muniz,
    bench_modified_band,
    bench_streaming_construction_and_query,
    bench_outliers,
    bench_random_projection,
    bench_functional_spatial,
    bench_band_depth,
    bench_dtw,
    bench_hausdorff,
    bench_fourier,
    bench_lp_distance,
    bench_norm_lp,
);
criterion_main!(benches);
