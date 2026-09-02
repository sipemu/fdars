//! Benchmark for the bundled shapelet-transform classifier pipeline.
//!
//! Measures the end-to-end `shapelet_classifier_fit` (discover → transform →
//! classify) on a small synthetic 2-class dataset. The dataset is kept small so
//! the benchmark runs quickly.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::matrix::FdMatrix;
use fdars_core::{shapelet_classifier_fit, ShapeletClassifierConfig, ShapeletDiscoveryConfig};

/// Two-class dataset: class 1 carries a triangular motif class 0 lacks.
fn labeled_dataset(n: usize, m: usize) -> (FdMatrix, Vec<usize>) {
    let mut flat = vec![0.0f64; n * m];
    let mut labels = vec![0usize; n];
    let motif_start = m / 2;
    let motif_len = (m / 4).max(1);
    for i in 0..n {
        let class1 = i % 2 == 1;
        labels[i] = usize::from(class1);
        let offset = 0.01 * (i as f64);
        for j in 0..m {
            let mut v = offset + (j as f64) * 0.001;
            let hash = (i.wrapping_mul(2654435761) ^ j.wrapping_mul(40503)) % 211;
            v += 0.05 * (hash as f64 / 211.0 - 0.5);
            if class1 && j >= motif_start && j < motif_start + motif_len {
                let k = j - motif_start;
                let half = motif_len / 2;
                let tri = if k <= half {
                    k as f64
                } else {
                    (motif_len - k) as f64
                };
                v += tri;
            }
            flat[i + j * n] = v;
        }
    }
    (FdMatrix::from_column_major(flat, n, m).unwrap(), labels)
}

fn bench_shapelet_classifier_fit(c: &mut Criterion) {
    let mut group = c.benchmark_group("shapelet_classifier_fit");
    // Small dataset keeps the bench quick.
    let (data, labels) = labeled_dataset(24, 24);
    let cfg = ShapeletClassifierConfig {
        discovery: ShapeletDiscoveryConfig {
            min_length: 3,
            max_length: 6,
            max_candidates: Some(500),
            max_shapelets: 4,
            seed: 0,
            ..Default::default()
        },
        ..Default::default()
    };

    group.bench_function("knn_n24_m24", |b| {
        b.iter(|| shapelet_classifier_fit(black_box(&data), black_box(&labels), black_box(&cfg)));
    });

    group.finish();
}

criterion_group!(benches, bench_shapelet_classifier_fit);
criterion_main!(benches);
