//! Exact DTW 1-NN search: cascading lower bounds + EAPrunedDTW vs brute force (GH #58).
//!
//! Data: 8 noisy random-walk prototypes (UCR-like), 500 references and 20
//! queries of length 128, Sakoe-Chiba radius 12 (~10 %), squared cost.
//! - `one_query`: a single query (sequential in both variants).
//! - `batch`: all 20 queries (both variants parallel over queries).
//!
//! Run: `cargo bench -p fdars-core --bench dtw_lb_benchmarks -- --quick`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::matrix::FdMatrix;
use fdars_core::{dtw_distance_windowed, dtw_nearest_neighbor, DtwWindow};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;

const M: usize = 128;
const N_REF: usize = 500;
const N_QUERY: usize = 20;

fn dataset(seed: u64) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let protos: Vec<Vec<f64>> = (0..8)
        .map(|_| {
            let mut v = 0.0;
            (0..M)
                .map(|_| {
                    v += rng.gen_range(-1.0..1.0);
                    v
                })
                .collect()
        })
        .collect();
    let mut sample = |k: usize| -> Vec<f64> {
        let shift = rng.gen_range(0..4usize);
        (0..M)
            .map(|j| protos[k % 8][(j + shift).min(M - 1)] + rng.gen_range(-0.5..0.5))
            .collect()
    };
    let refs: Vec<Vec<f64>> = (0..N_REF).map(&mut sample).collect();
    let queries: Vec<Vec<f64>> = (0..N_QUERY).map(&mut sample).collect();
    (refs, queries)
}

fn to_matrix(rows: &[Vec<f64>]) -> FdMatrix {
    let n = rows.len();
    let mut data = vec![0.0; n * M];
    for (i, r) in rows.iter().enumerate() {
        for (j, &v) in r.iter().enumerate() {
            data[i + j * n] = v;
        }
    }
    FdMatrix::from_column_major(data, n, M).unwrap()
}

fn brute_one(q: &[f64], refs: &[Vec<f64>], w: DtwWindow) -> (usize, f64) {
    let mut best = (usize::MAX, f64::INFINITY);
    for (k, r) in refs.iter().enumerate() {
        let d = dtw_distance_windowed(q, r, 2.0, w).unwrap();
        if d < best.1 {
            best = (k, d);
        }
    }
    best
}

fn brute_batch(queries: &[Vec<f64>], refs: &[Vec<f64>], w: DtwWindow) -> Vec<(usize, f64)> {
    #[cfg(feature = "parallel")]
    let it = queries.par_iter();
    #[cfg(not(feature = "parallel"))]
    let it = queries.iter();
    it.map(|q| brute_one(q, refs, w)).collect()
}

fn bench_dtw_nn(c: &mut Criterion) {
    let (refs, queries) = dataset(58);
    let (rm, qm) = (to_matrix(&refs), to_matrix(&queries));
    let w = DtwWindow::SakoeChiba { radius: 12 };

    // Sanity: identical answers.
    let nn = dtw_nearest_neighbor(&qm, &rm, 2.0, w).unwrap();
    let bf = brute_batch(&queries, &refs, w);
    for (q, &(k, d)) in bf.iter().enumerate() {
        assert_eq!((nn.indices[q], nn.distances[q].to_bits()), (k, d.to_bits()));
    }
    eprintln!(
        "dtw_lb pruning stats ({N_QUERY} queries x {N_REF} refs): {:?}",
        nn.stats
    );

    let q1 = to_matrix(&queries[..1]);
    let mut group = c.benchmark_group("dtw_1nn_one_query");
    group.bench_function("brute_force", |b| {
        b.iter(|| brute_one(black_box(&queries[0]), black_box(&refs), w))
    });
    group.bench_function("lb_cascade_eapruned", |b| {
        b.iter(|| dtw_nearest_neighbor(black_box(&q1), black_box(&rm), 2.0, w).unwrap())
    });
    group.finish();

    let mut group = c.benchmark_group("dtw_1nn_batch");
    group.sample_size(10);
    group.bench_function("brute_force", |b| {
        b.iter(|| brute_batch(black_box(&queries), black_box(&refs), w))
    });
    group.bench_function("lb_cascade_eapruned", |b| {
        b.iter(|| dtw_nearest_neighbor(black_box(&qm), black_box(&rm), 2.0, w).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_dtw_nn);
criterion_main!(benches);
