//! Benchmarks for the FOptDes optimal experimental design pipeline (v0.35.0).
//!
//! Measures the two public entry points on a SMALL fitted PACE model:
//! - `design_criterion` — scoring a fixed 5-point design (Trajectory + Score(A)).
//! - `optimal_design` — full greedy forward-selection (budget 5, Trajectory + Score(A)).
//!
//! The model is fit via the real `pace_fpca` path because `PaceFpcaResult` is
//! `#[non_exhaustive]` and cannot be struct-literal-constructed from this external
//! benches crate. The dataset is intentionally tiny so `cargo build --benches` and
//! `cargo bench` stay quick.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use fdars_core::irreg_fdata::IrregFdata;
use fdars_core::pace_fpca::{pace_fpca, PaceFpcaConfig, PaceFpcaResult};
use fdars_core::{design_criterion, optimal_design, DesignCriterion, OptDesConfig, OptimalityKind};

/// Fit a small synthetic PACE model on an `m`-point work grid (>= 2 components).
fn bench_model(m: usize) -> PaceFpcaResult {
    let argvals_list = vec![
        vec![0.1, 0.4, 0.7],
        vec![0.0, 0.3, 0.6, 0.9],
        vec![0.2, 0.5, 0.8],
        vec![0.0, 0.25, 0.5, 0.75, 1.0],
        vec![0.1, 0.5, 0.9],
        vec![0.0, 0.4, 0.8],
    ];
    let values_list: Vec<Vec<f64>> = argvals_list
        .iter()
        .enumerate()
        .map(|(i, ts)| {
            ts.iter()
                .map(|&t: &f64| (i as f64 + 1.0) * t.sin())
                .collect()
        })
        .collect();
    let data = IrregFdata::from_lists(&argvals_list, &values_list);

    let config = PaceFpcaConfig {
        ncomp: 2,
        bandwidth: 0.2,
        sigma2: 0.01,
        work_grid: (0..m).map(|i| i as f64 / (m - 1) as f64).collect(),
        alpha: 0.05,
    };
    pace_fpca(&data, &config).expect("bench model must fit")
}

fn bench_optimal_design(c: &mut Criterion) {
    let m = 51usize;
    let model = bench_model(m);
    let candidate_grid: Vec<f64> = model.argvals.clone();
    let fixed_design = [5usize, 12, 25, 38, 45];

    let mut group = c.benchmark_group("optimal_design");

    // design_criterion alone (Trajectory, 5 points).
    group.bench_function("design_criterion_trajectory_p5_m51", |b| {
        b.iter(|| {
            design_criterion(
                black_box(&model),
                black_box(&fixed_design),
                black_box(DesignCriterion::Trajectory),
            )
        });
    });

    // design_criterion alone (Score A, 5 points).
    group.bench_function("design_criterion_score_a_p5_m51", |b| {
        b.iter(|| {
            design_criterion(
                black_box(&model),
                black_box(&fixed_design),
                black_box(DesignCriterion::Score(OptimalityKind::A)),
            )
        });
    });

    // Full greedy selection (Trajectory, budget 5, grid 51).
    let cfg_traj = OptDesConfig {
        candidate_grid: candidate_grid.clone(),
        budget: 5,
        criterion: DesignCriterion::Trajectory,
    };
    group.bench_function("optimal_design_trajectory_budget5_m51", |b| {
        b.iter(|| optimal_design(black_box(&model), black_box(&cfg_traj)));
    });

    // Full greedy selection (Score A, budget 5, grid 51).
    let cfg_score = OptDesConfig {
        candidate_grid,
        budget: 5,
        criterion: DesignCriterion::Score(OptimalityKind::A),
    };
    group.bench_function("optimal_design_score_a_budget5_m51", |b| {
        b.iter(|| optimal_design(black_box(&model), black_box(&cfg_score)));
    });

    group.finish();
}

criterion_group!(benches, bench_optimal_design);
criterion_main!(benches);
