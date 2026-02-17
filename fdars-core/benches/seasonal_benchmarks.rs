//! Benchmarks for seasonal/periodicity detection methods
//!
//! Compares performance of:
//! - SAZED (ensemble method)
//! - Autoperiod (hybrid FFT + ACF)
//! - CFDAutoperiod (clustered filtered detrended)
//! - Basic FFT and ACF methods

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use fdars_core::matrix::FdMatrix;
use fdars_core::seasonal::{
    autoperiod, cfd_autoperiod, estimate_period_acf, estimate_period_fft, sazed,
};
use std::f64::consts::PI;

/// Generate a pure sine wave signal
fn generate_sine(m: usize, period: f64) -> (Vec<f64>, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period).sin())
        .collect();
    (data, argvals)
}

/// Generate a sine wave with linear trend
fn generate_sine_with_trend(m: usize, period: f64, trend_slope: f64) -> (Vec<f64>, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| trend_slope * t + (2.0 * PI * t / period).sin())
        .collect();
    (data, argvals)
}

/// Generate a noisy sine wave
fn generate_noisy_sine(m: usize, period: f64, noise_level: f64) -> (Vec<f64>, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .enumerate()
        .map(|(i, &t)| {
            let signal = (2.0 * PI * t / period).sin();
            // Deterministic pseudo-noise for reproducibility
            let noise = noise_level * ((17.3 * i as f64).sin());
            signal + noise
        })
        .collect();
    (data, argvals)
}

/// Generate multi-period signal
fn generate_multi_period(m: usize, period1: f64, period2: f64) -> (Vec<f64>, Vec<f64>) {
    let argvals: Vec<f64> = (0..m).map(|i| i as f64 * 0.1).collect();
    let data: Vec<f64> = argvals
        .iter()
        .map(|&t| (2.0 * PI * t / period1).sin() + 0.5 * (2.0 * PI * t / period2).sin())
        .collect();
    (data, argvals)
}

/// Benchmark SAZED with different signal lengths
fn bench_sazed(c: &mut Criterion) {
    let mut group = c.benchmark_group("SAZED");

    for size in [100, 200, 500, 1000, 2000].iter() {
        let (data, argvals) = generate_sine(*size, 2.0);

        group.bench_with_input(BenchmarkId::new("pure_sine", size), size, |b, _| {
            b.iter(|| sazed(black_box(&data), black_box(&argvals), None))
        });
    }

    // Benchmark with noisy signal
    let (data, argvals) = generate_noisy_sine(500, 2.0, 0.2);
    group.bench_function("noisy_500", |b| {
        b.iter(|| sazed(black_box(&data), black_box(&argvals), None))
    });

    // Benchmark with trend
    let (data, argvals) = generate_sine_with_trend(500, 2.0, 0.1);
    group.bench_function("trend_500", |b| {
        b.iter(|| sazed(black_box(&data), black_box(&argvals), None))
    });

    group.finish();
}

/// Benchmark Autoperiod with different signal lengths
fn bench_autoperiod(c: &mut Criterion) {
    let mut group = c.benchmark_group("Autoperiod");

    for size in [100, 200, 500, 1000, 2000].iter() {
        let (data, argvals) = generate_sine(*size, 2.0);

        group.bench_with_input(BenchmarkId::new("pure_sine", size), size, |b, _| {
            b.iter(|| autoperiod(black_box(&data), black_box(&argvals), None, None))
        });
    }

    // Benchmark with different n_candidates
    let (data, argvals) = generate_sine(500, 2.0);
    for n_cand in [3, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::new("candidates", n_cand),
            n_cand,
            |b, &n_cand| {
                b.iter(|| {
                    autoperiod(
                        black_box(&data),
                        black_box(&argvals),
                        Some(n_cand),
                        Some(10),
                    )
                })
            },
        );
    }

    // Benchmark with noisy signal
    let (data, argvals) = generate_noisy_sine(500, 2.0, 0.2);
    group.bench_function("noisy_500", |b| {
        b.iter(|| autoperiod(black_box(&data), black_box(&argvals), None, None))
    });

    group.finish();
}

/// Benchmark CFDAutoperiod with different signal lengths
fn bench_cfd_autoperiod(c: &mut Criterion) {
    let mut group = c.benchmark_group("CFDAutoperiod");

    for size in [100, 200, 500, 1000, 2000].iter() {
        let (data, argvals) = generate_sine(*size, 2.0);

        group.bench_with_input(BenchmarkId::new("pure_sine", size), size, |b, _| {
            b.iter(|| cfd_autoperiod(black_box(&data), black_box(&argvals), None, None))
        });
    }

    // Benchmark with trend (CFD's specialty)
    let (data, argvals) = generate_sine_with_trend(500, 2.0, 0.5);
    group.bench_function("trend_500", |b| {
        b.iter(|| cfd_autoperiod(black_box(&data), black_box(&argvals), None, None))
    });

    // Benchmark with multiple periods
    let (data, argvals) = generate_multi_period(500, 2.0, 5.0);
    group.bench_function("multi_period_500", |b| {
        b.iter(|| cfd_autoperiod(black_box(&data), black_box(&argvals), None, None))
    });

    group.finish();
}

/// Benchmark basic FFT period estimation
fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("FFT");

    for size in [100, 200, 500, 1000, 2000].iter() {
        let (data, argvals) = generate_sine(*size, 2.0);
        let mat = FdMatrix::from_column_major(data, 1, *size).unwrap();

        group.bench_with_input(BenchmarkId::new("pure_sine", size), size, |b, _| {
            b.iter(|| estimate_period_fft(black_box(&mat), black_box(&argvals)))
        });
    }

    group.finish();
}

/// Benchmark basic ACF period estimation
fn bench_acf(c: &mut Criterion) {
    let mut group = c.benchmark_group("ACF");

    for size in [100, 200, 500, 1000, 2000].iter() {
        let (data, argvals) = generate_sine(*size, 2.0);
        let max_lag = size / 2;

        group.bench_with_input(BenchmarkId::new("pure_sine", size), size, |b, _| {
            b.iter(|| estimate_period_acf(black_box(&data), 1, *size, black_box(&argvals), max_lag))
        });
    }

    group.finish();
}

/// Compare all methods on the same signal
fn bench_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Method_Comparison");

    let (data, argvals) = generate_sine(500, 2.0);
    let mat = FdMatrix::from_column_major(data.clone(), 1, 500).unwrap();
    let max_lag = 250;

    group.bench_function("FFT", |b| {
        b.iter(|| estimate_period_fft(black_box(&mat), black_box(&argvals)))
    });

    group.bench_function("ACF", |b| {
        b.iter(|| estimate_period_acf(black_box(&data), 1, 500, black_box(&argvals), max_lag))
    });

    group.bench_function("SAZED", |b| {
        b.iter(|| sazed(black_box(&data), black_box(&argvals), None))
    });

    group.bench_function("Autoperiod", |b| {
        b.iter(|| autoperiod(black_box(&data), black_box(&argvals), None, None))
    });

    group.bench_function("CFDAutoperiod", |b| {
        b.iter(|| cfd_autoperiod(black_box(&data), black_box(&argvals), None, None))
    });

    group.finish();
}

/// Benchmark with trended data (challenging scenario)
fn bench_trended_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("Trended_Comparison");

    let (data, argvals) = generate_sine_with_trend(500, 2.0, 0.3);
    let mat = FdMatrix::from_column_major(data.clone(), 1, 500).unwrap();
    let max_lag = 250;

    group.bench_function("FFT", |b| {
        b.iter(|| estimate_period_fft(black_box(&mat), black_box(&argvals)))
    });

    group.bench_function("ACF", |b| {
        b.iter(|| estimate_period_acf(black_box(&data), 1, 500, black_box(&argvals), max_lag))
    });

    group.bench_function("SAZED", |b| {
        b.iter(|| sazed(black_box(&data), black_box(&argvals), None))
    });

    group.bench_function("Autoperiod", |b| {
        b.iter(|| autoperiod(black_box(&data), black_box(&argvals), None, None))
    });

    group.bench_function("CFDAutoperiod", |b| {
        b.iter(|| cfd_autoperiod(black_box(&data), black_box(&argvals), None, None))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_sazed,
    bench_autoperiod,
    bench_cfd_autoperiod,
    bench_fft,
    bench_acf,
    bench_comparison,
    bench_trended_comparison,
);

criterion_main!(benches);
