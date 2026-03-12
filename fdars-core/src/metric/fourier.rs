//! Fourier-based semimetric for functional data.

use crate::iter_maybe_parallel;
use crate::matrix::FdMatrix;
#[cfg(feature = "parallel")]
use rayon::iter::ParallelIterator;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use super::{cross_distance_matrix, self_distance_matrix};

/// Compute Fourier coefficients for a curve using a pre-planned FFT.
fn fft_coefficients_with_plan(data: &[f64], nfreq: usize, fft: &dyn rustfft::Fft<f64>) -> Vec<f64> {
    let n = data.len();
    let nfreq = nfreq.min(n / 2);
    let mut buffer: Vec<Complex<f64>> = data.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buffer);
    buffer
        .iter()
        .take(nfreq + 1)
        .map(|c| c.norm() / n as f64)
        .collect()
}

/// Compute semimetric based on Fourier coefficients for self-distances.
pub fn fourier_self_1d(data: &FdMatrix, nfreq: usize) -> FdMatrix {
    let n = data.nrows();
    let m = data.ncols();
    if n == 0 || m == 0 {
        return FdMatrix::zeros(0, 0);
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(m);
    let rows: Vec<Vec<f64>> = (0..n).map(|i| data.row(i)).collect();
    let coeffs: Vec<Vec<f64>> = iter_maybe_parallel!(0..n)
        .map(|i| fft_coefficients_with_plan(&rows[i], nfreq, fft.as_ref()))
        .collect();
    self_distance_matrix(n, |i, j| {
        coeffs[i]
            .iter()
            .zip(coeffs[j].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    })
}

/// Compute semimetric based on Fourier coefficients for cross-distances.
pub fn fourier_cross_1d(data1: &FdMatrix, data2: &FdMatrix, nfreq: usize) -> FdMatrix {
    let n1 = data1.nrows();
    let n2 = data2.nrows();
    let m = data1.ncols();
    if n1 == 0 || n2 == 0 || m == 0 || data2.ncols() != m {
        return FdMatrix::zeros(0, 0);
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(m);
    let rows1: Vec<Vec<f64>> = (0..n1).map(|i| data1.row(i)).collect();
    let rows2: Vec<Vec<f64>> = (0..n2).map(|i| data2.row(i)).collect();
    let coeffs1: Vec<Vec<f64>> = iter_maybe_parallel!(0..n1)
        .map(|i| fft_coefficients_with_plan(&rows1[i], nfreq, fft.as_ref()))
        .collect();
    let coeffs2: Vec<Vec<f64>> = iter_maybe_parallel!(0..n2)
        .map(|i| fft_coefficients_with_plan(&rows2[i], nfreq, fft.as_ref()))
        .collect();
    cross_distance_matrix(n1, n2, |i, j| {
        coeffs1[i]
            .iter()
            .zip(coeffs2[j].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    })
}
