use num_complex::Complex;
use rustfft::FftPlanner;
use std::f64::consts::PI;

use super::{compute_mean_curve, unwrap_phase, InstantaneousPeriod};
use crate::matrix::FdMatrix;

/// Hilbert transform using FFT to compute analytic signal.
///
/// # Arguments
/// * `signal` - Input real signal
///
/// # Returns
/// Analytic signal as complex vector (real part = original, imaginary = Hilbert transform)
pub fn hilbert_transform(signal: &[f64]) -> Vec<Complex<f64>> {
    let n = signal.len();
    if n == 0 {
        return Vec::new();
    }

    let mut planner = FftPlanner::<f64>::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);

    // Forward FFT
    let mut buffer: Vec<Complex<f64>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft_forward.process(&mut buffer);

    // Create analytic signal in frequency domain
    // H[0] = 1 (DC), positive freqs = 2, Nyquist = 1 (even only), negative freqs = 0
    let half = n / 2;
    if n % 2 == 0 {
        // Even: bins 1..half-1 get 2, bin half (Nyquist) stays 1, rest 0
        for k in 1..half {
            buffer[k] *= 2.0;
        }
        for k in (half + 1)..n {
            buffer[k] = Complex::new(0.0, 0.0);
        }
    } else {
        // Odd: bins 1..=half get 2, rest 0 (no Nyquist bin)
        for k in 1..=half {
            buffer[k] *= 2.0;
        }
        for k in (half + 1)..n {
            buffer[k] = Complex::new(0.0, 0.0);
        }
    }

    // Inverse FFT
    fft_inverse.process(&mut buffer);

    // Normalize
    for c in &mut buffer {
        *c /= n as f64;
    }

    buffer
}

/// Estimate instantaneous period using Hilbert transform.
///
/// For series with drifting/changing period, this computes the period
/// at each time point using the analytic signal.
///
/// # Arguments
/// * `data` - Column-major matrix (n x m)
/// * `argvals` - Evaluation points
pub fn instantaneous_period(data: &FdMatrix, argvals: &[f64]) -> InstantaneousPeriod {
    let (n, m) = data.shape();
    if n == 0 || m < 4 || argvals.len() != m {
        return InstantaneousPeriod {
            period: Vec::new(),
            frequency: Vec::new(),
            amplitude: Vec::new(),
        };
    }

    // Compute mean curve
    let mean_curve = compute_mean_curve(data);

    // Remove DC component (detrend by subtracting mean)
    let dc: f64 = mean_curve.iter().sum::<f64>() / m as f64;
    let detrended: Vec<f64> = mean_curve.iter().map(|&x| x - dc).collect();

    // Compute analytic signal via Hilbert transform
    let analytic = hilbert_transform(&detrended);

    // Extract instantaneous amplitude and phase
    let amplitude: Vec<f64> = analytic.iter().map(|c| c.norm()).collect();

    let phase: Vec<f64> = analytic.iter().map(|c| c.im.atan2(c.re)).collect();

    // Unwrap phase
    let unwrapped_phase = unwrap_phase(&phase);

    // Compute instantaneous frequency (derivative of phase)
    let dt = (argvals[m - 1] - argvals[0]) / (m - 1) as f64;
    let mut inst_freq = vec![0.0; m];

    // Central differences for interior, forward/backward at boundaries
    if m > 1 {
        inst_freq[0] = (unwrapped_phase[1] - unwrapped_phase[0]) / dt / (2.0 * PI);
    }
    for j in 1..(m - 1) {
        inst_freq[j] = (unwrapped_phase[j + 1] - unwrapped_phase[j - 1]) / (2.0 * dt) / (2.0 * PI);
    }
    if m > 1 {
        inst_freq[m - 1] = (unwrapped_phase[m - 1] - unwrapped_phase[m - 2]) / dt / (2.0 * PI);
    }

    // Period = 1/frequency (handle near-zero frequencies)
    let period: Vec<f64> = inst_freq
        .iter()
        .map(|&f| {
            if f.abs() > 1e-10 {
                (1.0 / f).abs()
            } else {
                f64::INFINITY
            }
        })
        .collect();

    InstantaneousPeriod {
        period,
        frequency: inst_freq,
        amplitude,
    }
}
