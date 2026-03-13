use super::ToleranceBand;
use crate::matrix::FdMatrix;

/// Column-wise mean and standard deviation.
pub(super) fn pointwise_mean_std(data: &FdMatrix) -> (Vec<f64>, Vec<f64>) {
    let (n, m) = data.shape();
    let nf = n as f64;
    let mut means = vec![0.0; m];
    let mut stds = vec![0.0; m];

    for j in 0..m {
        let col = data.column(j);
        let mean = col.iter().sum::<f64>() / nf;
        means[j] = mean;
        let var = col.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / (nf - 1.0);
        stds[j] = var.sqrt();
    }
    (means, stds)
}

/// Inverse normal CDF (probit) via rational approximation (Abramowitz & Stegun 26.2.23).
pub(super) fn normal_quantile(p: f64) -> f64 {
    // Rational approximation coefficients (Abramowitz & Stegun 26.2.23)
    const C0: f64 = 2.515_517;
    const C1: f64 = 0.802_853;
    const C2: f64 = 0.010_328;
    const D1: f64 = 1.432_788;
    const D2: f64 = 0.189_269;
    const D3: f64 = 0.001_308;

    if p <= 0.0 || p >= 1.0 {
        return f64::NAN;
    }
    if (p - 0.5).abs() < 1e-15 {
        return 0.0;
    }

    // Use symmetry: for p < 0.5, compute for 1-p and negate
    let (sign, q) = if p < 0.5 { (-1.0, 1.0 - p) } else { (1.0, p) };

    let t = (-2.0 * (1.0 - q).ln()).sqrt();

    let numerator = C0 + C1 * t + C2 * t * t;
    let denominator = 1.0 + D1 * t + D2 * t * t + D3 * t * t * t;

    sign * (t - numerator / denominator)
}

/// Construct a tolerance band from center and half-width vectors.
pub(super) fn build_band(center: Vec<f64>, half_width: Vec<f64>) -> ToleranceBand {
    let lower: Vec<f64> = center
        .iter()
        .zip(half_width.iter())
        .map(|(&c, &h)| c - h)
        .collect();
    let upper: Vec<f64> = center
        .iter()
        .zip(half_width.iter())
        .map(|(&c, &h)| c + h)
        .collect();
    ToleranceBand {
        lower,
        upper,
        center,
        half_width,
    }
}

/// Extract the percentile value from a sorted slice.
pub(super) fn percentile_sorted(sorted: &mut [f64], p: f64) -> f64 {
    crate::helpers::sort_nan_safe(sorted);
    let idx = ((sorted.len() as f64 * p).ceil() as usize).min(sorted.len()) - 1;
    sorted[idx]
}

/// Validate common tolerance band parameters. Returns false if any are invalid.
pub(super) fn valid_band_params(
    n: usize,
    m: usize,
    ncomp: usize,
    nb: usize,
    coverage: f64,
) -> bool {
    n >= 3 && m > 0 && ncomp > 0 && nb > 0 && coverage > 0.0 && coverage < 1.0
}
