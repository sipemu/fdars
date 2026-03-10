//! Helper functions for numerical integration and common operations.

/// Small epsilon for numerical comparisons (e.g., avoiding division by zero).
pub const NUMERICAL_EPS: f64 = 1e-10;

/// Default convergence tolerance for iterative algorithms.
pub const DEFAULT_CONVERGENCE_TOL: f64 = 1e-6;

/// Extract curves from column-major data matrix.
///
/// Converts a flat column-major matrix into a vector of curve vectors,
/// where each curve contains all evaluation points for one observation.
///
/// # Arguments
/// * `data` - Functional data matrix (n x m)
///
/// # Returns
/// Vector of n curves, each containing m values
pub fn extract_curves(data: &crate::matrix::FdMatrix) -> Vec<Vec<f64>> {
    data.rows()
}

/// Compute L2 distance between two curves using integration weights.
///
/// # Arguments
/// * `curve1` - First curve values
/// * `curve2` - Second curve values
/// * `weights` - Integration weights
///
/// # Returns
/// L2 distance between the curves
pub fn l2_distance(curve1: &[f64], curve2: &[f64], weights: &[f64]) -> f64 {
    let mut dist_sq = 0.0;
    for i in 0..curve1.len() {
        let diff = curve1[i] - curve2[i];
        dist_sq += diff * diff * weights[i];
    }
    dist_sq.sqrt()
}

/// Compute Simpson's 1/3 rule integration weights for a grid.
///
/// For odd n (even number of intervals): standard composite Simpson's 1/3 rule.
/// For even n: Simpson's 1/3 for first n-1 points, trapezoidal for last interval.
/// For non-uniform grids: generalized Simpson's weights per sub-interval pair.
///
/// # Arguments
/// * `argvals` - Grid points (evaluation points)
///
/// # Returns
/// Vector of integration weights
pub fn simpsons_weights(argvals: &[f64]) -> Vec<f64> {
    let n = argvals.len();
    if n < 2 {
        return vec![1.0; n];
    }

    let mut weights = vec![0.0; n];

    if n == 2 {
        // Trapezoidal rule
        let h = argvals[1] - argvals[0];
        weights[0] = h / 2.0;
        weights[1] = h / 2.0;
        return weights;
    }

    // Check if grid is uniform
    let h0 = argvals[1] - argvals[0];
    let is_uniform = argvals
        .windows(2)
        .all(|w| ((w[1] - w[0]) - h0).abs() < 1e-12 * h0.abs());

    if is_uniform {
        simpsons_weights_uniform(&mut weights, n, h0);
    } else {
        simpsons_weights_nonuniform(&mut weights, argvals, n);
    }

    weights
}

/// Uniform grid Simpson's 1/3 weights.
fn simpsons_weights_uniform(weights: &mut [f64], n: usize, h0: f64) {
    let n_intervals = n - 1;
    if n_intervals % 2 == 0 {
        // Even number of intervals (odd n): pure Simpson's
        weights[0] = h0 / 3.0;
        weights[n - 1] = h0 / 3.0;
        for i in 1..n - 1 {
            weights[i] = if i % 2 == 1 {
                4.0 * h0 / 3.0
            } else {
                2.0 * h0 / 3.0
            };
        }
    } else {
        // Odd number of intervals (even n): Simpson's + trapezoidal for last
        let n_simp = n - 1;
        weights[0] = h0 / 3.0;
        weights[n_simp - 1] = h0 / 3.0;
        for i in 1..n_simp - 1 {
            weights[i] = if i % 2 == 1 {
                4.0 * h0 / 3.0
            } else {
                2.0 * h0 / 3.0
            };
        }
        weights[n_simp - 1] += h0 / 2.0;
        weights[n - 1] += h0 / 2.0;
    }
}

/// Non-uniform grid generalized Simpson's weights.
fn simpsons_weights_nonuniform(weights: &mut [f64], argvals: &[f64], n: usize) {
    let n_intervals = n - 1;
    let n_pairs = n_intervals / 2;

    for k in 0..n_pairs {
        let i0 = 2 * k;
        let i1 = i0 + 1;
        let i2 = i0 + 2;
        let h1 = argvals[i1] - argvals[i0];
        let h2 = argvals[i2] - argvals[i1];
        let h_sum = h1 + h2;

        weights[i0] += (2.0 * h1 - h2) * h_sum / (6.0 * h1);
        weights[i1] += h_sum * h_sum * h_sum / (6.0 * h1 * h2);
        weights[i2] += (2.0 * h2 - h1) * h_sum / (6.0 * h2);
    }

    if n_intervals % 2 == 1 {
        let h_last = argvals[n - 1] - argvals[n - 2];
        weights[n - 2] += h_last / 2.0;
        weights[n - 1] += h_last / 2.0;
    }
}

/// Compute 2D integration weights using tensor product of 1D weights.
///
/// Returns a flattened vector of weights for an m1 x m2 grid.
///
/// # Arguments
/// * `argvals_s` - Grid points in s direction
/// * `argvals_t` - Grid points in t direction
///
/// # Returns
/// Flattened vector of integration weights (row-major order)
pub fn simpsons_weights_2d(argvals_s: &[f64], argvals_t: &[f64]) -> Vec<f64> {
    let weights_s = simpsons_weights(argvals_s);
    let weights_t = simpsons_weights(argvals_t);
    let m1 = argvals_s.len();
    let m2 = argvals_t.len();

    let mut weights = vec![0.0; m1 * m2];
    for i in 0..m1 {
        for j in 0..m2 {
            weights[i * m2 + j] = weights_s[i] * weights_t[j];
        }
    }
    weights
}

/// Linear interpolation at point `t` using binary search.
///
/// Clamps to boundary values outside the domain of `x`.
pub fn linear_interp(x: &[f64], y: &[f64], t: f64) -> f64 {
    if t <= x[0] {
        return y[0];
    }
    let last = x.len() - 1;
    if t >= x[last] {
        return y[last];
    }

    let idx = match x.binary_search_by(|v| v.partial_cmp(&t).unwrap()) {
        Ok(i) => return y[i],
        Err(i) => i,
    };

    let t0 = x[idx - 1];
    let t1 = x[idx];
    let y0 = y[idx - 1];
    let y1 = y[idx];
    y0 + (y1 - y0) * (t - t0) / (t1 - t0)
}

/// Cumulative integration using Simpson's rule where possible.
///
/// For pairs of intervals uses Simpson's 1/3 rule for higher accuracy.
/// Falls back to trapezoidal for the last interval if n is even.
pub fn cumulative_trapz(y: &[f64], x: &[f64]) -> Vec<f64> {
    let n = y.len();
    let mut out = vec![0.0; n];
    if n < 2 {
        return out;
    }

    // Process pairs of intervals with Simpson's rule
    let mut k = 1;
    while k + 1 < n {
        let h1 = x[k] - x[k - 1];
        let h2 = x[k + 1] - x[k];
        let h_sum = h1 + h2;

        // Generalized Simpson's for this pair of intervals
        let integral = h_sum / 6.0
            * (y[k - 1] * (2.0 * h1 - h2) / h1
                + y[k] * h_sum * h_sum / (h1 * h2)
                + y[k + 1] * (2.0 * h2 - h1) / h2);

        out[k] = out[k - 1] + {
            // First sub-interval: use trapezoidal for the intermediate value
            0.5 * (y[k] + y[k - 1]) * h1
        };
        out[k + 1] = out[k - 1] + integral;
        k += 2;
    }

    // If there's a remaining interval, use trapezoidal
    if k < n {
        out[k] = out[k - 1] + 0.5 * (y[k] + y[k - 1]) * (x[k] - x[k - 1]);
    }

    out
}

/// Trapezoidal integration of `y` over `x`.
pub fn trapz(y: &[f64], x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for k in 1..y.len() {
        sum += 0.5 * (y[k] + y[k - 1]) * (x[k] - x[k - 1]);
    }
    sum
}

/// Numerical gradient with uniform spacing using 5-point stencil (O(h⁴)).
///
/// Interior points use the 5-point central difference:
///   `g[i] = (-y[i+2] + 8*y[i+1] - 8*y[i-1] + y[i-2]) / (12*h)`
///
/// Near-boundary points use appropriate forward/backward formulas.
pub fn gradient_uniform(y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    let mut g = vec![0.0; n];
    if n < 2 {
        return g;
    }
    if n == 2 {
        g[0] = (y[1] - y[0]) / h;
        g[1] = (y[1] - y[0]) / h;
        return g;
    }
    if n == 3 {
        g[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h);
        g[1] = (y[2] - y[0]) / (2.0 * h);
        g[2] = (y[0] - 4.0 * y[1] + 3.0 * y[2]) / (2.0 * h);
        return g;
    }
    if n == 4 {
        g[0] = (-3.0 * y[0] + 4.0 * y[1] - y[2]) / (2.0 * h);
        g[1] = (y[2] - y[0]) / (2.0 * h);
        g[2] = (y[3] - y[1]) / (2.0 * h);
        g[3] = (y[1] - 4.0 * y[2] + 3.0 * y[3]) / (2.0 * h);
        return g;
    }

    // n >= 5: use 5-point stencil for interior, 4-point formulas at boundaries
    // Left boundary: O(h³) forward formula
    g[0] = (-25.0 * y[0] + 48.0 * y[1] - 36.0 * y[2] + 16.0 * y[3] - 3.0 * y[4]) / (12.0 * h);
    g[1] = (-3.0 * y[0] - 10.0 * y[1] + 18.0 * y[2] - 6.0 * y[3] + y[4]) / (12.0 * h);

    // Interior: 5-point central difference O(h⁴)
    for i in 2..n - 2 {
        g[i] = (-y[i + 2] + 8.0 * y[i + 1] - 8.0 * y[i - 1] + y[i - 2]) / (12.0 * h);
    }

    // Right boundary: O(h³) backward formula
    g[n - 2] = (-y[n - 5] + 6.0 * y[n - 4] - 18.0 * y[n - 3] + 10.0 * y[n - 2] + 3.0 * y[n - 1])
        / (12.0 * h);
    g[n - 1] = (3.0 * y[n - 5] - 16.0 * y[n - 4] + 36.0 * y[n - 3] - 48.0 * y[n - 2]
        + 25.0 * y[n - 1])
        / (12.0 * h);
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simpsons_weights_uniform() {
        let argvals = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let weights = simpsons_weights(&argvals);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < NUMERICAL_EPS);
    }

    #[test]
    fn test_simpsons_weights_2d() {
        let argvals_s = vec![0.0, 0.5, 1.0];
        let argvals_t = vec![0.0, 0.5, 1.0];
        let weights = simpsons_weights_2d(&argvals_s, &argvals_t);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < NUMERICAL_EPS);
    }

    #[test]
    fn test_extract_curves() {
        // Column-major data: 2 observations, 3 points
        // obs 0: [1, 2, 3], obs 1: [4, 5, 6]
        let data = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        let mat = crate::matrix::FdMatrix::from_column_major(data, 2, 3).unwrap();
        let curves = extract_curves(&mat);
        assert_eq!(curves.len(), 2);
        assert_eq!(curves[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(curves[1], vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_l2_distance_identical() {
        let curve = vec![1.0, 2.0, 3.0];
        let weights = vec![0.25, 0.5, 0.25];
        let dist = l2_distance(&curve, &curve, &weights);
        assert!(dist.abs() < NUMERICAL_EPS);
    }

    #[test]
    fn test_l2_distance_different() {
        let curve1 = vec![0.0, 0.0, 0.0];
        let curve2 = vec![1.0, 1.0, 1.0];
        let weights = vec![0.25, 0.5, 0.25]; // sum = 1
        let dist = l2_distance(&curve1, &curve2, &weights);
        // dist^2 = 0.25*1 + 0.5*1 + 0.25*1 = 1.0, so dist = 1.0
        assert!((dist - 1.0).abs() < NUMERICAL_EPS);
    }

    #[test]
    fn test_n1_weights() {
        // Single point: fallback weight is 1.0 (degenerate case)
        let w = simpsons_weights(&[0.5]);
        assert_eq!(w.len(), 1);
        assert!((w[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_n2_weights() {
        let w = simpsons_weights(&[0.0, 1.0]);
        assert_eq!(w.len(), 2);
        // Trapezoidal: each weight should be 0.5
        assert!((w[0] - 0.5).abs() < 1e-12);
        assert!((w[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_mismatched_l2_distance() {
        // Mismatched lengths should not panic but may give garbage
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let w = vec![0.5, 0.5, 0.5];
        let d = l2_distance(&a, &b, &w);
        assert!(d.abs() < 1e-12, "Same vectors should have zero distance");
    }

    // ── trapz ──

    #[test]
    fn test_trapz_sine() {
        // ∫₀^π sin(x) dx = 2
        let m = 1000;
        let x: Vec<f64> = (0..m)
            .map(|i| std::f64::consts::PI * i as f64 / (m - 1) as f64)
            .collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();
        let result = trapz(&y, &x);
        assert!(
            (result - 2.0).abs() < 1e-4,
            "∫ sin(x) dx over [0,π] should be ~2, got {result}"
        );
    }

    // ── cumulative_trapz ──

    #[test]
    fn test_cumulative_trapz_matches_final() {
        let m = 100;
        let x: Vec<f64> = (0..m).map(|i| i as f64 / (m - 1) as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi).collect();
        let cum = cumulative_trapz(&y, &x);
        let total = trapz(&y, &x);
        assert!(
            (cum[m - 1] - total).abs() < 1e-12,
            "Final cumulative value should match trapz"
        );
    }

    // ── linear_interp ──

    #[test]
    fn test_linear_interp_boundary_clamp() {
        let x = vec![0.0, 0.5, 1.0];
        let y = vec![10.0, 20.0, 30.0];
        assert!((linear_interp(&x, &y, -1.0) - 10.0).abs() < 1e-12);
        assert!((linear_interp(&x, &y, 2.0) - 30.0).abs() < 1e-12);
        assert!((linear_interp(&x, &y, 0.25) - 15.0).abs() < 1e-12);
    }

    // ── gradient_uniform ──

    #[test]
    fn test_gradient_uniform_linear() {
        // f(x) = 3x → f'(x) = 3 everywhere
        let m = 50;
        let h = 1.0 / (m - 1) as f64;
        let y: Vec<f64> = (0..m).map(|i| 3.0 * i as f64 * h).collect();
        let g = gradient_uniform(&y, h);
        for i in 0..m {
            assert!(
                (g[i] - 3.0).abs() < 1e-10,
                "gradient of 3x should be 3 at i={i}, got {}",
                g[i]
            );
        }
    }
}
