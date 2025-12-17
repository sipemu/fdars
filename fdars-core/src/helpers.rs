//! Helper functions for numerical integration and common operations.

/// Compute Simpson's rule integration weights for non-uniform grid.
///
/// Returns weights for trapezoidal rule integration.
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

    // For non-uniform spacing, use composite trapezoidal rule
    for i in 0..n {
        if i == 0 {
            weights[i] = (argvals[1] - argvals[0]) / 2.0;
        } else if i == n - 1 {
            weights[i] = (argvals[n - 1] - argvals[n - 2]) / 2.0;
        } else {
            weights[i] = (argvals[i + 1] - argvals[i - 1]) / 2.0;
        }
    }

    weights
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simpsons_weights_uniform() {
        let argvals = vec![0.0, 0.25, 0.5, 0.75, 1.0];
        let weights = simpsons_weights(&argvals);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simpsons_weights_2d() {
        let argvals_s = vec![0.0, 0.5, 1.0];
        let argvals_t = vec![0.0, 0.5, 1.0];
        let weights = simpsons_weights_2d(&argvals_s, &argvals_t);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
