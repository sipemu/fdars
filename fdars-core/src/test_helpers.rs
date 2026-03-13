//! Shared test helper functions.
//!
//! This module is only compiled during testing.

/// Generate a uniform grid on \[0, 1\] with `n` points.
pub fn uniform_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}
