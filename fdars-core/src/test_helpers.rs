//! Shared test helper functions.
//!
//! This module is only compiled during testing.

/// Generate a uniform grid on \[0, 1\] with `n` points.
pub fn uniform_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

/// Compute the Adjusted Rand Index (ARI) between two label vectors.
///
/// ARI measures the agreement between two clusterings, adjusted for
/// chance. ARI = 1.0 means perfect agreement (up to label permutation);
/// ARI near 0 means no better than chance agreement.
///
/// Uses the Hubert & Arabie (1985) formula:
/// `ARI = (S - E) / (0.5 * (A + B) - E)`
/// where S = Σ C(n_ij, 2), A = Σ C(a_i, 2), B = Σ C(b_j, 2),
/// E = A * B / C(n, 2).
///
/// # Panics
///
/// Panics if `a.len() != b.len()`.
pub fn adjusted_rand_index(a: &[usize], b: &[usize]) -> f64 {
    assert_eq!(a.len(), b.len(), "adjusted_rand_index: label vectors must have equal length");
    let n = a.len();
    if n == 0 {
        return 1.0;
    }

    // Relabel both vectors to dense indices 0..k_a and 0..k_b
    fn dense_labels(labels: &[usize]) -> (Vec<usize>, usize) {
        let mut map = std::collections::HashMap::new();
        let mut next_id = 0usize;
        let mut out = Vec::with_capacity(labels.len());
        for &l in labels {
            let id = map.entry(l).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            out.push(*id);
        }
        (out, next_id)
    }

    let (a_dense, k_a) = dense_labels(a);
    let (b_dense, k_b) = dense_labels(b);

    // Build contingency table n_ij (k_a x k_b)
    let mut table = vec![0u64; k_a * k_b];
    for i in 0..n {
        table[a_dense[i] * k_b + b_dense[i]] += 1;
    }

    // choose2(x) = x*(x-1)/2
    let choose2 = |x: u64| x.saturating_sub(1) * x / 2;

    // S = Σ_ij C(n_ij, 2)
    let s: u64 = table.iter().map(|&v| choose2(v)).sum();

    // Row sums a_i, col sums b_j
    let row_sums: Vec<u64> = (0..k_a)
        .map(|i| (0..k_b).map(|j| table[i * k_b + j]).sum())
        .collect();
    let col_sums: Vec<u64> = (0..k_b)
        .map(|j| (0..k_a).map(|i| table[i * k_b + j]).sum())
        .collect();

    let sum_a: u64 = row_sums.iter().map(|&v| choose2(v)).sum();
    let sum_b: u64 = col_sums.iter().map(|&v| choose2(v)).sum();
    let cn2 = choose2(n as u64);

    if cn2 == 0 {
        // All points in one cluster — degenerate case
        return 1.0;
    }

    // Expected: E = sum_a * sum_b / C(n,2)  (in f64 to avoid overflow)
    let e = (sum_a as f64) * (sum_b as f64) / (cn2 as f64);
    let denom = 0.5 * (sum_a as f64 + sum_b as f64) - e;

    if denom.abs() < 1e-15 {
        // Degenerate case: all points in a single cluster in both
        return 1.0;
    }

    ((s as f64) - e) / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ari_identical() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let ari = adjusted_rand_index(&a, &a);
        assert!((ari - 1.0).abs() < 1e-9, "identical labels: ARI = {ari}");
    }

    #[test]
    fn test_ari_permutation() {
        // b is a relabeling of a: cluster 0 → 1, cluster 1 → 0
        let a = vec![0, 0, 1, 1, 0, 0];
        let b = vec![1, 1, 0, 0, 1, 1];
        let ari = adjusted_rand_index(&a, &b);
        assert!((ari - 1.0).abs() < 1e-9, "permuted labels: ARI = {ari}");
    }

    #[test]
    fn test_ari_random_vs_structured() {
        // Structured grouping: first half vs second half
        let a: Vec<usize> = (0..20).map(|i| if i < 10 { 0 } else { 1 }).collect();
        // Structured grouping perpendicular: alternating
        let b: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let ari = adjusted_rand_index(&a, &b);
        assert!(ari.abs() < 0.3, "unrelated labels: ARI = {ari} should be near 0");
    }

    #[test]
    fn test_ari_three_clusters_permuted() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![2, 2, 0, 0, 1, 1];
        let ari = adjusted_rand_index(&a, &b);
        assert!((ari - 1.0).abs() < 1e-9, "3-cluster permutation: ARI = {ari}");
    }
}
