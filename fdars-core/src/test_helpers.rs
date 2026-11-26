//! Shared test helper functions.
//!
//! This module is only compiled during testing.

/// Generate a uniform grid on \[0, 1\] with `n` points.
pub fn uniform_grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64).collect()
}

/// Generate a uniform grid on \[0, `end`\] with `n` points.
///
/// Uses the form `i * end / (n - 1)` (multiply before divide).
/// See also [`uniform_grid_range_div`] for the `i / (n - 1) * end` form,
/// which differs by floating-point rounding at some grid points.
pub fn uniform_grid_range(n: usize, end: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * end / (n - 1) as f64).collect()
}

/// Generate a uniform grid on \[0, `end`\] with `n` points.
///
/// Uses the form `i / (n - 1) * end` (divide before multiply).
/// This is NOT floating-point identical to [`uniform_grid_range`] for all `n`;
/// provided separately to preserve numeric identity at detrend test call sites.
pub fn uniform_grid_range_div(n: usize, end: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 / (n - 1) as f64 * end).collect()
}

/// Generate a uniform grid with `n` points spaced by `step` starting at 0.
///
/// Uses the form `i * step`, producing `[0.0, step, 2*step, ..., (n-1)*step]`.
/// This differs from [`uniform_grid_range`] (which uses `i * end / (n-1)`);
/// use this form at call sites that compute `i as f64 * step` directly.
pub fn uniform_step_grid(n: usize, step: f64) -> Vec<f64> {
    (0..n).map(|i| i as f64 * step).collect()
}

/// Binarize a continuous response at a threshold: values `>= threshold` → 1.0, else 0.0.
///
/// Used in tests to convert a continuous `y` into a binary response at its median.
pub fn binarize_at_median(y: &[f64], threshold: f64) -> Vec<f64> {
    y.iter()
        .map(|&v| if v >= threshold { 1.0 } else { 0.0 })
        .collect()
}

/// Build a flat n×m [`crate::matrix::FdMatrix`] with values `(idx * scale).sin()` (column-major).
///
/// Reproduces the repeated test idiom:
/// `FdMatrix::from_column_major((0..(n*m)).map(|i| (i as f64 * scale).sin()).collect(), n, m)`.
pub fn sin_matrix(n: usize, m: usize, scale: f64) -> crate::matrix::FdMatrix {
    crate::matrix::FdMatrix::from_column_major(
        (0..(n * m)).map(|i| (i as f64 * scale).sin()).collect(),
        n,
        m,
    )
    .expect("sin_matrix: n*m matches data length by construction")
}

/// Sine curve evaluated on a grid: `sin(2π · t[i])`.
///
/// Reproduces the repeated landmark test idiom:
/// `t.iter().map(|&ti| (2.0 * PI * ti).sin()).collect()`.
pub fn sine_curve(t: &[f64]) -> Vec<f64> {
    t.iter()
        .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
        .collect()
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
    assert_eq!(
        a.len(),
        b.len(),
        "adjusted_rand_index: label vectors must have equal length"
    );
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

/// FNV-1a fingerprint over the exact IEEE-754 bit patterns of `xs`.
///
/// Used by refactor bit-identity pins: any change in any output bit
/// (including sign of zero or NaN payload) changes the fingerprint.
pub fn bits_fingerprint(xs: &[f64]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in xs {
        for b in x.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
    }
    h
}

/// Portable reference pin: per-field value summaries compared with a tight
/// relative tolerance.
///
/// Literal bit-hash pins ([`bits_fingerprint`]) are only valid on one machine
/// when the computation calls transcendental functions (`exp`, `sin`, `atan2`,
/// …): glibc's libm picks a CPU-specific kernel (FMA / SSE2 / …) at load time,
/// and the kernels differ in the last ulp. Cross-machine regression pins
/// therefore record, for every output field, `[Σx, Σ|x|, Σx², Σ wᵢx]` with
/// position weights `wᵢ = (i + 1)/n`, and compare them at `rtol` (1e-10 in the
/// pins) relative to the field's own magnitude (`Σ|x|`, or `Σx²` for the
/// squared sum). Any change above rounding level in any field fails; exact
/// integers (indices, iteration counts) stay exact.
#[derive(Debug, Default, Clone)]
pub struct ValuePin(pub Vec<f64>);

impl ValuePin {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append the four summaries of one output field.
    pub fn push(&mut self, xs: &[f64]) {
        let n = xs.len().max(1) as f64;
        let (mut s, mut a, mut q, mut w) = (0.0, 0.0, 0.0, 0.0);
        for (i, &x) in xs.iter().enumerate() {
            s += x;
            a += x.abs();
            q += x * x;
            w += (i + 1) as f64 / n * x;
        }
        self.0.extend([s, a, q, w]);
    }

    /// Compare against a stored reference (as printed by a previous run).
    ///
    /// # Panics
    /// If the lengths differ or any summary deviates by more than `rtol` times
    /// its field's magnitude; the message prints the current values as a Rust
    /// literal so an *intentional* behaviour change can be re-pinned.
    pub fn assert_close(&self, expected: &[f64], rtol: f64, label: &str) {
        self.assert_close_by(expected, |_| rtol, label);
    }

    /// [`Self::assert_close`] with a per-field relative tolerance `rtol(field)`.
    pub fn assert_close_by(&self, expected: &[f64], rtol: impl Fn(usize) -> f64, label: &str) {
        let got = &self.0;
        let literal = format!("{got:?}");
        assert_eq!(
            got.len(),
            expected.len(),
            "{label}: pin length changed; current values {literal}"
        );
        for (k, (g, e)) in got.chunks(4).zip(expected.chunks(4)).enumerate() {
            let scale = [e[1], e[1], e[2], e[1]];
            for c in 0..4 {
                let tol = rtol(k) * scale[c].abs();
                assert!(
                    (g[c] - e[c]).abs() <= tol,
                    "{label}: field {k} summary {c}: {} vs pinned {} (tol {tol:e}); \
                     current values {literal}",
                    g[c],
                    e[c]
                );
            }
        }
    }
}

/// Fingerprint of this platform's libm: FNV-1a over the bits of `exp`, `ln`, `sin`,
/// `cos`, `tan`, `atan`, `atan2`, `asin`, `acos`, `powf` and `hypot` on a fixed set of
/// arguments.
///
/// glibc selects CPU-specific kernels for these functions that differ in the last
/// ulp. Chaotic computations (e.g. HMC trajectories, where one ulp flips an
/// accept/reject decision) cannot be pinned portably even with a tolerance, so such
/// pins are enforced only where this canary equals the recording machine's.
pub fn libm_canary() -> u64 {
    let u = lcg_values(20_000, 0x5eed_ca11);
    let mut out = Vec::with_capacity(u.len() * 11);
    for w in u.chunks(2) {
        let (a, b) = (w[0], w[1]);
        let x = 8.0 * a; // [-8, 8)
        out.extend([
            x.exp(),
            (x.abs() + 1e-3).ln(),
            (50.0 * a).sin(),
            (50.0 * b).cos(),
            (1.5 * a).tan(),
            (4.0 * a).atan(),
            a.atan2(b),
            a.asin(),
            b.acos(),
            (a.abs() + 0.5).powf(3.0 * b),
            a.hypot(b),
        ]);
    }
    bits_fingerprint(&out)
}

/// Deterministic pseudo-random values in \[-1, 1) from a 64-bit LCG.
///
/// Platform-independent (integer arithmetic + one exact division), so pins
/// built from it are portable.
pub fn lcg_values(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        })
        .collect()
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
        assert!(
            ari.abs() < 0.3,
            "unrelated labels: ARI = {ari} should be near 0"
        );
    }

    #[test]
    fn test_ari_three_clusters_permuted() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![2, 2, 0, 0, 1, 1];
        let ari = adjusted_rand_index(&a, &b);
        assert!(
            (ari - 1.0).abs() < 1e-9,
            "3-cluster permutation: ARI = {ari}"
        );
    }
}
