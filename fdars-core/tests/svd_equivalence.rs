//! Phase-6 numerical equivalence test: nalgebra vs faer SVD agree on singular values.
//!
//! This integration test lives here (rather than inline in `audit_hotpaths.rs`) because
//! the bench binary has `harness = false` — criterion provides its own main, so
//! `#[test]` items inside the bench file are not discovered by the Rust test harness.
//! Integration tests under `fdars-core/tests/` each link into their own test binary
//! and ARE discovered by `cargo test` (same pattern as `alloc_audit_fpca.rs`).
//!
//! **Run with:**
//! ```bash
//! export TMPDIR=/home/simonm/.cache/fdars-bench-tmp
//! cargo test -p fdars-core --features linalg -- svd_equivalence --nocapture
//! ```
//!
//! AUDIT-ONLY: no modification to `fdars-core/src/`. This test exercises the same
//! inputs that the `bench_p6_svd_comparison` bench measures, confirming the two
//! libraries agree before trusting the bench numbers.

/// Build the Simpson-weighted, column-centered matrix that `fdata_to_pc_1d` feeds to SVD.
///
/// Mirrors `generate_weighted_input` in `audit_hotpaths.rs` exactly — kept in sync.
#[cfg(feature = "linalg")]
fn generate_weighted_input(n: usize, m: usize) -> fdars_core::FdMatrix {
    use fdars_core::matrix::FdMatrix;
    use std::f64::consts::PI;

    // Deterministic curves: amplitude/phase variation per curve (matches generate_curves)
    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let mut raw = vec![0.0f64; n * m];
    for i in 0..n {
        let phase = 0.2 * ((i as f64 * 3.7 + 0.5).sin());
        let amp = 1.0 + 0.3 * ((i as f64 * 5.1 + 0.3).sin());
        for j in 0..m {
            let t = argvals[j];
            raw[i + j * n] = amp * (2.0 * PI * (t + phase)).sin();
        }
    }

    // Center each column (subtract per-column mean)
    let mut centered = vec![0.0f64; n * m];
    for j in 0..m {
        let col_sum: f64 = (0..n).map(|i| raw[i + j * n]).sum();
        let col_mean = col_sum / n as f64;
        for i in 0..n {
            centered[i + j * n] = raw[i + j * n] - col_mean;
        }
    }

    // Compute Simpson weights and sqrt-scale each column
    let weights = fdars_core::simpsons_weights(&argvals);
    let sqrt_weights: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    for j in 0..m {
        for i in 0..n {
            centered[i + j * n] *= sqrt_weights[j];
        }
    }

    FdMatrix::from_column_major(centered, n, m).unwrap()
}

/// Numerical equivalence check: nalgebra and faer agree on singular values within 1e-10.
///
/// Compares singular VALUES only (sign-ambiguity-safe — singular vectors have arbitrary
/// sign per column). Both libraries return singular values in nonincreasing order.
///
/// Also verifies thin-SVD shape: U is n×m for N > M (resolving RESEARCH Open Question 1).
#[cfg(feature = "linalg")]
#[test]
fn svd_equivalence() {
    let n = 500usize;
    let m = 200usize;
    let weighted = generate_weighted_input(n, m);

    // nalgebra path — SVD::new(mat, true, true) produces thin decomposition at N > M
    let dmatrix = weighted.to_dmatrix();
    let na_svd = nalgebra::linalg::SVD::new(dmatrix, true, true);
    let na_s = na_svd.singular_values; // DVector<f64>, nonincreasing order

    // faer path — thin_svd matches nalgebra's thin decomposition
    let mat_ref = faer::MatRef::<f64>::from_column_major_slice(weighted.as_slice(), n, m);
    let fa_svd = mat_ref.thin_svd().expect("faer thin_svd failed");
    // Collect singular values via iter() — as_slice() requires ContiguousFwd stride
    // which the diagonal may not guarantee.
    let fa_s: Vec<f64> = fa_svd.S().column_vector().iter().copied().collect();

    // Verify thin-SVD shape: for N > M, U should be n×m (resolving Open Question 1)
    assert_eq!(fa_svd.U().nrows(), n, "faer U.nrows() should equal n={n}");
    assert_eq!(
        fa_svd.U().ncols(),
        m,
        "faer U.ncols() should equal m={m} (thin SVD)"
    );

    // Compare singular values element-wise with relative tolerance 1e-10.
    // Only compare numerically significant values — values below 1e-8 * s[0] are
    // floating-point noise and can differ between backends without indicating a bug.
    let k = na_s.len().min(fa_s.len());
    assert!(
        k > 0,
        "both SVDs should produce at least one singular value"
    );
    let s0 = na_s[0].max(fa_s[0]);
    let abs_threshold = 1e-8 * s0; // below this both values are noise
    let mut compared = 0usize;
    for i in 0..k {
        if na_s[i] < abs_threshold && fa_s[i] < abs_threshold {
            // Both are numerical noise — skip rather than producing a spurious failure
            continue;
        }
        compared += 1;
        let rel_err = (na_s[i] - fa_s[i]).abs() / na_s[i].max(1e-12);
        assert!(
            rel_err < 1e-10,
            "singular value {i}: nalgebra={:.15e}, faer={:.15e}, rel_err={:.3e}",
            na_s[i],
            fa_s[i],
            rel_err
        );
    }
    assert!(
        compared > 0,
        "no significant singular values found to compare (s0={s0:.3e})"
    );
}
