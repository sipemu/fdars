//! Phase 99 (GEN-02) — end-to-end autodiff composition demo.
//!
//! Composes the broadened differentiable subset built across v0.44.0 into a
//! single `Scalar`-generic scalar objective and takes its gradient w.r.t. the
//! input curve, validated against central finite differences at BOTH forward-mode
//! `Dual` and reverse-mode `Var` (via `vjp`). This proves the autodiff types flow
//! end-to-end through the generalized hot-paths:
//!
//!   objective(curve) = predict_curve_generic(curve, fit)   // Phase 97 (FPCR prediction)
//!                    + inner_product(curve, ref, argvals)   // Phase 95 (functional inner product)
//!                    + modal_depth_generic(curve, refmat, h) // Phase 98 (differentiable depth)

use fdars_core::autodiff::{vjp, Dual, Scalar, Var};
use fdars_core::depth::modal_depth_generic;
use fdars_core::matrix::FdMatrix;
use fdars_core::scalar_on_function::{fregre_lm, predict_curve_generic, FregreLmResult};
use fdars_core::utility::inner_product;

/// The composed objective, generic over the scalar type `T`.
fn objective<T: Scalar>(
    curve: &[T],
    fit: &FregreLmResult,
    refmat: &FdMatrix,
    ref_curve: &[f64],
    argvals: &[f64],
    h: f64,
) -> T {
    let pred = predict_curve_generic::<T>(curve, fit);
    let ref_lifted: Vec<T> = ref_curve.iter().map(|&v| T::from_f64(v)).collect();
    let ip = inner_product::<T>(curve, &ref_lifted, argvals);
    let depth = modal_depth_generic::<T>(curve, refmat, h);
    pred + ip + depth
}

/// Build a small, well-conditioned FPCR fit (amplitude-scaled sinusoids, ncomp=1).
fn fixture() -> (FdMatrix, FregreLmResult, Vec<f64>, Vec<f64>, Vec<f64>, f64) {
    let (n, m) = (14usize, 9usize);
    let mut flat = vec![0.0; n * m];
    for i in 0..n {
        let amp = 1.0 + i as f64 * 0.15;
        for j in 0..m {
            let t = j as f64 / (m - 1) as f64;
            flat[i + j * n] = amp * (std::f64::consts::PI * t).sin();
        }
    }
    let data = FdMatrix::from_column_major(flat, n, m).unwrap();
    let y: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.15).collect();
    let fit = fregre_lm(&data, &y, None, 1).unwrap();

    let argvals: Vec<f64> = (0..m).map(|j| j as f64 / (m - 1) as f64).collect();
    let ref_curve: Vec<f64> = (0..m).map(|j| data[(1, j)]).collect();
    // Off-reference query so modal depth's L2 sqrt is differentiable (all dist_j > 0).
    let query: Vec<f64> = (0..m)
        .map(|j| data[(0, j)] + 0.07 + 0.01 * j as f64)
        .collect();
    (data, fit, argvals, ref_curve, query, 0.6)
}

#[test]
fn composed_objective_gradient_matches_finite_diff() {
    let (refmat, fit, argvals, ref_curve, query, h) = fixture();
    let m = query.len();

    let f64_obj =
        |c: &[f64]| -> f64 { objective::<f64>(c, &fit, &refmat, &ref_curve, &argvals, h) };

    // Reverse-mode: one vjp sweep gives all input gradients.
    let (rev_val, rev_grad) = vjp(
        |c: &[Var]| objective::<Var>(c, &fit, &refmat, &ref_curve, &argvals, h),
        &query,
    );
    // f64 value parity with the reverse-mode primal.
    assert!(
        (rev_val - f64_obj(&query)).abs() < 1e-9,
        "reverse-mode value {rev_val} vs f64 {}",
        f64_obj(&query)
    );

    let hfd = 1e-6_f64;
    for idx in 0..m {
        // Forward-mode tangent (seed coordinate idx).
        let cd: Vec<Dual> = query
            .iter()
            .enumerate()
            .map(|(j, &v)| {
                if j == idx {
                    Dual::seed(v)
                } else {
                    Dual::constant(v)
                }
            })
            .collect();
        let (_, fwd_tangent) =
            objective::<Dual>(&cd, &fit, &refmat, &ref_curve, &argvals, h).extract();

        // Central finite difference.
        let mut cp = query.clone();
        let mut cm = query.clone();
        cp[idx] += hfd;
        cm[idx] -= hfd;
        let fd = (f64_obj(&cp) - f64_obj(&cm)) / (2.0 * hfd);

        let tol = 1e-6 * (1.0 + fd.abs());
        assert!(
            (fwd_tangent - fd).abs() <= tol,
            "Dual: coord {idx} tangent={fwd_tangent} vs FD={fd}"
        );
        assert!(
            (rev_grad[idx] - fd).abs() <= tol,
            "Var: coord {idx} grad={} vs FD={fd}",
            rev_grad[idx]
        );
    }
}
