//! DIF-04 SC #3: the full differentiable surface must be reachable from BOTH
//! the crate root (`fdars_core::<symbol>`) and the prelude
//! (`fdars_core::prelude::*`). This external-crate integration test compiles
//! only if every symbol is publicly re-exported on both paths; each symbol is
//! touched at least once so none is dead code.
#![cfg(all(feature = "linalg", feature = "parallel"))]

// Crate-root path: every differentiable symbol named explicitly.
use fdars_core::{
    amplitude_distance_at_warp_generic, diff, directional_derivative, grad, jacobian,
    project_scores_generic, soft_dtw_distance_generic, Dual, FdMatrix, Scalar,
};

#[test]
fn reexports_reachable_via_crate_root() {
    // grad + jacobian + directional_derivative + diff.
    let (v, g) = grad(|x| x[0] * x[0], &[1.0]);
    assert_eq!(g.len(), 1);
    assert!((v - 1.0).abs() < 1e-12);

    let (jv, j) = jacobian(|x| vec![x[0] + x[1]], &[1.0, 2.0]);
    assert_eq!(jv.len(), 1);
    assert_eq!(j[0].len(), 2);

    let (_dv, dd) = directional_derivative(|x| x[0] * x[1], &[2.0, 3.0], &[1.0, 0.0]);
    assert!((dd - 3.0).abs() < 1e-12);

    let (dfv, dfd) = diff(|x: Dual| x * x, 3.0);
    assert!((dfv - 9.0).abs() < 1e-12 && (dfd - 6.0).abs() < 1e-12);

    // Scalar + Dual constructors.
    let d: Dual = Dual::seed(1.0);
    let _s: f64 = <f64 as Scalar>::one();
    assert_eq!(d.value, 1.0);

    // Generic ops: touch each with a live call (proves reachability + usage).
    let curve = [Dual::seed(0.1), Dual::constant(0.2)];
    let refr = [Dual::constant(0.15), Dual::constant(0.25)];
    let out = soft_dtw_distance_generic(&curve, &refr, 0.1);
    assert!(out.value.is_finite());

    let rotation = FdMatrix::from_column_major(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
    let scores = project_scores_generic(&curve, &[0.0, 0.0], &rotation, &[1.0, 1.0], 2);
    assert_eq!(scores.len(), 2);

    let argvals = [0.1_f64, 0.9];
    let q1_ref = [0.3_f64, 0.4];
    let warping = argvals;
    let weights = [0.5_f64, 0.5];
    let amp = amplitude_distance_at_warp_generic(&q1_ref, &curve, &warping, &argvals, &weights);
    assert!(amp.value.is_finite());
}

/// Prelude glob path: prove the same surface is reachable via `prelude::*`.
mod via_prelude {
    use fdars_core::prelude::*;

    #[test]
    fn reexports_reachable_via_prelude() {
        // grad / diff / Dual / Scalar reachable through the glob import.
        let (v, g) = grad(|x| x[0] * x[0] + x[1] * x[1], &[1.0, 2.0]);
        assert_eq!(g.len(), 2);
        assert!((v - 5.0).abs() < 1e-12);

        let (dv, dd) = diff(|x: Dual| x * x, 2.0);
        assert!((dv - 4.0).abs() < 1e-12 && (dd - 4.0).abs() < 1e-12);

        let _one: f64 = <f64 as Scalar>::one();
        let d = Dual::constant(3.0);
        assert_eq!(d.tangent, 0.0);

        // Generic ops reachable via the prelude glob too: touch each with a call.
        let curve = [Dual::seed(0.1), Dual::constant(0.2)];
        let refr = [Dual::constant(0.15), Dual::constant(0.25)];
        assert!(soft_dtw_distance_generic(&curve, &refr, 0.1)
            .value
            .is_finite());

        let rotation = FdMatrix::from_column_major(vec![1.0, 0.0, 0.0, 1.0], 2, 2).unwrap();
        let scores = project_scores_generic(&curve, &[0.0, 0.0], &rotation, &[1.0, 1.0], 2);
        assert_eq!(scores.len(), 2);

        let argvals = [0.1_f64, 0.9];
        let amp = amplitude_distance_at_warp_generic(
            &[0.3, 0.4],
            &curve,
            &argvals,
            &argvals,
            &[0.5, 0.5],
        );
        assert!(amp.value.is_finite());

        // jacobian reachable through the prelude glob.
        let (jv, _j) = jacobian(|x| vec![x[0] + x[1]], &[1.0, 2.0]);
        assert_eq!(jv.len(), 1);

        let (_val, dd2) = directional_derivative(|x| x[0] + x[1], &[1.0, 2.0], &[1.0, 1.0]);
        assert!((dd2 - 2.0).abs() < 1e-12);
    }
}
