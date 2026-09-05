//! Differentiable (generic-over-[`Scalar`]) elastic-distance path.
//!
//! This module holds the forward-mode-differentiable amplitude-distance
//! operation. Only the **fixed-warping** formulation is differentiable: given a
//! pre-computed warping `gamma: &[f64]` (a constant), the amplitude distance is
//! a smooth function of the second curve's sample values, so instantiating at
//! [`Dual`](crate::autodiff::Dual) yields exact gradients w.r.t. those values.
//!
//! The warp-**searched** [`elastic_distance`](crate::alignment::elastic_distance)
//! is intentionally **not** genericized: it routes through a discrete
//! dynamic-programming argmin over path cells. The optimal-path indices are
//! integers (piecewise-constant in the inputs), so forward-mode `Dual` would
//! propagate a *zero* gradient through the warping — misleading, not useful.
//! Differentiating through a warp search requires a continuous relaxation
//! (e.g. soft-DTW) or a fixed warp, which is what this module provides.
//!
//! # SRSF non-smoothness
//!
//! The SRSF q-transform `q = sign(f') * sqrt(|f'|)` has a square-root
//! singularity at `f' = 0`: [`Dual::sqrt`](crate::autodiff::Dual) yields a
//! non-finite tangent there. Callers must avoid sample points where the curve
//! derivative is zero (the same domain constraint as the existing f64 SRSF
//! code, where the value is likewise `0` with an ill-defined slope).

use crate::autodiff::Scalar;

/// Generic piecewise-linear interpolation of `curve` (values `S`) at `t`.
///
/// `argvals` and `t` are `f64` (the grid is a constant); the interpolated value
/// is a *linear* combination of two `S` curve values, so gradients flow through
/// the curve values. Values outside `[argvals[0], argvals[m-1]]` clamp to the
/// nearest endpoint.
fn generic_linear_interp<S: Scalar>(argvals: &[f64], curve: &[S], t: f64) -> S {
    let m = argvals.len();
    if m == 0 {
        return S::zero();
    }
    if t <= argvals[0] {
        return curve[0];
    }
    if t >= argvals[m - 1] {
        return curve[m - 1];
    }
    // Bracket: argvals[j] <= t < argvals[j+1].
    let j = argvals
        .partition_point(|&a| a <= t)
        .saturating_sub(1)
        .min(m - 2);
    let dt = argvals[j + 1] - argvals[j];
    if dt <= 0.0 {
        return curve[j];
    }
    let alpha = S::from_f64((t - argvals[j]) / dt);
    curve[j] * (S::one() - alpha) + curve[j + 1] * alpha
}

/// Generic SRSF via central differences: `q[j] = sign(f'[j]) * sqrt(|f'[j]|)`.
///
/// Uses uniform step `h = (argvals[m-1] - argvals[0]) / (m-1)` with central
/// differences in the interior and one-sided differences at the boundaries.
/// Non-smooth at `f' = 0` — [`Dual`](crate::autodiff::Dual)'s `abs`/`signum`
/// subdifferential convention handles it at runtime, but callers should avoid
/// zero-derivative sample points (see the module docs).
fn generic_srsf_central_diff<S: Scalar>(curve: &[S], argvals: &[f64]) -> Vec<S> {
    let m = curve.len();
    let h = if m > 1 {
        (argvals[m - 1] - argvals[0]) / (m - 1) as f64
    } else {
        1.0
    };
    let inv_h = S::from_f64(1.0 / h);
    let inv_2h = S::from_f64(1.0 / (2.0 * h));
    let mut q = vec![S::zero(); m];
    for (j, qj) in q.iter_mut().enumerate() {
        let deriv = if m == 1 {
            S::zero()
        } else if j == 0 {
            (curve[1] - curve[0]) * inv_h
        } else if j == m - 1 {
            (curve[m - 1] - curve[m - 2]) * inv_h
        } else {
            (curve[j + 1] - curve[j - 1]) * inv_2h
        };
        *qj = S::signum(deriv) * S::sqrt(S::abs(deriv));
    }
    q
}

/// Generic weighted L2 distance in SRSF space: `sqrt(sum_j (q1[j]-q2[j])^2 w[j])`.
///
/// `q1` (reference SRSF) and `weights` are `f64` constants; `q2` is generic, so
/// gradients flow through `q2`.
fn generic_l2_srsf_distance<S: Scalar>(q1: &[f64], q2: &[S], weights: &[f64]) -> S {
    let mut dist_sq = S::zero();
    for j in 0..q1.len() {
        let diff = S::from_f64(q1[j]) - q2[j];
        dist_sq += diff * diff * S::from_f64(weights[j]);
    }
    S::sqrt(dist_sq)
}

/// Differentiable SRSF amplitude distance at a **fixed** warping.
///
/// Computes the weighted-L2-in-SRSF-space distance between a pre-computed
/// reference SRSF `q1_ref` and the SRSF of `curve2` reparameterized by the fixed
/// `warping`. Only `curve2` is generic: gradients of the distance w.r.t.
/// `curve2[j]` flow through when `S = Dual`. `q1_ref`, `warping`, `argvals`, and
/// `weights` are `f64` constants.
///
/// The pipeline is: (1) reparameterize `curve2` by `warping` via generic linear
/// interpolation, (2) take the generic central-difference SRSF of the aligned
/// curve, (3) weighted-L2 distance against `q1_ref`.
///
/// Note the warp-searched [`elastic_distance`](crate::alignment::elastic_distance)
/// is *not* differentiable (discrete DP argmin) and is intentionally not
/// genericized (see the module docs). Callers must avoid sample points where the
/// aligned-curve derivative is zero (SRSF `sqrt(|f'|)` singularity).
///
/// # Arguments
/// * `q1_ref` - SRSF of the reference curve (pre-computed, `f64`)
/// * `curve2` - second curve sample values (`&[S]`, gradient flows here)
/// * `warping` - optimal warp `gamma` (fixed, pre-computed, `f64`)
/// * `argvals` - evaluation grid (`f64`)
/// * `weights` - integration weights (`f64`)
pub fn amplitude_distance_at_warp_generic<S: Scalar>(
    q1_ref: &[f64],
    curve2: &[S],
    warping: &[f64],
    argvals: &[f64],
    weights: &[f64],
) -> S {
    let m = argvals.len();
    let f2_aligned: Vec<S> = (0..m)
        .map(|j| generic_linear_interp(argvals, curve2, warping[j]))
        .collect();
    let q2_aligned = generic_srsf_central_diff(&f2_aligned, argvals);
    generic_l2_srsf_distance(q1_ref, &q2_aligned, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::Dual;
    use crate::helpers::simpsons_weights;
    use std::f64::consts::PI;

    // Grid on [0.1, 0.9] deliberately avoids derivative zeros of sin/cos(2*pi*t)
    // (which occur at t = 0.25, 0.75) so the SRSF sqrt(|f'|) stays smooth.
    fn setup() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = 20;
        let argvals: Vec<f64> = (0..n)
            .map(|i| 0.1 + 0.8 * i as f64 / (n - 1) as f64)
            .collect();
        // Reference curve f1 = sin(2*pi*t); its f64 central-diff SRSF is q1_ref.
        let f1: Vec<f64> = argvals.iter().map(|&t| (2.0 * PI * t).sin()).collect();
        let q1_ref = srsf_central_diff_f64(&f1, &argvals);
        // Second curve f2 = cos(2*pi*t) + 0.1 (derivative nonzero on the grid).
        let curve2: Vec<f64> = argvals
            .iter()
            .map(|&t| (2.0 * PI * t).cos() + 0.1)
            .collect();
        // Identity warping.
        let warping = argvals.clone();
        let weights = simpsons_weights(&argvals);
        (argvals, q1_ref, curve2, warping, weights)
    }

    // f64 reference central-diff SRSF (mirrors generic_srsf_central_diff at f64).
    fn srsf_central_diff_f64(curve: &[f64], argvals: &[f64]) -> Vec<f64> {
        generic_srsf_central_diff(curve, argvals)
    }

    #[test]
    fn amplitude_f64_parity() {
        // SC #7: generic path at f64 == hand-composed f64 reference
        // (interp -> central-diff SRSF -> weighted L2).
        let (argvals, q1_ref, curve2, warping, weights) = setup();
        let generic = amplitude_distance_at_warp_generic::<f64>(
            &q1_ref, &curve2, &warping, &argvals, &weights,
        );
        // Hand-composed reference.
        let m = argvals.len();
        let f2_aligned: Vec<f64> = (0..m)
            .map(|j| generic_linear_interp(&argvals, &curve2, warping[j]))
            .collect();
        let q2 = srsf_central_diff_f64(&f2_aligned, &argvals);
        let mut dist_sq = 0.0;
        for j in 0..m {
            let d = q1_ref[j] - q2[j];
            dist_sq += d * d * weights[j];
        }
        let reference = dist_sq.sqrt();
        assert!(
            (generic - reference).abs() <= 1e-10,
            "generic {generic} vs reference {reference}"
        );
    }

    #[test]
    fn amplitude_gradient_vs_fd() {
        // SC #4: Dual gradient w.r.t. curve2[j] vs central FD, tol <=1e-5.
        let (argvals, q1_ref, curve2, warping, weights) = setup();
        let m = curve2.len();
        let h = 1e-8;
        for j in 0..m {
            // Dual gradient: seed curve2[j].
            let c2_dual: Vec<Dual> = curve2
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if i == j {
                        Dual::seed(v)
                    } else {
                        Dual::constant(v)
                    }
                })
                .collect();
            let dual =
                amplitude_distance_at_warp_generic(&q1_ref, &c2_dual, &warping, &argvals, &weights)
                    .extract()
                    .1;
            // Central FD at f64.
            let mut cp = curve2.clone();
            let mut cm = curve2.clone();
            cp[j] += h;
            cm[j] -= h;
            let fp = amplitude_distance_at_warp_generic::<f64>(
                &q1_ref, &cp, &warping, &argvals, &weights,
            );
            let fm = amplitude_distance_at_warp_generic::<f64>(
                &q1_ref, &cm, &warping, &argvals, &weights,
            );
            let fd = (fp - fm) / (2.0 * h);
            assert!((dual - fd).abs() <= 1e-5, "j={j}: dual {dual} vs fd {fd}");
        }
    }
}
