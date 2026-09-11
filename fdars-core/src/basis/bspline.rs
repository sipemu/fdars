//! B-spline basis functions.

use crate::autodiff::Scalar;

/// Construct B-spline knot vector with extended boundary knots.
pub fn construct_bspline_knots(t_min: f64, t_max: f64, nknots: usize, order: usize) -> Vec<f64> {
    let dt = (t_max - t_min) / (nknots - 1) as f64;
    let mut knots = Vec::with_capacity(nknots + 2 * order);
    for i in 0..order {
        knots.push(t_min - (order - i) as f64 * dt);
    }
    for i in 0..nknots {
        knots.push(t_min + i as f64 * dt);
    }
    for i in 1..=order {
        knots.push(t_max + i as f64 * dt);
    }
    knots
}

/// Evaluate order-zero B-spline basis at a single point.
pub(super) fn evaluate_order_zero<T: Scalar>(
    t_val: T,
    knots: &[f64],
    t_max_knot_idx: usize,
) -> Vec<T> {
    let mut b0 = vec![T::zero(); knots.len() - 1];
    for j in 0..(knots.len() - 1) {
        let in_interval = if j == t_max_knot_idx - 1 {
            t_val >= T::from_f64(knots[j]) && t_val <= T::from_f64(knots[j + 1])
        } else {
            t_val >= T::from_f64(knots[j]) && t_val < T::from_f64(knots[j + 1])
        };
        if in_interval {
            b0[j] = T::one();
            break;
        }
    }
    b0
}

/// Compute one order of B-spline recurrence from the previous order.
pub(super) fn bspline_recurrence_step<T: Scalar>(
    b: &[T],
    knots: &[f64],
    t_val: T,
    k: usize,
) -> Vec<T> {
    (0..(knots.len() - k))
        .map(|j| {
            let d1 = knots[j + k - 1] - knots[j]; // f64 — stays f64
            let d2 = knots[j + k] - knots[j + 1]; // f64 — stays f64
            let left = if d1.abs() > 1e-10 {
                // division THEN multiply — preserves operation order for bit-identical f64 parity
                (t_val - T::from_f64(knots[j])) / T::from_f64(d1) * b[j]
            } else {
                T::zero()
            };
            let right = if d2.abs() > 1e-10 {
                (T::from_f64(knots[j + k]) - t_val) / T::from_f64(d2) * b[j + 1]
            } else {
                T::zero()
            };
            left + right
        })
        .collect()
}

/// Evaluate B-spline basis matrix using a pre-computed knot vector.
///
/// Unlike [`bspline_basis`], this function uses the provided knot vector
/// directly, allowing evaluation on a different grid than the one used
/// to construct the knots.
pub fn bspline_basis_from_knots<T: Scalar>(t: &[T], knots: &[f64], order: usize) -> Vec<T> {
    let n = t.len();
    let nbasis = knots.len() - order;
    // Find the index of the last interior knot for boundary handling
    let t_max_knot_idx = knots.len() - order - 1;

    let mut basis = vec![T::zero(); n * nbasis];

    for (ti, &t_val) in t.iter().enumerate() {
        let mut b = evaluate_order_zero(t_val, knots, t_max_knot_idx);

        for k in 2..=order {
            b = bspline_recurrence_step(&b, knots, t_val, k);
        }

        for j in 0..nbasis {
            basis[ti + j * n] = b[j];
        }
    }

    basis
}

/// Compute B-spline basis matrix for given knots and grid points.
///
/// Creates a B-spline basis with uniformly spaced knots extended beyond the data range.
/// For order k and nknots interior knots, produces nknots + order basis functions.
///
/// # Examples
///
/// ```
/// use fdars_core::basis::bspline::bspline_basis;
///
/// let t: Vec<f64> = (0..20).map(|i| i as f64 / 19.0).collect();
/// let basis = bspline_basis(&t, 5, 4);
/// // nbasis = nknots + order = 5 + 4 = 9
/// assert_eq!(basis.len(), 20 * 9);
/// ```
pub fn bspline_basis(t: &[f64], nknots: usize, order: usize) -> Vec<f64> {
    let n = t.len();
    let nbasis = nknots + order;

    let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let knots = construct_bspline_knots(t_min, t_max, nknots, order);
    let t_max_knot_idx = order + nknots - 1;

    let mut basis = vec![0.0; n * nbasis];

    for (ti, &t_val) in t.iter().enumerate() {
        let mut b = evaluate_order_zero(t_val, &knots, t_max_knot_idx);

        for k in 2..=order {
            b = bspline_recurrence_step(&b, &knots, t_val, k);
        }

        for j in 0..nbasis {
            basis[ti + j * n] = b[j];
        }
    }

    basis
}
