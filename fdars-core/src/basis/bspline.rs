//! B-spline basis functions.

/// Construct B-spline knot vector with extended boundary knots.
pub(super) fn construct_bspline_knots(
    t_min: f64,
    t_max: f64,
    nknots: usize,
    order: usize,
) -> Vec<f64> {
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
pub(super) fn evaluate_order_zero(t_val: f64, knots: &[f64], t_max_knot_idx: usize) -> Vec<f64> {
    let mut b0 = vec![0.0; knots.len() - 1];
    for j in 0..(knots.len() - 1) {
        let in_interval = if j == t_max_knot_idx - 1 {
            t_val >= knots[j] && t_val <= knots[j + 1]
        } else {
            t_val >= knots[j] && t_val < knots[j + 1]
        };
        if in_interval {
            b0[j] = 1.0;
            break;
        }
    }
    b0
}

/// Compute one order of B-spline recurrence from the previous order.
pub(super) fn bspline_recurrence_step(b: &[f64], knots: &[f64], t_val: f64, k: usize) -> Vec<f64> {
    (0..(knots.len() - k))
        .map(|j| {
            let d1 = knots[j + k - 1] - knots[j];
            let d2 = knots[j + k] - knots[j + 1];
            let left = if d1.abs() > 1e-10 {
                (t_val - knots[j]) / d1 * b[j]
            } else {
                0.0
            };
            let right = if d2.abs() > 1e-10 {
                (knots[j + k] - t_val) / d2 * b[j + 1]
            } else {
                0.0
            };
            left + right
        })
        .collect()
}

/// Compute B-spline basis matrix for given knots and grid points.
///
/// Creates a B-spline basis with uniformly spaced knots extended beyond the data range.
/// For order k and nknots interior knots, produces nknots + order basis functions.
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
