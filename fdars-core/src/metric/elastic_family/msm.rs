//! Move-split-merge distance (Stefan et al. 2013), aeon `msm_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    check_param, elastic_family_fns, last_cell, sqe, ElasticDistancePath, Lattice, Recipe, Ser,
    Traceback,
};
use crate::metric::window::DtwWindow;

/// aeon `_cost_independent`: split/merge cost of inserting `x` between `y`, `z`.
#[inline]
fn cost_independent(x: f64, y: f64, z: f64, c: f64) -> f64 {
    if (y <= x && x <= z) || (y >= x && x >= z) {
        c
    } else {
        c + (x - y).abs().min((x - z).abs())
    }
}

/// aeon `_cost_dependent` (squared distances over all channels): split/merge
/// cost of inserting `x` between `y` and `z`.
#[inline]
fn cost_dependent(x: &[f64], y: &[f64], z: &[f64], c: f64) -> f64 {
    let diameter = sqe(y, z);
    // `Σ_c ((y_c + z_c) / 2 − x_c)²`, i.e. aeon's squared distance of the
    // midpoint `(y + z) / 2.0` to `x`, without materialising the midpoint.
    let mut to_mid = 0.0;
    for ((&yc, &zc), &xc) in y.iter().zip(z).zip(x) {
        let t = (yc + zc) / 2.0 - xc;
        to_mid += t * t;
    }
    if to_mid <= diameter / 4.0 {
        c
    } else {
        let to_prev = sqe(y, x);
        let to_c = sqe(z, x);
        if to_prev < to_c {
            c + to_prev
        } else {
            c + to_c
        }
    }
}

/// aeon `_independent_cost_matrix` on one channel.
fn independent_lattice(x: &[f64], y: &[f64], bounds: &[(usize, usize)], c: f64) -> Lattice {
    let mut lat = Lattice::new(x.len(), y.len(), f64::INFINITY);
    let origin = (x[0] - y[0]).abs();
    lat.set(1, 1, origin);
    lat.fill(bounds, |a, b, diag, up, left| {
        if a == 0 && b == 0 {
            return origin;
        }
        let d1 = diag + (x[a] - y[b]).abs();
        let d2 = if a > 0 {
            up + cost_independent(x[a], x[a - 1], y[b], c)
        } else {
            f64::INFINITY
        };
        let d3 = if b > 0 {
            left + cost_independent(y[b], x[a], y[b - 1], c)
        } else {
            f64::INFINITY
        };
        d1.min(d2).min(d3)
    });
    lat
}

/// aeon `_msm_dependent_cost_matrix` (squared distances over all channels).
fn dependent_lattice(x: Ser, y: Ser, bounds: &[(usize, usize)], c: f64) -> Lattice {
    let mut lat = Lattice::new(x.len(), y.len(), f64::INFINITY);
    let origin = sqe(x.at(0), y.at(0));
    lat.set(1, 1, origin);
    lat.fill(bounds, |a, b, diag, up, left| {
        if a == 0 && b == 0 {
            return origin;
        }
        let (xa, yb) = (x.at(a), y.at(b));
        let d1 = diag + sqe(xa, yb);
        let d2 = if a > 0 {
            up + cost_dependent(xa, x.at(a - 1), yb, c)
        } else {
            f64::INFINITY
        };
        let d3 = if b > 0 {
            left + cost_dependent(yb, xa, y.at(b - 1), c)
        } else {
            f64::INFINITY
        };
        d1.min(d2).min(d3)
    });
    lat
}

/// aeon `_msm_independent_cost_matrix` (sum of the per-channel cost matrices,
/// in channel order) / `_msm_dependent_cost_matrix`.
fn msm_lattice(x: Ser, y: Ser, bounds: &[(usize, usize)], c: f64, independent: bool) -> Lattice {
    if !independent {
        return dependent_lattice(x, y, bounds, c);
    }
    if x.d == 1 {
        return independent_lattice(x.v, y.v, bounds, c);
    }
    let mut total = Lattice::new(x.len(), y.len(), 0.0);
    for ch in 0..x.d {
        total.add_assign(&independent_lattice(
            &x.channel(ch),
            &y.channel(ch),
            bounds,
            c,
        ));
    }
    total
}

fn recipe(
    c: f64,
    independent: bool,
) -> Result<Recipe<impl Fn(Ser, Ser, &[(usize, usize)]) -> Lattice + Sync>, FdarError> {
    check_param("c", c, true)?;
    Ok(Recipe {
        name: "msm",
        derivative: false,
        padded_output: false,
        lattice: move |x: Ser, y: Ser, b: &[(usize, usize)]| msm_lattice(x, y, b, c, independent),
        score: last_cell,
        path_score: last_cell,
        traceback: Traceback::MinReturn,
    })
}

elastic_family_fns! {
    label: "Move-split-merge (MSM)",
    distance: msm_distance,
    pairwise: msm_pairwise_distance,
    cost: msm_cost_matrix,
    path: msm_alignment_path,
    distance_mv: msm_distance_multivariate,
    pairwise_mv: msm_pairwise_distance_multivariate,
    cost_mv: msm_cost_matrix_multivariate,
    path_mv: msm_alignment_path_multivariate,
    params: (c: f64, independent: bool),
    params_doc: "A move costs `|x_i − y_j|`; a split or merge costs `c` (aeon default \
        `1.0`, finite and `≥ 0`) plus, when the new value is not between its neighbours, \
        the distance to the nearer one. `independent = true` (aeon default) is the \
        classical MSM; `independent = false` is aeon's dependent variant, which uses \
        squared distances throughout. For multivariate series `independent = true` sums \
        the per-channel MSM cost matrices and `independent = false` uses squared \
        Euclidean distances across channels (aeon).",
    example: ", 1.0, true",
    recipe: recipe(c, independent),
}
