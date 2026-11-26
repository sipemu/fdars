//! Move-split-merge distance (Stefan et al. 2013), aeon `msm_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    check_param, elastic_family_fns, last_cell, ElasticDistancePath, Lattice, Recipe, Traceback,
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

/// aeon `_cost_dependent` for one channel (squared distances).
#[inline]
fn cost_dependent(x: f64, y: f64, z: f64, c: f64) -> f64 {
    let sq = |a: f64, b: f64| (a - b) * (a - b);
    let diameter = sq(y, z);
    let mid = (y + z) / 2.0;
    if sq(mid, x) <= diameter / 4.0 {
        c
    } else {
        let to_prev = sq(y, x);
        let to_c = sq(z, x);
        if to_prev < to_c {
            c + to_prev
        } else {
            c + to_c
        }
    }
}

/// aeon `_independent_cost_matrix` / `_msm_dependent_cost_matrix` (univariate).
fn msm_lattice(
    x: &[f64],
    y: &[f64],
    bounds: &[(usize, usize)],
    c: f64,
    independent: bool,
) -> Lattice {
    let (dist, cost): (fn(f64, f64) -> f64, fn(f64, f64, f64, f64) -> f64) = if independent {
        (|a, b| (a - b).abs(), cost_independent)
    } else {
        (|a, b| (a - b) * (a - b), cost_dependent)
    };
    let mut lat = Lattice::new(x.len(), y.len(), f64::INFINITY);
    let origin = dist(x[0], y[0]);
    lat.set(1, 1, origin);
    lat.fill(bounds, |a, b, diag, up, left| {
        if a == 0 && b == 0 {
            return origin;
        }
        let d1 = diag + dist(x[a], y[b]);
        let d2 = if a > 0 {
            up + cost(x[a], x[a - 1], y[b], c)
        } else {
            f64::INFINITY
        };
        let d3 = if b > 0 {
            left + cost(y[b], x[a], y[b - 1], c)
        } else {
            f64::INFINITY
        };
        d1.min(d2).min(d3)
    });
    lat
}

fn recipe(
    c: f64,
    independent: bool,
) -> Result<Recipe<impl Fn(&[f64], &[f64], &[(usize, usize)]) -> Lattice + Sync>, FdarError> {
    check_param("c", c, true)?;
    Ok(Recipe {
        name: "msm",
        derivative: false,
        padded_output: false,
        lattice: move |x: &[f64], y: &[f64], b: &[(usize, usize)]| {
            msm_lattice(x, y, b, c, independent)
        },
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
    params: (c: f64, independent: bool),
    params_doc: "A move costs `|x_i − y_j|`; a split or merge costs `c` (aeon default \
        `1.0`, finite and `≥ 0`) plus, when the new value is not between its neighbours, \
        the distance to the nearer one. `independent = true` (aeon default) is the \
        classical MSM; `independent = false` is aeon's dependent variant, which for a \
        univariate series uses squared distances throughout.",
    example: ", 1.0, true",
    recipe: recipe(c, independent),
}
