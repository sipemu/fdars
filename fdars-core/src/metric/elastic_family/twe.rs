//! Time warp edit distance (Marteau 2009), aeon `twe_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    check_param, elastic_family_fns, euc, last_cell, ElasticDistancePath, Lattice, Recipe,
    Traceback,
};
use crate::metric::window::DtwWindow;

/// aeon `_twe_cost_matrix` on series padded with a leading `0` (aeon
/// `_pad_arrs`), time stamps = sample indices.
fn twe_lattice(x: &[f64], y: &[f64], bounds: &[(usize, usize)], nu: f64, lmbda: f64) -> Lattice {
    let mut lat = Lattice::new(x.len(), y.len(), f64::INFINITY);
    lat.set(0, 0, 0.0);
    let del_add = nu + lmbda;
    lat.fill(bounds, |a, b, diag, up, left| {
        let xp = if a == 0 { 0.0 } else { x[a - 1] };
        let yp = if b == 0 { 0.0 } else { y[b - 1] };
        let del_x = up + euc(xp, x[a]) + del_add;
        let del_y = left + euc(yp, y[b]) + del_add;
        let k = a.abs_diff(b);
        let matched = diag + euc(x[a], y[b]) + euc(xp, yp) + nu * ((k + k) as f64);
        del_x.min(del_y).min(matched)
    });
    lat
}

fn recipe(
    nu: f64,
    lmbda: f64,
) -> Result<Recipe<impl Fn(&[f64], &[f64], &[(usize, usize)]) -> Lattice + Sync>, FdarError> {
    check_param("nu", nu, true)?;
    check_param("lmbda", lmbda, true)?;
    Ok(Recipe {
        name: "twe",
        derivative: false,
        padded_output: false,
        lattice: move |x: &[f64], y: &[f64], b: &[(usize, usize)]| twe_lattice(x, y, b, nu, lmbda),
        score: last_cell,
        path_score: last_cell,
        traceback: Traceback::MinReturn,
    })
}

elastic_family_fns! {
    label: "Time warp edit distance (TWE)",
    distance: twe_distance,
    pairwise: twe_pairwise_distance,
    cost: twe_cost_matrix,
    path: twe_alignment_path,
    params: (nu: f64, lmbda: f64),
    params_doc: "Edit distance with stiffness `nu` (aeon default `0.001`; a match of \
        `x_i` with `y_j` pays `2·nu·|i − j|`) and deletion penalty `lmbda` (aeon default \
        `1.0`); both finite and `≥ 0`. As in aeon both series are padded with a leading \
        `0` and the sample indices serve as time stamps.",
    example: ", 0.001, 1.0",
    recipe: recipe(nu, lmbda),
}
