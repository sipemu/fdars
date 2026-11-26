//! Amerced DTW (Herrmann & Webb 2023), aeon `adtw_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    check_param, elastic_family_fns, last_cell, sqe, ElasticDistancePath, Lattice, Recipe, Ser,
    Traceback,
};
use crate::metric::window::DtwWindow;

/// aeon `_adtw_cost_matrix`: squared cost; vertical / horizontal steps pay
/// `warp_penalty`.
fn adtw_lattice(x: Ser, y: Ser, bounds: &[(usize, usize)], penalty: f64) -> Lattice {
    let mut lat = Lattice::new(x.len(), y.len(), f64::INFINITY);
    lat.set(0, 0, 0.0);
    lat.fill(bounds, |a, b, diag, up, left| {
        sqe(x.at(a), y.at(b)) + (up + penalty).min(left + penalty).min(diag)
    });
    lat
}

fn recipe(
    warp_penalty: f64,
) -> Result<Recipe<impl Fn(Ser, Ser, &[(usize, usize)]) -> Lattice + Sync>, FdarError> {
    check_param("warp_penalty", warp_penalty, true)?;
    Ok(Recipe {
        name: "adtw",
        derivative: false,
        padded_output: false,
        lattice: move |x: Ser, y: Ser, b: &[(usize, usize)]| adtw_lattice(x, y, b, warp_penalty),
        score: last_cell,
        path_score: last_cell,
        traceback: Traceback::MinReturn,
    })
}

elastic_family_fns! {
    label: "Amerced DTW (ADTW)",
    distance: adtw_distance,
    pairwise: adtw_pairwise_distance,
    cost: adtw_cost_matrix,
    path: adtw_alignment_path,
    distance_mv: adtw_distance_multivariate,
    pairwise_mv: adtw_pairwise_distance_multivariate,
    cost_mv: adtw_cost_matrix_multivariate,
    path_mv: adtw_alignment_path_multivariate,
    params: (warp_penalty: f64),
    params_doc: "DTW with squared local cost where every non-diagonal (warping) step \
        adds the constant `warp_penalty` (aeon default `1.0`, finite and `≥ 0`). \
        `warp_penalty = 0` is plain DTW. No root is taken.",
    example: ", 1.0",
    recipe: recipe(warp_penalty),
}
