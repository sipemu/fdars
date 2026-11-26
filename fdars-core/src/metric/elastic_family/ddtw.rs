//! Derivative DTW (Keogh & Pazzani 2001), aeon `ddtw_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{elastic_family_fns, last_cell, ElasticDistancePath, Lattice, Recipe, Traceback};
use crate::metric::window::DtwWindow;

/// aeon `_dtw_cost_matrix`: squared local cost, padded lattice with
/// `T[0][0] = 0` and `inf` borders.
pub(super) fn dtw_lattice(x: &[f64], y: &[f64], bounds: &[(usize, usize)]) -> Lattice {
    let mut lat = Lattice::new(x.len(), y.len(), f64::INFINITY);
    lat.set(0, 0, 0.0);
    lat.fill(bounds, |a, b, diag, up, left| {
        let d = x[a] - y[b];
        d * d + up.min(left).min(diag)
    });
    lat
}

type Builder = fn(&[f64], &[f64], &[(usize, usize)]) -> Lattice;

fn recipe() -> Result<Recipe<Builder>, FdarError> {
    Ok(Recipe {
        name: "ddtw",
        derivative: true,
        padded_output: false,
        lattice: dtw_lattice,
        score: last_cell,
        path_score: last_cell,
        traceback: Traceback::MinReturn,
    })
}

elastic_family_fns! {
    label: "Derivative DTW (DDTW)",
    distance: ddtw_distance,
    pairwise: ddtw_pairwise_distance,
    cost: ddtw_cost_matrix,
    path: ddtw_alignment_path,
    params: (),
    params_doc: "DTW with squared local cost on aeon's slope derivative \
        `q'_i = ((q_i − q_{i−1}) + (q_{i+1} − q_{i−1}) / 2) / 2` (Keogh & Pazzani), \
        evaluated at interior points only: a length-`n` series becomes a length-`n − 2` \
        derivative (series need `n ≥ 3`). The window, cost matrix and path all live in \
        derivative coordinates, exactly as in aeon. No root is taken.",
    example: "",
    recipe: recipe(),
}
