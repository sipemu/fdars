//! Weighted DTW and weighted derivative DTW (Jeong et al. 2011), aeon
//! `wdtw_*` / `wddtw_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    check_param, elastic_family_fns, last_cell, sqe, ElasticDistancePath, Lattice, Recipe, Ser,
    Traceback,
};
use crate::metric::window::DtwWindow;

/// aeon `_wdtw_cost_matrix`: squared cost times the logistic weight
/// `w(k) = 1 / (1 + exp(−g (k − max(n, m) / 2)))` of the offset `k = |i − j|`.
fn wdtw_lattice(x: Ser, y: Ser, bounds: &[(usize, usize)], g: f64) -> Lattice {
    let max_size = x.len().max(y.len());
    let half = max_size as f64 / 2.0;
    let weights: Vec<f64> = (0..max_size)
        .map(|k| 1.0 / (1.0 + (-g * (k as f64 - half)).exp()))
        .collect();
    let mut lat = Lattice::new(x.len(), y.len(), f64::INFINITY);
    lat.set(0, 0, 0.0);
    lat.fill(bounds, |a, b, diag, up, left| {
        sqe(x.at(a), y.at(b)) * weights[a.abs_diff(b)] + up.min(left).min(diag)
    });
    lat
}

fn recipe(
    name: &'static str,
    derivative: bool,
    g: f64,
) -> Result<Recipe<impl Fn(Ser, Ser, &[(usize, usize)]) -> Lattice + Sync>, FdarError> {
    check_param("g", g, false)?;
    Ok(Recipe {
        name,
        derivative,
        padded_output: false,
        lattice: move |x: Ser, y: Ser, b: &[(usize, usize)]| wdtw_lattice(x, y, b, g),
        score: last_cell,
        path_score: last_cell,
        traceback: Traceback::MinReturn,
    })
}

elastic_family_fns! {
    label: "Weighted DTW (WDTW)",
    distance: wdtw_distance,
    pairwise: wdtw_pairwise_distance,
    cost: wdtw_cost_matrix,
    path: wdtw_alignment_path,
    distance_mv: wdtw_distance_multivariate,
    pairwise_mv: wdtw_pairwise_distance_multivariate,
    cost_mv: wdtw_cost_matrix_multivariate,
    path_mv: wdtw_alignment_path_multivariate,
    params: (g: f64),
    params_doc: "Each squared local cost `(x_i − y_j)²` is multiplied by the logistic \
        weight `1 / (1 + exp(−g (|i − j| − max(n, m) / 2)))`, penalising large phase \
        offsets. `g` (aeon default `0.05`) controls the penalty steepness; it must be \
        finite. No root is taken.",
    example: ", 0.05",
    recipe: recipe("wdtw", false, g),
}

elastic_family_fns! {
    label: "Weighted derivative DTW (WDDTW)",
    distance: wddtw_distance,
    pairwise: wddtw_pairwise_distance,
    cost: wddtw_cost_matrix,
    path: wddtw_alignment_path,
    distance_mv: wddtw_distance_multivariate,
    pairwise_mv: wddtw_pairwise_distance_multivariate,
    cost_mv: wddtw_cost_matrix_multivariate,
    path_mv: wddtw_alignment_path_multivariate,
    params: (g: f64),
    params_doc: "WDTW (logistic weight with steepness `g`, aeon default `0.05`) applied \
        to aeon's slope derivative of both series (length `n − 2`, see \
        `ddtw_distance`); window, cost matrix and path are in derivative coordinates.",
    example: ", 0.05",
    recipe: recipe("wddtw", true, g),
}
