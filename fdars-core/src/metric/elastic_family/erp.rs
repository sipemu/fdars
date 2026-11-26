//! Edit distance with real penalty (Chen & Ng 2004), aeon `erp_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    check_param, elastic_family_fns, euc, last_cell, ElasticDistancePath, Lattice, Recipe,
    Traceback,
};
use crate::metric::window::DtwWindow;

/// aeon `_erp_cost_matrix`. Note aeon's border: every entry of the padded
/// first column holds the **total** gap cost `Σ_i |x_i − g|` (not a running
/// sum), likewise the first row; this is reproduced exactly.
fn erp_lattice(x: &[f64], y: &[f64], bounds: &[(usize, usize)], g: f64) -> Lattice {
    let gx: Vec<f64> = x.iter().map(|&v| euc(v, g)).collect();
    let gy: Vec<f64> = y.iter().map(|&v| euc(v, g)).collect();
    let x_sum: f64 = gx.iter().fold(0.0, |s, &v| s + v);
    let y_sum: f64 = gy.iter().fold(0.0, |s, &v| s + v);
    let (n, m) = (x.len(), y.len());
    let mut lat = Lattice::new(n, m, f64::INFINITY);
    for i in 1..=n {
        lat.set(i, 0, x_sum);
    }
    for j in 1..=m {
        lat.set(0, j, y_sum);
    }
    lat.set(0, 0, 0.0);
    lat.fill(bounds, |a, b, diag, up, left| {
        (diag + euc(x[a], y[b])).min(up + gx[a]).min(left + gy[b])
    });
    lat
}

fn recipe(
    g: f64,
) -> Result<Recipe<impl Fn(&[f64], &[f64], &[(usize, usize)]) -> Lattice + Sync>, FdarError> {
    check_param("g", g, false)?;
    Ok(Recipe {
        name: "erp",
        derivative: false,
        padded_output: false,
        lattice: move |x: &[f64], y: &[f64], b: &[(usize, usize)]| erp_lattice(x, y, b, g),
        score: last_cell,
        path_score: last_cell,
        traceback: Traceback::MinReturn,
    })
}

elastic_family_fns! {
    label: "Edit distance with real penalty (ERP)",
    distance: erp_distance,
    pairwise: erp_pairwise_distance,
    cost: erp_cost_matrix,
    path: erp_alignment_path,
    params: (g: f64),
    params_doc: "Matches cost `|x_i − y_j|`; a gap (skipping `x_i` or `y_j`) costs the \
        distance to the reference value `g` (aeon default `0.0`, finite). For univariate \
        series aeon's per-channel `g_arr` is this scalar. aeon's border convention (the \
        padded first row / column hold the total gap cost of the other series) is kept.",
    example: ", 0.0",
    recipe: recipe(g),
}
