//! Edit distance on real sequences (Chen et al. 2005), aeon `edr_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    check_param, elastic_family_fns, euc, in_band, last_cell, ElasticDistancePath, Lattice, Recipe,
    Ser, Traceback,
};
use crate::metric::window::DtwWindow;

/// Population standard deviation (numpy `np.std`, `ddof = 0`).
fn pop_std(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    (x.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt()
}

/// aeon `_edr_cost_matrix`, including its border initialisation (zeros on the
/// padded first column where row `i − 1` admits column 0, and on padded first-row
/// entry `j < m` where row 0 admits column `j − 1`, wrapping `−1` to `m − 1`).
fn edr_lattice(x: Ser, y: Ser, bounds: &[(usize, usize)], epsilon: Option<f64>) -> Lattice {
    let (n, m) = (x.len(), y.len());
    let eps = epsilon.unwrap_or_else(|| pop_std(x.v).max(pop_std(y.v)) / 4.0);
    let mut lat = Lattice::new(n, m, f64::INFINITY);
    for i in 1..=n {
        if in_band(bounds, i - 1, 0) {
            lat.set(i, 0, 0.0);
        }
    }
    for j in 0..m {
        let col = if j == 0 { m - 1 } else { j - 1 };
        if in_band(bounds, 0, col) {
            lat.set(0, j, 0.0);
        }
    }
    lat.set(0, 0, 0.0);
    lat.fill(bounds, |a, b, diag, up, left| {
        let cost = if euc(x.at(a), y.at(b)) < eps {
            0.0
        } else {
            1.0
        };
        (diag + cost).min(up + 1.0).min(left + 1.0)
    });
    lat
}

fn normalised(l: &Lattice, n: usize, m: usize) -> f64 {
    last_cell(l, n, m) / n.max(m) as f64
}

fn recipe(
    epsilon: Option<f64>,
) -> Result<Recipe<impl Fn(Ser, Ser, &[(usize, usize)]) -> Lattice + Sync>, FdarError> {
    if let Some(e) = epsilon {
        check_param("epsilon", e, true)?;
    }
    Ok(Recipe {
        name: "edr",
        derivative: false,
        padded_output: false,
        lattice: move |x: Ser, y: Ser, b: &[(usize, usize)]| edr_lattice(x, y, b, epsilon),
        score: normalised,
        path_score: normalised,
        traceback: Traceback::MinReturn,
    })
}

elastic_family_fns! {
    label: "Edit distance on real sequences (EDR)",
    distance: edr_distance,
    pairwise: edr_pairwise_distance,
    cost: edr_cost_matrix,
    path: edr_alignment_path,
    distance_mv: edr_distance_multivariate,
    pairwise_mv: edr_pairwise_distance_multivariate,
    cost_mv: edr_cost_matrix_multivariate,
    path_mv: edr_alignment_path_multivariate,
    params: (epsilon: Option<f64>),
    params_doc: "Counts edits: a match costs `0` if `|x_i − y_j| < epsilon` and `1` \
        otherwise, every gap costs `1`; the distance is the edit count divided by \
        `max(n, m)` (the cost matrix is unnormalised). `epsilon = None` (aeon default) \
        uses `max(sd(x), sd(y)) / 4` with population standard deviations, computed per \
        pair; a given `epsilon` must be finite and `≥ 0`.",
    example: ", None",
    recipe: recipe(epsilon),
}
