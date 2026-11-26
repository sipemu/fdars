//! Longest common subsequence distance (Vlachos et al. 2002), aeon `lcss_*`.

use crate::error::FdarError;
use crate::matrix::FdMatrix;

use super::{
    check_param, elastic_family_fns, euc, in_band, ElasticDistancePath, Lattice, Recipe, Ser,
    Traceback,
};
use crate::metric::window::DtwWindow;

/// aeon `_lcss_cost_matrix`: zero-initialised padded table of LCSS lengths.
fn lcss_lattice(x: Ser, y: Ser, bounds: &[(usize, usize)], epsilon: f64) -> Lattice {
    let mut lat = Lattice::new(x.len(), y.len(), 0.0);
    lat.fill(bounds, |a, b, diag, up, left| {
        if euc(x.at(a), y.at(b)) <= epsilon {
            1.0 + diag
        } else {
            left.max(up)
        }
    });
    lat
}

fn raw_score(l: &Lattice, n: usize, m: usize) -> f64 {
    1.0 - l.get(n, m) / n.min(m) as f64
}

fn clamped_score(l: &Lattice, n: usize, m: usize) -> f64 {
    raw_score(l, n, m).max(0.0)
}

/// Port of aeon's `compute_lcss_return_path` (matched pairs only). Where aeon's
/// loop never terminates (an out-of-window cell), the non-match step is taken.
pub(super) fn lcss_return_path(
    x: Ser,
    y: Ser,
    epsilon: f64,
    bounds: &[(usize, usize)],
    lat: &Lattice,
) -> Vec<(usize, usize)> {
    let (mut i, mut j) = (x.len(), y.len());
    let mut path = Vec::new();
    while i > 0 && j > 0 {
        if in_band(bounds, i - 1, j - 1) && euc(x.at(i - 1), y.at(j - 1)) <= epsilon {
            path.push((i - 1, j - 1));
            i -= 1;
            j -= 1;
        } else if lat.get(i - 1, j) > lat.get(i, j - 1) {
            i -= 1;
        } else {
            j -= 1;
        }
    }
    path.reverse();
    path
}

fn recipe(
    epsilon: f64,
) -> Result<Recipe<impl Fn(Ser, Ser, &[(usize, usize)]) -> Lattice + Sync>, FdarError> {
    check_param("epsilon", epsilon, true)?;
    Ok(Recipe {
        name: "lcss",
        derivative: false,
        padded_output: true,
        lattice: move |x: Ser, y: Ser, b: &[(usize, usize)]| lcss_lattice(x, y, b, epsilon),
        score: clamped_score,
        path_score: raw_score,
        traceback: Traceback::Lcss(epsilon),
    })
}

elastic_family_fns! {
    label: "Longest common subsequence (LCSS)",
    distance: lcss_distance,
    pairwise: lcss_pairwise_distance,
    cost: lcss_cost_matrix,
    path: lcss_alignment_path,
    distance_mv: lcss_distance_multivariate,
    pairwise_mv: lcss_pairwise_distance_multivariate,
    cost_mv: lcss_cost_matrix_multivariate,
    path_mv: lcss_alignment_path_multivariate,
    params: (epsilon: f64),
    params_doc: "Two samples match when `|x_i − y_j| ≤ epsilon` (aeon default `1.0`, \
        finite and `≥ 0`); the distance is `1 − LCSS / min(n, m)`, clamped at `0`. As in \
        aeon the cost matrix is the **padded** `(n + 1) × (m + 1)` table of common \
        subsequence lengths (zero outside the window), and the alignment path lists only \
        the matched pairs.",
    example: ", 1.0",
    recipe: recipe(epsilon),
}
