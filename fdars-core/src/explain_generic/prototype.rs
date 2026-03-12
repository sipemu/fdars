use crate::error::FdarError;
use crate::explain::{
    compute_kernel_mean, compute_witness, gaussian_kernel_matrix, greedy_prototype_selection,
    median_bandwidth, PrototypeCriticismResult,
};
use crate::matrix::FdMatrix;

use super::FpcPredictor;

/// Generic prototype/criticism selection for any FPC-based model.
///
/// # Errors
///
/// Returns [`FdarError::InvalidDimension`] if `data` has zero rows or its
/// column count does not match the model.
/// Returns [`FdarError::InvalidParameter`] if the model has zero components,
/// `n_prototypes` is zero, or `n_prototypes > n`.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn generic_prototype_criticism(
    model: &dyn FpcPredictor,
    data: &FdMatrix,
    n_prototypes: usize,
    n_criticisms: usize,
) -> Result<PrototypeCriticismResult, FdarError> {
    let (n, m) = data.shape();
    if n == 0 {
        return Err(FdarError::InvalidDimension {
            parameter: "data",
            expected: "n > 0".into(),
            actual: "0 rows".into(),
        });
    }
    if m != model.fpca_mean().len() {
        return Err(FdarError::InvalidDimension {
            parameter: "data columns",
            expected: model.fpca_mean().len().to_string(),
            actual: m.to_string(),
        });
    }
    let ncomp = model.ncomp();
    if ncomp == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "ncomp",
            message: "model has 0 components".into(),
        });
    }
    if n_prototypes == 0 {
        return Err(FdarError::InvalidParameter {
            parameter: "n_prototypes",
            message: "n_prototypes must be > 0".into(),
        });
    }
    if n_prototypes > n {
        return Err(FdarError::InvalidParameter {
            parameter: "n_prototypes",
            message: format!("n_prototypes {n_prototypes} > n {n}"),
        });
    }
    let n_crit = n_criticisms.min(n.saturating_sub(n_prototypes));

    let scores = model.project(data);
    let bandwidth = median_bandwidth(&scores, n, ncomp);
    let kernel = gaussian_kernel_matrix(&scores, ncomp, bandwidth);
    let mu_data = compute_kernel_mean(&kernel, n);

    let (selected, is_selected) = greedy_prototype_selection(&mu_data, &kernel, n, n_prototypes);
    let witness = compute_witness(&kernel, &mu_data, &selected, n);
    let prototype_witness: Vec<f64> = selected.iter().map(|&i| witness[i]).collect();

    let mut criticism_candidates: Vec<(usize, f64)> = (0..n)
        .filter(|i| !is_selected[*i])
        .map(|i| (i, witness[i].abs()))
        .collect();
    criticism_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let criticism_indices: Vec<usize> = criticism_candidates
        .iter()
        .take(n_crit)
        .map(|&(i, _)| i)
        .collect();
    let criticism_witness: Vec<f64> = criticism_indices.iter().map(|&i| witness[i]).collect();

    Ok(PrototypeCriticismResult {
        prototype_indices: selected,
        prototype_witness,
        criticism_indices,
        criticism_witness,
        bandwidth,
    })
}
