//! Component-wise gradient boosting for function-on-function regression (REG-06-02).
//!
//! Implements the bfpc (FPC-compression) variant of boosted FoFR: functional predictors
//! are compressed via `fdata_to_pc_1d` (FPCA), and the boosting core operates on the
//! resulting FPC score design matrices.
//!
//! **Status:** Skeleton — implementation delivered in Plan 02.
//!
//! # References
//!
//! FDboost CRAN documentation, `bfpc` / `bsignal` base-learners.
//!
//! # Divergences from FDboost
//!
//! Uses FPC score compression (`fdata_to_pc_1d`) rather than FDboost's `bsignal`
//! B-spline joint expansion. Simpler and dependency-free.

use crate::error::FdarError;
use crate::matrix::FdMatrix;
use super::{BoostFofrResult, BoostingConfig};

/// Component-wise gradient boosting for function-on-function regression.
///
/// # Arguments
///
/// * `x_data` — Slice of functional predictor matrices, one per predictor (each n × m_x).
/// * `x_argvals` — Evaluation grids for each functional predictor (one per predictor).
/// * `y_data` — Functional response matrix (n × m_y).
/// * `y_argvals` — Response grid evaluation points (length m_y).
/// * `config` — [`BoostingConfig`] for boosting parameters and FPC compression.
///
/// # Errors
///
/// Returns [`FdarError::ComputationFailed`] with `detail: "not yet implemented (Plan 02)"`.
/// Full implementation arrives in Plan 02.
#[must_use = "expensive computation whose result should not be discarded"]
pub fn boost_fofr(
    _x_data: &[&FdMatrix],
    _x_argvals: &[&[f64]],
    _y_data: &FdMatrix,
    _y_argvals: &[f64],
    _config: &BoostingConfig,
) -> Result<BoostFofrResult, FdarError> {
    // filled in Plan 02
    Err(FdarError::ComputationFailed {
        operation: "boost_fofr",
        detail: "not yet implemented (Plan 02)".to_string(),
    })
}
